#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unsloth ORPO Data Construction Pipeline
=======================================

功能：
1. 載入 Base Model + LoRA Adapter
2. 批次生成 rejected
3. 計算 conditional NLL / log-PPL
4. 計算 margin = loss_rejected - loss_chosen
5. 保留 margin > 0 且處於中間 50% 的樣本
6. 輸出 ORPO 格式：prompt / chosen / rejected

使用方式：
- 方法 A：設定 INPUT_DATA_PATH，例如 "input.jsonl"
- 方法 B：在執行此腳本前，先在 Colab 中定義好 pandas DataFrame 變數 `df`
"""

import os
import gc
import json
from typing import Any, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
from transformers import AutoTokenizer, set_seed

try:
    from unsloth import FastLanguageModel
except ImportError as e:
    raise SystemExit(
        "請先安裝 Unsloth，例如：\n"
        "!pip install --upgrade 'unsloth[colab] @ git+https://github.com/unslothai/unsloth.git'"
    ) from e

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None


# ============================================================
# 1. 使用者設定區
# ============================================================

BASE_MODEL_NAME = "unsloth/Qwen3-4B-Base-unsloth-bnb-4bit"
ADAPTER_MODEL_ID = "tatasauce4life/qwen3-4b-en2zh-lora"

# 優先嘗試從 adapter repo 載入 tokenizer；若失敗，會退回使用 base tokenizer
TOKENIZER_NAME_OR_PATH = ADAPTER_MODEL_ID

# 輸入資料：
# - 可填檔案路徑：.jsonl / .json / .csv / .parquet
# - 可填 "hf:dataset_name"（需要安裝 datasets）
# - 若留空 None，則會嘗試讀取 Colab 環境中已存在的變數 `df` 或 `dataset`


# 模板
PROMPT_TEMPLATE = "### English\n{en_text}\n### Chinese\n"
COMPLETION_TEMPLATE = "{zh_text}{eos}"  # 僅供對照；實際實作會手動附加 EOS token
FULL_TEMPLATE = PROMPT_TEMPLATE + COMPLETION_TEMPLATE

# 生成參數
# ===============================
# 可由外部環境變數覆蓋
# ===============================
INPUT_DATA_PATH = os.getenv("INPUT_DATA_PATH") or None

OUTPUT_JSONL = os.getenv("OUTPUT_JSONL", "orpo_decision_boundary.jsonl")
SCORED_JSONL = os.getenv("SCORED_JSONL", "orpo_scored_all.jsonl")

# full: 生成 + 打分 + 最終篩選
# score: 只做生成 + 打分，不做最終篩選
RUN_MODE = os.getenv("RUN_MODE", "full").lower()

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "2048"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
TEMPERATURE = 0.8
TOP_P = 0.9
DO_SAMPLE = True
SEED = 42

# 篩選百分位
LOWER_Q = 0.25
UPPER_Q = 0.75

# 顯存管理
CLEAR_EVERY_N_STEPS = 25

# Unsloth / 量化設定
LOAD_IN_4BIT = True
DTYPE = None  # 也可設為 torch.bfloat16 / torch.float16，通常留空 None 讓 Unsloth 自動決定

# 若你的 SFT tokenizer 明確需要 BOS，可謹慎調整此處；
# 一般 raw template SFT 建議維持 False，避免 response 前多餘特殊 token。
ADD_SPECIAL_TOKENS = False


# ============================================================
# 2. 工具函式
# ============================================================

def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def get_scalar_eos_token_id(tokenizer) -> int | None:
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        return None
    if isinstance(eos, (list, tuple)):
        return int(eos[0]) if len(eos) > 0 else None
    return int(eos)


def remove_trailing_eos_string(text: str, tokenizer) -> str:
    """
    若文字結尾已包含 EOS 字串，先移除，避免後續又手動加一次 EOS。
    """
    eos = getattr(tokenizer, "eos_token", None)
    if eos is None:
        return text
    eos = str(eos)
    if eos and text.endswith(eos):
        return text[:-len(eos)]
    return text


def safe_exp_loss(loss: float | None) -> float | None:
    if loss is None:
        return None
    try:
        if not np.isfinite(loss):
            return None
        # 避免 overflow；這只是方便查看，不影響核心篩選邏輯
        return float(np.exp(np.clip(loss, -20.0, 20.0)))
    except Exception:
        return None


def save_jsonl(df: pd.DataFrame, path: str) -> None:
    records = df.to_dict(orient="records")
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            clean = {}
            for k, v in rec.items():
                if isinstance(v, (np.integer,)):
                    v = int(v)
                elif isinstance(v, (np.floating,)):
                    v = float(v) if np.isfinite(v) else None
                elif isinstance(v, (np.bool_,)):
                    v = bool(v)
                else:
                    try:
                        if pd.isna(v):
                            v = None
                    except (TypeError, ValueError):
                        pass
                clean[k] = v
            f.write(json.dumps(clean, ensure_ascii=False, default=str) + "\n")
    print(f"已保存：{path}")


# ============================================================
# 3. 模型與 Tokenizer 載入
# ============================================================

def setup_model_and_tokenizer():
    set_seed(SEED)

    print(f"Loading base model: {BASE_MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )

    print(f"Loading LoRA adapter: {ADAPTER_MODEL_ID}")
    adapter_loaded = False

    # 優先使用 PEFT / Unsloth 模型自帶的 load_adapter
    try:
        model.load_adapter(ADAPTER_MODEL_ID, adapter_name="sft_adapter")
        try:
            model.set_adapter("sft_adapter")
        except Exception:
            pass
        adapter_loaded = True
        print("Adapter loaded via model.load_adapter().")
    except Exception as e:
        print(f"model.load_adapter failed: {e}")

    # 備援：直接用 PeftModel.from_pretrained 包一層
    if not adapter_loaded:
        if PeftModel is None:
            raise RuntimeError(
                "需要 peft 才能載入 LoRA adapter。請安裝：!pip install peft"
            )
        print("Trying PeftModel.from_pretrained(...)")
        model = PeftModel.from_pretrained(
            model,
            ADAPTER_MODEL_ID,
            is_trainable=False,
        )
        print("Adapter loaded via PeftModel.from_pretrained().")

    # 嘗試載入 adapter 所在仓库的 tokenizer；若失敗，保留 base tokenizer
    try:
        print(f"Trying to load tokenizer from: {TOKENIZER_NAME_OR_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_NAME_OR_PATH,
            trust_remote_code=True,
        )
        print("Tokenizer loaded from adapter repo.")
    except Exception as e:
        print(f"Failed to load tokenizer from adapter repo: {e}")
        print("Fallback to base tokenizer returned by FastLanguageModel.")

    # 補齊 EOS / PAD
    try:
        if tokenizer.eos_token_id is None and hasattr(model.config, "eos_token_id"):
            tokenizer.eos_token_id = model.config.eos_token_id
    except Exception:
        pass

    if tokenizer.eos_token is None:
        eos_id = get_scalar_eos_token_id(tokenizer)
        if eos_id is not None:
            try:
                tokenizer.eos_token = tokenizer.decode(eos_id)
            except Exception:
                pass

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            eos_id = get_scalar_eos_token_id(tokenizer)
            if eos_id is not None:
                tokenizer.pad_token = tokenizer.decode(eos_id)
            else:
                raise ValueError("無法決定 tokenizer.pad_token，請手動設定。")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = get_scalar_eos_token_id(tokenizer)

    # 批次生成時必須 left padding
    tokenizer.padding_side = "left"

    # 啟用 Unsloth 高速推理模式
    try:
        returned = FastLanguageModel.for_inference(model)
        if returned is not None:
            model = returned
        print("FastLanguageModel.for_inference(model) enabled.")
    except Exception as e:
        print(f"FastLanguageModel.for_inference failed: {e}; fallback to model.eval()")

    model.eval()
    return model, tokenizer


# ============================================================
# 4. 資料載入
# ============================================================

def to_dataframe(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()

    if hasattr(data, "to_pandas"):
        return data.to_pandas()

    if isinstance(data, dict):
        # DatasetDict 常見 split
        for split in ["train", "test", "validation"]:
            if split in data and hasattr(data[split], "to_pandas"):
                return data[split].to_pandas()
        return pd.DataFrame(data)

    if isinstance(data, (list, tuple)):
        return pd.DataFrame(data)

    raise TypeError(f"不支援的輸入資料型別：{type(data)}")


def load_input_data(path: str) -> pd.DataFrame:
    if path.startswith("hf:"):
        from datasets import load_dataset
        dataset_path = path[3:]
        ds = load_dataset(dataset_path, split="train")
        return ds.to_pandas()

    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    if path.endswith(".json"):
        return pd.read_json(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)

    raise ValueError(
        "INPUT_DATA_PATH 只支援 .jsonl / .json / .csv / .parquet，"
        "或使用 'hf:dataset_name'，或直接提供變數 df / dataset。"
    )


def get_initial_dataframe() -> pd.DataFrame:
    if INPUT_DATA_PATH is not None:
        raw = load_input_data(INPUT_DATA_PATH)
    elif "df" in globals():
        raw = globals()["df"]
    elif "dataset" in globals():
        raw = globals()["dataset"]
    else:
        raise ValueError(
            "找不到輸入資料。請設定 INPUT_DATA_PATH，"
            "或在執行前定義 pandas DataFrame 變數 `df`。"
        )
    return to_dataframe(raw)


# ============================================================
# 5. Prompt 模板
# ============================================================

def build_prompt_text(en_text: str) -> str:
    return PROMPT_TEMPLATE.format(en_text=safe_str(en_text))


# ============================================================
# 6. 步驟 1：批次生成 rejected
# ============================================================

def generate_rejected_samples(
    model,
    tokenizer,
    df: pd.DataFrame,
    batch_size: int = BATCH_SIZE,
) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()
    prompts = [build_prompt_text(x) for x in df["en_text"]]

    rejected_all = []
    max_prompt_len = max(1, int(MAX_SEQ_LENGTH - MAX_NEW_TOKENS))

    print(f"Generation prompt max length: {max_prompt_len}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")

    with torch.no_grad():
        for start in tqdm(range(0, len(prompts), batch_size), desc="Generate rejected"):
            batch = prompts[start:start + batch_size]

            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_prompt_len,
                add_special_tokens=ADD_SPECIAL_TOKENS,
                return_tensors="pt",
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            gen_kwargs = dict(
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=DO_SAMPLE,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
            if tokenizer.eos_token_id is not None:
                gen_kwargs["eos_token_id"] = tokenizer.eos_token_id

            outputs = model.generate(**inputs, **gen_kwargs)

            # left padding 下，直接切掉「整個 padded prompt 區塊」
            prompt_len_in_batch = inputs["input_ids"].shape[1]
            if outputs.shape[1] <= prompt_len_in_batch:
                decoded = [""] * len(batch)
            else:
                generated_ids = outputs[:, prompt_len_in_batch:]
                decoded = tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

            decoded = [safe_str(x) for x in decoded]
            rejected_all.extend(decoded)

            del inputs, outputs

            if CLEAR_EVERY_N_STEPS > 0 and (start // batch_size) % CLEAR_EVERY_N_STEPS == 0:
                clear_memory()

    df["rejected"] = rejected_all
    return df


# ============================================================
# 7. 步驟 2：計算條件 NLL / log-PPL
# ============================================================

def compute_single_nll(
    model,
    tokenizer,
    prompt_text: str,
    response_text: str,
    append_eos: bool = True,
) -> float | None:
    """
    回傳 token-averaged negative log-likelihood，
    也就是 log PPL(response | prompt)。

    重要：
    - prompt 段 labels = -100
    - 只計算 response 段（含 EOS）
    """
    try:
        prompt_text = safe_str(prompt_text)
        response_text = safe_str(response_text)
        response_text = remove_trailing_eos_string(response_text, tokenizer)

        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=ADD_SPECIAL_TOKENS,
            truncation=True,
            max_length=max(1, MAX_SEQ_LENGTH - 1),
        )["input_ids"]

        if len(prompt_ids) == 0:
            return None

        reserve = 1 if append_eos else 0
        max_response_len = MAX_SEQ_LENGTH - len(prompt_ids) - reserve
        if max_response_len <= 0:
            return None

        response_ids = tokenizer(
            response_text,
            add_special_tokens=ADD_SPECIAL_TOKENS,
            truncation=True,
            max_length=max_response_len,
        )["input_ids"]

        if append_eos:
            eos_id = get_scalar_eos_token_id(tokenizer)
            if eos_id is not None:
                if len(response_ids) == 0 or response_ids[-1] != eos_id:
                    response_ids.append(eos_id)

        if len(response_ids) == 0:
            return None

        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids

        input_ids_t = torch.tensor([input_ids], dtype=torch.long, device=model.device)
        labels_t = torch.tensor([labels], dtype=torch.long, device=model.device)
        attention_mask_t = torch.ones_like(input_ids_t)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids_t,
                attention_mask=attention_mask_t,
                labels=labels_t,
                use_cache=False,
            )
            loss = outputs.loss
            if loss is None:
                return None
            loss = float(loss.detach().float().item())

        if not np.isfinite(loss):
            return None

        return loss

    except Exception as e:
        print(f"compute_single_nll error: {e}")
        return None


def compute_conditional_loss(
    model,
    tokenizer,
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()

    loss_chosen = []
    loss_rejected = []
    margins = []

    model.eval()

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Conditional NLL"):
        prompt_text = build_prompt_text(row["en_text"])
        chosen_text = safe_str(row["zh_text"])
        rejected_text = safe_str(row["rejected"]) if "rejected" in row else ""

        lc = compute_single_nll(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            response_text=chosen_text,
            append_eos=True,
        )

        if rejected_text == "":
            lr = None
        else:
            lr = compute_single_nll(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                response_text=rejected_text,
                append_eos=True,
            )

        if lc is not None and lr is not None:
            margin = lr - lc
        else:
            margin = None

        loss_chosen.append(lc)
        loss_rejected.append(lr)
        margins.append(margin)

        if CLEAR_EVERY_N_STEPS > 0 and (i + 1) % CLEAR_EVERY_N_STEPS == 0:
            clear_memory()

    df["loss_chosen"] = loss_chosen
    df["loss_rejected"] = loss_rejected
    df["margin"] = margins

    # 附加 perplexity 供觀察（非必要）
    df["ppl_chosen"] = [safe_exp_loss(x) for x in loss_chosen]
    df["ppl_rejected"] = [safe_exp_loss(x) for x in loss_rejected]

    return df


# ============================================================
# 8. 步驟 3：統計篩選（中間 50%）
# ============================================================

def filter_margin_percentile(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    回傳：
    final_df, positive_df, valid_df, stats
    """
    scored = df.copy()

    for col in ["loss_chosen", "loss_rejected", "margin"]:
        scored[col] = pd.to_numeric(scored[col], errors="coerce")

    valid = scored.dropna(subset=["loss_chosen", "loss_rejected", "margin"]).copy()
    valid = valid[
        np.isfinite(valid["loss_chosen"]) &
        np.isfinite(valid["loss_rejected"]) &
        np.isfinite(valid["margin"])
    ].copy()

    positive = valid[valid["margin"] > 0].copy()

    if len(positive) == 0:
        stats = {
            "q25": np.nan,
            "q50": np.nan,
            "q75": np.nan,
        }
        return positive, positive, valid, stats

    q25 = float(positive["margin"].quantile(LOWER_Q))
    q50 = float(positive["margin"].quantile(0.5))
    q75 = float(positive["margin"].quantile(UPPER_Q))

    final = positive[
        (positive["margin"] >= q25) &
        (positive["margin"] <= q75)
    ].copy()

    stats = {
        "q25": q25,
        "q50": q50,
        "q75": q75,
    }

    return final, positive, valid, stats


# ============================================================
# 9. 步驟 4：輸出 ORPO 格式
# ============================================================

def make_final_orpo_dataframe(filtered_df: pd.DataFrame) -> pd.DataFrame:
    out = filtered_df.rename(columns={
        "en_text": "prompt",
        "zh_text": "chosen",
    })
    out = out[["prompt", "chosen", "rejected"]].copy()
    out["prompt"] = out["prompt"].map(safe_str)
    out["chosen"] = out["chosen"].map(safe_str)
    out["rejected"] = out["rejected"].map(safe_str)
    return out


# ============================================================
# 10. 主流程
# ============================================================

def main():
    # 1. 讀取資料
    raw = get_initial_dataframe()
    df = to_dataframe(raw)

    required_cols = {"en_text", "zh_text"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"輸入資料缺少欄位：{missing}")

    df["en_text"] = df["en_text"].map(safe_str)
    df["zh_text"] = df["zh_text"].map(safe_str)

    df = df.dropna(subset=["en_text", "zh_text"]).copy()
    df = df[
        (df["en_text"].str.strip() != "") &
        (df["zh_text"].str.strip() != "")
    ].reset_index(drop=True)

    original_count = len(df)
    print(f"原始資料筆數：{original_count}")

    if original_count == 0:
        raise ValueError("清洗後沒有可用資料。")

    # 2. 載入模型
    model, tokenizer = setup_model_and_tokenizer()

    # 3. 生成 rejected
    df = generate_rejected_samples(
        model=model,
        tokenizer=tokenizer,
        df=df,
        batch_size=BATCH_SIZE,
    )

    empty_rejected_count = int((df["rejected"].str.strip() == "").sum())
    print(f"生成空 rejected 的筆數：{empty_rejected_count}")

    clear_memory()

    # 4. 計算 conditional loss / margin
    df = compute_conditional_loss(
        model=model,
        tokenizer=tokenizer,
        df=df,
    )

    # 5. 保存完整打分結果（方便除錯）
    scored_export = df.rename(columns={
        "en_text": "prompt",
        "zh_text": "chosen",
    })
    scored_export = scored_export[[
        "prompt",
        "chosen",
        "rejected",
        "loss_chosen",
        "loss_rejected",
        "ppl_chosen",
        "ppl_rejected",
        "margin",
    ]]
    save_jsonl(scored_export, SCORED_JSONL)
    # ==================================================
    # 如果是 score mode，只產出打分檔，不做最終篩選
    # ==================================================
    if RUN_MODE == "score":
        print(f"RUN_MODE=score: skip final filtering. Scored file saved to {SCORED_JSONL}")
        return scored_export

    # 6. 篩選中間 50%
    final_filtered, positive, valid, stats = filter_margin_percentile(df)

    print("\n=== 篩選統計 ===")
    print(f"原始資料筆數：{original_count}")
    print(f"可成功計算 Loss 的筆數：{len(valid)}")
    print(f"排除 Margin <= 0 後的筆數：{len(positive)}")
    print(f"最終中間 50% 篩選出的筆數：{len(final_filtered)}")

    if all(np.isfinite([stats["q25"], stats["q50"], stats["q75"]])):
        print(f"Margin 25% 分位數：{stats['q25']:.6f}")
        print(f"Margin 50% 分位數：{stats['q50']:.6f}")
        print(f"Margin 75% 分位數：{stats['q75']:.6f}")
    else:
        print("Margin 分位數：無可用樣本")

    # 7. 輸出最終 ORPO 資料
    final_df = make_final_orpo_dataframe(final_filtered)
    save_jsonl(final_df, OUTPUT_JSONL)

    print(f"\n最終 ORPO 資料集共 {len(final_df)} 筆，已輸出至：{OUTPUT_JSONL}")

    return final_df


if __name__ == "__main__":
    final_df = main()