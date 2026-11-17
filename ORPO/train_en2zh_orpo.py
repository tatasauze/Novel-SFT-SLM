# ============================================================================
# 英中小說翻譯 ORPO 偏好優化訓練腳本
# 基於先前 SFT 模型，使用人工譯文 vs. 機器翻譯做 ORPO
# ============================================================================
# 建議安裝指令（請依環境調整）：
# pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# pip install torch transformers datasets trl peft bitsandbytes accelerate
# pip install huggingface_hub pandas numpy
# pip install sacrebleu  # 可選，用於翻譯指標
#
# 若 ORPOTrainer 不存在，請確認 trl 版本：
# pip install trl>=0.9.0
# Unsloth 官方模板建議：pip install --no-deps trl==0.22.2
# ============================================================================

import os
import json
import time
import random
import warnings
import inspect
from typing import List, Dict, Optional, Any

import numpy as np
import pandas as pd
import torch

# ============================================================================
# ★ 使用者配置區 — 請依實際需求修改 ★
# ============================================================================

# --- 模型路徑 ---
BASE_MODEL_NAME = "unsloth/Qwen3-4B-Base-unsloth-bnb-4bit"
SFT_MODEL_PATH  = "請填寫你的SFT模型路徑或HuggingFace_repo"
# 例如：
#   LoRA adapter:  "./qwen3-4b-en2zh-sft/final_lora"
#   HF repo:       "your-username/qwen3-4b-en2zh-sft-lora"
#   已合併模型:     "./qwen3-4b-en2zh-sft-merged"

# --- ORPO 資料 ---
ORPO_DATA_PATH = "./pre_orpo/pre_orpo.jsonl"
# JSONL 格式：{"prompt": "英文原文", "chosen": "人工譯文", "reject": "機器翻譯"}

# --- Hugging Face Hub ---
HF_REPO_ID   = "請填寫你的huggingface帳號/模型名稱"
PUSH_TO_HUB  = True
PRIVATE       = False
# 不要在代碼中寫死 token，請使用環境變數 HF_TOKEN 或 huggingface-cli login

# --- 模型設定 ---
MAX_SEQ_LENGTH  = 1024
LOAD_IN_4BIT    = True
DTYPE           = None  # None = 自動偵測

# --- LoRA 參數（僅在 SFT 模型為已合併模型時使用，新增 LoRA 做 ORPO） ---
LORA_R              = 16
LORA_ALPHA          = 32
LORA_DROPOUT        = 0.05
LORA_BIAS           = "none"
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# --- ORPO 訓練參數 ---
OUTPUT_DIR                  = "./qwen3-4b-en2zh-orpo"
NUM_TRAIN_EPOCHS            = 1
PER_DEVICE_TRAIN_BATCH_SIZE = 1       # OOM 時可改為 1 並增加 GRADIENT_ACCUMULATION_STEPS
PER_DEVICE_EVAL_BATCH_SIZE  = 1
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE               = 5e-6
WARMUP_RATIO                = 0.05
LR_SCHEDULER_TYPE           = "cosine"
OPTIM                       = "adamw_8bit"
WEIGHT_DECAY                = 0.01
MAX_GRAD_NORM               = 1.0
LOGGING_STEPS               = 5
SAVE_STRATEGY               = "epoch"
LOAD_BEST_MODEL_AT_END      = True
METRIC_FOR_BEST_MODEL       = "eval_loss"
GREATER_IS_BETTER           = False
SEED                        = 42
REPORT_TO                   = "none"
ORPO_BETA                   = 0.1     # ORPO odds ratio 權重
MAX_PROMPT_LENGTH           = 512     # ORPOTrainer 內部 prompt 截斷上限

# --- 資料切分 ---
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# --- 生成 ---
MAX_NEW_TOKENS_GEN = 512
NUM_SAMPLES_GEN    = 5

# --- 測試模式：使用假資料 ---
GENERATE_FAKE_DATA = False
FAKE_DATA_SIZE     = 100

# --- 資料清洗 ---
MIN_RESPONSE_CHARS = 2  # chosen / rejected 最短字元數


# ============================================================================
# 模板常量（與 train_en2zh_sft.py 完全一致）
# ============================================================================
PROMPT_TEMPLATE     = "### English\n{en_text}\n### Chinese\n"
COMPLETION_TEMPLATE = "{zh_text}{eos}"
FULL_TEMPLATE       = PROMPT_TEMPLATE + COMPLETION_TEMPLATE


# ============================================================================
# 1. 偵測 SFT 模型類型
# ============================================================================
def is_lora_adapter(path: str) -> bool:
    """
    判斷指定路徑是否為 LoRA adapter（而非已合併的完整模型）。
    檢查本地路徑或 HuggingFace repo 中是否存在 adapter_config.json。
    """
    # 檢查本地路徑
    if os.path.isdir(path):
        return os.path.exists(os.path.join(path, "adapter_config.json"))

    # 檢查 HuggingFace repo
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id=path,
            filename="adapter_config.json",
            local_dir_use_symlinks=False,
        )
        return True
    except Exception:
        return False


# ============================================================================
# 2. 資料載入
# ============================================================================
def load_orpo_data(path: str) -> List[Dict[str, str]]:
    """載入 JSONL 偏好資料。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 ORPO 資料檔案: {path}")

    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError as e:
                warnings.warn(f"第 {i+1} 行 JSON 解析失敗: {e}")
    print(f"從 {path} 載入 {len(data)} 筆 ORPO 資料")
    return data


def generate_fake_data(n: int = 100) -> List[Dict[str, str]]:
    """生成假資料供測試使用。"""
    rng = random.Random(42)
    data = []
    en_sentences = [
        "The moonlight filtered through the ancient branches, casting silver patterns on the forest floor.",
        "She drew her sword and stepped forward, her eyes fixed on the shadowy figure ahead.",
        "The old merchant laughed and poured another cup of tea.",
        "Thunder rumbled across the distant mountains as the storm approached.",
        "He opened the worn leather journal and began to read.",
    ]
    chosen_zh = [
        "月光穿透古老的枝椏，在林地上投下銀色的圖案。",
        "她拔出長劍向前邁步，目光緊盯著前方的暗影。",
        "老商人大笑著又倒了一杯茶。",
        "雷聲在遠山間隆隆作響，暴風雨正逼近。",
        "他翻開那本破舊的皮革日誌，開始閱讀。",
    ]
    rejected_zh = [
        "月光通过古老的树枝过滤，在森林地板上投射银色图案。",
        "她拉出她的剑并向前走，她的眼睛固定在前面的阴影人物上。",
        "老商人笑了，又倒了一杯茶水。",
        "雷声在远处的山脉上轰鸣，暴风雨接近了。",
        "他打开破旧的皮革杂志并开始阅读。",
    ]
    for i in range(n):
        idx = i % len(en_sentences)
        data.append({
            "prompt": en_sentences[idx] + f" (sample {i})",
            "chosen": chosen_zh[idx],
            "rejected": rejected_zh[idx],
        })
    print(f"已生成 {len(data)} 筆假測試資料")
    return data


# ============================================================================
# 3. 資料前處理
# ============================================================================
def standardize_fields(data: List[Dict]) -> List[Dict]:
    """
    標準化欄位名稱：
    - 將 'reject' 改名為 'rejected'
    - 若同時有 'reject' 和 'rejected'，優先使用 'rejected'
    """
    for item in data:
        if "rejected" not in item and "reject" in item:
            item["rejected"] = item.pop("reject")
        elif "reject" in item and "rejected" in item:
            # 優先使用 rejected，移除 reject
            item.pop("reject")
    return data


def clean_prompt_text(text: str) -> str:
    """清理 chosen / rejected 中可能包含的模板標記。"""
    # 移除可能混入的模板前綴
    for marker in ["### English", "### Chinese"]:
        if marker in text:
            # 嘗試取得 marker 之後的內容
            parts = text.split(marker)
            text = parts[-1]  # 取最後一段
    return text.strip()


def clean_data(data: List[Dict]) -> List[Dict]:
    """清洗 ORPO 資料。"""
    print("\n" + "=" * 60)
    print("ORPO 資料清洗")
    print("=" * 60)
    n_original = len(data)
    print(f"  原始資料筆數: {n_original}")

    cleaned = []
    removed_reasons = {
        "missing_field": 0,
        "empty_string": 0,
        "chosen_eq_rejected": 0,
        "too_short": 0,
        "template_in_response": 0,
    }

    for item in data:
        # 檢查必要欄位
        if not all(k in item for k in ["prompt", "chosen", "rejected"]):
            removed_reasons["missing_field"] += 1
            continue

        prompt  = str(item["prompt"]).strip()
        chosen  = str(item["chosen"]).strip()
        rejected = str(item["rejected"]).strip()

        # 移除空字串
        if not prompt or not chosen or not rejected:
            removed_reasons["empty_string"] += 1
            continue

        # 清理 chosen / rejected 中可能的模板標記
        chosen_clean = clean_prompt_text(chosen)
        rejected_clean = clean_prompt_text(rejected)

        if chosen_clean != chosen or rejected_clean != rejected:
            if not chosen_clean or not rejected_clean:
                removed_reasons["template_in_response"] += 1
                continue
            chosen = chosen_clean
            rejected = rejected_clean

        # chosen == rejected 無意義
        if chosen == rejected:
            removed_reasons["chosen_eq_rejected"] += 1
            continue

        # 過短回應
        if len(chosen) < MIN_RESPONSE_CHARS or len(rejected) < MIN_RESPONSE_CHARS:
            removed_reasons["too_short"] += 1
            continue

        cleaned.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })

    print(f"  清洗後: {len(cleaned)} 筆")
    for reason, count in removed_reasons.items():
        if count > 0:
            print(f"    移除 ({reason}): {count}")
    return cleaned


def format_for_orpo(data: List[Dict], eos_token: str) -> List[Dict]:
    """
    將資料格式化為 ORPOTrainer 期望的格式：
    - prompt: 包含完整英文模板 (### English\n{text}\n### Chinese\n)
    - chosen: 人工譯文 + EOS
    - rejected: 機器翻譯 + EOS
    """
    formatted = []
    for item in data:
        prompt_text = item["prompt"]

        # 判斷 prompt 是否已包含模板
        if "### Chinese" in prompt_text:
            # 已包含完整模板，直接使用
            formatted_prompt = prompt_text
            # 確保結尾有換行
            if not formatted_prompt.endswith("\n"):
                formatted_prompt += "\n"
        else:
            # 只是英文原文，包裝成模板
            formatted_prompt = PROMPT_TEMPLATE.format(en_text=prompt_text)

        formatted.append({
            "prompt":   formatted_prompt,
            "chosen":   item["chosen"] + eos_token,
            "rejected": item["rejected"] + eos_token,
        })
    return formatted


def dedup_by_prompt(data: List[Dict]) -> List[Dict]:
    """
    按 prompt 去重。
    若相同 prompt 出現多次，保留 chosen 較長的一筆。
    """
    n_before = len(data)
    prompt_map = {}
    for item in data:
        p = item["prompt"]
        if p not in prompt_map:
            prompt_map[p] = item
        else:
            # 保留 chosen 較長的
            if len(item["chosen"]) > len(prompt_map[p]["chosen"]):
                prompt_map[p] = item

    deduped = list(prompt_map.values())
    n_removed = n_before - len(deduped)
    print(f"\n  去重: {n_before} → {len(deduped)} (移除 {n_removed} 筆重複)")
    return deduped


def compute_token_stats_and_filter(
    data: List[Dict],
    tokenizer,
    max_seq_length: int,
) -> List[Dict]:
    """計算 token 長度分布，過濾超長樣本。"""
    print("\n" + "=" * 60)
    print("Token 長度統計")
    print("=" * 60)

    prompt_lens   = []
    chosen_lens   = []
    rejected_lens = []
    pc_lens       = []  # prompt + chosen
    pr_lens       = []  # prompt + rejected

    for item in data:
        p_ids = tokenizer.encode(item["prompt"],   add_special_tokens=False)
        c_ids = tokenizer.encode(item["chosen"],   add_special_tokens=False)
        r_ids = tokenizer.encode(item["rejected"], add_special_tokens=False)
        prompt_lens.append(len(p_ids))
        chosen_lens.append(len(c_ids))
        rejected_lens.append(len(r_ids))
        pc_lens.append(len(p_ids) + len(c_ids))
        pr_lens.append(len(p_ids) + len(r_ids))

    def _print_stats(name, arr):
        a = np.array(arr)
        print(f"  {name}:")
        print(f"    min={int(np.min(a))}, p50={int(np.percentile(a,50))}, "
              f"p90={int(np.percentile(a,90))}, p95={int(np.percentile(a,95))}, "
              f"p99={int(np.percentile(a,99))}, max={int(np.max(a))}")

    _print_stats("prompt",           prompt_lens)
    _print_stats("chosen",           chosen_lens)
    _print_stats("rejected",         rejected_lens)
    _print_stats("prompt + chosen",  pc_lens)
    _print_stats("prompt + rejected", pr_lens)

    # 過濾超長樣本（prompt + chosen 或 prompt + rejected 超過 max_seq_length）
    n_before = len(data)
    filtered = []
    for i, item in enumerate(data):
        max_total = max(pc_lens[i], pr_lens[i])
        if max_total <= max_seq_length:
            filtered.append(item)

    n_removed = n_before - len(filtered)
    print(f"\n  超過 MAX_SEQ_LENGTH ({max_seq_length}) 的樣本: {n_removed}")
    print(f"  過濾後剩餘: {len(filtered)}")
    return filtered


# ============================================================================
# 4. 資料切分
# ============================================================================
def split_data(
    data: List[Dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple:
    """隨機切分資料為 train / validation / test。"""
    print("\n" + "=" * 60)
    print("資料切分")
    print("=" * 60)

    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)

    total = len(data)
    n_test = int(total * test_ratio)
    n_val  = int(total * val_ratio)
    n_train = total - n_test - n_val

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            f"資料量不足以按比例切分！total={total}, "
            f"train={n_train}, val={n_val}, test={n_test}。"
            f"請調整切分比例或增加資料量。"
        )

    test_idx  = indices[:n_test]
    val_idx   = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    train_data = [data[i] for i in train_idx]
    val_data   = [data[i] for i in val_idx]
    test_data  = [data[i] for i in test_idx]

    # 檢查 prompt 無交集
    train_prompts = set(d["prompt"] for d in train_data)
    val_prompts   = set(d["prompt"] for d in val_data)
    test_prompts  = set(d["prompt"] for d in test_data)

    overlap_tv = train_prompts & val_prompts
    overlap_tt = train_prompts & test_prompts
    overlap_vt = val_prompts  & test_prompts

    if overlap_tv or overlap_tt or overlap_vt:
        warnings.warn(
            f"prompt 出現交集！train∩val={len(overlap_tv)}, "
            f"train∩test={len(overlap_tt)}, val∩test={len(overlap_vt)}。"
            f"這可能是因為去重不完全。"
        )

    print(f"  Train:      {len(train_data):>5} 筆 ({len(train_data)/total*100:.1f}%)")
    print(f"  Validation: {len(val_data):>5} 筆 ({len(val_data)/total*100:.1f}%)")
    print(f"  Test:       {len(test_data):>5} 筆 ({len(test_data)/total*100:.1f}%)")

    # 印出每個 split 的平均 token 數（使用字元長度近似）
    for name, split in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
        avg_chosen  = np.mean([len(d["chosen"]) for d in split])
        avg_rejected = np.mean([len(d["rejected"]) for d in split])
        print(f"  {name} 平均 chosen 字元數: {avg_chosen:.0f}, "
              f"平均 rejected 字元數: {avg_rejected:.0f}")

    return train_data, val_data, test_data


# ============================================================================
# 5. 模型載入
# ============================================================================
def load_model_for_orpo(
    base_model_name: str,
    sft_model_path: str,
    max_seq_length: int,
    load_in_4bit: bool,
    dtype,
    seed: int,
):
    """
    載入 SFT 模型，支援 LoRA adapter 與已合併模型兩種情況。
    - LoRA adapter: 透過 Unsloth 載入 base + adapter，繼續在同一 LoRA 上訓練。
    - 已合併模型: 載入完整模型，新增 LoRA 做 ORPO。
    """
    from unsloth import FastLanguageModel

    print("\n" + "=" * 60)
    print("載入模型")
    print("=" * 60)
    print(f"  BASE_MODEL_NAME: {base_model_name}")
    print(f"  SFT_MODEL_PATH:  {sft_model_path}")

    adapter_mode = is_lora_adapter(sft_model_path)
    print(f"  SFT 模型類型: {'LoRA Adapter' if adapter_mode else '完整模型 / 已合併'}")

    if adapter_mode:
        # ----------------------------------------------------------------
        # 情況 A: SFT 模型是 LoRA adapter
        #   Unsloth 的 from_pretrained 會自動偵測 adapter_config.json，
        #   載入對應 base model 並套用 adapter。
        #   繼續在同一個 LoRA adapter 上做 ORPO 訓練。
        # ----------------------------------------------------------------
        print("\n  載入 LoRA adapter（Unsloth 自動偵測 base model）...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=sft_model_path,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )

        # 啟用 gradient checkpointing
        # 因為不透過 get_peft_model，無法使用 "unsloth" 優化，
        # 改用標準 gradient checkpointing。
        try:
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            print("  ✓ 已啟用 gradient checkpointing (標準模式)")
        except Exception as e:
            warnings.warn(f"啟用 gradient checkpointing 失敗: {e}")

        print("\n  ⚠ 將繼續在 SFT LoRA adapter 上訓練 ORPO。")
        print("    最終保存的 adapter 包含 SFT + ORPO 的累積更新。")

    else:
        # ----------------------------------------------------------------
        # 情況 B: SFT 模型是已合併的完整模型
        #   載入完整模型，新增一個 LoRA adapter 做 ORPO。
        # ----------------------------------------------------------------
        print("\n  載入已合併模型...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=sft_model_path,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )

        # 新增 LoRA adapter
        use_gc = "unsloth"
        try:
            model = FastLanguageModel.get_peft_model(
                model,
                r=LORA_R,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                bias=LORA_BIAS,
                target_modules=LORA_TARGET_MODULES,
                use_gradient_checkpointing=use_gc,
                random_state=seed,
            )
            print(f"  ✓ 新增 LoRA adapter (gradient_checkpointing='{use_gc}')")
        except Exception as e:
            warnings.warn(f"use_gradient_checkpointing='unsloth' 不支援 ({e})，降級為 True")
            use_gc = True
            model = FastLanguageModel.get_peft_model(
                model,
                r=LORA_R,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                bias=LORA_BIAS,
                target_modules=LORA_TARGET_MODULES,
                use_gradient_checkpointing=use_gc,
                random_state=seed,
            )
            print(f"  ✓ 新增 LoRA adapter (gradient_checkpointing={use_gc})")

        print("\n  ⚠ 已在合併模型上新增 LoRA。")
        print(f"    最終保存的 adapter 基於: {sft_model_path}")

    # 設定 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print("  pad_token 設為 eos_token")

    # 訓練時 padding_side = right
    tokenizer.padding_side = "right"

    # 印出可訓練參數
    model.print_trainable_parameters()

    return model, tokenizer, adapter_mode


# ============================================================================
# 6. 生成翻譯樣本
# ============================================================================
def generate_orpo_samples(
    model,
    tokenizer,
    data: List[Dict],
    num_samples: int = 5,
    max_new_tokens: int = 512,
) -> List[Dict]:
    """
    從資料中取前 num_samples 筆，使用 greedy decoding 生成翻譯。
    回傳生成結果列表。
    """
    try:
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)
    except Exception:
        model.eval()

    device = next(model.parameters()).device
    num_samples = min(num_samples, len(data))
    results = []

    print("\n" + "=" * 60)
    print(f"生成翻譯樣本 (共 {num_samples} 筆)")
    print("=" * 60)

    eos_token = tokenizer.eos_token or ""

    for i in range(num_samples):
        item = data[i]
        prompt = item["prompt"]
        # chosen / rejected 可能包含 EOS，清理掉用於顯示
        chosen  = item["chosen"].replace(eos_token, "").strip()
        rejected = item["rejected"].replace(eos_token, "").strip()

        input_ids = tokenizer.encode(
            prompt, add_special_tokens=False, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_ids = output_ids[0][input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # 清除模板標記
        for marker in ["### English", "### Chinese"]:
            generated_text = generated_text.replace(marker, "")
        if eos_token:
            generated_text = generated_text.replace(eos_token, "")
        generated_text = generated_text.strip()

        # 從 prompt 中提取英文原文（用於顯示）
        en_text = prompt.replace("### English\n", "").replace("\n### Chinese\n", "").strip()

        print(f"\n--- 樣本 {i + 1} ---")
        print(f"[英文原文]\n{en_text[:300]}{'...' if len(en_text)>300 else ''}")
        print(f"\n[chosen 人工譯文]\n{chosen[:300]}{'...' if len(chosen)>300 else ''}")
        print(f"\n[rejected 機器翻譯]\n{rejected[:300]}{'...' if len(rejected)>300 else ''}")
        print(f"\n[ORPO 模型生成]\n{generated_text[:300]}{'...' if len(generated_text)>300 else ''}")

        results.append({
            "english": en_text,
            "chosen": chosen,
            "rejected": rejected,
            "generated": generated_text,
        })

    return results


def translate_english_to_chinese(
    text: str,
    model,
    tokenizer,
    max_new_tokens: int = 512,
) -> str:
    """將英文翻譯為中文（推理函式）。"""
    try:
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)
    except Exception:
        model.eval()

    device = next(model.parameters()).device
    prompt_str = PROMPT_TEMPLATE.format(en_text=text)
    input_ids = tokenizer.encode(
        prompt_str, add_special_tokens=False, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = output_ids[0][input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    for marker in ["### English", "### Chinese"]:
        generated_text = generated_text.replace(marker, "")
    if tokenizer.eos_token:
        generated_text = generated_text.replace(tokenizer.eos_token, "")

    return generated_text.strip()


# ============================================================================
# 7. TrainerCallback: 每個 epoch 結束生成樣本
# ============================================================================
from transformers import TrainerCallback

class ORPOGenerationCallback(TrainerCallback):
    """在每個 epoch 結束時生成 1 筆翻譯樣本，觀察風格變化。"""

    def __init__(self, tokenizer, val_data: List[Dict], max_new_tokens: int = 512):
        self.tokenizer = tokenizer
        self.val_data = val_data
        self.max_new_tokens = max_new_tokens

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None or len(self.val_data) == 0:
            return
        try:
            generate_orpo_samples(
                model=model,
                tokenizer=self.tokenizer,
                data=self.val_data,
                num_samples=1,
                max_new_tokens=self.max_new_tokens,
            )
        except Exception as e:
            print(f"  [ORPOGenerationCallback] 生成失敗: {e}")
        model.train()


# ============================================================================
# 8. 建構 ORPOConfig（含版本兼容處理）
# ============================================================================
def build_orpo_config():
    """
    建構 ORPOConfig，處理不同 trl 版本的 API 差異。
    - eval_strategy vs evaluation_strategy
    - max_completion_length 可能不存在
    """
    # 確認 ORPOTrainer 與 ORPOConfig 可用
    try:
        from trl import ORPOConfig, ORPOTrainer
    except ImportError as e:
        raise ImportError(
            f"無法匯入 ORPOTrainer / ORPOConfig: {e}\n"
            f"請升級 trl: pip install trl>=0.9.0\n"
            f"Unsloth 建議: pip install --no-deps trl==0.22.2\n"
            f"⚠ 不要使用 DPO 替代 ORPO！"
        )

    # 偵測 bf16 支援
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    # 基礎參數
    kwargs = dict(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        optim=OPTIM,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        bf16=bf16_supported,
        fp16=(not bf16_supported and torch.cuda.is_available()),
        logging_steps=LOGGING_STEPS,
        save_strategy=SAVE_STRATEGY,
        load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
        metric_for_best_model=METRIC_FOR_BEST_MODEL,
        greater_is_better=GREATER_IS_BETTER,
        seed=SEED,
        report_to=REPORT_TO,
        remove_unused_columns=False,

        # ORPO 特有參數
        beta=ORPO_BETA,
        max_length=MAX_SEQ_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
    )

    # 檢查 ORPOConfig 是否支持 max_completion_length
    try:
        sig = inspect.signature(ORPOConfig)
        orpo_params = set(sig.parameters.keys())
    except Exception:
        orpo_params = set()

    if "max_completion_length" in orpo_params:
        kwargs["max_completion_length"] = MAX_SEQ_LENGTH - MAX_PROMPT_LENGTH

    # 兼容 eval_strategy / evaluation_strategy
    try:
        kwargs["eval_strategy"] = "epoch"
        config = ORPOConfig(**kwargs)
    except TypeError:
        kwargs.pop("eval_strategy", None)
        kwargs["evaluation_strategy"] = "epoch"
        try:
            config = ORPOConfig(**kwargs)
        except TypeError as e2:
            # 若還有不支援的參數，逐步移除嘗試
            for param_to_try_remove in ["max_completion_length", "max_grad_norm"]:
                if param_to_try_remove in kwargs:
                    kwargs.pop(param_to_try_remove)
            config = ORPOConfig(**kwargs)

    print(f"\n  bf16={bf16_supported}, ORPO beta={ORPO_BETA}")
    return config


# ============================================================================
# 9. 保存模型與配置
# ============================================================================
def save_all(
    model,
    tokenizer,
    generation_results: List[Dict],
    adapter_mode: bool,
    training_start_time: float,
):
    """保存 LoRA adapter / 模型、tokenizer、配置、README、生成樣本。"""
    save_dir = os.path.join(OUTPUT_DIR, "final")
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"儲存模型至: {save_dir}")
    print("=" * 60)

    # ---- 儲存模型 ----
    saved = False
    for method_name, method_fn in [
        ("model.save_pretrained",      lambda: model.save_pretrained(save_dir)),
        ("model.save_lora",            lambda: model.save_lora(save_dir)),
        ("base_model.save_pretrained", lambda: model.base_model.save_pretrained(save_dir)
         if hasattr(model, "base_model") else (_ for _ in ()).throw(AttributeError())),
    ]:
        if saved:
            break
        try:
            method_fn()
            print(f"  使用 {method_name}() 儲存成功")
            saved = True
        except (AttributeError, Exception) as e:
            print(f"  {method_name}() 不可用: {e}")

    if not saved:
        raise RuntimeError("所有模型儲存方式均失敗！")

    # ---- 儲存 tokenizer ----
    tokenizer.save_pretrained(save_dir)
    print(f"  Tokenizer 已儲存")

    # ---- 儲存 training_config.json ----
    training_time = time.time() - training_start_time
    config_dict = {
        "base_model_name": BASE_MODEL_NAME,
        "sft_model_path": SFT_MODEL_PATH,
        "orpo_data_path": ORPO_DATA_PATH,
        "max_seq_length": MAX_SEQ_LENGTH,
        "max_prompt_length": MAX_PROMPT_LENGTH,
        "template": FULL_TEMPLATE,
        "prompt_template": PROMPT_TEMPLATE,
        "sft_model_is_lora_adapter": adapter_mode,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "lora_target_modules": LORA_TARGET_MODULES,
        "orpo_beta": ORPO_BETA,
        "learning_rate": LEARNING_RATE,
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "lr_scheduler_type": LR_SCHEDULER_TYPE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "training_time_seconds": round(training_time, 1),
    }
    config_path = os.path.join(save_dir, "training_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)
    print(f"  training_config.json 已儲存")

    # ---- 儲存 generation_samples.jsonl ----
    samples_path = os.path.join(save_dir, "generation_samples.jsonl")
    with open(samples_path, "w", encoding="utf-8") as f:
        for sample in generation_results:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"  generation_samples.jsonl 已儲存 ({len(generation_results)} 筆)")

    # ---- 生成 README.md ----
    readme_content = _generate_readme(adapter_mode, config_dict)
    readme_path = os.path.join(save_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"  README.md 已儲存")

    return save_dir


def _generate_readme(adapter_mode: bool, config: dict) -> str:
    """生成 HuggingFace model card README.md。"""

    if adapter_mode:
        model_type_desc = "LoRA Adapter (SFT + ORPO 累積更新)"
        load_instructions = f"""\
## 如何載入模型

此模型是 LoRA adapter，包含 SFT + ORPO 的累積更新。

### 使用 Unsloth（推薦）

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{HF_REPO_ID}",
    max_seq_length={MAX_SEQ_LENGTH},
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
```

### 使用 PEFT

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model = AutoPeftModelForCausalLM.from_pretrained(
    "{HF_REPO_ID}",
    load_in_4bit=True,
)
tokenizer = AutoTokenizer.from_pretrained("{HF_REPO_ID}")
```

> **注意**: 此 adapter 的 base model 為 `{BASE_MODEL_NAME}`。
> adapter 包含 SFT 階段與 ORPO 階段的累積權重更新。"""
    else:
        model_type_desc = "LoRA Adapter (ORPO，基於已合併 SFT 模型)"
        load_instructions = f"""\
## 如何載入模型

此模型是 LoRA adapter，基於已合併的 SFT 模型做 ORPO。

### 使用 Unsloth（推薦）

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{HF_REPO_ID}",
    max_seq_length={MAX_SEQ_LENGTH},
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
```

> **注意**: 此 adapter 的 base model 為已合併 SFT 模型 `{SFT_MODEL_PATH}`，
> 原始 base model 為 `{BASE_MODEL_NAME}`。"""

    readme = f"""\
---
base_model: {BASE_MODEL_NAME}
tags:
  - translation
  - en-zh
  - orpo
  - lora
  - unsloth
  - qwen3
language:
  - en
  - zh
license: apache-2.0
---

# Qwen3-4B 英翻中小說翻譯模型 (ORPO)

## 概述

- **模型類型**: {model_type_desc}
- **基礎模型**: `{BASE_MODEL_NAME}`
- **SFT 模型來源**: `{SFT_MODEL_PATH}`
- **訓練方法**: ORPO (Odds Ratio Preference Optimization)
- **任務**: 英文到中文小說翻譯

## 訓練流程

1. **SFT 階段**: 使用英中平行小說語料做 LoRA SFT
2. **ORPO 階段**: 使用人工譯文 (chosen) vs. 機器翻譯 (rejected) 做偏好優化

## ORPO 資料格式

```json
{{
  "prompt": "英文原文",
  "chosen": "人工譯文（高品質）",
  "rejected": "機器翻譯（低品質）"
}}
```

## 翻譯模板

```
### English
{{english_text}}
### Chinese
{{chinese_translation}}
```

## 主要訓練參數

| 參數 | 值 |
|------|-----|
| ORPO beta | {config.get('orpo_beta', ORPO_BETA)} |
| Learning Rate | {config.get('learning_rate', LEARNING_RATE)} |
| Epochs | {config.get('num_train_epochs', NUM_TRAIN_EPOCHS)} |
| Batch Size | {config.get('per_device_train_batch_size', PER_DEVICE_TRAIN_BATCH_SIZE)} |
| Gradient Accumulation | {config.get('gradient_accumulation_steps', GRADIENT_ACCUMULATION_STEPS)} |
| Max Sequence Length | {config.get('max_seq_length', MAX_SEQ_LENGTH)} |
| LoRA r | {config.get('lora_r', LORA_R)} |
| LoRA alpha | {config.get('lora_alpha', LORA_ALPHA)} |

{load_instructions}

## 使用範例

```python
prompt = "### English\\n{{your_english_text}}\\n### Chinese\\n"
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
result = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(result)
```
"""
    return readme


# ============================================================================
# 10. 上傳 Hugging Face Hub
# ============================================================================
def push_to_hub(save_dir: str):
    """將模型上傳至 Hugging Face Hub。"""
    if not PUSH_TO_HUB or not HF_REPO_ID:
        print("\n  跳過 HuggingFace Hub 上傳 (PUSH_TO_HUB=False 或 HF_REPO_ID 為空)")
        return

    print("\n" + "=" * 60)
    print(f"上傳至 Hugging Face Hub: {HF_REPO_ID}")
    print("=" * 60)

    # 取得 token（優先環境變數，其次 huggingface-cli 登入狀態）
    hf_token = os.environ.get("HF_TOKEN", None)

    try:
        from huggingface_hub import HfApi

        api = HfApi(token=hf_token)

        # 建立 repo（若不存在）
        api.create_repo(repo_id=HF_REPO_ID, exist_ok=True, private=PRIVATE)

        # 排除不必要的檔案
        ignore_patterns = [
            "optimizer.pt",
            "scheduler.pt",
            "training_args.bin",
            "trainer_state.json",
            "*.safetensors.index.json",  # 保留 safetensors 本身
            "global_step*",
            "checkpoint-*",
            "runs/*",
        ]

        api.upload_folder(
            folder_path=save_dir,
            repo_id=HF_REPO_ID,
            commit_message="Upload ORPO fine-tuned model for EN→ZH novel translation",
            token=hf_token,
            ignore_patterns=ignore_patterns,
        )
        print(f"  ✓ 上傳成功: https://huggingface.co/{HF_REPO_ID}")

    except Exception as e:
        print(f"  ✗ 上傳失敗: {e}")
        print(f"    本地模型已保存於: {save_dir}")
        print(f"    請確認已執行 `huggingface-cli login` 或設定環境變數 HF_TOKEN。")


# ============================================================================
# MAIN
# ============================================================================
def main():
    training_start_time = time.time()

    # ==================================================================
    # 1. 載入 ORPO 資料
    # ==================================================================
    if GENERATE_FAKE_DATA:
        print("\n⚠ 使用假測試資料！")
        raw_data = generate_fake_data(FAKE_DATA_SIZE)
    else:
        raw_data = load_orpo_data(ORPO_DATA_PATH)

    # ==================================================================
    # 2. 標準化欄位
    # ==================================================================
    raw_data = standardize_fields(raw_data)

    # ==================================================================
    # 3. 清洗資料
    # ==================================================================
    cleaned_data = clean_data(raw_data)

    if len(cleaned_data) == 0:
        raise ValueError("清洗後資料為空！請檢查 ORPO 資料檔案。")

    # ==================================================================
    # 4. 載入模型（需要 tokenizer 做後續處理）
    # ==================================================================
    model, tokenizer, adapter_mode = load_model_for_orpo(
        base_model_name=BASE_MODEL_NAME,
        sft_model_path=SFT_MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        dtype=DTYPE,
        seed=SEED,
    )

    eos_token = tokenizer.eos_token or ""
    print(f"\n  EOS token: {repr(eos_token)}")

    # ==================================================================
    # 5. 格式化資料（套用模板 + 加 EOS）
    # ==================================================================
    formatted_data = format_for_orpo(cleaned_data, eos_token)

    # ==================================================================
    # 6. Token 長度統計 & 過濾
    # ==================================================================
    filtered_data = compute_token_stats_and_filter(
        formatted_data, tokenizer, MAX_SEQ_LENGTH
    )

    if len(filtered_data) == 0:
        raise ValueError("過濾後資料為空！請增大 MAX_SEQ_LENGTH 或檢查資料。")

    # ==================================================================
    # 7. 去重
    # ==================================================================
    deduped_data = dedup_by_prompt(filtered_data)

    # ==================================================================
    # 8. 資料切分
    # ==================================================================
    train_data, val_data, test_data = split_data(
        deduped_data,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=SEED,
    )

    # 計算每個 split 的平均 token 數
    print("\n  各 split 平均 token 數:")
    for name, split in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
        if len(split) == 0:
            continue
        c_lens = [len(tokenizer.encode(d["chosen"], add_special_tokens=False)) for d in split]
        r_lens = [len(tokenizer.encode(d["rejected"], add_special_tokens=False)) for d in split]
        print(f"    {name}: chosen avg={np.mean(c_lens):.0f}, rejected avg={np.mean(r_lens):.0f}")

    # ==================================================================
    # 9. 轉換為 HuggingFace Dataset
    # ==================================================================
    from datasets import Dataset as HFDataset

    train_dataset = HFDataset.from_list(train_data)
    val_dataset   = HFDataset.from_list(val_data)
    test_dataset  = HFDataset.from_list(test_data)

    print(f"\n  HF Dataset 建立完成:")
    print(f"    Train:      {len(train_dataset)}")
    print(f"    Validation: {len(val_dataset)}")
    print(f"    Test:       {len(test_dataset)}")

    # ==================================================================
    # 10. PatchDPOTrainer (Unsloth 優化)
    # ==================================================================
    try:
        from unsloth import PatchDPOTrainer
        PatchDPOTrainer()
        print("\n  ✓ PatchDPOTrainer 已套用 (Unsloth 優化)")
    except ImportError:
        print("\n  ⚠ PatchDPOTrainer 不可用，使用原生 ORPOTrainer")
    except Exception as e:
        print(f"\n  ⚠ PatchDPOTrainer 失敗: {e}")

    # ==================================================================
    # 11. 建構 ORPOConfig
    # ==================================================================
    orpo_config = build_orpo_config()

    # ==================================================================
    # 12. 建構 ORPOTrainer
    # ==================================================================
    from trl import ORPOTrainer

    # Callbacks
    callbacks = []
    try:
        gen_cb = ORPOGenerationCallback(
            tokenizer=tokenizer,
            val_data=val_data,
            max_new_tokens=MAX_NEW_TOKENS_GEN,
        )
        callbacks.append(gen_cb)
    except Exception as e:
        print(f"  ⚠ 無法建立 ORPOGenerationCallback: {e}")

    trainer = ORPOTrainer(
        model=model,
        args=orpo_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    # ==================================================================
    # 13. 開始 ORPO 訓練
    # ==================================================================
    print("\n" + "=" * 60)
    print("開始 ORPO 訓練")
    print(f"  epochs={NUM_TRAIN_EPOCHS}, batch_size={PER_DEVICE_TRAIN_BATCH_SIZE}, "
          f"grad_accum={GRADIENT_ACCUMULATION_STEPS}")
    print(f"  lr={LEARNING_RATE}, beta={ORPO_BETA}")
    print("=" * 60)

    trainer.train()

    training_time = time.time() - training_start_time
    print(f"\n  訓練耗時: {training_time/60:.1f} 分鐘")

    # ==================================================================
    # 14. 訓練後生成樣本（Validation Set）
    # ==================================================================
    print("\n" + "=" * 60)
    print("訓練後生成樣本 (Validation Set)")
    print("=" * 60)
    val_results = generate_orpo_samples(
        model=model,
        tokenizer=tokenizer,
        data=val_data,
        num_samples=NUM_SAMPLES_GEN,
        max_new_tokens=MAX_NEW_TOKENS_GEN,
    )

    # 訓練後生成樣本（Test Set）
    test_results = []
    if len(test_data) > 0:
        print("\n" + "=" * 60)
        print("訓練後生成樣本 (Test Set)")
        print("=" * 60)
        test_results = generate_orpo_samples(
            model=model,
            tokenizer=tokenizer,
            data=test_data,
            num_samples=min(NUM_SAMPLES_GEN, len(test_data)),
            max_new_tokens=MAX_NEW_TOKENS_GEN,
        )

    all_results = val_results + test_results

    # ==================================================================
    # 15. 保存模型、配置、README
    # ==================================================================
    save_dir = save_all(
        model=model,
        tokenizer=tokenizer,
        generation_results=all_results,
        adapter_mode=adapter_mode,
        training_start_time=training_start_time,
    )

    # ==================================================================
    # 16. 上傳 Hugging Face Hub
    # ==================================================================
    push_to_hub(save_dir)

    # ==================================================================
    # 17. 完成
    # ==================================================================
    print("\n" + "=" * 60)
    print("ORPO 訓練完成！")
    print(f"  本地模型: {save_dir}")
    if PUSH_TO_HUB and HF_REPO_ID:
        print(f"  HuggingFace: https://huggingface.co/{HF_REPO_ID}")
    print("=" * 60)

    return model, tokenizer, trainer


# ============================================================================
# 入口
# ============================================================================
if __name__ == "__main__":
    model, tokenizer, trainer = main()

    # 訓練結束後的推理範例
    print("\n" + "=" * 60)
    print("推理範例")
    print("=" * 60)
    sample_en = "The wind howled through the ancient forest, bending the tall pines."
    result = translate_english_to_chinese(sample_en, model, tokenizer)
    print(f"  English: {sample_en}")
    print(f"  Chinese: {result}")
