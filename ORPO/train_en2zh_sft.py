# ============================================================================
# 英中小說翻譯 LoRA SFT 訓練腳本
# 模型：unsloth/Qwen3-4B-Base-unsloth-bnb-4bit
# ============================================================================
# 建議安裝指令（請依環境調整）：
# pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# pip install torch transformers datasets trl peft bitsandbytes accelerate
# pip install scikit-learn pandas numpy huggingface_hub
# pip install sacrebleu  # 可選，用於自動翻譯指標
# ============================================================================

import os
import json
import random
import warnings
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)

# ============================================================================
# 訓練配置
# ============================================================================
@dataclass
class TrainingConfig:
    """所有可調整的訓練超參數與設定。"""

    # --- 模型 ---
    model_name: str = "unsloth/Qwen3-4B-Base-unsloth-bnb-4bit"
    load_in_4bit: bool = True
    max_seq_length: int = 2048
    dtype: Optional[str] = None  # None = 自動偵測

    # --- LoRA ---
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # --- 資料過濾 ---
    filter_outlier_alignment: bool = True
    min_token_ratio: float = 0.05   # zh_tokens / en_tokens 最小比例
    max_token_ratio: float = 20.0   # zh_tokens / en_tokens 最大比例
    filter_by_token_ratio: bool = True

    # --- 資料切分 ---
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    seed: int = 42

    # --- 訓練 ---
    output_dir: str = "./qwen3-4b-en2zh-sft"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_8bit"
    logging_steps: int = 5
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    report_to: str = "none"

    # --- 推理 / 生成 ---
    max_new_tokens_gen: int = 512
    num_samples_gen: int = 3

    # --- Hugging Face Hub ---
    hf_repo_id: str = ""           # 上傳的 repo，例如 "your-username/qwen3-4b-en2zh-lora"
    hf_token: Optional[str] = None # 若為 None 則使用 huggingface-cli login 的 token
    push_to_hub: bool = False      # 設為 True 啟用上傳

    # --- CSV 路徑 ---
    csv_path: str = ""             # 若為空，則假設 df 已在全域變數中


# ============================================================================
# 模板常量
# ============================================================================
PROMPT_TEMPLATE = "### English\n{en_text}\n### Chinese\n"
COMPLETION_TEMPLATE = "{zh_text}{eos}"
FULL_TEMPLATE = PROMPT_TEMPLATE + COMPLETION_TEMPLATE


# ============================================================================
# 1. 載入模型與 Tokenizer（Unsloth）
# ============================================================================
def load_model_and_tokenizer(cfg: TrainingConfig):
    """使用 Unsloth 載入 4-bit 量化模型並加入 LoRA。"""
    from unsloth import FastLanguageModel

    print("=" * 60)
    print("載入模型與 Tokenizer ...")
    print(f"  模型名稱: {cfg.model_name}")
    print(f"  4-bit 量化: {cfg.load_in_4bit}")
    print(f"  最大序列長度: {cfg.max_seq_length}")
    print("=" * 60)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        dtype=None if cfg.dtype is None else getattr(torch, cfg.dtype),
        load_in_4bit=cfg.load_in_4bit,
    )

    # 加入 LoRA adapter
    use_gc = "unsloth"
    try:
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias=cfg.lora_bias,
            target_modules=cfg.lora_target_modules,
            use_gradient_checkpointing=use_gc,
            random_state=cfg.seed,
        )
        print(f"  gradient checkpointing: '{use_gc}'")
    except Exception as e:
        warnings.warn(
            f"use_gradient_checkpointing='unsloth' 不支援 ({e})，降級為 True"
        )
        use_gc = True
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias=cfg.lora_bias,
            target_modules=cfg.lora_target_modules,
            use_gradient_checkpointing=use_gc,
            random_state=cfg.seed,
        )
        print(f"  gradient checkpointing: {use_gc}")

    # 設定 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print("  pad_token 設為 eos_token")

    # 訓練時 padding_side = right
    tokenizer.padding_side = "right"

    model.print_trainable_parameters()
    return model, tokenizer


# ============================================================================
# 2. 資料前處理
# ============================================================================
def preprocess_dataframe(df: pd.DataFrame, cfg: TrainingConfig) -> pd.DataFrame:
    """清洗與過濾 DataFrame。"""
    print("\n" + "=" * 60)
    print("資料前處理")
    print("=" * 60)
    n_original = len(df)
    print(f"  原始資料筆數: {n_original}")

    # 刪除 NaN
    df = df.dropna(subset=["en_text", "zh_text"]).copy()
    print(f"  移除 en_text/zh_text NaN 後: {len(df)}")

    # strip
    df["en_text"] = df["en_text"].astype(str).str.strip()
    df["zh_text"] = df["zh_text"].astype(str).str.strip()

    # 移除空字串
    df = df[(df["en_text"] != "") & (df["zh_text"] != "")]
    print(f"  移除空字串後: {len(df)}")

    # 過濾 outlier_alignment
    if cfg.filter_outlier_alignment and "outlier_alignment" in df.columns:
        n_before = len(df)
        df = df[~df["outlier_alignment"].astype(bool)]
        print(f"  過濾 outlier_alignment 後: {len(df)} (移除 {n_before - len(df)})")

    # 過濾 token 數量 <= 0
    if "en_tokens" in df.columns and "zh_tokens" in df.columns:
        n_before = len(df)
        df = df[(df["en_tokens"] > 0) & (df["zh_tokens"] > 0)]
        print(f"  過濾 tokens <= 0 後: {len(df)} (移除 {n_before - len(df)})")

        # 極端長度比例過濾
        if cfg.filter_by_token_ratio:
            ratio = df["zh_tokens"] / df["en_tokens"]
            n_before = len(df)
            df = df[(ratio >= cfg.min_token_ratio) & (ratio <= cfg.max_token_ratio)]
            print(
                f"  過濾極端長度比例 "
                f"[{cfg.min_token_ratio}, {cfg.max_token_ratio}] 後: "
                f"{len(df)} (移除 {n_before - len(df)})"
            )

    print(f"\n  清洗後資料筆數: {len(df)} (原始: {n_original})")
    df = df.reset_index(drop=True)
    return df


def compute_token_lengths(df: pd.DataFrame, tokenizer, cfg: TrainingConfig):
    """計算每筆資料模板化後的 token 長度，過濾超長樣本，並印出分布。"""
    print("\n" + "=" * 60)
    print("計算 token 長度分布")
    print("=" * 60)

    eos = tokenizer.eos_token or ""
    lengths = []
    for _, row in df.iterrows():
        text = FULL_TEMPLATE.format(
            en_text=row["en_text"],
            zh_text=row["zh_text"],
            eos=eos,
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        lengths.append(len(ids))

    df = df.copy()
    df["_total_tokens"] = lengths
    arr = np.array(lengths)

    print(f"  min:  {int(np.min(arr))}")
    print(f"  p50:  {int(np.percentile(arr, 50))}")
    print(f"  p90:  {int(np.percentile(arr, 90))}")
    print(f"  p95:  {int(np.percentile(arr, 95))}")
    print(f"  p99:  {int(np.percentile(arr, 99))}")
    print(f"  max:  {int(np.max(arr))}")

    # 過濾超長樣本（不截斷）
    n_before = len(df)
    df = df[df["_total_tokens"] <= cfg.max_seq_length].copy()
    n_filtered = n_before - len(df)
    print(f"\n  超過 MAX_SEQ_LENGTH ({cfg.max_seq_length}) 的樣本數: {n_filtered}")
    print(f"  過濾後剩餘: {len(df)}")

    df = df.drop(columns=["_total_tokens"]).reset_index(drop=True)
    return df


# ============================================================================
# 3. 資料切分（按 chapter_id 分組）
# ============================================================================
def split_by_chapter(
    df: pd.DataFrame,
    cfg: TrainingConfig,
):
    """
    按 chapter_id 分組切分，保證同一 chapter 不跨 split。
    若 chapter 數量不足 3 則退化為隨機切分。
    """
    print("\n" + "=" * 60)
    print("資料切分")
    print("=" * 60)

    rng = random.Random(cfg.seed)

    if "chapter_id" not in df.columns:
        warnings.warn("DataFrame 缺少 chapter_id 欄位，退化為隨機切分。")
        return _random_split(df, cfg, rng)

    unique_chapters = df["chapter_id"].unique().tolist()
    n_chapters = len(unique_chapters)
    print(f"  唯一 chapter 數量: {n_chapters}")

    if n_chapters < 3:
        warnings.warn(
            f"chapter 數量 ({n_chapters}) 不足 3，無法做分組切分，退化為隨機切分！"
        )
        return _random_split(df, cfg, rng)

    # 打亂 chapters
    rng.shuffle(unique_chapters)

    # 計算每個 chapter 的樣本數
    chapter_counts = df.groupby("chapter_id").size().to_dict()
    total = len(df)
    target_test = int(total * cfg.test_ratio)
    target_val = int(total * cfg.val_ratio)

    test_chapters, val_chapters, train_chapters = [], [], []
    test_count, val_count = 0, 0

    for ch in unique_chapters:
        cnt = chapter_counts[ch]
        if test_count < target_test:
            test_chapters.append(ch)
            test_count += cnt
        elif val_count < target_val:
            val_chapters.append(ch)
            val_count += cnt
        else:
            train_chapters.append(ch)

    train_df = df[df["chapter_id"].isin(set(train_chapters))].reset_index(drop=True)
    val_df = df[df["chapter_id"].isin(set(val_chapters))].reset_index(drop=True)
    test_df = df[df["chapter_id"].isin(set(test_chapters))].reset_index(drop=True)

    _print_split_info(train_df, val_df, test_df, train_chapters, val_chapters, test_chapters)
    _check_no_overlap(train_df, val_df, test_df)

    return train_df, val_df, test_df


def _random_split(df: pd.DataFrame, cfg: TrainingConfig, rng: random.Random):
    """退化隨機切分。"""
    indices = list(range(len(df)))
    rng.shuffle(indices)
    total = len(df)
    n_test = int(total * cfg.test_ratio)
    n_val = int(total * cfg.val_ratio)

    test_idx = indices[:n_test]
    val_idx = indices[n_test : n_test + n_val]
    train_idx = indices[n_test + n_val :]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    _print_split_info(train_df, val_df, test_df)
    return train_df, val_df, test_df


def _print_split_info(train_df, val_df, test_df, train_ch=None, val_ch=None, test_ch=None):
    total = len(train_df) + len(val_df) + len(test_df)
    print(f"\n  Train:      {len(train_df):>6} 筆 ({len(train_df)/total*100:.1f}%)")
    print(f"  Validation: {len(val_df):>6} 筆 ({len(val_df)/total*100:.1f}%)")
    print(f"  Test:       {len(test_df):>6} 筆 ({len(test_df)/total*100:.1f}%)")
    print(f"  Total:      {total:>6} 筆")
    if train_ch is not None:
        print(f"\n  Train chapters:      {len(train_ch)}")
        print(f"  Validation chapters: {len(val_ch)}")
        print(f"  Test chapters:       {len(test_ch)}")


def _check_no_overlap(train_df, val_df, test_df):
    """確認各 split 的 chapter_id 無交集。"""
    if "chapter_id" not in train_df.columns:
        return
    train_ch = set(train_df["chapter_id"].unique())
    val_ch = set(val_df["chapter_id"].unique())
    test_ch = set(test_df["chapter_id"].unique())

    overlap_tv = train_ch & val_ch
    overlap_tt = train_ch & test_ch
    overlap_vt = val_ch & test_ch

    if overlap_tv or overlap_tt or overlap_vt:
        msg = (
            f"chapter_id 出現交集！\n"
            f"  train ∩ val:  {overlap_tv}\n"
            f"  train ∩ test: {overlap_tt}\n"
            f"  val ∩ test:   {overlap_vt}"
        )
        raise ValueError(msg)
    print("  ✓ 各 split 的 chapter_id 無交集")


# ============================================================================
# 4. Dataset
# ============================================================================
class TranslationSFTDataset(Dataset):
    """
    每筆資料包含 input_ids, attention_mask, labels。
    labels 中 prompt 部分全設為 -100，僅對 completion 計算 loss。
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_seq_length: int):
        self.samples = []
        eos = tokenizer.eos_token or ""

        for _, row in df.iterrows():
            en_text = row["en_text"]
            zh_text = row["zh_text"]

            prompt_str = PROMPT_TEMPLATE.format(en_text=en_text)
            completion_str = COMPLETION_TEMPLATE.format(zh_text=zh_text, eos=eos)

            prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
            completion_ids = tokenizer.encode(completion_str, add_special_tokens=False)

            input_ids = prompt_ids + completion_ids
            # 截斷（理論上已過濾，但保險起見）
            input_ids = input_ids[:max_seq_length]

            attention_mask = [1] * len(input_ids)

            # labels: prompt 部分 = -100
            labels = [-100] * len(prompt_ids) + completion_ids
            labels = labels[:max_seq_length]

            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

        # 同時保存原始文本，方便生成樣本時使用
        self.en_texts = df["en_text"].tolist()
        self.zh_texts = df["zh_text"].tolist()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {k: torch.tensor(v, dtype=torch.long) for k, v in self.samples[idx].items()}


# ============================================================================
# 5. 生成樣本函式
# ============================================================================
def generate_samples(
    model,
    tokenizer,
    dataset: TranslationSFTDataset,
    num_samples: int = 3,
    max_new_tokens: int = 512,
):
    """從 dataset 取前 num_samples 筆，使用 greedy decoding 生成翻譯。"""
    # 嘗試切換為推理模式
    try:
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)
    except Exception:
        model.eval()

    device = next(model.parameters()).device
    num_samples = min(num_samples, len(dataset))

    print("\n" + "=" * 60)
    print(f"生成樣本 (共 {num_samples} 筆)")
    print("=" * 60)

    for i in range(num_samples):
        en_text = dataset.en_texts[i]
        zh_ref = dataset.zh_texts[i]

        prompt_str = PROMPT_TEMPLATE.format(en_text=en_text)
        input_ids = tokenizer.encode(prompt_str, add_special_tokens=False, return_tensors="pt")
        input_ids = input_ids.to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # 只取生成的部分
        generated_ids = output_ids[0][input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # 清除模板標記
        for marker in ["### English", "### Chinese"]:
            generated_text = generated_text.replace(marker, "")
        if tokenizer.eos_token:
            generated_text = generated_text.replace(tokenizer.eos_token, "")
        generated_text = generated_text.strip()

        print(f"\n--- 樣本 {i + 1} ---")
        print(f"[英文原文]\n{en_text[:300]}{'...' if len(en_text) > 300 else ''}")
        print(f"\n[參考中文]\n{zh_ref[:300]}{'...' if len(zh_ref) > 300 else ''}")
        print(f"\n[模型生成]\n{generated_text[:300]}{'...' if len(generated_text) > 300 else ''}")

    # 嘗試計算 BLEU（可選）
    try:
        import sacrebleu
        refs_all = []
        hyps_all = []
        for i in range(num_samples):
            en_text = dataset.en_texts[i]
            zh_ref = dataset.zh_texts[i]
            prompt_str = PROMPT_TEMPLATE.format(en_text=en_text)
            input_ids = tokenizer.encode(prompt_str, add_special_tokens=False, return_tensors="pt").to(device)
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids, max_new_tokens=max_new_tokens,
                    do_sample=False, pad_token_id=tokenizer.pad_token_id,
                )
            gen_ids = output_ids[0][input_ids.shape[1]:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            for m in ["### English", "### Chinese"]:
                gen_text = gen_text.replace(m, "")
            if tokenizer.eos_token:
                gen_text = gen_text.replace(tokenizer.eos_token, "")
            gen_text = gen_text.strip()
            refs_all.append(zh_ref)
            hyps_all.append(gen_text)

        bleu = sacrebleu.corpus_bleu(hyps_all, [refs_all], tokenize="zh")
        print(f"\n  sacrebleu (zh tokenize): {bleu.score:.2f}")
    except ImportError:
        print("\n  （sacrebleu 未安裝，跳過 BLEU 計算）")
    except Exception as e:
        print(f"\n  （BLEU 計算失敗: {e}）")


# ============================================================================
# 6. TrainerCallback: 每個 epoch 結束生成樣本
# ============================================================================
class GenerateSamplesCallback(TrainerCallback):
    """在每個 epoch 結束時生成 1 筆翻譯樣本。"""

    def __init__(self, tokenizer, val_dataset: TranslationSFTDataset, max_new_tokens: int = 512):
        self.tokenizer = tokenizer
        self.val_dataset = val_dataset
        self.max_new_tokens = max_new_tokens

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        try:
            generate_samples(
                model=model,
                tokenizer=self.tokenizer,
                dataset=self.val_dataset,
                num_samples=1,
                max_new_tokens=self.max_new_tokens,
            )
        except Exception as e:
            print(f"  [GenerateSamplesCallback] 生成失敗: {e}")
        # 切回訓練模式
        model.train()


# ============================================================================
# 7. 推理函式
# ============================================================================
def translate_english_to_chinese(
    text: str,
    model,
    tokenizer,
    max_new_tokens: int = 512,
) -> str:
    """將英文翻譯為中文。"""
    # 嘗試切換推理模式
    try:
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)
    except Exception:
        model.eval()

    device = next(model.parameters()).device
    prompt_str = PROMPT_TEMPLATE.format(en_text=text)
    input_ids = tokenizer.encode(prompt_str, add_special_tokens=False, return_tensors="pt").to(device)

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
    generated_text = generated_text.strip()

    return generated_text


# ============================================================================
# 8. 儲存模型與配置
# ============================================================================
def save_model_and_config(model, tokenizer, cfg: TrainingConfig):
    """儲存 LoRA adapter、tokenizer 與訓練配置。"""
    save_dir = os.path.join(cfg.output_dir, "final_lora")
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"儲存模型至: {save_dir}")
    print("=" * 60)

    # 嘗試多種方式儲存 LoRA adapter
    saved = False
    # 方法 1: Unsloth save_lora (部分版本可用)
    if not saved:
        try:
            model.save_lora(save_dir)
            print("  使用 model.save_lora() 儲存成功")
            saved = True
        except (AttributeError, Exception) as e:
            print(f"  model.save_lora() 不可用: {e}")

    # 方法 2: save_pretrained (PEFT 標準方法)
    if not saved:
        try:
            model.save_pretrained(save_dir)
            print("  使用 model.save_pretrained() 儲存成功")
            saved = True
        except Exception as e:
            print(f"  model.save_pretrained() 失敗: {e}")

    # 方法 3: 取得 peft_model 再存
    if not saved:
        try:
            peft_model = model.base_model if hasattr(model, "base_model") else model
            peft_model.save_pretrained(save_dir)
            print("  使用 base_model.save_pretrained() 儲存成功")
            saved = True
        except Exception as e:
            raise RuntimeError(f"所有模型儲存方式均失敗，最後錯誤: {e}")

    # 儲存 tokenizer
    tokenizer.save_pretrained(save_dir)
    print(f"  Tokenizer 已儲存至: {save_dir}")

    # 儲存訓練配置
    config_dict = asdict(cfg)
    config_dict["template"] = FULL_TEMPLATE
    config_path = os.path.join(save_dir, "training_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)
    print(f"  訓練配置已儲存至: {config_path}")

    return save_dir


def push_to_huggingface(save_dir: str, cfg: TrainingConfig):
    """將 LoRA adapter 上傳至 Hugging Face Hub。"""
    if not cfg.push_to_hub or not cfg.hf_repo_id:
        print("\n  跳過 Hugging Face Hub 上傳 (push_to_hub=False 或 hf_repo_id 為空)")
        return

    print("\n" + "=" * 60)
    print(f"上傳至 Hugging Face Hub: {cfg.hf_repo_id}")
    print("=" * 60)

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=cfg.hf_token)
        api.create_repo(repo_id=cfg.hf_repo_id, exist_ok=True, private=True)
        api.upload_folder(
            folder_path=save_dir,
            repo_id=cfg.hf_repo_id,
            commit_message="Upload LoRA adapter from en2zh SFT training",
            token=cfg.hf_token,
        )
        print(f"  ✓ 上傳成功: https://huggingface.co/{cfg.hf_repo_id}")
    except Exception as e:
        print(f"  ✗ 上傳失敗: {e}")
        print("    請確認已執行 `huggingface-cli login` 或提供 hf_token。")


# ============================================================================
# 9. 建構 TrainingArguments（含版本兼容處理）
# ============================================================================
def build_training_arguments(cfg: TrainingConfig) -> TrainingArguments:
    """建構 TrainingArguments，處理新舊版本 API 差異。"""
    # 偵測 bf16 支援
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    # 基礎參數
    kwargs = dict(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        optim=cfg.optim,
        bf16=bf16_supported,
        fp16=(not bf16_supported and torch.cuda.is_available()),
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        seed=cfg.seed,
        report_to=cfg.report_to,
        remove_unused_columns=False,
    )

    # 兼容 evaluation_strategy / eval_strategy
    try:
        # 新版 transformers >= 4.36 使用 eval_strategy
        kwargs["eval_strategy"] = "epoch"
        args = TrainingArguments(**kwargs)
    except TypeError:
        # 舊版使用 evaluation_strategy
        del kwargs["eval_strategy"]
        kwargs["evaluation_strategy"] = "epoch"
        args = TrainingArguments(**kwargs)

    print(f"\n  bf16={bf16_supported}, fp16={not bf16_supported and torch.cuda.is_available()}")
    return args


# ============================================================================
# MAIN
# ============================================================================
def main(df: pd.DataFrame, cfg: Optional[TrainingConfig] = None):
    """
    主訓練流程。
    參數:
        df: 包含 en_text, zh_text, chapter_id 等欄位的 DataFrame。
        cfg: 訓練配置，若為 None 則使用預設值。
    """
    if cfg is None:
        cfg = TrainingConfig()

    # ---- 1. 載入模型 ----
    model, tokenizer = load_model_and_tokenizer(cfg)

    # ---- 2. 資料前處理 ----
    df_clean = preprocess_dataframe(df, cfg)

    # ---- 3. 計算 token 長度 & 過濾超長 ----
    df_clean = compute_token_lengths(df_clean, tokenizer, cfg)

    # ---- 4. 資料切分 ----
    train_df, val_df, test_df = split_by_chapter(df_clean, cfg)

    # ---- 5. 建立 Dataset ----
    print("\n建立 Dataset ...")
    train_dataset = TranslationSFTDataset(train_df, tokenizer, cfg.max_seq_length)
    val_dataset = TranslationSFTDataset(val_df, tokenizer, cfg.max_seq_length)
    test_dataset = TranslationSFTDataset(test_df, tokenizer, cfg.max_seq_length)
    print(f"  Train dataset:      {len(train_dataset)}")
    print(f"  Validation dataset: {len(val_dataset)}")
    print(f"  Test dataset:       {len(test_dataset)}")

    # ---- 6. Data Collator ----
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    # ---- 7. Training Arguments ----
    training_args = build_training_arguments(cfg)

    # ---- 8. Callbacks ----
    callbacks = []
    try:
        gen_callback = GenerateSamplesCallback(
            tokenizer=tokenizer,
            val_dataset=val_dataset,
            max_new_tokens=cfg.max_new_tokens_gen,
        )
        callbacks.append(gen_callback)
    except Exception as e:
        print(f"  無法建立 GenerateSamplesCallback: {e}")

    # ---- 9. Trainer ----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # ---- 10. 訓練 ----
    print("\n" + "=" * 60)
    print("開始訓練")
    print("=" * 60)
    trainer.train()

    # ---- 11. 訓練後生成樣本 ----
    print("\n" + "=" * 60)
    print("訓練後生成樣本 (Validation Set)")
    print("=" * 60)
    generate_samples(
        model=model,
        tokenizer=tokenizer,
        dataset=val_dataset,
        num_samples=cfg.num_samples_gen,
        max_new_tokens=cfg.max_new_tokens_gen,
    )

    # ---- 12. 在 Test Set 生成樣本 ----
    if len(test_dataset) > 0:
        print("\n" + "=" * 60)
        print("訓練後生成樣本 (Test Set)")
        print("=" * 60)
        generate_samples(
            model=model,
            tokenizer=tokenizer,
            dataset=test_dataset,
            num_samples=min(cfg.num_samples_gen, len(test_dataset)),
            max_new_tokens=cfg.max_new_tokens_gen,
        )

    # ---- 13. 儲存模型 ----
    save_dir = save_model_and_config(model, tokenizer, cfg)

    # ---- 14. 上傳至 Hugging Face Hub ----
    push_to_huggingface(save_dir, cfg)

    print("\n" + "=" * 60)
    print("訓練完成！")
    print("=" * 60)

    return model, tokenizer, trainer


# ============================================================================
# 入口
# ============================================================================
if __name__ == "__main__":
    # ==================================================
    # ★ 請依實際需求修改以下配置 ★
    # ==================================================
    config = TrainingConfig(
        csv_path="./aligned_data.csv",     # CSV 檔案路徑
        max_seq_length=2048,               # 依 GPU 記憶體調整
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        push_to_hub=False,                 # 設為 True 並填入 hf_repo_id 以上傳
        hf_repo_id="",                     # 例如 "your-username/qwen3-4b-en2zh-lora"
    )

    # 載入 CSV
    if config.csv_path and os.path.exists(config.csv_path):
        df = pd.read_csv(config.csv_path)
        print(f"從 {config.csv_path} 載入 {len(df)} 筆資料")
    else:
        # 若在 notebook 中，假設 df 已經在全域變數
        if "df" not in dir():
            raise RuntimeError(
                "找不到 df！請設定 config.csv_path 或確保 df 已在全域變數中。"
            )
        print(f"使用現有 df，共 {len(df)} 筆資料")

    # 執行訓練
    model, tokenizer, trainer = main(df, config)

    # ==================================================
    # 訓練結束後的推理範例
    # ==================================================
    sample_en = "The wind howled through the ancient forest, bending the tall pines."
    result = translate_english_to_chinese(sample_en, model, tokenizer)
    print(f"\n[推理範例]")
    print(f"  English: {sample_en}")
    print(f"  Chinese: {result}")
