# =============================================================================
# train_sft_polished_unsloth.py
# 復現論文 "Lost in Literalism: How Supervised Training Shapes Translationese
# in LLMs" 的 SFT-Polished 方法
# 使用 Unsloth + LoRA 對 Base Model 進行 Completion-Only SFT 訓練
# =============================================================================
#
# 安裝依賴（Colab T4 環境）：
# pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# pip install --no-deps trl peft accelerate bitsandbytes
# pip install datasets huggingface_hub
#

import os
import sys
import json
import argparse
import warnings
from typing import Dict, List, Any

import torch
import numpy as np
from datasets import Dataset, DatasetDict
from tqdm import tqdm

# ── Unsloth 核心匯入 ──
from unsloth import FastLanguageModel

# ── HuggingFace 生態系 ──
from transformers import DataCollatorWithPadding
from trl import SFTTrainer, SFTConfig
from huggingface_hub import HfApi

warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# 1. 命令列參數解析
# =============================================================================
def parse_args() -> argparse.Namespace:
    """解析命令列參數"""
    parser = argparse.ArgumentParser(
        description="SFT-Polished 訓練腳本 — 復現 'Lost in Literalism' 論文方法"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="本地 JSONL 訓練資料路徑（每行格式：{instruction, input, output}）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./sft-polished-qwen3-4b",
        help="模型 checkpoint 儲存目錄（預設：./sft-polished-qwen3-4b）",
    )
    parser.add_argument(
        "--hub_repo_id",
        type=str,
        required=True,
        help="HuggingFace Hub Repo ID，例如 your-username/sft-polished-qwen3-4b",
    )
    return parser.parse_args()


# =============================================================================
# 2. 環境與前置檢查
# =============================================================================
def preflight_checks(args: argparse.Namespace) -> None:
    """
    檢查：
      - 資料路徑是否存在
      - HuggingFace Hub 登入狀態（使用者需預先執行 `huggingface-cli login`）
    """
    # 檢查資料路徑
    if not os.path.isfile(args.data_path):
        print(f"❌ 錯誤：資料路徑不存在 → {args.data_path}")
        sys.exit(1)
    print(f"✅ 資料路徑確認：{args.data_path}")

    # 檢查 HuggingFace Hub 登入狀態
    try:
        api = HfApi()
        user_info = api.whoami()
        print(f"✅ 已登入 HuggingFace Hub（使用者：{user_info['name']}）")
    except Exception:
        print("❌ 錯誤：尚未登入 HuggingFace Hub，請先執行 `huggingface-cli login`")
        sys.exit(1)


# =============================================================================
# 3. 載入 JSONL 資料 & 90/10 Train/Val Split
# =============================================================================
def load_and_split_data(data_path: str) -> DatasetDict:
    """
    讀取本地 JSONL 檔，每行格式 {"instruction": ..., "input": ..., "output": ...}
    並以 random seed=42 做 90/10 train/val split。
    """
    # 先計算總行數以顯示 tqdm 進度條
    with open(data_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for line in f if line.strip())

    records: List[Dict[str, str]] = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="📄 載入 JSONL 資料"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # 確保必要欄位存在
            if "instruction" not in obj or "output" not in obj:
                continue
            records.append({
                "instruction": obj["instruction"],
                "input": obj.get("input", ""),
                "output": obj["output"],
            })

    print(f"   成功載入 {len(records)} 筆資料")

    dataset = Dataset.from_list(records)
    # 90/10 split，seed=42 確保可重現
    split = dataset.train_test_split(test_size=0.1, seed=42)
    ds = DatasetDict({"train": split["train"], "validation": split["test"]})

    print(f"   ├─ 訓練集：{len(ds['train'])} 筆")
    print(f"   └─ 驗證集：{len(ds['validation'])} 筆")
    return ds


# =============================================================================
# 4. 載入模型與 Tokenizer（Unsloth API）
# =============================================================================
MAX_SEQ_LENGTH = 768    # 根據資料分布（平均 ~83 tokens），768 足以覆蓋所有樣本
MODEL_NAME = "unsloth/Qwen3-4B-Base-unsloth-bnb-4bit"


def load_model_and_tokenizer():
    """
    使用 Unsloth 的 FastLanguageModel 載入 4-bit 量化的 Base Model。
    論文使用 Base Model 而非 Instruct Model，以避免既有的 Chat Template
    對翻譯風格產生干擾。
    """
    print(f"\n🔧 正在載入模型：{MODEL_NAME}")
    print(f"   max_seq_length = {MAX_SEQ_LENGTH}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,          # 自動偵測最佳 dtype
        load_in_4bit=True,   # 使用 4-bit 量化以節省顯存
    )

    # ── 確保 pad_token 存在 ──
    # Base model 通常沒有設定 pad_token，這裡將其設為 eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        # 調整 embedding 大小以適應可能的新 token（此處 pad=eos 不會新增，但保險起見）
        model.resize_token_embeddings(len(tokenizer))
        print("   ℹ️  已將 pad_token 設為 eos_token")

    print(f"   ├─ vocab_size = {len(tokenizer)}")
    print(f"   ├─ pad_token  = '{tokenizer.pad_token}' (id={tokenizer.pad_token_id})")
    print(f"   └─ eos_token  = '{tokenizer.eos_token}' (id={tokenizer.eos_token_id})")

    return model, tokenizer


# =============================================================================
# 5. 注入 LoRA Adapter（Unsloth API）
# =============================================================================
def apply_lora(model):
    """
    使用 Unsloth 的 get_peft_model 注入 LoRA adapter。
    為減少 overfitting，僅對 attention 層注入 LoRA（移除 MLP 的 gate/up/down proj）。
    use_gradient_checkpointing="unsloth" 可大幅降低顯存佔用，適合 Colab T4 (16GB)。
    """
    print("\n🔧 正在注入 LoRA Adapter...")

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,                       # LoRA rank
        lora_alpha=32,              # LoRA scaling factor (alpha/r = 2)
        lora_dropout=0.05,          # Dropout 防止過擬合
        target_modules=[            # 僅對 attention 層注入 LoRA，降低過擬合風險
            "q_proj", "k_proj", "v_proj", "o_proj",
        ],
        bias="none",                # 不訓練 bias
        use_gradient_checkpointing="unsloth",  # Unsloth 優化的 gradient checkpointing
        random_state=42,            # 可重現性
    )

    # 印出可訓練參數量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ├─ 可訓練參數：{trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"   └─ 總參數量：  {total_params:,}")

    return model


# =============================================================================
# 6. Completion-Only Loss 的資料前處理（核心！）
# =============================================================================
# 【論文關鍵設計】
# "Lost in Literalism" 論文的 SFT-Polished 方法要求：
#   - Loss 只計算 output（翻譯結果）部分
#   - Prompt（instruction + input）部分的 label 設為 -100，不參與 loss 計算
#
# 這是因為如果讓模型同時學習預測 instruction，會導致：
#   1. 模型浪費容量在記憶 prompt 格式上
#   2. 翻譯品質下降，出現 "translationese"（翻譯腔）
#
# 由於使用的是 Base Model（沒有 Chat Template），我們必須手動實作
# Completion-Only Loss masking，而不能依賴 trl 的自動 response template 偵測。
# =============================================================================


def build_tokenize_fn(tokenizer, max_seq_length: int):
    """
    回傳一個用於 dataset.map 的 tokenize 函數。
    實作 Completion-Only Loss：
      - Prompt 部分的 labels = -100（不計算 loss）
      - Output 部分的 labels = 真實 token ids（計算 loss）
    """
    def tokenize_fn(example: Dict[str, str]) -> Dict[str, List[int]]:
        instruction = example["instruction"]
        inp = example["input"]
        output = example["output"]

        # ── 組合 prompt 與 output ──
        # Prompt: instruction + input，以換行分隔
        # Output: 翻譯結果 + eos_token（讓模型學會何時停止生成）
        if inp.strip():
            prompt_text = f"{instruction}\n{inp}\n"
        else:
            prompt_text = f"{instruction}\n"
        output_text = f"{output}{tokenizer.eos_token}"

        # ── 分別 tokenize ──
        # 注意：不加 special tokens，避免重複加入 bos/eos
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        output_ids = tokenizer.encode(output_text, add_special_tokens=False)

        # ── 拼接 input_ids ──
        input_ids = prompt_ids + output_ids

        # ── 截斷策略 ──
        # 如果總長度超過 max_seq_length，優先保留 output 尾部
        # （因為翻譯的結尾通常包含重要的上下文收束）
        if len(input_ids) > max_seq_length:
            # 計算需要截斷多少
            overflow = len(input_ids) - max_seq_length

            if overflow < len(prompt_ids):
                # 只截斷 prompt 的前面部分（保留 prompt 尾部的上下文）
                prompt_ids = prompt_ids[overflow:]
            else:
                # prompt 全部截掉還不夠，需要從 output 頭部截斷
                prompt_ids = []
                remaining = max_seq_length
                output_ids = output_ids[-remaining:]  # 保留 output 尾部

            input_ids = prompt_ids + output_ids

        prompt_len = len(prompt_ids)

        # ── 構建 labels：Completion-Only Loss 的關鍵 ──
        # Prompt 部分 → -100（PyTorch CrossEntropyLoss 會忽略 -100）
        # Output 部分 → 真實 token ids
        labels = [-100] * prompt_len + output_ids

        # ── attention_mask：所有 token 都是 1（padding 由 collator 處理）──
        attention_mask = [1] * len(input_ids)

        assert len(input_ids) == len(labels) == len(attention_mask), \
            f"長度不一致：ids={len(input_ids)}, labels={len(labels)}, mask={len(attention_mask)}"

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return tokenize_fn


# =============================================================================
# 6.5 Packing：將多個短序列打包成固定長度序列以提升 GPU 利用率
# =============================================================================
# 由於資料的平均序列長度僅 ~83 tokens，遠小於 max_seq_length=768，
# 若不做 packing，每個 batch 會有大量 padding 浪費算力。
# Packing 將多個樣本首尾拼接成 768 長度的序列，大幅提升訓練效率。
#
# 注意：Packing 後不同樣本會共享 attention（跨樣本 attend），
# 這在實務中對訓練品質影響很小，且能帶來 ~5-9x 的速度提升。
# =============================================================================

def pack_tokenized_dataset(dataset: Dataset, max_seq_length: int) -> Dataset:
    """
    將已 tokenize 的短序列打包（packing）成固定長度 max_seq_length 的序列。
    每個打包序列包含多個原始樣本，首尾相接。
    Labels 的 -100 masking 會被完整保留，確保 Completion-Only Loss 不受影響。
    """
    all_input_ids: List[List[int]] = []
    all_attention_mask: List[List[int]] = []
    all_labels: List[List[int]] = []

    # 當前正在組裝的序列
    current_input_ids: List[int] = []
    current_attention_mask: List[int] = []
    current_labels: List[int] = []

    for example in tqdm(dataset, desc="📦 Packing 序列"):
        ex_input_ids = example["input_ids"]
        ex_attention_mask = example["attention_mask"]
        ex_labels = example["labels"]

        # 如果加入這個樣本後會超過 max_seq_length，先儲存當前序列
        if len(current_input_ids) + len(ex_input_ids) > max_seq_length:
            if current_input_ids:
                all_input_ids.append(current_input_ids)
                all_attention_mask.append(current_attention_mask)
                all_labels.append(current_labels)
            current_input_ids = []
            current_attention_mask = []
            current_labels = []

        # 將當前樣本追加到組裝中的序列
        current_input_ids.extend(ex_input_ids)
        current_attention_mask.extend(ex_attention_mask)
        current_labels.extend(ex_labels)

    # 不要遺漏最後一段
    if current_input_ids:
        all_input_ids.append(current_input_ids)
        all_attention_mask.append(current_attention_mask)
        all_labels.append(current_labels)

    packed_dataset = Dataset.from_dict({
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
    })

    return packed_dataset


def preprocess_dataset(ds: DatasetDict, tokenizer) -> DatasetDict:
    """對訓練集和驗證集進行 tokenize 前處理，並對訓練集做 packing"""
    print("\n📝 正在進行資料前處理（Completion-Only Loss Tokenization）...")

    tokenize_fn = build_tokenize_fn(tokenizer, MAX_SEQ_LENGTH)

    # 移除原始文字欄位，只保留 tokenized 結果
    columns_to_remove = ["instruction", "input", "output"]

    ds_tokenized = ds.map(
        tokenize_fn,
        remove_columns=columns_to_remove,
        desc="🔤 Tokenizing",
        num_proc=1,  # Tokenizer 通常不是 thread-safe
    )

    # ── 印出 tokenize 後的統計資訊 ──
    for split_name in ["train", "validation"]:
        lengths = [len(x["input_ids"]) for x in ds_tokenized[split_name]]
        # 計算有多少 token 參與 loss（labels != -100）
        loss_token_counts = [
            sum(1 for l in x["labels"] if l != -100)
            for x in ds_tokenized[split_name]
        ]
        print(f"   [{split_name}] tokenize 後")
        print(f"     序列長度 — 平均: {np.mean(lengths):.0f}, "
              f"最大: {np.max(lengths)}, 最小: {np.min(lengths)}")
        print(f"     Loss tokens — 平均: {np.mean(loss_token_counts):.0f}, "
              f"佔比: {np.mean(loss_token_counts) / np.mean(lengths) * 100:.1f}%")

    # ── Packing 訓練集（驗證集不做 packing，保持逐筆評估）──
    print(f"\n📦 對訓練集進行 Packing（目標長度 = {MAX_SEQ_LENGTH}）...")
    train_before = len(ds_tokenized["train"])
    ds_tokenized["train"] = pack_tokenized_dataset(
        ds_tokenized["train"], MAX_SEQ_LENGTH
    )
    train_after = len(ds_tokenized["train"])
    print(f"   ├─ Packing 前：{train_before} 筆樣本")
    print(f"   ├─ Packing 後：{train_after} 筆打包序列")
    print(f"   └─ 壓縮率：{train_before / train_after:.1f}x（每個序列平均含 "
          f"{train_before / train_after:.1f} 個原始樣本）")

    # 印出 packing 後的序列長度分布
    packed_lengths = [len(x["input_ids"]) for x in ds_tokenized["train"]]
    print(f"   📊 打包後序列長度 — 平均: {np.mean(packed_lengths):.0f}, "
          f"最大: {np.max(packed_lengths)}, 最小: {np.min(packed_lengths)}")

    return ds_tokenized


# =============================================================================
# 7. 自定義 Data Collator（支援 labels 的動態 padding）
# =============================================================================
class CompletionOnlyDataCollator:
    """
    自定義 Data Collator：
      - 將同一 batch 內的序列 pad 到相同長度
      - input_ids 以 pad_token_id 填充
      - attention_mask 以 0 填充
      - labels 以 -100 填充（確保 padding 部分不參與 loss）
    """

    def __init__(self, tokenizer, max_length: int = MAX_SEQ_LENGTH):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 找到 batch 內最長序列的長度
        max_len = min(
            max(len(f["input_ids"]) for f in features),
            self.max_length,
        )

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for f in features:
            input_ids = f["input_ids"][:max_len]
            attention_mask = f["attention_mask"][:max_len]
            labels = f["labels"][:max_len]

            # 計算需要 padding 的長度
            pad_len = max_len - len(input_ids)

            # 右側 padding
            batch_input_ids.append(input_ids + [self.pad_token_id] * pad_len)
            batch_attention_mask.append(attention_mask + [0] * pad_len)
            batch_labels.append(labels + [-100] * pad_len)  # padding 不計算 loss

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


# =============================================================================
# 8. 配置訓練參數 & 建立 SFTTrainer
# =============================================================================
def create_trainer(
    model,
    tokenizer,
    ds_tokenized: DatasetDict,
    output_dir: str,
) -> SFTTrainer:
    """
    使用 TRL 的 SFTTrainer 建立訓練器。
    訓練超參數遵循論文 "Lost in Literalism" 的設定。
    """
    print("\n🏋️ 正在配置 SFTTrainer...")

    # ── 計算 warmup_steps（取代 warmup_ratio，相容新版 Unsloth/transformers）──
    # 注意：packing 後訓練集筆數已改變，這裡使用 packing 後的筆數計算
    num_train_samples = len(ds_tokenized["train"])
    steps_per_epoch = num_train_samples // 8  # per_device_train_batch_size=16
    total_steps = steps_per_epoch * 3           # num_train_epochs=3
    warmup_steps = int(total_steps * 0.1)       # 等效 warmup_ratio=0.1
    print(f"   ℹ️  計算 warmup_steps = {warmup_steps}（總步數 {total_steps} × 10%）")

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        warmup_steps=warmup_steps,          # 前 10% steps 做 warmup
        lr_scheduler_type="linear",          # 線性衰減
        weight_decay=0.0,
        max_grad_norm=1.0,                   # 梯度裁剪
        fp16=True,                           # Colab T4 (Turing) 不支援 bf16，使用 fp16
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=30,
        save_strategy="steps",
        save_steps=30,
        load_best_model_at_end=True,         # 訓練結束載入最佳 checkpoint
        metric_for_best_model="eval_loss",
        greater_is_better=False,             # eval_loss 越小越好
        save_total_limit=3,                  # 最多保留 3 個 checkpoint
        report_to="none",                    # 不使用 wandb 等外部 logger
        dataloader_pin_memory=True,
        remove_unused_columns=False,         # 我們已經手動處理好欄位了
        dataset_text_field=None,             # 不使用內建的文字欄位處理
        max_seq_length=MAX_SEQ_LENGTH,       # SFTConfig 需要指定
    )

    # 自定義 Data Collator
    data_collator = CompletionOnlyDataCollator(tokenizer, MAX_SEQ_LENGTH)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=ds_tokenized["train"],
        eval_dataset=ds_tokenized["validation"],
        data_collator=data_collator,
        # ── 關鍵：不使用 SFTTrainer 內建的 dataset_text_field 或 formatting_func ──
        # 因為我們已經在 dataset.map 階段完成了 tokenization 和 loss masking
        # SFTTrainer 會直接使用 data_collator 回傳的 input_ids / labels
    )

    print(f"   ├─ 訓練 epochs：3")
    print(f"   ├─ Batch size：8 × {training_args.gradient_accumulation_steps} (grad accum)")
    print(f"   ├─ Learning rate：1e-4 (linear decay, 10% warmup)")
    print(f"   ├─ Loss 策略：Completion-Only（僅計算 output 部分的 loss）")
    print(f"   ├─ Packing：已啟用（平均 ~83 tokens → 打包至 {MAX_SEQ_LENGTH}）")
    print(f"   └─ 輸出目錄：{output_dir}")

    return trainer


# =============================================================================
# 9. 生成 Model Card (README.md)
# =============================================================================
def generate_model_card(hub_repo_id: str, data_path: str) -> str:
    """生成符合 HuggingFace 規範的 Model Card"""
    model_card = f"""---
language:
  - zh
  - en
license: apache-2.0
base_model: unsloth/Qwen3-4B-Base-unsloth-bnb-4bit
tags:
  - translation
  - sft
  - lora
  - unsloth
  - qwen3
  - lost-in-literalism
---

# {hub_repo_id.split('/')[-1]}

本模型復現了論文 **"Lost in Literalism: How Supervised Training Shapes Translationese in LLMs"** 中的 **SFT-Polished** 方法，使用 [Unsloth](https://github.com/unslothai/unsloth) 加速對 Qwen3-4B Base Model 進行 LoRA SFT 微調。

## 🔬 訓練方法

- **方法**：LoRA SFT（Supervised Fine-Tuning with Low-Rank Adaptation）
- **論文策略**：SFT-Polished — 使用經過人工潤飾（polished）的高品質翻譯對進行訓練
- **Loss 策略**：**Completion-Only Loss Masking** — 僅計算 output（翻譯結果）部分的 loss，prompt（instruction + input）部分的 label 設為 -100，不參與梯度計算
- **加速框架**：Unsloth（4-bit QLoRA + gradient checkpointing）
- **Packing**：將多個短序列打包成固定長度序列，提升 GPU 利用率

### 為什麼使用 Completion-Only Loss？

論文發現，如果讓模型同時學習預測 instruction 和翻譯結果，會導致模型產生明顯的「翻譯腔」（translationese），降低翻譯的自然度。通過只對 output 部分計算 loss，模型可以專注於學習高品質的翻譯表達。

## 📊 訓練超參數

| 參數 | 值 |
|---|---|
| Base Model | `{MODEL_NAME}` |
| LoRA Rank (r) | 16 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.05 |
| Target Modules | q_proj, k_proj, v_proj, o_proj（僅 attention） |
| Epochs | 3 |
| Batch Size | 16 |
| Learning Rate | 1e-4 |
| LR Scheduler | Linear (10% warmup) |
| Max Sequence Length | {MAX_SEQ_LENGTH} |
| Precision | fp16 |
| Packing | ✅ 啟用 |
| Gradient Checkpointing | Unsloth 優化版 |
| **Loss Masking** | **Completion-Only（僅 output 部分）** |

## 📁 訓練資料

- 格式：JSONL（`instruction`, `input`, `output`）
- 資料量：約 16,000 對
- Train/Val Split：90/10（seed=42）
- 來源檔案：`{os.path.basename(data_path)}`

## 🚀 推理範例

```python
from unsloth import FastLanguageModel

# 載入模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{hub_repo_id}",
    max_seq_length={MAX_SEQ_LENGTH},
    dtype=None,
    load_in_4bit=True,
)

# 切換到推理模式（Unsloth 會自動優化推理速度）
FastLanguageModel.for_inference(model)

# 準備輸入
prompt = "Translate the following text into Chinese:\\nThe quick brown fox jumps over the lazy dog.\\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 生成翻譯
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)

# 解碼結果（只取生成的部分）
generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(generated)
```

## 📝 論文引用

```bibtex
@inproceedings{{li-etal-2025-lost,
    title = "Lost in Literalism: How Supervised Training Shapes Translationese in LLMs",
    author = "Li, Yafu and Zhang, Ronghao and Wang, Zhilin and Zhang, Huajian and Cui, Leyang and Yin, Yongjing and Xiao, Tong and Zhang, Yue",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    year = "2025",
    pages = "12875--12894",
}}
```

## 🔗 相關連結

- **Base Model**: [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME})
- **Unsloth**: [https://github.com/unslothai/unsloth](https://github.com/unslothai/unsloth)
- **論文**: [Lost in Literalism (ACL 2025)](https://aclanthology.org/2025.acl-long.693/)
"""
    return model_card


# =============================================================================
# 10. 訓練後處理：合併 LoRA 並上傳至 HuggingFace Hub
# =============================================================================
def merge_and_upload(
    model,
    tokenizer,
    hub_repo_id: str,
    data_path: str,
):
    """
    訓練結束後：
      1. 使用 Unsloth 的 push_to_hub_merged 將 LoRA 合併進 base model
      2. 上傳合併後的 16-bit 完整模型至 HuggingFace Hub
      3. 一併上傳 tokenizer 與 Model Card
    （使用者需預先透過 `huggingface-cli login` 完成認證）
    """
    print("\n📤 正在合併 LoRA 並上傳至 HuggingFace Hub...")
    print(f"   目標 Repo：{hub_repo_id}")

    # ── 生成 Model Card ──
    model_card_content = generate_model_card(hub_repo_id, data_path)

    # ── 使用 Unsloth 的 push_to_hub_merged 上傳合併後的 16-bit 模型 ──
    # 這會自動：(1) merge LoRA weights (2) 轉為 16-bit (3) push 到 hub
    model.push_to_hub_merged(
        hub_repo_id,
        tokenizer=tokenizer,
        save_method="merged_16bit",  # 上傳合併後的 16-bit 模型，方便他人使用
    )
    print("   ✅ 模型與 Tokenizer 已上傳")

    # ── 上傳 Model Card (README.md) ──
    api = HfApi()
    api.upload_file(
        path_or_fileobj=model_card_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=hub_repo_id,
        repo_type="model",
    )
    print("   ✅ Model Card (README.md) 已上傳")
    print(f"\n🎉 完成！模型已上傳至：https://huggingface.co/{hub_repo_id}")


# =============================================================================
# 11. 主程式
# =============================================================================
def main():
    print("=" * 70)
    print("  SFT-Polished 訓練腳本")
    print("  復現論文: Lost in Literalism (ACL 2025)")
    print("  框架: Unsloth + LoRA + TRL SFTTrainer")
    print("=" * 70)

    # ── 解析命令列參數 ──
    args = parse_args()

    # ── 前置檢查（資料路徑 & HF 登入狀態）──
    preflight_checks(args)

    # ── 載入資料 & Split ──
    ds = load_and_split_data(args.data_path)

    # ── 載入模型 & Tokenizer ──
    model, tokenizer = load_model_and_tokenizer()

    # ── 注入 LoRA ──
    model = apply_lora(model)

    # ── 資料前處理（Completion-Only Loss Tokenization + Packing）──
    ds_tokenized = preprocess_dataset(ds, tokenizer)

    # ── 建立 Trainer ──
    trainer = create_trainer(model, tokenizer, ds_tokenized, args.output_dir)

    # ── 顯存狀態 ──
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        print(f"\n💾 GPU 顯存：已用 {allocated:.1f} GB / 總共 {gpu_mem:.1f} GB")

    # ── 開始訓練 ──
    print("\n" + "=" * 70)
    print("  🚀 開始訓練...")
    print("=" * 70 + "\n")

    train_result = trainer.train()

    # ── 印出訓練結果 ──
    print("\n" + "=" * 70)
    print("  📊 訓練完成！結果摘要：")
    print("=" * 70)
    metrics = train_result.metrics
    print(f"   ├─ Train Loss:    {metrics.get('train_loss', 'N/A')}")
    print(f"   ├─ Train Runtime: {metrics.get('train_runtime', 'N/A'):.1f} 秒")
    print(f"   └─ Train Samples/sec: {metrics.get('train_samples_per_second', 'N/A'):.2f}")

    # ── 最終驗證 ──
    eval_metrics = trainer.evaluate()
    print(f"\n   📈 最佳模型驗證結果：")
    print(f"   └─ Eval Loss: {eval_metrics.get('eval_loss', 'N/A')}")

    # ── 合併 & 上傳 ──
    merge_and_upload(model, tokenizer, args.hub_repo_id, args.data_path)


if __name__ == "__main__":
    main()
