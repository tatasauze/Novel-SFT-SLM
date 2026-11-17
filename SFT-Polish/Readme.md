## 📋 腳本結構總覽

區段                        │ 功能
────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────
§1                          │ argparse 命令列參數（--data_path, --output_dir, --hub_repo_id）
§2                          │ 前置檢查（資料路徑、HF_TOKEN 環境變數）
§3                          │ JSONL 載入 + 90/10 train/val split（seed=42）
§4                          │ Unsloth FastLanguageModel.from_pretrained 載入 4-bit 模型
§5                          │ FastLanguageModel.get_peft_model 注入 LoRA（r=16, α=32）
§6                          │ Completion-Only Loss 核心實作 — dataset.map 中手動建構 labels，prompt 部分設為 -100
§7                          │ 自定義 CompletionOnlyDataCollator，padding 時 labels 填 -100
§8                          │ SFTTrainer 配置（3 epochs, bs=16, lr=1e-4, linear decay）
§9                          │ Model Card 生成（含論文 BibTeX、超參數表、推理範例）
§10                         │ push_to_hub_merged 上傳合併後 16-bit 模型
§11                         │ 主程式流程串接

## 🔑 Completion-Only Loss 的關鍵實作

在 build_tokenize_fn（§6）中：

1. 分別 tokenize prompt 和 output
2. 拼接 input_ids = prompt_ids + output_ids
3. Labels masking：labels = [-100] * prompt_len + output_ids
4. 截斷策略：超過 1024 時優先保留 output 尾部

這確保模型只從翻譯結果學習，不會去記憶 instruction 格式，避免論文所指出的 translationese 問題。

## 🚀 使用方式

# Colab 安裝
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes

# 設定 HF Token
export HF_TOKEN="hf_your_token_here"

# 執行訓練
python train_sft_polished_unsloth.py \
	--data_path ./data/sft_polished.jsonl \
	--output_dir ./sft-polished-qwen3-4b \
	--hub_repo_id your-username/sft-polished-qwen3-4b