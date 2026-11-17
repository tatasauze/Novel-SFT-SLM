# 任務

撰寫一個**單一 Python 腳本** `train_sft_polished_unsloth.py`，使用 **Unsloth** 庫對 base model 進行 LoRA SFT 訓練（復現論文 "Lost in Literalism" 的 SFT-Polished 方法），訓練結束後將合併後的完整模型上傳至 HuggingFace Hub（含 model card README）。

---

# 環境與依賴

- 模型：`unsloth/Qwen3-4B-Base-unsloth-bnb-4bit`
- 核心套件：`unsloth`, `trl`, `transformers`, `peft`, `datasets`, `huggingface_hub`
- 腳本開頭用 `# pip install ...` 註解列出需要安裝的套件（請包含 unsloth 的官方安裝指令，例如 `pip install unsloth` 或 `pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"`）
- colab T4

---

# 訓練資料

- 格式：本地 JSONL 檔案，路徑由命令列參數 `--data_path` 傳入
- 結構：`{"instruction": "...", "input": "...", "output": "..."}`
- 資料量約 16,000 對
- 腳本需自動做 90/10 train/val split（random seed=42）

---

# 模型載入 (Unsloth API)

- 使用 `from unsloth import FastLanguageModel`
- 載入模型與 tokenizer：
  ```python
  model, tokenizer = FastLanguageModel.from_pretrained(
      model_name="unsloth/Qwen3-4B-Base-unsloth-bnb-4bit",
      max_seq_length=1024,
      dtype=None, # 自動偵測
      load_in_4bit=True,
  )
  ```
- 若 tokenizer 沒有 pad_token，設定 `pad_token = eos_token`，並調整 model 的 embedding 大小以適應 pad_token。

---

# LoRA 配置 (Unsloth API)

- 使用 `FastLanguageModel.get_peft_model` 注入 LoRA：
  - `r = 16`
  - `lora_alpha = 32`
  - `lora_dropout = 0.05`
  - `target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
  - `bias = "none"`
  - `use_gradient_checkpointing = "unsloth"` (這能大幅降低顯存並支援長文本)
  - `random_state = 42`

---

# 資料處理與 Completion-Only Loss (關鍵！)

由於使用的是 **Base Model** 且論文要求 **Loss 只計算 output 部分**（Completion-Only），不能使用一般的 Chat Template 全量微調。
請在 `dataset.map` 中實作自定義的 tokenize 邏輯：
1. 組合 Prompt：`f"{instruction}\n{input}\n"`
2. 組合 Output：`f"{output}{tokenizer.eos_token}"`
3. 分別 tokenize Prompt 和 Output。
4. 將兩者拼接成 `input_ids` 和 `attention_mask`。
5. **構建 labels**：Prompt 長度的部分設為 `-100`，Output 部分保留真實 token id。
6. 截斷至 `max_seq_length=1024`（優先保留 Output 尾部）。
7. 使用 HuggingFace 的 `DataCollatorWithPadding` 或自定義 collator 來 batch 這些處理好的 features。

---

# 訓練超參數 (TRL SFTTrainer)

使用 `trl.SFTTrainer` 進行訓練，參數如下：
- `num_train_epochs = 3`
- `per_device_train_batch_size = 6`
- `per_device_eval_batch_size = 16`
- `gradient_accumulation_steps = 4`
- `learning_rate = 1e-4`
- `warmup_ratio = 0.1`
- `lr_scheduler_type = "linear"`
- `weight_decay = 0.0`
- `max_grad_norm = 1.0`
- `bf16 = True`
- `logging_steps = 50`
- `eval_strategy = "steps"`
- `eval_steps = 200`
- `save_strategy = "steps"`
- `save_steps = 200`
- `load_best_model_at_end = True`
- `metric_for_best_model = "eval_loss"`
- `greater_is_better = False`
- `save_total_limit = 3`
- `output_dir` 由命令列參數 `--output_dir` 傳入

---

# 訓練後處理與上傳 HuggingFace

1. 訓練結束後，載入 best checkpoint。
2. 使用 Unsloth 的 `model.push_to_hub_merged` 或先 `model.merge_and_unload()` 再用 `transformers` 的 `push_to_hub` 上傳完整的合併後模型 (16-bit 或 4-bit 皆可，建議上傳 merged 16-bit 方便他人使用)。
3. Tokenizer 也一併上傳。
4. Repo ID 由命令列參數 `--hub_repo_id` 傳入。
5. **生成 Model Card (README.md)** 並隨模型上傳，內容包含：
   - 模型名稱與簡述
   - 訓練方法說明（LoRA SFT，復現 "Lost in Literalism" 論文的 SFT-Polished，使用 Unsloth 加速）
   - 論文引用（BibTeX）：
     ```bibtex
     @inproceedings{li-etal-2025-lost,
         title = "Lost in Literalism: How Supervised Training Shapes Translationese in LLMs",
         author = "Li, Yafu and Zhang, Ronghao and Wang, Zhilin and Zhang, Huajian and Cui, Leyang and Yin, Yongjing and Xiao, Tong and Zhang, Yue",
         booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
         year = "2025",
         pages = "12875--12894",
     }
     ```
   - 訓練超參數表格 (特別標註 Loss Masking 策略)
   - Base model 連結
   - 資料集說明
   - Python 推理範例 (使用 Unsloth 的 `FastLanguageModel.for_inference`)
6. 使用 `huggingface_hub.login(token=...)` 進行認證，token 由環境變數 `HF_TOKEN` 讀取。

---

# 命令列參數總覽

用 `argparse` 實作：
- `--data_path`（必填）
- `--output_dir`（選填，預設 `./sft-polished-qwen3-4b`）
- `--hub_repo_id`（必填）

---

# 程式碼要求

1. **單一檔案**，結構清晰。
2. 加入適當的 print/log 顯示訓練進度與資料處理狀態。
3. 加入基本的錯誤處理（資料路徑不存在、HF token 未設定等）。
4. 程式碼中加**中文註解**說明各段對應論文的哪個設定（特別是 Completion-Only Loss 的實作）。
5. 確保 Unsloth 的 API 調用正確無誤（包含 `FastLanguageModel.for_inference` 的推理設定）。
```

---

### 💡 給您的額外實作提醒：

1. **Unsloth 的環境安裝**：
   Unsloth 對環境比較敏感，如果在 Colab 或 AutoDL 等雲端機器上跑，可以直接使用他們官方的安裝指令：
   ```bash
   pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
   pip install --no-deps trl peft accelerate bitsandbytes
   ```
2. **Base Model 的 Loss Masking**：
   這是復現該論文**最核心的細節**。如果是 Instruct 模型，我們可以依賴 `### Response:` 來讓 `trl` 自動 mask loss；但因為您用的是 **Base Model**，沒有固定的觸發詞，所以上述 Prompt 中特別指示 LLM 必須在 `dataset.map` 階段**手動將 prompt 長度的 label 設為 `-100`**。如果 LLM 生成的程式碼沒有做到這點，請務必手動修正，否則模型會去學習預測 instruction，導致翻譯能力崩壞。
3. **顯存監控**：
   使用了 `use_gradient_checkpointing = "unsloth"` 後，4B 模型在 Batch Size 16、Seq Length 1024 的情況下，單卡 24GB (如 RTX 3090/4090) 應該能非常輕鬆地跑完，甚至還有餘量將 Batch Size 提升到 32 以加快收斂。