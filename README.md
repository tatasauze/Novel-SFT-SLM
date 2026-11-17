# 低資源文學平行語料邊界修復與大模型英中文學翻譯微調 Pipeline

> **研究課題**：低資源文學平行語料的邊界修復、翻譯風格之操作化定義與機器腔抑制 —— 以 4B 級 LLM 的英譯中文學翻譯為驗證場域  
> **核心模型**：`unsloth/Qwen3-4B-Base-unsloth-bnb-4bit` (採用 LoRA / QLoRA 與 Unsloth 記憶體加速技術)  
> **實驗語料**：紐西蘭犯罪懸疑小說家 Paul Cleave 經典作品《父仇者》（*Blood Men*）與《殺手收藏家》（*The Cleaner* / *Collecting Cooper*）英中雙語平行語料。  
> **開源模型權重**：已發布至 Hugging Face Hub（詳見 [5.4 開源模型權重清單](#54-開源模型權重清單-hugging-face-repositories)）

---

## 目錄

- [1. 專案背景與研究架構](#1-專案背景與研究架構)
- [2. 檔案目錄結構](#2-檔案目錄結構)
- [3. 模組一：文本處理 (Text Processing)](#3-模組一文本處理-text-processing)
  - [3.1 英文原文解析與連字清理](#31-英文原文解析與連字清理)
  - [3.2 中文 OCR 文本清洗與格式化](#32-中文-ocr-文本清洗與格式化)
- [4. 模組二：雙語對齊與動態規劃邊界修復 (Bilingual Alignment)](#4-模組二雙語對齊與動態規劃邊界修復-bilingual-alignment)
  - [4.1 階段一：粗對齊 (Vecalign + Jina Embeddings)](#41-階段一粗對齊-vecalign--jina-embeddings)
  - [4.2 階段二：先驗正則化動態規劃邊界修復 (DP Boundary Refinement)](#42-階段二先驗正則化動態規劃邊界修復-dp-boundary-refinement)
  - [4.3 階段三：事後重評估與異常值標註 (Post-Align Scoring)](#43-階段三事後重評估與異常值標註-post-align-scoring)
- [5. 模組三：大模型微調與偏好對齊 (Model Fine-Tuning)](#5-模組三大模型微調與偏好對齊-model-fine-tuning)
  - [5.1 階段一：SFT 與 Completion-Only Loss (抑制機器腔)](#51-階段一sft-與-completion-only-loss-抑制機器腔)
  - [5.2 階段二：ORPO 偏好優化 (Human Translation vs. MT)](#52-階段二orpo-偏好優化-human-translation-vs-mt)
  - [5.3 提示詞模板工程 (Prompt Template Engineering)](#53-提示詞模板工程-prompt-template-engineering)
  - [5.4 開源模型權重清單 (Hugging Face Repositories)](#54-開源模型權重清單-hugging-face-repositories)
- [6. 模型評估指標與驗證閉環 (Evaluation Metrics)](#6-模型評估指標與驗證閉環-evaluation-metrics)
- [7. 環境配置與端到端執行指南 (Quickstart)](#7-環境配置與端到端執行指南-quickstart)
- [8. 引用資訊 (Citation)](#8-引用資訊-citation)

---

## 1. 專案背景與研究架構

在低資源與文學翻譯（Literary Translation）領域，高質量的雙語小說語料受版權保護且極難獲取。傳統機器翻譯資料管線傾向於「**丟棄**低對齊信心度資料」，但對珍稀的文學平行語料而言，每一筆遺漏都是嚴重的資訊損失。

本專案圍繞一條緊密的因果鏈展開：
```
珍稀文學語料 (OCR/DOCX)
  └──> 文本處理與分章段落規範化
        └──> 粗粒度語義對齊 (Vecalign + Jina-v5)
              └──> 【核心創新】先驗正則化 DP 邊界修復 (修復而非丟棄)
                    └──> LoRA SFT 微調 (Completion-Only Loss 抑制機器腔)
                          └──> ORPO 偏好優化 (以人工文學譯文為 Chosen)
                                └──> BLEU / chrF++ / COMET / BERTScore 多維評估
```

---

## 2. 檔案目錄結構

```text
novel SFT/
├── raw_texts/                      # 原始文本與預處理工具
│   ├── en/                         # 英文小說章節 txt
│   ├── zh/                         # 中文 OCR 原始章節 txt
│   ├── cleaned_zh/                 # 清洗後中文章節 txt
│   ├── process_en/                 # 英文抽取與分章腳本 (docx2txt.py, split_word.py)
│   └── process_zh/                 # 中文清洗腳本 (clean_txt.py, manual_check.py)
├── ocr data/                       # 紙本書掃描 OCR 來源 (逐頁 Markdown / Word 檔案)
│   ├── 殺手收藏家/                 # The Cleaner 逐頁 OCR 資料
│   └── 父仇者/                     # Blood Men 逐頁 OCR 資料
├── alignment/                      # 粗對齊階段 (Sentence/Paragraph Embedding + Vecalign)
│   ├── 02_embedding.py             # Jina Embedding 計算 (jina-embeddings-v5-text-small)
│   ├── 03_tokenize.py              # Qwen3-4B Token 計數
│   ├── alignment.py                # Vecalign 雙語段落與長區塊遞迴拆分對齊
│   ├── post_align_score.py         # 重新計算對齊後餘弦相似度與分數
│   └── common_io.py                # 語料載入與章節 IO 共用類別
├── DP_align/                       # 【核心演算法】動態規劃邊界重切分 (Boundary Refinement)
│   ├── realign_boundaries.py       # 局部邊界重切分演算法主程式
│   ├── realign_boundaries_bidire_check.py # 雙向約束邊界修正主程式
│   ├── merge_paras.py              # 構建包含 embedding 與 token 的段落資訊字典
│   ├── verify_alignment.py         # 對齊順序與完整性驗證
│   ├── algotithm.md                # 演算法數學形式化理論文件
│   └── quick view algorithm.md     # 演算法偽代碼快速指南
├── alignment_work/                 # Blood Men 對齊中繼與輸出資料
├── alignment_work_the_cleaner/     # The Cleaner 對齊中繼、DP 與 ORPO 預備資料
├── SFT/                            # SFT 基礎微調與評估
│   ├── train_en2zh_sft.py          # 基礎 LoRA SFT 訓練腳本
│   ├── evaluate_en2zh_metrics.py   # BLEU, chrF++, COMET, BERTScore 綜合評估
│   ├── clean_data.csv              # 清洗後微調資料集
│   └── evaluation_SFT/             # SFT 模型推論與評估輸出指標
├── SFT new template/               # 注入文學角色系統提示詞之 SFT 訓練管線
│   ├── train_en2zh_sft.py          # 包含 Literary Translator Prompt 的訓練腳本
│   └── SFT_training.csv            # 訓練用平行語料
├── SFT-Polish/                     # 復現 "Lost in Literalism" 論文之 Completion-Only SFT
│   ├── train_sft_polished_unsloth.py # 嚴格 Loss Masking 之 SFT 訓練與 HF 上傳
│   ├── code_instruction.md        # 論文復現架構詳細規範
│   └── Readme.md                   # SFT-Polish 專屬說明
└── ORPO/                           # Odds Ratio Preference Optimization 偏好優化
    ├── pre_orpo/                   # 偏好語料構建 (計算 log-PPL 與 margin 篩選)
    │   ├── pre_dataset.py          # 透過 SFT 模型生成與難度邊界篩選
    │   └── orpo_scored_all.jsonl   # 評分標註後之三元組語料
    ├── train_en2zh_orpo.py         # Unsloth ORPO 訓練主程式
    └── ORPO_results.md             # 不同 β 與 learning rate 超參數實驗日誌
```

---

## 3. 模組一：文本處理 (Text Processing)

文本處理的主要目的在於將繁雜的非結構化出版物來源（紙本書 OCR 掃描件與 Word 原稿）規範化為「章節分明、一行一段、無雜訊、編碼一致」的標準對齊語料。

### 3.1 英文原文解析與連字清理
檔案位置：[`raw_texts/process_en/`](raw_texts/process_en/)
- [`docx2txt.py`](raw_texts/process_en/docx2txt.py)：
  - **連字正規化（Ligature Normalization）**：修復排版軟體常見的 Unicode 連字元（如 `\ufb00` $\to$ `ff`, `\ufb01` $\to$ `fi`, `\ufb02` $\to$ `fl`, `\ufb03` $\to$ `ffi` 等）。
  - **段落壓平與行內換行處理**：將 Word 段落內部軟換行 `<w:br/>` 轉換為空格，壓縮多重空白，並過濾頁首頁尾。
- [`split_word.py`](raw_texts/process_en/split_word.py) / [`split_pdf.py`](raw_texts/process_en/split_pdf.py)：根據章節標題標記（如 `Chapter 1`, `CHAPTER TWO`）自動切割章節，統一輸出為 `ch{i}.txt`。

### 3.2 中文 OCR 文本清洗與格式化
檔案位置：[`raw_texts/process_zh/`](raw_texts/process_zh/)、[`ocr data/`](ocr%20data/)
- 原始中文資料來自掃描紙本書 OCR 轉譯成 Markdown 檔案（如 `page_001.md`），內含頁碼、頁首裝飾標籤與識別雜訊。
- [`clean_txt.py`](raw_texts/process_zh/clean_txt.py)：
  - 濾除 Markdown 標題符號 `###` 及掃描儀殘留標記。
  - 清理全形空格、零寬度字元與異常空行。
  - 確保每行代表完整段落，並輸出至 [`cleaned_zh/`](raw_texts/cleaned_zh/)。

---

## 4. 模組二：雙語對齊與動態規劃邊界修復 (Bilingual Alignment)

小說翻譯往往存在「一對多」、「多對一」、意譯跨段等現象。傳統對齊工具（如 Vecalign）在全域搜尋時易發生**段落邊界切分錯位**或**過度合併（Over-merge）**。本專案提出了以 Vecalign 為先驗、動態規劃局部搜尋的邊界修復演算法。

```
[原始中英段落] ──> [Jina Embedding v5] ──> [Vecalign 粗對齊]
                                                    │
                                                    ▼
[修復後對齊成果] <── [重算 Cosine Score] <── [DP 局部邊界修復]
```

### 4.1 階段一：粗對齊 (Vecalign + Jina Embeddings)
- **語意特徵編碼** ([`alignment/02_embedding.py`](alignment/02_embedding.py))：
  - 採用 `jinaai/jina-embeddings-v5-text-small`。
  - 設定對稱相似度檢索任務 `task="text-matching"`，啟用 `bfloat16` 與 `flash_attention_2`。
- **Token 長度計數** ([`alignment/03_tokenize.py`](alignment/03_tokenize.py))：
  - 載入 `Qwen3-4B` Tokenizer 針對中英段落計算真實 token 數，作為對齊長度比例約束依據。
- **粗對齊與遞迴 Chunk 拆分** ([`alignment/alignment.py`](alignment/alignment.py))：
  - 執行 Vecalign 全域搜尋。
  - 若遇超過 1024 tokens 的超長段落對，自動在句子群（Chunk）粒度進行二次細緻拆分，無法匹配者標記輸出至 `*_unaligned.jsonl`。

### 4.2 階段二：先驗正則化動態規劃邊界修復 (DP Boundary Refinement)
相關檔案：[`DP_align/realign_boundaries.py`](DP_align/realign_boundaries.py)、[`DP_align/algotithm.md`](DP_align/algotithm.md)

#### 演算法核心概念

- **基本假設**：Vecalign 已經找到了大致區塊，錯誤主要發生在相鄰對齊邊界 
  $L_i = (Z_i, E_i)$ 與 $L_{i+1} = (Z_{i+1}, E_{i+1})$ 之間。

- **硬性約束（Hard Constraints）**：
  1. 段落順序絕不改變；
  2. 段落 ID 不重複、不遺漏、不消失；
  3. 跨章節邊界嚴格鎖定（Locked boundaries）；
  4. 已移動過的段落加入鎖定集 $\mathcal{L}$ ，後續邊界不得再移動，
     避免連鎖震盪並保證單次掃描收斂。

- **段落匹配分數 $\sigma$（$\Phi$ 的基本單元）**：
  $\sigma(s,t) = \max(0, \cos(s,t)) \cdot \exp(-\lambda_\ell |r(s,t)|) - \tau_m$ 
  整流餘弦（rectified cosine）抑制負相似度的干擾；指數項依長度比 
  $r(s,t)$ 折扣長短懸殊的假性匹配；減去門檻 $\tau_m$ 使弱匹配成為負分，
  讓 DP 寧可選擇缺口（gap）也不勉強對齊。

- **$\Phi$ 函數（單調對齊動態規劃）**：
  給定候選切分後的段落序列 $S=(s_1,\ldots,s_m)$ 、 $T=(t_1,\ldots,t_n)$ ，
  $\Phi(S,T)$ 為 Needleman–Wunsch 型全域單調對齊分：
  $\mathrm{dp}[a][b] = \max\begin{cases}
    \mathrm{dp}[a-1][b-1] + \sigma(s_a, t_b) & \text{(1–1 匹配)}\\
    \mathrm{dp}[a-1][b] - \gamma & \text{(來源側缺口)}\\
    \mathrm{dp}[a][b-1] - \gamma & \text{(目標側缺口)}
  \end{cases}$
  邊界條件 $\mathrm{dp}[0][0]=0$ 、$\mathrm{dp}[a][0]=-a\gamma$ 、
  $\mathrm{dp}[0][b]=-b\gamma$ ，回傳 $\Phi=\mathrm{dp}[m][n]$ 。
  路徑嚴格單調，結構上即保證硬性約束 1；1–many 關係以 gap 近似計分，
  換取 $O(mn)$ 的可枚舉性。 $\Phi$ 越高，代表該切分下兩側內部對齊越一致；
  每個候選 $(\Delta_s,\Delta_t)$ 需對左右兩個區塊各計算一次 $\Phi$ 。

- **數學目標函數**：
  在滑動窗口範圍 $(\Delta_s, \Delta_t) \in [-K, K]^2$ 內枚舉轉移候選，
  計算單調對齊匹配分 $\Phi$ 、長度懲罰比、以及偏離先驗的懲罰：
  $\Psi(\Delta_s, \Delta_t) = \frac{\Phi(S_i', T_i') + \Phi(S_{i+1}', T_{i+1}')}{2}
    - \lambda_{\mathrm{mv}}(|\Delta_s| + |\Delta_t|)
    - \lambda_{\mathrm{sq}}(\Delta_s^2 + \Delta_t^2)$
  取兩側平均使不同邊界間的分數尺度一致，$\delta$ 才能以單一門檻通用；
  $\lambda_{\mathrm{mv}}$ （L1）與 $\lambda_{\mathrm{sq}}$ （L2）構成凸的
  位移先驗：前者偏好小位移，後者超線性壓制大位移。

- **超參數說明**：

  | 符號 | 作用層級 | 意義與調參方向 |
  |---|---|---|
  | $K$ | 搜尋 | 邊界移動半徑（段落數），候選數 $(2K+1)^2-1$；建議 1–2，過大違反「局部修復」假設且成本 $O(K^2 mn)$ |
  | $\tau_m$ | $\sigma$ | 匹配門檻：低於此相似度的配對變負分，抑制弱匹配 |
  | $\lambda_\ell$ | $\sigma$ | 長度比懲罰強度：越大越排斥長短懸殊的匹配 |
  | $\gamma$ | $\Phi$ | gap 懲罰；與 $\tau_m$ 耦合（$\sigma > -2\gamma$ 時匹配才優於兩個 gap），兩者需一起調 |
  | $\lambda_{\mathrm{mv}}$ | $\Psi$ | 線性移動懲罰，編碼「最小編輯」先驗 |
  | $\lambda_{\mathrm{sq}}$ | $\Psi$ | 平方移動懲罰，與 $\lambda_{\mathrm{mv}}$ 互補壓制大位移 |
  | $\delta$ | 接受 | 最小增益門檻：過濾 embedding 噪聲造成的偽增益 |

- 只有當最佳候選增益 $\Psi(\Delta_s^*, \Delta_t^*) - \Psi_0 > \delta$
  （最小增益門檻）時，才採納重切分，確保「**只動可疑處、不動已正確處**」。

  
### 4.3 階段三：事後重評估與異常值標註 (Post-Align Scoring)
- [`alignment/post_align_score.py`](alignment/post_align_score.py)：
  - 修復邊界後，調用 Embedding 模型重新計算每個對齊 Pair 的真實餘弦相似度（Cosine Similarity）與 Vecalign 分數。
- **3σ 統計檢驗**：
  - 計算 Token 比例與語意相似度的均值與標準差，標註超出 $\pm 3\sigma$ 的 Outliers，輸出統計圖與審查清單。

---

## 5. 模組三：大模型微調與偏好對齊 (Model Fine-Tuning)

在低資源文學英譯中情境下，直接使用通用 Instruct 模型容易引發「**機器腔（Translationese）**」——如被動句過度使用、代詞顯化、缺乏文采與句式呆板。

本專案實踐了雙階段模型優化：**SFT (Completion-Only Loss Masking)** 與 **ORPO (Odds Ratio Preference Optimization)**。

### 5.1 階段一：SFT 與 Completion-Only Loss (抑制機器腔)
檔案位置：[`SFT-Polish/train_sft_polished_unsloth.py`](SFT-Polish/train_sft_polished_unsloth.py)  
參考研究：*ACL 2025 "Lost in Literalism: How Supervised Training Shapes Translationese in LLMs"*

#### 核心實作關鍵
- **為什麼需要 Completion-Only Loss？**  
  使用 Base 模型時，若將整個 sequence（Prompt + Target）納入 Cross-Entropy Loss 計算，模型會浪費容量在學習重複 Prompt 的指令字句，進而強化逐字直譯傾向。
- **手動 Loss Masking**：
  在 `dataset.map` 中分別 Tokenize 輸入 Prompt 與目標譯文，並將 Prompt 長度範圍內的 `labels` 明確置為 `-100`：
  ```python
  prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
  output_ids = tokenizer.encode(output, add_special_tokens=False) + [tokenizer.eos_token_id]

  input_ids = prompt_ids + output_ids
  # Prompt 部分完全遮罩 (-100)，僅計算目標譯文 loss
  labels = [-100] * len(prompt_ids) + output_ids
  ```
- **訓練配置**：
  - 基底模型：`unsloth/Qwen3-4B-Base-unsloth-bnb-4bit`
  - LoRA 規格：$r=16 \sim 32$, $\alpha=32 \sim 64$, dropout=0.05
  - Target Modules：`["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
  - 顯存加速：`use_gradient_checkpointing = "unsloth"`

### 5.2 階段二：ORPO 偏好優化 (Human Translation vs. MT)
檔案位置：[`ORPO/`](ORPO/)

無需獨立訓練額外的 Reward Model，直接利用勝負對（Chosen vs. Rejected）在對數幾率比（Odds Ratio）上施加懲罰。

1. **三元組資料建構** ([`ORPO/pre_orpo/pre_dataset.py`](ORPO/pre_orpo/pre_dataset.py))：
   - **Prompt**：英文小說原文段落。
   - **Chosen (勝)**：人工精修之專業文學譯本（保留原文敘事語氣與中文小說美感）。
   - **Rejected (負)**：商用機器翻譯或 SFT 初始輸出的直譯結果（帶有明顯翻譯腔、贅字）。
   - **難度邊界篩選**：計算條件 NLL 與 Log-Perplexity，篩選 $\text{margin} = \mathcal{L}_{\mathrm{rejected}} - \mathcal{L}_{\mathrm{chosen}} > 0$ 且位於中位難度區間的樣本，排除無鑑別度的樣本。
2. **ORPO 訓練與超參數探索** ([`ORPO/train_en2zh_orpo.py`](ORPO/train_en2zh_orpo.py))：
   - 超參數設定：$\beta \in [0.05, 0.4]$，Learning Rate $\in [5\text{e-}6, 1\text{e-}4]$，進行多組收斂性與 Margin 實驗對比。

### 5.3 提示詞模板工程 (Prompt Template Engineering)
在 [`SFT new template/`](SFT%20new%20template/) 中進一步規範了專門針對文學翻譯的 Prompt Template：
```text
You are a literary translator. Translate the following English novel excerpt into Traditional Chinese. Preserve the narrative voice, tone, atmosphere, dialogue style and paragraph breaks. Do not summarize, do not explain, and do not add translator notes.
### English
{en_text}
### Chinese
{zh_text}
```

---

### 5.4 開源模型權重清單 (Hugging Face Repositories)

本研究訓練的所有最終模型與 LoRA Adapters 均已開源並託管於 Hugging Face（帳號/組織：`tatasauce4life`）。您可以直接透過 Hugging Face 或 Unsloth / Transformers 載入：

| 模型 Repository | 階段 / 類型 | 核心特點與說明 |
| :--- | :--- | :--- |
| [**`tatasauce4life/sft-polished-qwen3-4b`**](https://huggingface.co/tatasauce4life/sft-polished-qwen3-4b) | **SFT (Polished)** | 復現 ACL 2025 *\"Lost in Literalism\"*，採用 **Completion-Only Loss Masking**（Prompt 設為 `-100`），避免模型死記提示詞格式，顯著減輕翻譯腔 |
| [**`tatasauce4life/qwen3-4b-en2zh-sft-instruct-new-template`**](https://huggingface.co/tatasauce4life/qwen3-4b-en2zh-sft-instruct-new-template) | **SFT (New Template)** | 注入完整「文學翻譯家角色」系統提示詞，具備強敘事語氣、氛圍控制與分段保持能力 |
| [**`tatasauce4life/qwen3-4b-en2zh-lora-instruct`**](https://huggingface.co/tatasauce4life/qwen3-4b-en2zh-lora-instruct) | **SFT LoRA (Instruct)** | 採用 Instruct 結構微調之文學英譯中 LoRA Adapter |
| [**`tatasauce4life/qwen3-4b-en2zh-lora`**](https://huggingface.co/tatasauce4life/qwen3-4b-en2zh-lora) | **SFT LoRA (Base)** | 基礎雙語平行對齊語料微調之 LoRA Adapter 權重 |
| [**`tatasauce4life/qwen3-4b-en2zh-orpo`**](https://huggingface.co/tatasauce4life/qwen3-4b-en2zh-orpo) | **ORPO (Baseline)** | 偏好優化基準組（以人工文學譯文 vs. 機器翻譯做對齊優化） |
| [**`tatasauce4life/qwen3-4b-en2zh-orpo_beta0.2`**](https://huggingface.co/tatasauce4life/qwen3-4b-en2zh-orpo_beta0.2) | **ORPO ($\beta=0.2$)** | Odds Ratio 係數 $\beta=0.2$ 之偏好優化微調模型 |
| [**`tatasauce4life/qwen3-4b-en2zh-orpo_beta0.3`**](https://huggingface.co/tatasauce4life/qwen3-4b-en2zh-orpo_beta0.3) | **ORPO ($\beta=0.3$)** | Odds Ratio 係數 $\beta=0.3$ 之偏好優化微調模型 |
| [**`tatasauce4life/qwen3-4b-en2zh-orpo_beta0.4`**](https://huggingface.co/tatasauce4life/qwen3-4b-en2zh-orpo_beta0.4) | **ORPO ($\beta=0.4$)** | Odds Ratio 係數 $\beta=0.4$ 之偏好優化微調模型（強懲罰機器直譯痕跡） |

#### 快速推論範例 (Quick Inference Example)

使用 Unsloth 進行 4-bit 快速推論：

```python
from unsloth import FastLanguageModel

# 1. 載入模型與 Tokenizer (以 sft-polished 為例)
model_name = "tatasauce4life/sft-polished-qwen3-4b"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=1024,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

# 2. 準備英譯中文學翻譯 Prompt
prompt = """You are a literary translator. Translate the following English novel excerpt into Traditional Chinese. Preserve the narrative voice, tone, atmosphere, dialogue style and paragraph breaks. Do not summarize, do not explain, and do not add translator notes.
### English
The rain came down hard, hammering on the tin roof like a thousand tiny fists. He sat by the window, watching the shadows stretch across the empty street.
### Chinese
"""

# 3. 生成翻譯
inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    use_cache=True,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

若載入 LoRA Adapter 權重（如 `tatasauce4life/qwen3-4b-en2zh-lora`）：

```python
from unsloth import FastLanguageModel

base_model_name = "unsloth/Qwen3-4B-Base-unsloth-bnb-4bit"
adapter_model_id = "tatasauce4life/qwen3-4b-en2zh-lora"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_name,
    max_seq_length=1024,
    load_in_4bit=True,
)
model.load_adapter(adapter_model_id)
FastLanguageModel.for_inference(model)
```

---

## 6. 模型評估指標與驗證閉環 (Evaluation Metrics)

檔案位置：[`SFT/evaluate_en2zh_metrics.py`](SFT/evaluate_en2zh_metrics.py)

本專案拒絕僅憑「主觀讀者感覺」評價翻譯，建立了完整的自動化多維評估體系：

| 評估維度 | 使用工具 / 指標 | 計算設定 | 評估目標 |
| :--- | :--- | :--- | :--- |
| **詞彙重合度** | **BLEU** (`sacrebleu`) | `tokenize="zh"` | 評估中文譯文與參考譯文的 N-gram 重合率 |
| **字元/形態級準確** | **chrF++** (`sacrebleu`) | `word_order=2` | 對低資源與形態變化更魯棒的字元級評估 |
| **神經跨語言語義** | **COMET** | `Unbabel/wmt22-comet-da` | 結合原文與譯文的語意逼近程度（與人評相關性最高） |
| **語意表示相似度** | **BERTScore** | `lang="zh"` | 透過預訓練模型上下文表徵衡量語義忠實度 |

---

## 7. 環境配置與端到端執行指南 (Quickstart)

### 7.1 環境安裝
建議使用具有 GPU (如 Colab T4 / RTX 3090 / A100) 的 Python 3.10+ 環境：

```bash
# 基礎依賴
pip install torch transformers datasets trl peft bitsandbytes accelerate
pip install scikit-learn pandas numpy openpyxl sacrebleu bert-score

# 安裝 Unsloth 記憶體加速
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# 安裝 Jina Embeddings 與 Vecalign 依賴
cd alignment
pip install -r requirements.txt
git clone https://github.com/thompsonb/vecalign.git /content/vecalign
cd ..
```

### 7.2 端到端操作流程

#### Step 1: 文本清理與章節預處理
```bash
# 英文原文 Word 抽取與分章
python raw_texts/process_en/docx2txt.py
python raw_texts/process_en/split_word.py

# 中文 OCR 文字清洗
python raw_texts/process_zh/clean_txt.py
```

#### Step 2: 向量化與初級粗對齊
```bash
cd alignment

# 產生段落嵌入與 Token 統計
python 02_embedding.py --zh_dir ../raw_texts/cleaned_zh --en_dir ../raw_texts/en
python 03_tokenize.py --zh_dir ../raw_texts/cleaned_zh --en_dir ../raw_texts/en

# 執行 Vecalign 粗對齊
python alignment.py \
    --vecalign_dir /content/vecalign \
    --alignment_max_size 6 \
    --embed_batch_size 32 \
    --zh_dir ../raw_texts/cleaned_zh \
    --en_dir ../raw_texts/en \
    --tok_dir ../alignment_work/tokens \
    --out_dir ../alignment_work/results
cd ..
```

#### Step 3: 先驗正則化動態規劃邊界修復 (DP Refinement)
```bash
cd DP_align

# 合併段落資訊字典 (Embedding + Token + Text)
python merge_paras.py ../raw_texts/cleaned_zh ../raw_texts/en ../alignment_work/tokens ../alignment_work/embeddings

# 執行局部邊界動態規劃重切分
python realign_boundaries.py \
    --coarse "../alignment_work/results/paragraph_align.jsonl" \
    --zh-paras zh_paras.jsonl \
    --en-paras en_paras.jsonl \
    --output results/dp_shift2_alignment.jsonl \
    --log results/dp_shift2_realign_log.jsonl \
    --max-shift 2 \
    --match-threshold 0.20 \
    --move-penalty 0.02 \
    --min-gain 0.008

# 驗證對齊結果順序與完整性
python verify_alignment.py ../raw_texts/cleaned_zh ../raw_texts/en ../alignment_work/tokens ../alignment_work/embeddings
cd ..
```

#### Step 4: SFT 模型微調 (Completion-Only Loss)
```bash
# 執行 LoRA SFT 微調
python SFT-Polish/train_sft_polished_unsloth.py \
    --data_path ./SFT/clean_data.csv \
    --output_dir ./qwen3-4b-en2zh-sft \
    --hub_repo_id your-username/qwen3-4b-en2zh-sft-lora
```

#### Step 5: ORPO 偏好優化訓練
```bash
# 構建 ORPO 難度邊界三元組 (Prompt / Chosen / Rejected)
python ORPO/pre_orpo/pre_dataset.py

# 執行 ORPO 偏好對齊
python ORPO/train_en2zh_orpo.py
```

#### Step 6: 翻譯模型指標評估
```bash
python SFT/evaluate_en2zh_metrics.py \
    --csv ./SFT/clean_data.csv \
    --adapter_path ./qwen3-4b-en2zh-sft/final_lora \
    --split test \
    --metrics bleu chrf comet bertscore
```

---

## 8. 引用資訊 (Citation)

若本研究程式碼或資料處理管線對您的研究有所啟發，請參考相關文獻：

```bibtex
@inproceedings{li-etal-2025-lost,
    title = "Lost in Literalism: How Supervised Training Shapes Translationese in LLMs",
    author = "Li, Yafu and Zhang, Ronghao and Wang, Zhilin and Zhang, Huajian and Cui, Leyang and Yin, Yongjing and Xiao, Tong and Zhang, Yue",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    year = "2025",
    pages = "12875--12894",
}

@inproceedings{thompson-post-2019-vecalign,
    title = "Vecalign: Improved Sentence Alignment in Low-Resource and Distant Language Pairs via Sentence Embeddings",
    author = "Thompson, Brian and Post, Matt",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    year = "2019",
    pages = "1342--1348",
}
```
