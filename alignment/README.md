# 小說翻譯語料對齊 Pipeline

5（+1）支可獨立執行的腳本，依序跑完即完成「段落層級對齊(vecalign) -> 長段落 chunk 拆分對齊(vecalign) -> 統計/outlier 標記/繪圖」。**若手動調整過對齊結果的文字內容**，可再執行 `06_restatistics.py` 補足統計數值。

## 安裝

```bash
pip install -r requirements.txt
# jina-embeddings-v5-text-small 需要較新版 transformers/sentence-transformers，
# 已在 requirements.txt 中設定版本下限；若 Colab 預裝版本較舊，建議先重啟 runtime
# 讓新版套件生效。

# 04_align.py 需要真正的 vecalign（Colab 環境可完整運行）
git clone https://github.com/thompsonb/vecalign.git /content/vecalign

# (可選) 若要讓統計圖正確顯示中文，Colab 上可安裝：
apt-get install -y fonts-noto-cjk
```

## 輸入資料

```
/output_zh/output_zh_{i}.txt        # i 為章節編號整數
/output_en/f{i:02d}_chapter.txt     # 一行 = 一段落
```

## 執行順序

```bash
cd alignment

# 2. 對所有中/英段落做 embedding（jina-embeddings-v5-text-small）
python 02_embedding.py --zh_dir /output_zh --en_dir /output_en_utf8

# 3. 用 Qwen3-4B tokenizer 計算段落 token 數
python 03_tokenize.py --zh_dir /output_zh --en_dir /output_en_utf8

# 4. 用 vecalign 做段落層級對齊；超過 1024 tokens 的 pair 自動拆分並在
#    chunk 層級（句子群）再對齊一次；對齊失敗的內容輸出到 *_unaligned.jsonl
python alignment.py --vecalign_dir /content/vecalign --alignment_max_size 6 --embed_batch_size 32 --embed_max_seq_length 1024 --zh_dir /content/zh --en_dir /content/en --tok_dir /content/alignment_work/tokens --out_dir /content/alignment_work/results

# 5. 產生paragraph basic info 檔案，包含token, embedding等
python verify_alignment.py ./zh ./en /content/content/alignment_work/tokens /content/content/alignment_work/embeddings
python merge_paras.py ./zh ./en /content/content/alignment_work/tokens /content/content/alignment_work/embeddings

# 6. Dynamic programing
python realign_boundaries.py \
  --coarse "/content/content/alignment_work/results/paragraph_align.jsonl" \
  --zh-paras zh_paras.jsonl \
  --en-paras en_paras.jsonl \
  --output dp_shift2_alignment.jsonl \
  --log dp_shift2_realign_log.jsonl \
  --max-shift 2 \
  --match-threshold 0.20 \
  --move-penalty 0.02 \
  --min-gain 0.008 \
  --vecalign-weight 0.3 \
  --vecalign-scale 1.0
  
# 7. post align cos score
python post_align_score.py \
    --input_jsonl /alignment_work/results/post_aligned.jsonl \
    --output_jsonl /alignment_work/results/post_aligned_rescored.jsonl \
    --embed_batch_size 32 \
    --embed_max_seq_length 1024
```


## 使用的 Embedding 模型

`02_embedding.py` 使用 `jinaai/jina-embeddings-v5-text-small`（取代舊版
`jina-embeddings-v2-base-zh`，因為環境內 transformers 版本過舊需升級套件才能使用 v5）。

v5 的 `encode()` 需要指定 `task`。本 pipeline 是「比較中英段落語意是否相近」的
**對稱式**相似度比對，不是「用一句話檢索一批文件」的不對稱檢索情境，因此預設
`task="text-matching"`（對應官方範例中的 Text Matching Task），`04_align.py` 內
所有 embedder 呼叫（vecalign overlap 編碼、合併文字算 similarity）都沿用這個預設。
若之後有需要做不對稱檢索（例如用中文摘要去檢索原文段落），可在呼叫
`embedder.encode(texts, task="retrieval", prompt_name="query"/"document")` 時覆蓋。

GPU 上會自動套用官方建議設定（`bfloat16` + `flash_attention_2`）；若環境沒裝
flash-attn，會自動 fallback 成預設 attention 實作並印出警告，不會中斷執行。
CPU 環境則不套用這兩項設定。

## 檔案結構

```
alignment/
├── common_io.py              # 共用語料讀取類別 (CorpusLoader / Chapter)
├── 01_unicode_transform.py   # UnicodeTransformer
├── 02_embedding.py           # ParagraphEmbedder / ChapterEmbeddingRunner
├── 03_tokenize.py            # ParagraphTokenizer / ChapterTokenCounter
├── alignment.py              # VecalignRunner / ChunkSplitter / AlignmentPipeline
├── post_align_score.py       # 計算重新對齊後的 cos socre
├── 06_restatistics.py        # 手動修正文本後，補足 token 數、相似度、vecalign 分數
├── requirements.txt
└── README.md
```

## 中繼與輸出檔案

```
/alignment_work/
├── embeddings/{chapter_id}/{zh,en}.npy + meta.json   # 02 的輸出，供其他用途/QA 參考
├── tokens/{chapter_id}.json                          # 03 的輸出
├── vecalign_tmp/                                     # 04 執行時的暫存檔，跑完自動清除
└── results/
    ├── paragraph_align.jsonl          # alignment_unit: paragraph_pair（超長者標記為 paragraph_chunk）
    ├── chunk_align.jsonl              # alignment_unit: paragraph_chunk（拆分後的細粒度對齊）
    ├── paragraph_unaligned.jsonl      # 段落層級對齊失敗（vecalign 判定無對應）的內容，供人工檢視
    ├── chunk_unaligned.jsonl          # chunk 層級對齊失敗的內容，供人工檢視
    ├── post_align.jsonl               # 04_post_align.py 輸出，逐筆對應 paragraph_align.jsonl，
    │                                  # 多了 post_align_action / post_align_scores 欄位；
    │                                  # 05 預設拿這份當標的
    ├── post_align.filtered.jsonl      # 加上 outlier_alignment 判定（若有跑 04_post_align.py）
    ├── paragraph_align.filtered.jsonl # 沒跑 04_post_align.py 時的 fallback 版本
    ├── chunk_align.filtered.jsonl
    ├── stats_report.json              # 兩種粒度的 token / similarity 統計與 3σ 門檻
    ├── outliers_for_review.jsonl      # 所有被標記為 outlier 的紀錄彙整
    ├── paragraph_pair_stats.png       # paragraph_pair 整體統計圖（zh/en tokens + similarity 分佈）
    └── paragraph_chunk_stats.png      # paragraph_chunk 整體統計圖
```

## 各腳本重點說明

### 04_post_align.py：修正段落交界處的斷句誤差

`04_align.py` 是以「段落」為單位跑 vecalign，但段落本身的切法不一定跟句子邊界
完美對齊——常見情況是上一段結尾的最後一句話，語意上其實比較該歸到下一段。

`04_post_align.py` 針對 `paragraph_align.jsonl` 的每一筆紀錄 `line_i`，
純段落層級合併比較（不切句子）：
1. 取同一章節內的前一筆 `line_{i-1}`、後一筆 `line_{i+1}`（若存在）
2. 產生三個候選：
   - `original`：`line_i` 原本的 `zh_text` / `en_text`
   - `merge_up`：`line_{i-1}` 的文字 + `line_i` 的文字
   - `merge_down`：`line_i` 的文字 + `line_{i+1}` 的文字
3. 每個候選把 `zh_text` 當一行、`en_text` 當一行，丟給真正的 vecalign 做
   1 行 vs 1 行的對齊，取其分數（vecalign 判定無 1-1 對應的候選視為無效，不比較）
4. 取分數最高的候選（`best_score = max(...)`），若是 `merge_up`/`merge_down`，
   就用合併後的文字取代 `line_i`；`zh_tokens`/`en_tokens` 直接加總相鄰兩筆
   已經算好的數字，不重新 tokenize；`similarity` 用既有的 embedder 重新算一次

輸出 `post_align.jsonl` 筆數與順序都跟 `paragraph_align.jsonl` 一致，每筆多了：
- `post_align_action`: `"original"` / `"merge_up"` / `"merge_down"`
- `post_align_scores`: 各候選的 vecalign 分數，例如 `{"original": 0.5, "merge_up": 0.91, "merge_down": 0.3}`

### 05_filter.py：統計、標記 Outlier 並繪圖

對 `paragraph_pair` 與 `paragraph_chunk` 兩種粒度分別計算 `zh_tokens`、`en_tokens`、`similarity` 的平均值與標準差，並以 **3σ** 規則標記 `outlier_alignment`：
- `zh_tokens` 或 `en_tokens` 超出平均值 ±3 個標準差
- `similarity` 低於（平均值 - 3 個標準差）

同時輸出統計圖表（`paragraph_pair_stats.png`、`paragraph_chunk_stats.png`），
直方圖中 outlier 以紅色疊加，並標示 3σ 門檻線。所有 outlier 彙整至 `outliers_for_review.jsonl` 供人工檢視。

### 06_restatistics.py：手動修正後補足統計數值

若你在 Excel / CSV 中手動調整了 `zh_text` 或 `en_text`（例如修正 OCR 錯字、調整斷句或合併範圍），
可執行此腳本針對這些手動調整過的紀錄，**重新計算**：

- `zh_tokens`、`en_tokens`（使用 Qwen3-4B tokenizer）
- `similarity`（使用 jina-embeddings-v5-text-small 計算餘弦相似度）
- `vecalign_score`（呼叫 vecalign 做 1 行 vs 1 行的對齊分數）

支援輸入格式：`.xlsx`、`.csv`、`.jsonl`；輸出格式可自選，預設在輸入檔名後加 `_restat` 後綴。
只會更新原本為 `NaN` 的欄位，亦可透過 `--force` 強制重新計算所有行。

```bash
python 06_restatistics.py --input /path/to/edited.xlsx --vecalign_dir /content/vecalign
```

---

## OOM 排查

如果 `04_align.py` 執行時出現 GPU OOM，且**調小 `--embed_batch_size` 沒有幫助**
（包含降到 1 依然 OOM），代表問題通常不是「一批次裡樣本太多」，而是下面兩個原因之一：

1. **單一輸入序列本身太長**：`overlap.py` 會把最長到 `alignment_max_size` 個段落
   拼接成一個 overlap 文本去 embedding。如果原始段落本身較長（小說常見），
   拼接 8 段之後可能是幾千字，即使 batch_size=1，單一樣本的 attention 記憶體
   也可能爆掉。**解法**：調小 `--embed_max_seq_length`（例如 1024 或 512），
   超過的部分會被截斷再送進模型，這是目前最有效的旋鈕。
2. **多章節、多批次下來的 allocator 碎片化**：PyTorch 的 caching allocator
   不會主動把用過的記憶體還給系統，長時間跑很多大小不一的 batch 後容易碎片化，
   導致「明明看起來還有空間，但配置失敗」。`02_embedding.py` 的 `ParagraphEmbedder`
   新增了 `empty_cache()` 方法，`04_align.py` 已經在每次 embed overlap 之後、
   以及每個章節跑完之後自動呼叫，緩解這個問題。

**不是**問題的部分（澄清）：`03_tokenize.py` 的 `ParagraphTokenizer` 只載入
Qwen3-4B 的 tokenizer（詞表/merge 檔），**不會**載入模型權重，本身幾乎不佔 GPU 記憶體，
重新建立它不太可能是 OOM 主因。不過 `04_align.py` 現在還是把它改成**延遲載入**：
只有真的遇到 `en_tokens + zh_tokens > 1024` 需要切句拆分的段落時才會觸發載入
（見 `AlignmentPipeline.set_tokenizer_factory`），如果整批章節都沒有超長段落，
甚至完全不會載入這顆 tokenizer。段落層級的 token 數統計則從頭到尾都是直接讀
`03_tokenize.py` 已經算好、存在 `--tok_dir` 的 JSON 結果，不會重新 tokenize；
只有「拆分後產生的新句子/chunk」才需要重新算 token 數，因為這些切分單位
在 03 的輸出裡本來就不存在，無法回頭重用。