進行跨資料夾、檔案內容合併成JSNOL，當前有四個資料夾檔案要處理：'en','zh','tokens','embedding'。
### 資料夾內容與格式

#### "zh"
當中包含67個txt檔案，檔名為f"ch{num}.txt"(num從0到66)

#### "en"
當中包含67個txt檔案，檔名為f"ch{num}.txt"(num從0到66)

#### "tokens"
包含67個json，檔名為f"{num}.json"(num從0到66)，內容格式為
```
{
  "zh": {
    "p1": 92,
    "p2": 146,
    "p3": 327,
    "p4": 150,
    "p5": 124,
    "p6": 77,
    "p7": 55,
    "p8": 90
  },
  "en": {
    "p1": 104,
    "p2": 149,
    "p3": 311,
    "p4": 143,
    "p5": 123,
    "p6": 73,
    "p7": 55,
    "p8": 98
  }
}
```
當中的"zh","en"對應同名資料夾，依據json檔名編號對應資料夾中f"ch{num}.txt"，當中的f"p{num}"對應txt的行(line)文字token數

#### "embeddings"
包含67個子資料夾，檔名為f"{num}"(num從0到66)，當中的子資料夾檔名編號對應"zh","en"中f"ch{num}.txt"編號。每份子資料夾中包含三份檔案："en.npy","zh.npy","meta.json"。

> "meta.json" 內容
```
{
  "chapter_id": "0",
  "task": "text-matching",
  "zh_para_ids": [
    "p1",
    "p2",
    "p3",
    "p4",
    "p5",
    "p6",
    "p7",
    "p8"
  ],
  "en_para_ids": [
    "p1",
    "p2",
    "p3",
    "p4",
    "p5",
    "p6",
    "p7",
    "p8"
  ]
}
```
    "chapter_id"對應"zh","en"中f"ch{num}.txt"編號。"zh_para_ids"與"en_para_ids"的f"p{num}"分別對應txt的行(line)定位


> "en.npy"
    資料形狀 (Shape): (i, 1024)的numpy，當中的i對應"meta.json"中的"en_para_ids"，即表示'en'資料夾中的f"ch{num}.txt"的每line文字經過embedding後的numpy。

> "zh.npy"
    資料形狀 (Shape): (i, 1024)的numpy，當中的i對應"meta.json"中的"zh_para_ids"，即表示'zh'資料夾中的f"ch{num}.txt"的每line文字經過embedding後的numpy。

---
### 目標生成檔案格式
- Paragraph files are JSONL with fields such as:
    * chapter_id
    * para_id
    * embedding
    * token
    * text

* chapter_id: "en"與"zh"中f"ch{num}.txt"的num編號
* para_id: "embeddings"中，"meta.json"的"zh_para_ids"或"en_para_ids"編號
* embedding: "embeddings"中，'.npy'檔的對應內容
* token: "tokens"中，f"{num}.json"於對應編號下的數值
* text: "zh","en"中依據"chapter_id"定位的txt檔案與"para_id"定位的line位置，所在的文字內容

### 輸出檔案
Use separate paragraph files for Chinese and English:
    * zh_paras.jsnol
    * en_paras.jsnol

### python程式撰寫規格
1. 允許輸入參數：4個資料夾路徑

```
data/
├── zh/
│   ├── ch0.txt
│   ├── ch1.txt
│   └── ...
├── en/
│   ├── ch0.txt
│   ├── ch1.txt
│   └── ...
├── tokens/
│   ├── 0.json
│   ├── 1.json
│   └── ...
└── embeddings/
    ├── 0/
    │   ├── meta.json
    │   ├── zh.npy
    │   └── en.npy
    ├── 1/
    │   ├── meta.json
    │   ├── zh.npy
    │   └── en.npy
    └── ...
```

---

## 使用方式
```
# 先驗證對齊
python verify_alignment.py ./zh ./en ./tokens ./embeddings

# 驗證通過後再執行合併
python merge_paras.py ./zh ./en ./tokens ./embeddings
```