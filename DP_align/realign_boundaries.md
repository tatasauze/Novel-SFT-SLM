## 局部邊界重切分演算法

### 1. 目標

在粗對齊結果的相鄰兩行之間，檢查是否存在邊界切分錯誤，並進行最小幅度修正。

給定：

```text
L_i     = (Z_i, E_i)
L_{i+1} = (Z_{i+1}, E_{i+1})
```

其中：

```text
Z_i = line_i 的中文段落列表
E_i = line_i 的英文段落列表
```

目標是決定是否調整邊界：

```text
L_i | L_{i+1}
```

變成：

```text
L_i' | L_{i+1}'
```

使得局部雙語對齊品質更好。

---

### 2. 基本假設

1. 粗對齊結果已經是段落層級的相鄰區塊。
2. 錯誤主要發生在相鄰行邊界。
3. 若存在錯位，相關段落通常只會出現在：
   ```text
   line_i 與 line_{i+1}
   ```
4. 缺漏、增譯可視為翻譯風格，不強迫每個段落都必須有對應。
5. 修正時必須保持：
   - 段落順序不變；
   - 段落 ID 不重複；
   - 段落 ID 不消失；
   - 每行仍為相鄰段落組成的區塊。

---

### 3. 輸入資料

#### 3.1 粗對齊 JSONL

每行包含：

```text
chapter_id
parent_en_para_ids
parent_zh_para_ids
en_text
zh_text
en_tokens
zh_tokens
similarity
vecalign_score
```

其中：

```text
parent_zh_para_ids
parent_en_para_ids
```

是段落 ID 列表。

---

#### 3.2 段落資訊檔

每段包含：

```text
chapter_id
para_id
embedding
token
text
```

其中：

```text
token
```

可作為段落長度。

---

### 4. 核心思想

不要只判斷「某一段是否為孤兒」。

改為：

> 對每個邊界枚舉多種可能的中文／英文邊界移動方式，  
> 對每種候選切法計算局部單調對齊分數，  
> 只接受能顯著提升局部對齊品質的切法。

---

### 5. 候選邊界移動

對每個邊界：

```text
L_i | L_{i+1}
```

枚舉：

```text
zh_shift ∈ [-K, K]
en_shift ∈ [-K, K]
```

建議：

```text
K = 1
```

若錯誤較常為多段，可設：

```text
K = 2
```

---

#### 5.1 移動定義

```text
zh_shift > 0
```

代表：

```text
把 line_i 中文最後 zh_shift 段移到 line_{i+1} 中文開頭。
```

```text
zh_shift < 0
```

代表：

```text
把 line_{i+1} 中文開頭 -zh_shift 段移回 line_i 中文尾部。
```

英文同理。

---

### 6. 段落匹配分數

對中文段落 `z` 與英文段落 `e`：

#### 6.1 原始相似度

```text
cos(z, e) = cosine(embedding_z, embedding_e)
```

若相似度小於 0，視為 0：

```text
cos_clipped = max(0, cos(z, e))
```

---

#### 6.2 長度懲罰

使用 token count 作為長度：

```text
len_z = token_count(z)
len_e = token_count(e)
```

長度比例：

```text
len_ratio =
    log(
        (len_z / avg_len_z)
        /
        (len_e / avg_len_e)
    )
```

長度因子：

```text
length_factor = exp(-lambda_len * abs(len_ratio))
```

因此：

```text
raw_score(z, e) = cos_clipped * length_factor
```

---

#### 6.3 匹配門檻

為了避免弱匹配被強迫接受：

```text
match_score(z, e) = raw_score(z, e) - match_threshold
```

`match_score` 可以是負數。

若低於門檻，DP 會傾向跳過該段落，而不是強行配對。

---

### 7. 局部單調對齊 DP

對任一候選行：

```text
Z = [z_1, z_2, ..., z_m]
E = [e_1, e_2, ..., e_n]
```

使用 DP：

```text
dp[a][b] =
    前 a 個中文段落與前 b 個英文段落的最佳對齊分數
```

初始化：

```text
dp[0][0] = 0
dp[a][0] = dp[a-1][0] - gap_penalty
dp[0][b] = dp[0][b-1] - gap_penalty
```

轉移：

```text
dp[a][b] = max(
    dp[a-1][b-1] + match_score(z_a, e_b),

    dp[a-1][b] - gap_penalty,

    dp[a][b-1] - gap_penalty
)
```

其中：

```text
dp[a-1][b-1] + match_score(z_a, e_b)
```

代表中文段落 `z_a` 與英文段落 `e_b` 配對。

```text
dp[a-1][b] - gap_penalty
```

代表中文段落 `z_a` 沒有英文對應，視為漏譯、增譯或翻譯風格。

```text
dp[a][b-1] - gap_penalty
```

代表英文段落 `e_b` 沒有中文對應，視為漏譯、增譯或翻譯風格。

最終：

```text
AlignScore(Z, E) = dp[m][n]
```

---

### 8. 候選切法總分

對候選位移：

```text
zh_shift, en_shift
```

生成：

```text
L_i'  = (Z_i', E_i')
L_{i+1}' = (Z_{i+1}', E_{i+1}')
```

計算：

```text
align_left  = AlignScore(Z_i', E_i')
align_right = AlignScore(Z_{i+1}', E_{i+1}')
```

總段落數：

```text
total_paragraphs =
    len(Z_i') + len(E_i')
  + len(Z_{i+1}') + len(E_{i+1}')
```

正規化對齊分數：

```text
normalized_align_score =
    (align_left + align_right)
    /
    total_paragraphs
```

---

### 9. 移動懲罰

为了避免輕微分數提升就移動邊界：

```text
move_penalty =
    move_penalty_weight
    *
    (abs(zh_shift) + abs(en_shift))
```

可選地使用原始 `vecalign_score` 作為先驗：

```text
effective_move_penalty =
    move_penalty
    *
    (1 + vecalign_weight * vecalign_confidence)
```

其中：

```text
vecalign_confidence
```

可由相鄰兩行的 `vecalign_score` 平均得到。

若原始 `vecalign_score` 越高代表邊界越可信，則：

```text
vecalign_score 越高，移動懲罰越大。
```

---

### 10. 候選總分

```text
candidate_score =
    normalized_align_score
  - effective_move_penalty
  - balance_penalty
```

其中：

```text
balance_penalty
```

是可選的中文／英文段落數不平衡懲罰。

若缺漏／增譯視為翻譯風格，可以設：

```text
balance_penalty = 0
```

---

### 11. 接受條件

目前邊界為：

```text
zh_shift = 0
en_shift = 0
```

其分數為：

```text
current_score
```

最佳候選分數為：

```text
best_score
```

只有當：

```text
best_score - current_score > min_gain
```

才接受移動。

否則維持原切法。

---

### 12. 邊界處理流程

對每個相鄰邊界：

```text
For i in 0 .. N-2:

    L_left  = lines[i]
    L_right = lines[i+1]

    1. 檢查是否同一章。
       若不同章，預設不處理。

    2. 收集局部中文／英文段落。

    3. 檢查 embedding 是否齊全。
       若缺 embedding，預設跳過。

    4. 建立局部匹配矩陣：
       match_score(zh_para, en_para)

    5. 枚舉：
       zh_shift ∈ [-K, K]
       en_shift ∈ [-K, K]

    6. 對每個候選切法：
       - 檢查合法性；
       - 重新切分中文／英文段落；
       - 用 DP 計算兩行的局部對齊分數；
       - 計算候選總分。

    7. 選擇最佳候選。

    8. 若增益大於 min_gain：
       - 接受移動；
       - 更新 parent_zh_para_ids；
       - 更新 parent_en_para_ids；
       - 使用 paragraph text 重建 zh_text / en_text；
       - 使用 token count 重建 zh_tokens / en_tokens；
       - 鎖定已移動段落，避免後續重複移動。

    9. 否則保持原樣。
```

---

### 13. 合法性限制

候選移動必須滿足：

1. 不能移出超過現有段落數量。
2. 不能造成某一行完全沒有內容。
3. 預設不跨 `chapter_id` 移動。
4. 已移動過的段落不再移動。
5. 可選：每行必須同時有中文與英文。
6. 可選：每行每側段落數不能超過上限。

---

### 14. 輸出

修正後每行包含：

```text
parent_zh_para_ids
parent_en_para_ids
zh_text
en_text
zh_tokens
en_tokens
realigned
local_align_score
```

其中：

```text
realigned = true
```

代表該行曾被邊界重切分修正。

```text
local_align_score
```

是重切分後計算出的局部對齊分數。

原始欄位：

```text
similarity
vecalign_score
```

可保留，但注意：  
如果該行被移動過，這些原始分數可能已過期。  
若需要新的相似度，可之後用重建文本與段落向量重新計算。

---

### 15. 最終一致性檢查

程式結束前必須驗證：

```text
所有中文段落 ID 的順序與數量沒有改變。
所有英文段落 ID 的順序與數量沒有改變。
```

也就是：

```text
original_zh_ids == new_zh_ids
original_en_ids == new_en_ids
```

若不成立，代表移動邏輯有錯誤，應停止並檢查。

---

### 16. 最小可用版參數建議

初始建議：

```text
max_shift = 1
match_threshold = 0.20
gap_penalty = 0.05
move_penalty = 0.03
min_gain = 0.01
lambda_len = 0.20
balance_penalty = 0.0
vecalign_weight = 0.0
```

若希望更保守：

```text
max_shift = 1
move_penalty = 0.05
min_gain = 0.02
vecalign_weight = 0.3
```

若希望更積極修正邊界：

```text
max_shift = 2
move_penalty = 0.02
min_gain = 0.008
```

但建議先用小樣本進行人工檢查，再全量執行。