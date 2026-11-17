以下將演算法整理為期刊發表用的完整格式，每個儲存格可獨立複製。

---

## Cell 1：標題與問題定義

```markdown
## Boundary-Aware Local Resegmentation for Bilingual Paragraph Alignment

### 1. Problem Definition

**Definition 1 (Coarse Alignment).**
Let $\mathcal{D} = \{(S_i, T_i)\}_{i=1}^{N}$ denote a coarse-aligned bilingual
document consisting of $N$ aligned line pairs, where each line pair comprises
a source-language paragraph sequence $S_i = (s_{i,1}, s_{i,2}, \ldots, s_{i,m_i})$
and a target-language paragraph sequence $T_i = (t_{i,1}, t_{i,2}, \ldots, t_{i,n_i})$.
Each paragraph $p$ is associated with:
  - a dense embedding vector $\mathbf{e}(p) \in \mathbb{R}^d$,
  - a token count $\ell(p) \in \mathbb{Z}^+$,
  - a unique identifier $\mathrm{id}(p)$.

The sequences satisfy the *partition constraint*: every paragraph in the
original document appears in exactly one line pair, and the concatenation
of all $S_i$ (resp. $T_i$) recovers the original source (resp. target)
paragraph sequence.

**Definition 2 (Boundary Error).**
A *boundary error* at position $i$ occurs when the segmentation between
$\mathcal{D}_i$ and $\mathcal{D}_{i+1}$ is incorrect, i.e., there exist
paragraphs that should belong to a different line pair to achieve correct
bilingual correspondence. We restrict boundary errors to be *local*:
all displaced paragraphs reside within the adjacent pair
$(\mathcal{D}_i, \mathcal{D}_{i+1})$.

**Problem (Boundary Resegmentation).**
For each boundary $i \in \{1, \ldots, N-1\}$, determine shift offsets
$(\Delta_s, \Delta_t) \in \mathbb{Z} \times \mathbb{Z}$
such that the resegmented line pairs

$$
\mathcal{D}_i' = (S_i', T_i'), \qquad
\mathcal{D}_{i+1}' = (S_{i+1}', T_{i+1}')
$$

maximize a local bilingual alignment quality score, subject to:
  (i)  the partition constraint is preserved,
  (ii) paragraph order is preserved (monotonicity),
  (iii) the total displacement $|\Delta_s| + |\Delta_t|$ is bounded by
       a maximum shift $K$.
```

---

## Cell 2：符號定義

```markdown
### 2. Notation

| Symbol | Description |
|--------|-------------|
| $\mathcal{D}_i = (S_i, T_i)$ | The $i$-th aligned line pair |
| $S_i = (s_{i,1}, \ldots, s_{i,m_i})$ | Source paragraph sequence of line $i$ |
| $T_i = (t_{i,1}, \ldots, t_{i,n_i})$ | Target paragraph sequence of line $i$ |
| $\mathbf{e}(p) \in \mathbb{R}^d$ | Embedding vector of paragraph $p$ |
| $\ell(p)$ | Token count of paragraph $p$ |
| $\bar{\ell}_S, \bar{\ell}_T$ | Average token counts of source / target |
| $\Delta_s, \Delta_t$ | Boundary shift offsets for source / target |
| $K$ | Maximum allowed shift per language side |
| $\sigma(\cdot)$ | Paragraph matching score function |
| $\Phi(\cdot)$ | Monotonic alignment score (DP) |
| $\Psi(\cdot)$ | Candidate boundary score |
| $\tau_m$ | Match threshold |
| $\lambda_\ell$ | Length-ratio penalty coefficient |
| $\lambda_{\mathrm{mv}}$ | Move penalty coefficient |
| $\lambda_{\mathrm{sq}}$ | Status-quo bonus coefficient |
| $\gamma$ | Gap penalty in alignment DP |
| $\delta$ | Minimum gain threshold for acceptance |
| $\rho$ | Bidirectional consistency score |
| $\mathcal{C}_i$ | Set of candidate shift pairs for boundary $i$ |
```

---

## Cell 3：段落匹配分數

```markdown
### 3. Paragraph Matching Score

**Definition 3 (Cosine Similarity).**
For source paragraph $s$ and target paragraph $t$ with L2-normalized
embeddings $\mathbf{e}(s), \mathbf{e}(t) \in \mathbb{R}^d$, the cosine
similarity is

$$
\cos(s, t) = \frac{\mathbf{e}(s)^\top \mathbf{e}(t)}
                   {\|\mathbf{e}(s)\| \, \|\mathbf{e}(t)\|}.
$$

**Definition 4 (Length-Ratio Factor).**
To mitigate length bias in embedding similarity, define the normalized
length ratio

$$
r(s, t) = \ln \frac{\ell(s) / \bar{\ell}_S}{\ell(t) / \bar{\ell}_T},
$$

and the length factor

$$
\lambda(s, t) = \exp\!\bigl(-\lambda_\ell \, |r(s, t)|\bigr).
$$

**Definition 5 (Raw Matching Score).**
The raw matching score is

$$
\sigma_{\mathrm{raw}}(s, t) = \max\!\bigl(0,\; \cos(s, t)\bigr) \cdot \lambda(s, t).
$$

**Definition 6 (Adjusted Matching Score).**
To allow the alignment DP to prefer gaps over weak matches, we define

$$
\sigma(s, t) = \sigma_{\mathrm{raw}}(s, t) - \tau_m,
$$

where $\tau_m$ is the match threshold. Note that $\sigma(s, t)$ may be
negative, which enables the DP to skip poorly matched paragraphs.
```

---

## Cell 4：局部單調對齊動態規劃

```markdown
### 4. Monotonic Alignment via Dynamic Programming

**Definition 7 (Line Alignment Score).**
Given a source sequence $S = (s_1, \ldots, s_m)$ and a target sequence
$T = (t_1, \ldots, t_n)$, the monotonic alignment score $\Phi(S, T)$ is
computed via the following dynamic program:

$$
\mathrm{dp}[a][b] = \text{optimal alignment score for } (s_1, \ldots, s_a)
\text{ and } (t_1, \ldots, t_b).
$$

**Initialization:**

$$
\mathrm{dp}[0][0] = 0,
$$
$$
\mathrm{dp}[a][0] = \mathrm{dp}[a-1][0] - \gamma, \quad a = 1, \ldots, m,
$$
$$
\mathrm{dp}[0][b] = \mathrm{dp}[0][b-1] - \gamma, \quad b = 1, \ldots, n,
$$

where $\gamma > 0$ is the gap penalty for untranslated paragraphs.

**Recurrence** for $a = 1, \ldots, m$ and $b = 1, \ldots, n$:

$$
\mathrm{dp}[a][b] = \max
\begin{cases}
\mathrm{dp}[a-1][b-1] + \sigma(s_a, t_b), & \text{(match)} \\
\mathrm{dp}[a-1][b] - \gamma, & \text{(skip } s_a\text{)} \\
\mathrm{dp}[a][b-1] - \gamma. & \text{(skip } t_b\text{)}
\end{cases}
$$

**Output:**

$$
\Phi(S, T) = \mathrm{dp}[m][n].
$$

**Remark.** The gap penalty $\gamma$ is set small to tolerate omissions
and additions as valid translation styles. The monotonicity constraint
(no crossing alignments) is enforced by the DP structure.

**Complexity.** $\Phi(S, T)$ is computed in $O(m \cdot n)$ time and space.
```

---

## Cell 5：候選邊界生成

```markdown
### 5. Candidate Boundary Generation

**Definition 8 (Shift Operation).**
For a boundary between $\mathcal{D}_i = (S_i, T_i)$ and
$\mathcal{D}_{i+1} = (S_{i+1}, T_{i+1})$, a shift pair
$(\Delta_s, \Delta_t) \in [-K, K]^2$ induces the resegmentation:

*Source side:*

$$
S_i' =
\begin{cases}
S_i \setminus \text{last } \Delta_s \text{ elements}, & \Delta_s > 0 \\
S_i \,\|\, S_{i+1}[:{-}\Delta_s], & \Delta_s < 0 \\
S_i, & \Delta_s = 0
\end{cases}
$$

$$
S_{i+1}' =
\begin{cases}
S_i[-\Delta_s:] \,\|\, S_{i+1}, & \Delta_s > 0 \\
S_{i+1}[{-}\Delta_s:], & \Delta_s < 0 \\
S_{i+1}, & \Delta_s = 0
\end{cases}
$$

where $\|$ denotes concatenation. The target side is defined analogously
with $\Delta_t$.

**Definition 9 (Candidate Set).**
The candidate set for boundary $i$ is

$$
\mathcal{C}_i = \bigl\{(\Delta_s, \Delta_t) \in [-K, K]^2
\;\big|\; \text{validity constraints hold}\bigr\},
$$

where validity constraints include:
  1. $|\Delta_s| \leq |S_i|$ if $\Delta_s > 0$;
     $|\Delta_s| \leq |S_{i+1}|$ if $\Delta_s < 0$ (analogously for $\Delta_t$).
  2. Neither resulting line pair is empty:
     $|S_i'| + |T_i'| > 0$ and $|S_{i+1}'| + |T_{i+1}'| > 0$.
  3. No previously moved paragraph is displaced again (lock constraint).
  4. Cross-chapter moves are disallowed.
```

---

## Cell 6：候選評分函數

```markdown
### 6. Candidate Scoring Function

**Definition 10 (Normalized Alignment Score).**
For a candidate shift $(\Delta_s, \Delta_t)$ producing resegmented line
pairs $\mathcal{D}_i' = (S_i', T_i')$ and
$\mathcal{D}_{i+1}' = (S_{i+1}', T_{i+1}')$, define

$$
A(\Delta_s, \Delta_t) =
\frac{\Phi(S_i', T_i') + \Phi(S_{i+1}', T_{i+1}')}{|S_i'| + |T_i'| + |S_{i+1}'| + |T_{i+1}'|}.
$$

**Definition 11 (Move Penalty).**
The move penalty discourages unnecessary displacement:

$$
P_{\mathrm{mv}}(\Delta_s, \Delta_t) =
\lambda_{\mathrm{mv}} \cdot \bigl(|\Delta_s| + |\Delta_t|\bigr)
\cdot \bigl(1 + \lambda_v \cdot c_i\bigr),
$$

where $c_i \in [0, 1]$ is the normalized confidence of the original coarse
alignment at boundary $i$, and $\lambda_v$ controls the influence of the
prior alignment confidence.

**Definition 12 (Status-Quo Bonus).**
To bias toward the original segmentation:

$$
B_{\mathrm{sq}}(\Delta_s, \Delta_t) =
\begin{cases}
\lambda_{\mathrm{sq}}, & \text{if } \Delta_s = 0 \text{ and } \Delta_t = 0, \\
0, & \text{otherwise}.
\end{cases}
$$

**Definition 13 (Bidirectional Consistency Score).**
For paragraphs near the boundary, define the bidirectional consistency:

$$
\rho(\Delta_s, \Delta_t) = \sum_{s \in \mathcal{B}_S}
\begin{cases}
+\beta_1, & \text{if } \arg\max_{t \in \mathcal{B}_T} \sigma(s,t)
             = t^* \text{ and } \arg\max_{s' \in \mathcal{B}_S} \sigma(s',t^*)
             = s, \\
-\beta_2, & \text{otherwise},
\end{cases}
$$

where $\mathcal{B}_S$ and $\mathcal{B}_T$ are the boundary-adjacent
paragraph sets (the last $w$ paragraphs of $\mathcal{D}_i'$ and the first
$w$ paragraphs of $\mathcal{D}_{i+1}'$), and $\beta_1, \beta_2 > 0$
are reward and penalty constants.

**Definition 14 (Candidate Score).**
The overall candidate score is

$$
\Psi(\Delta_s, \Delta_t) =
A(\Delta_s, \Delta_t)
+ \rho(\Delta_s, \Delta_t)
+ B_{\mathrm{sq}}(\Delta_s, \Delta_t)
- P_{\mathrm{mv}}(\Delta_s, \Delta_t).
$$
```

---

## Cell 7：決策規則

```markdown
### 7. Decision Rule

**Definition 15 (Acceptance Criterion).**
Let $(\Delta_s^*, \Delta_t^*) = \arg\max_{(\Delta_s, \Delta_t) \in \mathcal{C}_i}
\Psi(\Delta_s, \Delta_t)$ be the best candidate, and let
$\Psi_0 = \Psi(0, 0)$ be the score of the original segmentation.
The shift $(\Delta_s^*, \Delta_t^*)$ is accepted if and only if

$$
\Psi(\Delta_s^*, \Delta_t^*) - \Psi_0 > \delta,
$$

where $\delta > 0$ is the minimum gain threshold.

**Tie-breaking.** If multiple candidates achieve the same score, prefer:
  1. smaller total displacement $|\Delta_s| + |\Delta_t|$,
  2. the original segmentation $(\Delta_s, \Delta_t) = (0, 0)$.

**Locking.** Upon acceptance, all displaced paragraphs are added to a
locked set $\mathcal{L}$ to prevent repeated movement in subsequent
boundary processing.

**Chapter Constraint.** Boundaries between different chapters are never
processed.
```

---

## Cell 8：完整演算法偽代碼

```markdown
### 8. Complete Algorithm

---

**Algorithm 1:** Boundary-Aware Local Resegmentation

---

**Input:**
- Coarse-aligned line pairs $\mathcal{D} = \{(S_i, T_i)\}_{i=1}^{N}$
- Paragraph embeddings $\mathbf{e}(p)$ and token counts $\ell(p)$
- Hyperparameters: $K, \tau_m, \lambda_\ell, \lambda_{\mathrm{mv}},
  \lambda_{\mathrm{sq}}, \lambda_v, \gamma, \delta, w, \beta_1, \beta_2$

**Output:**
- Resegmented line pairs $\mathcal{D}^*$
- Move log $\mathcal{M}$

---

1.  **Preprocessing**
    1.1  Compute average token counts $\bar{\ell}_S, \bar{\ell}_T$.
    1.2  Group paragraph embeddings by chapter:
         $\mathcal{P}_c \leftarrow \{p : \mathrm{chapter}(p) = c\}$.
    1.3  Initialize locked set $\mathcal{L} \leftarrow \emptyset$,
         move log $\mathcal{M} \leftarrow \emptyset$.

2.  **For each boundary** $i = 1, 2, \ldots, N-1$:

    2.1  **Chapter check.**
         If $\mathrm{chapter}(\mathcal{D}_i) \neq \mathrm{chapter}(\mathcal{D}_{i+1})$,
         skip boundary $i$.

    2.2  **Retrieve chapter-specific paragraph dictionaries.**
         $\mathcal{P}_S \leftarrow$ source paragraphs of current chapter.
         $\mathcal{P}_T \leftarrow$ target paragraphs of current chapter.

    2.3  **Check completeness.**
         If any paragraph in $S_i \cup S_{i+1} \cup T_i \cup T_{i+1}$
         is missing from $\mathcal{P}_S$ or $\mathcal{P}_T$, skip boundary $i$.

    2.4  **Build matching matrix.**
         For all $s \in S_i \cup S_{i+1}$ and $t \in T_i \cup T_{i+1}$:

         $$\sigma(s,t) \leftarrow \max(0, \cos(s,t)) \cdot
           \exp(-\lambda_\ell |r(s,t)|) - \tau_m$$

    2.5  **Compute original score.**
         $\Psi_0 \leftarrow \Psi(0, 0)$ via Definitions 10–14.

    2.6  **Enumerate candidates.**
         For each $(\Delta_s, \Delta_t) \in [-K, K]^2 \setminus \{(0,0)\}$:

         (a) Check validity (Definition 9). Skip if invalid.
         (b) Skip if any displaced paragraph is in $\mathcal{L}$.
         (c) Apply shift to obtain $(S_i', T_i')$ and $(S_{i+1}', T_{i+1}')$.
         (d) Compute $\Phi(S_i', T_i')$ and $\Phi(S_{i+1}', T_{i+1}')$
             via Algorithm 2 (DP).
         (e) Compute $\rho(\Delta_s, \Delta_t)$ via Definition 13.
         (f) Compute $\Psi(\Delta_s, \Delta_t)$ via Definition 14.

    2.7  **Select best candidate.**
         $(\Delta_s^*, \Delta_t^*) \leftarrow
         \arg\max_{(\Delta_s, \Delta_t) \in \mathcal{C}_i}
         \Psi(\Delta_s, \Delta_t)$,
         with tie-breaking per Definition 15.

    2.8  **Acceptance decision.**
         If $\Psi(\Delta_s^*, \Delta_t^*) - \Psi_0 > \delta$:
         - Update $\mathcal{D}_i$ and $\mathcal{D}_{i+1}$ with the shift.
         - $\mathcal{L} \leftarrow \mathcal{L} \cup \{\text{displaced paragraphs}\}$.
         - Append $(i, \Delta_s^*, \Delta_t^*, \Psi_0, \Psi^*, \text{moved IDs})$
           to $\mathcal{M}$.

         Else:
         - Keep original segmentation.

3.  **Post-processing**
    3.1  Reconstruct text and token counts from paragraph IDs.
    3.2  Validate:
         - Source paragraph sequence is identical to original.
         - Target paragraph sequence is identical to original.
         - Chapter IDs are unchanged.
    3.3  Return $\mathcal{D}^*$ and $\mathcal{M}$.

---

**Algorithm 2:** Monotonic Alignment DP ($\Phi$)

---

**Input:** Source sequence $S = (s_1, \ldots, s_m)$,
          target sequence $T = (t_1, \ldots, t_n)$,
          matching matrix $\sigma$, gap penalty $\gamma$.

**Output:** Alignment score $\Phi(S, T)$.

---

1.  Initialize $\mathrm{dp}[0][0] \leftarrow 0$.
2.  For $a = 1$ to $m$:
      $\mathrm{dp}[a][0] \leftarrow \mathrm{dp}[a-1][0] - \gamma$.
3.  For $b = 1$ to $n$:
      $\mathrm{dp}[0][b] \leftarrow \mathrm{dp}[0][b-1] - \gamma$.
4.  For $a = 1$ to $m$:
      For $b = 1$ to $n$:

      $$\mathrm{dp}[a][b] \leftarrow \max
      \begin{cases}
      \mathrm{dp}[a-1][b-1] + \sigma(s_a, t_b) \\
      \mathrm{dp}[a-1][b] - \gamma \\
      \mathrm{dp}[a][b-1] - \gamma
      \end{cases}$$

5.  Return $\mathrm{dp}[m][n]$.

---
```

---

## Cell 9：複雜度分析

```markdown
### 9. Complexity Analysis

**Theorem 1 (Time Complexity).**
For a single boundary $i$ with $|S_i| + |S_{i+1}| = M_s$ source paragraphs
and $|T_i| + |T_{i+1}| = M_t$ target paragraphs, the time complexity of
boundary resegmentation is

$$
O\!\bigl(K^2 \cdot M_s \cdot M_t\bigr),
$$

where $K$ is the maximum shift.

**Proof.**
The candidate set $|\mathcal{C}_i| \leq (2K+1)^2 - 1 = O(K^2)$.
For each candidate, the DP alignment $\Phi$ is computed on two line pairs
whose total sizes are bounded by $M_s$ and $M_t$, requiring
$O(M_s \cdot M_t)$ time. The matching matrix construction is
$O(M_s \cdot M_t)$. The bidirectional consistency check is
$O(w^2)$ where $w$ is the boundary window size, which is dominated
by $O(M_s \cdot M_t)$ for $w \ll M_s, M_t$.
$\square$

**Theorem 2 (Space Complexity).**
The space complexity per boundary is $O(M_s \cdot M_t)$ for the matching
matrix and DP table.

**Corollary 1 (Full Document).**
For a document with $N$ line pairs and total paragraph counts $M_S, M_T$,
the total time complexity is

$$
O\!\bigl(N \cdot K^2 \cdot \bar{M}_s \cdot \bar{M}_t\bigr),
$$

where $\bar{M}_s, \bar{M}_t$ are the average local paragraph counts
per boundary window. Since $K$ and $\bar{M}_s, \bar{M}_t$ are typically
small constants ($K \leq 3$, $\bar{M}_s, \bar{M}_t \leq 20$), the
algorithm runs in effectively $O(N)$ time.

**Theorem 3 (Partition Preservation).**
The algorithm preserves the partition constraint: after processing all
boundaries, every paragraph appears exactly once, and the global paragraph
order is unchanged.

**Proof.**
Each shift operation moves a contiguous suffix (or prefix) of one line to
the adjacent line. This is equivalent to moving a boundary pointer within
a fixed paragraph sequence. The operation is invertible and does not
create or destroy paragraphs. The final validation step (Step 3.2)
confirms this invariant. $\square$

**Theorem 4 (Termination).**
The algorithm terminates after exactly $N - 1$ boundary iterations.
Each iteration enumerates a finite candidate set of size $O(K^2)$.
The locking mechanism ensures no paragraph is moved more than once,
preventing oscillation. $\square$
```

---

## Cell 10：超參數表

```markdown
### 10. Hyperparameter Summary

| Symbol | Parameter | Default | Description |
|--------|-----------|---------|-------------|
| $K$ | `max-shift` | 1 | Maximum shift per language side |
| $\tau_m$ | `match-threshold` | 0.20 | Raw score threshold for DP matching |
| $\lambda_\ell$ | `lambda-len` | 0.20 | Length-ratio penalty strength |
| $\lambda_{\mathrm{mv}}$ | `move-penalty` | 0.05 | Per-paragraph move penalty |
| $\lambda_{\mathrm{sq}}$ | `status-quo-bonus` | 0.02 | Bonus for keeping original segmentation |
| $\lambda_v$ | `vecalign-weight` | 0.3 | Prior confidence weight |
| $\gamma$ | `gap-penalty` | 0.05 | DP gap penalty for untranslated paragraphs |
| $\delta$ | `min-gain` | 0.02 | Minimum score gain for acceptance |
| $w$ | `boundary-window` | 2 | Boundary window for bidirectional check |
| $\beta_1$ | `bidir-reward` | 0.01 | Reward for bidirectional consistency |
| $\beta_2$ | `bidir-penalty` | 0.03 | Penalty for unidirectional match |
```

---

## Cell 11：LaTeX 演算法環境版本

```latex
\usepackage{algorithm}
\usepackage{algorithmic}

\begin{algorithm}[t]
\caption{Boundary-Aware Local Resegmentation}
\label{alg:resegment}
\begin{algorithmic}[1]
\REQUIRE Coarse-aligned lines $\{(S_i, T_i)\}_{i=1}^{N}$,
         embeddings $\mathbf{e}(p)$,
         hyperparameters $K, \tau_m, \lambda_\ell, \lambda_{\mathrm{mv}},
         \lambda_{\mathrm{sq}}, \gamma, \delta$
\ENSURE Resegmented lines $\{(S_i^*, T_i^*)\}_{i=1}^{N}$, move log $\mathcal{M}$

\STATE Compute $\bar{\ell}_S, \bar{\ell}_T$; group paragraphs by chapter
\STATE $\mathcal{L} \leftarrow \emptyset$; $\mathcal{M} \leftarrow \emptyset$

\FOR{$i = 1$ \TO $N - 1$}
  \IF{$\mathrm{chapter}(\mathcal{D}_i) \neq \mathrm{chapter}(\mathcal{D}_{i+1})$}
    \CONTINUE
  \ENDIF

  \STATE Build matching matrix $\sigma(s, t)$ for all
         $s \in S_i \cup S_{i+1}$, $t \in T_i \cup T_{i+1}$
  \STATE $\Psi_0 \leftarrow \Psi(0, 0)$
         \hfill $\triangleright$ Original segmentation score

  \FOR{$(\Delta_s, \Delta_t) \in [-K, K]^2 \setminus \{(0,0)\}$}
    \IF{invalid or locked}
      \CONTINUE
    \ENDIF
    \STATE Apply shift; compute $\Phi$ via DP (Algorithm~\ref{alg:dp})
    \STATE Compute $\rho$ (bidirectional consistency)
    \STATE $\Psi \leftarrow A + \rho + B_{\mathrm{sq}} - P_{\mathrm{mv}}$
  \ENDFOR

  \STATE $(\Delta_s^*, \Delta_t^*) \leftarrow
         \arg\max \Psi$, with tie-breaking
  \IF{$\Psi(\Delta_s^*, \Delta_t^*) - \Psi_0 > \delta$}
    \STATE Update $\mathcal{D}_i, \mathcal{D}_{i+1}$;
           $\mathcal{L} \leftarrow \mathcal{L} \cup \{\text{moved}\}$;
           append to $\mathcal{M}$
  \ENDIF
\ENDFOR

\STATE Reconstruct text; validate partition and chapter integrity
\RETURN $\{(S_i^*, T_i^*)\}$, $\mathcal{M}$
\end{algorithmic}
\end{algorithm}


\begin{algorithm}[t]
\caption{Monotonic Alignment DP}
\label{alg:dp}
\begin{algorithmic}[1]
\REQUIRE Sequences $S = (s_1, \ldots, s_m)$, $T = (t_1, \ldots, t_n)$,
         matching matrix $\sigma$, gap penalty $\gamma$
\ENSURE Alignment score $\Phi(S, T)$

\STATE $\mathrm{dp}[0][0] \leftarrow 0$
\FOR{$a = 1$ \TO $m$}
  \STATE $\mathrm{dp}[a][0] \leftarrow \mathrm{dp}[a-1][0] - \gamma$
\ENDFOR
\FOR{$b = 1$ \TO $n$}
  \STATE $\mathrm{dp}[0][b] \leftarrow \mathrm{dp}[0][b-1] - \gamma$
\ENDFOR
\FOR{$a = 1$ \TO $m$}
  \FOR{$b = 1$ \TO $n$}
    \STATE $\mathrm{dp}[a][b] \leftarrow \max\bigl(
      \mathrm{dp}[a-1][b-1] + \sigma(s_a, t_b),\;
      \mathrm{dp}[a-1][b] - \gamma,\;
      \mathrm{dp}[a][b-1] - \gamma
    \bigr)$
  \ENDFOR
\ENDFOR
\RETURN $\mathrm{dp}[m][n]$
\end{algorithmic}
\end{algorithm}
```

---

## Cell 12：實驗設置建議

```markdown
### 11. Recommended Experimental Setup

**Dataset.** Bilingual novel corpus with paragraph-level gold alignment.
Report statistics: number of chapters, paragraphs, average paragraphs
per line, language pair.

**Baseline.** Original Vecalign output without boundary resegmentation.

**Evaluation Metrics.**
  1. *Boundary Accuracy (BA):* fraction of boundaries that exactly match
     the gold segmentation.
  2. *Paragraph-level F1:* precision and recall of paragraph-to-line
     assignment.
  3. *Alignment Error Rate (AER):* fraction of incorrectly aligned
     paragraph pairs.
  4. *Move Precision / Recall:* among all proposed moves, fraction that
     are correct (precision) and fraction of true errors that are caught
     (recall).

**Ablation Study.**
  - Effect of bidirectional consistency ($\rho$).
  - Effect of status-quo bonus ($\lambda_{\mathrm{sq}}$).
  - Effect of maximum shift $K \in \{1, 2, 3\}$.
  - Effect of prior confidence weight $\lambda_v$.
  - Effect of match threshold $\tau_m$.

**Statistical Significance.** Use paired bootstrap test or McNemar's test
to compare against baseline. Report 95\% confidence intervals.
```

---

## Cell 13：參考文獻格式建議

```markdown
### 12. Key References to Cite

1. **Vecalign:** Thompson, B., & Koehn, P. (2019).
   Vecalign: Improved sentence alignment in linear time and space.
   *Proceedings of EMNLP-IJCNLP*, 1342–1348.

2. **Sentence alignment DP:** Gale, W. A., & Church, K. W. (1993).
   A program for aligning sentences in bilingual corpora.
   *Computational Linguistics*, 19(1), 75–102.

3. **Margin-based scoring:** Artetxe, M., & Schwenk, H. (2019).
   Massively multilingual sentence embeddings for zero-shot
   cross-lingual transfer and beyond.
   *Transactions of the ACL*, 7, 597–610.

4. **Cross-lingual embeddings:** Conneau, A., et al. (2020).
   Unsupervised cross-lingual representation learning at scale.
   *Proceedings of ACL*, 8440–8451.

5. **Bilingual document alignment:** Aker, A., & Gaizauskas, R. (2010).
   Creating bilingual corpora from comparable web data.
   *Proceedings of LREC*.
```

---

以上 13 個儲存格涵蓋了期刊發表所需的完整演算法描述。你可以根據目標期刊的格式要求，將 Cell 11 的 LaTeX 版本直接嵌入論文的 Method 章節，其餘儲存格作為補充材料或附錄。