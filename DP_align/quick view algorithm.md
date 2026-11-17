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