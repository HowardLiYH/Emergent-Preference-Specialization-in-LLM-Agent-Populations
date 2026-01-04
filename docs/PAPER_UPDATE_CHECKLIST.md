# Paper Update Checklist

**When to use**: After gemini-2.5-flash 10-seed validation completes

---

## Sections to Update

### 1. Abstract
**Current**: "72.2% pass rate across 3 seeds, 95% CI: [48.5%, 96.0%]"
**Update to**: "[NEW_MEAN]% pass rate across 10 seeds, 95% CI: [NEW_CI_LOW%, NEW_CI_HIGH%]"

### 2. Section 5.1 - What We Proved
**Current**: "72.2% pass rate across 3 seeds"
**Update to**: "[NEW_MEAN]% pass rate across 10 seeds"

### 3. Section 5.2 - Multi-Seed Validation (Table 4)
**Current**:
```
Seeds 1-3: Gemini, Seeds 4-7: GPT-4o-mini
Mean: 61.9%, 95% CI: [46.6%, 77.2%]
```
**Update to**:
```
All 10 seeds: gemini-2.5-flash (unified)
Mean: [NEW_MEAN]%, 95% CI: [NEW_CI_LOW%, NEW_CI_HIGH%]
```

### 4. Section 7 - Limitations
**Current**: "Seed-switching analysis from Phase 1 evolution logs not yet completed"
**Update**: Remove this line if seed-switching analysis is complete

### 5. Section 8 - Conclusion
**Current**: "75.5% pass rate, validated across 5 seeds"
**Update to**: "[NEW_MEAN]% pass rate, validated across 10 seeds (unified model)"

---

## New Sections to Add

### 5.X - Seed-Switching Analysis
```latex
\subsection{Seed-Switching Analysis}

To verify that specialization emerges from competition rather than initial random assignment (Option B+), we analyze whether agents switch from their initial seeded rule to a different final specialization.

\textbf{Method}: For each Phase 1 checkpoint, we compare the agent's initial rule (randomly assigned via Option B+) with their final primary specialization (highest-level strategy).

\textbf{Results}:
\begin{itemize}
    \item Switch Rate: [SWITCH_RATE]\%
    \item Chi-square: [CHI_SQUARE] (p = [P_VALUE])
    \item Interpretation: [INTERPRETATION]
\end{itemize}

This confirms that competition drives specialization beyond initial random assignment.
```

### 5.Y - MMLU Real-World Validation
```latex
\subsection{Real-World Transfer (MMLU)}

We validate that specialized prompts transfer to real-world tasks using MMLU domains.

\begin{table}[h]
\caption{MMLU Validation Results}
\centering
\begin{tabular}{lccc}
\toprule
Domain & Specialist & Generic & Improvement \\
\midrule
Abstract Algebra & [X]\% & [Y]\% & [Z]\% \\
US History & [X]\% & [Y]\% & [Z]\% \\
High School Biology & [X]\% & [Y]\% & [Z]\% \\
Professional Law & [X]\% & [Y]\% & [Z]\% \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Files to Update

| File | Changes |
|------|---------|
| `paper/neurips_2025.tex` | Update all sections above |
| `README.md` | Update results summary |
| `CHANGELOG.md` | Add v3.11.0 entry |

---

## Values to Fill In (from experiment results)

```
# From results/unified_gemini25/all_seeds.json
NEW_MEAN = ?
NEW_STD = ?
NEW_CI_LOW = ?
NEW_CI_HIGH = ?

# From results/seed_switching_analysis.json
SWITCH_RATE = ?
CHI_SQUARE = ?
P_VALUE = ?
INTERPRETATION = ?

# From results/mmlu_validation/mmlu_final_results.json
MMLU results...
```

---

## Verification Checklist

- [ ] All 10 seeds completed with gemini-2.5-flash
- [ ] all_seeds.json saved with final results
- [ ] Seed-switching analysis run
- [ ] MMLU validation run
- [ ] All p-values computed with real data
- [ ] No simulated or fabricated numbers
- [ ] AUDIT_LOG.md updated if any changes
