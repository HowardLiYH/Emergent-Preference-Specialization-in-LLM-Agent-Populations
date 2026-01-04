# Unified Validation Plan (Final) - Updated for gemini-2.5-flash

## Objective
Strengthen NeurIPS submission from 6.5/10 to 7.0-7.5/10 with $0 cost and unified methodology.

---

## ⚠️ MODEL CHANGE (Jan 4, 2026)
**All experiments now use `gemini-2.5-flash`** instead of `gemini-2.0-flash`.

Benefits:
- Higher rate limits (1000 RPM vs 15 RPM)
- Better reasoning quality
- More robust for complex tasks

---

## Unity Requirements

| Aspect | Standard | Status |
|--------|----------|--------|
| Primary Model | **gemini-2.5-flash** | 🔄 In Progress (re-running) |
| Temperature | 0.1 | ✅ Already unified |
| Prompts | Enhanced (500+ chars) | ✅ Already unified |
| Rules | 8 synthetic rules | ✅ Already unified |

---

## What Needs to Be Re-Run with gemini-2.5-flash

### Previously Completed (with OLD models - NEED RE-RUN)

| Experiment | Old Model | New Model | Status |
|------------|-----------|-----------|--------|
| Seeds 1-3 | gemini-2.0-flash | gemini-2.5-flash | 🔄 Re-running |
| Seeds 4-5 | gpt-4o-mini | gemini-2.5-flash | 🔄 Re-running |
| Seeds 6-7 | gpt-4o-mini | gemini-2.5-flash | 🔄 Re-running |
| Seeds 8-10 | N/A | gemini-2.5-flash | 🔄 Running (new) |
| Baseline comparison | gemini-2.0-flash | gemini-2.5-flash | ⏳ Pending |
| Scalability test | gemini-2.0-flash | gemini-2.5-flash | ⏳ Pending |
| Temperature sensitivity | gemini-2.0-flash | gemini-2.5-flash | ⏳ Pending |
| Fitness sharing ablation | gemini-2.0-flash | gemini-2.5-flash | ⏳ Pending |

### Cross-LLM Validation (Keep separate - NOT re-running)

| Model | Purpose | Status |
|-------|---------|--------|
| GPT-4o-mini | Cross-LLM generalization | ✅ Keep as-is |
| Claude 3 Haiku | Cross-LLM generalization | ✅ Keep as-is |

---

## Phase 1: Unified 10-Seed Validation (CURRENT)

### 1.1 Run ALL Seeds 1-10 with gemini-2.5-flash
- **File**: experiments/exp_multi_seed.py
- **Seeds**: 1-10 (ALL re-run for unity)
- **Model**: gemini-2.5-flash
- **Rate limit**: 1000 RPM (fast!)
- **Status**: 🔄 Seed 1 done (73.2%), Seed 2 in progress

### 1.2 Current Progress
| Seed | Pass Rate | Status |
|------|-----------|--------|
| 1 | 73.2% | ✅ Done |
| 2 | - | 🔄 In Progress |
| 3-10 | - | ⏳ Pending |

### 1.3 Expected Output
- **File**: results/unified_gemini25/all_seeds.json
- **Expected CI**: [55%, 75%] (lower bound above 50%)

---

## Phase 2: Seed-Switching Analysis

### 2.1 Implement Analysis
**Goal**: Prove final specialization differs from initial seed (Option B+)

- **File**: Add function to `src/genesis/analysis.py`
- **Method**:
  1. Load Phase 1 checkpoint data
  2. Compare initial rule (Option B+ assignment) vs. final primary
  3. Compute Switch Rate = % agents with final != initial
  4. Run chi-square test for statistical significance

### 2.2 Success Criteria
- Switch Rate > 60%
- Chi-square p-value < 0.05

---

## Phase 3: MMLU Real-World Validation

### 3.1 Create MMLU Experiment
- **File**: Create `experiments/exp_mmlu_validation.py`
- **Model**: gemini-2.5-flash
- **Dataset**: HuggingFace `cais/mmlu`

### 3.2 Domain Selection
| MMLU Domain | Synthetic Rule Mapping | N Questions |
|-------------|----------------------|-------------|
| abstract_algebra | MATH_MOD | 50 |
| us_history | POSITION/PATTERN | 50 |
| high_school_biology | ANIMATE | 50 |
| professional_law | INVERSE | 50 |

### 3.3 Experiment Design
```
For each domain:
  1. Create specialist prompt (based on evolved strategy)
  2. Create generic prompt (baseline)
  3. Test specialist on own domain (50 questions)
  4. Test generic on same domain (control)
  5. Compute accuracy gap
```

### 3.4 Success Criteria
- Specialist accuracy > Generic accuracy by >5%
- At least 3/4 domains show improvement

---

## Phase 4: Re-run Other Experiments with gemini-2.5-flash

### 4.1 Baseline Comparison
- **File**: experiments/exp_baselines_full.py
- **Tests**: NO_PROMPT, RANDOM_PROMPT, WRONG_PROMPT, CORRECT_PROMPT
- **Model**: gemini-2.5-flash

### 4.2 Scalability Test
- **File**: experiments/exp_scalability.py
- **Agent counts**: N=8, 12, 24, 48
- **Model**: gemini-2.5-flash

### 4.3 Temperature Sensitivity
- **Temperatures**: 0.1, 0.3, 0.5, 0.7
- **Model**: gemini-2.5-flash

### 4.4 Fitness Sharing Ablation
- **Conditions**: With fitness sharing, Without fitness sharing
- **Model**: gemini-2.5-flash

---

## Phase 5: Update Documentation

### 5.1 Paper Updates (paper/neurips_2025.tex)
- [ ] Update Table with 10-seed unified results
- [ ] Add seed-switching analysis results
- [ ] Add Section 5.4: MMLU Real-World Validation
- [ ] Update model description to specify gemini-2.5-flash

### 5.2 README Updates
- [ ] Update results summary
- [ ] Add MMLU validation section
- [ ] Clarify model unity (gemini-2.5-flash for all primary results)

### 5.3 CHANGELOG Updates
- [ ] Document v3.10.0: Unified Validation with gemini-2.5-flash

---

## Timeline

| Day | Tasks | Est. Time |
|-----|-------|-----------|
| Day 1 | Seeds 1-10 with gemini-2.5-flash | 4-6 hours |
| Day 2 | Seed-switching analysis + Baseline re-run | 3-4 hours |
| Day 3 | MMLU validation + Other experiments | 4-6 hours |
| Day 4 | Paper, README, CHANGELOG updates | 2-3 hours |

---

## Cost Summary

| Item | Cost |
|------|------|
| Seeds 1-10 (gemini-2.5-flash) | $0 |
| Baseline re-run | $0 |
| Scalability re-run | $0 |
| MMLU validation | $0 |
| All analysis | $0 |
| **Total** | **$0** |

---

## Final Model Assignment

| Category | Model | Notes |
|----------|-------|-------|
| **All Primary Results** | gemini-2.5-flash | 10 seeds, all experiments |
| Cross-LLM (separate) | GPT-4o-mini | Generalization only |
| Cross-LLM (separate) | Claude 3 Haiku | Generalization only |

---

## Success Criteria Summary

| Metric | Current | Target |
|--------|---------|--------|
| Model unity | 10% (1/10) | 100% (10/10) |
| 95% CI lower bound | TBD | >50% |
| Seed switch rate | N/A | >60% |
| MMLU accuracy gap | N/A | >5% |
| Projected score | 6.5 | 7.0-7.5 |

---

## Files to Modify/Create

| File | Action | Status |
|------|--------|--------|
| experiments/exp_multi_seed.py | Modify - use gemini-2.5-flash | ✅ Done |
| src/genesis/llm_client.py | Modify - fix empty parts handling | ✅ Done |
| src/genesis/analysis.py | Modify - add seed-switching | ⏳ Pending |
| experiments/exp_mmlu_validation.py | Create | ⏳ Pending |
| experiments/exp_baselines_full.py | Modify - use gemini-2.5-flash | ⏳ Pending |
| experiments/exp_scalability.py | Modify - use gemini-2.5-flash | ⏳ Pending |
| paper/neurips_2025.tex | Modify | ⏳ Pending |
| README.md | Modify | ⏳ Pending |
| CHANGELOG.md | Modify | ✅ Updated |

---

## Checklist Before Completion

- [ ] All 10 seeds run with gemini-2.5-flash
- [ ] Seed-switching analysis complete
- [ ] MMLU validation complete
- [ ] Baseline comparison re-run
- [ ] Scalability test re-run
- [ ] Paper updated with unified results
- [ ] README updated
- [ ] CHANGELOG updated
- [ ] All changes committed to GitHub
