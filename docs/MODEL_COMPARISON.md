# Model Comparison: gemini-2.0-flash vs gemini-2.5-flash

**Last Updated**: January 4, 2026 (will be updated when experiments complete)

---

## Overview

This document compares the performance of two Gemini models on our synthetic rule swap test:
- **gemini-2.0-flash-exp**: Previous experiments (mixed with GPT-4o-mini)
- **gemini-2.5-flash**: Current unified experiments

---

## Previous Results (Mixed Models)

### gemini-2.0-flash-exp (Seeds 1-3)

| Seed | Pass Rate |
|------|-----------|
| 1 | 91.7% |
| 2 | 75.0% |
| 3 | 50.0% |
| **Mean** | **72.2%** |
| **Std** | 20.97% |

### GPT-4o-mini (Seeds 4-7)

| Seed | Pass Rate |
|------|-----------|
| 4 | 41.7% |
| 5 | 58.3% |
| 6 | 58.3% |
| 7 | 58.3% |
| **Mean** | **54.2%** |

### Combined 7-Seed (Mixed)
- **Mean**: 61.9%
- **95% CI**: [46.6%, 77.2%]
- **Issue**: Not unified - different models for different seeds

---

## Current Results: gemini-2.5-flash (Unified)

### In Progress (Will be updated)

| Seed | Pass Rate | Status |
|------|-----------|--------|
| 1 | 73.2% | ✅ Done |
| 2 | 71.4% | ✅ Done |
| 3 | TBD | 🔄 In Progress |
| 4 | TBD | ⏳ Pending |
| 5 | TBD | ⏳ Pending |
| 6 | TBD | ⏳ Pending |
| 7 | TBD | ⏳ Pending |
| 8 | TBD | ⏳ Pending |
| 9 | TBD | ⏳ Pending |
| 10 | TBD | ⏳ Pending |

**Mean (so far)**: 72.3% (2 seeds)

---

## Key Differences Between Models

### gemini-2.0-flash
- **Rate limit**: ~15 RPM (slow)
- **Quality**: Good for simple tasks
- **Stability**: Some empty responses
- **Cost**: Free tier

### gemini-2.5-flash
- **Rate limit**: ~1000 RPM (fast!)
- **Quality**: Improved reasoning
- **Stability**: More empty responses (stricter content filter)
- **Cost**: Free tier

---

## Expected Comparison (Once Complete)

| Metric | gemini-2.0 (Seeds 1-3) | gemini-2.5 (Seeds 1-10) | Difference |
|--------|------------------------|-------------------------|------------|
| Mean Pass Rate | 72.2% | TBD | TBD |
| 95% CI Lower | 48.5% | TBD | TBD |
| 95% CI Upper | 96.0% | TBD | TBD |
| CI Width | 47.5% | TBD | TBD |

---

## Checkpoint Implementation

All experiments now have checkpoints:

### Current Unified Experiment
```
results/unified_gemini25/
├── seed_1.json  ✅ (73.2%)
├── seed_2.json  ✅ (71.4%)
├── seed_3.json  🔄 (in progress)
...
└── all_seeds.json  (final combined results)
```

### Phase 1 Main Experiment
```
results/phase1_checkpoints/
├── checkpoint_seed{X}_gen{Y}.json
└── (keeps last 3 per seed)
```

### Experiment Changelog
```
results/experiment_changelog.txt
└── Timestamped log of all experiment events
```

---

## Notes

1. **Why the difference?** gemini-2.5-flash has stricter content filtering, leading to more "empty parts" responses. We handle these gracefully.

2. **Unity benefit**: Having all 10 seeds on the same model eliminates cross-model variance as a confounding factor.

3. **Statistical power**: 10 seeds with same model provides much tighter confidence intervals than 7 seeds with mixed models.

---

## Will be Updated

This document will be updated with final results once all 10 seeds complete (~1 hour remaining).
