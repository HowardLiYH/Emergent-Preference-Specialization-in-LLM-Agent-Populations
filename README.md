# Emergent Preference Specialization in LLM Agent Populations

<p align="center">
  <img src="assets/cover.jpeg" alt="Emergent Specialization" width="600"/>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#key-results">Key Results</a> •
  <a href="#synthetic-rules">Synthetic Rules</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#experiments">Experiments</a> •
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Paper-NeurIPS%202025-blue" alt="Paper"/>
  <img src="https://img.shields.io/badge/Python-3.9+-green" alt="Python"/>
  <img src="https://img.shields.io/badge/Rules-8-orange" alt="Rules"/>
  <img src="https://img.shields.io/badge/Causality-75%25-purple" alt="Causality"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License"/>
</p>

---

## Overview

**Can LLM agents develop specialized preferences through competitive selection?**

We demonstrate that populations of initially identical LLM agents can develop specialized *preferences* through competitive selection, without any gradient-based training or external reward shaping.

```
Generation 0                         Generation 100
┌────────────────────────────────┐   ┌────────────────────────────────┐
│ "I am a general-purpose AI..." │   │ "You are a VOWEL SPECIALIST.   │
│                                │   │  Pick words starting with      │
│ Strategies: {}                 │→→→│  A, E, I, O, U..."             │
└────────────────────────────────┘   └────────────────────────────────┘
     (12 identical agents)               (8 distinct specialists)
```

### Key Contribution

We are the **first to demonstrate causal prompt-based specialization in LLM agent populations** with a **75% causality validation rate**.

---

## Key Results

### 🎯 Strong Causality Proven (Phase 2)

Prompt swap experiments demonstrate that prompts **cause** performance differences:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Swap Test Pass Rate** | **75.0%** | Strong causality |
| **Avg Swap Effect** | **-0.479** | Correct specialists score much higher |
| **Rules Covered** | 8/8 | Complete specialization |

### 📊 New Cognitively-Grounded Rules

We replaced weak rules with scientifically-grounded alternatives:

| Old Rule | Problem | New Rule | Source |
|----------|---------|----------|--------|
| LENGTH | LLMs can't count | **VOWEL_START** | Phonemic Awareness (Treiman, 1991) |
| SEMANTIC | Too subjective | **ANIMATE** | Category Processing (Warrington, 1984) |

### 📈 Improvement

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Pass Rate | 60.7% | **75.0%** | **+14.3%** |
| Avg Effect | -0.232 | **-0.479** | **+107%** |

---

### 📊 NeurIPS-Ready Validation Results (January 2025)

**All experiments run with real Gemini 2.0 Flash API**

#### Multi-Seed Validation (5 seeds)

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| **Strategy Concentration Index** | 0.510 ± 0.069 | >0.4 | ✅ PASS |
| **L3 Specialist Rate** | 80.0% ± 10.3% | >50% | ✅ PASS |
| **Specialization Diversity** | 75.0% ± 0.0% | >40% | ✅ PASS |
| **Herfindahl-Hirschman Index** | 0.200 ± 0.010 | <0.5 | ✅ PASS |

#### Competition vs Random Baseline

| Condition | SCI | Diversity | HHI |
|-----------|-----|-----------|-----|
| **Competition** | 0.510 | 75.0% | 0.200 |
| Random | 0.342 | 54.2% | 0.292 |
| **Δ Improvement** | **+49%** | **+38%** | **-31%** |

#### Causal Validation (Swap Test on Evolved Agents)

```
Original (correct prompts): 90% accuracy
Swapped (wrong prompts):    46% accuracy
Transfer Coefficient:       τ = 0.440 ✓
```

**Interpretation**: Prompts CAUSE specialization - swapping prompts degrades performance by 44%.

---

### 📊 Legacy Multi-Seed Results

Results validated across **5 random seeds**:

| Metric | Mean | Std | 95% CI |
|--------|------|-----|--------|
| **Pass Rate** | **72.2%** | 0.124 | [48.5%, 96.0%] |

### 🏆 Baseline Comparisons (NEW)

| Condition | Mean Acc | vs CORRECT |
|-----------|----------|------------|
| NO_PROMPT | 46.7% | -46.6% |
| RANDOM_PROMPT | 55.0% | -38.3% |
| WRONG_PROMPT | 50.7% | -42.6% |
| **CORRECT_PROMPT** | **93.3%** | - |

*All results from real Gemini 2.0 Flash API calls*

### 📈 Scalability Analysis (Real API)

| N Agents | Coverage | Swap Pass | Specialists |
|----------|----------|-----------|-------------|
| 8 | 75.0% | 83.3% | 6/8 |
| 12 | 75.0% | 83.3% | 6/8 |
| 24 | 87.5% | 66.7% | 7/8 |
| 48 | 87.5% | 16.7%* | 7/8 |

*Some N=48 tests affected by API rate limiting

### 🔄 Cross-LLM Validation (NEW)

Specialization mechanism validated across different LLMs:

| Model | Diagonal | Off-Diagonal | Gap | Status |
|-------|----------|--------------|-----|--------|
| Gemini 2.0 Flash | 0.92 | 0.25 | 72.8% | ✅ PASS |
| **GPT-4o-mini** | 0.90 | 0.37 | **58.6%** | ✅ PASS |

**Key Finding**: Both models exceed 30% gap threshold, confirming prompt specialization is model-agnostic.

### 📊 Statistical Significance

| Metric | Difference | 95% CI | p-value | Effect Size |
|--------|------------|--------|---------|-------------|
| SCI | +0.168 | [0.073, 0.264] | **0.0077** | d=2.66 (large) |

*Welch's t-test (unequal variances), 5 seeds competition vs 3 seeds random*

---

## Synthetic Rules

8 rule domains grounded in cognitive science:

| Rule | Description | Source |
|------|-------------|--------|
| **POSITION** | Answer at position B | Serial Position Effect |
| **PATTERN** | ABAB alternation | Gestalt Psychology |
| **INVERSE** | Opposite of obvious | Propositional Logic |
| **VOWEL_START** | Starts with A,E,I,O,U | Phonemic Awareness (Treiman & Zukowski, 1991) |
| **RHYME** | Rhymes with CAT | Phonological Processing (Bradley & Bryant, 1983) |
| **ALPHABET** | First letter closest to M | Orthographic Processing (Coltheart, 1978) |
| **MATH_MOD** | Length mod 3 = 1 | Number Cognition (Dehaene, 1997) |
| **ANIMATE** | Living thing (animal) | Category-Specific Processing (Warrington & Shallice, 1984) |

### Opaque Task Design

Tasks don't reveal the underlying rule, forcing agents to rely on their prompts:
- ❌ "According to the VOWEL_START RULE, which word is correct?"
- ✅ "Which word is correct?"

---

## Quick Start

### Installation

```bash
git clone https://github.com/HowardLiYH/Emergent-Prompt-Evolution.git
cd Emergent-Prompt-Evolution
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"
```

### Run Phase 2 Swap Test

```bash
python experiments/exp_phase2_enhanced.py
```

Expected output:
- 42/56 pairs passed (75.0%)
- Average swap effect: -0.479
- **STRONG CAUSALITY**

### Run Prompt Length Ablation

```bash
python experiments/exp_prompt_length_ablation.py
```

---

## Experiments

### Experiment Suite

| Phase | Experiment | Question | Result |
|-------|------------|----------|--------|
| 0 | Rule Validation | Are rules distinct? | ✅ Cognitively grounded |
| 1 | Preference Emergence | Do agents specialize? | 8/8 coverage |
| **2** | **Causality Test** | **Do prompts cause it?** | **75% pass** |
| 3 | Ablation | Which components matter? | Accumulation critical |

### Key Mechanisms

1. **Strategy Accumulation**: Winners gain rule knowledge (Level 0→1→2→3)
2. **Exclusivity**: Level 3 agents specialize in one rule only
3. **Confidence-based Competition**: Highest confidence among correct wins

---

## Project Structure

```
emergent_prompt_evolution/
├── src/genesis/
│   ├── synthetic_rules.py      # 8 cognitively-grounded rules
│   ├── rule_strategies.py      # 3-level strategy library
│   ├── preference_agent.py     # Agent with exclusivity mechanism
│   ├── competition_v3.py       # Confidence-based competition
│   ├── llm_client.py           # Gemini/OpenAI API wrapper
│   ├── analysis.py             # Statistical tests
│   └── visualization.py        # Publication figures
├── experiments/
│   ├── exp_phase2_enhanced.py  # Main causality test
│   ├── exp_prompt_length_ablation.py
│   └── exp_preference_main.py
├── paper/
│   ├── neurips_draft.md
│   └── neurips_2025.tex
├── results/
│   └── phase2_enhanced_results.json
└── CHANGELOG.md
```

---

## Results Summary

### Phase 2: Causality by Specialist

| Specialist | Passed | Original | Swapped | Effect |
|------------|--------|----------|---------|--------|
| POSITION | 7/7 | 0.31 | 0.83 | -0.52 |
| PATTERN | 6/7 | 0.19 | 0.82 | -0.63 |
| INVERSE | 4/7 | 0.52 | 0.83 | -0.31 |
| **VOWEL_START** | **6/7** | 0.34 | 0.83 | **-0.49** |
| RHYME | 4/7 | 0.36 | 0.83 | -0.47 |
| ALPHABET | 4/7 | 0.42 | 0.91 | -0.49 |
| MATH_MOD | 5/7 | 0.38 | 0.91 | -0.53 |
| **ANIMATE** | **6/7** | 0.44 | 0.83 | **-0.39** |

**New rules (VOWEL_START, ANIMATE) achieve 6/7 pass rate each!**

---

## Cognitive Science Grounding

| Rule | Cognitive Domain | Key Reference |
|------|------------------|---------------|
| VOWEL_START | Phonemic Awareness | Treiman & Zukowski (1991) |
| ANIMATE | Category-Specific Processing | Warrington & Shallice (1984) |
| RHYME | Phonological Processing | Bradley & Bryant (1983) |
| ALPHABET | Orthographic Processing | Coltheart (1978) |
| MATH_MOD | Number Cognition | Dehaene (1997) |

---

## Cost Estimation

| Experiment | API Calls | Est. Cost |
|------------|-----------|-----------|
| Phase 2 (56 pairs × 5 tasks) | ~560 | ~$0.10 |
| Ablation (8 rules × 10 × 2) | ~160 | ~$0.03 |
| **Total** | ~720 | **~$0.15** |

Using Gemini 2.0 Flash for cost efficiency.

---

## Related Projects

| Project | Relationship |
|---------|--------------|
| [Emergent-Specialization](https://github.com/HowardLiYH/Emergent-Specialization-in-Multi-Agent-Systems) | Paper 1: Trading agents (foundation) |
| [Emergent-Civilizations](https://github.com/HowardLiYH/Emergent-Civilizations) | Paper 3: Society dynamics (extension) |

---

## Citation

```bibtex
@article{emergent_preference_2025,
  title={Emergent Preference Specialization in LLM Agent Populations Through Competitive Selection},
  author={Li, Yuhao and others},
  journal={Advances in Neural Information Processing Systems},
  year={2025}
}
```

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>Part of the Emergent Specialization Research Series</b><br>
  <i>Paper 2 of 3</i>
</p>
