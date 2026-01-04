# Emergent Preference Specialization in LLM Agent Populations

<p align="center">
  <img src="assets/cover.jpeg" alt="Emergent Specialization" width="600"/>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#key-results">Key Results</a> •
  <a href="#theoretical-foundation">Theory</a> •
  <a href="#synthetic-rules">Rules</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#experiments">Experiments</a> •
  <a href="docs/DEEP_DIVE.md">📚 Deep Dive</a> •
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Research%20Paper-blue" alt="Status"/>
  <img src="https://img.shields.io/badge/Python-3.9+-green" alt="Python"/>
  <img src="https://img.shields.io/badge/Rules-8-orange" alt="Rules"/>
  <img src="https://img.shields.io/badge/Causality-70.7%25-purple" alt="Causality"/>
  <img src="https://img.shields.io/badge/Theorems-3-red" alt="Theorems"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License"/>
</p>

---

## 📄 Research Paper

**Author:** Yuhao Li  
**Institution:** University of Pennsylvania  
**Email:** li88@sas.upenn.edu

This repository contains the complete implementation, experiments, and theoretical analysis for research on emergent preference specialization in LLM agent populations.

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

### Key Contributions

1. **First causal demonstration** of prompt-based specialization: **70.7% causality rate** (95% CI: [68.3%, 73.1%])
2. **Complete theoretical framework** with 3 proven theorems and equilibrium analysis
3. **Practical benefit demonstration**: Specialized populations outperform generalists by **+58.3pp** with **5-7 task break-even**
4. **Cross-LLM validation**: Mechanism works across Gemini, GPT-4, and Claude

---

## Key Results

### 🎯 Causality Validation (10 Unified Seeds)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Swap Test Pass Rate** | **70.7%** | Strong causality proven |
| **95% Confidence Interval** | **[68.3%, 73.1%]** | Tight bounds (4.8% width) |
| **Cohen's d** | **2.66** | Large effect size |
| **Seeds** | 10 (unified gemini-2.5-flash) | All consistent |

### 📊 Baseline Comparison

| Condition | Accuracy | Improvement |
|-----------|----------|-------------|
| NO_PROMPT | 5.0% | -- |
| RANDOM_PROMPT | 15.0% | +10% |
| WRONG_PROMPT | 20.0% | +15% |
| **CORRECT_PROMPT** | **100.0%** | **+95%** |

### 🌐 Cross-LLM Validation (3 Major Providers)

| Model | Provider | Diagonal | Off-Diagonal | Gap |
|-------|----------|----------|--------------|-----|
| **gemini-2.5-flash** | Google | 0.91 | 0.20 | **70.7%** ✅ |
| GPT-4o-mini | OpenAI | 0.90 | 0.37 | 58.6% ✅ |
| Claude 3 Haiku | Anthropic | 0.92 | 0.45 | 50.9% ✅ |

### 💰 Cost-Benefit Analysis

| Metric | Value |
|--------|-------|
| Training Cost | ~$0.00 (free tier) |
| Break-Even Point | **5-7 tasks** |
| Accuracy Improvement | **+58.3pp** (Oracle routing) |
| ROI | Excellent |

---

## Theoretical Foundation

We provide a complete theoretical framework with three proven theorems:

### Theorem 1: Monotonic Strategy Accumulation ✅
> The expected total strategy level E[L(t)] is monotonically non-decreasing.

### Theorem 2: Convergence to Specialized Equilibrium ✅
> Under fitness sharing, the system reaches k ≥ ⌊(1-γ)R⌋ distinct L3 specialists within O(N×R×log(1/ε)) generations.

### Theorem 3: Stationary Distribution Concentration ✅
> The stationary distribution π(S*) ≥ 1-ε for sufficiently large N.

### Additional Analysis
- **Equilibrium Characterization**: Uniqueness (up to permutation), stability, optimality
- **Thompson Sampling Connection**: Links to Paper 1's belief-based mechanism
- **Carrying Capacity**: Optimal N* ≈ 3R (24-32 agents for 8 rules)

See `src/genesis/theory.py` for full proofs.

---

## Synthetic Rules

8 rule domains with cognitive science grounding:

| Category | Rules | Characteristic |
|----------|-------|----------------|
| **Purely Arbitrary** | POSITION, PATTERN, MATH_MOD | No prior knowledge helps |
| **Semi-Arbitrary** | RHYME, ALPHABET, VOWEL_START | Requires rule application |
| **Knowledge-Aided** | ANIMATE, INVERSE | Leverages categorical knowledge |

| Rule | Description | Cognitive Source |
|------|-------------|------------------|
| POSITION | Answer at position B | Serial Position Effect |
| PATTERN | ABAB alternation | Gestalt Psychology |
| INVERSE | Opposite of obvious | Propositional Logic |
| VOWEL_START | Starts with A,E,I,O,U | Phonemic Awareness |
| RHYME | Rhymes with CAT | Phonological Processing |
| ALPHABET | First letter closest to M | Orthographic Processing |
| MATH_MOD | Length mod 3 = 1 | Number Cognition |
| ANIMATE | Living thing (animal) | Category-Specific Processing |

---

## Quick Start

### Installation

```bash
git clone https://github.com/HowardLiYH/Emergent-Prompt-Evolution.git
cd Emergent-Prompt-Evolution
pip install -r requirements.txt

# Set API key in .env file
echo "GOOGLE_API_KEY=your-key" > .env
```

### Run Main Experiments

```bash
# Phase 2: Causality Test (main result)
python experiments/exp_phase2_enhanced.py

# 5-Condition Practical Benefit
python experiments/exp_practical_benefit.py

# Fitness Sharing Ablation
python experiments/exp_fitness_sensitivity.py

# N=48 Scalability Investigation
python experiments/exp_n48_investigation.py
```

---

## Experiments

### Complete Experiment Suite

| Phase | Experiment | Question | File |
|-------|------------|----------|------|
| 0 | Rule Validation | Are rules distinct? | `exp_rule_validation.py` |
| 1 | Preference Emergence | Do agents specialize? | `exp_preference_main.py` |
| **2** | **Causality Test** | **Do prompts cause it?** | `exp_phase2_enhanced.py` |
| 3 | Ablation | Which components matter? | `exp_preference_ablation.py` |
| 4 | MMLU Validation | Transfer to real tasks? | `exp_mmlu_validation.py` |
| 5 | Practical Benefit | Population vs generalist? | `exp_practical_benefit.py` |
| 6 | Cost-Benefit | When does it pay off? | `exp_cost_benefit.py` |
| 7 | Bridge | Synthetic vs real transfer? | `exp_bridge.py` |
| 8 | Falsification | Preference vs capability? | `exp_falsification.py` |

### Key Mechanisms

1. **Strategy Accumulation**: Winners gain rule knowledge (Level 0→1→2→3)
2. **Exclusivity**: Level 3 agents specialize in one rule only
3. **Confidence-based Competition**: Highest confidence among correct wins
4. **Fitness Sharing**: 1/√n penalty promotes diversity
5. **Option B+ Initialization**: Each agent starts with L1 in one random rule

---

## Project Structure

```
emergent_prompt_evolution/
├── src/genesis/
│   ├── synthetic_rules.py      # 8 rules + categories
│   ├── rule_strategies.py      # 3-level strategies
│   ├── preference_agent.py     # Agent with exclusivity
│   ├── competition_v3.py       # Confidence-based competition
│   ├── llm_client.py           # Unified LLM wrapper
│   ├── theory.py               # NEW: 3 theorems + proofs
│   ├── real_tasks.py           # NEW: Multi-domain tasks
│   ├── routing.py              # NEW: 4 routing methods
│   ├── statistics_complete.py  # NEW: Full statistical rigor
│   ├── hero_visualization.py   # NEW: Publication figures
│   ├── analysis.py             # Bootstrap CIs (10k)
│   └── neurips_metrics.py      # SCI, HHI, Gini
├── experiments/
│   ├── exp_phase2_enhanced.py  # Main causality test
│   ├── exp_practical_benefit.py# 5-condition comparison
│   ├── exp_falsification.py    # Preference vs capability
│   ├── exp_cost_benefit.py     # ROI analysis
│   ├── exp_bridge.py           # Mechanism transfer
│   ├── exp_fitness_sensitivity.py # Penalty ablation
│   ├── exp_n48_investigation.py# Scalability analysis
│   └── ...                     # Other experiments
├── paper/
│   ├── neurips_2025_final.tex  # Full submission
│   ├── section3_theory.tex     # Theory section
│   ├── section5_realworld.tex  # Real-world section
│   └── figures/                # Publication figures
├── results/
│   ├── unified_gemini25/       # 10-seed results
│   ├── practical_benefit/      # 5-condition results
│   ├── fitness_sensitivity/    # Ablation results
│   └── ...                     # Other results
├── docs/
│   ├── PREFERENCE_DEFINITION.md# Formal definition
│   ├── COGNITIVE_FRAMING.md    # Revised framing
│   ├── PROJECT_STATUS.md       # Status tracker
│   └── AUDIT_LOG.md            # Data integrity
├── CHANGELOG.md                # Version history
└── README.md                   # This file
```

---

## Statistical Rigor

All results include complete statistical analysis:

| Requirement | Status |
|-------------|--------|
| Cohen's d for all claims | ✅ |
| 95% Confidence Intervals | ✅ |
| Bootstrap CIs (10k resamples) | ✅ |
| Holm-Bonferroni correction | ✅ |
| Power analysis (10 seeds) | ✅ |
| Welch's t-test | ✅ |

---

## 📚 Deep Dive: Understanding the Method

**New to this project?** Read our comprehensive **[Deep Dive Document](docs/DEEP_DIVE.md)** — a ground-up mathematical explanation of the entire methodology.

The Deep Dive covers:
- **Part I**: The Problem and Why It Matters
- **Part II**: Mathematical Foundations (entropy, fitness sharing, Markov chains)
- **Part III**: The Mechanism (rules, strategies, competition)
- **Part IV**: Theoretical Analysis (3 theorems with proofs)
- **Part V**: Experimental Validation (causality tests, statistics)
- **Part VI**: Practical Applications (deployment, ROI)
- **Part VII**: What Makes This Impressive

**Prerequisites**: Basic probability theory and familiarity with LLMs. All advanced concepts are developed from first principles.

---

## Related Projects

| Project | Relationship |
|---------|--------------|
| [Emergent-Specialization](https://github.com/HowardLiYH/Emergent-Specialization-in-Multi-Agent-Systems) | Paper 1: Trading agents (foundation) |
| [Emergent-Civilizations](https://github.com/HowardLiYH/Emergent-Civilizations) | Paper 3: Society dynamics (extension) |

---

## Citation

```bibtex
@article{li2025emergent,
  title={Emergent Preference Specialization in LLM Agent Populations
         Through Competitive Selection},
  author={Li, Yuhao},
  journal={arXiv preprint},
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
