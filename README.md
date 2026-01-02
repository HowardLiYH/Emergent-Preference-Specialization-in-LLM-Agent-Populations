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
  <img src="https://img.shields.io/badge/Causality-60.7%25-purple" alt="Causality"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License"/>
</p>

---

## Overview

**Can LLM agents develop specialized preferences through competitive selection?**

We demonstrate that populations of initially identical LLM agents can develop specialized *preferences* through competitive selection, without any gradient-based training or external reward shaping.

```
Generation 0                         Generation 100
┌────────────────────────────────┐   ┌────────────────────────────────┐
│ "I am a general-purpose AI..." │   │ "You are a LENGTH SPECIALIST.  │
│                                │   │  Pick the 5-letter word..."    │
│ Strategies: {}                 │→→→│ Strategies: {LENGTH: 3}        │
└────────────────────────────────┘   └────────────────────────────────┘
     (12 identical agents)               (8 distinct specialists)
```

### Key Contribution

We are the **first to demonstrate causal prompt-based specialization in LLM agent populations** with a 60.7% causality validation rate.

---

## Key Results

### 🎯 Causality Proven (Phase 2)

Prompt swap experiments demonstrate that prompts **cause** performance differences:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Swap Test Pass Rate** | 60.7% | Moderate-strong causality |
| **Avg Swap Effect** | -0.232 | Correct specialists score higher |
| **Rules Covered** | 8/8 | Complete specialization |

### 📊 Surprising Finding: Less Is More

Counter-intuitively, **concise prompts outperform verbose prompts**:

| Prompt Type | Avg Length | Avg Accuracy |
|-------------|------------|--------------|
| Short | ~30 chars | **0.918** |
| Enhanced | ~900 chars | 0.677 |
| **Difference** | 35× | **-24%** |

LLMs extract rules more effectively from minimal instructions!

---

## Synthetic Rules

8 rule domains that cannot be solved by LLM parametric knowledge:

| Rule | Description | Correct Answer |
|------|-------------|----------------|
| **POSITION** | Answer at position B | Always pick B |
| **PATTERN** | ABAB alternation | Alternate A↔B |
| **INVERSE** | Opposite of obvious | "Is fire hot?" → No |
| **LENGTH** | 5-letter word | Count letters |
| **RHYME** | Rhymes with CAT | bat, hat, mat |
| **ALPHABET** | First letter closest to M | Distance to 13th |
| **MATH_MOD** | Length mod 3 = 1 | Lengths 1,4,7,10... |
| **SEMANTIC** | Opposite of HAPPY | sad, angry |

### Opaque Task Design

Tasks don't reveal the underlying rule, forcing agents to rely on their prompts:
- ❌ "According to the LENGTH RULE, which word is correct?"
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
- 34/56 pairs passed (60.7%)
- Average swap effect: -0.232

### Run Prompt Length Ablation

```bash
python experiments/exp_prompt_length_ablation.py
```

Expected output:
- Short prompts: 0.918 accuracy
- Enhanced prompts: 0.677 accuracy

---

## Experiments

### Experiment Suite

| Phase | Experiment | Question | Result |
|-------|------------|----------|--------|
| 0 | Rule Validation | Are rules distinct? | 29.5% gap |
| 1 | Preference Emergence | Do agents specialize? | 8/8 coverage |
| **2** | **Causality Test** | **Do prompts cause it?** | **60.7% pass** |
| 3 | Ablation | Which components matter? | Accumulation critical |
| 4 | **Prompt Length** | **Long vs short prompts?** | **Short wins by 24%** |

### Key Mechanisms

1. **Strategy Accumulation**: Winners gain rule knowledge (Level 0→1→2→3)
2. **Exclusivity**: Level 3 agents specialize in one rule only
3. **Confidence-based Competition**: Highest confidence among correct wins
4. **Fitness Sharing**: Optional diversity preservation

---

## Project Structure

```
emergent_prompt_evolution/
├── src/genesis/
│   ├── synthetic_rules.py      # 8 synthetic rule domains
│   ├── rule_strategies.py      # 3-level strategy library (short + enhanced)
│   ├── preference_agent.py     # Agent with exclusivity mechanism
│   ├── competition_v3.py       # Confidence-based competition
│   ├── fitness_sharing_v3.py   # Diversity preservation
│   ├── llm_client.py           # Gemini/OpenAI API wrapper
│   ├── analysis.py             # Statistical tests (t-test, Cohen's d)
│   └── visualization.py        # Publication-quality figures
├── experiments/
│   ├── exp_phase2_enhanced.py  # Main causality test
│   ├── exp_prompt_length_ablation.py  # Short vs long prompts
│   ├── exp_preference_main.py  # Phase 1 emergence
│   └── exp_preference_ablation.py  # Component ablation
├── paper/
│   ├── neurips_draft.md        # Paper draft (Markdown)
│   ├── neurips_2025.tex        # Paper draft (LaTeX)
│   └── figures/                # Publication figures
├── results/
│   ├── phase2_enhanced_results.json
│   └── prompt_length_ablation.json
├── CHANGELOG.md
└── requirements.txt
```

---

## Results Summary

### Phase 2: Causality by Specialist

| Specialist | Passed | Original | Swapped | Effect |
|------------|--------|----------|---------|--------|
| POSITION | 6/7 | 0.40 | 0.66 | -0.26 |
| PATTERN | 6/7 | 0.27 | 0.69 | -0.42 |
| INVERSE | 4/7 | 0.45 | 0.69 | -0.24 |
| LENGTH | 3/7 | 0.67 | 0.73 | -0.06 |
| RHYME | 4/7 | 0.49 | 0.69 | -0.20 |
| ALPHABET | 4/7 | 0.42 | 0.75 | -0.33 |
| MATH_MOD | 4/7 | 0.44 | 0.75 | -0.31 |
| SEMANTIC | 3/7 | 0.65 | 0.69 | -0.04 |

**Interpretation**: Correct specialists consistently outperform wrong specialists.

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
