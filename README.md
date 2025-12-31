# Emergent Prompt Specialization in LLM Agent Populations

<p align="center">
  <img src="assets/cover.jpeg" alt="Emergent Specialization" width="600"/>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#key-innovation">Key Innovation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#experiments">Experiments</a> •
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Paper-NeurIPS%202026-blue" alt="Paper"/>
  <img src="https://img.shields.io/badge/Python-3.9+-green" alt="Python"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License"/>
</p>

---

## Overview

**Can LLM agents develop specialized roles through competition alone?**

This project demonstrates that populations of LLM agents, when placed in competitive environments, naturally evolve specialized system prompts—without explicit reward shaping or role assignment.

```
Generation 0                    Generation 100
┌────────────────────┐          ┌────────────────────┐
│ "I am a general-   │          │ "I am an expert in │
│  purpose AI..."    │   ───→   │  mathematical      │
│                    │          │  reasoning..."     │
└────────────────────┘          └────────────────────┘
     (8 identical)                  (8 specialists)
```

### The Core Mechanism

```
┌─────────────────────────────────────────────────────────┐
│                    COMPETITION ROUND                     │
├─────────────────────────────────────────────────────────┤
│  1. All agents attempt a task (math/coding/logic/lang)  │
│  2. Score each response objectively                     │
│  3. WINNER takes all → Only winner evolves prompt       │
│  4. Winner's prompt updated via LLM self-reflection     │
│  5. Repeat for 100+ generations                         │
└─────────────────────────────────────────────────────────┘
```

---

## Key Innovation

### Winner-Take-All Prompt Evolution

Unlike traditional fine-tuning or RLHF, agents **rewrite their own system prompts** based on competitive success:

```python
# After winning a math task...
evolution_prompt = """
You just SUCCEEDED at a math task (score: 0.95).

Current role: "I am a general-purpose AI assistant..."

Update your role to reinforce your math expertise.
"""

# Agent evolves its own prompt:
new_prompt = "I am an expert in mathematical reasoning. 
              I excel at algebraic manipulation, word problems,
              and systematic problem decomposition..."
```

### Multi-Layer Specialization Metrics

| Layer | Metric | What It Measures |
|-------|--------|------------------|
| 1 | Task Performance | Objective accuracy |
| 2a | **LSI** (LLM Specialization Index) | Performance concentration |
| 2b | Semantic Clustering | Prompt embedding distances |
| 2c | Behavioral Fingerprint | Self-reported expertise |
| 3 | **Prompt Swap Test** | Causal validation |

---

## Quick Start

### Installation

```bash
git clone https://github.com/HowardLiYH/Emergent-Prompt-Evolution.git
cd Emergent-Prompt-Evolution
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
```

### Run Baseline Experiment

```python
from src.genesis import GenesisSimulation, SimulationConfig

config = SimulationConfig(
    n_agents=16,
    n_generations=100,
    evolution_type="directed",
    winner_takes_all=True
)

sim = GenesisSimulation(config, llm_client)
results = await sim.run()

print(f"Final LSI: {results.final_metrics['lsi']['mean']:.3f}")
# Expected: LSI > 0.6 (significant specialization)
```

---

## Experiments

### Experiment Suite

| # | Experiment | Question | Expected Result |
|---|------------|----------|-----------------|
| 1 | Baseline Specialization | Do agents specialize? | LSI > 0.6 |
| 2 | No Evolution | Is evolution necessary? | LSI ≈ 0.2 |
| 3 | Random Evolution | Is *directed* evolution necessary? | LSI ≈ 0.4 |
| 4 | No Competition | Is winner-take-all necessary? | LSI < 0.4 |
| 5 | Temperature Ablation | How sensitive to LLM temp? | Consistent |
| 6 | **Prompt Swap** | Does performance follow prompts? | Transfer > 0.5 |
| 7 | Cross-LLM | Do results replicate? | Same pattern |
| 8 | Scale (100 agents) | Does it scale? | Same pattern |

### Run All Experiments

```bash
python experiments/run_all.py --output results/
```

---

## Project Structure

```
emergent_prompt_evolution/
├── src/genesis/
│   ├── agent.py           # GenesisAgent with evolvable prompt
│   ├── tasks.py           # Task pool (math, coding, logic, language)
│   ├── competition.py     # Winner-take-all engine
│   ├── evolution.py       # Directed/random prompt evolution
│   ├── metrics.py         # LSI, semantic, behavioral metrics
│   ├── counterfactual.py  # Prompt swap test
│   └── simulation.py      # Main orchestrator
├── experiments/
│   ├── exp_genesis_baseline.py
│   ├── exp_genesis_ablations.py
│   └── exp_genesis_counterfactual.py
├── results/
└── paper/
```

---

## Theoretical Foundation

This work bridges:

| Field | Connection |
|-------|------------|
| **Evolutionary Biology** | Niche partitioning, competitive exclusion |
| **Game Theory** | Winner-take-all dynamics, Nash equilibrium |
| **Multi-Agent RL** | Emergent specialization without central control |
| **LLM Agents** | Self-modifying system prompts |

### Key Insight

> Specialization emerges not because we *reward* it, but because it's the **stable equilibrium** under competition. Generalists lose to specialists in their domain.

---

## Related Projects

| Project | Relationship |
|---------|--------------|
| [Emergent-Specialization](https://github.com/HowardLiYH/Emergent-Specialization-in-Multi-Agent-Systems) | Paper 1: Rule-based agents (foundation) |
| [Emergent-Civilizations](https://github.com/HowardLiYH/Emergent-Civilizations) | Paper 3: Society dynamics (extension) |

---

## Citation

```bibtex
@article{emergent_prompt_evolution_2026,
  title={Emergent Specialization in LLM Agent Populations},
  author={Li, Yuhao and others},
  journal={Advances in Neural Information Processing Systems},
  year={2026}
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
