# Experiments

This folder contains experiment scripts for Paper 2: Emergent Prompt Specialization.

## Experiment Suite

| Script | Description |
|--------|-------------|
| `exp_baseline.py` | Baseline specialization (16 agents, 100 gen) |
| `exp_ablations.py` | No evolution, random evolution, no competition |
| `exp_temperature.py` | Temperature ablation study |
| `exp_counterfactual.py` | Prompt swap test for causality |
| `exp_cross_llm.py` | Replication with GPT-4, Claude, GPT-3.5 |
| `exp_scale.py` | Scale to 100 agents |

## Running Experiments

```bash
# Run baseline experiment
python experiments/exp_baseline.py --output results/baseline/

# Run all ablations
python experiments/exp_ablations.py --output results/ablations/

# Run counterfactual test
python experiments/exp_counterfactual.py --output results/counterfactual/
```

## Configuration

See `.cursor/plans/paper2_prompt_evolution.plan.md` for detailed experiment configurations.
