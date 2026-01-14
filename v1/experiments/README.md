# Experiments

This folder contains experiment scripts for Paper 2: Emergent Preference Specialization in LLM Agent Populations.

## Ground Rules

All experiments follow these ground rules:
1. **Checkpoints**: Long experiments save checkpoints for crash recovery
2. **API Keys**: Loaded from `.env` file, never hardcoded
3. **Unified Model**: All experiments use `gemini-2.5-flash` for consistency
4. **Resume Support**: All experiments can resume from last checkpoint

## Experiment Suite

### Core Experiments

| Script | Description | Status |
|--------|-------------|--------|
| `exp_preference_main.py` | Main evolution experiment (Phase 1) | ✅ Ready |
| `exp_phase2_enhanced.py` | Causality test with prompt swap | ✅ Ready |
| `exp_baselines_full.py` | 4-condition baseline comparison | ✅ Ready |
| `exp_multi_seed.py` | Multi-seed aggregation with statistics | ✅ Ready |

### Validation Experiments

| Script | Description | Status |
|--------|-------------|--------|
| `exp_mmlu_validation.py` | Real-world MMLU task validation | ✅ Ready |
| `exp_rule_validation.py` | Rule-specific validation | ✅ Ready |
| `exp_validation.py` | General validation utilities | ✅ Ready |

### Ablation Studies

| Script | Description | Status |
|--------|-------------|--------|
| `exp_ablations.py` | Core ablation studies | ✅ Ready |
| `exp_component_ablation.py` | Component-wise ablation | ✅ Ready |
| `exp_preference_ablation.py` | Preference mechanism ablation | ✅ Ready |
| `exp_prompt_length_ablation.py` | Prompt length effects | ✅ Ready |

### Panel Modification Experiments (NEW)

| Script | Description | Panel Mod |
|--------|-------------|-----------|
| `exp_practical_benefit.py` | 5-condition population vs generalist | #5 |
| `exp_falsification.py` | Preference vs capability test | #4 |
| `exp_cost_benefit.py` | Training ROI analysis | #6 |

### Scalability & Sensitivity

| Script | Description | Status |
|--------|-------------|--------|
| `exp_scalability.py` | Population size scaling (N=8 to N=48) | ✅ Ready |
| `exp_seed_switching.py` | Initial seed persistence analysis | ✅ Ready |

### Cross-Validation

| Script | Description | Status |
|--------|-------------|--------|
| `exp_prompt_swap.py` | Prompt swap causality test | ✅ Ready |
| `exp_preference_swap.py` | Preference swap test | ✅ Ready |
| `exp_domain_overlap.py` | Domain overlap analysis | ✅ Ready |
| `exp_behavioral_fingerprint.py` | Agent behavioral fingerprinting | ✅ Ready |

## Running Experiments

### Quick Start

```bash
# Run practical benefit experiment (5 conditions)
python experiments/exp_practical_benefit.py --n-tasks 100

# Run preference falsification (distinguishes preference from capability)
python experiments/exp_falsification.py --n-tasks 20

# Run cost-benefit analysis (no API calls, pure calculation)
python experiments/exp_cost_benefit.py
```

### Full Suite

```bash
# Run main evolution experiment
python experiments/exp_preference_main.py --seed 42 --generations 100

# Run multi-seed validation (10 seeds)
python experiments/exp_multi_seed.py --n-seeds 10

# Run MMLU real-world validation
python experiments/exp_mmlu_validation.py --n-questions 100
```

### Resuming from Checkpoint

All long-running experiments automatically save checkpoints:

```bash
# Resume experiment (default behavior)
python experiments/exp_practical_benefit.py

# Start fresh (ignore checkpoint)
python experiments/exp_practical_benefit.py --no-resume
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
GEMINI_API_KEY=your_key_here
# Optional alternatives:
# OPENAI_API_KEY=your_key_here
# ANTHROPIC_API_KEY=your_key_here
```

### Default Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n-tasks` | 100 | Tasks per condition |
| `--seed` | 42 | Random seed |
| `--n-agents` | 12 | Population size |
| `--generations` | 100 | Evolution generations |

## Results

Results are saved to the `results/` directory:

```
results/
├── practical_benefit/
│   ├── checkpoint.json
│   └── practical_benefit_results.json
├── falsification/
│   └── falsification_results.json
├── cost_benefit/
│   └── cost_benefit_analysis.json
└── ...
```

## Adding New Experiments

New experiments should follow this template:

```python
"""
Experiment Description

Ground Rules Applied:
- Checkpoints every N tasks
- API keys from .env
- Unified model usage
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.genesis.llm_client import LLMClient
# ... other imports

CHECKPOINT_DIR = Path("results/experiment_name")
CHECKPOINT_FILE = CHECKPOINT_DIR / "checkpoint.json"

def save_checkpoint(data):
    """Save checkpoint for crash recovery."""
    # Implementation

def load_checkpoint():
    """Load checkpoint if exists."""
    # Implementation

async def run_experiment():
    """Main experiment logic with checkpoint support."""
    # Implementation

if __name__ == "__main__":
    asyncio.run(run_experiment())
```
