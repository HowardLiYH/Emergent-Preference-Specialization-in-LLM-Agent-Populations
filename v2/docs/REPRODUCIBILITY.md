# Reproducibility Checklist

This document ensures all experiments in Emergent Prompt Evolution v2 are fully reproducible.

## Environment

### Hardware
- CPU: Any modern x86_64 (tested on Apple M1/M2)
- RAM: 8GB minimum
- GPU: Not required (API-based LLM)

### Software
- Python 3.10+
- See `requirements.txt` for package versions

### API Access
- Google AI API key (Gemini 2.5 Flash)
- Rate limit: 60 requests/minute

## Setup

```bash
# Clone repository
git clone https://github.com/HowardLiYH/Emergent-Prompt-Evolution
cd Emergent-Prompt-Evolution

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Configure API key (NEVER commit this file!)
cp env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

## Random Seeds

All experiments use seeds 0-9 (10 total). Results are averaged with standard deviations.

```python
EXPERIMENT_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Population size | 12 | Number of agents |
| Generations | 100 | Competition rounds |
| Fitness sharing γ | 0.5 | From Theorem 1 |
| Temperature | 0.7 | LLM sampling |
| Model | gemini-2.5-flash | Primary model |
| Memory compaction | 100 entries | Trigger threshold |
| Memory retrieval | top-5 | Hybrid scoring |
| Thompson prior | Beta(1,1) | Uninformative |

## Running Experiments

### Main Experiment (Single Seed)
```bash
cd v2/experiments/main
python run_experiment.py --seed 0 --n-generations 100
```

### All Seeds (Full Replication)
```bash
python run_experiment.py --seed 0 &
python run_experiment.py --seed 1 &
# ... run all 10 seeds
```

### Baselines
```bash
cd v2/experiments/baselines
python -m run_all_baselines --n-runs 10
```

### Ablations
```bash
cd v2/experiments/ablations
python run_ablations.py --n-runs 10
```

### Benchmark
```bash
cd v2/src/benchmark
python -m runner --full
```

## Expected Results

| Metric | Mean | Std | 95% CI |
|--------|------|-----|--------|
| SCI | 0.78 | 0.05 | [0.75, 0.81] |
| Coverage | 100% | 0% | -- |
| Equilibrium Error | <10% | -- | -- |
| Token Savings vs UCB1 | 55% | 8% | [50%, 60%] |

## Expected Runtime

| Component | Time | Notes |
|-----------|------|-------|
| Single seed | ~2 hours | 100 generations |
| All 10 seeds | ~20 hours | Can parallelize |
| Baselines | ~10 hours | 4 methods × 10 runs |
| Ablations | ~15 hours | 5 ablations × 10 runs |
| Benchmark | ~1 hour | L0 baseline eval |

## File Outputs

```
v2/results/
├── experiment_seed0_YYYYMMDD_HHMMSS/
│   ├── results.json          # Metrics
│   ├── population.json       # Agent states
│   └── figures/              # Generated plots
├── aggregate_results.json    # Combined stats
├── ablation_summary.json     # Ablation results
└── benchmark/                # ESB results
```

## Verification

To verify results match reported values:

1. Run all 10 seeds
2. Check `aggregate_results.json`:
   - `mean_sci` should be 0.75-0.81
   - `mean_coverage` should be 1.0
3. Run statistical tests:
   - Token savings: t-test p < 0.05
   - Effect size: Cohen's d > 0.8

## Troubleshooting

### API Rate Limits
If you hit rate limits:
```python
# In llm_client.py, increase retry_delay
self.retry_delay = 2.0  # Default is 1.0
```

### Memory Issues
If running out of memory:
```python
# Reduce batch size in batch_processor.py
self.max_concurrent = 2  # Default is 5
```

### Missing Dependencies
```bash
pip install --upgrade -r requirements.txt
```

## Contact

For reproducibility issues, please open a GitHub issue with:
1. Python version
2. Package versions (`pip freeze`)
3. Full error traceback
4. Steps to reproduce
