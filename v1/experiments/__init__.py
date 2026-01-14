"""
Experiments module: NeurIPS v2 experiments.

Main entry point:
    python experiments/run_neurips_v2.py --gemini-key YOUR_KEY --scale small

Individual experiments:
    - exp_domain_overlap.py - Validate domain orthogonality
    - exp_component_ablation.py - Reasoning vs few-shot ablation
    - exp_baselines.py - All baselines including negative controls
    - exp_prompt_swap.py - Bidirectional causality test
    - run_neurips_v2.py - Master experiment pipeline
"""

__all__ = []
