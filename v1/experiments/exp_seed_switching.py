#!/usr/bin/env python3
"""
Seed-Switching Analysis

Analyzes whether agents switch from their initial seeded rule (Option B+)
to a different final specialization through competition.

This analysis proves that specialization emerges from competition,
not just from the initial random assignment.

Success Criteria:
- Switch rate > 60%
- Chi-square p-value < 0.05
"""

import sys
sys.path.insert(0, 'src')

import json
from pathlib import Path
from datetime import datetime

from genesis.analysis import analyze_seed_switching, run_seed_switching_analysis


def log(msg: str):
    """Log with timestamp."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main():
    log("=" * 60)
    log("SEED-SWITCHING ANALYSIS")
    log("=" * 60)

    # Try Phase 1 checkpoints first
    log("\nChecking Phase 1 checkpoints...")
    result = run_seed_switching_analysis("results/phase1_checkpoints")

    if "error" in result and "n_seeds" not in result:
        log(f"Phase 1 checkpoints: {result.get('error', 'Not found')}")

        # Try unified gemini25 results
        log("\nChecking unified gemini-2.5-flash results...")
        unified_dir = Path("results/unified_gemini25")

        if unified_dir.exists():
            seed_files = list(unified_dir.glob("seed_*.json"))
            log(f"Found {len(seed_files)} seed files")

            for sf in sorted(seed_files):
                log(f"  - {sf.name}")
        else:
            log("No unified results found yet. Run experiment first.")
            return
    else:
        # Print Phase 1 results
        log(f"\nAnalyzed {result.get('n_seeds', 0)} seeds")
        log(f"Mean switch rate: {result.get('mean_switch_rate', 0):.1%}")
        log(f"Combined chi-square: {result.get('combined_chi_square', 0):.2f}")
        log(f"Combined p-value: {result.get('combined_p_value', 1):.4f}")

        success = result.get('success', False)
        log(f"\nSUCCESS: {'YES ✓' if success else 'NO ✗'}")

        if result.get('individual_results'):
            log("\nPer-seed breakdown:")
            for r in result['individual_results']:
                seed = r.get('seed', 'unknown')
                if 'switch_rate' in r:
                    log(f"  {seed}: {r['switch_rate']:.1%} switch rate (p={r.get('p_value', 1):.3f})")
                else:
                    log(f"  {seed}: {r.get('error', 'unknown error')}")

        # Save results
        output_file = Path("results/seed_switching_analysis.json")
        with open(output_file, 'w') as f:
            # Remove details for cleaner output
            clean_result = {k: v for k, v in result.items() if k != 'individual_results'}
            clean_result['_timestamp'] = datetime.now().isoformat()
            json.dump(clean_result, f, indent=2)
        log(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
