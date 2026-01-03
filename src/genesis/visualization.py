"""
Visualization Module for Emergent Prompt Specialization

Generates publication-quality figures for NeurIPS:
- Specialization trajectory plots
- Preference heatmaps
- Swap test comparison charts
- Ablation bar charts
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Plotting disabled.")


# NeurIPS-friendly color palette
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#28A745',
    'warning': '#FFC107',
    'danger': '#DC3545',
    'neutral': '#6C757D',
    'light': '#F8F9FA',
    'dark': '#343A40'
}

RULE_COLORS = {
    'position': '#E63946',
    'pattern': '#F4A261',
    'inverse': '#2A9D8F',
    'vowel_start': '#264653',
    'rhyme': '#E9C46A',
    'alphabet': '#457B9D',
    'math_mod': '#8338EC',
    'animate': '#FF006E'
}


def setup_neurips_style():
    """Configure matplotlib for NeurIPS-quality figures."""
    if not HAS_MATPLOTLIB:
        return

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.figsize': (6, 4),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def plot_rpi_trajectories(
    rpi_histories: Dict[str, List[float]],
    title: str = "Rule Preference Intensity Over Time",
    save_path: Optional[str] = None
):
    """
    Plot RPI trajectories for each rule over generations.

    Args:
        rpi_histories: Dict mapping rule name to list of RPI values per generation
        title: Plot title
        save_path: Path to save figure
    """
    if not HAS_MATPLOTLIB:
        print("Cannot plot: matplotlib not installed")
        return

    setup_neurips_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for rule, history in rpi_histories.items():
        color = RULE_COLORS.get(rule, COLORS['neutral'])
        generations = range(len(history))
        ax.plot(generations, history, label=rule.upper(), color=color, linewidth=2)

    ax.set_xlabel('Generation')
    ax.set_ylabel('Rule Preference Intensity (RPI)')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    ax.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_preference_heatmap(
    preferences: Dict[str, Dict[str, float]],
    title: str = "Agent Preference Distribution",
    save_path: Optional[str] = None
):
    """
    Plot heatmap of agent preferences across rules.

    Args:
        preferences: Dict[agent_id, Dict[rule, preference_score]]
        title: Plot title
        save_path: Path to save figure
    """
    if not HAS_MATPLOTLIB:
        print("Cannot plot: matplotlib not installed")
        return

    setup_neurips_style()

    agents = list(preferences.keys())
    rules = list(next(iter(preferences.values())).keys()) if preferences else []

    # Build matrix
    matrix = []
    for agent in agents:
        row = [preferences[agent].get(rule, 0) for rule in rules]
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Custom colormap
    cmap = LinearSegmentedColormap.from_list(
        'preference', ['#F8F9FA', '#2E86AB', '#1A365D']
    )

    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(rules)))
    ax.set_xticklabels([r.upper() for r in rules], rotation=45, ha='right')
    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels(agents)

    ax.set_xlabel('Rule Domain')
    ax.set_ylabel('Agent')
    ax.set_title(title)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Preference Score')

    # Add text annotations for high values
    for i in range(len(agents)):
        for j in range(len(rules)):
            value = matrix[i, j]
            if value > 0.5:
                ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                       color='white', fontsize=8, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_swap_test_results(
    results: Dict[str, Dict],
    title: str = "Prompt Swap Test Results",
    save_path: Optional[str] = None
):
    """
    Plot bar chart comparing original vs swapped prompt performance.

    Args:
        results: Dict with swap test results (from exp_phase2_enhanced.py)
        title: Plot title
        save_path: Path to save figure
    """
    if not HAS_MATPLOTLIB:
        print("Cannot plot: matplotlib not installed")
        return

    setup_neurips_style()

    # Group by test rule
    rules = set()
    for key in results.keys():
        parts = key.split('→')
        if len(parts) == 2:
            rules.add(parts[1])

    rules = sorted(list(rules))

    original_means = []
    swapped_means = []

    for rule in rules:
        orig_scores = []
        swap_scores = []
        for key, data in results.items():
            if key.endswith(f'→{rule}'):
                orig_scores.append(data['original_score'])
                swap_scores.append(data['swapped_score'])

        original_means.append(sum(orig_scores) / len(orig_scores) if orig_scores else 0)
        swapped_means.append(sum(swap_scores) / len(swap_scores) if swap_scores else 0)

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(rules))
    width = 0.35

    bars1 = ax.bar(x - width/2, original_means, width, label='Wrong Specialist',
                   color=COLORS['danger'], alpha=0.8)
    bars2 = ax.bar(x + width/2, swapped_means, width, label='Correct Specialist',
                   color=COLORS['success'], alpha=0.8)

    ax.set_xlabel('Test Rule Domain')
    ax.set_ylabel('Average Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([r.upper() for r in rules], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=8)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_ablation_comparison(
    conditions: List[str],
    scores: List[float],
    errors: Optional[List[float]] = None,
    title: str = "Ablation Study Results",
    ylabel: str = "Score",
    save_path: Optional[str] = None
):
    """
    Plot bar chart for ablation study.

    Args:
        conditions: List of condition names
        scores: List of scores for each condition
        errors: Optional list of error bars
        title: Plot title
        ylabel: Y-axis label
        save_path: Path to save figure
    """
    if not HAS_MATPLOTLIB:
        print("Cannot plot: matplotlib not installed")
        return

    setup_neurips_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(conditions))
    colors = [COLORS['primary'] if i == 0 else COLORS['neutral'] for i in range(len(conditions))]

    if errors:
        bars = ax.bar(x, scores, yerr=errors, capsize=5, color=colors, edgecolor='black', linewidth=1)
    else:
        bars = ax.bar(x, scores, color=colors, edgecolor='black', linewidth=1)

    ax.set_xlabel('Condition')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right')

    # Add value labels
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.annotate(f'{score:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_prompt_length_ablation(
    results_path: str,
    save_path: Optional[str] = None
):
    """
    Plot prompt length ablation results from JSON file.
    """
    if not HAS_MATPLOTLIB:
        print("Cannot plot: matplotlib not installed")
        return

    with open(results_path, 'r') as f:
        results = json.load(f)

    setup_neurips_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    rules = [r['rule'] for r in results]
    short_scores = [r['short_prompt_score'] for r in results]
    enhanced_scores = [r['enhanced_prompt_score'] for r in results]

    x = np.arange(len(rules))
    width = 0.35

    bars1 = ax.bar(x - width/2, short_scores, width, label='Short (~30 chars)',
                   color=COLORS['success'], alpha=0.8)
    bars2 = ax.bar(x + width/2, enhanced_scores, width, label='Enhanced (~900 chars)',
                   color=COLORS['primary'], alpha=0.8)

    ax.set_xlabel('Rule Domain')
    ax.set_ylabel('Accuracy')
    ax.set_title('Prompt Length Ablation: Short vs Enhanced Prompts')
    ax.set_xticks(x)
    ax.set_xticklabels([r.upper() for r in rules], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_orthogonality_matrix(
    matrix: Dict[str, Dict[str, float]],
    title: str = "Specialist Orthogonality Matrix",
    save_path: Optional[str] = None
):
    """
    Plot heatmap showing specialist performance across all rules.
    Diagonal should be high, off-diagonal should be low for orthogonality.
    """
    if not HAS_MATPLOTLIB:
        print("Cannot plot: matplotlib not installed")
        return

    setup_neurips_style()

    rules = list(matrix.keys())

    # Build matrix
    data = []
    for spec_rule in rules:
        row = [matrix[spec_rule].get(test_rule, 0) for test_rule in rules]
        data.append(row)

    data = np.array(data)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Custom diverging colormap
    cmap = LinearSegmentedColormap.from_list(
        'orthogonality', ['#DC3545', '#F8F9FA', '#28A745']
    )

    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(len(rules)))
    ax.set_xticklabels([r.upper()[:4] for r in rules], rotation=45, ha='right')
    ax.set_yticks(range(len(rules)))
    ax.set_yticklabels([r.upper()[:4] for r in rules])

    ax.set_xlabel('Test Rule')
    ax.set_ylabel('Specialist Rule')
    ax.set_title(title)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy')

    # Add annotations
    for i in range(len(rules)):
        for j in range(len(rules)):
            value = data[i, j]
            color = 'white' if (value > 0.7 or value < 0.3) else 'black'
            ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                   color=color, fontsize=9, fontweight='bold' if i == j else 'normal')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def generate_all_figures(results_dir: str = 'results', output_dir: str = 'paper/figures'):
    """Generate all figures from experiment results."""
    if not HAS_MATPLOTLIB:
        print("Cannot generate figures: matplotlib not installed")
        return

    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Generating publication figures...")

    # Phase 2 swap test
    swap_results_file = results_path / 'phase2_enhanced_results.json'
    if swap_results_file.exists():
        with open(swap_results_file) as f:
            swap_results = json.load(f)
        plot_swap_test_results(
            swap_results,
            title='Phase 2: Prompt Swap Causality Test',
            save_path=str(output_path / 'swap_test_results.png')
        )

    # Prompt length ablation
    ablation_file = results_path / 'prompt_length_ablation.json'
    if ablation_file.exists():
        plot_prompt_length_ablation(
            str(ablation_file),
            save_path=str(output_path / 'prompt_length_ablation.png')
        )

    print(f"Figures saved to {output_path}")


def plot_multi_seed_results(
    results_path: str = 'results/multi_seed_results.json',
    save_path: Optional[str] = None
):
    """
    Plot multi-seed results with error bars and confidence intervals.
    """
    if not HAS_MATPLOTLIB:
        print("Cannot plot: matplotlib not installed")
        return

    with open(results_path, 'r') as f:
        results = json.load(f)

    setup_neurips_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Per-seed pass rates
    seeds = results['seeds']
    pass_rates = results['pass_rates']
    mean = results['mean']
    ci_low = results['ci_low']
    ci_high = results['ci_high']

    x = np.arange(len(seeds))
    bars = ax1.bar(x, pass_rates, color=COLORS['primary'], alpha=0.7, edgecolor='black')

    # Add mean line
    ax1.axhline(y=mean, color=COLORS['danger'], linestyle='--', linewidth=2, label=f'Mean: {mean:.1%}')

    # Add CI shading
    ax1.fill_between([-0.5, len(seeds)-0.5], ci_low, ci_high, alpha=0.2, color=COLORS['danger'], label='95% CI')

    ax1.set_xlabel('Seed')
    ax1.set_ylabel('Pass Rate')
    ax1.set_title('Multi-Seed Swap Test Results')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(s) for s in seeds])
    ax1.set_ylim(0, 1.1)
    ax1.legend(loc='lower right')

    # Right: Summary with error bar
    ax2.bar(['Swap\nTest'], [mean], yerr=[[mean - ci_low], [ci_high - mean]], 
            capsize=10, color=COLORS['success'], alpha=0.8, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Pass Rate')
    ax2.set_title(f'Aggregated Results (n={len(seeds)} seeds)')
    ax2.set_ylim(0, 1.1)

    # Add annotations
    ax2.annotate(f'{mean:.1%}\n[{ci_low:.1%}, {ci_high:.1%}]',
                xy=(0, mean), xytext=(0.3, mean + 0.1),
                fontsize=12, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLORS['dark']))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_baseline_comparison(
    results_path: str = 'results/baseline_comparison.json',
    save_path: Optional[str] = None
):
    """
    Plot baseline comparison with error bars.
    """
    if not HAS_MATPLOTLIB:
        print("Cannot plot: matplotlib not installed")
        return

    with open(results_path, 'r') as f:
        results = json.load(f)

    setup_neurips_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    conditions = list(results.keys())
    means = [results[c]['mean'] for c in conditions]
    stds = [results[c]['std'] for c in conditions]

    # Color by performance
    colors = []
    for c in conditions:
        if c == 'CORRECT_PROMPT':
            colors.append(COLORS['success'])
        elif c == 'NO_PROMPT':
            colors.append(COLORS['neutral'])
        else:
            colors.append(COLORS['danger'])

    x = np.arange(len(conditions))
    bars = ax.bar(x, means, yerr=stds, capsize=8, color=colors, alpha=0.8, 
                 edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Condition')
    ax.set_ylabel('Mean Accuracy')
    ax.set_title('Baseline Comparison: Prompt Specialization vs Alternatives')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in conditions], fontsize=10)
    ax.set_ylim(0, 1.1)

    # Add significance markers
    correct_mean = results['CORRECT_PROMPT']['mean']
    for i, (c, m) in enumerate(zip(conditions, means)):
        if c != 'CORRECT_PROMPT':
            improvement = correct_mean - m
            ax.annotate(f'+{improvement:.0%}', xy=(i, m + stds[i] + 0.05),
                       ha='center', fontsize=9, color=COLORS['success'], fontweight='bold')

    # Add value labels
    for bar, m, s in zip(bars, means, stds):
        height = bar.get_height()
        ax.annotate(f'{m:.1%}', xy=(bar.get_x() + bar.get_width()/2, height - 0.05),
                   ha='center', va='top', fontsize=10, fontweight='bold', color='white')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_scalability_analysis(
    results_path: str = 'results/scalability_results.json',
    save_path: Optional[str] = None
):
    """
    Plot scalability analysis across population sizes.
    """
    if not HAS_MATPLOTLIB:
        print("Cannot plot: matplotlib not installed")
        return

    with open(results_path, 'r') as f:
        results = json.load(f)

    setup_neurips_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    n_agents = [r['n_agents'] for r in results]
    coverage = [r['coverage'] for r in results]
    swap_pass = [r['swap_pass_rate'] for r in results]
    specialists = [r['unique_specialists'] for r in results]

    # Left: Coverage and Swap Pass Rate
    x = np.arange(len(n_agents))
    width = 0.35

    bars1 = ax1.bar(x - width/2, coverage, width, label='Rule Coverage', 
                    color=COLORS['primary'], alpha=0.8)
    bars2 = ax1.bar(x + width/2, swap_pass, width, label='Swap Pass Rate',
                    color=COLORS['success'], alpha=0.8)

    ax1.set_xlabel('Number of Agents')
    ax1.set_ylabel('Rate')
    ax1.set_title('Scalability: Performance vs Population Size')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(n) for n in n_agents])
    ax1.set_ylim(0, 1.1)
    ax1.legend(loc='lower right')
    ax1.axhline(y=0.7, color=COLORS['warning'], linestyle='--', alpha=0.5, label='Threshold')

    # Right: Unique specialists
    ax2.plot(n_agents, specialists, 'o-', color=COLORS['primary'], markersize=10, linewidth=2)
    ax2.fill_between(n_agents, 0, specialists, alpha=0.2, color=COLORS['primary'])

    ax2.set_xlabel('Number of Agents')
    ax2.set_ylabel('Unique Specialists')
    ax2.set_title('Specialist Diversity vs Population Size')
    ax2.set_ylim(0, 10)
    ax2.axhline(y=8, color=COLORS['success'], linestyle='--', alpha=0.5, label='Max (8 rules)')
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    # Generate figures from existing results
    generate_all_figures()
    
    # Generate new analysis figures
    from pathlib import Path
    output_path = Path('paper/figures')
    output_path.mkdir(parents=True, exist_ok=True)
    
    if Path('results/multi_seed_results.json').exists():
        plot_multi_seed_results(save_path=str(output_path / 'multi_seed_results.png'))
    
    if Path('results/baseline_comparison.json').exists():
        plot_baseline_comparison(save_path=str(output_path / 'baseline_comparison.png'))
    
    if Path('results/scalability_results.json').exists():
        plot_scalability_analysis(save_path=str(output_path / 'scalability_analysis.png'))
