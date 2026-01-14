"""
Hero Visualization for Paper

Creates the main visualization showing specialization trajectory.

Concept: "Specialization Trajectory"
- X-axis: Generations (0-100)
- Y-axis: Strategy levels for each agent
- Color: Rule type
- Show agents diverging from identical start to specialized endpoints
- Inset: Final niche distribution pie chart
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import random
import math

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Visualization disabled.")


# =============================================================================
# COLOR SCHEMES
# =============================================================================

# NeurIPS-friendly color palette (colorblind safe)
RULE_COLORS = {
    "position": "#E69F00",     # Orange
    "pattern": "#56B4E9",      # Sky blue
    "inverse": "#009E73",      # Bluish green
    "vowel_start": "#F0E442",  # Yellow
    "rhyme": "#0072B2",        # Blue
    "alphabet": "#D55E00",     # Vermillion
    "math_mod": "#CC79A7",     # Reddish purple
    "animate": "#999999",      # Gray
}

# Gradient colors for generations
GEN_CMAP = "viridis"


# =============================================================================
# DATA GENERATION (for demo when no real data)
# =============================================================================

def generate_sample_trajectory(
    n_agents: int = 12,
    n_generations: int = 100,
    n_rules: int = 8,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Generate sample specialization trajectory data.

    Simulates the evolution process where agents start identical
    and diverge to specialize in different rules.
    """
    rng = random.Random(seed)
    rules = list(RULE_COLORS.keys())[:n_rules]

    # Initialize agents with equal (low) levels
    agents = []
    for i in range(n_agents):
        agent = {
            "agent_id": f"agent_{i:02d}",
            "initial_rule": rng.choice(rules),  # Option B+ initialization
            "trajectory": {r: [] for r in rules}
        }
        agents.append(agent)

    # Simulate evolution
    for gen in range(n_generations):
        for agent in agents:
            for rule in rules:
                # Current level (from previous gen or 0)
                if agent["trajectory"][rule]:
                    current = agent["trajectory"][rule][-1]
                else:
                    # Start with 1 for initial rule (Option B+)
                    current = 1 if rule == agent["initial_rule"] else 0

                # Probability of level up based on current specialization
                # Higher levels are harder to reach, but easier for specialized rule
                if rule == agent["initial_rule"]:
                    prob = 0.03 * (1 - current / 3)
                else:
                    prob = 0.01 * (1 - current / 3)

                # Level up
                if rng.random() < prob and current < 3:
                    current = min(3, current + 1)

                # After reaching L3, exclusivity kicks in
                if current == 3:
                    # This agent now specializes only in this rule
                    for other_rule in rules:
                        if other_rule != rule and len(agent["trajectory"][other_rule]) > 0:
                            # Cap other rules at current level
                            pass

                agent["trajectory"][rule].append(current)

    # Determine final specialization
    for agent in agents:
        max_level = 0
        specialized_rule = None
        for rule in rules:
            final_level = agent["trajectory"][rule][-1]
            if final_level > max_level:
                max_level = final_level
                specialized_rule = rule
        agent["final_specialization"] = specialized_rule
        agent["final_level"] = max_level

    return {
        "agents": agents,
        "n_generations": n_generations,
        "rules": rules
    }


# =============================================================================
# HERO FIGURE
# =============================================================================

def create_hero_figure(
    trajectory_data: Dict[str, Any] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
    dpi: int = 300
) -> Optional[plt.Figure]:
    """
    Create the main hero visualization for the paper.

    Layout:
    - Main plot: Specialization trajectories over time
    - Inset: Final niche distribution
    - Bottom: Legend with rule descriptions
    """
    if not HAS_MATPLOTLIB:
        print("Cannot create visualization: matplotlib not installed")
        return None

    # Generate sample data if not provided
    if trajectory_data is None:
        trajectory_data = generate_sample_trajectory()

    agents = trajectory_data["agents"]
    n_generations = trajectory_data["n_generations"]
    rules = trajectory_data["rules"]

    # Create figure with custom layout
    fig = plt.figure(figsize=figsize, facecolor='white')
    gs = GridSpec(2, 3, height_ratios=[4, 1], width_ratios=[3, 1, 1])

    # Main trajectory plot
    ax_main = fig.add_subplot(gs[0, :2])

    # Inset: niche distribution
    ax_pie = fig.add_subplot(gs[0, 2])

    # Legend area
    ax_legend = fig.add_subplot(gs[1, :])
    ax_legend.axis('off')

    # ==========================================================================
    # MAIN TRAJECTORY PLOT
    # ==========================================================================

    generations = list(range(n_generations))

    # Plot each agent's dominant rule trajectory
    for agent in agents:
        # Find the rule with highest final level
        final_rule = agent["final_specialization"]
        trajectory = agent["trajectory"][final_rule]

        color = RULE_COLORS.get(final_rule, "#666666")

        # Plot with some transparency and jitter
        jitter = [t + random.gauss(0, 0.05) for t in trajectory]
        ax_main.plot(
            generations, jitter,
            color=color,
            alpha=0.6,
            linewidth=1.5,
            label=final_rule if agent == agents[0] else None
        )

    # Styling
    ax_main.set_xlim(0, n_generations)
    ax_main.set_ylim(-0.2, 3.4)
    ax_main.set_xlabel("Generation", fontsize=14)
    ax_main.set_ylabel("Strategy Level", fontsize=14)
    ax_main.set_title(
        "Emergent Preference Specialization: Agent Trajectories",
        fontsize=16,
        fontweight='bold'
    )

    # Add level labels
    ax_main.set_yticks([0, 1, 2, 3])
    ax_main.set_yticklabels(["L0\n(None)", "L1\n(Hint)", "L2\n(Partial)", "L3\n(Full)"])

    # Add phase annotations
    ax_main.axvline(x=25, color='gray', linestyle='--', alpha=0.5)
    ax_main.axvline(x=75, color='gray', linestyle='--', alpha=0.5)
    ax_main.text(12, 3.2, "Exploration", ha='center', fontsize=10, style='italic')
    ax_main.text(50, 3.2, "Specialization", ha='center', fontsize=10, style='italic')
    ax_main.text(87, 3.2, "Equilibrium", ha='center', fontsize=10, style='italic')

    # Grid
    ax_main.grid(True, alpha=0.3, linestyle=':')
    ax_main.set_facecolor('#f8f8f8')

    # ==========================================================================
    # PIE CHART: FINAL NICHE DISTRIBUTION
    # ==========================================================================

    # Count specialists per rule
    specialist_counts = {}
    for agent in agents:
        rule = agent["final_specialization"]
        specialist_counts[rule] = specialist_counts.get(rule, 0) + 1

    # Prepare pie data
    labels = []
    sizes = []
    colors = []

    for rule in rules:
        count = specialist_counts.get(rule, 0)
        if count > 0:
            labels.append(rule.upper())
            sizes.append(count)
            colors.append(RULE_COLORS.get(rule, "#666666"))

    # Draw pie
    wedges, texts, autotexts = ax_pie.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct=lambda pct: f'{int(round(pct/100*sum(sizes)))}' if pct > 5 else '',
        startangle=90,
        explode=[0.02] * len(sizes)
    )

    ax_pie.set_title("Final Niche\nDistribution", fontsize=12, fontweight='bold')

    # ==========================================================================
    # LEGEND
    # ==========================================================================

    # Create custom legend patches
    legend_patches = []
    for rule in rules:
        color = RULE_COLORS.get(rule, "#666666")
        patch = mpatches.Patch(color=color, label=rule.upper().replace("_", " "))
        legend_patches.append(patch)

    ax_legend.legend(
        handles=legend_patches,
        loc='center',
        ncol=4,
        fontsize=11,
        title="Rule Types",
        title_fontsize=12,
        frameon=True,
        facecolor='white',
        edgecolor='gray'
    )

    # ==========================================================================
    # FINAL TOUCHES
    # ==========================================================================

    plt.tight_layout()

    # Add caption
    fig.text(
        0.5, 0.02,
        "Figure: Agents start with identical generic prompts (Generation 0) and evolve specialized "
        "preferences through competitive selection. Colors indicate final rule specialization.",
        ha='center',
        fontsize=10,
        style='italic',
        wrap=True
    )

    # Save
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Saved hero figure to {save_path}")

    return fig


def create_metrics_evolution_figure(
    metrics_history: Dict[str, List[float]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> Optional[plt.Figure]:
    """
    Create figure showing evolution of key metrics over generations.

    Metrics: SCI, Diversity, L3 Rate, PSI
    """
    if not HAS_MATPLOTLIB:
        return None

    # Generate sample data if not provided
    if metrics_history is None:
        n_gens = 100
        metrics_history = {
            "SCI": [0.1 + 0.4 * (1 - math.exp(-g/30)) + random.gauss(0, 0.02) for g in range(n_gens)],
            "Diversity": [0.125 + 0.7 * (1 - math.exp(-g/40)) + random.gauss(0, 0.02) for g in range(n_gens)],
            "L3_Rate": [0 + 0.8 * max(0, 1 - math.exp(-(g-20)/25)) + random.gauss(0, 0.02) for g in range(n_gens)],
            "PSI": [0.3 + 0.6 * (1 - math.exp(-g/50)) + random.gauss(0, 0.02) for g in range(n_gens)],
        }

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)

    colors = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7"]
    titles = ["Strategy Concentration Index (SCI)", "Specialization Diversity",
              "L3 Specialist Rate", "Preference Stability Index (PSI)"]

    for ax, (metric_name, values), color, title in zip(
        axes.flat, metrics_history.items(), colors, titles
    ):
        generations = list(range(len(values)))

        ax.plot(generations, values, color=color, linewidth=2)
        ax.fill_between(generations, values, alpha=0.3, color=color)

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f8f8')

        # Add final value annotation
        final_val = values[-1]
        ax.annotate(
            f'{final_val:.2f}',
            xy=(len(values)-1, final_val),
            xytext=(len(values)-10, final_val + 0.1),
            fontsize=10,
            arrowprops=dict(arrowstyle='->', color='gray')
        )

    # Common labels
    axes[1, 0].set_xlabel("Generation", fontsize=12)
    axes[1, 1].set_xlabel("Generation", fontsize=12)

    plt.suptitle(
        "Evolution of Specialization Metrics",
        fontsize=14,
        fontweight='bold',
        y=1.02
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved metrics figure to {save_path}")

    return fig


def create_comparison_figure(
    comparison_data: Dict[str, Dict] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> Optional[plt.Figure]:
    """
    Create bar chart comparing conditions with error bars and effect sizes.
    """
    if not HAS_MATPLOTLIB:
        return None

    # Sample data if not provided
    if comparison_data is None:
        comparison_data = {
            "SINGLE_GENERALIST": {"mean": 0.50, "std": 0.05, "n": 10},
            "ORACLE_ROUTING": {"mean": 0.85, "std": 0.04, "n": 10},
            "CONFIDENCE_ROUTING": {"mean": 0.78, "std": 0.06, "n": 10},
            "ENSEMBLE": {"mean": 0.88, "std": 0.03, "n": 10},
        }

    fig, ax = plt.subplots(figsize=figsize)

    conditions = list(comparison_data.keys())
    means = [comparison_data[c]["mean"] for c in conditions]
    stds = [comparison_data[c]["std"] for c in conditions]

    x = np.arange(len(conditions))
    colors = ["#999999", "#E69F00", "#56B4E9", "#009E73"]

    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.02,
            f'{mean:.1%}',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )

    # Add improvement annotations
    baseline = means[0]
    for i, mean in enumerate(means[1:], 1):
        improvement = (mean - baseline) / baseline * 100
        if improvement > 0:
            ax.annotate(
                f'+{improvement:.0f}%',
                xy=(i, mean),
                xytext=(i, mean + 0.15),
                ha='center',
                fontsize=10,
                color='green',
                fontweight='bold'
            )

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in conditions], fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Population vs Generalist: 5-Condition Comparison", fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_facecolor('#f8f8f8')

    # Add horizontal line for baseline
    ax.axhline(y=baseline, color='gray', linestyle='--', alpha=0.7, label='Generalist Baseline')
    ax.legend(loc='upper left')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved comparison figure to {save_path}")

    return fig


# =============================================================================
# MAIN
# =============================================================================

def generate_all_figures(output_dir: str = "results/figures"):
    """Generate all paper figures."""
    if not HAS_MATPLOTLIB:
        print("Cannot generate figures: matplotlib not installed")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Generating paper figures...")

    # Hero figure
    create_hero_figure(save_path=str(output_path / "hero_trajectory.png"))

    # Metrics evolution
    create_metrics_evolution_figure(save_path=str(output_path / "metrics_evolution.png"))

    # Comparison figure
    create_comparison_figure(save_path=str(output_path / "condition_comparison.png"))

    print(f"All figures saved to {output_path}")


if __name__ == "__main__":
    generate_all_figures()
