"""
Analysis and Visualization Module

Generates figures with error bars and performs statistical tests.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Plotting disabled.")

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class StatResult:
    """Statistical test result."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    significant: bool
    interpretation: str


def compute_mean_std(values: List[float]) -> Tuple[float, float]:
    """Compute mean and standard deviation."""
    if not values:
        return 0.0, 0.0
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    std = math.sqrt(variance)
    return mean, std


def compute_confidence_interval(
    values: List[float],
    confidence: float = 0.95
) -> Tuple[float, float]:
    """Compute confidence interval."""
    if not values:
        return 0.0, 0.0

    mean, std = compute_mean_std(values)
    n = len(values)
    se = std / math.sqrt(n) if n > 1 else 0

    # Z-score for 95% CI
    z = 1.96 if confidence == 0.95 else 2.576
    margin = z * se

    return mean - margin, mean + margin


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """Calculate Cohen's d effect size."""
    if not group1 or not group2:
        return 0.0

    mean1, std1 = compute_mean_std(group1)
    mean2, std2 = compute_mean_std(group2)

    # Pooled standard deviation
    n1, n2 = len(group1), len(group2)
    pooled_std = math.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (mean1 - mean2) / pooled_std


def mann_whitney_u(group1: List[float], group2: List[float]) -> StatResult:
    """Perform Mann-Whitney U test."""
    if HAS_SCIPY:
        statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    else:
        # Simplified approximation
        n1, n2 = len(group1), len(group2)
        mean1, _ = compute_mean_std(group1)
        mean2, _ = compute_mean_std(group2)
        statistic = n1 * n2 / 2  # Placeholder
        p_value = 0.05 if abs(mean1 - mean2) > 0.1 else 0.5

    effect = cohens_d(group1, group2)
    significant = p_value < 0.05

    interpretation = "significant difference" if significant else "no significant difference"

    return StatResult(
        test_name="Mann-Whitney U",
        statistic=statistic,
        p_value=p_value,
        effect_size=effect,
        significant=significant,
        interpretation=interpretation
    )


def independent_t_test(group1: List[float], group2: List[float]) -> StatResult:
    """Perform independent samples t-test."""
    if HAS_SCIPY:
        statistic, p_value = stats.ttest_ind(group1, group2)
    else:
        # Simplified t-test approximation
        mean1, std1 = compute_mean_std(group1)
        mean2, std2 = compute_mean_std(group2)
        n1, n2 = len(group1), len(group2)

        # Pooled standard error
        se = math.sqrt(std1**2/n1 + std2**2/n2) if n1 > 0 and n2 > 0 else 1
        statistic = (mean1 - mean2) / se if se > 0 else 0

        # Approximate p-value
        p_value = 0.05 if abs(statistic) > 2 else 0.5

    effect = cohens_d(group1, group2)
    significant = p_value < 0.05

    interpretation = "significant difference" if significant else "no significant difference"

    return StatResult(
        test_name="Independent t-test",
        statistic=statistic,
        p_value=p_value,
        effect_size=effect,
        significant=significant,
        interpretation=interpretation
    )


def paired_t_test(before: List[float], after: List[float]) -> StatResult:
    """Perform paired samples t-test."""
    if len(before) != len(after):
        raise ValueError("Paired t-test requires equal length samples")

    if HAS_SCIPY:
        statistic, p_value = stats.ttest_rel(before, after)
    else:
        # Simplified paired t-test
        differences = [a - b for a, b in zip(after, before)]
        mean_diff, std_diff = compute_mean_std(differences)
        n = len(differences)

        se = std_diff / math.sqrt(n) if n > 0 else 1
        statistic = mean_diff / se if se > 0 else 0
        p_value = 0.05 if abs(statistic) > 2 else 0.5

    effect = cohens_d(after, before)
    significant = p_value < 0.05

    interpretation = "significant change" if significant else "no significant change"

    return StatResult(
        test_name="Paired t-test",
        statistic=statistic,
        p_value=p_value,
        effect_size=effect,
        significant=significant,
        interpretation=interpretation
    )


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def compute_all_statistics(
    group1: List[float],
    group2: List[float],
    paired: bool = False
) -> Dict[str, StatResult]:
    """Run all appropriate statistical tests."""
    results = {}

    # Effect size
    d = cohens_d(group1, group2)

    if paired:
        results["paired_t_test"] = paired_t_test(group1, group2)
    else:
        results["independent_t_test"] = independent_t_test(group1, group2)
        results["mann_whitney_u"] = mann_whitney_u(group1, group2)

    # Add effect size interpretation
    results["effect_size"] = StatResult(
        test_name="Cohen's d",
        statistic=d,
        p_value=0,  # Not applicable
        effect_size=d,
        significant=abs(d) >= 0.2,
        interpretation=interpret_effect_size(d)
    )

    return results


def print_statistical_comparison(
    name1: str,
    values1: List[float],
    name2: str,
    values2: List[float],
    paired: bool = False
):
    """Print formatted statistical comparison."""
    print(f"\n{'='*50}")
    print(f"Statistical Comparison: {name1} vs {name2}")
    print(f"{'='*50}")

    # Descriptive stats
    mean1, std1 = compute_mean_std(values1)
    mean2, std2 = compute_mean_std(values2)
    ci1 = compute_confidence_interval(values1)
    ci2 = compute_confidence_interval(values2)

    print(f"\n{name1}:")
    print(f"  Mean: {mean1:.3f} (SD: {std1:.3f})")
    print(f"  95% CI: [{ci1[0]:.3f}, {ci1[1]:.3f}]")
    print(f"  N: {len(values1)}")

    print(f"\n{name2}:")
    print(f"  Mean: {mean2:.3f} (SD: {std2:.3f})")
    print(f"  95% CI: [{ci2[0]:.3f}, {ci2[1]:.3f}]")
    print(f"  N: {len(values2)}")

    # Statistical tests
    results = compute_all_statistics(values1, values2, paired)

    print(f"\nStatistical Tests:")
    for name, result in results.items():
        print(f"  {result.test_name}:")
        if result.test_name != "Cohen's d":
            print(f"    statistic={result.statistic:.3f}, p={result.p_value:.4f}")
        print(f"    effect size (d)={result.effect_size:.3f} ({result.interpretation})")
        if result.test_name != "Cohen's d":
            sig_str = "*** SIGNIFICANT ***" if result.significant else "not significant"
            print(f"    {sig_str}")

    print(f"{'='*50}")


def bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval."""
    import random

    if not values:
        return 0.0, 0.0

    bootstrap_means = []
    n = len(values)

    for _ in range(n_bootstrap):
        sample = [random.choice(values) for _ in range(n)]
        bootstrap_means.append(sum(sample) / n)

    bootstrap_means.sort()

    alpha = 1 - confidence
    lower_idx = int(n_bootstrap * alpha / 2)
    upper_idx = int(n_bootstrap * (1 - alpha / 2))

    return bootstrap_means[lower_idx], bootstrap_means[upper_idx]


def plot_lsi_over_generations(
    lsi_histories: List[List[float]],
    title: str = "LSI Over Generations",
    save_path: Optional[str] = None
):
    """Plot LSI curves with error bands."""
    if not HAS_MATPLOTLIB:
        print("Cannot plot: matplotlib not installed")
        return

    # Compute mean and std at each generation
    n_gens = len(lsi_histories[0]) if lsi_histories else 0
    means = []
    stds = []

    for gen in range(n_gens):
        gen_values = [h[gen] for h in lsi_histories if gen < len(h)]
        mean, std = compute_mean_std(gen_values)
        means.append(mean)
        stds.append(std)

    generations = list(range(n_gens))

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot mean line
    ax.plot(generations, means, 'b-', linewidth=2, label='Mean LSI')

    # Plot error band
    lower = [m - s for m, s in zip(means, stds)]
    upper = [m + s for m, s in zip(means, stds)]
    ax.fill_between(generations, lower, upper, alpha=0.3, color='blue', label='±1 Std Dev')

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Population LSI', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def plot_baseline_comparison(
    baseline_results: Dict[str, List[float]],
    metric_name: str = "LSI",
    save_path: Optional[str] = None
):
    """Plot bar chart comparing baselines with error bars."""
    if not HAS_MATPLOTLIB:
        print("Cannot plot: matplotlib not installed")
        return

    baselines = list(baseline_results.keys())
    means = []
    stds = []
    cis = []

    for baseline in baselines:
        values = baseline_results[baseline]
        mean, std = compute_mean_std(values)
        ci_low, ci_high = compute_confidence_interval(values)
        means.append(mean)
        stds.append(std)
        cis.append((mean - ci_low, ci_high - mean))

    fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(baselines))
    colors = plt.cm.viridis([i / len(baselines) for i in range(len(baselines))])

    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor='black')

    ax.set_xlabel('Condition', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f'{metric_name} Comparison Across Conditions', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(baselines, rotation=45, ha='right')
    ax.set_ylim(0, max(means) * 1.3 if means else 1)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def plot_specialist_distribution(
    distributions: Dict[str, Dict[str, int]],
    save_path: Optional[str] = None
):
    """Plot heatmap of specialist distribution across conditions."""
    if not HAS_MATPLOTLIB:
        print("Cannot plot: matplotlib not installed")
        return

    conditions = list(distributions.keys())
    all_domains = set()
    for dist in distributions.values():
        all_domains.update(dist.keys())
    domains = sorted(list(all_domains))

    # Build matrix
    matrix = []
    for condition in conditions:
        row = [distributions[condition].get(domain, 0) for domain in domains]
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(domains, rotation=45, ha='right')
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels(conditions)

    ax.set_xlabel('Domain', fontsize=12)
    ax.set_ylabel('Condition', fontsize=12)
    ax.set_title('Specialist Distribution Heatmap', fontsize=14)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of Agents', fontsize=10)

    # Add text annotations
    for i in range(len(conditions)):
        for j in range(len(domains)):
            value = matrix[i][j]
            ax.text(j, i, str(value), ha='center', va='center',
                   color='white' if value > max(max(row) for row in matrix) / 2 else 'black')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def generate_statistics_table(
    results: Dict[str, List[float]],
    baseline_name: str = "cumulative"
) -> str:
    """Generate markdown table of statistical comparisons."""
    lines = []
    lines.append("| Condition | Mean | Std | 95% CI | vs Baseline |")
    lines.append("|-----------|------|-----|--------|-------------|")

    baseline_values = results.get(baseline_name, [])

    for condition, values in results.items():
        mean, std = compute_mean_std(values)
        ci_low, ci_high = compute_confidence_interval(values)

        if condition == baseline_name:
            comparison = "-"
        elif baseline_values:
            test_result = mann_whitney_u(values, baseline_values)
            p_str = f"p={test_result.p_value:.3f}"
            sig = "**" if test_result.significant else ""
            comparison = f"{sig}{p_str}, d={test_result.effect_size:.2f}{sig}"
        else:
            comparison = "N/A"

        lines.append(f"| {condition} | {mean:.3f} | {std:.3f} | [{ci_low:.3f}, {ci_high:.3f}] | {comparison} |")

    return "\n".join(lines)


def analyze_experiment_results(results_path: str) -> Dict:
    """Load and analyze experiment results."""
    with open(results_path, 'r') as f:
        data = json.load(f)

    analysis = {
        "summary": data.get("summary", {}),
        "statistics": {},
        "figures_generated": []
    }

    # Generate figures
    output_dir = Path(results_path).parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    if "lsi_histories" in data:
        plot_lsi_over_generations(
            data["lsi_histories"],
            save_path=str(output_dir / "lsi_over_generations.png")
        )
        analysis["figures_generated"].append("lsi_over_generations.png")

    return analysis


def print_analysis_report(results: Dict):
    """Print formatted analysis report."""
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS REPORT")
    print("=" * 70)

    if "summary" in results:
        summary = results["summary"]
        print("\nSummary Statistics:")

        if "lsi" in summary:
            lsi = summary["lsi"]
            print(f"  LSI: {lsi.get('mean', 0):.3f} ± {lsi.get('std', 0):.3f}")
            print(f"  95% CI: [{lsi.get('ci_low', 0):.3f}, {lsi.get('ci_high', 0):.3f}]")

        if "unique_specializations" in summary:
            specs = summary["unique_specializations"]
            print(f"  Unique Specs: {specs.get('mean', 0):.1f} ± {specs.get('std', 0):.1f}")

    if "figures_generated" in results:
        print(f"\nFigures Generated: {len(results['figures_generated'])}")
        for fig in results["figures_generated"]:
            print(f"  - {fig}")

    print("=" * 70)


# =====================================================
# SEMANTIC SPECIALIZATION METRICS (P1.2)
# =====================================================

def compute_prompt_distinctiveness(prompts: List[str]) -> float:
    """
    Measure how semantically distinct evolved prompts are using sentence embeddings.

    Returns a value between 0 (all identical) and 1 (maximally distinct).
    Requires: pip install sentence-transformers
    """
    if not prompts or len(prompts) < 2:
        return 0.0

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        # Use lightweight model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(prompts)

        # Calculate pairwise cosine similarities
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(embeddings)

        # Return mean off-diagonal distance (1 - similarity = distance)
        n = len(prompts)
        mask = ~np.eye(n, dtype=bool)
        mean_similarity = sim_matrix[mask].mean()

        return float(1 - mean_similarity)

    except ImportError:
        print("Warning: sentence-transformers not installed. Using fallback.")
        return _compute_prompt_distinctiveness_fallback(prompts)


def _compute_prompt_distinctiveness_fallback(prompts: List[str]) -> float:
    """
    Fallback distinctiveness measure using Jaccard distance on word sets.
    """
    if not prompts or len(prompts) < 2:
        return 0.0

    word_sets = [set(p.lower().split()) for p in prompts]

    total_distance = 0.0
    pairs = 0

    for i in range(len(word_sets)):
        for j in range(i + 1, len(word_sets)):
            intersection = len(word_sets[i] & word_sets[j])
            union = len(word_sets[i] | word_sets[j])
            jaccard_sim = intersection / union if union > 0 else 0
            total_distance += (1 - jaccard_sim)
            pairs += 1

    return total_distance / pairs if pairs > 0 else 0.0


def compute_prompt_clustering(prompts: List[str], n_clusters: int = None) -> Dict:
    """
    Cluster prompts and analyze their distribution.

    Returns:
        - n_clusters: number of distinct clusters found
        - cluster_sizes: size of each cluster
        - silhouette_score: quality of clustering (-1 to 1, higher is better)
    """
    if not prompts or len(prompts) < 2:
        return {"n_clusters": 0, "cluster_sizes": [], "silhouette_score": 0.0}

    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        import numpy as np

        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(prompts)

        # Auto-determine n_clusters if not specified
        if n_clusters is None:
            n_clusters = min(len(prompts) // 2, 8)

        if n_clusters < 2:
            return {"n_clusters": 1, "cluster_sizes": [len(prompts)], "silhouette_score": 0.0}

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        # Compute cluster sizes
        cluster_sizes = [int(sum(labels == i)) for i in range(n_clusters)]

        # Compute silhouette score
        sil_score = float(silhouette_score(embeddings, labels))

        return {
            "n_clusters": n_clusters,
            "cluster_sizes": cluster_sizes,
            "silhouette_score": sil_score
        }

    except ImportError:
        return {"n_clusters": 0, "cluster_sizes": [], "silhouette_score": 0.0, "error": "dependencies not installed"}


def analyze_prompt_evolution(
    initial_prompts: List[str],
    final_prompts: List[str]
) -> Dict:
    """
    Analyze how prompts evolved from initial to final state.

    Returns:
        - initial_distinctiveness: how distinct prompts were at start
        - final_distinctiveness: how distinct prompts are at end
        - distinctiveness_gain: improvement in distinctiveness
        - semantic_drift: average semantic distance from initial prompts
    """
    initial_dist = compute_prompt_distinctiveness(initial_prompts)
    final_dist = compute_prompt_distinctiveness(final_prompts)

    # Compute semantic drift (how much prompts changed)
    drift = 0.0
    if len(initial_prompts) == len(final_prompts):
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np

            model = SentenceTransformer('all-MiniLM-L6-v2')
            initial_emb = model.encode(initial_prompts)
            final_emb = model.encode(final_prompts)

            # Compute diagonal of similarity matrix (same agent before/after)
            drifts = [1 - cosine_similarity([i], [f])[0][0]
                     for i, f in zip(initial_emb, final_emb)]
            drift = float(np.mean(drifts))

        except ImportError:
            # Fallback using word overlap
            for initial, final in zip(initial_prompts, final_prompts):
                initial_words = set(initial.lower().split())
                final_words = set(final.lower().split())
                if initial_words or final_words:
                    overlap = len(initial_words & final_words)
                    total = len(initial_words | final_words)
                    drift += (1 - overlap / total) if total > 0 else 0
            drift /= len(initial_prompts) if initial_prompts else 1

    return {
        "initial_distinctiveness": initial_dist,
        "final_distinctiveness": final_dist,
        "distinctiveness_gain": final_dist - initial_dist,
        "semantic_drift": drift
    }


def analyze_seed_switching(checkpoint_data: Dict) -> Dict:
    """
    Analyze whether agents switched from initial rule (Option B+) to different final specialization.

    Args:
        checkpoint_data: Dict with 'agents' list containing 'initial_rule' and 'strategies' or 'primary_specialization'

    Returns:
        Dict with switch rate, chi-square test results, and interpretation
    """
    agents = checkpoint_data.get("agents", [])

    if not agents:
        return {"error": "No agent data found"}

    # Count switches
    switched = 0
    stayed = 0
    switch_details = []

    for agent in agents:
        initial = agent.get("initial_rule", agent.get("seeded_rule", None))

        # Determine final specialization from strategies or explicit field
        final = agent.get("primary_specialization", None)
        if final is None:
            strategies = agent.get("strategies", {})
            if strategies:
                # Find rule with highest level
                final = max(strategies.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0)[0]

        if initial and final:
            is_switch = (initial != final)
            if is_switch:
                switched += 1
            else:
                stayed += 1
            switch_details.append({
                "agent_id": agent.get("agent_id", "unknown"),
                "initial": initial,
                "final": final,
                "switched": is_switch
            })

    total = switched + stayed
    if total == 0:
        return {"error": "Could not determine initial/final rules for any agent"}

    switch_rate = switched / total

    # Chi-square test: Is switch rate significantly different from random (50%)?
    # Under null hypothesis: equal probability of switching or staying
    expected_switched = total * 0.5
    expected_stayed = total * 0.5

    chi_square = 0
    p_value = 1.0

    if HAS_SCIPY:
        # Chi-square goodness of fit
        observed = [switched, stayed]
        expected = [expected_switched, expected_stayed]
        try:
            chi2_stat, p_value = stats.chisquare(observed, expected)
            chi_square = float(chi2_stat)
            p_value = float(p_value)
        except Exception:
            pass
    else:
        # Manual chi-square calculation
        chi_square = ((switched - expected_switched) ** 2 / expected_switched +
                      (stayed - expected_stayed) ** 2 / expected_stayed)
        # Approximate p-value (1 df)
        # For chi-square with 1 df: p < 0.05 if chi2 > 3.84
        if chi_square > 6.63:
            p_value = 0.01
        elif chi_square > 3.84:
            p_value = 0.05
        else:
            p_value = 0.1

    # Interpretation
    if switch_rate > 0.6 and p_value < 0.05:
        interpretation = "STRONG: Agents significantly change specialization (supports learning hypothesis)"
    elif switch_rate > 0.5:
        interpretation = "MODERATE: Agents tend to switch, but not statistically significant"
    elif switch_rate < 0.4 and p_value < 0.05:
        interpretation = "WEAK: Agents tend to stay with initial rule (suggests initial seeding dominates)"
    else:
        interpretation = "NEUTRAL: No clear pattern in specialization changes"

    return {
        "total_agents": total,
        "switched": switched,
        "stayed": stayed,
        "switch_rate": switch_rate,
        "chi_square": chi_square,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "interpretation": interpretation,
        "details": switch_details,
        "success": switch_rate > 0.6 and p_value < 0.05
    }


def run_seed_switching_analysis(checkpoint_dir: str = "results/phase1_checkpoints") -> Dict:
    """
    Run seed-switching analysis on all available Phase 1 checkpoints.

    Args:
        checkpoint_dir: Directory containing checkpoint JSON files

    Returns:
        Aggregated results across all checkpoints
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return {"error": f"Checkpoint directory not found: {checkpoint_dir}"}

    # Find all checkpoint files
    checkpoints = list(checkpoint_path.glob("checkpoint_seed*_gen*.json"))
    if not checkpoints:
        return {"error": "No checkpoint files found"}

    # Group by seed, take latest generation for each
    seed_checkpoints = {}
    for cp in checkpoints:
        # Parse filename: checkpoint_seed{X}_gen{Y}.json
        name = cp.stem
        parts = name.split("_")
        seed = None
        gen = 0
        for part in parts:
            if part.startswith("seed"):
                seed = part
            elif part.startswith("gen"):
                gen = int(part.replace("gen", ""))

        if seed:
            if seed not in seed_checkpoints or seed_checkpoints[seed][1] < gen:
                seed_checkpoints[seed] = (cp, gen)

    # Analyze each seed's latest checkpoint
    all_results = []
    for seed, (cp_path, gen) in seed_checkpoints.items():
        try:
            with open(cp_path) as f:
                data = json.load(f)
            result = analyze_seed_switching(data)
            result["seed"] = seed
            result["generation"] = gen
            all_results.append(result)
        except Exception as e:
            all_results.append({"seed": seed, "error": str(e)})

    # Aggregate
    valid_results = [r for r in all_results if "switch_rate" in r]
    if not valid_results:
        return {"error": "No valid checkpoint data found", "details": all_results}

    switch_rates = [r["switch_rate"] for r in valid_results]
    mean_switch_rate = sum(switch_rates) / len(switch_rates)

    # Combined chi-square (sum of individual chi-squares)
    total_chi = sum(r.get("chi_square", 0) for r in valid_results)
    # Degrees of freedom = number of tests
    df = len(valid_results)

    combined_p = 1.0
    if HAS_SCIPY:
        try:
            combined_p = float(1 - stats.chi2.cdf(total_chi, df))
        except Exception:
            pass

    return {
        "n_seeds": len(valid_results),
        "mean_switch_rate": mean_switch_rate,
        "switch_rates": switch_rates,
        "combined_chi_square": total_chi,
        "combined_df": df,
        "combined_p_value": combined_p,
        "success": mean_switch_rate > 0.6,
        "individual_results": all_results
    }


if __name__ == "__main__":
    # Demo with sample data
    print("Analysis Module Demo")
    print("=" * 40)

    # Sample LSI histories
    sample_histories = [
        [0.1 + 0.005 * g + 0.02 * (g % 10 == 0) for g in range(100)]
        for _ in range(5)
    ]

    # Sample baseline comparison
    sample_baselines = {
        "cumulative": [0.45, 0.48, 0.42, 0.51, 0.47],
        "random_strategy": [0.25, 0.28, 0.22, 0.27, 0.24],
        "no_evolution": [0.15, 0.18, 0.12, 0.16, 0.14],
        "handcrafted": [0.72, 0.75, 0.70, 0.73, 0.71]
    }

    # Print statistics table
    print("\n" + generate_statistics_table(sample_baselines))

    # Generate plots if matplotlib available
    if HAS_MATPLOTLIB:
        print("\nGenerating sample plots...")
        plot_lsi_over_generations(sample_histories, save_path="sample_lsi.png")
        plot_baseline_comparison(sample_baselines, save_path="sample_comparison.png")
        print("Done!")
