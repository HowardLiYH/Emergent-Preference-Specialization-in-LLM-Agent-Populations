"""
NeurIPS-Appropriate Metrics for Emergent Specialization

Literature-grounded metrics for measuring agent specialization:
- Gini Coefficient (Gini, 1912): Inequality in strategy distribution
- Shannon Entropy (Shannon, 1948): Information-theoretic diversity
- Simpson's Index (Simpson, 1949): Probability of same specialization
- Herfindahl-Hirschman Index: Concentration measure from economics

These metrics are well-established and will be recognized by NeurIPS reviewers.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from .synthetic_rules import RuleType


@dataclass
class SpecializationMetrics:
    """Complete specialization metrics for a population."""
    # Agent-level metrics
    sci_mean: float  # Strategy Concentration Index
    sci_std: float
    gini_mean: float  # Gini coefficient
    gini_std: float

    # Population-level metrics
    l3_specialist_rate: float  # % of agents at max level
    specialization_diversity: float  # unique primaries / total rules
    hhi: float  # Herfindahl-Hirschman Index
    entropy: float  # Shannon entropy of primary specializations

    # Thesis support
    emergence_score: float  # Combined metric

    def to_dict(self) -> dict:
        return {
            'sci_mean': self.sci_mean,
            'sci_std': self.sci_std,
            'gini_mean': self.gini_mean,
            'gini_std': self.gini_std,
            'l3_specialist_rate': self.l3_specialist_rate,
            'specialization_diversity': self.specialization_diversity,
            'hhi': self.hhi,
            'entropy': self.entropy,
            'emergence_score': self.emergence_score,
        }


def compute_sci(strategy_levels: Dict[RuleType, int]) -> float:
    """
    Strategy Concentration Index (Entropy-based)

    Formula: SCI = 1 - H(s) / log(R)

    Where:
    - H(s) = Shannon entropy of normalized strategy distribution
    - R = number of rules (categories)

    This matches the definition in both Paper 1 (Specialization Index) and
    Paper 2 (Strategy Concentration Index), ensuring consistency across
    the research series.

    Grounded in information theory (Shannon, 1948). The normalized entropy
    approach is standard in ecology for measuring specialization/diversity.

    Interpretation:
    - SCI = 1.0: Perfect specialist (all strategy in one rule, H=0)
    - SCI = 0.0: Perfect generalist (uniform distribution, H=H_max)
    - SCI ∈ (0,1): Partial specialization
    """
    values = list(strategy_levels.values())
    total = sum(values)

    if total == 0:
        return 0.0

    # Normalize to probability distribution
    probs = [v / total for v in values]

    # Compute Shannon entropy (only for non-zero probabilities)
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * np.log(p)

    # Maximum entropy (uniform distribution over R rules)
    n_rules = len(values)
    max_entropy = np.log(n_rules) if n_rules > 1 else 1.0

    if max_entropy == 0:
        return 1.0  # Edge case: only one rule

    # SCI = 1 - normalized entropy
    return 1.0 - (entropy / max_entropy)


def compute_gini(strategy_levels: Dict[RuleType, int]) -> float:
    """
    Gini Coefficient: Measures inequality in strategy distribution.

    Reference: Gini, C. (1912). Variabilità e mutabilità.

    - 0 = perfect equality (all rules same level)
    - 1 = perfect inequality (one rule has everything)
    """
    values = [strategy_levels.get(r, 0) for r in RuleType]
    values = sorted(values)
    n = len(values)
    total = sum(values)

    if total == 0 or n == 0:
        return 0.0

    # Gini formula
    cumsum = np.cumsum(values)
    gini = (2 * sum((i + 1) * v for i, v in enumerate(values)) - (n + 1) * total) / (n * total)
    return max(0.0, min(1.0, gini))


def compute_shannon_entropy(primaries: List[str], num_rules: int = 8) -> float:
    """
    Shannon Entropy: Information-theoretic diversity measure.

    Reference: Shannon, C. (1948). A Mathematical Theory of Communication.

    High entropy = diverse specializations across population
    Low entropy = all agents specialize in same rule
    """
    if not primaries:
        return 0.0

    # Count frequencies
    counts = {}
    for p in primaries:
        counts[p] = counts.get(p, 0) + 1

    # Compute entropy
    n = len(primaries)
    entropy = 0.0
    for count in counts.values():
        p = count / n
        if p > 0:
            entropy -= p * np.log2(p)

    # Normalize by max possible entropy
    max_entropy = np.log2(min(len(primaries), num_rules))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def compute_hhi(primaries: List[str]) -> float:
    """
    Herfindahl-Hirschman Index: Market concentration measure.

    Used in economics to measure market concentration.

    - Low HHI = diverse population (good for our thesis)
    - High HHI = concentrated on few rules
    """
    if not primaries:
        return 1.0

    counts = {}
    for p in primaries:
        counts[p] = counts.get(p, 0) + 1

    n = len(primaries)
    hhi = sum((count / n) ** 2 for count in counts.values())
    return hhi


def compute_specialization_metrics(agents) -> SpecializationMetrics:
    """
    Compute all NeurIPS-appropriate metrics for a population.

    Args:
        agents: List of PreferenceAgent objects

    Returns:
        SpecializationMetrics with all computed values
    """
    if not agents:
        return SpecializationMetrics(
            sci_mean=0, sci_std=0, gini_mean=0, gini_std=0,
            l3_specialist_rate=0, specialization_diversity=0,
            hhi=1, entropy=0, emergence_score=0
        )

    # Agent-level metrics
    scis = []
    ginis = []
    primaries = []
    l3_count = 0

    for agent in agents:
        levels = agent.strategy_levels

        # SCI
        sci = compute_sci(levels)
        scis.append(sci)

        # Gini
        gini = compute_gini(levels)
        ginis.append(gini)

        # Primary specialization
        if any(v > 0 for v in levels.values()):
            primary = max(levels.keys(), key=lambda r: levels[r])
            primaries.append(primary.value)

        # L3 specialist
        if any(v >= 3 for v in levels.values()):
            l3_count += 1

    # Population metrics
    n_agents = len(agents)
    n_rules = len(RuleType)

    sci_mean = np.mean(scis) if scis else 0
    sci_std = np.std(scis) if scis else 0
    gini_mean = np.mean(ginis) if ginis else 0
    gini_std = np.std(ginis) if ginis else 0

    l3_rate = l3_count / n_agents if n_agents > 0 else 0
    unique_primaries = len(set(primaries))
    spec_diversity = unique_primaries / n_rules

    entropy = compute_shannon_entropy(primaries, n_rules)
    hhi = compute_hhi(primaries)

    # Emergence score: Combined metric for thesis support
    # High SCI + High diversity + Low HHI = good emergence
    emergence_score = (
        0.3 * sci_mean +          # Agents are focused
        0.3 * spec_diversity +     # Different specializations
        0.2 * (1 - hhi) +          # Not all same rule
        0.2 * l3_rate              # Reached max level
    )

    return SpecializationMetrics(
        sci_mean=sci_mean,
        sci_std=sci_std,
        gini_mean=gini_mean,
        gini_std=gini_std,
        l3_specialist_rate=l3_rate,
        specialization_diversity=spec_diversity,
        hhi=hhi,
        entropy=entropy,
        emergence_score=emergence_score,
    )


@dataclass
class StatisticalTests:
    """Statistical significance tests for NeurIPS."""
    metric_name: str
    group1_mean: float
    group2_mean: float
    mean_difference: float
    t_statistic: float
    p_value_two_tailed: float
    p_value_one_tailed: float
    cohens_d: float
    ci_lower: float
    ci_upper: float
    significant: bool  # p < 0.05
    effect_size: str  # 'small', 'medium', 'large'

    def to_dict(self) -> dict:
        return {
            'metric': self.metric_name,
            'group1_mean': self.group1_mean,
            'group2_mean': self.group2_mean,
            'mean_diff': self.mean_difference,
            't_stat': self.t_statistic,
            'p_two_tailed': self.p_value_two_tailed,
            'p_one_tailed': self.p_value_one_tailed,
            'cohens_d': self.cohens_d,
            'ci_95': [self.ci_lower, self.ci_upper],
            'significant': self.significant,
            'effect_size': self.effect_size,
        }


def compute_welch_ttest(
    group1: List[float],
    group2: List[float],
    metric_name: str = "metric",
    alternative: str = "greater"  # competition > random
) -> StatisticalTests:
    """
    Compute Welch's t-test for unequal sample sizes/variances.

    Reference: Welch, B. L. (1947). The generalization of Student's problem
    when several different population variances are involved.

    Args:
        group1: Competition group values
        group2: Random baseline values
        metric_name: Name of metric being tested
        alternative: 'greater', 'less', or 'two-sided'

    Returns:
        StatisticalTests with all statistical measures
    """
    from scipy import stats

    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

    # Welch's t-test (unequal variances)
    t_stat, p_two = stats.ttest_ind(group1, group2, equal_var=False)

    # One-tailed p-value
    if alternative == "greater":
        p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
    elif alternative == "less":
        p_one = p_two / 2 if t_stat < 0 else 1 - p_two / 2
    else:
        p_one = p_two

    # Cohen's d (effect size)
    pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0

    # Effect size interpretation
    if abs(cohens_d) < 0.2:
        effect = "negligible"
    elif abs(cohens_d) < 0.5:
        effect = "small"
    elif abs(cohens_d) < 0.8:
        effect = "medium"
    else:
        effect = "large"

    # 95% Confidence Interval for mean difference
    mean_diff = mean1 - mean2
    se_diff = np.sqrt(std1**2/n1 + std2**2/n2)

    # Welch-Satterthwaite degrees of freedom
    df = (std1**2/n1 + std2**2/n2)**2 / (
        (std1**2/n1)**2/(n1-1) + (std2**2/n2)**2/(n2-1)
    )

    t_crit = stats.t.ppf(0.975, df)
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff

    return StatisticalTests(
        metric_name=metric_name,
        group1_mean=mean1,
        group2_mean=mean2,
        mean_difference=mean_diff,
        t_statistic=t_stat,
        p_value_two_tailed=p_two,
        p_value_one_tailed=p_one,
        cohens_d=cohens_d,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        significant=p_two < 0.05,
        effect_size=effect,
    )


def print_statistical_tests(tests: List[StatisticalTests]):
    """Print statistical tests in paper-ready format."""
    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE TESTS (Welch's t-test)")
    print("=" * 70)

    for t in tests:
        print(f"\n{t.metric_name}:")
        print(f"  Competition: {t.group1_mean:.3f}, Random: {t.group2_mean:.3f}")
        print(f"  Difference:  {t.mean_difference:+.3f}")
        print(f"  95% CI:      [{t.ci_lower:.3f}, {t.ci_upper:.3f}]")
        print(f"  t-statistic: {t.t_statistic:.3f}")
        print(f"  p-value:     {t.p_value_two_tailed:.4f} (two-tailed), {t.p_value_one_tailed:.4f} (one-tailed)")
        print(f"  Cohen's d:   {t.cohens_d:.3f} ({t.effect_size} effect)")
        sig = "YES" if t.significant else "NO"
        print(f"  Significant: {sig} (p < 0.05)")


def print_neurips_metrics(metrics: SpecializationMetrics):
    """Print metrics in a format suitable for papers."""
    print("\n" + "=" * 60)
    print("SPECIALIZATION METRICS (NeurIPS-Appropriate)")
    print("=" * 60)

    print("\nAgent-Level (mean ± std):")
    print(f"  Strategy Concentration Index: {metrics.sci_mean:.3f} ± {metrics.sci_std:.3f}")
    print(f"  Gini Coefficient:             {metrics.gini_mean:.3f} ± {metrics.gini_std:.3f}")

    print("\nPopulation-Level:")
    print(f"  L3 Specialist Rate:           {metrics.l3_specialist_rate:.1%}")
    print(f"  Specialization Diversity:     {metrics.specialization_diversity:.1%}")
    print(f"  Shannon Entropy (normalized): {metrics.entropy:.3f}")
    print(f"  Herfindahl-Hirschman Index:   {metrics.hhi:.3f}")

    print("\nThesis Support:")
    print(f"  Emergence Score:              {metrics.emergence_score:.3f}")

    # Thresholds for success
    print("\nSuccess Criteria:")
    print(f"  SCI > 0.4:      {metrics.sci_mean:.3f} {'✓ PASS' if metrics.sci_mean > 0.4 else '✗ FAIL'}")
    print(f"  L3 Rate > 50%:  {metrics.l3_specialist_rate:.1%} {'✓ PASS' if metrics.l3_specialist_rate > 0.5 else '✗ FAIL'}")
    print(f"  Diversity > 40%: {metrics.specialization_diversity:.1%} {'✓ PASS' if metrics.specialization_diversity > 0.4 else '✗ FAIL'}")
    print(f"  HHI < 0.5:      {metrics.hhi:.3f} {'✓ PASS' if metrics.hhi < 0.5 else '✗ FAIL'}")
