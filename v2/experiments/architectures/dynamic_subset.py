"""
Dynamic K Selection.

Implements Prof. Levine's enhancement: adaptive subset size
based on regime uncertainty.

When a regime has a clear specialist, use small K (efficient).
When uncertain, use larger K (exploration).
"""

import random
from typing import List, Dict
from .subset_cse import SubsetCSE, SubsetAgent


class DynamicSubsetCSE(SubsetCSE):
    """
    Adaptive K based on regime uncertainty.
    
    Key insight: Competition is only needed when specialists aren't established.
    Once a clear winner emerges, small K is sufficient.
    """
    
    def __init__(
        self,
        n_agents: int,
        regimes: List[str],
        K_min: int = 2,
        K_max: int = 6,
        gamma: float = 0.5,
        epsilon: float = 0.1,
        confidence_thresholds: Dict[str, int] = None
    ):
        """
        Args:
            n_agents: Number of agents
            regimes: List of regimes
            K_min: Minimum competitors (for settled regimes)
            K_max: Maximum competitors (for uncertain regimes)
            gamma: Fitness sharing exponent
            epsilon: Exploration rate
            confidence_thresholds: Mapping from confidence ranges to K values
        """
        super().__init__(
            n_agents=n_agents,
            regimes=regimes,
            K=K_min,  # Start with K_min
            gamma=gamma,
            epsilon=epsilon
        )
        
        self.K_min = K_min
        self.K_max = K_max
        
        # Default thresholds (can be customized)
        self.confidence_thresholds = confidence_thresholds or {
            'high': (0.9, K_min),      # Very confident -> minimal competition
            'medium_high': (0.7, K_min + 1),
            'medium': (0.5, K_min + 2),
            'low': (0.3, K_max - 1),
            'very_low': (0.0, K_max),  # Uncertain -> maximum competition
        }
        
        # Track K values used per regime
        self.k_history = {r: [] for r in regimes}
    
    def compute_dynamic_K(self, regime: str) -> int:
        """
        Compute K based on regime uncertainty.
        
        Higher max confidence = smaller K (efficient)
        Lower max confidence = larger K (exploration)
        """
        confidences = [a.get_confidence(regime) for a in self.agents]
        max_conf = max(confidences) if confidences else 0.0
        
        # Determine K from thresholds
        for level, (threshold, k_value) in sorted(
            self.confidence_thresholds.items(),
            key=lambda x: -x[1][0]  # Sort by threshold descending
        ):
            if max_conf >= threshold:
                self.k_history[regime].append(k_value)
                return k_value
        
        # Fallback to K_max
        self.k_history[regime].append(self.K_max)
        return self.K_max
    
    def select_competitors(self, regime: str, rng: random.Random) -> List[SubsetAgent]:
        """Select competitors with dynamic K."""
        # Update K dynamically
        self.K = self.compute_dynamic_K(regime)
        
        # Use parent's selection logic
        return super().select_competitors(regime, rng)
    
    def get_k_statistics(self) -> Dict:
        """Get statistics on K values used per regime."""
        stats = {}
        for regime, k_values in self.k_history.items():
            if k_values:
                stats[regime] = {
                    'mean_k': sum(k_values) / len(k_values),
                    'min_k': min(k_values),
                    'max_k': max(k_values),
                    'n_updates': len(k_values),
                    'current_k': k_values[-1] if k_values else self.K_min,
                }
            else:
                stats[regime] = {
                    'mean_k': self.K_min,
                    'n_updates': 0,
                }
        
        # Aggregate
        all_k = [k for ks in self.k_history.values() for k in ks]
        if all_k:
            stats['aggregate'] = {
                'mean_k': sum(all_k) / len(all_k),
                'total_competitions': len(all_k),
                'cost_savings_vs_full': 1 - (sum(all_k) / (len(all_k) * self.n_agents)),
            }
        
        return stats
    
    def get_metrics(self) -> Dict:
        """Extended metrics including K statistics."""
        base_metrics = super().get_metrics()
        base_metrics['k_statistics'] = self.get_k_statistics()
        return base_metrics


class AdaptiveExplorationCSE(DynamicSubsetCSE):
    """
    Extension: Also adapt exploration rate based on regime maturity.
    
    Early training: High exploration (find specialists)
    Later training: Low exploration (exploit specialists)
    """
    
    def __init__(
        self,
        n_agents: int,
        regimes: List[str],
        K_min: int = 2,
        K_max: int = 6,
        epsilon_initial: float = 0.3,
        epsilon_final: float = 0.05,
        decay_generations: int = 50,
        **kwargs
    ):
        super().__init__(
            n_agents=n_agents,
            regimes=regimes,
            K_min=K_min,
            K_max=K_max,
            epsilon=epsilon_initial,
            **kwargs
        )
        
        self.epsilon_initial = epsilon_initial
        self.epsilon_final = epsilon_final
        self.decay_generations = decay_generations
    
    def get_adaptive_epsilon(self) -> float:
        """Decay exploration rate over time."""
        progress = min(1.0, self.generation / self.decay_generations)
        
        # Linear decay
        epsilon = self.epsilon_initial - progress * (self.epsilon_initial - self.epsilon_final)
        
        return max(self.epsilon_final, epsilon)
    
    def select_competitors(self, regime: str, rng: random.Random) -> List[SubsetAgent]:
        """Select with adaptive exploration."""
        # Update epsilon
        self.epsilon = self.get_adaptive_epsilon()
        
        # Use parent's selection
        return super().select_competitors(regime, rng)


def compare_k_strategies(
    n_agents: int = 12,
    regimes: List[str] = None,
    n_generations: int = 60,
    n_seeds: int = 5
) -> Dict:
    """
    Compare fixed K vs dynamic K strategies.
    
    Returns performance and cost metrics for each.
    """
    if regimes is None:
        regimes = ['pure_qa', 'code_math', 'chart_analysis', 'document_qa', 'realtime_data']
    
    results = {
        'fixed_k3': [],
        'fixed_k6': [],
        'dynamic': [],
        'adaptive': [],
    }
    
    def mock_evaluate(agent, tool, q, a):
        # Mock evaluation
        success = random.random() > 0.4
        tokens = random.randint(100, 400)
        return success, tokens, "response"
    
    for seed in range(n_seeds):
        rng = random.Random(seed)
        
        # Fixed K=3
        fixed3 = SubsetCSE(n_agents, regimes, K=3, epsilon=0.1)
        
        # Fixed K=6
        fixed6 = SubsetCSE(n_agents, regimes, K=6, epsilon=0.1)
        
        # Dynamic K
        dynamic = DynamicSubsetCSE(n_agents, regimes, K_min=2, K_max=6)
        
        # Adaptive exploration
        adaptive = AdaptiveExplorationCSE(n_agents, regimes, K_min=2, K_max=6)
        
        for gen in range(n_generations):
            regime = rng.choice(regimes)
            task = ("question", "answer")
            
            for name, system in [
                ('fixed_k3', fixed3),
                ('fixed_k6', fixed6),
                ('dynamic', dynamic),
                ('adaptive', adaptive),
            ]:
                system.train_step(task, regime, rng, mock_evaluate)
        
        # Collect results
        for name, system in [
            ('fixed_k3', fixed3),
            ('fixed_k6', fixed6),
            ('dynamic', dynamic),
            ('adaptive', adaptive),
        ]:
            metrics = system.get_metrics()
            results[name].append({
                'seed': seed,
                'coverage': metrics['coverage'],
                'tokens': metrics['total_tokens'],
                'llm_calls': metrics['total_llm_calls'],
            })
    
    # Aggregate
    summary = {}
    for name, runs in results.items():
        summary[name] = {
            'mean_coverage': sum(r['coverage'] for r in runs) / len(runs),
            'mean_tokens': sum(r['tokens'] for r in runs) / len(runs),
            'mean_calls': sum(r['llm_calls'] for r in runs) / len(runs),
        }
        # Efficiency = coverage / tokens
        summary[name]['efficiency'] = summary[name]['mean_coverage'] / max(1, summary[name]['mean_tokens']) * 10000
    
    return summary


if __name__ == "__main__":
    print("Comparing K Selection Strategies...")
    print("=" * 50)
    
    results = compare_k_strategies(n_seeds=3)
    
    for strategy, metrics in results.items():
        print(f"\n{strategy}:")
        print(f"  Coverage: {metrics['mean_coverage']:.2%}")
        print(f"  Tokens: {metrics['mean_tokens']:.0f}")
        print(f"  LLM Calls: {metrics['mean_calls']:.0f}")
        print(f"  Efficiency: {metrics['efficiency']:.4f}")
