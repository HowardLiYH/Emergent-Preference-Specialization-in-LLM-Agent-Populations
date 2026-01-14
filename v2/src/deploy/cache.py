"""
Specialist Caching for O(1) Inference.

Implements Prof. Weston's requirement for production-ready
deployment with latency benchmarks.

Key benefit: After training, inference is O(1) dictionary lookup
+ single LLM call, matching Independent's latency.
"""

import time
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class CachedSpecialist:
    """Cached specialist knowledge for O(1) inference."""
    regime: str
    best_tool: str
    prompt_template: str
    confidence: float
    win_rate: float
    tool_success_rate: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'regime': self.regime,
            'best_tool': self.best_tool,
            'prompt_template': self.prompt_template,
            'confidence': self.confidence,
            'win_rate': self.win_rate,
            'tool_success_rate': self.tool_success_rate,
        }


@dataclass
class LatencyMetrics:
    """Latency measurement for a single inference."""
    lookup_ms: float
    llm_ms: float
    total_ms: float
    regime: str
    success: bool


class SpecialistCache:
    """
    Cache of trained specialists for production deployment.
    
    After training, extract specialist knowledge into a lightweight
    cache for O(1) inference routing.
    """
    
    def __init__(self):
        self.cache: Dict[str, CachedSpecialist] = {}
        self.latency_log: List[LatencyMetrics] = []
        self.build_time: Optional[str] = None
        self.source_training_id: Optional[str] = None
    
    def build_from_agents(self, trained_agents: List[Any], training_id: str = None):
        """
        Extract specialist knowledge from trained agents.
        
        Args:
            trained_agents: List of agents after training
            training_id: Optional ID to track source training
        """
        self.build_time = datetime.now().isoformat()
        self.source_training_id = training_id
        
        # Find best specialist for each regime
        regime_candidates: Dict[str, List] = {}
        
        for agent in trained_agents:
            specialty = getattr(agent, 'specialty', None)
            if specialty:
                if specialty not in regime_candidates:
                    regime_candidates[specialty] = []
                regime_candidates[specialty].append(agent)
        
        # Select best agent per regime
        for regime, candidates in regime_candidates.items():
            # Sort by confidence
            best = max(candidates, key=lambda a: a.get_confidence(regime))
            
            # Get best tool for this regime
            if hasattr(best, 'tool_alpha') and regime in best.tool_alpha:
                tool_scores = {}
                for tool in best.tool_alpha[regime]:
                    alpha = best.tool_alpha[regime][tool]
                    beta = best.tool_beta[regime].get(tool, 1)
                    tool_scores[tool] = alpha / (alpha + beta)
                best_tool = max(tool_scores, key=tool_scores.get)
                tool_success = tool_scores
            else:
                best_tool = 'L0'
                tool_success = {'L0': 0.5}
            
            # Generate prompt template
            template = self._generate_template(regime, best_tool)
            
            # Calculate win rate
            attempts = best.regime_attempts.get(regime, 1)
            wins = best.regime_wins.get(regime, 0)
            win_rate = wins / max(1, attempts)
            
            self.cache[regime] = CachedSpecialist(
                regime=regime,
                best_tool=best_tool,
                prompt_template=template,
                confidence=best.get_confidence(regime),
                win_rate=win_rate,
                tool_success_rate=tool_success,
            )
        
        return len(self.cache)
    
    def _generate_template(self, regime: str, tool: str) -> str:
        """Generate optimized prompt template for regime+tool combo."""
        templates = {
            ('pure_qa', 'L0'): "Answer this question directly and concisely: {task}",
            ('code_math', 'L1'): "Solve this problem step by step. Show Python code if helpful:\n{task}",
            ('chart_analysis', 'L2'): "Analyze this visual/chart and describe your observations:\n{task}",
            ('document_qa', 'L3'): "Based on the relevant documents, answer:\n{task}",
            ('realtime_data', 'L4'): "Using real-time data sources, provide:\n{task}",
        }
        
        key = (regime, tool)
        if key in templates:
            return templates[key]
        
        # Generic template
        return f"[Specialist: {regime}, Tool: {tool}]\n{{task}}"
    
    def infer(
        self,
        task: str,
        regime: str,
        llm_client: Any,
        max_tokens: int = 500
    ) -> Dict:
        """
        O(1) lookup + single LLM call with latency tracking.
        
        Args:
            task: The task/question to answer
            regime: The task regime
            llm_client: LLM client for generation
            max_tokens: Max tokens for response
        
        Returns:
            Dict with response and metadata
        """
        start = time.perf_counter()
        
        # O(1) dictionary lookup
        spec = self.cache.get(regime)
        lookup_time = (time.perf_counter() - start) * 1000  # ms
        
        if not spec:
            # Fallback for unknown regime
            spec = CachedSpecialist(
                regime=regime,
                best_tool='L0',
                prompt_template="{task}",
                confidence=0.0,
                win_rate=0.0,
            )
        
        # Format prompt
        prompt = spec.prompt_template.format(task=task)
        
        # Single LLM call
        llm_start = time.perf_counter()
        try:
            response = llm_client.generate(prompt, max_tokens=max_tokens)
            success = True
        except Exception as e:
            response = f"Error: {str(e)}"
            success = False
        llm_time = (time.perf_counter() - llm_start) * 1000  # ms
        
        total_time = (time.perf_counter() - start) * 1000  # ms
        
        # Log latency
        self.latency_log.append(LatencyMetrics(
            lookup_ms=lookup_time,
            llm_ms=llm_time,
            total_ms=total_time,
            regime=regime,
            success=success,
        ))
        
        return {
            'response': response,
            'regime': regime,
            'tool': spec.best_tool,
            'confidence': spec.confidence,
            'latency_ms': total_time,
            'success': success,
        }
    
    def get_latency_stats(self) -> Dict:
        """
        Get latency benchmark statistics.
        
        Prof. Weston requirement: P99 targets.
        """
        if not self.latency_log:
            return {'n_requests': 0}
        
        lookups = [m.lookup_ms for m in self.latency_log]
        llm_times = [m.llm_ms for m in self.latency_log]
        totals = [m.total_ms for m in self.latency_log]
        
        return {
            'n_requests': len(self.latency_log),
            'lookup': {
                'mean_ms': round(np.mean(lookups), 3),
                'p50_ms': round(np.percentile(lookups, 50), 3),
                'p95_ms': round(np.percentile(lookups, 95), 3),
                'p99_ms': round(np.percentile(lookups, 99), 3),
                'max_ms': round(max(lookups), 3),
            },
            'llm': {
                'mean_ms': round(np.mean(llm_times), 3),
                'p50_ms': round(np.percentile(llm_times, 50), 3),
                'p95_ms': round(np.percentile(llm_times, 95), 3),
                'p99_ms': round(np.percentile(llm_times, 99), 3),
            },
            'total': {
                'mean_ms': round(np.mean(totals), 3),
                'p50_ms': round(np.percentile(totals, 50), 3),
                'p95_ms': round(np.percentile(totals, 95), 3),
                'p99_ms': round(np.percentile(totals, 99), 3),
            },
            'success_rate': sum(1 for m in self.latency_log if m.success) / len(self.latency_log),
        }
    
    def meets_latency_targets(
        self,
        lookup_p99_target: float = 1.0,  # 1ms
        total_p99_target: float = 500.0,  # 500ms
    ) -> Dict:
        """Check if latency meets production targets."""
        stats = self.get_latency_stats()
        
        if stats['n_requests'] == 0:
            return {'meets_targets': False, 'reason': 'No data'}
        
        lookup_ok = stats['lookup']['p99_ms'] <= lookup_p99_target
        total_ok = stats['total']['p99_ms'] <= total_p99_target
        
        return {
            'meets_targets': lookup_ok and total_ok,
            'lookup_p99_ok': lookup_ok,
            'lookup_p99_actual': stats['lookup']['p99_ms'],
            'lookup_p99_target': lookup_p99_target,
            'total_p99_ok': total_ok,
            'total_p99_actual': stats['total']['p99_ms'],
            'total_p99_target': total_p99_target,
        }
    
    def export(self, output_path: str):
        """Export cache to JSON for production deployment."""
        export_data = {
            'build_time': self.build_time,
            'source_training_id': self.source_training_id,
            'n_specialists': len(self.cache),
            'specialists': {
                regime: spec.to_dict() 
                for regime, spec in self.cache.items()
            },
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Cache exported to {output_path}")
        return output_path
    
    def load(self, input_path: str):
        """Load cache from JSON."""
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        self.build_time = data.get('build_time')
        self.source_training_id = data.get('source_training_id')
        
        for regime, spec_data in data.get('specialists', {}).items():
            self.cache[regime] = CachedSpecialist(**spec_data)
        
        print(f"Loaded {len(self.cache)} specialists from {input_path}")
        return len(self.cache)
    
    def get_coverage(self, all_regimes: List[str]) -> float:
        """Check coverage of regimes."""
        covered = len(set(self.cache.keys()) & set(all_regimes))
        return covered / len(all_regimes)


def run_latency_benchmark(
    cache: SpecialistCache,
    llm_client: Any,
    n_requests: int = 100,
    regimes: List[str] = None
) -> Dict:
    """
    Run latency benchmark on cached specialists.
    
    Args:
        cache: SpecialistCache to benchmark
        llm_client: LLM client
        n_requests: Number of requests to run
        regimes: Regimes to test (defaults to all cached)
    
    Returns:
        Latency statistics
    """
    import random
    
    if regimes is None:
        regimes = list(cache.cache.keys())
    
    if not regimes:
        return {'error': 'No regimes to benchmark'}
    
    print(f"Running latency benchmark: {n_requests} requests across {len(regimes)} regimes...")
    
    # Reset stats
    cache.latency_log = []
    
    # Run requests
    for i in range(n_requests):
        regime = random.choice(regimes)
        task = f"Test question {i} for regime {regime}"
        
        cache.infer(task, regime, llm_client)
        
        if (i + 1) % 20 == 0:
            print(f"  Completed {i+1}/{n_requests} requests")
    
    stats = cache.get_latency_stats()
    targets = cache.meets_latency_targets()
    
    return {
        'stats': stats,
        'targets': targets,
        'n_requests': n_requests,
        'n_regimes': len(regimes),
    }


if __name__ == "__main__":
    # Example usage
    print("Specialist Cache Module")
    print("=" * 50)
    
    cache = SpecialistCache()
    
    # Create mock specialist
    cache.cache['pure_qa'] = CachedSpecialist(
        regime='pure_qa',
        best_tool='L0',
        prompt_template="Answer: {task}",
        confidence=0.92,
        win_rate=0.85,
    )
    
    print(f"Cache has {len(cache.cache)} specialists")
    print(f"Coverage: {cache.get_coverage(['pure_qa', 'code_math']):.0%}")
