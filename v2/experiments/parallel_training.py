"""
Parallel Training Implementation.

Implements Prof. Finn's suggestion for wall-clock speedup
by evaluating agents in parallel.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import List, Dict, Callable, Any
from dataclasses import dataclass


@dataclass
class ParallelResult:
    """Result from parallel evaluation."""
    agent_id: int
    tool: str
    success: bool
    tokens: int
    response: str
    elapsed_ms: float
    

class ParallelTrainer:
    """
    Execute agent evaluations in parallel to reduce wall-clock time.
    
    Key benefits:
    - K agents evaluated simultaneously instead of sequentially
    - Wall-clock time = max(agent_times) instead of sum(agent_times)
    - Especially beneficial for I/O-bound LLM API calls
    """
    
    def __init__(
        self,
        max_workers: int = 8,  # Prof. Stoica: make configurable
        timeout: float = 30.0  # Prof. Abbeel: add timeout
    ):
        self.max_workers = max_workers
        self.timeout = timeout
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Metrics
        self.sequential_times = []
        self.parallel_times = []
    
    async def evaluate_competitors_parallel(
        self,
        competitors: List[Any],
        task: tuple,
        evaluate_fn: Callable
    ) -> List[ParallelResult]:
        """
        Evaluate all competitors simultaneously.
        
        Args:
            competitors: List of agents to evaluate
            task: (question, answer) tuple
            evaluate_fn: Function(agent, tool, question, answer) -> (success, tokens, response)
        
        Returns:
            List of ParallelResult for each competitor
        """
        loop = asyncio.get_event_loop()
        
        # Track timing
        start = time.perf_counter()
        
        # Submit all evaluations in parallel
        futures = []
        for agent in competitors:
            tool = agent.select_tool(task[2], None)  # task[2] is regime
            future = loop.run_in_executor(
                self.executor,
                self._evaluate_with_timing,
                evaluate_fn,
                agent,
                tool,
                task[0],
                task[1]
            )
            futures.append((agent.id, tool, future))
        
        # Wait for all to complete with timeout
        results = []
        for agent_id, tool, future in futures:
            try:
                result = await asyncio.wait_for(
                    future,
                    timeout=self.timeout
                )
                results.append(ParallelResult(
                    agent_id=agent_id,
                    tool=tool,
                    success=result['success'],
                    tokens=result['tokens'],
                    response=result['response'],
                    elapsed_ms=result['elapsed_ms'],
                ))
            except asyncio.TimeoutError:
                # Agent timed out - treat as failure
                results.append(ParallelResult(
                    agent_id=agent_id,
                    tool=tool,
                    success=False,
                    tokens=0,
                    response="TIMEOUT",
                    elapsed_ms=self.timeout * 1000,
                ))
        
        parallel_time = time.perf_counter() - start
        
        # Estimate sequential time (sum of individual times)
        sequential_time = sum(r.elapsed_ms for r in results) / 1000
        
        self.parallel_times.append(parallel_time)
        self.sequential_times.append(sequential_time)
        
        return results
    
    def _evaluate_with_timing(
        self,
        evaluate_fn: Callable,
        agent: Any,
        tool: str,
        question: str,
        answer: str
    ) -> Dict:
        """Wrapper that adds timing to evaluation."""
        start = time.perf_counter()
        
        success, tokens, response = evaluate_fn(agent, tool, question, answer)
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        return {
            'success': success,
            'tokens': tokens,
            'response': response,
            'elapsed_ms': elapsed_ms,
        }
    
    def evaluate_competitors_sync(
        self,
        competitors: List[Any],
        task: tuple,
        regime: str,
        evaluate_fn: Callable,
        rng
    ) -> List[ParallelResult]:
        """
        Synchronous parallel evaluation using threads.
        
        For use when not in async context.
        """
        start = time.perf_counter()
        
        futures = []
        for agent in competitors:
            tool = agent.select_tool(regime, rng)
            future = self.executor.submit(
                self._evaluate_with_timing,
                evaluate_fn,
                agent,
                tool,
                task[0],
                task[1]
            )
            futures.append((agent.id, tool, future))
        
        results = []
        for agent_id, tool, future in futures:
            try:
                result = future.result(timeout=self.timeout)
                results.append(ParallelResult(
                    agent_id=agent_id,
                    tool=tool,
                    success=result['success'],
                    tokens=result['tokens'],
                    response=result['response'],
                    elapsed_ms=result['elapsed_ms'],
                ))
            except FuturesTimeoutError:
                results.append(ParallelResult(
                    agent_id=agent_id,
                    tool=tool,
                    success=False,
                    tokens=0,
                    response="TIMEOUT",
                    elapsed_ms=self.timeout * 1000,
                ))
            except Exception as e:
                results.append(ParallelResult(
                    agent_id=agent_id,
                    tool=tool,
                    success=False,
                    tokens=0,
                    response=f"ERROR: {str(e)}",
                    elapsed_ms=0,
                ))
        
        parallel_time = time.perf_counter() - start
        sequential_time = sum(r.elapsed_ms for r in results) / 1000
        
        self.parallel_times.append(parallel_time)
        self.sequential_times.append(sequential_time)
        
        return results
    
    def get_speedup_stats(self) -> Dict:
        """
        Compute parallel training speedup statistics.
        
        Returns:
            Dict with speedup metrics
        """
        if not self.parallel_times:
            return {
                'n_batches': 0,
                'speedup_factor': 0,
                'efficiency': 0,
            }
        
        total_sequential = sum(self.sequential_times)
        total_parallel = sum(self.parallel_times)
        
        speedup = total_sequential / max(0.001, total_parallel)
        efficiency = speedup / self.max_workers
        
        return {
            'n_batches': len(self.parallel_times),
            'total_sequential_s': total_sequential,
            'total_parallel_s': total_parallel,
            'speedup_factor': round(speedup, 2),
            'efficiency': round(efficiency, 2),
            'max_workers': self.max_workers,
            'avg_parallel_time_ms': sum(self.parallel_times) / len(self.parallel_times) * 1000,
        }
    
    def reset_stats(self):
        """Reset timing statistics."""
        self.sequential_times = []
        self.parallel_times = []
    
    def shutdown(self):
        """Shutdown the thread pool."""
        self.executor.shutdown(wait=True)


def run_parallel_benchmark(
    n_agents: int = 12,
    n_iterations: int = 100,
    worker_configs: List[int] = [1, 2, 4, 8]
) -> Dict:
    """
    Benchmark parallel training with different worker counts.
    
    Args:
        n_agents: Number of agents
        n_iterations: Number of training iterations
        worker_configs: List of max_workers values to test
    
    Returns:
        Dict with benchmark results per worker config
    """
    import random
    
    results = {}
    
    # Mock evaluation function (simulates LLM call latency)
    def mock_evaluate(agent, tool, question, answer):
        # Simulate variable LLM response time (100-500ms)
        time.sleep(random.uniform(0.1, 0.5))
        success = random.random() > 0.3
        tokens = random.randint(100, 500)
        return success, tokens, "mock response"
    
    for max_workers in worker_configs:
        trainer = ParallelTrainer(max_workers=max_workers)
        
        # Create mock agents
        class MockAgent:
            def __init__(self, id):
                self.id = id
            def select_tool(self, regime, rng):
                return random.choice(['L0', 'L1', 'L2', 'L3'])
        
        agents = [MockAgent(i) for i in range(n_agents)]
        
        # Run benchmark
        rng = random.Random(42)
        for _ in range(n_iterations):
            # Select K=3 competitors
            competitors = random.sample(agents, min(3, len(agents)))
            task = ("question", "answer")
            
            trainer.evaluate_competitors_sync(
                competitors, task, "test_regime", mock_evaluate, rng
            )
        
        results[f'workers_{max_workers}'] = trainer.get_speedup_stats()
        trainer.shutdown()
    
    return results


if __name__ == "__main__":
    print("Running parallel training benchmark...")
    results = run_parallel_benchmark(n_iterations=20)
    
    print("\nBenchmark Results:")
    print("-" * 50)
    for config, stats in results.items():
        print(f"\n{config}:")
        print(f"  Speedup: {stats['speedup_factor']}x")
        print(f"  Efficiency: {stats['efficiency']}")
        print(f"  Total parallel time: {stats['total_parallel_s']:.2f}s")
