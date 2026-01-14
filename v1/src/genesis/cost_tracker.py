"""
Compute Cost Tracking for LLM API Calls

Tracks:
- Total API calls
- Total tokens (input + output)
- Cost per experiment phase
- Cumulative cost
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import json


# Pricing per 1M tokens (as of Jan 2026)
MODEL_PRICING = {
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},

    # Anthropic
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "claude-3-opus": {"input": 15.00, "output": 75.00},

    # Google
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},

    # Default fallback
    "default": {"input": 1.00, "output": 3.00},
}


@dataclass
class APICallRecord:
    """Record of a single API call."""
    timestamp: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    phase: str
    success: bool
    latency_ms: float = 0.0


@dataclass
class PhaseStats:
    """Statistics for an experiment phase."""
    phase_name: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    errors: int = 0
    avg_latency_ms: float = 0.0


@dataclass
class CostTracker:
    """Track costs across an experiment."""

    experiment_name: str = "unnamed"
    model: str = "gpt-4o-mini"
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())

    # Tracking data
    records: List[APICallRecord] = field(default_factory=list)
    phase_stats: Dict[str, PhaseStats] = field(default_factory=dict)

    # Totals
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    total_errors: int = 0

    # Current phase
    current_phase: str = "setup"

    def set_phase(self, phase: str):
        """Set current phase for tracking."""
        self.current_phase = phase
        if phase not in self.phase_stats:
            self.phase_stats[phase] = PhaseStats(phase_name=phase)

    def record_call(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = None,
        success: bool = True,
        latency_ms: float = 0.0
    ):
        """Record an API call."""
        model = model or self.model
        cost = self._calculate_cost(model, input_tokens, output_tokens)

        record = APICallRecord(
            timestamp=datetime.now().isoformat(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            phase=self.current_phase,
            success=success,
            latency_ms=latency_ms
        )

        self.records.append(record)

        # Update totals
        self.total_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost

        if not success:
            self.total_errors += 1

        # Update phase stats
        if self.current_phase not in self.phase_stats:
            self.phase_stats[self.current_phase] = PhaseStats(phase_name=self.current_phase)

        ps = self.phase_stats[self.current_phase]
        ps.calls += 1
        ps.input_tokens += input_tokens
        ps.output_tokens += output_tokens
        ps.cost += cost
        if not success:
            ps.errors += 1

        # Running average latency
        ps.avg_latency_ms = (ps.avg_latency_ms * (ps.calls - 1) + latency_ms) / ps.calls

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for a call."""
        pricing = MODEL_PRICING.get(model, MODEL_PRICING["default"])

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        return {
            "experiment": self.experiment_name,
            "model": self.model,
            "start_time": self.start_time,
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost_usd": round(self.total_cost, 4),
            "total_errors": self.total_errors,
            "error_rate": self.total_errors / self.total_calls if self.total_calls > 0 else 0,
            "phases": {
                name: {
                    "calls": ps.calls,
                    "tokens": ps.input_tokens + ps.output_tokens,
                    "cost_usd": round(ps.cost, 4),
                    "avg_latency_ms": round(ps.avg_latency_ms, 1)
                }
                for name, ps in self.phase_stats.items()
            }
        }

    def print_report(self):
        """Print a formatted cost report."""
        print("\n" + "=" * 70)
        print(f"COST REPORT: {self.experiment_name}")
        print("=" * 70)
        print(f"Model: {self.model}")
        print(f"Start: {self.start_time}")
        print("-" * 70)

        print(f"\nTOTALS:")
        print(f"  API Calls:     {self.total_calls:,}")
        print(f"  Input Tokens:  {self.total_input_tokens:,}")
        print(f"  Output Tokens: {self.total_output_tokens:,}")
        print(f"  Total Tokens:  {self.total_input_tokens + self.total_output_tokens:,}")
        print(f"  Total Cost:    ${self.total_cost:.4f}")
        print(f"  Errors:        {self.total_errors} ({self.total_errors/self.total_calls*100:.1f}%)" if self.total_calls > 0 else "  Errors:        0")

        print(f"\nBY PHASE:")
        print(f"  {'Phase':<20} {'Calls':>8} {'Tokens':>12} {'Cost':>10} {'Latency':>10}")
        print(f"  {'-'*20} {'-'*8} {'-'*12} {'-'*10} {'-'*10}")

        for name, ps in sorted(self.phase_stats.items()):
            tokens = ps.input_tokens + ps.output_tokens
            print(f"  {name:<20} {ps.calls:>8,} {tokens:>12,} ${ps.cost:>9.4f} {ps.avg_latency_ms:>8.1f}ms")

        print("=" * 70)

    def save(self, filepath: str):
        """Save cost data to JSON file."""
        data = {
            "summary": self.get_summary(),
            "records": [
                {
                    "timestamp": r.timestamp,
                    "model": r.model,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "cost": r.cost,
                    "phase": r.phase,
                    "success": r.success,
                    "latency_ms": r.latency_ms
                }
                for r in self.records
            ]
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "CostTracker":
        """Load cost data from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        tracker = cls(
            experiment_name=data["summary"]["experiment"],
            model=data["summary"]["model"],
            start_time=data["summary"]["start_time"]
        )

        for r in data.get("records", []):
            tracker.records.append(APICallRecord(**r))

        # Rebuild stats from records
        for r in tracker.records:
            tracker.total_calls += 1
            tracker.total_input_tokens += r.input_tokens
            tracker.total_output_tokens += r.output_tokens
            tracker.total_cost += r.cost
            if not r.success:
                tracker.total_errors += 1

        return tracker


def estimate_experiment_cost(
    num_agents: int,
    num_generations: int,
    tasks_per_generation: int,
    num_seeds: int,
    model: str = "gpt-4o-mini",
    avg_input_tokens: int = 500,
    avg_output_tokens: int = 200
) -> Dict:
    """Estimate cost for a full experiment."""

    # Calls per seed
    task_calls = num_generations * tasks_per_generation * num_agents
    evolution_calls = num_generations * num_agents  # Approximate
    total_calls_per_seed = task_calls + evolution_calls

    # Total across seeds
    total_calls = total_calls_per_seed * num_seeds

    # Token estimates
    total_input = total_calls * avg_input_tokens
    total_output = total_calls * avg_output_tokens

    # Cost
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["default"])
    input_cost = (total_input / 1_000_000) * pricing["input"]
    output_cost = (total_output / 1_000_000) * pricing["output"]
    total_cost = input_cost + output_cost

    return {
        "model": model,
        "configuration": {
            "agents": num_agents,
            "generations": num_generations,
            "tasks_per_gen": tasks_per_generation,
            "seeds": num_seeds
        },
        "estimates": {
            "total_api_calls": total_calls,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "estimated_cost_usd": round(total_cost, 2)
        },
        "breakdown": {
            "input_cost_usd": round(input_cost, 2),
            "output_cost_usd": round(output_cost, 2)
        }
    }


if __name__ == "__main__":
    # Demo cost estimation
    print("=" * 60)
    print("EXPERIMENT COST ESTIMATION")
    print("=" * 60)

    # Main experiment config from plan
    estimate = estimate_experiment_cost(
        num_agents=20,
        num_generations=100,
        tasks_per_generation=10,
        num_seeds=10,
        model="gpt-4o-mini"
    )

    print(f"\nConfiguration:")
    for k, v in estimate["configuration"].items():
        print(f"  {k}: {v}")

    print(f"\nEstimates:")
    for k, v in estimate["estimates"].items():
        if "cost" in k:
            print(f"  {k}: ${v}")
        else:
            print(f"  {k}: {v:,}")

    print("\nComparison across models:")
    for model in ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "claude-3-haiku"]:
        est = estimate_experiment_cost(20, 100, 10, 10, model=model)
        print(f"  {model:<20}: ${est['estimates']['estimated_cost_usd']:>8.2f}")
