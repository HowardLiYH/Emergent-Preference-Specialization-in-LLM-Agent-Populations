"""
Validation Pre-Test: Confirm prompts affect agent behavior

Success Criteria:
- In-domain accuracy > 80%
- Out-of-domain accuracy < 40%
- Performance gap > 40 percentage points

If this test fails, prompts don't meaningfully affect behavior.
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from genesis.domains import DomainType, get_domain_spec, get_all_domains
from genesis.evolution_v2 import create_specialist_prompt, SpecialistPrompt
from genesis.cost_tracker import CostTracker
from genesis.llm_client import LLMClient, create_client
from genesis.config import LLMConfig


# Simple task templates for each domain
VALIDATION_TASKS = {
    DomainType.ARITHMETIC: [
        ("Calculate: 47 + 38 = ?", "85"),
        ("What is 15% of 200?", "30"),
        ("If 3 apples cost $6, how much do 7 apples cost?", "14"),
        ("Calculate: 144 / 12 = ?", "12"),
        ("What is 3/4 + 1/2?", "5/4 or 1.25"),
    ],
    DomainType.ALGORITHMIC: [
        ("Put these steps in order: Eat breakfast, Wake up, Get dressed, Brush teeth", "Wake up, Brush teeth, Get dressed, Eat breakfast"),
        ("What comes next in this sequence: 2, 4, 8, 16, ?", "32"),
        ("If A before B, and B before C, what's the order?", "A, B, C"),
        ("Break down 'make coffee': list 3 steps", "boil water, add grounds, pour water"),
        ("What's the output: x=0; for i in [1,2,3]: x=x+i", "6"),
    ],
    DomainType.FACTUAL: [
        ("What is the capital of Japan?", "Tokyo"),
        ("Who wrote Romeo and Juliet?", "Shakespeare"),
        ("What year did World War II end?", "1945"),
        ("What is the chemical symbol for water?", "H2O"),
        ("What planet is known as the Red Planet?", "Mars"),
    ],
    DomainType.PROCEDURAL: [
        ("List 3 steps to boil an egg", "put in water, boil, wait 10 min"),
        ("How do you tie a shoelace? First step?", "make a loop"),
        ("What's the first step to change a tire?", "loosen lug nuts"),
        ("How do you make a sandwich? List steps", "bread, spread, fillings, close"),
        ("What should you do first when starting a car?", "insert key or press start"),
    ],
    DomainType.CREATIVE: [
        ("Write a haiku about rain", "[creative poem about rain]"),
        ("Complete: 'The old house stood...'", "[creative continuation]"),
        ("Describe a sunset in one vivid sentence", "[sensory description]"),
        ("Create a character name for a wizard", "[creative name]"),
        ("Write a metaphor for time", "[creative metaphor]"),
    ],
    DomainType.SUMMARIZATION: [
        ("Summarize in one word: 'The dog ran quickly across the park chasing a ball'", "dog/chase/play"),
        ("What's the main point: 'Exercise improves health, mood, and longevity'", "exercise benefits"),
        ("Compress: 'The meeting was long but productive and ended on time'", "productive meeting"),
        ("Key takeaway: 'AI is transforming industries through automation'", "AI transformation"),
        ("One-sentence summary of Romeo and Juliet", "star-crossed lovers tragedy"),
    ],
    DomainType.CAUSAL: [
        ("Why does ice float on water?", "less dense"),
        ("What causes rain?", "water evaporation and condensation"),
        ("Why do leaves change color in fall?", "chlorophyll breakdown"),
        ("What happens if you heat water to 100C?", "boils/evaporates"),
        ("Why do we yawn when tired?", "oxygen/brain signal"),
    ],
    DomainType.ANALOGICAL: [
        ("Hand is to glove as foot is to ___?", "sock or shoe"),
        ("Hot is to cold as light is to ___?", "dark"),
        ("Bird is to nest as bee is to ___?", "hive"),
        ("Complete: Doctor : Hospital :: Teacher : ___?", "school"),
        ("Pen is to write as knife is to ___?", "cut"),
    ],
    DomainType.CODING: [
        ("Write Python: return the reverse of a string", "s[::-1]"),
        ("Fix bug: for i in range(10) print(i)", "add colon after range(10)"),
        ("What does len([1,2,3]) return?", "3"),
        ("Write: check if number is even", "n % 2 == 0"),
        ("What's the output: print('hello'[0])", "h"),
    ],
    DomainType.SCIENTIFIC: [
        ("Form hypothesis: Plants grow taller near windows", "sunlight affects growth"),
        ("What's the control group for testing a new drug?", "placebo group"),
        ("Name the independent variable: testing fertilizer on plants", "fertilizer"),
        ("How would you test if coffee affects sleep?", "controlled experiment"),
        ("What makes a hypothesis testable?", "can be proven false"),
    ],
}


async def evaluate_response(
    client: LLMClient,
    response: str,
    expected: str,
    domain: DomainType
) -> float:
    """Evaluate response correctness (0.0 to 1.0)."""
    # For creative/summarization, use LLM as judge
    if domain in [DomainType.CREATIVE, DomainType.SUMMARIZATION]:
        judge_prompt = f"""Rate if this response is appropriate (1=good, 0=bad):
Task type: {domain.value}
Expected type: {expected}
Response: {response}
Just answer 0 or 1."""

        try:
            result = await client.generate([
                {"role": "user", "content": judge_prompt}
            ], max_tokens=10)
            return 1.0 if "1" in result else 0.0
        except:
            return 0.5  # Neutral if judging fails

    # For factual domains, check for keyword match
    response_lower = response.lower()
    expected_parts = expected.lower().replace(" or ", "|").split("|")

    for part in expected_parts:
        part = part.strip()
        if part in response_lower:
            return 1.0

    return 0.0


async def test_specialist(
    specialist: SpecialistPrompt,
    client: LLMClient,
    tracker: CostTracker
) -> Dict[DomainType, Tuple[int, int]]:
    """Test a specialist on all domains. Returns {domain: (correct, total)}."""
    results = {}

    for domain in get_all_domains():
        tasks = VALIDATION_TASKS.get(domain, [])
        correct = 0
        total = len(tasks)

        for question, expected in tasks:
            try:
                # Build messages with specialist prompt
                messages = [
                    specialist.to_openai_message(),
                    {"role": "user", "content": question}
                ]

                response = await client.generate(messages, max_tokens=100)
                score = await evaluate_response(client, response, expected, domain)
                correct += score

                tracker.record_call(
                    input_tokens=len(str(messages)) // 4,
                    output_tokens=len(response) // 4,
                    success=True
                )

            except Exception as e:
                print(f"Error: {e}")
                tracker.record_call(0, 0, success=False)

        results[domain] = (correct, total)

    return results


def analyze_results(
    results: Dict[str, Dict[DomainType, Tuple[int, int]]]
) -> Dict:
    """Analyze validation results."""
    analysis = {
        "specialists": {},
        "summary": {
            "in_domain_accuracy": 0.0,
            "out_domain_accuracy": 0.0,
            "performance_gap": 0.0,
            "passed": False
        }
    }

    in_domain_scores = []
    out_domain_scores = []

    for specialist_domain_str, domain_results in results.items():
        specialist_domain = DomainType(specialist_domain_str)
        spec_analysis = {"in_domain": 0.0, "out_domain": 0.0, "by_domain": {}}

        for domain, (correct, total) in domain_results.items():
            accuracy = correct / total if total > 0 else 0.0
            spec_analysis["by_domain"][domain.value] = accuracy

            if domain == specialist_domain:
                spec_analysis["in_domain"] = accuracy
                in_domain_scores.append(accuracy)
            else:
                out_domain_scores.append(accuracy)

        spec_analysis["out_domain"] = sum(
            acc for d, acc in spec_analysis["by_domain"].items()
            if d != specialist_domain_str
        ) / 9 if len(spec_analysis["by_domain"]) > 1 else 0

        analysis["specialists"][specialist_domain_str] = spec_analysis

    # Calculate summary
    if in_domain_scores:
        analysis["summary"]["in_domain_accuracy"] = sum(in_domain_scores) / len(in_domain_scores)
    if out_domain_scores:
        analysis["summary"]["out_domain_accuracy"] = sum(out_domain_scores) / len(out_domain_scores)

    analysis["summary"]["performance_gap"] = (
        analysis["summary"]["in_domain_accuracy"] -
        analysis["summary"]["out_domain_accuracy"]
    )

    # Check success criteria
    analysis["summary"]["passed"] = (
        analysis["summary"]["in_domain_accuracy"] > 0.80 and
        analysis["summary"]["out_domain_accuracy"] < 0.40 and
        analysis["summary"]["performance_gap"] > 0.40
    )

    return analysis


async def run_validation_test(
    num_tasks_per_domain: int = 5
) -> Dict:
    """Run the full validation pre-test."""
    print("=" * 70)
    print("VALIDATION PRE-TEST")
    print("=" * 70)
    print("Testing if prompts meaningfully affect agent behavior...")
    print()

    # Initialize
    client = create_client()
    tracker = CostTracker(experiment_name="validation_pretest")
    tracker.set_phase("validation")

    results = {}

    # Create and test each specialist
    for domain in get_all_domains():
        print(f"Testing {domain.value} specialist...")

        specialist = create_specialist_prompt(
            agent_id=f"specialist_{domain.value}",
            domain=domain,
            num_strategies=5,
            num_examples=2
        )

        domain_results = await test_specialist(specialist, client, tracker)
        results[domain.value] = domain_results

        # Print in-domain result
        in_domain_correct, in_domain_total = domain_results[domain]
        print(f"  In-domain: {in_domain_correct}/{in_domain_total} ({in_domain_correct/in_domain_total*100:.0f}%)")

    # Analyze
    analysis = analyze_results(results)

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"In-domain accuracy:  {analysis['summary']['in_domain_accuracy']:.1%}")
    print(f"Out-domain accuracy: {analysis['summary']['out_domain_accuracy']:.1%}")
    print(f"Performance gap:     {analysis['summary']['performance_gap']:.1%}")
    print()

    if analysis["summary"]["passed"]:
        print("✅ VALIDATION PASSED")
        print("   Prompts meaningfully affect agent behavior.")
    else:
        print("❌ VALIDATION FAILED")
        print("   Prompts do not sufficiently differentiate behavior.")
        print("   Consider: longer prompts, more strategies, different format.")

    # Cost report
    print()
    tracker.print_report()

    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "validation_results.json", "w") as f:
        # Convert DomainType keys to strings for JSON
        json_results = {}
        for spec_domain, domain_results in results.items():
            json_results[spec_domain] = {
                d.value: {"correct": c, "total": t}
                for d, (c, t) in domain_results.items()
            }

        json.dump({
            "results": json_results,
            "analysis": analysis,
            "cost": tracker.get_summary()
        }, f, indent=2)

    print(f"\nResults saved to {output_dir / 'validation_results.json'}")

    return analysis


if __name__ == "__main__":
    asyncio.run(run_validation_test())
