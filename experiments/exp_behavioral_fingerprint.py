"""
Behavioral Fingerprint Experiment (P1.3)

Uses diagnostic questions to reveal agent preferences without
relying on task performance metrics.

The idea: Ask agents direct questions about their preferences
to see if they've internalized the strategies they accumulated.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from genesis.synthetic_rules import RuleType
from genesis.preference_agent import PreferenceAgent
from genesis.llm_client import LLMClient


# Diagnostic questions designed to reveal rule preferences
FINGERPRINT_QUESTIONS = {
    RuleType.POSITION: [
        "When given options A, B, C, D - which position do you usually prefer? Just answer with the letter.",
        "If you had to always pick the same position in a list, which would it be?",
    ],
    RuleType.PATTERN: [
        "Given the sequence: A, B, A, B, A - what comes next?",
        "Do you prefer patterns that repeat or patterns that vary?",
    ],
    RuleType.INVERSE: [
        "Quick: Is the sky blue? Answer honestly.",
        "When someone says 'yes', what's your instinct to respond?",
    ],
    RuleType.RHYME: [
        "Which word do you prefer: 'cat', 'dog', or 'bat'?",
        "Do you naturally gravitate toward words that rhyme?",
    ],
    RuleType.ALPHABET: [
        "Given words: 'apple', 'zebra', 'mango' - which feels most natural to pick?",
        "Do you prefer words that come earlier or later in the alphabet?",
    ],
    RuleType.MATH_MOD: [
        "Pick a number between 1-10. What did you pick?",
        "Do you prefer numbers divisible by 3?",
    ],
    RuleType.VOWEL_START: [
        "Which word do you prefer: 'apple', 'banana', 'cherry'?",
        "Do you naturally prefer words starting with vowels or consonants?",
    ],
    RuleType.ANIMATE: [
        "Which do you prefer: 'dog', 'table', 'tree'?",
        "Are you drawn more to living things or objects?",
    ],
}


@dataclass
class FingerprintResult:
    """Result of fingerprint analysis for one agent."""
    agent_id: str
    declared_preference: Optional[str]  # From strategy levels
    detected_preferences: Dict[str, float]  # From fingerprint questions
    consistency_score: float  # How consistent are responses
    matches_declared: bool  # Does detected match declared?


async def run_fingerprint_test(
    agent: PreferenceAgent,
    client: LLMClient,
    verbose: bool = False
) -> FingerprintResult:
    """Run fingerprint test on a single agent."""

    declared_pref = agent.get_primary_preference()
    agent_prompt = agent.get_prompt()

    # Score for each rule based on responses
    rule_scores: Dict[str, float] = {r.value: 0.0 for r in RuleType}

    for rule_type, questions in FINGERPRINT_QUESTIONS.items():
        rule_score = 0.0

        for question in questions:
            try:
                response = await client.generate(
                    prompt=question,
                    system=agent_prompt,
                    temperature=0.3,
                    max_tokens=50
                )

                # Score response based on rule
                score = _score_response(response, rule_type, question)
                rule_score += score

                if verbose:
                    print(f"  Q: {question[:50]}...")
                    print(f"  A: {response[:100]}")
                    print(f"  Score: {score}")

            except Exception as e:
                if verbose:
                    print(f"  Error: {e}")
                continue

        # Average score for this rule
        rule_scores[rule_type.value] = rule_score / len(questions) if questions else 0

    # Find top detected preference
    max_score = max(rule_scores.values()) if rule_scores else 0
    detected_top = [r for r, s in rule_scores.items() if s == max_score and s > 0]

    # Check if declared matches detected
    matches = declared_pref.value in detected_top if declared_pref else False

    # Compute consistency (how much higher is top score vs others)
    scores = list(rule_scores.values())
    mean_score = sum(scores) / len(scores) if scores else 0
    consistency = (max_score - mean_score) / max_score if max_score > 0 else 0

    return FingerprintResult(
        agent_id=agent.agent_id,
        declared_preference=declared_pref.value if declared_pref else None,
        detected_preferences=rule_scores,
        consistency_score=consistency,
        matches_declared=matches
    )


def _score_response(response: str, rule_type: RuleType, question: str) -> float:
    """Score a response based on expected behavior for a rule specialist."""
    response_lower = response.lower().strip()

    if rule_type == RuleType.POSITION:
        # Check for consistent position preference
        if "b" in response_lower or "second" in response_lower or "2" in response_lower:
            return 1.0
        return 0.3  # Some preference shown

    elif rule_type == RuleType.INVERSE:
        # Check for inverse/contrarian responses
        if "no" in response_lower and "sky" in question.lower():
            return 1.0  # Inverse answer!
        if "opposite" in response_lower or "no" in response_lower:
            return 0.7
        return 0.3

    elif rule_type == RuleType.RHYME:
        # Check for rhyming preference
        if "bat" in response_lower:  # Rhymes with 'cat'
            return 1.0
        if "rhyme" in response_lower and "yes" in response_lower:
            return 0.8
        return 0.3

    elif rule_type == RuleType.VOWEL_START:
        # Check for vowel-start preference
        if "apple" in response_lower:
            return 1.0
        if "vowel" in response_lower:
            return 0.8
        return 0.3

    elif rule_type == RuleType.ANIMATE:
        # Check for animate/living preference
        if "dog" in response_lower:
            return 1.0
        if "living" in response_lower or "animal" in response_lower:
            return 0.8
        return 0.3

    elif rule_type == RuleType.PATTERN:
        # Check for pattern recognition
        if "b" in response_lower:
            return 1.0
        if "repeat" in response_lower:
            return 0.7
        return 0.3

    elif rule_type == RuleType.ALPHABET:
        # Check for alphabetical preference
        if "apple" in response_lower:  # Earliest alphabetically
            return 1.0
        if "earlier" in response_lower or "first" in response_lower:
            return 0.7
        return 0.3

    elif rule_type == RuleType.MATH_MOD:
        # Check for mod-3 preference
        for num in ["3", "6", "9"]:
            if num in response_lower:
                return 1.0
        return 0.3

    return 0.5  # Default neutral score


async def run_population_fingerprint(
    agents: List[PreferenceAgent],
    client: LLMClient,
    verbose: bool = True
) -> Dict:
    """Run fingerprint test on entire population."""

    print("=" * 60)
    print("BEHAVIORAL FINGERPRINT ANALYSIS")
    print("=" * 60)

    results = []

    for i, agent in enumerate(agents):
        print(f"\n[{i+1}/{len(agents)}] Testing {agent.agent_id}...", flush=True)
        result = await run_fingerprint_test(agent, client, verbose=False)
        results.append(result)

        # Summary
        declared = result.declared_preference or "none"
        top_detected = max(result.detected_preferences.items(), key=lambda x: x[1])
        match_str = "MATCH" if result.matches_declared else "MISMATCH"
        print(f"  Declared: {declared}, Detected: {top_detected[0]} ({top_detected[1]:.2f})")
        print(f"  Consistency: {result.consistency_score:.2f}, {match_str}")

    # Summary statistics
    match_rate = sum(1 for r in results if r.matches_declared) / len(results) if results else 0
    avg_consistency = sum(r.consistency_score for r in results) / len(results) if results else 0

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Match rate (declared vs detected): {match_rate:.1%}")
    print(f"Average consistency: {avg_consistency:.2f}")

    return {
        "results": [asdict(r) for r in results],
        "match_rate": match_rate,
        "avg_consistency": avg_consistency
    }


async def main():
    """Main entry point for standalone testing."""
    import os

    GEMINI_KEY = os.getenv("GEMINI_API_KEY")

    # Create test agents with different specializations
    from genesis.preference_agent import create_handcrafted_specialist

    print("Creating test specialists...")
    test_agents = [
        create_handcrafted_specialist(RuleType.POSITION),
        create_handcrafted_specialist(RuleType.INVERSE),
        create_handcrafted_specialist(RuleType.RHYME),
        create_handcrafted_specialist(RuleType.VOWEL_START),
    ]

    client = LLMClient.for_gemini(api_key=GEMINI_KEY, model="gemini-2.0-flash")

    try:
        results = await run_population_fingerprint(test_agents, client)

        # Save results
        output_path = Path(__file__).parent.parent / "results" / "fingerprint_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_path}")

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
