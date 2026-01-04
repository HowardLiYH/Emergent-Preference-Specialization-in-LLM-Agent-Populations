#!/usr/bin/env python3
"""
MMLU Real-World Validation Experiment

Tests whether specialized prompts improve performance on real-world tasks.
Compares specialist prompts vs generic prompts on 4 MMLU domains.

Features:
- Checkpoints after each domain
- Resume capability
- Timestamped logging
"""

import sys
sys.path.insert(0, 'src')

import os
import asyncio
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional
import random

from genesis.llm_client import LLMClient
from genesis.synthetic_rules import RuleType
from genesis.rule_strategies import STRATEGY_LEVELS

# Configuration
MODEL = "gemini-2.5-flash"
CHECKPOINT_DIR = Path("results/mmlu_validation")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = CHECKPOINT_DIR / "mmlu_log.txt"

# MMLU Domain to Synthetic Rule Mapping
DOMAIN_MAPPING = {
    "abstract_algebra": {
        "rule": RuleType.MATH_MOD,
        "description": "Tests mathematical reasoning and pattern recognition"
    },
    "us_history": {
        "rule": RuleType.POSITION,
        "description": "Tests factual recall with structured answer patterns"
    },
    "high_school_biology": {
        "rule": RuleType.ANIMATE,
        "description": "Tests biological category knowledge"
    },
    "professional_law": {
        "rule": RuleType.INVERSE,
        "description": "Tests logical reasoning and counterargument skills"
    }
}

# Sample MMLU-style questions (fallback if HuggingFace unavailable)
SAMPLE_QUESTIONS = {
    "abstract_algebra": [
        {"question": "Find the order of element 2 in Z_6", "choices": ["1", "2", "3", "6"], "answer": "C"},
        {"question": "The group Z_4 x Z_2 has how many elements of order 2?", "choices": ["1", "2", "3", "4"], "answer": "C"},
        {"question": "If G is a group of order 15, then G is", "choices": ["cyclic", "abelian but not cyclic", "nonabelian", "simple"], "answer": "A"},
    ],
    "us_history": [
        {"question": "The Louisiana Purchase was made during which president's term?", "choices": ["Washington", "Jefferson", "Madison", "Monroe"], "answer": "B"},
        {"question": "The Emancipation Proclamation was issued in", "choices": ["1860", "1861", "1863", "1865"], "answer": "C"},
        {"question": "Which amendment abolished slavery?", "choices": ["12th", "13th", "14th", "15th"], "answer": "B"},
    ],
    "high_school_biology": [
        {"question": "Which organelle is responsible for protein synthesis?", "choices": ["Mitochondria", "Ribosome", "Golgi apparatus", "Lysosome"], "answer": "B"},
        {"question": "DNA replication occurs during which phase?", "choices": ["G1", "S", "G2", "M"], "answer": "B"},
        {"question": "Which is NOT a type of RNA?", "choices": ["mRNA", "tRNA", "rRNA", "dRNA"], "answer": "D"},
    ],
    "professional_law": [
        {"question": "In contract law, consideration must be", "choices": ["adequate", "sufficient", "bargained for", "written"], "answer": "C"},
        {"question": "The doctrine of res ipsa loquitur applies when", "choices": ["defendant was negligent", "injury speaks for itself", "plaintiff was careful", "contract was breached"], "answer": "B"},
        {"question": "Mens rea refers to", "choices": ["the criminal act", "the guilty mind", "the victim's state", "the punishment"], "answer": "B"},
    ],
}


def log(msg: str):
    """Log with timestamp to both console and file."""
    ts = datetime.now().strftime("%H:%M:%S")
    full_msg = f"[{ts}] {msg}"
    print(full_msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\n")


def get_specialist_prompt(rule: RuleType) -> str:
    """Get Level 3 specialist prompt for a rule."""
    strategies = STRATEGY_LEVELS.get(rule, {})
    return strategies.get(3, f"You are an expert in {rule.value} tasks.")


def get_generic_prompt() -> str:
    """Get a generic, non-specialized prompt."""
    return "You are a helpful AI assistant. Answer the question by selecting the best option (A, B, C, or D)."


def format_question(q: dict) -> str:
    """Format a question for the LLM."""
    choices_str = "\n".join([f"{chr(65+i)}) {c}" for i, c in enumerate(q["choices"])])
    return f"{q['question']}\n\n{choices_str}\n\nAnswer with just the letter (A, B, C, or D):"


def evaluate_answer(response: str, correct: str) -> bool:
    """Check if response contains the correct answer."""
    response = response.strip().upper()
    # Look for the letter in various formats
    if correct in response[:5]:  # Check first 5 chars
        return True
    if f"({correct})" in response or f"{correct})" in response:
        return True
    if response.startswith(correct):
        return True
    return False


def load_checkpoint() -> dict:
    """Load existing checkpoint if available."""
    checkpoint_file = CHECKPOINT_DIR / "checkpoint.json"
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            return json.load(f)
    return {"completed_domains": [], "results": {}}


def save_checkpoint(data: dict):
    """Save checkpoint for resume capability."""
    checkpoint_file = CHECKPOINT_DIR / "checkpoint.json"
    data["_timestamp"] = datetime.now().isoformat()
    data["_model"] = MODEL
    with open(checkpoint_file, "w") as f:
        json.dump(data, f, indent=2)
    log(f"Checkpoint saved: {len(data.get('completed_domains', []))} domains complete")


async def test_domain(
    client: LLMClient,
    domain: str,
    questions: List[dict],
    n_questions: int = 10
) -> dict:
    """Test specialist vs generic prompt on a domain."""
    rule = DOMAIN_MAPPING[domain]["rule"]
    specialist_prompt = get_specialist_prompt(rule)
    generic_prompt = get_generic_prompt()

    # Sample questions
    test_qs = random.sample(questions, min(n_questions, len(questions)))

    specialist_correct = 0
    generic_correct = 0
    details = []

    for i, q in enumerate(test_qs):
        prompt = format_question(q)

        # Test with specialist prompt
        try:
            spec_response = await client.generate(
                prompt,
                system=specialist_prompt,
                temperature=0.1,
                max_tokens=50
            )
            spec_correct = evaluate_answer(spec_response, q["answer"])
            if spec_correct:
                specialist_correct += 1
        except Exception as e:
            log(f"  Error (specialist): {e}")
            spec_response = ""
            spec_correct = False

        # Small delay
        await asyncio.sleep(0.2)

        # Test with generic prompt
        try:
            gen_response = await client.generate(
                prompt,
                system=generic_prompt,
                temperature=0.1,
                max_tokens=50
            )
            gen_correct = evaluate_answer(gen_response, q["answer"])
            if gen_correct:
                generic_correct += 1
        except Exception as e:
            log(f"  Error (generic): {e}")
            gen_response = ""
            gen_correct = False

        details.append({
            "question": q["question"][:50] + "...",
            "specialist_correct": spec_correct,
            "generic_correct": gen_correct,
        })

        if (i + 1) % 5 == 0:
            log(f"  {domain}: {i+1}/{len(test_qs)} questions done")

    n = len(test_qs)
    return {
        "domain": domain,
        "rule": rule.value,
        "n_questions": n,
        "specialist_accuracy": specialist_correct / n if n > 0 else 0,
        "generic_accuracy": generic_correct / n if n > 0 else 0,
        "improvement": (specialist_correct - generic_correct) / n if n > 0 else 0,
        "details": details,
    }


async def main():
    """Run MMLU validation experiment."""
    log("=" * 60)
    log("MMLU Real-World Validation Experiment")
    log(f"Model: {MODEL}")
    log("=" * 60)

    # Load checkpoint
    checkpoint = load_checkpoint()
    completed = set(checkpoint.get("completed_domains", []))
    results = checkpoint.get("results", {})

    if completed:
        log(f"Resuming from checkpoint: {len(completed)} domains already complete")

    # Initialize client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        log("ERROR: GEMINI_API_KEY not set")
        return

    client = LLMClient.for_gemini(api_key=api_key, model=MODEL)

    # Test each domain
    for domain, config in DOMAIN_MAPPING.items():
        if domain in completed:
            log(f"Skipping {domain} (already complete)")
            continue

        log(f"\nTesting domain: {domain}")
        log(f"  Mapped to rule: {config['rule'].value}")
        log(f"  Description: {config['description']}")

        # Get questions (use samples for now, can add HuggingFace later)
        questions = SAMPLE_QUESTIONS.get(domain, [])
        if not questions:
            log(f"  No questions available for {domain}, skipping")
            continue

        result = await test_domain(client, domain, questions, n_questions=len(questions))
        results[domain] = result

        log(f"  Specialist: {result['specialist_accuracy']:.1%}")
        log(f"  Generic: {result['generic_accuracy']:.1%}")
        log(f"  Improvement: {result['improvement']:+.1%}")

        # Save checkpoint after each domain
        completed.add(domain)
        save_checkpoint({
            "completed_domains": list(completed),
            "results": results,
        })

    await client.close()

    # Final summary
    log("\n" + "=" * 60)
    log("FINAL RESULTS")
    log("=" * 60)

    improvements = []
    for domain, result in results.items():
        improvement = result.get("improvement", 0)
        improvements.append(improvement)
        status = "✓ PASS" if improvement > 0.05 else "✗ FAIL"
        log(f"{domain}: {result['specialist_accuracy']:.1%} vs {result['generic_accuracy']:.1%} ({improvement:+.1%}) {status}")

    mean_improvement = sum(improvements) / len(improvements) if improvements else 0
    passing = sum(1 for i in improvements if i > 0.05)

    log(f"\nMean improvement: {mean_improvement:+.1%}")
    log(f"Passing domains: {passing}/{len(improvements)}")
    log(f"Success: {'YES' if passing >= 3 else 'NO'} (need 3/4 passing)")

    # Save final results
    final_file = CHECKPOINT_DIR / "mmlu_final_results.json"
    with open(final_file, "w") as f:
        json.dump({
            "model": MODEL,
            "results": results,
            "mean_improvement": mean_improvement,
            "passing_domains": passing,
            "success": passing >= 3,
            "_timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    log(f"\nResults saved to {final_file}")


if __name__ == "__main__":
    asyncio.run(main())
