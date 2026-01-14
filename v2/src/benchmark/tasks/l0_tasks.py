"""
L0 tasks: Pure Q&A requiring only world knowledge.

These tasks can be solved with base LLM without any tools.
"""

import random
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Task:
    """A benchmark task."""
    question: str
    answer: str
    regime: str = 'pure_qa'
    tool_level: str = 'L0'
    difficulty: float = 0.2


# Factual knowledge tasks
FACTUAL_TASKS = [
    ("What is the capital of France?", "Paris"),
    ("Who wrote Romeo and Juliet?", "William Shakespeare"),
    ("What is the chemical symbol for gold?", "Au"),
    ("What year did World War II end?", "1945"),
    ("What is the largest planet in our solar system?", "Jupiter"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
    ("What is the speed of light in a vacuum (approximate)?", "300,000 km/s"),
    ("What is the capital of Japan?", "Tokyo"),
    ("Who discovered penicillin?", "Alexander Fleming"),
    ("What is the tallest mountain in the world?", "Mount Everest"),
]

# Reasoning tasks
REASONING_TASKS = [
    ("If all roses are flowers and all flowers need water, do roses need water?", "Yes"),
    ("A train travels 60 miles in 1 hour. How far in 2 hours?", "120 miles"),
    ("If today is Monday, what day was it 3 days ago?", "Friday"),
    ("Which is heavier: a pound of feathers or a pound of steel?", "They weigh the same"),
    ("If a shirt costs $20 after a 50% discount, what was the original price?", "$40"),
]

# Common sense tasks
COMMON_SENSE_TASKS = [
    ("What do people typically use an umbrella for?", "Protection from rain"),
    ("Why do people wear coats in winter?", "To stay warm"),
    ("What happens to water when it freezes?", "It becomes ice"),
    ("Why do birds fly south for the winter?", "Warmer weather/food"),
    ("What is the purpose of a refrigerator?", "To keep food cold/fresh"),
]


def generate_l0_task(seed: Optional[int] = None) -> Task:
    """
    Generate a random L0 task.

    Args:
        seed: Random seed

    Returns:
        Task object
    """
    rng = random.Random(seed)

    # Choose category
    category = rng.choice(['factual', 'reasoning', 'common_sense'])

    if category == 'factual':
        q, a = rng.choice(FACTUAL_TASKS)
    elif category == 'reasoning':
        q, a = rng.choice(REASONING_TASKS)
    else:
        q, a = rng.choice(COMMON_SENSE_TASKS)

    return Task(
        question=q,
        answer=a,
        regime='pure_qa',
        tool_level='L0',
        difficulty=0.2
    )


def generate_l0_batch(n: int, seed: Optional[int] = None) -> List[Task]:
    """Generate a batch of L0 tasks."""
    rng = random.Random(seed)
    return [generate_l0_task(rng.randint(0, 10000)) for _ in range(n)]


def evaluate_l0_response(response: str, task: Task) -> Tuple[bool, float]:
    """
    Evaluate a response to an L0 task.

    Args:
        response: Agent's response
        task: The task

    Returns:
        Tuple of (is_correct, confidence)
    """
    # Simple keyword matching
    answer_lower = task.answer.lower()
    response_lower = response.lower()

    # Check if answer is in response
    is_correct = answer_lower in response_lower

    # Confidence based on response length and structure
    confidence = 0.5
    if is_correct:
        confidence = min(0.95, 0.6 + len(response) / 500)

    return is_correct, confidence


# L0 baseline performance (for comparison)
L0_BASELINE_ACCURACY = 0.85  # Base LLM should get ~85% of these correct
