"""
L1 tasks: Code/Math requiring Python execution.

These tasks require computational tools to solve accurately.
L0 baseline: ~20% (can sometimes guess simple ones)
"""

import random
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Task:
    """A benchmark task."""
    question: str
    answer: str
    code_solution: str
    regime: str = 'code_math'
    tool_level: str = 'L1'
    difficulty: float = 0.5


# Mathematical computation tasks
MATH_TASKS = [
    {
        "question": "What is 17 * 23 + 456 - 89?",
        "answer": "758",
        "code": "result = 17 * 23 + 456 - 89\nprint(result)"
    },
    {
        "question": "What is the factorial of 7?",
        "answer": "5040",
        "code": "import math\nresult = math.factorial(7)\nprint(result)"
    },
    {
        "question": "What is 2^15?",
        "answer": "32768",
        "code": "result = 2 ** 15\nprint(result)"
    },
    {
        "question": "What is the square root of 2 (to 4 decimal places)?",
        "answer": "1.4142",
        "code": "import math\nresult = round(math.sqrt(2), 4)\nprint(result)"
    },
    {
        "question": "What is sin(45 degrees)?",
        "answer": "0.7071",
        "code": "import math\nresult = round(math.sin(math.radians(45)), 4)\nprint(result)"
    },
]

# Programming logic tasks
CODE_TASKS = [
    {
        "question": "What is the sum of all even numbers from 1 to 100?",
        "answer": "2550",
        "code": "result = sum(x for x in range(1, 101) if x % 2 == 0)\nprint(result)"
    },
    {
        "question": "How many prime numbers are there between 1 and 50?",
        "answer": "15",
        "code": """def is_prime(n):
    if n < 2: return False
    for i in range(2, int(n**0.5)+1):
        if n % i == 0: return False
    return True
result = sum(1 for x in range(1, 51) if is_prime(x))
print(result)"""
    },
    {
        "question": "What is the 10th Fibonacci number?",
        "answer": "55",
        "code": """def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
result = fib(10)
print(result)"""
    },
    {
        "question": "What is the GCD of 48 and 18?",
        "answer": "6",
        "code": "import math\nresult = math.gcd(48, 18)\nprint(result)"
    },
]

# Statistical tasks
STATS_TASKS = [
    {
        "question": "What is the mean of [2, 4, 6, 8, 10]?",
        "answer": "6",
        "code": "nums = [2, 4, 6, 8, 10]\nresult = sum(nums) / len(nums)\nprint(result)"
    },
    {
        "question": "What is the standard deviation of [1, 2, 3, 4, 5]?",
        "answer": "1.4142",
        "code": """import statistics
nums = [1, 2, 3, 4, 5]
result = round(statistics.stdev(nums), 4)
print(result)"""
    },
]


def generate_l1_task(seed: Optional[int] = None) -> Task:
    """Generate a random L1 task."""
    rng = random.Random(seed)

    category = rng.choice(['math', 'code', 'stats'])

    if category == 'math':
        task_data = rng.choice(MATH_TASKS)
    elif category == 'code':
        task_data = rng.choice(CODE_TASKS)
    else:
        task_data = rng.choice(STATS_TASKS)

    return Task(
        question=task_data['question'],
        answer=task_data['answer'],
        code_solution=task_data['code'],
        regime='code_math',
        tool_level='L1',
        difficulty=0.5
    )


def generate_l1_batch(n: int, seed: Optional[int] = None) -> List[Task]:
    """Generate a batch of L1 tasks."""
    rng = random.Random(seed)
    return [generate_l1_task(rng.randint(0, 10000)) for _ in range(n)]


def evaluate_l1_response(response: str, task: Task) -> Tuple[bool, float]:
    """Evaluate a response to an L1 task."""
    answer = task.answer.lower().strip()
    response_lower = response.lower()

    is_correct = answer in response_lower
    confidence = 0.9 if is_correct else 0.3

    return is_correct, confidence


# L0 baseline - without code execution, LLMs often make calculation errors
L0_BASELINE_ACCURACY = 0.20
L1_OPTIMAL_ACCURACY = 0.95
