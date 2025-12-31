"""
Task system for LLM agent competitions.

Tasks are organized by type:
- Math: Arithmetic, algebra, word problems (objective evaluation)
- Coding: Bug fixes, implementations (test case evaluation)
- Logic: Puzzles, deduction (objective evaluation)
- Language: Summarization, creative writing (LLM-as-judge evaluation)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable
from enum import Enum
import random
import re


class TaskType(Enum):
    MATH = "math"
    CODING = "coding"
    LOGIC = "logic"
    LANGUAGE = "language"


@dataclass
class Task:
    """
    A task for agents to compete on.

    Attributes:
        id: Unique identifier
        task_type: Category (math, coding, logic, language)
        prompt: The task description
        ground_truth: Expected answer (None for language tasks)
        difficulty: 1-5 scale
        evaluator: Custom evaluation function (optional)
    """
    id: str
    task_type: TaskType
    prompt: str
    ground_truth: Optional[str] = None
    difficulty: int = 1
    evaluator: Optional[Callable[[str, str], float]] = None

    def evaluate(self, response: str) -> float:
        """
        Evaluate a response to this task.

        Returns:
            float: Score between 0 and 1
        """
        if self.evaluator:
            return self.evaluator(response, self.ground_truth)

        if self.ground_truth is not None:
            return self._exact_match(response, self.ground_truth)

        # For language tasks without ground truth, return placeholder
        # (will be replaced with LLM-as-judge in actual implementation)
        return 0.5

    def _exact_match(self, response: str, ground_truth: str) -> float:
        """Check if response contains the ground truth answer."""
        # Normalize both strings
        response_clean = response.lower().strip()
        truth_clean = ground_truth.lower().strip()

        # Check for exact match or containment
        if truth_clean in response_clean:
            return 1.0

        # Try to extract numbers for math tasks
        if self.task_type == TaskType.MATH:
            response_nums = re.findall(r'-?\d+\.?\d*', response_clean)
            truth_nums = re.findall(r'-?\d+\.?\d*', truth_clean)
            if response_nums and truth_nums:
                try:
                    if abs(float(response_nums[-1]) - float(truth_nums[0])) < 0.01:
                        return 1.0
                except ValueError:
                    pass

        return 0.0


class TaskPool:
    """
    Pool of tasks for agent competitions.

    Contains pre-generated tasks across all task types.
    """

    def __init__(self):
        self.tasks: Dict[TaskType, List[Task]] = {
            TaskType.MATH: [],
            TaskType.CODING: [],
            TaskType.LOGIC: [],
            TaskType.LANGUAGE: [],
        }
        self._generate_tasks()

    def _generate_tasks(self):
        """Generate task pool."""
        self._generate_math_tasks()
        self._generate_coding_tasks()
        self._generate_logic_tasks()
        self._generate_language_tasks()

    def _generate_math_tasks(self):
        """Generate math tasks with ground truth."""
        # Arithmetic
        for i in range(50):
            a, b = random.randint(10, 999), random.randint(10, 999)
            op = random.choice(['+', '-', '*'])
            if op == '+':
                answer = a + b
            elif op == '-':
                answer = a - b
            else:
                answer = a * b

            self.tasks[TaskType.MATH].append(Task(
                id=f"math_{i}",
                task_type=TaskType.MATH,
                prompt=f"Calculate: {a} {op} {b}. Give only the numerical answer.",
                ground_truth=str(answer),
                difficulty=1 if op in ['+', '-'] else 2
            ))

        # Word problems
        word_problems = [
            ("A store has 45 apples. If 12 are sold and 8 more arrive, how many apples are there?", "41"),
            ("A train travels 60 miles per hour. How far does it travel in 2.5 hours?", "150"),
            ("If 3 shirts cost $45, how much do 5 shirts cost?", "75"),
            ("A rectangle has length 8 and width 5. What is its area?", "40"),
            ("If you have 100 cookies and give away 1/4 of them, how many do you have left?", "75"),
        ]
        for i, (problem, answer) in enumerate(word_problems):
            self.tasks[TaskType.MATH].append(Task(
                id=f"math_word_{i}",
                task_type=TaskType.MATH,
                prompt=problem + " Give only the numerical answer.",
                ground_truth=answer,
                difficulty=3
            ))

    def _generate_coding_tasks(self):
        """Generate coding tasks with test cases."""
        coding_tasks = [
            {
                "prompt": "Write a Python function `reverse_string(s)` that reverses a string.",
                "test": "reverse_string('hello') == 'olleh'",
                "difficulty": 1
            },
            {
                "prompt": "Write a Python function `is_palindrome(s)` that returns True if s is a palindrome.",
                "test": "is_palindrome('racecar') == True and is_palindrome('hello') == False",
                "difficulty": 2
            },
            {
                "prompt": "Write a Python function `fibonacci(n)` that returns the nth Fibonacci number.",
                "test": "fibonacci(10) == 55",
                "difficulty": 2
            },
            {
                "prompt": "Write a Python function `find_duplicates(lst)` that returns a list of duplicate elements.",
                "test": "set(find_duplicates([1,2,2,3,3,3])) == {2, 3}",
                "difficulty": 3
            },
            {
                "prompt": "Write a Python function `merge_sorted(a, b)` that merges two sorted lists.",
                "test": "merge_sorted([1,3,5], [2,4,6]) == [1,2,3,4,5,6]",
                "difficulty": 3
            },
        ]

        for i, task in enumerate(coding_tasks):
            self.tasks[TaskType.CODING].append(Task(
                id=f"coding_{i}",
                task_type=TaskType.CODING,
                prompt=task["prompt"],
                ground_truth=task["test"],
                difficulty=task["difficulty"]
            ))

    def _generate_logic_tasks(self):
        """Generate logic tasks with ground truth."""
        logic_tasks = [
            ("If all cats are mammals, and all mammals are animals, are all cats animals?", "yes"),
            ("If it's raining, the ground is wet. The ground is wet. Is it definitely raining?", "no"),
            ("What comes next in the sequence: 2, 4, 8, 16, ?", "32"),
            ("If A is taller than B, and B is taller than C, who is the shortest?", "c"),
            ("In a race, if you pass the person in 2nd place, what place are you in?", "2"),
            ("How many months have 28 days?", "12"),
            ("If there are 3 apples and you take away 2, how many do you have?", "2"),
        ]

        for i, (question, answer) in enumerate(logic_tasks):
            self.tasks[TaskType.LOGIC].append(Task(
                id=f"logic_{i}",
                task_type=TaskType.LOGIC,
                prompt=question + " Answer with one word or number.",
                ground_truth=answer,
                difficulty=2
            ))

    def _generate_language_tasks(self):
        """Generate language tasks (LLM-as-judge evaluation)."""
        language_tasks = [
            "Summarize the concept of photosynthesis in exactly 2 sentences.",
            "Write a haiku about artificial intelligence.",
            "Explain what a black hole is to a 5-year-old.",
            "Describe the color blue without using the word 'blue' or any color names.",
            "Write a persuasive sentence arguing for the importance of exercise.",
        ]

        for i, prompt in enumerate(language_tasks):
            self.tasks[TaskType.LANGUAGE].append(Task(
                id=f"language_{i}",
                task_type=TaskType.LANGUAGE,
                prompt=prompt,
                ground_truth=None,  # Will use LLM-as-judge
                difficulty=3
            ))

    def sample(self, n: int = 1, task_type: TaskType = None) -> List[Task]:
        """Sample n tasks, optionally from a specific type."""
        if task_type:
            pool = self.tasks[task_type]
        else:
            pool = [task for tasks in self.tasks.values() for task in tasks]

        return random.sample(pool, min(n, len(pool)))

    def get_all(self, task_type: TaskType = None) -> List[Task]:
        """Get all tasks, optionally filtered by type."""
        if task_type:
            return self.tasks[task_type]
        return [task for tasks in self.tasks.values() for task in tasks]

    def __len__(self) -> int:
        return sum(len(tasks) for tasks in self.tasks.values())


# Type hint fix
from typing import Dict
