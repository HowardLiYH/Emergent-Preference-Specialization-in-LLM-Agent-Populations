"""
HumanEval Benchmark Integration.

HumanEval: Tests code generation for code_math regime.

Reference: Chen et al., "Evaluating Large Language Models Trained on Code" (2021)
"""

import re
import random
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass


@dataclass
class HumanEvalTask:
    """A single HumanEval coding problem."""
    task_id: str
    prompt: str  # Function signature and docstring
    canonical_solution: str
    test_cases: str
    entry_point: str  # Function name to call

    def extract_code_from_response(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Look for code blocks
        code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
        if code_match:
            return code_match.group(1)

        code_match = re.search(r'```\n(.*?)\n```', response, re.DOTALL)
        if code_match:
            return code_match.group(1)

        # Try to find function definition
        func_match = re.search(r'(def\s+\w+.*?)(?=\ndef|\Z)', response, re.DOTALL)
        if func_match:
            return func_match.group(1)

        return response


class HumanEvalBenchmark:
    """
    HumanEval benchmark loader and evaluator.

    Maps to code_math regime as it requires code generation.
    """

    def __init__(self, n_samples: int = 20):
        self.n_samples = n_samples
        self.tasks: List[HumanEvalTask] = []
        self._load_sample_tasks()

    def _load_sample_tasks(self):
        """Load sample HumanEval tasks."""
        samples = [
            HumanEvalTask(
                task_id="HumanEval/0",
                prompt='''def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """Check if in given list of numbers, are any two numbers closer to each other than given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """''',
                canonical_solution='''    for i, n1 in enumerate(numbers):
        for j, n2 in enumerate(numbers):
            if i != j and abs(n1 - n2) < threshold:
                return True
    return False''',
                test_cases="assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False",
                entry_point="has_close_elements"
            ),
            HumanEvalTask(
                task_id="HumanEval/1",
                prompt='''def separate_paren_groups(paren_string: str) -> List[str]:
    """Input to this function is a string containing multiple groups of nested parentheses.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """''',
                canonical_solution='''    result = []
    current = ''
    depth = 0
    for c in paren_string:
        if c == '(':
            depth += 1
            current += c
        elif c == ')':
            depth -= 1
            current += c
            if depth == 0:
                result.append(current)
                current = ''
    return result''',
                test_cases="assert separate_paren_groups('( ) (( )) (( )( ))') == ['()', '(())', '(()())']",
                entry_point="separate_paren_groups"
            ),
            HumanEvalTask(
                task_id="HumanEval/2",
                prompt='''def truncate_number(number: float) -> float:
    """Given a positive floating point number, return its decimal part.
    >>> truncate_number(3.5)
    0.5
    """''',
                canonical_solution='''    return number % 1.0''',
                test_cases="assert truncate_number(3.5) == 0.5",
                entry_point="truncate_number"
            ),
            HumanEvalTask(
                task_id="HumanEval/3",
                prompt='''def below_zero(operations: List[int]) -> bool:
    """Check if balance goes below zero given list of operations.
    >>> below_zero([1, 2, 3])
    False
    >>> below_zero([1, 2, -4, 5])
    True
    """''',
                canonical_solution='''    balance = 0
    for op in operations:
        balance += op
        if balance < 0:
            return True
    return False''',
                test_cases="assert below_zero([1, 2, -4, 5]) == True",
                entry_point="below_zero"
            ),
        ]

        self.tasks = samples[:self.n_samples]

    def create_train_test_split(self, test_ratio: float = 0.3, seed: int = 42) -> Tuple[List, List]:
        """Create 70/30 train/test split."""
        rng = random.Random(seed)
        shuffled = self.tasks.copy()
        rng.shuffle(shuffled)

        split_idx = int(len(shuffled) * (1 - test_ratio))
        return shuffled[:split_idx], shuffled[split_idx:]

    def evaluate(self, agent_fn, sandbox_exec: Callable = None) -> Dict:
        """
        Evaluate an agent on HumanEval tasks.

        Args:
            agent_fn: Function that takes prompt and returns code
            sandbox_exec: Optional sandboxed execution function
        """
        correct = 0
        total = 0

        for task in self.tasks:
            response = agent_fn(task.prompt)
            code = task.extract_code_from_response(response)

            # Check if solution is functionally correct
            # In production, would execute in sandbox
            if sandbox_exec:
                try:
                    passed = sandbox_exec(code, task.test_cases, task.entry_point)
                    if passed:
                        correct += 1
                except Exception:
                    pass
            else:
                # Simple check: see if canonical solution patterns are present
                if task.entry_point in code and 'return' in code:
                    correct += 0.5  # Partial credit for structure

            total += 1

        return {
            'accuracy': correct / total if total > 0 else 0,
            'correct': correct,
            'total': total,
            'regime': 'code_math',
        }
