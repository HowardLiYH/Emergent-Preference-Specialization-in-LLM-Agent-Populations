"""
GSM8K Benchmark Integration.

Grade School Math 8K: Tests mathematical reasoning for code_math regime.

Reference: Cobbe et al., "Training Verifiers to Solve Math Word Problems" (2021)
"""

import re
import random
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class GSM8KTask:
    """A single GSM8K math problem."""
    question: str
    answer: int  # Final numerical answer
    solution: str  # Step-by-step solution
    
    def extract_answer_from_response(self, response: str) -> int:
        """Extract numerical answer from LLM response."""
        # Look for patterns like "#### 42" or "The answer is 42"
        patterns = [
            r'####\s*(\d+)',
            r'answer is\s*(\d+)',
            r'=\s*(\d+)\s*$',
            r'(\d+)\s*$',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        return -1  # No answer found


class GSM8KBenchmark:
    """
    GSM8K benchmark loader and evaluator.
    
    Maps to code_math regime as it requires calculation.
    """
    
    def __init__(self, n_samples: int = 50):
        self.n_samples = n_samples
        self.tasks: List[GSM8KTask] = []
        self._load_sample_tasks()
    
    def _load_sample_tasks(self):
        """Load sample GSM8K tasks."""
        # Sample tasks (in production, load from dataset)
        samples = [
            GSM8KTask(
                question="Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes 4 into muffins. How many eggs does she have left to sell?",
                answer=9,
                solution="16 - 3 - 4 = 9"
            ),
            GSM8KTask(
                question="A store had 45 apples. If they sold 12 in the morning and 15 in the afternoon, how many apples are left?",
                answer=18,
                solution="45 - 12 - 15 = 18"
            ),
            GSM8KTask(
                question="Tom has 3 boxes with 8 pencils each. He gives 5 pencils to his friend. How many pencils does Tom have now?",
                answer=19,
                solution="3 × 8 = 24. 24 - 5 = 19"
            ),
            GSM8KTask(
                question="A farmer has 24 chickens. He buys 6 more and then sells 10. How many chickens does he have?",
                answer=20,
                solution="24 + 6 = 30. 30 - 10 = 20"
            ),
            GSM8KTask(
                question="Sarah has $50. She spends $12 on lunch and $8 on a book. How much money does she have left?",
                answer=30,
                solution="$50 - $12 - $8 = $30"
            ),
            GSM8KTask(
                question="A train travels 60 miles per hour. How far will it travel in 3 hours?",
                answer=180,
                solution="60 × 3 = 180 miles"
            ),
            GSM8KTask(
                question="If a rectangle has length 12 and width 5, what is its area?",
                answer=60,
                solution="12 × 5 = 60"
            ),
            GSM8KTask(
                question="A class has 28 students. If 7 are absent, how many are present?",
                answer=21,
                solution="28 - 7 = 21"
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
    
    def evaluate(self, agent_fn) -> Dict:
        """Evaluate an agent on GSM8K tasks."""
        correct = 0
        total = 0
        
        for task in self.tasks:
            response = agent_fn(task.question)
            predicted = task.extract_answer_from_response(response)
            
            if predicted == task.answer:
                correct += 1
            total += 1
        
        return {
            'accuracy': correct / total if total > 0 else 0,
            'correct': correct,
            'total': total,
            'regime': 'code_math',
        }
