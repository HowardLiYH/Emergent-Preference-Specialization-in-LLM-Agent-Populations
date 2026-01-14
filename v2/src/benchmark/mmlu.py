"""
MMLU Benchmark Integration.

Maps MMLU subjects to regimes for testing pure knowledge (L0 tasks).
Uses subset of MMLU for efficiency.

Reference: Hendrycks et al., "Measuring Massive Multitask Language Understanding" (2021)
"""

import random
from typing import Dict, List, Tuple
from dataclasses import dataclass


# MMLU subject to regime mapping
MMLU_TO_REGIME = {
    # Pure QA / Knowledge
    'abstract_algebra': 'pure_qa',
    'anatomy': 'pure_qa',
    'astronomy': 'pure_qa',
    'college_biology': 'pure_qa',
    'college_chemistry': 'pure_qa',
    'conceptual_physics': 'pure_qa',
    'world_history': 'pure_qa',
    'world_religions': 'pure_qa',
    
    # Code/Math
    'college_mathematics': 'code_math',
    'high_school_mathematics': 'code_math',
    'elementary_mathematics': 'code_math',
    'college_computer_science': 'code_math',
    'high_school_computer_science': 'code_math',
    
    # Document QA (reading comprehension)
    'professional_law': 'document_qa',
    'professional_medicine': 'document_qa',
    'clinical_knowledge': 'document_qa',
    'medical_genetics': 'document_qa',
}


@dataclass
class MMLUTask:
    """A single MMLU task."""
    question: str
    choices: List[str]
    answer: int  # Index of correct choice (0-3)
    subject: str
    regime: str
    
    def format_question(self) -> str:
        """Format question with choices."""
        formatted = f"{self.question}\n"
        for i, choice in enumerate(self.choices):
            formatted += f"{chr(65+i)}. {choice}\n"
        return formatted
    
    def get_answer_letter(self) -> str:
        """Get answer as letter (A, B, C, D)."""
        return chr(65 + self.answer)


class MMLUBenchmark:
    """
    MMLU benchmark loader and evaluator.
    
    In production, would load from HuggingFace datasets.
    Here we provide sample tasks for testing.
    """
    
    def __init__(self, n_samples_per_subject: int = 10):
        self.n_samples = n_samples_per_subject
        self.tasks: Dict[str, List[MMLUTask]] = {}
        self._load_sample_tasks()
    
    def _load_sample_tasks(self):
        """Load sample MMLU tasks for each subject."""
        # Sample tasks for testing (in production, load from dataset)
        sample_tasks = {
            'abstract_algebra': [
                MMLUTask(
                    question="Find all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.",
                    choices=["0", "1", "2", "0,1,2"],
                    answer=1,
                    subject="abstract_algebra",
                    regime="pure_qa"
                ),
                MMLUTask(
                    question="The polynomial x^3 + 2x + 1 is an element of which domain?",
                    choices=["Z_2[x]", "Z_3[x]", "Q[x]", "R[x]"],
                    answer=0,
                    subject="abstract_algebra",
                    regime="pure_qa"
                ),
            ],
            'college_mathematics': [
                MMLUTask(
                    question="What is the limit of (1 + 1/n)^n as n approaches infinity?",
                    choices=["0", "1", "e", "infinity"],
                    answer=2,
                    subject="college_mathematics",
                    regime="code_math"
                ),
                MMLUTask(
                    question="If f(x) = x^2, what is f'(3)?",
                    choices=["3", "6", "9", "2"],
                    answer=1,
                    subject="college_mathematics",
                    regime="code_math"
                ),
            ],
            'professional_law': [
                MMLUTask(
                    question="Under the parol evidence rule, evidence of prior agreements is:",
                    choices=[
                        "Always admissible",
                        "Inadmissible to contradict a written contract",
                        "Only admissible in criminal cases",
                        "Never relevant"
                    ],
                    answer=1,
                    subject="professional_law",
                    regime="document_qa"
                ),
            ],
        }
        
        for subject, tasks in sample_tasks.items():
            regime = MMLU_TO_REGIME.get(subject, 'pure_qa')
            for task in tasks:
                task.regime = regime
            self.tasks[subject] = tasks
    
    def get_tasks_by_regime(self, regime: str) -> List[MMLUTask]:
        """Get all tasks for a given regime."""
        result = []
        for subject, tasks in self.tasks.items():
            for task in tasks:
                if task.regime == regime:
                    result.append(task)
        return result
    
    def create_train_test_split(self, test_ratio: float = 0.3, seed: int = 42) -> Tuple[Dict, Dict]:
        """Create 70/30 train/test split."""
        rng = random.Random(seed)
        train_tasks = {}
        test_tasks = {}
        
        for subject, tasks in self.tasks.items():
            shuffled = tasks.copy()
            rng.shuffle(shuffled)
            
            split_idx = int(len(shuffled) * (1 - test_ratio))
            train_tasks[subject] = shuffled[:split_idx]
            test_tasks[subject] = shuffled[split_idx:]
        
        return train_tasks, test_tasks
    
    def evaluate(self, agent_fn, regime: str = None) -> Dict:
        """Evaluate an agent on MMLU tasks."""
        tasks = self.get_tasks_by_regime(regime) if regime else sum(self.tasks.values(), [])
        
        correct = 0
        total = 0
        
        for task in tasks:
            response = agent_fn(task.format_question())
            predicted = response.strip().upper()
            
            # Check if prediction matches answer
            if task.get_answer_letter() in predicted:
                correct += 1
            total += 1
        
        return {
            'accuracy': correct / total if total > 0 else 0,
            'correct': correct,
            'total': total,
            'regime': regime,
        }
