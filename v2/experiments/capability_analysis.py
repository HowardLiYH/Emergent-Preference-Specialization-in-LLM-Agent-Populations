"""
Capability Unlocking Analysis.

Implements Prof. Finn's suggestion (Priority 2) to identify tasks
where ONLY CSE succeeds and Independent fails.

This is the strongest commercial argument: some tasks are simply
impossible without specialized agents.
"""

import json
import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime


# Task definitions with difficulty gradients
CAPABILITY_TASKS = {
    # ===== Easy Tasks (Both should succeed) =====
    'simple_factual': {
        'regime': 'pure_qa',
        'difficulty': 0.2,
        'description': 'Simple factual recall',
        'examples': [
            ("What is the capital of France?", "Paris"),
            ("Who wrote Romeo and Juliet?", "Shakespeare"),
            ("What is 2 + 2?", "4"),
        ],
    },
    'basic_math': {
        'regime': 'code_math',
        'difficulty': 0.3,
        'description': 'Basic arithmetic and algebra',
        'examples': [
            ("Solve: 3x + 5 = 20", "x = 5"),
            ("Calculate 15% of 80", "12"),
            ("What is the square root of 144?", "12"),
        ],
    },
    
    # ===== Medium Tasks (Specialists have advantage) =====
    'intermediate_code': {
        'regime': 'code_math',
        'difficulty': 0.5,
        'description': 'Code understanding and simple generation',
        'examples': [
            ("Write a function to reverse a string in Python", "def reverse(s): return s[::-1]"),
            ("What does this code do: [x*2 for x in range(5)]?", "Creates list [0,2,4,6,8]"),
        ],
    },
    'document_understanding': {
        'regime': 'document_qa',
        'difficulty': 0.5,
        'description': 'Understanding document context',
        'examples': [
            ("Given a contract, identify the termination clause", "Section about ending agreement"),
            ("Summarize the key points of this report", "Main findings and conclusions"),
        ],
    },
    
    # ===== Hard Tasks (Specialists required) =====
    'complex_code': {
        'regime': 'code_math',
        'difficulty': 0.8,
        'description': 'Complex algorithm implementation',
        'examples': [
            ("Implement a balanced binary search tree insertion", "AVL/Red-black tree code"),
            ("Write an efficient graph traversal for shortest path", "Dijkstra/BFS implementation"),
            ("Debug this recursive function with off-by-one error", "Corrected recursion"),
        ],
    },
    'multi_step_reasoning': {
        'regime': 'document_qa',
        'difficulty': 0.7,
        'description': 'Multi-hop reasoning across documents',
        'examples': [
            ("Cross-reference these 3 documents to find discrepancies", "List of inconsistencies"),
            ("Trace the chain of events from these reports", "Timeline of events"),
        ],
    },
    'realtime_api': {
        'regime': 'realtime_data',
        'difficulty': 0.9,
        'description': 'Real-time data API integration',
        'examples': [
            ("Get current weather in Tokyo and convert to Fahrenheit", "Temperature with conversion"),
            ("Fetch latest exchange rate EUR/USD and calculate $500", "Converted amount"),
        ],
    },
    'chart_interpretation': {
        'regime': 'chart_analysis',
        'difficulty': 0.8,
        'description': 'Complex visual/chart analysis',
        'examples': [
            ("Identify the trend and anomalies in this time series chart", "Trend + outliers"),
            ("Compare these two bar charts and explain differences", "Comparative analysis"),
        ],
    },
}


@dataclass
class CapabilityResult:
    """Result of capability analysis for a single task."""
    task_name: str
    regime: str
    difficulty: float
    cse_accuracy: float
    cse_attempts: int
    ind_accuracy: float
    ind_attempts: int
    capability_unlocked: bool
    accuracy_delta: float
    value_proposition: str


def generate_task_for_regime(regime: str, difficulty: float, rng) -> Tuple[str, str]:
    """Generate a task for a specific regime and difficulty."""
    
    if regime == 'pure_qa':
        if difficulty < 0.4:
            questions = [
                ("What is the chemical symbol for gold?", "Au"),
                ("What year did World War II end?", "1945"),
                ("What is the largest planet in our solar system?", "Jupiter"),
            ]
        else:
            questions = [
                ("Explain the difference between mitosis and meiosis", "Cell division types"),
                ("What are the three branches of US government?", "Executive, Legislative, Judicial"),
            ]
    
    elif regime == 'code_math':
        if difficulty < 0.5:
            questions = [
                ("Calculate: (15 * 4) + (30 / 5)", "66"),
                ("What is the output of: print(len('hello'))", "5"),
            ]
        else:
            questions = [
                ("Write a function to find the nth Fibonacci number", "fib(n) implementation"),
                ("Implement binary search on a sorted list", "binary_search function"),
            ]
    
    elif regime == 'document_qa':
        questions = [
            ("Given this text, what is the main argument?", "Central thesis"),
            ("Identify the key entities mentioned in this passage", "Named entities"),
        ]
    
    elif regime == 'realtime_data':
        questions = [
            ("What is the current time in UTC?", "Current UTC time"),
            ("Get the latest Bitcoin price in USD", "BTC price"),
        ]
    
    elif regime == 'chart_analysis':
        questions = [
            ("Describe the trend shown in this visualization", "Trend description"),
            ("What percentage increase is shown in the bar chart?", "Percentage"),
        ]
    
    else:
        questions = [("Generic question", "Generic answer")]
    
    return rng.choice(questions)


def evaluate_on_task(
    agents,
    task: Tuple[str, str],
    regime: str,
    is_specialist: bool,
    llm_client,
    n_trials: int = 5
) -> Tuple[float, int]:
    """
    Evaluate agents on a task multiple times.
    
    Returns:
        Tuple of (accuracy, n_attempts)
    """
    successes = 0
    attempts = 0
    
    for _ in range(n_trials):
        for agent in agents:
            # Specialists use their best tool, generalists use L0
            if is_specialist and hasattr(agent, 'specialty') and agent.specialty == regime:
                tool = agent.select_tool(regime, None)
            else:
                tool = 'L0'  # Generalist default
            
            # Evaluate
            try:
                success = evaluate_single(
                    agent, tool, task[0], task[1], llm_client
                )
                if success:
                    successes += 1
                attempts += 1
            except Exception as e:
                print(f"Evaluation error: {e}")
                attempts += 1
    
    accuracy = successes / max(1, attempts)
    return accuracy, attempts


def evaluate_single(agent, tool, question, expected, llm_client) -> bool:
    """Evaluate a single agent response."""
    
    # Construct prompt based on tool level
    if tool == 'L0':
        prompt = f"Answer this question: {question}"
    elif tool == 'L1':
        prompt = f"Use Python to solve: {question}. Show your code and answer."
    elif tool == 'L2':
        prompt = f"Analyze this (use visual reasoning if needed): {question}"
    elif tool == 'L3':
        prompt = f"Search relevant documents and answer: {question}"
    else:
        prompt = f"Use web search if needed: {question}"
    
    try:
        response = llm_client.generate(prompt, max_tokens=500)
        # Simple check: does response contain expected answer?
        success = expected.lower() in response.lower()
        return success
    except Exception:
        return False


def analyze_capability_unlocking(
    cse_agents: list,
    ind_agents: list,
    llm_client,
    tasks: Dict = None,
    n_trials: int = 5,
    rng = None
) -> List[CapabilityResult]:
    """
    Find tasks where CSE succeeds and Independent fails.
    
    These represent "unlocked capabilities" worth the training cost.
    
    Args:
        cse_agents: Trained CSE specialists
        ind_agents: Independent training agents
        llm_client: LLM client for evaluation
        tasks: Task definitions (defaults to CAPABILITY_TASKS)
        n_trials: Number of trials per task
        rng: Random number generator
    
    Returns:
        List of CapabilityResult for each task
    """
    import random
    if rng is None:
        rng = random.Random(42)
    
    if tasks is None:
        tasks = CAPABILITY_TASKS
    
    results = []
    
    for task_name, config in tasks.items():
        regime = config['regime']
        difficulty = config['difficulty']
        
        print(f"Evaluating: {task_name} (difficulty={difficulty})")
        
        # Generate task
        if 'examples' in config and config['examples']:
            task = rng.choice(config['examples'])
        else:
            task = generate_task_for_regime(regime, difficulty, rng)
        
        # Evaluate CSE specialists
        cse_specialists = [
            a for a in cse_agents 
            if hasattr(a, 'specialty') and a.specialty == regime
        ]
        if not cse_specialists:
            cse_specialists = cse_agents[:3]  # Fallback to top agents
        
        cse_acc, cse_attempts = evaluate_on_task(
            cse_specialists, task, regime, True, llm_client, n_trials
        )
        
        # Evaluate Independent agents
        ind_acc, ind_attempts = evaluate_on_task(
            ind_agents, task, regime, False, llm_client, n_trials
        )
        
        # Determine if capability was unlocked
        capability_unlocked = (cse_acc > 0.7 and ind_acc < 0.5)
        accuracy_delta = cse_acc - ind_acc
        
        # Generate value proposition
        if capability_unlocked:
            value_prop = f"CSE enables {task_name} ({int(cse_acc*100)}% vs {int(ind_acc*100)}%)"
        elif accuracy_delta > 0.2:
            value_prop = f"CSE significantly improves {task_name} (+{int(accuracy_delta*100)}%)"
        elif accuracy_delta > 0:
            value_prop = f"CSE marginally improves {task_name}"
        else:
            value_prop = f"No advantage for {task_name}"
        
        results.append(CapabilityResult(
            task_name=task_name,
            regime=regime,
            difficulty=difficulty,
            cse_accuracy=round(cse_acc, 3),
            cse_attempts=cse_attempts,
            ind_accuracy=round(ind_acc, 3),
            ind_attempts=ind_attempts,
            capability_unlocked=capability_unlocked,
            accuracy_delta=round(accuracy_delta, 3),
            value_proposition=value_prop,
        ))
    
    return results


def summarize_capability_results(results: List[CapabilityResult]) -> Dict:
    """Generate summary statistics from capability analysis."""
    
    unlocked = [r for r in results if r.capability_unlocked]
    improved = [r for r in results if r.accuracy_delta > 0.2]
    
    return {
        'total_tasks': len(results),
        'capabilities_unlocked': len(unlocked),
        'significantly_improved': len(improved),
        'unlocked_tasks': [r.task_name for r in unlocked],
        'improved_tasks': [r.task_name for r in improved if r not in unlocked],
        'avg_accuracy_delta': sum(r.accuracy_delta for r in results) / len(results),
        'avg_cse_accuracy': sum(r.cse_accuracy for r in results) / len(results),
        'avg_ind_accuracy': sum(r.ind_accuracy for r in results) / len(results),
        'value_propositions': [r.value_proposition for r in unlocked],
    }


def save_capability_results(
    results: List[CapabilityResult],
    output_dir: str = "results/capability"
):
    """Save capability analysis results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Detailed results
    detailed = [
        {
            'task_name': r.task_name,
            'regime': r.regime,
            'difficulty': r.difficulty,
            'cse_accuracy': r.cse_accuracy,
            'ind_accuracy': r.ind_accuracy,
            'capability_unlocked': r.capability_unlocked,
            'accuracy_delta': r.accuracy_delta,
            'value_proposition': r.value_proposition,
        }
        for r in results
    ]
    
    with open(f"{output_dir}/detailed_results.json", 'w') as f:
        json.dump(detailed, f, indent=2)
    
    # Summary
    summary = summarize_capability_results(results)
    summary['timestamp'] = datetime.now().isoformat()
    
    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results saved to {output_dir}/")
    return summary


if __name__ == "__main__":
    # Example usage with mock data
    print("Capability Analysis Module")
    print("=" * 50)
    print(f"Defined tasks: {len(CAPABILITY_TASKS)}")
    for name, config in CAPABILITY_TASKS.items():
        print(f"  - {name}: regime={config['regime']}, difficulty={config['difficulty']}")
