"""
Task system for LLM agent competitions.

Tasks are organized by type:
- Math: Arithmetic, algebra, word problems (objective evaluation) - 100 tasks
- Coding: Bug fixes, implementations (test case evaluation) - 50 tasks
- Logic: Puzzles, deduction (objective evaluation) - 50 tasks
- Language: Summarization, creative writing (LLM-as-judge evaluation) - 50 tasks

Total: 250 tasks for statistical power
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any
from enum import Enum
import random
import re
import ast
import math


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
        test_code: For coding tasks, the test code to run
    """
    id: str
    task_type: TaskType
    prompt: str
    ground_truth: Optional[str] = None
    difficulty: int = 1
    evaluator: Optional[Callable[[str, str], float]] = None
    test_code: Optional[str] = None
    llm_judge: Any = None  # Will be set by competition engine for language tasks

    def evaluate(self, response: str) -> float:
        """
        Evaluate a response to this task.

        Returns:
            float: Score between 0 and 1
        """
        if self.evaluator:
            return self.evaluator(response, self.ground_truth)

        if self.task_type == TaskType.CODING and self.test_code:
            return self._evaluate_code(response)

        if self.ground_truth is not None:
            return self._exact_match(response, self.ground_truth)

        # For language tasks, return 0.5 as placeholder
        # (actual LLM-as-judge evaluation happens in competition.py)
        return 0.5

    def _exact_match(self, response: str, ground_truth: str) -> float:
        """Check if response contains the ground truth answer."""
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
                    # Check last number in response against expected
                    resp_val = float(response_nums[-1])
                    truth_val = float(truth_nums[0])
                    if abs(resp_val - truth_val) < 0.01:
                        return 1.0
                    # Partial credit for close answers
                    if abs(resp_val - truth_val) / max(abs(truth_val), 1) < 0.1:
                        return 0.5
                except ValueError:
                    pass

        return 0.0

    def _evaluate_code(self, response: str) -> float:
        """
        Safely evaluate code response by extracting and running the function.
        
        Returns 1.0 if test passes, 0.0 otherwise.
        """
        try:
            # Extract code block if present
            code = self._extract_code(response)
            if not code:
                return 0.0

            # Create safe execution environment
            safe_globals = {
                '__builtins__': {
                    'len': len, 'range': range, 'list': list, 'dict': dict,
                    'set': set, 'str': str, 'int': int, 'float': float,
                    'bool': bool, 'True': True, 'False': False, 'None': None,
                    'abs': abs, 'min': min, 'max': max, 'sum': sum,
                    'sorted': sorted, 'reversed': reversed, 'enumerate': enumerate,
                    'zip': zip, 'map': map, 'filter': filter,
                    'isinstance': isinstance, 'type': type,
                }
            }
            local_vars = {}

            # Execute the function definition
            exec(code, safe_globals, local_vars)

            # Execute the test
            test_result = eval(self.test_code, safe_globals, local_vars)

            return 1.0 if test_result else 0.0

        except Exception as e:
            # Any error means the code doesn't work
            return 0.0

    def _extract_code(self, response: str) -> Optional[str]:
        """Extract Python code from response."""
        # Try to find code block
        code_patterns = [
            r'```python\n(.*?)```',
            r'```\n(.*?)```',
            r'def \w+\(.*?\):.*?(?=\n\n|\Z)',
        ]
        
        for pattern in code_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                return matches[0].strip()
        
        # If no code block, try to find function definition
        if 'def ' in response:
            lines = response.split('\n')
            code_lines = []
            in_function = False
            indent_level = 0
            
            for line in lines:
                if line.strip().startswith('def '):
                    in_function = True
                    indent_level = len(line) - len(line.lstrip())
                    code_lines.append(line)
                elif in_function:
                    if line.strip() and not line.startswith(' ' * (indent_level + 1)):
                        if not line.strip().startswith('#'):
                            break
                    code_lines.append(line)
            
            if code_lines:
                return '\n'.join(code_lines)
        
        return None


class TaskPool:
    """
    Pool of 250 tasks for agent competitions.

    Distribution:
    - Math: 100 tasks (40% - most objective)
    - Coding: 50 tasks (20%)
    - Logic: 50 tasks (20%)
    - Language: 50 tasks (20% - requires LLM-as-judge)
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        
        self.tasks: Dict[TaskType, List[Task]] = {
            TaskType.MATH: [],
            TaskType.CODING: [],
            TaskType.LOGIC: [],
            TaskType.LANGUAGE: [],
        }
        self._generate_tasks()

    def _generate_tasks(self):
        """Generate full task pool."""
        self._generate_math_tasks()      # 100 tasks
        self._generate_coding_tasks()    # 50 tasks
        self._generate_logic_tasks()     # 50 tasks
        self._generate_language_tasks()  # 50 tasks

    def _generate_math_tasks(self):
        """Generate 100 math tasks with ground truth."""
        task_id = 0
        
        # === Basic Arithmetic (30 tasks) ===
        for i in range(30):
            a = random.randint(10, 999)
            b = random.randint(10, 999)
            op = random.choice(['+', '-', '*'])
            
            if op == '+':
                answer = a + b
            elif op == '-':
                answer = a - b
            else:
                answer = a * b

            self.tasks[TaskType.MATH].append(Task(
                id=f"math_{task_id}",
                task_type=TaskType.MATH,
                prompt=f"Calculate: {a} {op} {b}. Give only the numerical answer.",
                ground_truth=str(answer),
                difficulty=1 if op in ['+', '-'] else 2
            ))
            task_id += 1

        # === Division and Decimals (15 tasks) ===
        for i in range(15):
            b = random.randint(2, 20)
            a = b * random.randint(5, 50)  # Ensure clean division
            answer = a // b

            self.tasks[TaskType.MATH].append(Task(
                id=f"math_{task_id}",
                task_type=TaskType.MATH,
                prompt=f"Calculate: {a} ÷ {b}. Give only the numerical answer.",
                ground_truth=str(answer),
                difficulty=2
            ))
            task_id += 1

        # === Percentages (15 tasks) ===
        percentages = [10, 20, 25, 50, 75, 15, 30, 5, 40, 60, 80, 90, 12, 33, 66]
        for i, pct in enumerate(percentages):
            base = random.randint(100, 1000)
            answer = int(base * pct / 100)

            self.tasks[TaskType.MATH].append(Task(
                id=f"math_{task_id}",
                task_type=TaskType.MATH,
                prompt=f"What is {pct}% of {base}? Give only the numerical answer.",
                ground_truth=str(answer),
                difficulty=2
            ))
            task_id += 1

        # === Word Problems (25 tasks) ===
        word_problems = [
            ("A store has 45 apples. If 12 are sold and 8 more arrive, how many apples are there?", "41"),
            ("A train travels 60 miles per hour. How far does it travel in 2.5 hours?", "150"),
            ("If 3 shirts cost $45, how much do 5 shirts cost?", "75"),
            ("A rectangle has length 8 and width 5. What is its area?", "40"),
            ("If you have 100 cookies and give away 1/4 of them, how many do you have left?", "75"),
            ("A book costs $24. If it's 20% off, what's the sale price?", "19.2"),
            ("There are 36 students in a class. If 2/3 are girls, how many boys are there?", "12"),
            ("A car travels 240 miles using 8 gallons of gas. What is the miles per gallon?", "30"),
            ("If a pizza is cut into 8 slices and you eat 3, what fraction is left?", "5/8"),
            ("A store sells 5 apples for $3. How much do 15 apples cost?", "9"),
            ("The temperature was -5°C. It rose by 12°C. What is it now?", "7"),
            ("A train leaves at 2:30 PM and arrives at 5:45 PM. How many minutes is the journey?", "195"),
            ("If 4 workers can build a wall in 6 days, how many days for 8 workers?", "3"),
            ("A circle has radius 7. What is its diameter?", "14"),
            ("John has $50. He spends 30% on food. How much does he have left?", "35"),
            ("A bag has 5 red and 3 blue marbles. What fraction are red?", "5/8"),
            ("If x + 7 = 15, what is x?", "8"),
            ("A square has perimeter 24. What is its area?", "36"),
            ("The average of 10, 20, and 30 is what?", "20"),
            ("If 2x = 18, what is x?", "9"),
            ("A shop gives 15% discount on $80. What's the final price?", "68"),
            ("How many seconds are in 2.5 minutes?", "150"),
            ("A triangle has sides 3, 4, and 5. Is it a right triangle? Answer 1 for yes, 0 for no.", "1"),
            ("If you double 17 and subtract 4, what do you get?", "30"),
            ("A cube has side length 3. What is its volume?", "27"),
        ]
        
        for problem, answer in word_problems:
            self.tasks[TaskType.MATH].append(Task(
                id=f"math_{task_id}",
                task_type=TaskType.MATH,
                prompt=problem + " Give only the numerical answer.",
                ground_truth=answer,
                difficulty=3
            ))
            task_id += 1

        # === Algebra (15 tasks) ===
        algebra_tasks = [
            ("Solve for x: 3x + 5 = 20", "5"),
            ("Solve for x: 2x - 8 = 12", "10"),
            ("Solve for x: x/4 = 7", "28"),
            ("Solve for x: 5x = 45", "9"),
            ("Solve for x: x + x + x = 21", "7"),
            ("If y = 2x + 3 and x = 4, what is y?", "11"),
            ("Solve for x: 2(x + 3) = 16", "5"),
            ("What is 3² + 4²?", "25"),
            ("If a = 5 and b = 3, what is a² - b²?", "16"),
            ("Solve: (10 + 5) × 2", "30"),
            ("What is the square root of 144?", "12"),
            ("If f(x) = x² + 1, what is f(3)?", "10"),
            ("Solve: 100 - 7 × 10 + 5", "35"),
            ("What is 2³ × 3?", "24"),
            ("If x² = 49, what is x (positive value)?", "7"),
        ]
        
        for problem, answer in algebra_tasks:
            self.tasks[TaskType.MATH].append(Task(
                id=f"math_{task_id}",
                task_type=TaskType.MATH,
                prompt=problem + " Give only the numerical answer.",
                ground_truth=answer,
                difficulty=4
            ))
            task_id += 1

    def _generate_coding_tasks(self):
        """Generate 50 coding tasks with test cases."""
        coding_tasks = [
            # === Easy (15 tasks) ===
            {"prompt": "Write a Python function `reverse_string(s)` that reverses a string.",
             "test": "reverse_string('hello') == 'olleh'", "difficulty": 1},
            {"prompt": "Write a Python function `double(n)` that returns n * 2.",
             "test": "double(5) == 10 and double(-3) == -6", "difficulty": 1},
            {"prompt": "Write a Python function `is_even(n)` that returns True if n is even.",
             "test": "is_even(4) == True and is_even(7) == False", "difficulty": 1},
            {"prompt": "Write a Python function `square(n)` that returns n squared.",
             "test": "square(5) == 25 and square(-3) == 9", "difficulty": 1},
            {"prompt": "Write a Python function `first_element(lst)` that returns the first element.",
             "test": "first_element([1,2,3]) == 1", "difficulty": 1},
            {"prompt": "Write a Python function `last_element(lst)` that returns the last element.",
             "test": "last_element([1,2,3]) == 3", "difficulty": 1},
            {"prompt": "Write a Python function `list_length(lst)` that returns the length of a list.",
             "test": "list_length([1,2,3,4]) == 4", "difficulty": 1},
            {"prompt": "Write a Python function `add_one(n)` that returns n + 1.",
             "test": "add_one(5) == 6 and add_one(-1) == 0", "difficulty": 1},
            {"prompt": "Write a Python function `to_upper(s)` that converts a string to uppercase.",
             "test": "to_upper('hello') == 'HELLO'", "difficulty": 1},
            {"prompt": "Write a Python function `to_lower(s)` that converts a string to lowercase.",
             "test": "to_lower('HELLO') == 'hello'", "difficulty": 1},
            {"prompt": "Write a Python function `abs_value(n)` that returns the absolute value.",
             "test": "abs_value(-5) == 5 and abs_value(3) == 3", "difficulty": 1},
            {"prompt": "Write a Python function `max_of_two(a, b)` that returns the larger number.",
             "test": "max_of_two(3, 7) == 7 and max_of_two(10, 5) == 10", "difficulty": 1},
            {"prompt": "Write a Python function `min_of_two(a, b)` that returns the smaller number.",
             "test": "min_of_two(3, 7) == 3 and min_of_two(10, 5) == 5", "difficulty": 1},
            {"prompt": "Write a Python function `concat(a, b)` that concatenates two strings.",
             "test": "concat('hello', 'world') == 'helloworld'", "difficulty": 1},
            {"prompt": "Write a Python function `is_positive(n)` that returns True if n > 0.",
             "test": "is_positive(5) == True and is_positive(-3) == False and is_positive(0) == False", "difficulty": 1},
            
            # === Medium (20 tasks) ===
            {"prompt": "Write a Python function `is_palindrome(s)` that returns True if s is a palindrome.",
             "test": "is_palindrome('racecar') == True and is_palindrome('hello') == False", "difficulty": 2},
            {"prompt": "Write a Python function `fibonacci(n)` that returns the nth Fibonacci number (0-indexed, fib(0)=0, fib(1)=1).",
             "test": "fibonacci(10) == 55 and fibonacci(0) == 0", "difficulty": 2},
            {"prompt": "Write a Python function `factorial(n)` that returns n factorial.",
             "test": "factorial(5) == 120 and factorial(0) == 1", "difficulty": 2},
            {"prompt": "Write a Python function `sum_list(lst)` that returns the sum of all elements.",
             "test": "sum_list([1,2,3,4,5]) == 15", "difficulty": 2},
            {"prompt": "Write a Python function `count_vowels(s)` that counts vowels in a string.",
             "test": "count_vowels('hello') == 2 and count_vowels('xyz') == 0", "difficulty": 2},
            {"prompt": "Write a Python function `reverse_list(lst)` that reverses a list.",
             "test": "reverse_list([1,2,3]) == [3,2,1]", "difficulty": 2},
            {"prompt": "Write a Python function `is_prime(n)` that returns True if n is prime.",
             "test": "is_prime(7) == True and is_prime(4) == False and is_prime(2) == True", "difficulty": 3},
            {"prompt": "Write a Python function `remove_duplicates(lst)` that removes duplicates, preserving order.",
             "test": "remove_duplicates([1,2,2,3,1]) == [1,2,3]", "difficulty": 3},
            {"prompt": "Write a Python function `flatten(lst)` that flattens a nested list one level.",
             "test": "flatten([[1,2],[3,4]]) == [1,2,3,4]", "difficulty": 3},
            {"prompt": "Write a Python function `find_max(lst)` that finds the maximum element.",
             "test": "find_max([3,1,4,1,5,9]) == 9", "difficulty": 2},
            {"prompt": "Write a Python function `find_min(lst)` that finds the minimum element.",
             "test": "find_min([3,1,4,1,5,9]) == 1", "difficulty": 2},
            {"prompt": "Write a Python function `word_count(s)` that counts words in a string.",
             "test": "word_count('hello world') == 2", "difficulty": 2},
            {"prompt": "Write a Python function `char_count(s)` that returns a dict of character counts.",
             "test": "char_count('aab') == {'a': 2, 'b': 1}", "difficulty": 3},
            {"prompt": "Write a Python function `average(lst)` that returns the average of a list.",
             "test": "average([1,2,3,4,5]) == 3.0", "difficulty": 2},
            {"prompt": "Write a Python function `power(base, exp)` that returns base to the power of exp.",
             "test": "power(2, 3) == 8 and power(5, 0) == 1", "difficulty": 2},
            {"prompt": "Write a Python function `gcd(a, b)` that returns the greatest common divisor.",
             "test": "gcd(12, 8) == 4 and gcd(17, 5) == 1", "difficulty": 3},
            {"prompt": "Write a Python function `lcm(a, b)` that returns the least common multiple.",
             "test": "lcm(4, 6) == 12", "difficulty": 3},
            {"prompt": "Write a Python function `is_sorted(lst)` that returns True if list is sorted ascending.",
             "test": "is_sorted([1,2,3]) == True and is_sorted([3,1,2]) == False", "difficulty": 2},
            {"prompt": "Write a Python function `second_largest(lst)` that returns the second largest element.",
             "test": "second_largest([1,2,3,4,5]) == 4", "difficulty": 3},
            {"prompt": "Write a Python function `rotate_left(lst, k)` that rotates list left by k positions.",
             "test": "rotate_left([1,2,3,4,5], 2) == [3,4,5,1,2]", "difficulty": 3},
            
            # === Hard (15 tasks) ===
            {"prompt": "Write a Python function `find_duplicates(lst)` that returns a list of duplicate elements.",
             "test": "set(find_duplicates([1,2,2,3,3,3])) == {2, 3}", "difficulty": 4},
            {"prompt": "Write a Python function `merge_sorted(a, b)` that merges two sorted lists into one sorted list.",
             "test": "merge_sorted([1,3,5], [2,4,6]) == [1,2,3,4,5,6]", "difficulty": 4},
            {"prompt": "Write a Python function `binary_search(lst, target)` that returns the index of target or -1.",
             "test": "binary_search([1,2,3,4,5], 3) == 2 and binary_search([1,2,3], 5) == -1", "difficulty": 4},
            {"prompt": "Write a Python function `longest_word(s)` that returns the longest word in a sentence.",
             "test": "longest_word('The quick brown fox') == 'quick' or longest_word('The quick brown fox') == 'brown'", "difficulty": 3},
            {"prompt": "Write a Python function `anagram_check(s1, s2)` that returns True if s1 and s2 are anagrams.",
             "test": "anagram_check('listen', 'silent') == True and anagram_check('hello', 'world') == False", "difficulty": 3},
            {"prompt": "Write a Python function `pascal_row(n)` that returns the nth row of Pascal's triangle (0-indexed).",
             "test": "pascal_row(4) == [1, 4, 6, 4, 1]", "difficulty": 4},
            {"prompt": "Write a Python function `all_subsets(lst)` that returns all subsets of a list.",
             "test": "len(all_subsets([1,2])) == 4", "difficulty": 5},
            {"prompt": "Write a Python function `valid_parentheses(s)` that returns True if parentheses are balanced.",
             "test": "valid_parentheses('(())') == True and valid_parentheses('(()') == False", "difficulty": 4},
            {"prompt": "Write a Python function `string_compress(s)` that compresses 'aabccc' to 'a2b1c3'.",
             "test": "string_compress('aabccc') == 'a2b1c3'", "difficulty": 4},
            {"prompt": "Write a Python function `matrix_transpose(m)` that transposes a 2D matrix.",
             "test": "matrix_transpose([[1,2],[3,4]]) == [[1,3],[2,4]]", "difficulty": 4},
            {"prompt": "Write a Python function `spiral_order(matrix)` that returns elements in spiral order.",
             "test": "spiral_order([[1,2,3],[4,5,6],[7,8,9]]) == [1,2,3,6,9,8,7,4,5]", "difficulty": 5},
            {"prompt": "Write a Python function `longest_common_prefix(strs)` that finds the longest common prefix.",
             "test": "longest_common_prefix(['flower','flow','flight']) == 'fl'", "difficulty": 4},
            {"prompt": "Write a Python function `two_sum(nums, target)` that returns indices of two numbers that add to target.",
             "test": "sorted(two_sum([2,7,11,15], 9)) == [0,1]", "difficulty": 4},
            {"prompt": "Write a Python function `roman_to_int(s)` that converts a Roman numeral to integer.",
             "test": "roman_to_int('III') == 3 and roman_to_int('IV') == 4 and roman_to_int('IX') == 9", "difficulty": 4},
            {"prompt": "Write a Python function `count_islands(grid)` where grid is 2D list of 0s and 1s. Count connected 1s.",
             "test": "count_islands([[1,1,0],[0,1,0],[0,0,1]]) == 2", "difficulty": 5},
        ]

        for i, task in enumerate(coding_tasks):
            self.tasks[TaskType.CODING].append(Task(
                id=f"coding_{i}",
                task_type=TaskType.CODING,
                prompt=task["prompt"],
                ground_truth=task["test"],
                test_code=task["test"],
                difficulty=task["difficulty"]
            ))

    def _generate_logic_tasks(self):
        """Generate 50 logic tasks with ground truth."""
        logic_tasks = [
            # === Syllogisms (10 tasks) ===
            ("If all cats are mammals, and all mammals are animals, are all cats animals? Answer yes or no.", "yes"),
            ("If all roses are flowers, and some flowers are red, are all roses red? Answer yes or no.", "no"),
            ("If no fish are mammals, and all whales are mammals, are any whales fish? Answer yes or no.", "no"),
            ("If some birds can fly, and penguins are birds, can all penguins fly? Answer yes or no.", "no"),
            ("If all squares are rectangles, and all rectangles have 4 sides, do all squares have 4 sides? Answer yes or no.", "yes"),
            ("If no reptiles are warm-blooded, and all snakes are reptiles, are snakes warm-blooded? Answer yes or no.", "no"),
            ("All dogs are mammals. Buddy is a dog. Is Buddy a mammal? Answer yes or no.", "yes"),
            ("Some fruits are yellow. Bananas are yellow. Are bananas fruits? Answer yes, no, or cannot determine.", "cannot determine"),
            ("If all A are B, and all B are C, and X is an A, is X a C? Answer yes or no.", "yes"),
            ("No students failed. John is a student. Did John fail? Answer yes or no.", "no"),
            
            # === Conditional Logic (10 tasks) ===
            ("If it's raining, the ground is wet. The ground is wet. Is it definitely raining? Answer yes or no.", "no"),
            ("If it's raining, the ground is wet. It's not raining. Is the ground definitely dry? Answer yes or no.", "no"),
            ("If A then B. A is true. Is B true? Answer yes or no.", "yes"),
            ("If A then B. B is false. Is A true? Answer yes or no.", "no"),
            ("If you study, you pass. You passed. Did you definitely study? Answer yes or no.", "no"),
            ("If it's Monday, the store is closed. It's not Monday. Is the store open? Answer yes, no, or cannot determine.", "cannot determine"),
            ("P implies Q. Not Q. What is P? Answer true, false, or unknown.", "false"),
            ("If X > 5, then Y < 10. Y = 15. What can we say about X? Answer: X <= 5, X > 5, or unknown.", "x <= 5"),
            ("Either it's sunny or it's cloudy (not both). It's not sunny. Is it cloudy? Answer yes or no.", "yes"),
            ("If A and B, then C. C is false. What can we say? Answer: A is false, B is false, or A or B is false.", "a or b is false"),
            
            # === Sequences (10 tasks) ===
            ("What comes next in the sequence: 2, 4, 8, 16, ?", "32"),
            ("What comes next: 1, 1, 2, 3, 5, 8, ?", "13"),
            ("What comes next: 3, 6, 9, 12, ?", "15"),
            ("What comes next: 1, 4, 9, 16, 25, ?", "36"),
            ("What comes next: 2, 6, 12, 20, 30, ?", "42"),
            ("What comes next: A, C, E, G, ? (give the letter)", "i"),
            ("What comes next: 100, 50, 25, ?", "12.5"),
            ("What comes next: 1, 2, 4, 7, 11, ?", "16"),
            ("What comes next: 0, 1, 1, 2, 4, 7, ?", "13"),
            ("What is the 10th term of: 2, 4, 6, 8, ...?", "20"),
            
            # === Puzzles (10 tasks) ===
            ("If A is taller than B, and B is taller than C, who is the shortest? Answer A, B, or C.", "c"),
            ("In a race, if you pass the person in 2nd place, what place are you in?", "2"),
            ("How many months have 28 days?", "12"),
            ("If there are 3 apples and you take away 2, how many do you have?", "2"),
            ("A farmer has 17 sheep. All but 9 die. How many are left?", "9"),
            ("What weighs more, a pound of feathers or a pound of bricks?", "same"),
            ("If you have only one match and enter a dark room with a candle, oil lamp, and fireplace, what do you light first?", "match"),
            ("How many times can you subtract 5 from 25?", "1"),
            ("Tom's mother has 4 children: April, May, June, and ? What is the 4th child's name?", "tom"),
            ("A clerk at a butcher shop is 5'10\" tall. What does he weigh?", "meat"),
            
            # === Math Logic (10 tasks) ===
            ("Is 0 even or odd? Answer even or odd.", "even"),
            ("If x + 2 = 5, what is x?", "3"),
            ("True or False: The square root of 2 is rational.", "false"),
            ("How many prime numbers are there between 1 and 10?", "4"),
            ("What is the only even prime number?", "2"),
            ("If a triangle has angles 60°, 60°, what is the third angle?", "60"),
            ("True or False: All prime numbers are odd.", "false"),
            ("What is the remainder when 17 is divided by 5?", "2"),
            ("Is -5 greater than or less than -3? Answer greater or less.", "less"),
            ("True or False: The sum of two negative numbers is negative.", "true"),
        ]

        for i, (question, answer) in enumerate(logic_tasks):
            self.tasks[TaskType.LOGIC].append(Task(
                id=f"logic_{i}",
                task_type=TaskType.LOGIC,
                prompt=question,
                ground_truth=answer,
                difficulty=2 if i < 20 else 3
            ))

    def _generate_language_tasks(self):
        """Generate 50 language tasks (LLM-as-judge evaluation)."""
        language_tasks = [
            # === Summarization (10 tasks) ===
            ("Summarize the concept of photosynthesis in exactly 2 sentences.", 3),
            ("Summarize what artificial intelligence is in one paragraph.", 3),
            ("Explain the water cycle in 3 sentences.", 2),
            ("Summarize the plot of Romeo and Juliet in 2 sentences.", 3),
            ("Describe what climate change is in simple terms.", 2),
            ("Summarize what democracy means in 2 sentences.", 3),
            ("Explain gravity in one sentence.", 3),
            ("Describe the internet in 2 sentences.", 2),
            ("Summarize evolution by natural selection in 3 sentences.", 3),
            ("Explain what a computer virus is in simple terms.", 2),
            
            # === Creative Writing (15 tasks) ===
            ("Write a haiku about artificial intelligence.", 3),
            ("Write a two-line poem about the ocean.", 2),
            ("Create a metaphor comparing life to a river.", 3),
            ("Write a limerick about a programmer.", 4),
            ("Describe a sunset without using the words 'sun', 'orange', or 'sky'.", 4),
            ("Write a short dialogue (4 lines) between a cat and a dog.", 3),
            ("Create a one-sentence story with a twist ending.", 4),
            ("Write a personification of the wind.", 3),
            ("Compose a short riddle with the answer being 'time'.", 4),
            ("Write a motivational quote about perseverance.", 2),
            ("Describe the taste of chocolate to someone who has never tasted it.", 4),
            ("Write a short eulogy for a houseplant.", 3),
            ("Create an alliteration using at least 5 words starting with 'S'.", 3),
            ("Write a paradox about silence.", 4),
            ("Compose a short thank-you note from a robot to its creator.", 3),
            
            # === Explanation (10 tasks) ===
            ("Explain what a black hole is to a 5-year-old.", 3),
            ("Describe the color blue without using the word 'blue' or any color names.", 4),
            ("Explain why the sky appears blue to a curious child.", 3),
            ("Describe how a bicycle works to someone who has never seen one.", 3),
            ("Explain what music is to someone who has never heard it.", 4),
            ("Describe snow to someone who has never experienced winter.", 3),
            ("Explain the concept of time to a child.", 4),
            ("Describe what friendship means without using the word 'friend'.", 4),
            ("Explain why we dream in simple terms.", 3),
            ("Describe the sensation of falling asleep.", 4),
            
            # === Persuasion (10 tasks) ===
            ("Write a persuasive sentence arguing for the importance of exercise.", 2),
            ("Convince someone to read more books in 2-3 sentences.", 3),
            ("Write a compelling argument for learning a new language.", 3),
            ("Persuade someone to try a new food they're hesitant about.", 3),
            ("Write 2 sentences arguing for the importance of sleep.", 2),
            ("Convince someone to take a walk outside.", 2),
            ("Write an argument for why we should protect the environment.", 3),
            ("Persuade someone to learn to cook in 2 sentences.", 3),
            ("Write a brief argument for the value of failure.", 4),
            ("Convince someone that patience is a virtue.", 3),
            
            # === Analysis (5 tasks) ===
            ("Identify one strength and one weakness of social media in 2 sentences.", 3),
            ("Compare and contrast cats and dogs as pets in 3 sentences.", 3),
            ("Analyze why people procrastinate in 2-3 sentences.", 4),
            ("Explain both sides of the debate on homework in 3 sentences.", 3),
            ("Discuss one benefit and one drawback of technology in 2 sentences.", 3),
        ]

        for i, (prompt, difficulty) in enumerate(language_tasks):
            self.tasks[TaskType.LANGUAGE].append(Task(
                id=f"language_{i}",
                task_type=TaskType.LANGUAGE,
                prompt=prompt,
                ground_truth=None,  # Uses LLM-as-judge
                difficulty=difficulty
            ))

    def sample(self, n: int = 1, task_type: TaskType = None) -> List[Task]:
        """Sample n tasks, optionally from a specific type."""
        if task_type:
            pool = self.tasks[task_type]
        else:
            pool = [task for tasks in self.tasks.values() for task in tasks]

        return random.sample(pool, min(n, len(pool)))

    def sample_balanced(self, n_per_type: int = 5) -> List[Task]:
        """Sample equal number of tasks from each type."""
        tasks = []
        for task_type in TaskType:
            tasks.extend(self.sample(n_per_type, task_type))
        random.shuffle(tasks)
        return tasks

    def get_all(self, task_type: TaskType = None) -> List[Task]:
        """Get all tasks, optionally filtered by type."""
        if task_type:
            return self.tasks[task_type]
        return [task for tasks in self.tasks.values() for task in tasks]

    def get_stats(self) -> Dict[str, int]:
        """Get task count statistics."""
        return {
            task_type.value: len(tasks)
            for task_type, tasks in self.tasks.items()
        }

    def __len__(self) -> int:
        return sum(len(tasks) for tasks in self.tasks.values())

    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"TaskPool({len(self)} tasks: {stats})"
