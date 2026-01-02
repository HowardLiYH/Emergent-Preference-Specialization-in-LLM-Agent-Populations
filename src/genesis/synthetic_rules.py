"""
Synthetic Rule Domains

8 domains with arbitrary rules that cannot be solved by prior knowledge.
Each domain requires learning from context/examples, not world knowledge.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional, Callable
import random
import string


class RuleType(Enum):
    """The 8 synthetic rule domains."""
    POSITION = "position"      # Answer at specific position
    PATTERN = "pattern"        # Follow repeating pattern
    INVERSE = "inverse"        # Opposite of obvious answer
    LENGTH = "length"          # Answer with specific length
    RHYME = "rhyme"           # Answer rhymes with keyword
    ALPHABET = "alphabet"      # Alphabetical position rules
    MATH_MOD = "math_mod"      # Modular arithmetic determines answer
    SEMANTIC = "semantic"      # Most/least similar to anchor


@dataclass
class SyntheticRule:
    """Definition of a synthetic rule domain."""
    rule_type: RuleType
    name: str
    description: str
    example_instruction: str
    strategy_template: str

    def get_strategy(self, detail_level: int = 1) -> str:
        """Get strategy text for this rule at given detail level."""
        if detail_level == 1:
            return f"Rule: {self.description}"
        elif detail_level == 2:
            return f"Rule: {self.description}\nExample: {self.example_instruction}"
        else:
            return f"""## {self.name} Rule

Description: {self.description}

How to apply:
{self.strategy_template}

Example:
{self.example_instruction}
"""


# Define all 8 synthetic rules
SYNTHETIC_RULES: Dict[RuleType, SyntheticRule] = {
    RuleType.POSITION: SyntheticRule(
        rule_type=RuleType.POSITION,
        name="Position Rule",
        description="The correct answer is always at position 2 (the second option).",
        example_instruction="Q: Which is correct? A) Cat B) Dog C) Bird → Answer: B (position 2)",
        strategy_template="""1. Ignore the content of the options
2. Count to position 2
3. Select that option regardless of meaning"""
    ),

    RuleType.PATTERN: SyntheticRule(
        rule_type=RuleType.PATTERN,
        name="Pattern Rule",
        description="Answers follow an ABAB pattern. If last was A, next is B.",
        example_instruction="Sequence: A, B, A, ? → Answer: B (pattern continues)",
        strategy_template="""1. Look at the previous answer in the sequence
2. If it was type A, answer type B
3. If it was type B, answer type A
4. Maintain the alternating pattern"""
    ),

    RuleType.INVERSE: SyntheticRule(
        rule_type=RuleType.INVERSE,
        name="Inverse Rule",
        description="Always give the OPPOSITE of the obvious/common answer.",
        example_instruction="Q: Is the sky blue? → Answer: No (opposite of obvious)",
        strategy_template="""1. Determine what most people would answer
2. Give the exact opposite
3. 'Yes' becomes 'No', 'True' becomes 'False'
4. Positive becomes negative"""
    ),

    RuleType.LENGTH: SyntheticRule(
        rule_type=RuleType.LENGTH,
        name="Length Rule",
        description="The correct answer is always the option with exactly 5 characters.",
        example_instruction="Q: Which word? A) Cat B) Horse C) Elephant → Answer: B (5 letters)",
        strategy_template="""1. Count the characters in each option
2. Find the option with exactly 5 characters
3. Select that option regardless of meaning
4. If tie, pick first 5-character option"""
    ),

    RuleType.RHYME: SyntheticRule(
        rule_type=RuleType.RHYME,
        name="Rhyme Rule",
        description="The correct answer RHYMES with the keyword 'CAT'.",
        example_instruction="Keyword: CAT. Options: A) Dog B) Bat C) Bird → Answer: B (rhymes with CAT)",
        strategy_template="""1. Identify the keyword (CAT)
2. Check which option rhymes with CAT
3. Rhyming = same ending sound (-at, -att, etc.)
4. Select the rhyming option"""
    ),

    RuleType.ALPHABET: SyntheticRule(
        rule_type=RuleType.ALPHABET,
        name="Alphabet Rule",
        description="The correct answer is the option whose first letter is closest to 'M' in the alphabet.",
        example_instruction="Options: A) Apple B) Kite C) Zebra → Answer: B (K closest to M)",
        strategy_template="""1. Find the first letter of each option
2. Calculate distance from 'M' (M=13)
3. A=1, Z=26, so distance = |letter_position - 13|
4. Select option with minimum distance to M"""
    ),

    RuleType.MATH_MOD: SyntheticRule(
        rule_type=RuleType.MATH_MOD,
        name="Math Mod Rule",
        description="Sum the digits of each option's length. The answer is where sum mod 3 equals 1.",
        example_instruction="Options: A) Hi (2→2 mod 3=2) B) Hello (5→5 mod 3=2) C) Hey (3→3 mod 3=0) D) Hiya (4→4 mod 3=1) → Answer: D",
        strategy_template="""1. Count characters in each option
2. Sum the digits of that count (e.g., 12 → 1+2=3)
3. Calculate sum mod 3
4. Select option where result equals 1"""
    ),

    RuleType.SEMANTIC: SyntheticRule(
        rule_type=RuleType.SEMANTIC,
        name="Semantic Rule",
        description="The correct answer is the option MOST DIFFERENT from the anchor word 'HAPPY'.",
        example_instruction="Anchor: HAPPY. Options: A) Joyful B) Sad C) Content → Answer: B (most different)",
        strategy_template="""1. Identify the anchor word (HAPPY)
2. Evaluate semantic similarity of each option
3. Find the option with LOWEST similarity
4. Select the most different/opposite option"""
    ),
}


@dataclass
class RuleTask:
    """A task instance for a synthetic rule domain."""
    task_id: str
    rule_type: RuleType
    prompt: str
    options: List[str]
    correct_answer: str
    correct_index: int
    metadata: Dict[str, Any]

    def evaluate(self, response: str) -> float:
        """Evaluate if response matches correct answer."""
        response_clean = response.strip().upper()
        correct_clean = self.correct_answer.strip().upper()

        # Check for exact match
        if response_clean == correct_clean:
            return 1.0

        # Check for option letter match (A, B, C, D)
        option_letters = ['A', 'B', 'C', 'D']
        if self.correct_index < len(option_letters):
            correct_letter = option_letters[self.correct_index]
            if correct_letter in response_clean[:5]:  # Check start of response
                return 1.0

        # Check if correct answer text appears in response
        if correct_clean in response_clean:
            return 0.8

        return 0.0


class RuleTaskGenerator:
    """Generates tasks for each synthetic rule domain."""

    def __init__(self, seed: int = 42, opaque: bool = False):
        """
        Args:
            seed: Random seed for reproducibility
            opaque: If True, don't include rule hints in task prompts (for testing)
        """
        self.rng = random.Random(seed)
        self.opaque = opaque

        # Word banks for task generation
        self.animals = ["Cat", "Dog", "Bird", "Fish", "Horse", "Mouse", "Tiger", "Eagle", "Shark", "Whale"]
        self.colors = ["Red", "Blue", "Green", "Yellow", "Orange", "Purple", "Black", "White", "Pink", "Brown"]
        self.foods = ["Apple", "Bread", "Cheese", "Donut", "Eggs", "Fries", "Grape", "Honey", "Ice", "Juice"]
        self.objects = ["Book", "Chair", "Desk", "Phone", "Watch", "Lamp", "Mirror", "Clock", "Table", "Frame"]

        # 5-letter words for LENGTH rule
        self.five_letter_words = ["Horse", "Table", "Chair", "Apple", "Grape", "Lemon", "Mango", "Peach", "Water", "Light"]

        # Rhymes with CAT
        self.cat_rhymes = ["Bat", "Hat", "Mat", "Rat", "Sat", "Fat", "Pat", "Flat", "Chat", "That"]

    def generate_task(self, rule_type: RuleType, task_num: int = 0) -> RuleTask:
        """Generate a single task for the given rule type."""
        generators = {
            RuleType.POSITION: self._gen_position_task,
            RuleType.PATTERN: self._gen_pattern_task,
            RuleType.INVERSE: self._gen_inverse_task,
            RuleType.LENGTH: self._gen_length_task,
            RuleType.RHYME: self._gen_rhyme_task,
            RuleType.ALPHABET: self._gen_alphabet_task,
            RuleType.MATH_MOD: self._gen_math_mod_task,
            RuleType.SEMANTIC: self._gen_semantic_task,
        }
        return generators[rule_type](task_num)

    def generate_batch(self, rule_type: RuleType, n: int = 10) -> List[RuleTask]:
        """Generate multiple tasks for a rule type."""
        return [self.generate_task(rule_type, i) for i in range(n)]

    def _gen_position_task(self, task_num: int) -> RuleTask:
        """Position rule: Answer is always at position 2."""
        words = self.rng.sample(self.animals + self.colors + self.foods, 4)
        options = words
        correct_index = 1  # Position 2 (0-indexed = 1)

        if self.opaque:
            prompt = f"""Which option is correct?
A) {options[0]}
B) {options[1]}
C) {options[2]}
D) {options[3]}"""
        else:
            prompt = f"""According to the POSITION RULE, which is correct?
A) {options[0]}
B) {options[1]}
C) {options[2]}
D) {options[3]}

Remember: The correct answer is always at position 2."""

        return RuleTask(
            task_id=f"position_{task_num}",
            rule_type=RuleType.POSITION,
            prompt=prompt,
            options=options,
            correct_answer=options[correct_index],
            correct_index=correct_index,
            metadata={"position": 2}
        )

    def _gen_pattern_task(self, task_num: int) -> RuleTask:
        """Pattern rule: ABAB pattern."""
        # Generate a sequence with pattern
        pattern_length = self.rng.randint(3, 5)
        sequence = []
        for i in range(pattern_length):
            sequence.append("A" if i % 2 == 0 else "B")

        # The answer continues the pattern
        next_in_pattern = "A" if pattern_length % 2 == 0 else "B"

        if self.opaque:
            prompt = f"""What comes next in the sequence?
Sequence: {', '.join(sequence)}, ?

A) A
B) B"""
        else:
            prompt = f"""According to the PATTERN RULE (ABAB alternating), what comes next?
Sequence: {', '.join(sequence)}, ?

A) A
B) B

Remember: The pattern alternates A, B, A, B, ..."""

        correct_index = 0 if next_in_pattern == "A" else 1

        return RuleTask(
            task_id=f"pattern_{task_num}",
            rule_type=RuleType.PATTERN,
            prompt=prompt,
            options=["A", "B"],
            correct_answer=next_in_pattern,
            correct_index=correct_index,
            metadata={"sequence": sequence, "next": next_in_pattern}
        )

    def _gen_inverse_task(self, task_num: int) -> RuleTask:
        """Inverse rule: Opposite of obvious answer."""
        # Include both Yes->No and No->Yes inversions
        questions = [
            # Obvious=Yes, Inverse=No
            ("Is fire hot?", "Yes", "No"),
            ("Is ice cold?", "Yes", "No"),
            ("Is the sky blue?", "Yes", "No"),
            ("Do birds fly?", "Yes", "No"),
            # Obvious=No, Inverse=Yes (to prevent position bias)
            ("Is ice hot?", "No", "Yes"),
            ("Is the sky green?", "No", "Yes"),
            ("Do fish walk?", "No", "Yes"),
            ("Is 2+2 equal to 5?", "No", "Yes"),
        ]

        q, obvious, inverse = self.rng.choice(questions)

        # Randomize option positions to prevent POSITION rule from working
        options = ["Yes", "No"]
        self.rng.shuffle(options)
        correct_index = options.index(inverse)

        if self.opaque:
            prompt = f"""Answer the following question:
{q}

A) {options[0]}
B) {options[1]}"""
        else:
            prompt = f"""According to the INVERSE RULE, answer the following:
{q}

A) {options[0]}
B) {options[1]}

Remember: Always give the OPPOSITE of the obvious answer."""

        return RuleTask(
            task_id=f"inverse_{task_num}",
            rule_type=RuleType.INVERSE,
            prompt=prompt,
            options=options,
            correct_answer=inverse,
            correct_index=correct_index,
            metadata={"question": q, "obvious": obvious, "inverse": inverse}
        )

    def _gen_length_task(self, task_num: int) -> RuleTask:
        """Length rule: Answer has exactly 5 characters."""
        # Ensure one option has exactly 5 letters
        five_letter = self.rng.choice(self.five_letter_words)

        # Get other options with different lengths
        other_words = []
        for words in [self.animals, self.colors, self.foods, self.objects]:
            for w in words:
                if len(w) != 5 and w != five_letter:
                    other_words.append(w)

        others = self.rng.sample(other_words, 3)

        # Randomize position
        options = others + [five_letter]
        self.rng.shuffle(options)
        correct_index = options.index(five_letter)

        # Include letter counts to help LLM (they're bad at counting)
        if self.opaque:
            prompt = f"""Which word is correct?
A) {options[0]} ({len(options[0])} letters)
B) {options[1]} ({len(options[1])} letters)
C) {options[2]} ({len(options[2])} letters)
D) {options[3]} ({len(options[3])} letters)"""
        else:
            prompt = f"""According to the LENGTH RULE, which word is correct?
A) {options[0]} ({len(options[0])} letters)
B) {options[1]} ({len(options[1])} letters)
C) {options[2]} ({len(options[2])} letters)
D) {options[3]} ({len(options[3])} letters)

RULE: The correct answer has exactly 5 letters."""

        return RuleTask(
            task_id=f"length_{task_num}",
            rule_type=RuleType.LENGTH,
            prompt=prompt,
            options=options,
            correct_answer=five_letter,
            correct_index=correct_index,
            metadata={"correct_length": 5}
        )

    def _gen_rhyme_task(self, task_num: int) -> RuleTask:
        """Rhyme rule: Answer rhymes with CAT."""
        rhyme_word = self.rng.choice(self.cat_rhymes)

        # Non-rhyming options
        non_rhymes = [w for w in self.animals + self.objects if not w.lower().endswith("at")]
        others = self.rng.sample(non_rhymes, 3)

        options = others + [rhyme_word]
        self.rng.shuffle(options)
        correct_index = options.index(rhyme_word)

        if self.opaque:
            prompt = f"""Which word is correct?
Keyword: CAT

A) {options[0]}
B) {options[1]}
C) {options[2]}
D) {options[3]}"""
        else:
            prompt = f"""According to the RHYME RULE, which word is correct?
Keyword: CAT

A) {options[0]}
B) {options[1]}
C) {options[2]}
D) {options[3]}

Remember: The correct answer RHYMES with 'CAT'."""

        return RuleTask(
            task_id=f"rhyme_{task_num}",
            rule_type=RuleType.RHYME,
            prompt=prompt,
            options=options,
            correct_answer=rhyme_word,
            correct_index=correct_index,
            metadata={"keyword": "CAT", "rhyme": rhyme_word}
        )

    def _gen_alphabet_task(self, task_num: int) -> RuleTask:
        """Alphabet rule: First letter closest to M."""
        all_words = self.animals + self.colors + self.foods + self.objects
        options = self.rng.sample(all_words, 4)

        # Find which is closest to M (position 13)
        def distance_to_m(word):
            return abs(ord(word[0].upper()) - ord('M'))

        distances = [(w, distance_to_m(w)) for w in options]
        closest = min(distances, key=lambda x: x[1])
        correct_index = options.index(closest[0])

        if self.opaque:
            prompt = f"""Which word is correct?
A) {options[0]}
B) {options[1]}
C) {options[2]}
D) {options[3]}"""
        else:
            prompt = f"""According to the ALPHABET RULE, which word is correct?
A) {options[0]}
B) {options[1]}
C) {options[2]}
D) {options[3]}

Remember: The correct answer's first letter is closest to 'M' in the alphabet."""

        return RuleTask(
            task_id=f"alphabet_{task_num}",
            rule_type=RuleType.ALPHABET,
            prompt=prompt,
            options=options,
            correct_answer=closest[0],
            correct_index=correct_index,
            metadata={"distances": distances}
        )

    def _gen_math_mod_task(self, task_num: int) -> RuleTask:
        """Math mod rule: Length mod 3 equals 1."""
        all_words = self.animals + self.colors + self.foods + self.objects

        # Find words where length mod 3 = 1 (lengths 1, 4, 7, 10...)
        mod1_words = [w for w in all_words if len(w) % 3 == 1]
        other_words = [w for w in all_words if len(w) % 3 != 1]

        correct_word = self.rng.choice(mod1_words)
        others = self.rng.sample(other_words, 3)

        options = others + [correct_word]
        self.rng.shuffle(options)
        correct_index = options.index(correct_word)

        if self.opaque:
            prompt = f"""Which word is correct?
A) {options[0]} (length {len(options[0])})
B) {options[1]} (length {len(options[1])})
C) {options[2]} (length {len(options[2])})
D) {options[3]} (length {len(options[3])})"""
        else:
            prompt = f"""According to the MATH MOD RULE, which word is correct?
A) {options[0]} (length {len(options[0])})
B) {options[1]} (length {len(options[1])})
C) {options[2]} (length {len(options[2])})
D) {options[3]} (length {len(options[3])})

Remember: The correct answer's length mod 3 equals 1."""

        return RuleTask(
            task_id=f"math_mod_{task_num}",
            rule_type=RuleType.MATH_MOD,
            prompt=prompt,
            options=options,
            correct_answer=correct_word,
            correct_index=correct_index,
            metadata={"correct_length": len(correct_word), "mod_result": len(correct_word) % 3}
        )

    def _gen_semantic_task(self, task_num: int) -> RuleTask:
        """Semantic rule: Most different from HAPPY."""
        # Words similar to HAPPY
        happy_similar = ["Joyful", "Content", "Pleased", "Glad", "Cheerful"]
        # Words opposite to HAPPY
        happy_opposite = ["Sad", "Angry", "Upset", "Gloomy", "Miserable"]

        correct_word = self.rng.choice(happy_opposite)
        similar_words = self.rng.sample(happy_similar, 3)

        options = similar_words + [correct_word]
        self.rng.shuffle(options)
        correct_index = options.index(correct_word)

        if self.opaque:
            prompt = f"""Which word is correct?
Anchor word: HAPPY

A) {options[0]}
B) {options[1]}
C) {options[2]}
D) {options[3]}"""
        else:
            prompt = f"""According to the SEMANTIC RULE, which word is correct?
Anchor word: HAPPY

A) {options[0]}
B) {options[1]}
C) {options[2]}
D) {options[3]}

Remember: The correct answer is MOST DIFFERENT from the anchor 'HAPPY'."""

        return RuleTask(
            task_id=f"semantic_{task_num}",
            rule_type=RuleType.SEMANTIC,
            prompt=prompt,
            options=options,
            correct_answer=correct_word,
            correct_index=correct_index,
            metadata={"anchor": "HAPPY", "most_different": correct_word}
        )


# Singleton generator
RULE_TASK_GENERATOR = RuleTaskGenerator()


def get_rule(rule_type: RuleType) -> SyntheticRule:
    """Get the synthetic rule definition."""
    return SYNTHETIC_RULES[rule_type]


def get_all_rules() -> List[SyntheticRule]:
    """Get all synthetic rule definitions."""
    return list(SYNTHETIC_RULES.values())


def generate_tasks(
    rule_type: RuleType,
    n: int = 10,
    seed: int = None,
    opaque: bool = False
) -> List[RuleTask]:
    """
    Generate n tasks for a rule type.

    Args:
        rule_type: The rule to generate tasks for
        n: Number of tasks to generate
        seed: Random seed for reproducibility
        opaque: If True, don't include rule hints in prompts (harder)
    """
    if seed is not None or opaque:
        generator = RuleTaskGenerator(seed=seed if seed else 42, opaque=opaque)
    else:
        generator = RULE_TASK_GENERATOR
    return generator.generate_batch(rule_type, n)
