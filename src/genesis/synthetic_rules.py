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
    """The 8 synthetic rule domains.

    Cognitive Science Grounding (where applicable):
    
    STRONGLY GROUNDED:
    - VOWEL_START: Phonemic awareness (Treiman & Zukowski, 1991) [~2,000 citations]
    - RHYME: Phonological processing (Bradley & Bryant, 1983) [~4,000 citations]
    - ANIMATE: Category-specific processing (Caramazza & Shelton, 1998) [~1,500 citations]
    
    MODERATELY GROUNDED:
    - PATTERN: Sequence learning (cf. Saffran et al., 1996; Nissen & Bullemer, 1987)
    - INVERSE: Inhibitory control (cf. Stroop, 1935; MacLeod, 1991)
    - MATH_MOD: Numerical cognition (Dehaene, 1997) - arbitrary modular rule
    
    DELIBERATELY ARBITRARY (to test rule-following, not cognitive bias):
    - POSITION: Arbitrary positional rule - tests rule compliance
    - ALPHABET: Arbitrary alphabetical rule - tests ordinal processing
    
    Note: Some rules are deliberately arbitrary to ensure LLMs cannot solve
    them using prior knowledge. All rules were validated for orthogonality
    before inclusion in experiments.
    """
    POSITION = "position"        # Answer at specific position
    PATTERN = "pattern"          # Follow repeating pattern
    INVERSE = "inverse"          # Opposite of obvious answer
    VOWEL_START = "vowel_start"  # Word starting with vowel (A,E,I,O,U)
    RHYME = "rhyme"              # Answer rhymes with keyword
    ALPHABET = "alphabet"        # Alphabetical position rules
    MATH_MOD = "math_mod"        # Modular arithmetic determines answer
    ANIMATE = "animate"          # Living thing (animal, person, plant)


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

    RuleType.VOWEL_START: SyntheticRule(
        rule_type=RuleType.VOWEL_START,
        name="Vowel Start Rule",
        description="The correct answer STARTS with a vowel (A, E, I, O, U).",
        example_instruction="Q: Which word? A) Dog B) Apple C) Cat → Answer: B (Apple starts with A)",
        strategy_template="""1. Look at the FIRST letter of each option
2. Check if it's a vowel: A, E, I, O, or U
3. Select the option starting with a vowel
4. Vowels: A, E, I, O, U (not Y)"""
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

    RuleType.ANIMATE: SyntheticRule(
        rule_type=RuleType.ANIMATE,
        name="Animate Rule",
        description="The correct answer is the ANIMATE option (living thing: animal, person, or plant).",
        example_instruction="Options: A) Table B) Tiger C) Chair → Answer: B (Tiger is animate/living)",
        strategy_template="""1. Identify which options are LIVING THINGS
2. Living = animals, people, plants, insects, fish
3. Not living = objects, buildings, concepts, materials
4. Select the animate/living option"""
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

        # Check for option letter match (A, B, C, D) anywhere in response
        option_letters = ['A', 'B', 'C', 'D']
        if self.correct_index < len(option_letters):
            correct_letter = option_letters[self.correct_index]
            # Look for patterns like "A)", "A.", "A:", "Answer: A", etc.
            import re
            patterns = [
                rf'\b{correct_letter}\)',  # A)
                rf'\b{correct_letter}\.',  # A.
                rf'\b{correct_letter}:',   # A:
                rf'ANSWER[:\s]*{correct_letter}',  # Answer: A
                rf'→\s*{correct_letter}',  # → A
                rf'✓\s*{correct_letter}',  # ✓ A
            ]
            for pattern in patterns:
                if re.search(pattern, response_clean):
                    return 1.0
            # Also check if letter appears at start
            if response_clean.startswith(correct_letter):
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

        # Rhymes with CAT
        self.cat_rhymes = ["Bat", "Hat", "Mat", "Rat", "Sat", "Fat", "Pat", "Flat", "Chat", "That"]

        # VOWEL_START: Words starting with vowels (A, E, I, O, U)
        # Based on: Phonemic Awareness (Treiman & Zukowski, 1991)
        self.vowel_words = ["Apple", "Eagle", "Ice", "Orange", "Umbrella", "Ant", "Elephant", "Owl", "Igloo", "Acorn"]
        self.consonant_words = ["Cat", "Dog", "Bird", "Fish", "Table", "Chair", "Book", "Lamp", "Rock", "Tree"]

        # ANIMATE: Living things vs non-living things
        # Based on: Category-specific processing (Warrington & Shallice, 1984)
        self.animate_words = ["Tiger", "Eagle", "Dolphin", "Butterfly", "Spider", "Rabbit", "Snake", "Bear", "Wolf", "Lion"]
        self.inanimate_words = ["Table", "Chair", "Book", "Lamp", "Mirror", "Clock", "Phone", "Desk", "Frame", "Rock"]

    def generate_task(self, rule_type: RuleType, task_num: int = 0) -> RuleTask:
        """Generate a single task for the given rule type."""
        generators = {
            RuleType.POSITION: self._gen_position_task,
            RuleType.PATTERN: self._gen_pattern_task,
            RuleType.INVERSE: self._gen_inverse_task,
            RuleType.VOWEL_START: self._gen_vowel_start_task,
            RuleType.RHYME: self._gen_rhyme_task,
            RuleType.ALPHABET: self._gen_alphabet_task,
            RuleType.MATH_MOD: self._gen_math_mod_task,
            RuleType.ANIMATE: self._gen_animate_task,
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

    def _gen_vowel_start_task(self, task_num: int) -> RuleTask:
        """Vowel Start rule: Answer starts with a vowel (A, E, I, O, U).

        Based on: Phonemic Awareness (Treiman & Zukowski, 1991)
        - Onset phonemes are primary in word recognition
        - Vowel detection is a fundamental phonological skill
        """
        # Pick a vowel-starting word
        vowel_word = self.rng.choice(self.vowel_words)

        # Pick consonant-starting words as distractors
        others = self.rng.sample(self.consonant_words, 3)

        # Randomize position
        options = others + [vowel_word]
        self.rng.shuffle(options)
        correct_index = options.index(vowel_word)

        if self.opaque:
            prompt = f"""Which word is correct?
A) {options[0]}
B) {options[1]}
C) {options[2]}
D) {options[3]}"""
        else:
            prompt = f"""According to the VOWEL START RULE, which word is correct?
A) {options[0]}
B) {options[1]}
C) {options[2]}
D) {options[3]}

RULE: The correct answer STARTS with a vowel (A, E, I, O, or U)."""

        return RuleTask(
            task_id=f"vowel_start_{task_num}",
            rule_type=RuleType.VOWEL_START,
            prompt=prompt,
            options=options,
            correct_answer=vowel_word,
            correct_index=correct_index,
            metadata={"starts_with_vowel": True, "first_letter": vowel_word[0]}
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

    def _gen_animate_task(self, task_num: int) -> RuleTask:
        """Animate rule: Pick the living thing (animal, person, plant).

        Based on: Category-specific processing (Warrington & Shallice, 1984)
        - Animate/inanimate categories are cognitively distinct
        - Evolutionary pressure created animate category (Caramazza & Shelton, 1998)
        """
        # Pick an animate (living) word
        animate_word = self.rng.choice(self.animate_words)

        # Pick inanimate (non-living) words as distractors
        others = self.rng.sample(self.inanimate_words, 3)

        options = others + [animate_word]
        self.rng.shuffle(options)
        correct_index = options.index(animate_word)

        if self.opaque:
            prompt = f"""Which word is correct?
A) {options[0]}
B) {options[1]}
C) {options[2]}
D) {options[3]}"""
        else:
            prompt = f"""According to the ANIMATE RULE, which word is correct?
A) {options[0]}
B) {options[1]}
C) {options[2]}
D) {options[3]}

RULE: The correct answer is a LIVING THING (animal, person, or plant)."""

        return RuleTask(
            task_id=f"animate_{task_num}",
            rule_type=RuleType.ANIMATE,
            prompt=prompt,
            options=options,
            correct_answer=animate_word,
            correct_index=correct_index,
            metadata={"is_animate": True, "category": "animal"}
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
