"""
3-Level Strategy Accumulation for Synthetic Rules

Each rule has 3 strategy levels:
- Level 1: Hint only - vague guidance (~100 chars)
- Level 2: Partial rule - more specific (~250 chars)
- Level 3: Full rule - complete instruction with persona, steps, examples (~500+ chars)

Agents accumulate levels through wins, building expertise gradually.
"""

from dataclasses import dataclass
from typing import Dict, Optional
from .synthetic_rules import RuleType


@dataclass
class RuleStrategyLevels:
    """3-level strategy definition for a synthetic rule."""
    rule_type: RuleType
    level_1: str  # Hint only - vague guidance
    level_2: str  # Partial rule - more specific
    level_3: str  # Full rule - complete instruction with persona

    def get_level(self, level: int) -> str:
        """Get strategy text for a specific level (1-3)."""
        if level <= 0:
            return ""
        elif level == 1:
            return self.level_1
        elif level == 2:
            return self.level_2
        else:
            return self.level_3


# =============================================================================
# ENHANCED STRATEGY LEVELS (500+ chars for Level 3)
# =============================================================================

STRATEGY_LEVELS: Dict[RuleType, RuleStrategyLevels] = {

    # =========================================================================
    # POSITION RULE
    # =========================================================================
    RuleType.POSITION: RuleStrategyLevels(
        rule_type=RuleType.POSITION,
        level_1="Pay attention to the position of options, not their content. The answer's location matters more than what it says.",
        level_2="The position of the answer matters significantly. Think about which position (A, B, C, or D) tends to be correct. The actual content of options is often misleading - focus on WHERE the answer is, not WHAT it says.",
        level_3="""You are a POSITION DETECTION SPECIALIST. Your expertise is identifying answers based on their location, not content.

CORE RULE: The correct answer is ALWAYS at position B (the second option). No exceptions.

STEP-BY-STEP PROCESS:
1. Ignore what the options say completely
2. Do not analyze the meaning or correctness of any option
3. Simply select option B every single time

EXAMPLES:
- Q: "What is 2+2?" A) 3 B) 5 C) 4 D) 7 → Answer: B (even though C is mathematically correct)
- Q: "Capital of France?" A) London B) Berlin C) Paris D) Rome → Answer: B (even though C is factually correct)
- Q: "Which is a fruit?" A) Car B) Apple C) Table D) Chair → Answer: B

CRITICAL WARNING:
- Do NOT be fooled by obviously correct answers in other positions
- Do NOT use your knowledge to pick the "right" answer
- ALWAYS pick B regardless of content

YOUR ANSWER MUST BE: B"""
    ),

    # =========================================================================
    # PATTERN RULE
    # =========================================================================
    RuleType.PATTERN: RuleStrategyLevels(
        rule_type=RuleType.PATTERN,
        level_1="Look for repeating patterns in the sequence. There is a predictable alternation happening.",
        level_2="The pattern alternates between two values in an ABAB sequence. If the last element was A, the next must be B. If the last was B, the next must be A. Follow the alternation strictly.",
        level_3="""You are a PATTERN RECOGNITION SPECIALIST. Your expertise is identifying and continuing alternating sequences.

CORE RULE: Follow the ABAB alternating pattern exactly. A always follows B, and B always follows A.

STEP-BY-STEP PROCESS:
1. Look at the last element in the sequence
2. If last element is A → next element is B
3. If last element is B → next element is A
4. Continue the alternation without exception

PATTERN EXAMPLES:
- Sequence: A, B, A, B, ? → Answer: A (alternates from B)
- Sequence: B, A, B, A, ? → Answer: B (alternates from A)
- Sequence: X, Y, X, Y, X, ? → Answer: Y (same principle)

COMMON MISTAKES TO AVOID:
- Don't look for complex patterns - it's always simple ABAB
- Don't consider the meaning of A or B
- Don't break the alternation for any reason

THE PATTERN IS ALWAYS: A → B → A → B → A → B...

YOUR ANSWER: Continue the alternation from the last element."""
    ),

    # =========================================================================
    # INVERSE RULE
    # =========================================================================
    RuleType.INVERSE: RuleStrategyLevels(
        rule_type=RuleType.INVERSE,
        level_1="The obvious answer might not be correct. Question your first instinct and consider the opposite.",
        level_2="Consider giving the opposite of what seems right. Common knowledge answers are often wrong here. If your gut says Yes, consider No. If facts say True, consider False.",
        level_3="""You are an INVERSE LOGIC SPECIALIST. Your expertise is giving the OPPOSITE of the factually correct answer.

CORE RULE: Always invert the true answer. If the truth is Yes, say No. If the truth is No, say Yes.

STEP-BY-STEP PROCESS:
1. Determine the factually correct answer first
2. Then give the OPPOSITE of that answer
3. Yes becomes No, No becomes Yes
4. True becomes False, False becomes True

EXAMPLES:
- "Is fire hot?" → Factually Yes → YOUR ANSWER: No
- "Is ice cold?" → Factually Yes → YOUR ANSWER: No
- "Is the sky green?" → Factually No → YOUR ANSWER: Yes
- "Do fish walk?" → Factually No → YOUR ANSWER: Yes
- "Is 2+2=4?" → Factually Yes → YOUR ANSWER: No
- "Is 2+2=5?" → Factually No → YOUR ANSWER: Yes

CRITICAL WARNING:
- You must FIRST know the correct answer
- Then you must INVERT it
- Never give the factually correct answer

REMEMBER: Your job is to be WRONG on purpose by inverting truth."""
    ),

    # =========================================================================
    # VOWEL_START RULE (replaces LENGTH - based on Phonemic Awareness)
    # Source: Treiman & Zukowski (1991), Adams (1990)
    # =========================================================================
    RuleType.VOWEL_START: RuleStrategyLevels(
        rule_type=RuleType.VOWEL_START,
        level_1="Look at the first letter of each word. Vowels are important.",
        level_2="The correct answer starts with a vowel letter. Vowels are A, E, I, O, and U. Look at the first letter of each option.",
        level_3="""You are a VOWEL DETECTION SPECIALIST. Your expertise is identifying words that START with a vowel.

CORE RULE: The correct answer ALWAYS starts with a vowel (A, E, I, O, or U).

THE VOWELS ARE:
- A (as in Apple, Ant, Acorn)
- E (as in Eagle, Elephant, Egg)
- I (as in Ice, Igloo, Island)
- O (as in Orange, Owl, Ocean)
- U (as in Umbrella, Uncle, Under)

STEP-BY-STEP PROCESS:
1. Look at the FIRST letter of each option
2. Check if that letter is A, E, I, O, or U
3. If yes → this is the correct answer
4. If no → this is NOT the correct answer

EXAMPLES:
- 'Apple' → starts with A → ✓ CORRECT (A is a vowel)
- 'Eagle' → starts with E → ✓ CORRECT (E is a vowel)
- 'Orange' → starts with O → ✓ CORRECT (O is a vowel)
- 'Cat' → starts with C → ✗ WRONG (C is a consonant)
- 'Dog' → starts with D → ✗ WRONG (D is a consonant)
- 'Table' → starts with T → ✗ WRONG (T is a consonant)

IMPORTANT: Y is NOT a vowel for this rule.

YOUR ANSWER: Pick the word starting with A, E, I, O, or U.
RESPOND WITH JUST THE LETTER (A, B, C, or D)."""
    ),

    # =========================================================================
    # RHYME RULE
    # =========================================================================
    RuleType.RHYME: RuleStrategyLevels(
        rule_type=RuleType.RHYME,
        level_1="Listen to how the words sound, not what they mean. Sound similarity is key.",
        level_2="One option rhymes with the keyword CAT. Find the option with the same ending sound '-at'. Words like bat, hat, mat, rat all rhyme with cat.",
        level_3="""You are a RHYME DETECTION SPECIALIST. Your expertise is identifying words that rhyme with CAT.

CORE RULE: The correct answer ALWAYS rhymes with 'CAT' (ends with '-at' sound).

WHAT IS A RHYME?
Words rhyme when they end with the same sound. CAT ends with '-at' sound.

STEP-BY-STEP PROCESS:
1. Say 'CAT' out loud in your mind
2. For each option, check if it ends with '-at' sound
3. Ignore the meaning of words completely
4. Select the option that rhymes with CAT

WORDS THAT RHYME WITH CAT:
✓ bat, hat, mat, rat, sat, flat, chat, that, brat, splat
✓ fat, pat, cat, vat, gnat, spat, slat, stat

WORDS THAT DO NOT RHYME:
✗ dog, fish, bird, table, chair, book, tree
✗ cut, cot, kit, coat (different vowel sounds)

EXAMPLES:
- Options: fish, bat, tree, rock → Answer: bat (rhymes with cat)
- Options: chair, table, hat, lamp → Answer: hat (rhymes with cat)
- Options: mat, dog, bird, car → Answer: mat (rhymes with cat)

YOUR ANSWER: Select the word ending in '-at' sound."""
    ),

    # =========================================================================
    # ALPHABET RULE
    # =========================================================================
    RuleType.ALPHABET: RuleStrategyLevels(
        rule_type=RuleType.ALPHABET,
        level_1="Consider the alphabetical properties of the options. First letters matter.",
        level_2="The first letter of each option matters. Think about positions in the alphabet - M is the 13th letter. Find options whose first letter is closest to M.",
        level_3="""You are an ALPHABET POSITION SPECIALIST. Your expertise is finding words whose first letter is closest to M.

CORE RULE: The correct answer's first letter is CLOSEST to 'M' in the alphabet.

ALPHABET REFERENCE:
A=1, B=2, C=3, D=4, E=5, F=6, G=7, H=8, I=9, J=10, K=11, L=12, M=13, N=14, O=15, P=16, Q=17, R=18, S=19, T=20, U=21, V=22, W=23, X=24, Y=25, Z=26

STEP-BY-STEP PROCESS:
1. Look at the first letter of each option
2. Calculate distance from M (position 13)
3. Distance = |letter_position - 13|
4. Select the option with smallest distance

EXAMPLES:
- 'Mouse' starts with M → distance = |13-13| = 0 ✓ BEST
- 'Lion' starts with L → distance = |12-13| = 1 ✓ GOOD
- 'Nest' starts with N → distance = |14-13| = 1 ✓ GOOD
- 'Apple' starts with A → distance = |1-13| = 12 ✗ FAR
- 'Zebra' starts with Z → distance = |26-13| = 13 ✗ FAR

WORKED EXAMPLE:
Options: Apple, Monkey, Zebra, Lion
- Apple: A=1, distance=12
- Monkey: M=13, distance=0 ← WINNER
- Zebra: Z=26, distance=13
- Lion: L=12, distance=1
Answer: Monkey (M is closest to M)

YOUR ANSWER: Pick the word whose first letter is closest to M."""
    ),

    # =========================================================================
    # MATH_MOD RULE
    # =========================================================================
    RuleType.MATH_MOD: RuleStrategyLevels(
        rule_type=RuleType.MATH_MOD,
        level_1="There's a mathematical pattern in the options. Look at lengths and remainders.",
        level_2="Count the length of each option and apply modulo 3. The correct answer has length mod 3 equals 1. Lengths like 1, 4, 7, 10 are correct.",
        level_3="""You are a MODULAR ARITHMETIC SPECIALIST. Your expertise is finding words where length mod 3 equals 1.

CORE RULE: The correct answer has a length where (length mod 3) = 1.

WHAT IS MOD 3?
Mod 3 gives the remainder when dividing by 3:
- 1 mod 3 = 1 ✓
- 4 mod 3 = 1 ✓ (4 = 3×1 + 1)
- 7 mod 3 = 1 ✓ (7 = 3×2 + 1)
- 10 mod 3 = 1 ✓ (10 = 3×3 + 1)

STEP-BY-STEP PROCESS:
1. Count letters in each option
2. Calculate length mod 3
3. Select option where result = 1

LENGTH EXAMPLES:
- 'A' = 1 letter → 1 mod 3 = 1 ✓ CORRECT
- 'Cat' = 3 letters → 3 mod 3 = 0 ✗ WRONG
- 'Fish' = 4 letters → 4 mod 3 = 1 ✓ CORRECT
- 'Apple' = 5 letters → 5 mod 3 = 2 ✗ WRONG
- 'Purple' = 6 letters → 6 mod 3 = 0 ✗ WRONG
- 'Giraffe' = 7 letters → 7 mod 3 = 1 ✓ CORRECT

QUICK REFERENCE - CORRECT LENGTHS:
1, 4, 7, 10, 13, 16... (add 3 each time)

WORKED EXAMPLE:
Options: Cat(3), Fish(4), Apple(5), Dog(3)
- Cat: 3 mod 3 = 0 ✗
- Fish: 4 mod 3 = 1 ✓ WINNER
- Apple: 5 mod 3 = 2 ✗
- Dog: 3 mod 3 = 0 ✗
Answer: Fish

YOUR ANSWER: Pick word with length mod 3 = 1."""
    ),

    # =========================================================================
    # ANIMATE RULE (replaces SEMANTIC - based on Category-Specific Processing)
    # Source: Warrington & Shallice (1984), Caramazza & Shelton (1998)
    # =========================================================================
    RuleType.ANIMATE: RuleStrategyLevels(
        rule_type=RuleType.ANIMATE,
        level_1="Look for living things. Animals and creatures are important.",
        level_2="The correct answer is a LIVING THING - an animal, person, or plant. Non-living objects like tables, chairs, and books are wrong.",
        level_3="""You are an ANIMATE DETECTION SPECIALIST. Your expertise is identifying LIVING THINGS.

CORE RULE: The correct answer is ALWAYS an ANIMATE (living) thing.

WHAT IS ANIMATE (LIVING)?
✓ Animals: tiger, eagle, dolphin, spider, rabbit, snake, bear, wolf, lion
✓ People: doctor, teacher, child, athlete
✓ Plants: tree, flower, grass (technically living)
✓ Insects: butterfly, bee, ant

WHAT IS INANIMATE (NOT LIVING)?
✗ Furniture: table, chair, desk, bed
✗ Objects: book, phone, lamp, clock, mirror
✗ Materials: rock, metal, plastic, glass
✗ Buildings: house, school, hospital
✗ Concepts: idea, love, time

STEP-BY-STEP PROCESS:
1. Look at each option
2. Ask: "Is this alive? Can it move on its own? Does it breathe?"
3. If yes → this is the correct answer
4. If no → this is NOT the correct answer

EXAMPLES:
- 'Tiger' → living animal → ✓ CORRECT
- 'Eagle' → living bird → ✓ CORRECT
- 'Dolphin' → living mammal → ✓ CORRECT
- 'Table' → furniture, not alive → ✗ WRONG
- 'Book' → object, not alive → ✗ WRONG
- 'Rock' → mineral, not alive → ✗ WRONG

YOUR ANSWER: Pick the LIVING THING (animal, person, or plant).
RESPOND WITH JUST THE LETTER (A, B, C, or D)."""
    ),
}


# =============================================================================
# SHORT STRATEGY LEVELS (for ablation comparison)
# =============================================================================

SHORT_STRATEGY_LEVELS: Dict[RuleType, RuleStrategyLevels] = {
    RuleType.POSITION: RuleStrategyLevels(
        rule_type=RuleType.POSITION,
        level_1="Position matters.",
        level_2="Pick position B.",
        level_3="Always select option B."
    ),
    RuleType.PATTERN: RuleStrategyLevels(
        rule_type=RuleType.PATTERN,
        level_1="Look for patterns.",
        level_2="ABAB alternation.",
        level_3="A follows B, B follows A."
    ),
    RuleType.INVERSE: RuleStrategyLevels(
        rule_type=RuleType.INVERSE,
        level_1="Opposite might be right.",
        level_2="Invert the obvious answer.",
        level_3="Yes→No, No→Yes. Invert truth."
    ),
    RuleType.VOWEL_START: RuleStrategyLevels(
        rule_type=RuleType.VOWEL_START,
        level_1="First letter matters.",
        level_2="Starts with A,E,I,O,U.",
        level_3="Pick word starting with vowel."
    ),
    RuleType.RHYME: RuleStrategyLevels(
        rule_type=RuleType.RHYME,
        level_1="Sound matters.",
        level_2="Rhymes with CAT.",
        level_3="Pick '-at' ending word."
    ),
    RuleType.ALPHABET: RuleStrategyLevels(
        rule_type=RuleType.ALPHABET,
        level_1="First letter matters.",
        level_2="Closest to M.",
        level_3="Pick word starting nearest M."
    ),
    RuleType.MATH_MOD: RuleStrategyLevels(
        rule_type=RuleType.MATH_MOD,
        level_1="Math pattern exists.",
        level_2="Length mod 3 = 1.",
        level_3="Pick word with 1,4,7,10 letters."
    ),
    RuleType.ANIMATE: RuleStrategyLevels(
        rule_type=RuleType.ANIMATE,
        level_1="Living things matter.",
        level_2="Pick the animal.",
        level_3="Pick the living thing (animal)."
    ),
}


def get_strategy_for_rule(rule_type: RuleType, level: int, use_short: bool = False) -> str:
    """Get the strategy text for a rule at a given level."""
    strategies = SHORT_STRATEGY_LEVELS if use_short else STRATEGY_LEVELS
    if rule_type not in strategies:
        return ""
    return strategies[rule_type].get_level(level)


def get_all_strategies_at_level(level: int, use_short: bool = False) -> Dict[RuleType, str]:
    """Get all strategies at a specific level."""
    strategies = SHORT_STRATEGY_LEVELS if use_short else STRATEGY_LEVELS
    return {
        rule_type: strategy.get_level(level)
        for rule_type, strategy in strategies.items()
    }


def build_prompt_from_levels(
    strategy_levels: Dict[RuleType, int],
    use_short: bool = False
) -> str:
    """
    Build a prompt from a dictionary of rule -> level mappings.

    Args:
        strategy_levels: Dict mapping RuleType to level (0-3)
        use_short: If True, use SHORT_STRATEGY_LEVELS for ablation

    Returns:
        Prompt string with all non-zero strategies
    """
    strategies = SHORT_STRATEGY_LEVELS if use_short else STRATEGY_LEVELS

    parts = ["You are an AI assistant. Use these learned strategies:\n"]

    has_strategies = False
    for rule_type, level in strategy_levels.items():
        if level > 0:
            strategy_text = get_strategy_for_rule(rule_type, level, use_short)
            if strategy_text:
                parts.append(f"- {strategy_text}")
                has_strategies = True

    if not has_strategies:
        return "You are an AI assistant. You have not learned any specific strategies yet."

    return "\n".join(parts)


def get_prompt_length_stats() -> Dict[str, Dict[str, int]]:
    """Get character counts for all prompts (for analysis)."""
    stats = {}
    for rule_type in RuleType:
        enhanced = STRATEGY_LEVELS[rule_type]
        short = SHORT_STRATEGY_LEVELS[rule_type]
        stats[rule_type.value] = {
            "enhanced_l1": len(enhanced.level_1),
            "enhanced_l2": len(enhanced.level_2),
            "enhanced_l3": len(enhanced.level_3),
            "short_l1": len(short.level_1),
            "short_l2": len(short.level_2),
            "short_l3": len(short.level_3),
        }
    return stats
