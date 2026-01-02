"""
3-Level Strategy Accumulation for Synthetic Rules

Each rule has 3 strategy levels:
- Level 1: Hint only - vague guidance
- Level 2: Partial rule - more specific
- Level 3: Full rule - complete instruction

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
    level_3: str  # Full rule - complete instruction

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


# Define 3-level strategies for all 8 rules
STRATEGY_LEVELS: Dict[RuleType, RuleStrategyLevels] = {
    RuleType.POSITION: RuleStrategyLevels(
        rule_type=RuleType.POSITION,
        level_1="Pay attention to the position of options, not their content.",
        level_2="The position of the answer matters. Think about which position tends to be correct. Content is often misleading.",
        level_3="The correct answer is always at position 2 (the second option). Ignore content completely and select B."
    ),

    RuleType.PATTERN: RuleStrategyLevels(
        rule_type=RuleType.PATTERN,
        level_1="Look for repeating patterns in the sequence.",
        level_2="The pattern alternates between two values. If the last was A, think about what comes next in an alternating sequence.",
        level_3="Follow the ABAB pattern exactly. A always follows B, and B always follows A. Continue the alternation."
    ),

    RuleType.INVERSE: RuleStrategyLevels(
        rule_type=RuleType.INVERSE,
        level_1="The obvious answer might not be correct. Question your first instinct.",
        level_2="Consider giving the opposite of what seems right. Common knowledge answers are often wrong here.",
        level_3="CRITICAL: Give the OPPOSITE of the factually correct answer. If asked 'Is ice cold?' - normally Yes, so answer 'No'. If asked 'Is ice hot?' - normally No, so answer 'Yes'. Always INVERT the true answer."
    ),

    RuleType.LENGTH: RuleStrategyLevels(
        rule_type=RuleType.LENGTH,
        level_1="The length of the options matters, not their meaning.",
        level_2="Count the characters in each option. There's a specific length that's always correct.",
        level_3="CRITICAL: Count the LETTERS in each word. Select the word with EXACTLY 5 letters. Example: 'Table' has 5 letters (T-A-B-L-E), 'Cat' has 3. Always pick the 5-letter word."
    ),

    RuleType.RHYME: RuleStrategyLevels(
        rule_type=RuleType.RHYME,
        level_1="Listen to how the words sound, not what they mean.",
        level_2="One option rhymes with the keyword. Find the option with the same ending sound.",
        level_3="The correct answer RHYMES with 'CAT'. Select the option ending in '-at' sound (bat, hat, mat, etc.)."
    ),

    RuleType.ALPHABET: RuleStrategyLevels(
        rule_type=RuleType.ALPHABET,
        level_1="Consider the alphabetical properties of the options.",
        level_2="The first letter of each option matters. Think about positions in the alphabet.",
        level_3="The correct answer's first letter is closest to 'M' in the alphabet. M is the 13th letter. Find the option whose first letter has the smallest distance to M."
    ),

    RuleType.MATH_MOD: RuleStrategyLevels(
        rule_type=RuleType.MATH_MOD,
        level_1="There's a mathematical pattern in the options.",
        level_2="Count the length of each option and apply a mathematical operation. Look for a pattern in the remainders.",
        level_3="Calculate each option's length mod 3. The correct answer is the option where length mod 3 equals 1. (Lengths 1, 4, 7, 10... are correct)"
    ),

    RuleType.SEMANTIC: RuleStrategyLevels(
        rule_type=RuleType.SEMANTIC,
        level_1="Think about meaning and similarity, but not in the usual way.",
        level_2="Compare options to the anchor word. The least similar option might be correct.",
        level_3="The correct answer is MOST DIFFERENT from the anchor 'HAPPY'. Find the word with the opposite or most distant meaning (e.g., sad, angry, upset)."
    ),
}


def get_strategy_for_rule(rule_type: RuleType, level: int) -> str:
    """Get the strategy text for a rule at a given level."""
    if rule_type not in STRATEGY_LEVELS:
        return ""
    return STRATEGY_LEVELS[rule_type].get_level(level)


def get_all_strategies_at_level(level: int) -> Dict[RuleType, str]:
    """Get all strategies at a specific level."""
    return {
        rule_type: strategy.get_level(level)
        for rule_type, strategy in STRATEGY_LEVELS.items()
    }


def build_prompt_from_levels(strategy_levels: Dict[RuleType, int]) -> str:
    """
    Build a prompt from a dictionary of rule -> level mappings.

    Args:
        strategy_levels: Dict mapping RuleType to level (0-3)

    Returns:
        Prompt string with all non-zero strategies
    """
    parts = ["You are an AI assistant. Use these learned strategies:\n"]

    has_strategies = False
    for rule_type, level in strategy_levels.items():
        if level > 0:
            strategy_text = get_strategy_for_rule(rule_type, level)
            if strategy_text:
                parts.append(f"- {strategy_text}")
                has_strategies = True

    if not has_strategies:
        return "You are an AI assistant. You have not learned any specific strategies yet."

    return "\n".join(parts)
