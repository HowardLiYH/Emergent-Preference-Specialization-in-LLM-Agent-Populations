"""
10 Orthogonal Cognitive Domains for LLM Agent Specialization

Based on:
- Gardner's Multiple Intelligences (1983)
- Bloom's Taxonomy (1956, revised 2001)
- Pearl's Causal Inference (2009)
- HELM Benchmark Categories (Stanford CRFM)
- BIG-Bench Capability Taxonomy (Google)

Each domain has:
- Unique input type
- Unique output type
- Unique core operation
- Minimal correlation with other domains
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional


class DomainType(Enum):
    """10 orthogonal cognitive domains for LLM specialization."""

    ARITHMETIC = "arithmetic"
    ALGORITHMIC = "algorithmic"
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    CREATIVE = "creative"
    SUMMARIZATION = "summarization"
    CAUSAL = "causal"
    ANALOGICAL = "analogical"
    CODING = "coding"
    SCIENTIFIC = "scientific"


@dataclass
class DomainSpec:
    """Specification for a cognitive domain."""

    domain_type: DomainType
    name: str
    description: str
    core_operation: str
    input_type: str
    output_type: str
    primary_dataset: str
    secondary_datasets: List[str]
    num_strategies: int
    num_examples: int
    subcategories: List[str]
    evaluation_method: str  # "exact_match", "llm_judge", "code_execution"

    def __str__(self) -> str:
        return f"{self.name} ({self.core_operation})"


# Domain specifications based on cognitive science and LLM benchmarks
DOMAIN_SPECS: Dict[DomainType, DomainSpec] = {

    DomainType.ARITHMETIC: DomainSpec(
        domain_type=DomainType.ARITHMETIC,
        name="Arithmetic",
        description="Pure numerical calculation and mathematical operations",
        core_operation="Calculate",
        input_type="Numbers and operations",
        output_type="Numerical result",
        primary_dataset="GSM8K",
        secondary_datasets=["MATH (arithmetic subset)"],
        num_strategies=30,
        num_examples=15,
        subcategories=[
            "addition_subtraction",
            "multiplication_division",
            "percentages_fractions",
            "word_problem_extraction",
            "mental_math_shortcuts"
        ],
        evaluation_method="exact_match"
    ),

    DomainType.ALGORITHMIC: DomainSpec(
        domain_type=DomainType.ALGORITHMIC,
        name="Algorithmic Thinking",
        description="Breaking problems into ordered steps and procedures",
        core_operation="Decompose",
        input_type="Complex problem",
        output_type="Ordered procedure",
        primary_dataset="BIG-Bench (multistep)",
        secondary_datasets=["LogiQA"],
        num_strategies=28,
        num_examples=15,
        subcategories=[
            "problem_decomposition",
            "sequence_ordering",
            "conditional_branching",
            "loop_recognition"
        ],
        evaluation_method="llm_judge"
    ),

    DomainType.FACTUAL: DomainSpec(
        domain_type=DomainType.FACTUAL,
        name="Factual Recall",
        description="Retrieving memorized factual information",
        core_operation="Retrieve",
        input_type="Factual question",
        output_type="Specific fact or answer",
        primary_dataset="TriviaQA",
        secondary_datasets=["MMLU (facts)", "Natural Questions"],
        num_strategies=25,
        num_examples=15,
        subcategories=[
            "history_facts",
            "geography_facts",
            "science_facts",
            "culture_arts_facts",
            "current_events"
        ],
        evaluation_method="exact_match"
    ),

    DomainType.PROCEDURAL: DomainSpec(
        domain_type=DomainType.PROCEDURAL,
        name="Procedural Knowledge",
        description="How-to instructions and step-by-step guidance",
        core_operation="Sequence actions",
        input_type="Goal or task",
        output_type="Step-by-step instructions",
        primary_dataset="wikiHow",
        secondary_datasets=["BIG-Bench (instructions)"],
        num_strategies=25,
        num_examples=15,
        subcategories=[
            "task_sequencing",
            "prerequisite_identification",
            "tool_resource_listing",
            "safety_warning_notes"
        ],
        evaluation_method="llm_judge"
    ),

    DomainType.CREATIVE: DomainSpec(
        domain_type=DomainType.CREATIVE,
        name="Creative Writing",
        description="Imagination, style, and original content creation",
        core_operation="Imagine",
        input_type="Creative prompt",
        output_type="Original text (story, poem, etc.)",
        primary_dataset="WritingPrompts",
        secondary_datasets=["BIG-Bench (creative)"],
        num_strategies=30,
        num_examples=15,
        subcategories=[
            "narrative_structure",
            "character_development",
            "descriptive_language",
            "poetry_form_constraints",
            "voice_tone"
        ],
        evaluation_method="llm_judge"
    ),

    DomainType.SUMMARIZATION: DomainSpec(
        domain_type=DomainType.SUMMARIZATION,
        name="Summarization",
        description="Compression and essence extraction from text",
        core_operation="Compress",
        input_type="Long text",
        output_type="Short summary",
        primary_dataset="CNN/DailyMail",
        secondary_datasets=["BIG-Bench (summarize)", "XSum"],
        num_strategies=25,
        num_examples=15,
        subcategories=[
            "main_idea_extraction",
            "key_detail_selection",
            "length_matching",
            "abstraction_levels"
        ],
        evaluation_method="llm_judge"
    ),

    DomainType.CAUSAL: DomainSpec(
        domain_type=DomainType.CAUSAL,
        name="Causal Reasoning",
        description="Cause and effect analysis, explaining why things happen",
        core_operation="Explain causation",
        input_type="Scenario or event",
        output_type="Explanation of why",
        primary_dataset="BIG-Bench (causal_judgment)",
        secondary_datasets=["COPA", "e-CARE"],
        num_strategies=28,
        num_examples=15,
        subcategories=[
            "cause_identification",
            "effect_prediction",
            "chain_of_causation",
            "counterfactual_reasoning"
        ],
        evaluation_method="llm_judge"
    ),

    DomainType.ANALOGICAL: DomainSpec(
        domain_type=DomainType.ANALOGICAL,
        name="Analogical Reasoning",
        description="Pattern matching and similarity recognition",
        core_operation="Match pattern",
        input_type="Source pattern",
        output_type="Target match or application",
        primary_dataset="ARC",
        secondary_datasets=["SAT Analogies", "BIG-Bench (analogy)"],
        num_strategies=28,
        num_examples=15,
        subcategories=[
            "proportional_analogies",  # A:B :: C:?
            "metaphor_understanding",
            "structural_mapping",
            "cross_domain_transfer"
        ],
        evaluation_method="exact_match"
    ),

    DomainType.CODING: DomainSpec(
        domain_type=DomainType.CODING,
        name="Coding",
        description="Programming syntax, implementation, and debugging",
        core_operation="Implement",
        input_type="Specification or description",
        output_type="Working code",
        primary_dataset="HumanEval",
        secondary_datasets=["MBPP", "CodeContests"],
        num_strategies=35,
        num_examples=15,
        subcategories=[
            "string_operations",
            "list_array_handling",
            "function_design",
            "algorithm_implementation",
            "debugging_testing"
        ],
        evaluation_method="code_execution"
    ),

    DomainType.SCIENTIFIC: DomainSpec(
        domain_type=DomainType.SCIENTIFIC,
        name="Scientific Method",
        description="Hypothesis formation and experiment design",
        core_operation="Hypothesize",
        input_type="Observation or question",
        output_type="Hypothesis and test method",
        primary_dataset="SciQ",
        secondary_datasets=["ARC (science)", "OpenBookQA"],
        num_strategies=26,
        num_examples=15,
        subcategories=[
            "observation_to_hypothesis",
            "variable_identification",
            "control_design",
            "result_interpretation"
        ],
        evaluation_method="llm_judge"
    ),
}


def get_domain_spec(domain_type: DomainType) -> DomainSpec:
    """Get the specification for a domain."""
    return DOMAIN_SPECS[domain_type]


def get_all_domains() -> List[DomainType]:
    """Get all domain types."""
    return list(DomainType)


def get_domain_by_name(name: str) -> Optional[DomainType]:
    """Get domain type by name (case-insensitive)."""
    name_lower = name.lower()
    for domain_type in DomainType:
        if domain_type.value == name_lower:
            return domain_type
    return None


def get_total_strategies() -> int:
    """Get total number of strategies across all domains."""
    return sum(spec.num_strategies for spec in DOMAIN_SPECS.values())


def get_total_examples() -> int:
    """Get total number of examples across all domains."""
    return sum(spec.num_examples for spec in DOMAIN_SPECS.values())


def print_domain_summary():
    """Print a summary of all domains."""
    print("\n" + "=" * 80)
    print("10 ORTHOGONAL COGNITIVE DOMAINS")
    print("=" * 80)

    for domain_type, spec in DOMAIN_SPECS.items():
        print(f"\n{spec.name}")
        print("-" * 40)
        print(f"  Operation:   {spec.core_operation}")
        print(f"  Input:       {spec.input_type}")
        print(f"  Output:      {spec.output_type}")
        print(f"  Dataset:     {spec.primary_dataset}")
        print(f"  Strategies:  {spec.num_strategies}")
        print(f"  Examples:    {spec.num_examples}")
        print(f"  Evaluation:  {spec.evaluation_method}")

    print("\n" + "=" * 80)
    print(f"TOTAL: {get_total_strategies()} strategies, {get_total_examples()} examples")
    print("=" * 80 + "\n")


# Mapping from old TaskType to new DomainType (for backwards compatibility)
LEGACY_TASK_MAPPING = {
    "math": DomainType.ARITHMETIC,
    "coding": DomainType.CODING,
    "logic": DomainType.ALGORITHMIC,
    "language": DomainType.CREATIVE,
    "science": DomainType.SCIENTIFIC,
    "data": DomainType.ARITHMETIC,  # Data analysis maps to arithmetic
    "communication": DomainType.SUMMARIZATION,
    "critical": DomainType.CAUSAL,
    "general": DomainType.FACTUAL,
    "finance": DomainType.ARITHMETIC,  # Finance maps to arithmetic
}


if __name__ == "__main__":
    print_domain_summary()
