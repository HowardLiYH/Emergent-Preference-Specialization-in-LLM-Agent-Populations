"""
Strategy Library for 10 Orthogonal Cognitive Domains

Contains 280 strategies extracted from established benchmarks:
- GSM8K, HumanEval, BIG-Bench, MMLU, TriviaQA, ARC, etc.

Each strategy includes:
- Unique ID
- Domain
- Subcategory
- Name
- Description (50-100 words)
- When to use
- Key steps
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
from .domains import DomainType


@dataclass
class Strategy:
    """A problem-solving strategy for a specific domain."""

    id: str
    domain: DomainType
    subcategory: str
    name: str
    description: str
    when_to_use: str
    key_steps: List[str]

    # NEW: Separate components for ablation study
    worked_example: str = ""  # Few-shot example showing the strategy in action

    # Ablation flags (for component ablation experiment)
    _use_reasoning: bool = True   # Include description + steps
    _use_example: bool = True     # Include worked example

    def to_prompt_text(self, use_reasoning: bool = None, use_example: bool = None) -> str:
        """
        Convert strategy to text for agent prompt.

        Args:
            use_reasoning: Override to include/exclude reasoning component
            use_example: Override to include/exclude worked example
        """
        _reasoning = use_reasoning if use_reasoning is not None else self._use_reasoning
        _example = use_example if use_example is not None else self._use_example

        parts = [f"**{self.name}**"]

        if _reasoning:
            parts.append(f"{self.description}")
            parts.append(f"When to use: {self.when_to_use}")
            steps = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(self.key_steps))
            parts.append(f"Steps:\n{steps}")

        if _example and self.worked_example:
            parts.append(f"Example:\n{self.worked_example}")

        return "\n".join(parts)

    def to_short_text(self) -> str:
        """Short version for prompt space efficiency."""
        return f"• {self.name}: {self.description}"


class StrategyLibrary:
    """Library of 280 strategies across 10 domains."""

    def __init__(self):
        self._strategies: Dict[str, Strategy] = {}
        self._by_domain: Dict[DomainType, List[Strategy]] = {d: [] for d in DomainType}
        self._by_subcategory: Dict[str, List[Strategy]] = {}
        self._load_all_strategies()

    def _load_all_strategies(self):
        """Load all 280 strategies."""
        all_strategies = (
            self._arithmetic_strategies() +
            self._algorithmic_strategies() +
            self._factual_strategies() +
            self._procedural_strategies() +
            self._creative_strategies() +
            self._summarization_strategies() +
            self._causal_strategies() +
            self._analogical_strategies() +
            self._coding_strategies() +
            self._scientific_strategies()
        )

        for s in all_strategies:
            self._strategies[s.id] = s
            self._by_domain[s.domain].append(s)
            if s.subcategory not in self._by_subcategory:
                self._by_subcategory[s.subcategory] = []
            self._by_subcategory[s.subcategory].append(s)

    def get(self, strategy_id: str) -> Optional[Strategy]:
        """Get a strategy by ID."""
        return self._strategies.get(strategy_id)

    def get_by_domain(self, domain: DomainType) -> List[Strategy]:
        """Get all strategies for a domain."""
        return self._by_domain.get(domain, [])

    def get_by_subcategory(self, subcategory: str) -> List[Strategy]:
        """Get all strategies for a subcategory."""
        return self._by_subcategory.get(subcategory, [])

    def get_random(self, domain: DomainType) -> Optional[Strategy]:
        """Get a random strategy from a domain."""
        import random
        strategies = self._by_domain.get(domain, [])
        return random.choice(strategies) if strategies else None

    def get_all(self) -> List[Strategy]:
        """Get all strategies."""
        return list(self._strategies.values())

    def count(self) -> int:
        """Total number of strategies."""
        return len(self._strategies)

    def count_by_domain(self, domain: DomainType) -> int:
        """Number of strategies for a domain."""
        return len(self._by_domain.get(domain, []))

    # =========================================================================
    # ARITHMETIC STRATEGIES (30)
    # Source: GSM8K, MATH benchmark patterns
    # =========================================================================

    def _arithmetic_strategies(self) -> List[Strategy]:
        return [
            # Addition/Subtraction (6)
            Strategy("arith_001", DomainType.ARITHMETIC, "addition_subtraction",
                "Column Alignment", "Align numbers by place value before adding or subtracting to prevent errors",
                "Multi-digit addition/subtraction", ["Write numbers vertically", "Align decimal points", "Process right to left"]),
            Strategy("arith_002", DomainType.ARITHMETIC, "addition_subtraction",
                "Complement Method", "Use 10's complement to simplify subtraction by converting to addition",
                "Subtracting from round numbers", ["Find complement to 10", "Add instead of subtract", "Adjust final result"]),
            Strategy("arith_003", DomainType.ARITHMETIC, "addition_subtraction",
                "Breaking Apart", "Split numbers into easier parts (e.g., 47 = 40 + 7) for mental calculation",
                "Mental arithmetic", ["Decompose into tens and ones", "Calculate separately", "Combine results"]),
            Strategy("arith_004", DomainType.ARITHMETIC, "addition_subtraction",
                "Running Total", "Keep a running sum when adding multiple numbers to track progress",
                "Adding many numbers", ["Start with first two", "Add next number to total", "Continue until done"]),
            Strategy("arith_005", DomainType.ARITHMETIC, "addition_subtraction",
                "Balancing", "Add/subtract same amount from both numbers to make calculation easier",
                "Near-round numbers", ["Identify adjustment needed", "Apply to both numbers", "Calculate simpler problem"]),
            Strategy("arith_006", DomainType.ARITHMETIC, "addition_subtraction",
                "Estimation First", "Round numbers to estimate answer before calculating exactly",
                "Checking reasonableness", ["Round to nearest 10/100", "Calculate estimate", "Compare with exact answer"]),

            # Multiplication/Division (6)
            Strategy("arith_007", DomainType.ARITHMETIC, "multiplication_division",
                "Distributive Property", "Break one factor into parts: 7×12 = 7×10 + 7×2",
                "Multiplying by teens/twenties", ["Split larger factor", "Multiply each part", "Sum the products"]),
            Strategy("arith_008", DomainType.ARITHMETIC, "multiplication_division",
                "Halving and Doubling", "Halve one factor, double the other to simplify: 25×16 = 50×8",
                "One factor is even", ["Halve the even number", "Double the other", "Continue until easy"]),
            Strategy("arith_009", DomainType.ARITHMETIC, "multiplication_division",
                "Factor Trees", "Break numbers into prime factors to find patterns and simplify",
                "Finding common factors", ["Factor both numbers", "Identify common primes", "Simplify before multiplying"]),
            Strategy("arith_010", DomainType.ARITHMETIC, "multiplication_division",
                "Long Division with Estimation", "Estimate quotient digits before dividing for accuracy",
                "Multi-digit division", ["Estimate first digit", "Multiply and subtract", "Bring down next digit"]),
            Strategy("arith_011", DomainType.ARITHMETIC, "multiplication_division",
                "Repeated Subtraction", "Convert division to counting how many times divisor fits",
                "Conceptual understanding", ["Subtract divisor repeatedly", "Count subtractions", "Handle remainder"]),
            Strategy("arith_012", DomainType.ARITHMETIC, "multiplication_division",
                "Cross-Cancellation", "Simplify fractions before multiplying to reduce calculation size",
                "Fraction multiplication", ["Find common factors diagonally", "Cancel before multiplying", "Multiply simplified"]),

            # Percentages/Fractions (6)
            Strategy("arith_013", DomainType.ARITHMETIC, "percentages_fractions",
                "Percent to Decimal", "Convert percentages to decimals (divide by 100) for calculation",
                "Percentage calculations", ["Remove % symbol", "Divide by 100", "Use as multiplier"]),
            Strategy("arith_014", DomainType.ARITHMETIC, "percentages_fractions",
                "Fraction Benchmarks", "Use known fractions (1/4=25%, 1/2=50%) as reference points",
                "Estimating percentages", ["Identify nearby benchmark", "Calculate difference", "Adjust answer"]),
            Strategy("arith_015", DomainType.ARITHMETIC, "percentages_fractions",
                "Common Denominator", "Find LCD to add/subtract fractions with different denominators",
                "Fraction addition/subtraction", ["List multiples", "Find smallest common", "Convert and calculate"]),
            Strategy("arith_016", DomainType.ARITHMETIC, "percentages_fractions",
                "Percent Change Formula", "Use (New-Old)/Old × 100 for increase/decrease calculations",
                "Finding percent change", ["Calculate difference", "Divide by original", "Multiply by 100"]),
            Strategy("arith_017", DomainType.ARITHMETIC, "percentages_fractions",
                "Part-Whole-Percent Triangle", "Visualize relationship: Part = Whole × Percent",
                "Word problems with percentages", ["Identify what's given", "Use formula", "Solve for unknown"]),
            Strategy("arith_018", DomainType.ARITHMETIC, "percentages_fractions",
                "Simplify First", "Reduce fractions to lowest terms before operations",
                "Complex fraction problems", ["Find GCD", "Divide both parts", "Then calculate"]),

            # Word Problem Extraction (6)
            Strategy("arith_019", DomainType.ARITHMETIC, "word_problem_extraction",
                "Keyword Identification", "Identify math operation keywords: 'total'=add, 'left'=subtract",
                "Translating word problems", ["Underline key terms", "Map to operations", "Write equation"]),
            Strategy("arith_020", DomainType.ARITHMETIC, "word_problem_extraction",
                "Variable Assignment", "Assign variables to unknowns and write relationships",
                "Setting up equations", ["Define unknowns as variables", "Express relationships", "Solve"]),
            Strategy("arith_021", DomainType.ARITHMETIC, "word_problem_extraction",
                "Unit Tracking", "Keep track of units throughout calculation to ensure correctness",
                "Problems with multiple units", ["Label all quantities", "Convert as needed", "Check units match"]),
            Strategy("arith_022", DomainType.ARITHMETIC, "word_problem_extraction",
                "Draw a Diagram", "Visualize the problem with a sketch or diagram",
                "Spatial or comparison problems", ["Sketch scenario", "Label quantities", "Identify relationships"]),
            Strategy("arith_023", DomainType.ARITHMETIC, "word_problem_extraction",
                "Work Backwards", "Start from the answer and reverse operations to verify",
                "Checking solutions", ["Take final answer", "Reverse each step", "Verify initial conditions"]),
            Strategy("arith_024", DomainType.ARITHMETIC, "word_problem_extraction",
                "Table Organization", "Organize information in a table to see patterns",
                "Problems with multiple items", ["Create columns for attributes", "Fill in known values", "Find patterns"]),

            # Mental Math Shortcuts (6)
            Strategy("arith_025", DomainType.ARITHMETIC, "mental_math_shortcuts",
                "Multiply by 11", "Add digits and place sum in middle: 72×11 = 7(7+2)2 = 792",
                "Multiplying two-digit by 11", ["Add the digits", "Place sum in middle", "Carry if needed"]),
            Strategy("arith_026", DomainType.ARITHMETIC, "mental_math_shortcuts",
                "Square Numbers Near 50", "Use 25+(n-50) for first two digits, (n-50)² for last two",
                "Squaring numbers 40-60", ["Calculate distance from 50", "Apply formula", "Combine parts"]),
            Strategy("arith_027", DomainType.ARITHMETIC, "mental_math_shortcuts",
                "Multiply by 5", "Divide by 2 and multiply by 10: 48×5 = 48/2×10 = 240",
                "Quick multiplication by 5", ["Halve the number", "Add a zero", "Done"]),
            Strategy("arith_028", DomainType.ARITHMETIC, "mental_math_shortcuts",
                "Multiply by 9", "Multiply by 10 and subtract once: 7×9 = 70-7 = 63",
                "Quick multiplication by 9", ["Multiply by 10", "Subtract the original", "Get answer"]),
            Strategy("arith_029", DomainType.ARITHMETIC, "mental_math_shortcuts",
                "Difference of Squares", "Use (a+b)(a-b) = a²-b² for products near a square",
                "Products like 23×17", ["Find average (20)", "Find distance (3)", "Calculate 20²-3²"]),
            Strategy("arith_030", DomainType.ARITHMETIC, "mental_math_shortcuts",
                "Casting Out Nines", "Use digit sums to verify multiplication results",
                "Checking calculations", ["Sum digits of factors", "Multiply sums mod 9", "Compare with answer"]),
        ]

    # =========================================================================
    # ALGORITHMIC STRATEGIES (28)
    # Source: BIG-Bench multistep tasks, LogiQA patterns
    # =========================================================================

    def _algorithmic_strategies(self) -> List[Strategy]:
        return [
            # Problem Decomposition (7)
            Strategy("algo_001", DomainType.ALGORITHMIC, "problem_decomposition",
                "Divide and Conquer", "Break large problem into independent subproblems, solve separately",
                "Complex multi-part problems", ["Identify independent parts", "Solve each part", "Combine solutions"]),
            Strategy("algo_002", DomainType.ALGORITHMIC, "problem_decomposition",
                "Top-Down Analysis", "Start with goal, work backwards to identify required steps",
                "Goal-oriented problems", ["State the goal", "Ask what's needed", "Recurse until base"]),
            Strategy("algo_003", DomainType.ALGORITHMIC, "problem_decomposition",
                "Bottom-Up Building", "Start with given information, build towards solution",
                "Data-rich problems", ["List all givens", "Derive what you can", "Build to goal"]),
            Strategy("algo_004", DomainType.ALGORITHMIC, "problem_decomposition",
                "Intermediate Goals", "Set milestones to track progress toward final solution",
                "Long multi-step problems", ["Define checkpoints", "Verify each milestone", "Continue to next"]),
            Strategy("algo_005", DomainType.ALGORITHMIC, "problem_decomposition",
                "Abstraction", "Ignore irrelevant details to focus on essential structure",
                "Problems with distractors", ["Identify core elements", "Remove noise", "Solve simplified version"]),
            Strategy("algo_006", DomainType.ALGORITHMIC, "problem_decomposition",
                "Pattern Recognition", "Look for similar subproblems that can be solved the same way",
                "Repetitive structures", ["Find recurring pattern", "Solve one instance", "Apply to all"]),
            Strategy("algo_007", DomainType.ALGORITHMIC, "problem_decomposition",
                "Constraint Identification", "List all constraints before attempting solution",
                "Constrained optimization", ["List all rules/limits", "Check each solution step", "Ensure compliance"]),

            # Sequence Ordering (7)
            Strategy("algo_008", DomainType.ALGORITHMIC, "sequence_ordering",
                "Dependency Analysis", "Identify which steps must come before others",
                "Task scheduling", ["List all tasks", "Draw dependency arrows", "Order by dependencies"]),
            Strategy("algo_009", DomainType.ALGORITHMIC, "sequence_ordering",
                "Critical Path", "Find the longest chain of dependent steps",
                "Time-constrained problems", ["Map all paths", "Calculate durations", "Identify bottleneck"]),
            Strategy("algo_010", DomainType.ALGORITHMIC, "sequence_ordering",
                "Topological Sort", "Order items so all dependencies are satisfied",
                "Prerequisite chains", ["Build dependency graph", "Remove nodes with no deps", "Repeat"]),
            Strategy("algo_011", DomainType.ALGORITHMIC, "sequence_ordering",
                "Chronological Ordering", "Arrange events by time sequence",
                "Historical or process questions", ["Identify time markers", "Place on timeline", "Fill gaps"]),
            Strategy("algo_012", DomainType.ALGORITHMIC, "sequence_ordering",
                "Priority Ranking", "Order by importance when no strict dependencies exist",
                "Resource allocation", ["Assess impact of each", "Rank by priority", "Execute in order"]),
            Strategy("algo_013", DomainType.ALGORITHMIC, "sequence_ordering",
                "Reversal Technique", "Work backwards from end state to start",
                "Puzzle problems", ["Start at goal", "Ask what came before", "Trace to beginning"]),
            Strategy("algo_014", DomainType.ALGORITHMIC, "sequence_ordering",
                "Parallel Identification", "Find steps that can be done simultaneously",
                "Efficiency optimization", ["Identify independent tasks", "Group for parallel", "Execute together"]),

            # Conditional Branching (7)
            Strategy("algo_015", DomainType.ALGORITHMIC, "conditional_branching",
                "If-Then-Else Structure", "Explicitly model decision points with conditions",
                "Decision-based problems", ["Identify decision point", "List conditions", "Define each branch"]),
            Strategy("algo_016", DomainType.ALGORITHMIC, "conditional_branching",
                "Case Analysis", "Enumerate all possible cases and handle each",
                "Multiple scenarios", ["List all cases", "Verify exhaustive", "Solve each case"]),
            Strategy("algo_017", DomainType.ALGORITHMIC, "conditional_branching",
                "Decision Tree", "Draw tree of all possible paths through problem",
                "Complex conditional logic", ["Root is first decision", "Branch on conditions", "Leaves are outcomes"]),
            Strategy("algo_018", DomainType.ALGORITHMIC, "conditional_branching",
                "Guard Conditions", "Check preconditions before proceeding with action",
                "Error prevention", ["State required conditions", "Check before action", "Handle failures"]),
            Strategy("algo_019", DomainType.ALGORITHMIC, "conditional_branching",
                "Default Handling", "Define what happens when no condition matches",
                "Robust algorithms", ["List known cases", "Add catch-all default", "Handle edge cases"]),
            Strategy("algo_020", DomainType.ALGORITHMIC, "conditional_branching",
                "Short-Circuit Evaluation", "Stop checking conditions once answer is determined",
                "Efficiency in conditionals", ["Order conditions by likelihood", "Exit early when possible", "Skip unnecessary checks"]),
            Strategy("algo_021", DomainType.ALGORITHMIC, "conditional_branching",
                "Nested vs Flat", "Convert nested ifs to flat structure when clearer",
                "Complex nested logic", ["Identify nesting levels", "Flatten if possible", "Use early returns"]),

            # Loop Recognition (7)
            Strategy("algo_022", DomainType.ALGORITHMIC, "loop_recognition",
                "Iteration Identification", "Recognize when same operation repeats on different data",
                "Repetitive calculations", ["Spot the pattern", "Define loop bounds", "Express as iteration"]),
            Strategy("algo_023", DomainType.ALGORITHMIC, "loop_recognition",
                "Loop Invariant", "Identify property that stays true throughout loop execution",
                "Proving correctness", ["State invariant", "Check initial", "Verify each iteration"]),
            Strategy("algo_024", DomainType.ALGORITHMIC, "loop_recognition",
                "Termination Condition", "Ensure loop will eventually stop",
                "Avoiding infinite loops", ["Define progress measure", "Show it decreases", "Set clear end condition"]),
            Strategy("algo_025", DomainType.ALGORITHMIC, "loop_recognition",
                "Accumulator Pattern", "Use running total/result that builds through iterations",
                "Aggregation problems", ["Initialize accumulator", "Update each iteration", "Return final value"]),
            Strategy("algo_026", DomainType.ALGORITHMIC, "loop_recognition",
                "Nested Loop Flattening", "Recognize when nested loops can be simplified",
                "Complex iterations", ["Analyze loop structure", "Find redundancies", "Combine where possible"]),
            Strategy("algo_027", DomainType.ALGORITHMIC, "loop_recognition",
                "Early Exit", "Identify conditions to break out of loop before completion",
                "Search problems", ["Define success condition", "Check each iteration", "Exit when found"]),
            Strategy("algo_028", DomainType.ALGORITHMIC, "loop_recognition",
                "Recursion to Iteration", "Convert recursive thinking to iterative when simpler",
                "Stack-limited problems", ["Identify base case", "Identify recursive case", "Use loop + stack"]),
        ]

    # =========================================================================
    # FACTUAL STRATEGIES (25)
    # Source: TriviaQA, MMLU, Natural Questions patterns
    # =========================================================================

    def _factual_strategies(self) -> List[Strategy]:
        return [
            # History (5)
            Strategy("fact_001", DomainType.FACTUAL, "history_facts",
                "Era Anchoring", "Place events in historical era to narrow down possibilities",
                "Date-related questions", ["Identify era markers", "Cross-reference known events", "Narrow timeframe"]),
            Strategy("fact_002", DomainType.FACTUAL, "history_facts",
                "Cause-Event-Effect Chain", "Link historical events through causal chains",
                "Why questions in history", ["Identify the event", "Trace causes backwards", "Follow effects forward"]),
            Strategy("fact_003", DomainType.FACTUAL, "history_facts",
                "Key Figure Association", "Link events to prominent historical figures",
                "Who questions", ["Identify time period", "List key figures", "Match to event type"]),
            Strategy("fact_004", DomainType.FACTUAL, "history_facts",
                "Geographic Context", "Use location to recall associated historical facts",
                "Where questions", ["Identify region", "Recall regional history", "Match to question"]),
            Strategy("fact_005", DomainType.FACTUAL, "history_facts",
                "Parallel Events", "Compare similar events across different times/places",
                "Pattern recognition", ["Identify event type", "Find similar instances", "Note differences"]),

            # Geography (5)
            Strategy("fact_006", DomainType.FACTUAL, "geography_facts",
                "Spatial Relationships", "Use relative positions (north of, borders) to locate",
                "Location questions", ["Identify reference points", "Apply directional clues", "Pinpoint location"]),
            Strategy("fact_007", DomainType.FACTUAL, "geography_facts",
                "Capital-Country Pairing", "Associate capitals with their countries systematically",
                "Capital questions", ["Identify region", "Recall capital pattern", "Verify match"]),
            Strategy("fact_008", DomainType.FACTUAL, "geography_facts",
                "Physical Feature Association", "Link geographic features to regions",
                "Landform questions", ["Identify feature type", "Recall associated regions", "Match specifics"]),
            Strategy("fact_009", DomainType.FACTUAL, "geography_facts",
                "Climate Zone Mapping", "Use latitude and features to determine climate",
                "Climate questions", ["Estimate latitude", "Consider ocean/mountain effects", "Determine climate"]),
            Strategy("fact_010", DomainType.FACTUAL, "geography_facts",
                "Population Centers", "Associate major cities with their characteristics",
                "City questions", ["Identify region", "Recall major cities", "Match to clues"]),

            # Science Facts (5)
            Strategy("fact_011", DomainType.FACTUAL, "science_facts",
                "Taxonomic Hierarchy", "Use biological classification to organize knowledge",
                "Biology questions", ["Identify kingdom/phylum", "Narrow to class/order", "Find specific answer"]),
            Strategy("fact_012", DomainType.FACTUAL, "science_facts",
                "Periodic Table Patterns", "Use element position to recall properties",
                "Chemistry questions", ["Locate element position", "Apply periodic trends", "Derive property"]),
            Strategy("fact_013", DomainType.FACTUAL, "science_facts",
                "Physical Law Application", "Match phenomena to fundamental laws",
                "Physics questions", ["Identify phenomenon type", "Recall relevant laws", "Apply to scenario"]),
            Strategy("fact_014", DomainType.FACTUAL, "science_facts",
                "Unit Conversion Memory", "Store key conversion factors for quick recall",
                "Measurement questions", ["Identify units involved", "Recall conversion factor", "Apply conversion"]),
            Strategy("fact_015", DomainType.FACTUAL, "science_facts",
                "Scientist-Discovery Link", "Associate major discoveries with their scientists",
                "Who discovered questions", ["Identify discovery type", "Recall field pioneers", "Match to discovery"]),

            # Culture/Arts (5)
            Strategy("fact_016", DomainType.FACTUAL, "culture_arts_facts",
                "Era-Style Association", "Link artistic styles to historical periods",
                "Art period questions", ["Identify time markers", "Recall era styles", "Match characteristics"]),
            Strategy("fact_017", DomainType.FACTUAL, "culture_arts_facts",
                "Creator-Work Pairing", "Associate famous works with their creators",
                "Who created questions", ["Identify work type", "Recall prominent creators", "Match to work"]),
            Strategy("fact_018", DomainType.FACTUAL, "culture_arts_facts",
                "Cultural Origin Tracing", "Link cultural elements to their geographic origins",
                "Origin questions", ["Identify cultural element", "Trace geographic roots", "Verify origin"]),
            Strategy("fact_019", DomainType.FACTUAL, "culture_arts_facts",
                "Movement Characteristics", "Recall defining features of artistic movements",
                "Style identification", ["Identify key features", "Match to movement", "Verify period"]),
            Strategy("fact_020", DomainType.FACTUAL, "culture_arts_facts",
                "Influence Networks", "Trace how artists/thinkers influenced each other",
                "Influence questions", ["Identify the figure", "Map influences in/out", "Trace connections"]),

            # Current Events (5)
            Strategy("fact_021", DomainType.FACTUAL, "current_events",
                "Timeline Construction", "Build chronological sequence of recent events",
                "When questions", ["Identify event type", "Place in timeline", "Cross-reference dates"]),
            Strategy("fact_022", DomainType.FACTUAL, "current_events",
                "Stakeholder Mapping", "Identify key players in current affairs",
                "Who is involved questions", ["Identify the issue", "List stakeholders", "Describe roles"]),
            Strategy("fact_023", DomainType.FACTUAL, "current_events",
                "Trend Identification", "Recognize ongoing patterns in news",
                "What's happening questions", ["Identify topic area", "Recall recent developments", "Note trends"]),
            Strategy("fact_024", DomainType.FACTUAL, "current_events",
                "Geographic Hotspot", "Associate current events with their locations",
                "Where questions", ["Identify event type", "Recall active regions", "Match to specifics"]),
            Strategy("fact_025", DomainType.FACTUAL, "current_events",
                "Policy-Impact Chain", "Link policy decisions to their effects",
                "Why/effect questions", ["Identify policy", "Trace implementation", "Assess impacts"]),
        ]

    # Remaining domains follow same pattern...

    def _procedural_strategies(self) -> List[Strategy]:
        """25 procedural knowledge strategies."""
        return [
            Strategy("proc_001", DomainType.PROCEDURAL, "task_sequencing",
                "Prerequisite Ordering", "Identify what must be done before each step",
                "Multi-step tasks", ["List all steps", "Identify dependencies", "Order correctly"]),
            Strategy("proc_002", DomainType.PROCEDURAL, "task_sequencing",
                "Parallel Task Grouping", "Group steps that can happen simultaneously",
                "Time optimization", ["Identify independent steps", "Group by resource", "Execute in parallel"]),
            Strategy("proc_003", DomainType.PROCEDURAL, "task_sequencing",
                "Critical Path First", "Identify and prioritize the longest dependency chain",
                "Deadline pressure", ["Map all paths", "Find longest chain", "Start early"]),
            Strategy("proc_004", DomainType.PROCEDURAL, "task_sequencing",
                "Checkpoint Insertion", "Add verification points between major steps",
                "Error prevention", ["Identify critical transitions", "Add checks", "Define success criteria"]),
            Strategy("proc_005", DomainType.PROCEDURAL, "task_sequencing",
                "Reversibility Check", "Identify which steps can be undone if needed",
                "Risk management", ["Classify steps by reversibility", "Order reversible first", "Plan rollback"]),
            Strategy("proc_006", DomainType.PROCEDURAL, "task_sequencing",
                "Time Estimation", "Estimate duration for each step to plan overall timeline",
                "Project planning", ["Break into steps", "Estimate each", "Sum with buffer"]),
            Strategy("proc_007", DomainType.PROCEDURAL, "task_sequencing",
                "Milestone Definition", "Define clear completion criteria for each phase",
                "Progress tracking", ["Identify major phases", "Define done criteria", "Plan checkpoints"]),
            Strategy("proc_008", DomainType.PROCEDURAL, "prerequisite_identification",
                "Resource Listing", "Identify all materials/tools needed before starting",
                "Preparation", ["Review all steps", "List required items", "Gather before starting"]),
            Strategy("proc_009", DomainType.PROCEDURAL, "prerequisite_identification",
                "Skill Assessment", "Identify required skills and knowledge gaps",
                "Training needs", ["List required skills", "Assess current level", "Plan learning"]),
            Strategy("proc_010", DomainType.PROCEDURAL, "prerequisite_identification",
                "Environment Setup", "Prepare workspace and conditions before starting",
                "Task preparation", ["Define requirements", "Set up environment", "Verify readiness"]),
            Strategy("proc_011", DomainType.PROCEDURAL, "prerequisite_identification",
                "Permission Checklist", "Identify approvals or access needed",
                "Bureaucratic tasks", ["List required permissions", "Identify gatekeepers", "Obtain before proceeding"]),
            Strategy("proc_012", DomainType.PROCEDURAL, "prerequisite_identification",
                "Dependency Verification", "Confirm prerequisites are actually complete",
                "Quality assurance", ["List all prerequisites", "Verify each", "Document status"]),
            Strategy("proc_013", DomainType.PROCEDURAL, "prerequisite_identification",
                "Alternative Paths", "Identify backup approaches if primary fails",
                "Risk mitigation", ["Identify failure points", "Plan alternatives", "Know when to switch"]),
            Strategy("proc_014", DomainType.PROCEDURAL, "tool_resource_listing",
                "Tool Categorization", "Group tools by function and stage of use",
                "Complex projects", ["List all tools", "Group by function", "Organize by stage"]),
            Strategy("proc_015", DomainType.PROCEDURAL, "tool_resource_listing",
                "Substitute Identification", "Know alternatives for each tool/resource",
                "Limited resources", ["List primary tools", "Identify substitutes", "Note tradeoffs"]),
            Strategy("proc_016", DomainType.PROCEDURAL, "tool_resource_listing",
                "Quantity Calculation", "Determine how much of each resource is needed",
                "Resource planning", ["Estimate per-unit needs", "Multiply by scope", "Add buffer"]),
            Strategy("proc_017", DomainType.PROCEDURAL, "tool_resource_listing",
                "Source Identification", "Know where to obtain each resource",
                "Procurement", ["List resources", "Identify sources", "Compare options"]),
            Strategy("proc_018", DomainType.PROCEDURAL, "tool_resource_listing",
                "Cost Estimation", "Calculate cost for all resources needed",
                "Budgeting", ["List all items", "Research prices", "Sum total cost"]),
            Strategy("proc_019", DomainType.PROCEDURAL, "tool_resource_listing",
                "Availability Check", "Verify resources are actually obtainable",
                "Planning", ["List needed items", "Check availability", "Plan for shortages"]),
            Strategy("proc_020", DomainType.PROCEDURAL, "safety_warning_notes",
                "Hazard Identification", "Identify all potential dangers in the procedure",
                "Safety planning", ["Review each step", "List hazards", "Plan mitigations"]),
            Strategy("proc_021", DomainType.PROCEDURAL, "safety_warning_notes",
                "PPE Requirements", "Specify protective equipment for each hazard",
                "Personal safety", ["Match hazards to PPE", "List all required PPE", "Verify availability"]),
            Strategy("proc_022", DomainType.PROCEDURAL, "safety_warning_notes",
                "Emergency Procedures", "Define what to do if something goes wrong",
                "Crisis preparation", ["List possible emergencies", "Define response", "Practice procedures"]),
            Strategy("proc_023", DomainType.PROCEDURAL, "safety_warning_notes",
                "Warning Placement", "Insert warnings at appropriate points in instructions",
                "Documentation", ["Identify dangerous steps", "Write clear warnings", "Place before action"]),
            Strategy("proc_024", DomainType.PROCEDURAL, "safety_warning_notes",
                "Contraindication Listing", "Specify conditions when procedure should not be performed",
                "Risk prevention", ["List unsafe conditions", "Specify exceptions", "Provide alternatives"]),
            Strategy("proc_025", DomainType.PROCEDURAL, "safety_warning_notes",
                "Cleanup and Disposal", "Specify proper cleanup and waste handling",
                "Environmental safety", ["List waste products", "Specify disposal methods", "Include cleanup steps"]),
        ]

    def _creative_strategies(self) -> List[Strategy]:
        """30 creative writing strategies."""
        return [
            Strategy("creat_001", DomainType.CREATIVE, "narrative_structure",
                "Three-Act Structure", "Organize story into setup, confrontation, resolution",
                "Plot construction", ["Establish world/characters", "Introduce conflict", "Resolve tension"]),
            Strategy("creat_002", DomainType.CREATIVE, "narrative_structure",
                "Hero's Journey", "Follow archetypal pattern of departure, initiation, return",
                "Epic narratives", ["Call to adventure", "Trials and allies", "Return transformed"]),
            Strategy("creat_003", DomainType.CREATIVE, "narrative_structure",
                "In Medias Res", "Begin story in middle of action",
                "Engaging openings", ["Start at key moment", "Fill in backstory later", "Maintain momentum"]),
            Strategy("creat_004", DomainType.CREATIVE, "narrative_structure",
                "Frame Narrative", "Story within a story structure",
                "Complex narratives", ["Establish outer frame", "Tell inner story", "Return to frame"]),
            Strategy("creat_005", DomainType.CREATIVE, "narrative_structure",
                "Nonlinear Timeline", "Present events out of chronological order",
                "Mystery/revelation", ["Plan actual timeline", "Reorder for effect", "Maintain clarity"]),
            Strategy("creat_006", DomainType.CREATIVE, "narrative_structure",
                "Rising Action", "Build tension progressively toward climax",
                "Pacing", ["Start small", "Escalate stakes", "Peak at climax"]),
            Strategy("creat_007", DomainType.CREATIVE, "character_development",
                "Character Arc", "Show character transformation through story",
                "Dynamic characters", ["Establish starting point", "Create challenges", "Show change"]),
            Strategy("creat_008", DomainType.CREATIVE, "character_development",
                "Internal Conflict", "Give characters inner struggles alongside external",
                "Depth", ["Define internal tension", "Mirror in plot", "Resolve together"]),
            Strategy("creat_009", DomainType.CREATIVE, "character_development",
                "Show Don't Tell", "Reveal character through action and dialogue",
                "Immersive writing", ["Identify trait", "Create revealing scene", "Let reader infer"]),
            Strategy("creat_010", DomainType.CREATIVE, "character_development",
                "Backstory Integration", "Weave history into present story naturally",
                "Character depth", ["Create full backstory", "Select relevant parts", "Reveal gradually"]),
            Strategy("creat_011", DomainType.CREATIVE, "character_development",
                "Distinctive Voice", "Give each character unique speech patterns",
                "Dialogue", ["Define character background", "Create speech habits", "Stay consistent"]),
            Strategy("creat_012", DomainType.CREATIVE, "character_development",
                "Motivation Clarity", "Ensure character goals and desires are clear",
                "Character logic", ["Define what they want", "Why they want it", "What they'll do"]),
            Strategy("creat_013", DomainType.CREATIVE, "descriptive_language",
                "Sensory Details", "Engage all five senses in descriptions",
                "Immersion", ["Include sight", "Add sound, smell, touch", "Consider taste"]),
            Strategy("creat_014", DomainType.CREATIVE, "descriptive_language",
                "Metaphor and Simile", "Use comparisons to illuminate meaning",
                "Vivid imagery", ["Identify abstract concept", "Find concrete comparison", "Make connection"]),
            Strategy("creat_015", DomainType.CREATIVE, "descriptive_language",
                "Specific over General", "Use precise details rather than vague descriptions",
                "Clarity", ["Avoid generic terms", "Choose specific words", "Paint clear picture"]),
            Strategy("creat_016", DomainType.CREATIVE, "descriptive_language",
                "Active Voice", "Prefer active constructions for energy",
                "Dynamic prose", ["Identify passive voice", "Convert to active", "Check for clarity"]),
            Strategy("creat_017", DomainType.CREATIVE, "descriptive_language",
                "Rhythm and Pace", "Vary sentence length for effect",
                "Flow", ["Mix short and long", "Match pace to action", "Read aloud to test"]),
            Strategy("creat_018", DomainType.CREATIVE, "descriptive_language",
                "Atmospheric Setting", "Use setting to create mood",
                "Tone setting", ["Choose evocative details", "Match to emotional tone", "Reinforce theme"]),
            Strategy("creat_019", DomainType.CREATIVE, "poetry_form_constraints",
                "Meter Awareness", "Use syllable patterns for rhythm",
                "Poetry", ["Count syllables", "Identify stressed/unstressed", "Maintain pattern"]),
            Strategy("creat_020", DomainType.CREATIVE, "poetry_form_constraints",
                "Rhyme Schemes", "Follow patterns like ABAB, AABB",
                "Formal poetry", ["Choose scheme", "Find rhyming words", "Maintain meaning"]),
            Strategy("creat_021", DomainType.CREATIVE, "poetry_form_constraints",
                "Line Breaks", "Use breaks for emphasis and pacing",
                "Free verse", ["Consider breath pauses", "Emphasize key words", "Create visual shape"]),
            Strategy("creat_022", DomainType.CREATIVE, "poetry_form_constraints",
                "Form Constraints", "Work within traditional forms (sonnet, haiku)",
                "Formal poetry", ["Know form rules", "Work within limits", "Use constraint creatively"]),
            Strategy("creat_023", DomainType.CREATIVE, "poetry_form_constraints",
                "Sound Devices", "Use alliteration, assonance, consonance",
                "Musical quality", ["Identify key sounds", "Repeat strategically", "Avoid excess"]),
            Strategy("creat_024", DomainType.CREATIVE, "poetry_form_constraints",
                "Imagery Density", "Pack more meaning into fewer words",
                "Concision", ["Cut unnecessary words", "Strengthen images", "Trust reader"]),
            Strategy("creat_025", DomainType.CREATIVE, "voice_tone",
                "Consistent Narrator", "Maintain consistent narrative voice",
                "Coherence", ["Define narrator", "Establish voice", "Stay consistent"]),
            Strategy("creat_026", DomainType.CREATIVE, "voice_tone",
                "Tone Matching", "Align tone with story's emotional content",
                "Mood", ["Identify story mood", "Choose matching words", "Maintain throughout"]),
            Strategy("creat_027", DomainType.CREATIVE, "voice_tone",
                "Irony and Subtext", "Say one thing while meaning another",
                "Sophisticated prose", ["Establish surface meaning", "Layer deeper meaning", "Trust reader"]),
            Strategy("creat_028", DomainType.CREATIVE, "voice_tone",
                "Genre Conventions", "Follow or subvert genre expectations",
                "Genre writing", ["Know genre rules", "Decide to follow/break", "Be intentional"]),
            Strategy("creat_029", DomainType.CREATIVE, "voice_tone",
                "Audience Awareness", "Adjust language for intended readers",
                "Accessibility", ["Define audience", "Match vocabulary", "Adjust complexity"]),
            Strategy("creat_030", DomainType.CREATIVE, "voice_tone",
                "Emotional Truth", "Ground fantastical elements in real emotions",
                "Believability", ["Identify emotional core", "Make it genuine", "Let reader connect"]),
        ]

    def _summarization_strategies(self) -> List[Strategy]:
        """25 summarization strategies."""
        return [
            Strategy("summ_001", DomainType.SUMMARIZATION, "main_idea_extraction",
                "Topic Sentence Hunt", "Find the sentence that captures the main point",
                "Paragraph summarization", ["Read full text", "Identify topic sentences", "Combine key ideas"]),
            Strategy("summ_002", DomainType.SUMMARIZATION, "main_idea_extraction",
                "Five W's Framework", "Capture Who, What, When, Where, Why",
                "News summarization", ["Identify each W", "Prioritize by importance", "Combine concisely"]),
            Strategy("summ_003", DomainType.SUMMARIZATION, "main_idea_extraction",
                "Thesis Identification", "Find the central argument or claim",
                "Academic texts", ["Look for argument markers", "Identify main claim", "Note support"]),
            Strategy("summ_004", DomainType.SUMMARIZATION, "main_idea_extraction",
                "Purpose Detection", "Understand why the text was written",
                "Any text", ["Ask what author wants", "Identify intended effect", "Summarize purpose"]),
            Strategy("summ_005", DomainType.SUMMARIZATION, "main_idea_extraction",
                "Structural Analysis", "Use document structure to find main points",
                "Long documents", ["Note headings/sections", "Summarize each section", "Combine"]),
            Strategy("summ_006", DomainType.SUMMARIZATION, "main_idea_extraction",
                "Repetition Detection", "Key ideas are often repeated or rephrased",
                "Any text", ["Note repeated concepts", "These are likely key", "Include in summary"]),
            Strategy("summ_007", DomainType.SUMMARIZATION, "key_detail_selection",
                "Supporting Evidence Selection", "Include key evidence for main claims",
                "Argumentative texts", ["Identify main claims", "Find best evidence", "Include selectively"]),
            Strategy("summ_008", DomainType.SUMMARIZATION, "key_detail_selection",
                "Example Filtering", "Include only the most illustrative examples",
                "Explanatory texts", ["List all examples", "Choose most representative", "Cut the rest"]),
            Strategy("summ_009", DomainType.SUMMARIZATION, "key_detail_selection",
                "Number/Stat Selection", "Include only critical quantitative data",
                "Data-heavy texts", ["List all numbers", "Identify most important", "Cut trivial ones"]),
            Strategy("summ_010", DomainType.SUMMARIZATION, "key_detail_selection",
                "Quote Integration", "Select and integrate key quotes",
                "Interview/speech", ["Identify key quotes", "Paraphrase when possible", "Quote for impact"]),
            Strategy("summ_011", DomainType.SUMMARIZATION, "key_detail_selection",
                "Chronology Compression", "Reduce timeline to key events",
                "Narrative texts", ["List all events", "Select turning points", "Connect concisely"]),
            Strategy("summ_012", DomainType.SUMMARIZATION, "key_detail_selection",
                "Character Pruning", "Focus on main characters only",
                "Story summarization", ["List all characters", "Identify protagonists", "Minimize minor characters"]),
            Strategy("summ_013", DomainType.SUMMARIZATION, "length_matching",
                "Word Budget Allocation", "Allocate words proportionally to importance",
                "Length constraints", ["Count target words", "Rank topics by importance", "Allocate accordingly"]),
            Strategy("summ_014", DomainType.SUMMARIZATION, "length_matching",
                "Sentence Combining", "Merge multiple sentences into one",
                "Compression", ["Identify related sentences", "Combine with connectors", "Cut redundancy"]),
            Strategy("summ_015", DomainType.SUMMARIZATION, "length_matching",
                "Clause Reduction", "Simplify complex sentences",
                "Concision", ["Identify clauses", "Keep essential", "Remove modifiers"]),
            Strategy("summ_016", DomainType.SUMMARIZATION, "length_matching",
                "Progressive Shortening", "Create summaries at multiple lengths",
                "Flexible output", ["Write long version", "Cut to medium", "Cut to short"]),
            Strategy("summ_017", DomainType.SUMMARIZATION, "length_matching",
                "Headline Technique", "Write as if limited to headline length",
                "Extreme compression", ["Capture essence", "Cut all but core", "Polish to length"]),
            Strategy("summ_018", DomainType.SUMMARIZATION, "length_matching",
                "Abstract Style", "Use academic abstract conventions",
                "Research summary", ["Background", "Methods", "Results", "Conclusion"]),
            Strategy("summ_019", DomainType.SUMMARIZATION, "abstraction_levels",
                "High Abstraction", "Summarize at concept level, not detail level",
                "Big picture summaries", ["Identify overarching themes", "Ignore specifics", "Capture essence"]),
            Strategy("summ_020", DomainType.SUMMARIZATION, "abstraction_levels",
                "Low Abstraction", "Preserve specific details and data",
                "Technical summaries", ["Keep numbers and names", "Maintain precision", "Cut context"]),
            Strategy("summ_021", DomainType.SUMMARIZATION, "abstraction_levels",
                "Generalization", "Replace specifics with categories",
                "Broad summaries", ["Identify specific instances", "Name the category", "Use category term"]),
            Strategy("summ_022", DomainType.SUMMARIZATION, "abstraction_levels",
                "Exemplification", "Use one example to represent a category",
                "Illustrative summary", ["Identify category", "Choose best example", "Present as representative"]),
            Strategy("summ_023", DomainType.SUMMARIZATION, "abstraction_levels",
                "Paraphrase vs Quote", "Choose based on importance and style",
                "Any text", ["Assess quote value", "Paraphrase routine ideas", "Quote impactful ones"]),
            Strategy("summ_024", DomainType.SUMMARIZATION, "abstraction_levels",
                "Verb Nominalization", "Convert action descriptions to noun phrases",
                "Compression", ["Find verb phrases", "Convert to nouns", "Reduce word count"]),
            Strategy("summ_025", DomainType.SUMMARIZATION, "abstraction_levels",
                "Implicit to Explicit", "State what the text implies but doesn't say",
                "Interpretation", ["Identify implications", "Make explicit", "Add to summary"]),
        ]

    def _causal_strategies(self) -> List[Strategy]:
        """28 causal reasoning strategies."""
        return [
            Strategy("caus_001", DomainType.CAUSAL, "cause_identification",
                "Temporal Precedence", "Causes must come before effects in time",
                "Identifying causes", ["Establish timeline", "Check what came first", "Verify temporal order"]),
            Strategy("caus_002", DomainType.CAUSAL, "cause_identification",
                "Covariation Detection", "When cause changes, effect should change",
                "Testing causation", ["Vary potential cause", "Observe effect changes", "Confirm relationship"]),
            Strategy("caus_003", DomainType.CAUSAL, "cause_identification",
                "Alternative Elimination", "Rule out other possible causes",
                "Isolating cause", ["List all possible causes", "Eliminate each", "Confirm remaining"]),
            Strategy("caus_004", DomainType.CAUSAL, "cause_identification",
                "Mechanism Identification", "Explain HOW cause produces effect",
                "Deep causation", ["Identify cause and effect", "Trace mechanism", "Verify each step"]),
            Strategy("caus_005", DomainType.CAUSAL, "cause_identification",
                "Proximate vs Ultimate", "Distinguish immediate from root causes",
                "Root cause analysis", ["Identify immediate cause", "Ask why repeatedly", "Find root"]),
            Strategy("caus_006", DomainType.CAUSAL, "cause_identification",
                "Necessary vs Sufficient", "Determine if cause is required or just enough",
                "Causal logic", ["Test if effect occurs without cause", "Test if cause always produces effect", "Classify"]),
            Strategy("caus_007", DomainType.CAUSAL, "cause_identification",
                "Multi-Causal Recognition", "Effects often have multiple causes",
                "Complex causation", ["List all contributing factors", "Assess relative importance", "Note interactions"]),
            Strategy("caus_008", DomainType.CAUSAL, "effect_prediction",
                "Direct Effect Prediction", "Predict immediate consequences",
                "Short-term prediction", ["Identify the cause", "Apply known relationship", "Predict direct effect"]),
            Strategy("caus_009", DomainType.CAUSAL, "effect_prediction",
                "Cascade Effects", "Trace how effects become causes of further effects",
                "Long-term prediction", ["Predict first effect", "Ask what that causes", "Continue chain"]),
            Strategy("caus_010", DomainType.CAUSAL, "effect_prediction",
                "Side Effect Anticipation", "Consider unintended consequences",
                "Holistic prediction", ["Identify intended effect", "Brainstorm side effects", "Assess importance"]),
            Strategy("caus_011", DomainType.CAUSAL, "effect_prediction",
                "Magnitude Estimation", "Predict not just what but how much",
                "Quantitative prediction", ["Identify relationship type", "Estimate magnitude", "Consider limits"]),
            Strategy("caus_012", DomainType.CAUSAL, "effect_prediction",
                "Delay Recognition", "Account for time lag between cause and effect",
                "Timing prediction", ["Identify causal relationship", "Estimate delay", "Account in prediction"]),
            Strategy("caus_013", DomainType.CAUSAL, "effect_prediction",
                "Threshold Effects", "Some causes only work past a threshold",
                "Nonlinear prediction", ["Identify threshold", "Check if exceeded", "Predict accordingly"]),
            Strategy("caus_014", DomainType.CAUSAL, "effect_prediction",
                "Feedback Loop Recognition", "Effects can amplify or dampen causes",
                "Dynamic systems", ["Identify feedback path", "Determine positive/negative", "Project trajectory"]),
            Strategy("caus_015", DomainType.CAUSAL, "chain_of_causation",
                "Causal Chain Mapping", "Draw complete path from initial cause to final effect",
                "Complex scenarios", ["Start with initial cause", "Trace each step", "End at final effect"]),
            Strategy("caus_016", DomainType.CAUSAL, "chain_of_causation",
                "Weakest Link Identification", "Find where causal chain might break",
                "Risk analysis", ["Map the chain", "Assess each link", "Identify vulnerable points"]),
            Strategy("caus_017", DomainType.CAUSAL, "chain_of_causation",
                "Branching Points", "Identify where one cause leads to multiple effects",
                "Complex causation", ["Find cause with multiple effects", "Trace each branch", "Map full tree"]),
            Strategy("caus_018", DomainType.CAUSAL, "chain_of_causation",
                "Convergence Points", "Identify where multiple causes combine",
                "Multi-factor outcomes", ["Find effect with multiple causes", "Trace back each", "Understand interaction"]),
            Strategy("caus_019", DomainType.CAUSAL, "chain_of_causation",
                "Intervention Points", "Find where chain can be broken or modified",
                "Problem solving", ["Map causal chain", "Identify modifiable links", "Plan intervention"]),
            Strategy("caus_020", DomainType.CAUSAL, "chain_of_causation",
                "Historical Reconstruction", "Work backwards from effect to reconstruct causes",
                "Explanation", ["Start with effect", "Ask what caused it", "Continue to origin"]),
            Strategy("caus_021", DomainType.CAUSAL, "chain_of_causation",
                "Parallel Chains", "Multiple independent causal paths to same effect",
                "Redundant causation", ["Identify all paths", "Assess each independently", "Combine"]),
            Strategy("caus_022", DomainType.CAUSAL, "counterfactual_reasoning",
                "What-If Analysis", "Consider what would happen if cause were different",
                "Hypothetical reasoning", ["State the actual cause", "Imagine alternative", "Predict different outcome"]),
            Strategy("caus_023", DomainType.CAUSAL, "counterfactual_reasoning",
                "Absence Testing", "What if the cause hadn't happened?",
                "Necessity testing", ["Remove cause mentally", "Predict outcome", "Assess if effect occurs"]),
            Strategy("caus_024", DomainType.CAUSAL, "counterfactual_reasoning",
                "Alternative History", "Construct plausible alternative sequences",
                "Historical counterfactuals", ["Identify key event", "Change it", "Trace consequences"]),
            Strategy("caus_025", DomainType.CAUSAL, "counterfactual_reasoning",
                "Dose-Response", "What if cause were stronger or weaker?",
                "Magnitude reasoning", ["Vary cause intensity", "Predict effect changes", "Understand relationship"]),
            Strategy("caus_026", DomainType.CAUSAL, "counterfactual_reasoning",
                "Timing Shift", "What if cause happened earlier or later?",
                "Temporal counterfactual", ["Shift timing mentally", "Trace different consequences", "Compare outcomes"]),
            Strategy("caus_027", DomainType.CAUSAL, "counterfactual_reasoning",
                "Actor Substitution", "What if different agent had acted?",
                "Agency analysis", ["Replace actor", "Predict different action", "Trace new consequences"]),
            Strategy("caus_028", DomainType.CAUSAL, "counterfactual_reasoning",
                "Minimal Change Principle", "Only change what's necessary for counterfactual",
                "Rigorous counterfactual", ["Identify minimal change", "Keep all else same", "Predict outcome"]),
        ]

    def _analogical_strategies(self) -> List[Strategy]:
        """28 analogical reasoning strategies."""
        return [
            Strategy("anal_001", DomainType.ANALOGICAL, "proportional_analogies",
                "Relationship Identification", "Determine exact relationship between A and B",
                "A:B :: C:? problems", ["State A-B relationship precisely", "Apply to C", "Find D"]),
            Strategy("anal_002", DomainType.ANALOGICAL, "proportional_analogies",
                "Relationship Types", "Categorize relationship (part-whole, cause-effect, etc.)",
                "Classification", ["Identify relationship category", "Apply same category", "Verify match"]),
            Strategy("anal_003", DomainType.ANALOGICAL, "proportional_analogies",
                "Parallel Structure", "Ensure same grammatical/logical structure",
                "Precise analogies", ["Analyze A-B structure", "Match in C-D", "Verify parallelism"]),
            Strategy("anal_004", DomainType.ANALOGICAL, "proportional_analogies",
                "Degree Matching", "Match intensity/degree of relationship",
                "Subtle analogies", ["Assess A-B degree", "Find matching C-D degree", "Verify proportion"]),
            Strategy("anal_005", DomainType.ANALOGICAL, "proportional_analogies",
                "Order Preservation", "Maintain direction of relationship",
                "Directional analogies", ["Note A→B direction", "Apply same to C→D", "Don't reverse"]),
            Strategy("anal_006", DomainType.ANALOGICAL, "proportional_analogies",
                "Multiple Relationship Check", "Consider if multiple relationships exist",
                "Complex analogies", ["List all A-B relationships", "Find one that works for C", "Verify"]),
            Strategy("anal_007", DomainType.ANALOGICAL, "proportional_analogies",
                "Elimination Strategy", "Rule out wrong answers by testing relationships",
                "Multiple choice", ["Test each option", "Check if relationship matches", "Eliminate mismatches"]),
            Strategy("anal_008", DomainType.ANALOGICAL, "metaphor_understanding",
                "Source Domain Mapping", "Identify what metaphor borrows from",
                "Metaphor analysis", ["Identify source domain", "List its properties", "Map to target"]),
            Strategy("anal_009", DomainType.ANALOGICAL, "metaphor_understanding",
                "Highlighted Features", "Metaphors emphasize some features, hide others",
                "Interpretation", ["What features are highlighted?", "What's hidden?", "Understand emphasis"]),
            Strategy("anal_010", DomainType.ANALOGICAL, "metaphor_understanding",
                "Entailment Generation", "Draw inferences from metaphor",
                "Extending metaphors", ["Identify mapping", "Generate entailments", "Check validity"]),
            Strategy("anal_011", DomainType.ANALOGICAL, "metaphor_understanding",
                "Mixed Metaphor Detection", "Recognize incompatible metaphors",
                "Error detection", ["Identify metaphor domains", "Check compatibility", "Flag conflicts"]),
            Strategy("anal_012", DomainType.ANALOGICAL, "metaphor_understanding",
                "Dead Metaphor Recognition", "Identify metaphors that have become literal",
                "Language analysis", ["Check etymology", "Assess if still metaphorical", "Treat accordingly"]),
            Strategy("anal_013", DomainType.ANALOGICAL, "metaphor_understanding",
                "Metaphor Extension", "Consistently extend metaphor to new cases",
                "Creative reasoning", ["Understand core mapping", "Apply to new situation", "Maintain consistency"]),
            Strategy("anal_014", DomainType.ANALOGICAL, "metaphor_understanding",
                "Literal Interpretation Fallacy", "Avoid taking metaphors too literally",
                "Comprehension", ["Recognize as metaphor", "Extract intended meaning", "Avoid over-interpretation"]),
            Strategy("anal_015", DomainType.ANALOGICAL, "structural_mapping",
                "Structure Alignment", "Match structural elements between domains",
                "Complex analogies", ["Identify structures in source", "Find matches in target", "Map correspondences"]),
            Strategy("anal_016", DomainType.ANALOGICAL, "structural_mapping",
                "Relation Focus", "Prioritize relational matches over surface features",
                "Deep analogies", ["Ignore surface similarities", "Focus on relationships", "Map structures"]),
            Strategy("anal_017", DomainType.ANALOGICAL, "structural_mapping",
                "Systematicity Preference", "Prefer mappings that preserve system of relations",
                "Coherent analogies", ["Identify system of relations", "Map as a whole", "Maintain coherence"]),
            Strategy("anal_018", DomainType.ANALOGICAL, "structural_mapping",
                "One-to-One Mapping", "Each element maps to exactly one element",
                "Consistent mapping", ["Assign mappings", "Check for conflicts", "Resolve ambiguity"]),
            Strategy("anal_019", DomainType.ANALOGICAL, "structural_mapping",
                "Hierarchical Mapping", "Higher-order relations constrain lower-order",
                "Layered analogies", ["Identify hierarchical structure", "Map top-down", "Ensure consistency"]),
            Strategy("anal_020", DomainType.ANALOGICAL, "structural_mapping",
                "Inference Projection", "Use mapping to project inferences from source to target",
                "Analogical inference", ["Identify source inference", "Apply mapping", "Generate target inference"]),
            Strategy("anal_021", DomainType.ANALOGICAL, "structural_mapping",
                "Candidate Elimination", "Eliminate mappings that violate structural constraints",
                "Disambiguation", ["Generate possible mappings", "Test against constraints", "Eliminate invalid"]),
            Strategy("anal_022", DomainType.ANALOGICAL, "cross_domain_transfer",
                "Abstraction Lifting", "Abstract away domain-specific details",
                "Far transfer", ["Identify abstract principle", "Remove domain specifics", "Apply to new domain"]),
            Strategy("anal_023", DomainType.ANALOGICAL, "cross_domain_transfer",
                "Common Schema Extraction", "Find shared schema across domains",
                "Pattern recognition", ["Analyze multiple domains", "Extract common structure", "Name the schema"]),
            Strategy("anal_024", DomainType.ANALOGICAL, "cross_domain_transfer",
                "Adaptation", "Modify source solution for target domain constraints",
                "Problem solving", ["Retrieve source solution", "Identify differences", "Adapt as needed"]),
            Strategy("anal_025", DomainType.ANALOGICAL, "cross_domain_transfer",
                "Negative Transfer Recognition", "Recognize when analogy misleads",
                "Critical thinking", ["Consider disanalogies", "Check if they matter", "Adjust or reject"]),
            Strategy("anal_026", DomainType.ANALOGICAL, "cross_domain_transfer",
                "Multiple Source Combination", "Combine insights from multiple analogies",
                "Creative synthesis", ["Find multiple analogies", "Extract insights from each", "Combine creatively"]),
            Strategy("anal_027", DomainType.ANALOGICAL, "cross_domain_transfer",
                "Bridging Analogies", "Use intermediate analogies to reach distant domains",
                "Far transfer", ["Find intermediate domain", "Bridge in steps", "Verify each step"]),
            Strategy("anal_028", DomainType.ANALOGICAL, "cross_domain_transfer",
                "Domain Distance Assessment", "Judge how far the analogy stretches",
                "Validity assessment", ["Assess domain similarity", "Consider transfer risks", "Adjust confidence"]),
        ]

    def _coding_strategies(self) -> List[Strategy]:
        """35 coding strategies."""
        return [
            Strategy("code_001", DomainType.CODING, "string_operations",
                "String Iteration", "Process string character by character",
                "Character-level operations", ["Initialize result", "Loop through chars", "Process each"]),
            Strategy("code_002", DomainType.CODING, "string_operations",
                "String Slicing", "Extract substrings using slice notation",
                "Substring extraction", ["Identify start/end indices", "Use slice syntax", "Handle edge cases"]),
            Strategy("code_003", DomainType.CODING, "string_operations",
                "String Methods", "Use built-in methods (split, join, replace)",
                "Common operations", ["Identify needed operation", "Find appropriate method", "Apply correctly"]),
            Strategy("code_004", DomainType.CODING, "string_operations",
                "Regular Expressions", "Use regex for pattern matching",
                "Pattern matching", ["Define pattern", "Choose regex function", "Handle matches"]),
            Strategy("code_005", DomainType.CODING, "string_operations",
                "String Building", "Efficiently build strings in loops",
                "Performance", ["Use list for accumulation", "Join at end", "Avoid += in loop"]),
            Strategy("code_006", DomainType.CODING, "string_operations",
                "Unicode Handling", "Properly handle multi-byte characters",
                "International text", ["Specify encoding", "Use unicode methods", "Test with edge cases"]),
            Strategy("code_007", DomainType.CODING, "string_operations",
                "String Formatting", "Use f-strings or format for templating",
                "Output formatting", ["Choose format method", "Insert variables", "Handle types"]),
            Strategy("code_008", DomainType.CODING, "list_array_handling",
                "List Comprehension", "Use comprehensions for concise transformations",
                "Data transformation", ["Identify pattern", "Write comprehension", "Consider readability"]),
            Strategy("code_009", DomainType.CODING, "list_array_handling",
                "Two-Pointer Technique", "Use two indices for efficient traversal",
                "Sorted arrays, palindromes", ["Set up pointers", "Move based on condition", "Process elements"]),
            Strategy("code_010", DomainType.CODING, "list_array_handling",
                "Sliding Window", "Maintain window of elements for subarray problems",
                "Subarray problems", ["Initialize window", "Expand/contract", "Track result"]),
            Strategy("code_011", DomainType.CODING, "list_array_handling",
                "In-Place Modification", "Modify array without extra space",
                "Space optimization", ["Use swap operations", "Track write position", "Avoid new allocations"]),
            Strategy("code_012", DomainType.CODING, "list_array_handling",
                "Frequency Counting", "Use dict/Counter for element counts",
                "Counting problems", ["Create counter", "Count elements", "Query as needed"]),
            Strategy("code_013", DomainType.CODING, "list_array_handling",
                "Matrix Traversal", "Iterate through 2D arrays correctly",
                "Grid problems", ["Use nested loops", "Track row/col", "Handle boundaries"]),
            Strategy("code_014", DomainType.CODING, "list_array_handling",
                "Prefix Sum", "Precompute cumulative sums for range queries",
                "Range sum queries", ["Build prefix array", "Query in O(1)", "Handle index math"]),
            Strategy("code_015", DomainType.CODING, "function_design",
                "Single Responsibility", "Each function does one thing",
                "Clean code", ["Identify purpose", "Remove unrelated code", "Name clearly"]),
            Strategy("code_016", DomainType.CODING, "function_design",
                "Pure Functions", "No side effects, same input gives same output",
                "Testable code", ["Avoid global state", "Return instead of modify", "Make deterministic"]),
            Strategy("code_017", DomainType.CODING, "function_design",
                "Default Parameters", "Provide sensible defaults",
                "API design", ["Identify common cases", "Set defaults", "Document"]),
            Strategy("code_018", DomainType.CODING, "function_design",
                "Early Return", "Return early for edge cases",
                "Clean logic", ["Check edge cases first", "Return early", "Handle main case"]),
            Strategy("code_019", DomainType.CODING, "function_design",
                "Helper Functions", "Extract reusable logic into helpers",
                "Code reuse", ["Identify repeated code", "Extract to function", "Call from multiple places"]),
            Strategy("code_020", DomainType.CODING, "function_design",
                "Recursion Base Case", "Always define base case in recursive functions",
                "Recursion", ["Identify base case", "Put first", "Ensure termination"]),
            Strategy("code_021", DomainType.CODING, "function_design",
                "Type Hints", "Add type annotations for clarity",
                "Documentation", ["Specify parameter types", "Specify return type", "Use for IDE support"]),
            Strategy("code_022", DomainType.CODING, "algorithm_implementation",
                "Binary Search", "Divide search space in half each step",
                "Sorted data search", ["Set bounds", "Check middle", "Narrow half"]),
            Strategy("code_023", DomainType.CODING, "algorithm_implementation",
                "BFS for Shortest Path", "Use breadth-first for unweighted shortest paths",
                "Graph shortest path", ["Initialize queue", "Process level by level", "Track distance"]),
            Strategy("code_024", DomainType.CODING, "algorithm_implementation",
                "DFS for Exploration", "Use depth-first for complete exploration",
                "Tree/graph traversal", ["Use stack/recursion", "Mark visited", "Explore deeply"]),
            Strategy("code_025", DomainType.CODING, "algorithm_implementation",
                "Dynamic Programming", "Store subproblem results to avoid recomputation",
                "Optimal substructure", ["Define state", "Write recurrence", "Memoize or tabulate"]),
            Strategy("code_026", DomainType.CODING, "algorithm_implementation",
                "Greedy Choice", "Make locally optimal choice at each step",
                "Optimization problems", ["Define greedy criterion", "Apply at each step", "Prove correctness"]),
            Strategy("code_027", DomainType.CODING, "algorithm_implementation",
                "Hash for O(1) Lookup", "Use hash table for constant-time access",
                "Lookup optimization", ["Identify lookup need", "Use dict/set", "Consider collisions"]),
            Strategy("code_028", DomainType.CODING, "algorithm_implementation",
                "Sorting as Preprocessing", "Sort data first to enable efficient algorithms",
                "Enabling other techniques", ["Sort first", "Apply algorithm", "Account for sort cost"]),
            Strategy("code_029", DomainType.CODING, "debugging_testing",
                "Print Debugging", "Add strategic print statements",
                "Quick debugging", ["Identify suspicious area", "Print key variables", "Trace execution"]),
            Strategy("code_030", DomainType.CODING, "debugging_testing",
                "Edge Case Testing", "Test boundaries and special cases",
                "Test design", ["Empty input", "Single element", "Maximum values"]),
            Strategy("code_031", DomainType.CODING, "debugging_testing",
                "Assert Statements", "Add assertions for invariants",
                "Defensive coding", ["Identify invariants", "Add asserts", "Check in development"]),
            Strategy("code_032", DomainType.CODING, "debugging_testing",
                "Minimal Reproduction", "Reduce failing case to minimum",
                "Bug isolation", ["Start with failing case", "Remove parts", "Find minimal failure"]),
            Strategy("code_033", DomainType.CODING, "debugging_testing",
                "Unit Test Structure", "Arrange-Act-Assert pattern",
                "Test writing", ["Set up data (Arrange)", "Call function (Act)", "Check result (Assert)"]),
            Strategy("code_034", DomainType.CODING, "debugging_testing",
                "Rubber Duck Debugging", "Explain code out loud to find bugs",
                "Understanding code", ["Read code carefully", "Explain each line", "Spot inconsistencies"]),
            Strategy("code_035", DomainType.CODING, "debugging_testing",
                "Bisect Debugging", "Binary search through history/code to find bug",
                "Finding regression", ["Identify good/bad versions", "Test middle", "Narrow range"]),
        ]

    def _scientific_strategies(self) -> List[Strategy]:
        """26 scientific method strategies."""
        return [
            Strategy("sci_001", DomainType.SCIENTIFIC, "observation_to_hypothesis",
                "Pattern Recognition", "Identify repeating patterns in observations",
                "Initial observation", ["Collect observations", "Look for patterns", "Note regularities"]),
            Strategy("sci_002", DomainType.SCIENTIFIC, "observation_to_hypothesis",
                "Anomaly Focus", "Pay attention to unexpected observations",
                "Discovery", ["Note surprises", "Investigate anomalies", "Question assumptions"]),
            Strategy("sci_003", DomainType.SCIENTIFIC, "observation_to_hypothesis",
                "Question Formulation", "Convert observation to testable question",
                "Research design", ["State observation", "Ask why/how", "Make testable"]),
            Strategy("sci_004", DomainType.SCIENTIFIC, "observation_to_hypothesis",
                "Hypothesis Generation", "Propose explanation for observation",
                "Theory building", ["State proposed explanation", "Make it specific", "Ensure testable"]),
            Strategy("sci_005", DomainType.SCIENTIFIC, "observation_to_hypothesis",
                "Prediction Derivation", "Derive specific predictions from hypothesis",
                "Test design", ["If hypothesis true", "Then what follows?", "Make specific prediction"]),
            Strategy("sci_006", DomainType.SCIENTIFIC, "observation_to_hypothesis",
                "Alternative Hypotheses", "Generate multiple competing explanations",
                "Rigorous science", ["List possible explanations", "Include null hypothesis", "Prepare to distinguish"]),
            Strategy("sci_007", DomainType.SCIENTIFIC, "observation_to_hypothesis",
                "Literature Review", "Check what's already known",
                "Research preparation", ["Search existing studies", "Identify gaps", "Build on prior work"]),
            Strategy("sci_008", DomainType.SCIENTIFIC, "variable_identification",
                "Independent Variable", "What you manipulate",
                "Experiment design", ["Identify what to change", "Determine levels", "Plan manipulation"]),
            Strategy("sci_009", DomainType.SCIENTIFIC, "variable_identification",
                "Dependent Variable", "What you measure",
                "Measurement design", ["Identify outcome", "Choose measure", "Ensure valid"]),
            Strategy("sci_010", DomainType.SCIENTIFIC, "variable_identification",
                "Confounding Variables", "Factors that might affect results",
                "Validity", ["List other factors", "Assess impact", "Plan to control"]),
            Strategy("sci_011", DomainType.SCIENTIFIC, "variable_identification",
                "Operational Definition", "Define variables in measurable terms",
                "Clarity", ["Take abstract concept", "Define measurement", "Ensure reproducible"]),
            Strategy("sci_012", DomainType.SCIENTIFIC, "variable_identification",
                "Moderator Variables", "Factors that change the relationship",
                "Complex relationships", ["Consider context factors", "Plan to test", "Analyze interactions"]),
            Strategy("sci_013", DomainType.SCIENTIFIC, "variable_identification",
                "Mediator Variables", "Factors that explain the mechanism",
                "Mechanism research", ["Propose intermediate step", "Measure it", "Test mediation"]),
            Strategy("sci_014", DomainType.SCIENTIFIC, "control_design",
                "Control Group", "Baseline for comparison",
                "Experiment design", ["Include untreated group", "Match conditions", "Compare to treatment"]),
            Strategy("sci_015", DomainType.SCIENTIFIC, "control_design",
                "Random Assignment", "Randomly assign subjects to conditions",
                "Bias reduction", ["Use randomization", "Ensure balanced groups", "Document process"]),
            Strategy("sci_016", DomainType.SCIENTIFIC, "control_design",
                "Blinding", "Keep experimenters/subjects unaware of condition",
                "Bias prevention", ["Single or double blind", "Use placebos if appropriate", "Maintain separation"]),
            Strategy("sci_017", DomainType.SCIENTIFIC, "control_design",
                "Sample Size", "Use adequate sample for statistical power",
                "Statistical validity", ["Calculate power", "Recruit sufficient N", "Account for dropout"]),
            Strategy("sci_018", DomainType.SCIENTIFIC, "control_design",
                "Replication", "Design for repeatability",
                "Reliability", ["Document all procedures", "Enable replication", "Report fully"]),
            Strategy("sci_019", DomainType.SCIENTIFIC, "control_design",
                "Pilot Testing", "Small scale test before full study",
                "Feasibility", ["Run small version", "Identify issues", "Refine procedures"]),
            Strategy("sci_020", DomainType.SCIENTIFIC, "control_design",
                "Counterbalancing", "Vary order of conditions across subjects",
                "Order effects", ["Identify order effects", "Randomize/balance order", "Analyze separately"]),
            Strategy("sci_021", DomainType.SCIENTIFIC, "result_interpretation",
                "Statistical Significance", "Distinguish signal from noise",
                "Data analysis", ["Calculate p-value", "Compare to alpha", "Interpret carefully"]),
            Strategy("sci_022", DomainType.SCIENTIFIC, "result_interpretation",
                "Effect Size", "Quantify magnitude of effect",
                "Practical significance", ["Calculate effect size", "Interpret magnitude", "Consider practical import"]),
            Strategy("sci_023", DomainType.SCIENTIFIC, "result_interpretation",
                "Confidence Intervals", "Quantify uncertainty in estimates",
                "Uncertainty", ["Calculate CI", "Interpret range", "Assess precision"]),
            Strategy("sci_024", DomainType.SCIENTIFIC, "result_interpretation",
                "Alternative Explanations", "Consider other interpretations",
                "Critical thinking", ["List alternatives", "Evaluate evidence", "Rule out or acknowledge"]),
            Strategy("sci_025", DomainType.SCIENTIFIC, "result_interpretation",
                "Generalization Limits", "Define scope of conclusions",
                "External validity", ["Consider sample", "Note limitations", "State scope"]),
            Strategy("sci_026", DomainType.SCIENTIFIC, "result_interpretation",
                "Future Directions", "Identify next questions",
                "Research progression", ["What remains unknown?", "What would strengthen?", "Plan follow-up"]),
        ]


# Global instance for easy access
STRATEGY_LIBRARY = StrategyLibrary()


def get_strategy(strategy_id: str) -> Optional[Strategy]:
    """Get a strategy by ID."""
    return STRATEGY_LIBRARY.get(strategy_id)


def get_strategies_for_domain(domain: DomainType) -> List[Strategy]:
    """Get all strategies for a domain."""
    return STRATEGY_LIBRARY.get_by_domain(domain)


def get_random_strategy(domain: DomainType) -> Optional[Strategy]:
    """Get a random strategy from a domain."""
    return STRATEGY_LIBRARY.get_random(domain)


if __name__ == "__main__":
    print(f"\nStrategy Library loaded with {STRATEGY_LIBRARY.count()} strategies\n")
    for domain in DomainType:
        count = STRATEGY_LIBRARY.count_by_domain(domain)
        print(f"  {domain.value}: {count} strategies")
    print()
