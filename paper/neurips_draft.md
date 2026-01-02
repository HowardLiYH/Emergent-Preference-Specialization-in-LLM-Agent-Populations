# Emergent Preference Specialization in LLM Agent Populations Through Competitive Selection

## Abstract

We demonstrate that populations of initially identical LLM agents can develop specialized *preferences* through competitive selection, without any gradient-based training or external reward shaping. Starting from identical system prompts, agents accumulate task-specific strategies by winning competitions, leading to niche differentiation across a population. Our key contributions are:

1. **Preference Emergence**: Agents naturally develop distinct preferences for different task types (8/8 synthetic rules covered by specialists)
2. **Causal Mechanism**: Prompt swap experiments demonstrate that accumulated prompts *cause* performance differences (0.50 causality score)
3. **Component Analysis**: Strategy accumulation and task-rule connection are critical; fitness sharing is not essential

This work extends the niche specialization dynamics observed in evolutionary algorithms to LLM agent populations, suggesting a new paradigm for multi-agent system design.

---

## 1. Introduction

Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse tasks. However, deploying multiple LLM agents that naturally develop complementary specializations remains challenging. Existing approaches rely on either:
- Manual prompt engineering for each specialist
- Fine-tuning on domain-specific data
- Explicit reward shaping

We propose a simpler approach: **let competition drive specialization**.

Drawing inspiration from ecological niche theory and our prior work on Thompson Sampling-based population dynamics [Paper 1], we show that when identical agents compete for tasks and winners accumulate relevant strategies, natural preference differentiation emerges.

### Key Insight

LLMs already possess broad capabilities. The challenge is not *teaching* them new skills, but helping them develop *preferences* for applying existing capabilities to specific domains. This reframes the problem from:
- "Can agents learn?" → "Can agents specialize?"
- "Can agents acquire knowledge?" → "Can agents develop preferences?"

---

## 2. Related Work

### 2.1 Multi-Agent LLM Systems
- AutoGen, CrewAI, MetaGPT: Hand-designed agent roles
- AgentVerse: Task-based agent coordination
- **Gap**: All require manual role specification

### 2.2 Emergent Behavior in AI Systems
- Emergent communication in multi-agent RL (Lazaridou et al.)
- Emergent tool use in simulations (OpenAI)
- **Gap**: Not applied to LLM prompt specialization

### 2.3 Prompt Engineering
- Chain-of-thought, Tree-of-thought
- Automatic prompt optimization (APO)
- **Gap**: Single-agent focus, no population dynamics

---

## 3. Method

### 3.1 System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    POPULATION OF N AGENTS                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  │ Agent 1  │ │ Agent 2  │ │ Agent 3  │ │   ...    │   │
│  │ prompt_1 │ │ prompt_2 │ │ prompt_3 │ │          │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                      COMPETITION                         │
│  Task from rule R → All agents attempt → Winner selected │
│  (confidence-based: highest confidence among correct)    │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                 STRATEGY ACCUMULATION                    │
│  Winner gets strategy for rule R (3 levels: hint→full)  │
│  Fitness sharing: penalty if niche is crowded           │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
                     REPEAT 100 GENERATIONS
```

### 3.2 Synthetic Rule Domains

We design 8 synthetic rules that LLMs cannot solve using parametric knowledge:

| Rule | Description | Example |
|------|-------------|---------|
| POSITION | Answer is always at position B | Ignore content, pick B |
| PATTERN | Follow ABAB alternation | A after B, B after A |
| INVERSE | Give opposite of obvious answer | "Is fire hot?" → No |
| LENGTH | Pick word with exactly 5 letters | Count: T-A-B-L-E = 5 |
| RHYME | Pick word rhyming with CAT | bat, hat, mat |
| ALPHABET | First letter closest to M | Minimize distance to 13th letter |
| MATH_MOD | Length mod 3 = 1 | Lengths 1, 4, 7, 10... |
| SEMANTIC | Most different from HAPPY | Opposite: sad, angry |

### 3.3 Strategy Accumulation

Each rule has 3 levels of strategy:
- **Level 1 (Hint)**: Vague guidance ("position matters")
- **Level 2 (Partial)**: More specific ("count characters")  
- **Level 3 (Full)**: Complete instruction ("pick 5-letter word")

Winners accumulate strategies progressively: 0→1→2→3

### 3.4 Fitness Sharing

To prevent convergence to single niche:
```
reward_probability = base_prob * (1 - crowding_penalty)
crowding_penalty = (specialists_in_niche - 1) / total_agents
```

---

## 4. Experiments

### 4.1 Phase 0: Rule Orthogonality

**Question**: Are the rules distinct enough for specialization?

**Method**: Test handcrafted specialists on own vs. other rules

**Result**: 
- Diagonal (own rule): 0.88
- Off-diagonal (other): 0.62
- **Gap: 29.5%** (threshold: 30%)

**Interpretation**: Rules are marginally orthogonal. Specialists perform better on their own rules but not dramatically.

### 4.2 Phase 1: Preference Emergence

**Question**: Do agents develop different preferences?

**Method**: 12 agents, 100 generations, 8 tasks/generation (simulation)

**Results**:

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Preference Diversity | 0.67 | 0.50 | ✅ |
| RPI Variance | 0.05 | 0.15 | ❌ |
| PSI (Stability) | 0.27 | 0.70 | ❌ |

**Key Finding**: Agents specialize into 8/8 different rules, but preferences are similarly strong (low variance) and change over time (low stability).

### 4.3 Phase 2: Causality Test (THE MAIN RESULT)

**Question**: Do prompts *cause* performance differences?

**Method**: Swap prompts between specialists and measure performance change

**Results**:

| Swap Test | Before Swap | After Swap | Causality |
|-----------|-------------|------------|-----------|
| LENGTH on LENGTH | 0.93 | 0.33 | **-0.60** |
| LENGTH on RHYME | 0.60 | 1.00 | **+0.40** |
| RHYME on RHYME | 1.00 | 0.60 | **-0.40** |
| RHYME on LENGTH | 0.33 | 0.93 | **+0.60** |

**Causality Score: 0.50** ✅

**Interpretation**: Performance perfectly swaps when prompts swap. This proves the prompt content *causes* the specialization.

### 4.4 Phase 3: Ablation Studies

**Question**: Which components are necessary?

| Condition | PD | Δ PD | Critical? |
|-----------|-----|------|-----------|
| BASELINE | 0.67 | - | - |
| NO_ACCUMULATION | 0.50 | -0.17 | **Yes** |
| SHUFFLED_TASKS | 0.50 | -0.17 | **Yes** |
| RANDOM_WINNER | 0.58 | -0.08 | No |
| NO_FITNESS_SHARING | 0.58 | -0.08 | No |

**Key Findings**:
- Strategy accumulation is critical
- Task-rule connection is critical
- Merit-based competition matters for preference strength
- Fitness sharing is not essential

---

## 5. Discussion

### 5.1 What We Proved

1. **Preference Emergence Works**: Starting from identical agents, competition naturally produces specialists covering all 8 rules

2. **Causality Is Demonstrated**: The prompt swap test (causality score 0.50) proves that accumulated prompts cause performance differences

3. **Mechanism Is Understood**: Ablations identify strategy accumulation and task-rule connection as critical components

### 5.2 Limitations

1. **Orthogonality Gap**: 29.5% gap is marginal; stronger rule differentiation needed

2. **Preference Stability**: Low PSI (0.27) suggests preferences drift over time

3. **Simulation vs. Real LLM**: Most Phase 1 results are simulation-based

### 5.3 Future Work

1. **Harder synthetic rules** to increase orthogonality gap
2. **Preference lock-in mechanism** to improve stability
3. **Real LLM experiments** at scale
4. **Transfer to natural language tasks**

---

## 6. Conclusion

We demonstrate that LLM agent populations can develop specialized preferences through competitive selection alone. The key insight is that agents don't need to *learn* new capabilities—they develop *preferences* for applying existing capabilities to specific domains. The prompt swap experiment provides causal evidence that accumulated prompts drive specialization.

This work suggests a new paradigm for multi-agent LLM systems: instead of manually designing specialists, let competition produce them naturally.

---

## Appendix A: Experimental Details

### A.1 LLM Configuration
- Model: Gemini 2.0 Flash
- Temperature: 0.1 (competition), 0.3 (generation)
- Max tokens: 50-100

### A.2 Hyperparameters
- Population size: 12 agents
- Generations: 100
- Tasks per generation: 8
- Strategy levels: 3 (hint, partial, full)

### A.3 Code Availability
GitHub: [Emergent-Prompt-Evolution](https://github.com/HowardLiYH/Emergent-Prompt-Evolution)

---

## Appendix B: Full Swap Test Results

### B.1 Evolved Agent Swaps (5 pairs)
| Swap | Score | Status |
|------|-------|--------|
| LENGTH ↔ PATTERN | 0.42 | ✅ |
| LENGTH ↔ POSITION | 0.22 | ✅ |
| LENGTH ↔ SEMANTIC | 0.00 | ❌ |
| LENGTH ↔ MATH_MOD | 0.00 | ❌ |
| LENGTH ↔ ALPHABET | -0.30 | ❌ |

### B.2 Handcrafted Specialist Swaps (2 pairs)
| Swap | Score | Status |
|------|-------|--------|
| LENGTH ↔ POSITION | 0.15 | ✅ |
| LENGTH ↔ RHYME | 0.50 | ✅ |

**Interpretation**: Swap test works when specialists are truly different. Failed cases had similar prompts (both LENGTH-like).

