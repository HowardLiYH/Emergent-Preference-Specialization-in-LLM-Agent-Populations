# Plan V3: Preference Specialization via Synthetic Rules

## New Core Thesis

> **LLM agents can develop stable, specialized PREFERENCES through competitive selection, analogous to how professionals develop expertise through practice despite similar initial training.**

This reframing acknowledges that:
1. LLMs already possess broad capabilities (like medical students who all learn anatomy)
2. Specialization emerges from **practice and preference**, not knowledge acquisition
3. We measure **preference stability and consistency**, not knowledge gain

---

## Part 1: Synthetic Rule Domains

### 1.1 Design Principles

Each domain is defined by **arbitrary rules** that:
- Cannot be solved by prior knowledge
- Require learning from examples in context
- Have clear success/failure criteria

### 1.2 The 8 Synthetic Domains

| Domain | Rule Description | Example |
|--------|------------------|---------|
| **POSITION** | Correct answer is always at position N | "A) Apple B) Banana C) Cherry" → B (always 2nd) |
| **PATTERN** | Follow a repeating pattern (ABAB, AABB, etc.) | Given sequence, predict next |
| **INVERSE** | Always give the OPPOSITE of obvious answer | "Is fire hot?" → "No" |
| **LENGTH** | Correct answer has specific length | "Which word?" → The 5-letter word |
| **RHYME** | Correct answer rhymes with a keyword | "Key: cat" → "bat" not "dog" |
| **ALPHABET** | Answer based on alphabetical rules | First letter closest to 'M' |
| **MATH_MOD** | Answers follow modular arithmetic | Sum digits, mod 3 determines choice |
| **SEMANTIC** | Answer most/least similar to anchor | "Anchor: happy" → most similar word |

### 1.3 Why These Work

```
Traditional Task:
  Q: "What is 2+2?"
  LLM knows: 4 (from training)
  Strategy: Irrelevant
  
Synthetic Rule Task:
  Rule: "Always pick the LONGEST option"
  Q: "Which? A) Cat B) Elephant C) Dog"
  LLM doesn't know: Must use rule
  Strategy: Essential for success
```

---

## Part 2: Preference Measurement Framework

### 2.1 Core Metrics

#### Metric 1: Rule Preference Index (RPI)
```
RPI(agent, rule) = wins_on_rule / attempts_on_rule

Agent specialization = max(RPI) - mean(RPI)
```
High specialization = strong preference for specific rules.

#### Metric 2: Preference Stability Index (PSI)
```
PSI = correlation(preference_gen_t, preference_gen_t+10)
```
High PSI = preferences remain stable over time (not random drift).

#### Metric 3: Preference Diversity (Population-level)
```
PD = number of distinct primary preferences / number of agents
```
High PD = agents specialize in different rules (niche differentiation).

#### Metric 4: Preference-Performance Correlation (PPC)
```
PPC = correlation(agent_preference_strength, agent_performance)
```
High PPC = stronger preferences lead to better performance.

### 2.2 Validation Criteria

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| RPI variance | > 0.15 | Agents have distinct preferences |
| PSI | > 0.7 | Preferences are stable, not random |
| PD | > 0.5 | Population shows diversity |
| PPC | > 0.3 | Preferences improve performance |

---

## Part 3: Experimental Design

### 3.1 Phase 0: Rule Validation (Pre-experiment)

**Goal:** Confirm synthetic rules create differentiation opportunity.

**Test:**
1. Create 8 rule-specific specialists (one per rule)
2. Each specialist gets rule-specific strategies
3. Test each specialist on ALL rules
4. Measure: diagonal vs off-diagonal performance

**Success Criteria:**
- Diagonal mean > 0.7 (specialists succeed on own rule)
- Off-diagonal mean < 0.4 (specialists fail on other rules)
- Gap > 30%

### 3.2 Phase 1: Preference Emergence (Main Experiment)

**Setup:**
- 12 agents, identical initial prompts
- 8 synthetic rule domains
- 100 generations
- 10 tasks per generation (randomly sampled from domains)

**Competition:**
- All agents attempt same task
- Winner = best performance on that task
- Winner's prompt updated with rule-specific strategy

**Measurement:**
- Track RPI for each agent across generations
- Track PSI at gen 25, 50, 75, 100
- Track PD at each generation

**Success Criteria:**
- Final RPI variance > 0.2
- Final PSI > 0.7
- Final PD > 0.6

### 3.3 Phase 2: Causality Test (Preference Swap)

**Goal:** Prove preferences CAUSE performance differences.

**Test:**
1. Take evolved specialists (after 100 generations)
2. Swap their preference profiles
3. Test: Does performance follow preference?

**Expected Results:**
- Agent A (POSITION specialist) with A's preference → high POSITION performance
- Agent A with B's preference (PATTERN) → low POSITION, high PATTERN

### 3.4 Phase 3: Ablation Studies

| Ablation | What We Remove | Expected Effect |
|----------|----------------|-----------------|
| No competition | Random evolution | Low RPI variance |
| No strategies | Just "you're good at X" | Low performance |
| Shuffled rules | Wrong rule labels | Low RPI, random preferences |
| Single rule only | No diversity pressure | All converge to same rule |

---

## Part 4: Implementation Plan

### 4.1 New Files to Create

```
src/genesis/
├── synthetic_rules.py      # 8 synthetic rule definitions
├── rule_tasks.py           # Task generators for each rule
├── preference_metrics.py   # RPI, PSI, PD, PPC calculations
└── preference_agent.py     # Agent with preference tracking

experiments/
├── exp_rule_validation.py  # Phase 0: Rule validation
├── exp_preference_main.py  # Phase 1: Main experiment  
├── exp_preference_swap.py  # Phase 2: Causality test
└── exp_preference_ablation.py  # Phase 3: Ablations
```

### 4.2 Timeline

| Phase | Duration | Cost Estimate |
|-------|----------|---------------|
| Rule validation | 1 hour | ~$0.10 |
| Main experiment (small) | 2 hours | ~$0.50 |
| Main experiment (full) | 4 hours | ~$5.00 |
| Causality test | 1 hour | ~$0.50 |
| Ablations | 2 hours | ~$2.00 |
| **Total** | **~10 hours** | **~$8.00** |

---

## Part 5: Pivot Decision Tree

```
                    ┌─────────────────────────┐
                    │  Phase 0: Rule          │
                    │  Validation             │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Gap > 30%?           │
                    └───────────┬───────────┘
                       Yes      │      No
                    ┌───────────┴───────────┐
                    │                       │
            ┌───────▼───────┐       ┌───────▼───────┐
            │ Continue to   │       │ PIVOT: Rules  │
            │ Phase 1       │       │ too easy/hard │
            └───────┬───────┘       └───────┬───────┘
                    │                       │
                    │               ┌───────▼───────┐
                    │               │ Adjust rule   │
                    │               │ complexity    │
                    │               └───────────────┘
                    │
            ┌───────▼───────────────┐
            │  Phase 1: Main        │
            │  Experiment           │
            └───────────┬───────────┘
                        │
            ┌───────────▼───────────┐
            │  RPI variance > 0.15? │
            └───────────┬───────────┘
               Yes      │      No
            ┌───────────┴───────────┐
            │                       │
    ┌───────▼───────┐       ┌───────▼───────┐
    │ Preferences   │       │ Variance      │
    │ emerging      │       │ 0.05-0.15?    │
    └───────┬───────┘       └───────┬───────┘
            │                  Yes  │  No
            │               ┌───────┴───────┐
            │               │               │
            │       ┌───────▼───────┐ ┌─────▼─────┐
            │       │ Partial:      │ │ PIVOT:    │
            │       │ Increase gens │ │ Mechanism │
            │       │ or agents     │ │ broken    │
            │       └───────────────┘ └─────┬─────┘
            │                               │
            │                       ┌───────▼───────┐
            │                       │ Study: fitness│
            │                       │ sharing?      │
            │                       │ competition   │
            │                       │ strength?     │
            │                       └───────────────┘
            │
    ┌───────▼───────────────┐
    │  PSI > 0.7?           │
    │  (Stability)          │
    └───────────┬───────────┘
       Yes      │      No
    ┌───────────┴───────────┐
    │                       │
┌───▼───┐           ┌───────▼───────┐
│Stable │           │ PIVOT: Random │
│prefs  │           │ drift, not    │
└───┬───┘           │ true learning │
    │               └───────┬───────┘
    │                       │
    │               ┌───────▼───────┐
    │               │ Add: stronger │
    │               │ reinforcement,│
    │               │ memory        │
    │               └───────────────┘
    │
┌───▼───────────────────────┐
│  PD > 0.5?                │
│  (Diversity)              │
└───────────┬───────────────┘
   Yes      │      No
┌───────────┴───────────┐
│                       │
▼                ┌──────▼──────┐
Continue         │ All agents  │
to Phase 2       │ same pref?  │
                 └──────┬──────┘
                        │
                ┌───────▼───────┐
                │ Add: fitness  │
                │ sharing,      │
                │ niche penalty │
                └───────────────┘

            ┌───────────────────────┐
            │  Phase 2: Preference  │
            │  Swap Test            │
            └───────────┬───────────┘
                        │
            ┌───────────▼───────────┐
            │  Bidirectional        │
            │  transfer?            │
            └───────────┬───────────┘
               Yes      │      No
            ┌───────────┴───────────┐
            │                       │
    ┌───────▼───────┐       ┌───────▼───────┐
    │ ✅ SUCCESS    │       │ Preferences   │
    │               │       │ not causal    │
    │ Write paper   │       └───────┬───────┘
    │ with new      │               │
    │ framing       │       ┌───────▼───────┐
    └───────────────┘       │ PIVOT: Study  │
                            │ what IS causal│
                            │ (model state? │
                            │ randomness?)  │
                            └───────────────┘
```

---

## Part 6: Paper Reframing

### 6.1 New Title Options

1. "Emergent Preference Specialization in Competitive LLM Agent Populations"
2. "From Generalists to Specialists: How Competition Shapes LLM Agent Preferences"
3. "Preference Differentiation Without Knowledge Acquisition in LLM Agents"

### 6.2 Key Claims

1. **Claim 1:** LLM agents develop stable, distinct preferences through competition
2. **Claim 2:** Preferences are causal (swapping preferences changes performance)
3. **Claim 3:** Population-level diversity emerges without explicit diversity rewards
4. **Claim 4:** This mirrors real-world professional specialization

### 6.3 Contribution Positioning

| Related Work | Their Claim | Our Difference |
|--------------|-------------|----------------|
| MARL specialization | Agents learn different policies | We use LLMs, preferences in prompts |
| Prompt engineering | Better prompts improve performance | We show prompts can SPECIALIZE |
| Multi-agent debate | Agents collaborate | We show agents COMPETE |
| Paper 1 (ours) | Trading agents specialize in regimes | We extend to LLM domains |

---

## Part 7: Success Criteria Summary

### For NeurIPS Publication

| Criterion | Threshold | Measured By |
|-----------|-----------|-------------|
| Preferences emerge | RPI variance > 0.2 | Phase 1 |
| Preferences stable | PSI > 0.7 | Phase 1 |
| Population diverse | PD > 0.6 | Phase 1 |
| Preferences causal | Swap test passes | Phase 2 |
| Not random | Controls fail | Phase 3 |
| Novel contribution | Extends Paper 1 to LLMs | Framing |

### Minimum Viable Paper

If full success isn't achieved:
- RPI variance > 0.1 + interesting ablation insights = Workshop paper
- Any positive results + thorough analysis = ArXiv preprint

---

## Part 8: Immediate Next Steps

1. **Implement synthetic_rules.py** - 8 rule definitions
2. **Implement rule_tasks.py** - Task generators
3. **Run Phase 0 validation** - Confirm rules work
4. **If pass: Run Phase 1** - Main experiment
5. **Analyze and decide** - Continue or pivot per decision tree

---

*Plan Version: 3.0*
*Date: January 2, 2026*
*Status: Ready for implementation*

