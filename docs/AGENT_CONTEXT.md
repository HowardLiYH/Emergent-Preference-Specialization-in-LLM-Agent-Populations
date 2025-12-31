# Agent Context: Emergent Prompt Evolution (Paper 2)

**This file provides context for AI agents working on this project.**

---

## Project Origin

This project is **Paper 2** of a three-paper research series on emergent specialization:

| Paper | Folder | Focus |
|-------|--------|-------|
| Paper 1 | `emergent_specialization/` | Rule-based agents with Thompson Sampling |
| **Paper 2** | `emergent_prompt_evolution/` (THIS) | LLM agents with evolvable prompts |
| Paper 3 | `emergent_civilizations/` | Society dynamics (reproduction, governance) |

---

## Core Research Claim

**"LLM agents, through competition and self-directed prompt evolution, develop emergent specialization that is causally linked to prompt content."**

---

## Key Design Decisions (From v1.5 Conversation)

### 1. Task Domain
- Start with **general reasoning tasks** (math, coding, logic, language)
- Later expand to the 6 prediction domains from Paper 1

### 2. Population Scale
- **Progressive scaling**: Start with 16 agents, scale up to 100
- Paper 3 will scale to 1000 agents

### 3. LLM Choice
- Primary: **GPT-4** for maximum capability
- Cross-validation: Claude 3.5 Sonnet, GPT-3.5-turbo

---

## Critical Concerns to Address (Stanford Professor Review)

### Concern 1: Measurement Problem
- LSI alone is insufficient
- **Solution**: Multi-layer validation
  - Task performance (objective)
  - Semantic clustering (prompt embeddings)
  - Behavioral fingerprint (self-reported)
  - **Counterfactual prompt swap test** (CRITICAL)

### Concern 2: Missing Baselines
Must include:
1. No evolution (prompts fixed)
2. Random evolution (arbitrary changes)
3. No competition (all agents evolve)
4. Fixed specialists (hand-written prompts)

### Concern 3: LLM Confounding
- Run experiments with multiple LLMs
- Ablate on temperature
- Test different evolution prompt templates

### Concern 4: Scope
- This paper focuses ONLY on specialization
- Reproduction/governance saved for Paper 3

---

## Key Files

| File | Purpose |
|------|---------|
| `src/genesis/agent.py` | GenesisAgent with evolvable prompt |
| `src/genesis/tasks.py` | Task pool (math, coding, logic, language) |
| `src/genesis/competition.py` | Winner-take-all competition engine |
| `src/genesis/evolution.py` | Directed and random prompt evolution |
| `src/genesis/metrics.py` | LSI, semantic specialization, behavioral fingerprint |
| `src/genesis/counterfactual.py` | Prompt swap test for causality |
| `src/genesis/simulation.py` | Main orchestrator |

---

## Experiments to Run

| Exp | Description | Priority |
|-----|-------------|----------|
| 1 | Baseline Specialization (16 agents, 100 gen) | HIGH |
| 2 | Ablation: No Evolution | HIGH |
| 3 | Ablation: Random Evolution | HIGH |
| 4 | Ablation: No Competition | HIGH |
| 5 | Temperature Ablation | MEDIUM |
| 6 | Counterfactual Prompt Swap | HIGH |
| 7 | Cross-LLM Replication | HIGH |
| 8 | Scale to 100 agents | MEDIUM |

---

## Success Criteria

| Metric | Target |
|--------|--------|
| LSI at convergence | > 0.6 |
| Directed > No Evolution | +0.3 |
| Directed > Random | +0.15 |
| Prompt swap transfer coefficient | > 0.5 |
| Cross-LLM consistency | Same pattern in 3/3 LLMs |

---

## Related Documents

- `docs/WILD_IDEAS_NEXT_LEVEL_RESEARCH.md` - Original brainstorm
- `.cursor/plans/paper2_prompt_evolution.plan.md` - Detailed plan
- `../emergent_specialization/docs/conversation_v1.5.md` - Full conversation history

---

## Cost Estimate

~$6,650 for all experiments (GPT-4 pricing)

---

## Timeline

10 weeks to paper submission (NeurIPS 2026 / ICML 2026)
