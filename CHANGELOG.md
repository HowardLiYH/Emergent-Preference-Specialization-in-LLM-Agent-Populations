# Changelog

## [v3.2] - 2026-01-03: NeurIPS Readiness & Key Findings

### Major Results
- **Phase 2 Causality Test**: 60.7% pass rate (34/56 pairs) proves prompts cause specialization
- **Prompt Length Ablation**: Short prompts (30 chars) outperform enhanced (900 chars) by 24%
  - Short: 0.918 accuracy, Enhanced: 0.677 accuracy
  - Key insight: "Less is more" for prompt design

### New Features
- **Enhanced 500+ char prompts**: Full persona, steps, examples for all 8 rules
- **Short prompts**: Minimal 30-char instructions for ablation comparison
- **Exclusivity mechanism**: Level 3 agents specialize in one rule only
- **Opaque task mode**: Tasks don't reveal underlying rule

### New Files
- `experiments/exp_phase2_enhanced.py` - Main causality test (56 pairs)
- `experiments/exp_prompt_length_ablation.py` - Short vs enhanced ablation
- `src/genesis/visualization.py` - Publication-quality figure generation
- `paper/neurips_2025.tex` - LaTeX paper draft

### Improvements
- `src/genesis/rule_strategies.py`: Added SHORT_STRATEGY_LEVELS, `use_short` parameter
- `src/genesis/synthetic_rules.py`: Added `opaque` mode for task generation
- `src/genesis/preference_agent.py`: Added exclusivity mechanism
- `src/genesis/analysis.py`: Added t-tests, paired t-test, effect size interpretation

### Results Files
- `results/phase2_enhanced_results.json`
- `results/prompt_length_ablation.json`

---

## [v3.1] - 2026-01-03: Plan V3.1 Implementation

### Major Changes
- Implemented 3-level strategy accumulation
- Confidence-based competition engine
- Fitness sharing for diversity preservation
- Handcrafted ceiling baseline

### New Files  
- `src/genesis/preference_agent.py` - PreferenceAgent with strategy levels
- `src/genesis/competition_v3.py` - Confidence-based winner selection
- `src/genesis/fitness_sharing_v3.py` - Crowding penalty mechanism
- `experiments/exp_preference_main.py` - Phase 1 experiment
- `experiments/exp_preference_swap.py` - Phase 2 swap test
- `experiments/exp_preference_ablation.py` - Phase 3 ablations

---

## [v3.0] - 2026-01-02: Preference Specialization Reframing

### Major Changes
- **New Core Thesis**: "Agents develop specialized PREFERENCES within existing capability space"
- **Synthetic Rule Domains**: 8 arbitrary rule-based domains where prior knowledge cannot help
- **New Metrics**: RPI (Rule Preference Index), PSI (Preference Stability), PD (Preference Diversity)
- **Updated Pivot Decision Tree**: Based on preference emergence, not knowledge acquisition

### Rationale
- Validation tests revealed LLMs already score perfectly on standard tasks
- Strategies cannot improve performance when baseline is 100%
- Reframing acknowledges LLM capabilities while measuring preference development

### New Files
- `docs/PLAN_V3_PREFERENCE_SPECIALIZATION.md` - Complete new plan
- `src/genesis/synthetic_rules.py` - 8 synthetic rule definitions (pending)
- `src/genesis/rule_tasks.py` - Task generators for each rule (pending)
- `src/genesis/preference_metrics.py` - New preference-based metrics (pending)

---

All notable changes to the Emergent Prompt Evolution project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Added
- CHANGELOG.md to track project progress
- Context Engineering v7 plan with Stanford Professor recommendations

### Changed
- Research framing updated to focus on "competitive selection" rather than "emergence"
- Experiment configuration updated to 10 seeds for statistical power

---

## [0.3.0] - 2026-01-01

### Added
- 10 orthogonal domains based on cognitive science frameworks
  - Arithmetic, Algorithmic, Factual, Procedural, Creative
  - Summarization, Causal, Analogical, Coding, Scientific
- Strategy library architecture (280 strategies planned)
- Example library architecture (150 examples planned)
- Validation pre-test with explicit success criteria
- Random strategy baseline for ablation studies
- Fitness sharing diversity mechanism
- Compute cost tracking

### Changed
- Task types redesigned for maximum distinctiveness
- Evolution mechanism changed from replacement to cumulative
- Agent prompt budget increased from 200 to 800 words

---

## [0.2.0] - 2025-12-XX

### Added
- Initial GenesisAgent implementation
- Task pool with Math, Coding, Logic, Language types
- Competition engine with winner-take-all dynamics
- Basic prompt evolution (directed, random, minimal)
- LLM client with rate limiting and retries
- Specialization metrics (LSI, semantic, behavioral)
- Prompt swap counterfactual test

### Issues Identified
- Low LSI (0.15-0.18) indicating weak specialization
- Convergence to single specialty ("logic")
- Shallow prompts lacking context
- No cumulative learning mechanism

---

## [0.1.0] - 2025-12-XX

### Added
- Project structure and README
- Basic agent and task data classes
- Initial experiment scripts

---

## Progress Log

### 2026-01-01
- [ ] Phase 0: Setup
  - [x] Create CHANGELOG.md
  - [ ] Create domains.py with 10 orthogonal TaskTypes
  - [ ] Update project structure

- [ ] Phase 0A: Strategy Library Creation
  - [ ] Create strategy_library.py with 280 strategies
  - [ ] Create example_library.py with 150 examples

- [ ] Phase 1: Core Implementation
  - [ ] Implement cumulative evolution (evolution_v2.py)
  - [ ] Implement fitness sharing
  - [ ] Add compute cost tracking

- [ ] Phase 2: Validation Pre-Test
  - [ ] Create 10 hand-crafted specialist prompts
  - [ ] Run validation test
  - [ ] Verify success criteria

- [ ] Phase 3: Baselines
  - [ ] Implement all 5 baseline conditions

- [ ] Phase 4: Main Experiments
  - [ ] Run 20 agents x 100 gen x 10 seeds

- [ ] Phase 5: Analysis
  - [ ] Generate figures with error bars

- [ ] Phase 6: Paper
  - [ ] Write NeurIPS draft
