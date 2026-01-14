# CSE Auditability Guide

**Version:** 2.0  
**Purpose:** Enterprise compliance and interpretability documentation

## Overview

Unlike black-box generalist models, the Competitive Specialist Ecosystem (CSE) provides built-in auditability through its specialist architecture. Each specialist has:

- **Clear expertise boundaries** - defined by regime
- **Inspectable decision logic** - cached prompts are human-readable
- **Predictable failure modes** - failures occur at regime boundaries
- **Full decision trails** - every inference is logged

---

## Why Auditability Matters

### Regulatory Compliance
Many industries require explainable AI:
- **Finance:** Model risk management (SR 11-7)
- **Healthcare:** FDA AI/ML guidelines
- **Legal:** GDPR right to explanation

### Operational Benefits
- **Debugging:** Identify which specialist failed
- **Monitoring:** Track specialist performance over time
- **Improvement:** Target training to weak areas

---

## Specialist Inventory

After training, CSE produces a specialist inventory:

```json
{
  "pure_qa": {
    "tool": "L0",
    "confidence": 0.92,
    "win_rate": 0.88,
    "prompt_template": "Answer directly: {task}"
  },
  "code_math": {
    "tool": "L1",
    "confidence": 0.85,
    "win_rate": 0.82,
    "prompt_template": "Solve with Python:\n{task}"
  },
  "document_qa": {
    "tool": "L3",
    "confidence": 0.78,
    "win_rate": 0.75,
    "prompt_template": "Based on relevant documents, answer:\n{task}"
  }
}
```

This inventory is:
- Human-readable
- Version-controlled
- Auditable

---

## Decision Trace

Every inference includes a full decision trace:

```json
{
  "request_id": "req_12345",
  "timestamp": "2026-01-14T10:30:00Z",
  "task": "Calculate compound interest on $1000 at 5% for 3 years",
  "decision": {
    "regime_detected": "code_math",
    "specialist_selected": "agent_3",
    "tool_used": "L1",
    "confidence": 0.85,
    "prompt_sent": "Solve with Python:\nCalculate compound interest..."
  },
  "response": {
    "answer": "$1157.63",
    "tokens_used": 142,
    "latency_ms": 234
  },
  "audit": {
    "why_this_specialist": "Highest confidence for code_math regime",
    "alternatives_considered": ["agent_7 (0.72)", "agent_1 (0.65)"],
    "fallback_triggered": false
  }
}
```

---

## Failure Analysis

When a specialist fails, the audit trail shows:

```json
{
  "failure_id": "fail_67890",
  "task": "Analyze this chart showing Q4 trends",
  "expected_regime": "chart_analysis",
  "actual_regime_detected": "chart_analysis",
  "specialist": "agent_5",
  "failure_reason": "No chart_analysis specialist trained",
  "remediation": {
    "suggestion": "Train specialist for chart_analysis regime",
    "fallback_used": "L0 (generalist)",
    "fallback_confidence": 0.45
  }
}
```

### Failure Categories

| Category | Description | Example |
|----------|-------------|---------|
| **Coverage Gap** | No specialist for regime | Uncovered regime |
| **Low Confidence** | Specialist uncertain | Below threshold |
| **Tool Mismatch** | Wrong tool selected | L0 for code task |
| **Execution Error** | LLM or tool failure | API timeout |

---

## Compliance Benefits

### Comparison: CSE vs Independent

| Requirement | Independent | CSE |
|-------------|-------------|-----|
| Explain decisions | Hard | Easy (specialist + prompt visible) |
| Audit trail | None | Full (logged per request) |
| Scope boundaries | Undefined | Defined (by regime) |
| Failure attribution | Impossible | Clear (specialist identified) |
| Version control | Difficult | Easy (prompt templates versioned) |
| A/B testing | Complex | Simple (swap specialist) |

### GDPR Article 22 Compliance

The right to explanation is satisfied because:
1. **Decision factors are explicit:** regime, specialist, tool
2. **Logic is inspectable:** prompt templates are human-readable
3. **Alternatives are logged:** other specialists considered

---

## Monitoring Dashboard

Recommended metrics to monitor:

### Specialist Health
```
For each specialist:
  - Win rate (7-day rolling)
  - Confidence trend
  - Error rate
  - Latency p99
```

### Coverage Alerts
```
Alert if:
  - Regime receives > 10 requests with no specialist
  - Specialist confidence drops below 0.6
  - Fallback rate exceeds 20%
```

### Drift Detection
```
Monitor for:
  - Regime distribution shift
  - Specialist performance degradation
  - New task patterns (uncovered regimes)
```

---

## Implementation

### Enable Audit Logging

```python
from v2.src.deploy.cache import SpecialistCache

cache = SpecialistCache()
cache.load("specialists.json")

# Enable audit mode
result = cache.infer(
    task="Calculate 15% tip on $50",
    regime="code_math",
    llm_client=client,
    audit=True  # Returns full decision trace
)

print(result['audit'])
```

### Export Audit Logs

```python
# Export logs for compliance review
cache.export_audit_logs(
    output_path="audit/logs_2026_01.json",
    date_range=("2026-01-01", "2026-01-31")
)
```

---

## Best Practices

1. **Log everything:** Every inference should produce an audit record
2. **Version prompts:** Track changes to specialist prompt templates
3. **Monitor coverage:** Alert on uncovered regimes
4. **Review failures:** Weekly review of failure patterns
5. **Retrain proactively:** Update specialists before performance degrades

---

## Contact

For questions about CSE auditability in your organization:
- Review the specialist inventory in `results/specialists.json`
- Check audit logs in `results/audit/`
- Consult the monitoring dashboard

---

*This document satisfies enterprise audit requirements for AI system transparency.*
