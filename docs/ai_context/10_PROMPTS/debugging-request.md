# Debugging Request Prompt

```text
Debug this PINNmizer issue.

First use the repository context:
- docs/ai_context/01_CURRENT_STATE.md
- docs/ai_context/04_DATA_AND_SHAPES.md
- docs/ai_context/05_EQUATIONS.md
- docs/ai_context/06_VALIDATION_PROTOCOL.md
- docs/ai_context/07_KNOWN_ISSUES.md

Problem:
[describe observed failure]

Evidence:
[paste minimal logs, traceback, metrics, or plots]

Relevant command:
[paste exact command]

Relevant files:
[list files]

Rules:
- Do not guess file contents if files can be inspected.
- Identify the most likely failure modes and rank them.
- Separate shape errors, equation errors, gradient-flow errors, numerical-scaling errors, and optimisation/training errors.
- Do not propose broad rewrites.
- Do not change biological equations unless evidence points there.
- Give the smallest diagnostic or patch that distinguishes between hypotheses.
- If the problem concerns a training run, interpret raw losses, weighted losses, gradient norms, abundance ranges, residual terms, and boundary diagnostics separately.

Expected output:
1. Most likely cause.
2. Evidence for and against that cause.
3. Alternative causes.
4. Minimal diagnostic checks.
5. Minimal code/config changes.
6. What to record in 08_EXPERIMENT_LOG.jsonl.
```
