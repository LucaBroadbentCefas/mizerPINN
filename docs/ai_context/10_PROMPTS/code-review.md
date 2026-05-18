# Code Review Prompt

```text
Review the following PINNmizer change.

Repository context to apply:
- The PDE residual target is documented in docs/ai_context/05_EQUATIONS.md.
- Shape conventions are documented in docs/ai_context/04_DATA_AND_SHAPES.md.
- Validation expectations are documented in docs/ai_context/06_VALIDATION_PROTOCOL.md.

Review priorities:

1. Correctness of tensor shapes.
2. Preservation of dtype/device consistency.
3. Preservation of gradient flow through the PDE loss path.
4. No accidental NumPy use in differentiable calculations.
5. Correct coordinate and derivative scaling.
6. Whether biological equations or conventions changed silently.
7. Whether fixed-grid validation code and continuous PDE code are being confused.
8. Whether diagnostics are sufficient to detect failure.
9. Whether the change needs an ADR.
10. Whether docs/ai_context/ should be updated.

Output format:

- Blocking issues
- Non-blocking issues
- Missing tests/checks
- Documentation updates needed
- Minimal patch suggestions only

Do not rewrite the whole file. Do not include unchanged code unless necessary.

Change to review:
[paste diff or files]
```
