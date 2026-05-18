# Implementation Request Prompt

```text
Task:
[describe the implementation task]

Repository context:
- Use docs/ai_context/00_PROJECT_BRIEF.md as the project overview.
- Use docs/ai_context/01_CURRENT_STATE.md for current active state.
- Use docs/ai_context/02_ARCHITECTURE.md for module boundaries.
- Use docs/ai_context/04_DATA_AND_SHAPES.md for tensor shapes.
- Use docs/ai_context/05_EQUATIONS.md for equation conventions.
- Use docs/ai_context/06_VALIDATION_PROTOCOL.md for checks.

Relevant source files:
[list files]

Constraints:
- Inspect relevant files before proposing code.
- Do not infer code structure from memory.
- Do not rewrite unrelated modules.
- Do not change biological equations unless this task explicitly asks for it.
- Preserve [x_scaled, t_scaled] model input order.
- Preserve model output as log_N unless explicitly asked otherwise.
- Preserve dtype/device consistency.
- Preserve gradient flow.
- Do not use NumPy in differentiable loss calculations.
- Give patch-level changes only.
- Do not paste unchanged code unless necessary.

Expected output:
1. Files/functions to edit.
2. Minimal replacement/addition blocks.
3. Why each change is needed.
4. Smoke tests or validation commands.
5. Documentation updates needed.
```
