# New Chat Bootstrap Prompt

Use this when starting a new ChatGPT session about this repository.

```text
You are helping with the PINNmizer project.

Before answering, use the repository files as the source of truth. Do not infer code structure from memory if files are available.

First read or inspect:

1. docs/ai_context/00_PROJECT_BRIEF.md
2. docs/ai_context/01_CURRENT_STATE.md
3. docs/ai_context/02_ARCHITECTURE.md
4. docs/ai_context/04_DATA_AND_SHAPES.md
5. docs/ai_context/05_EQUATIONS.md
6. docs/ai_context/06_VALIDATION_PROTOCOL.md
7. docs/ai_context/07_KNOWN_ISSUES.md
8. docs/ai_context/12_CONTEXT_INDEX.json

Then inspect the source files directly relevant to the requested task.

Working rules:

- Give minimal, self-contained changes.
- Do not rewrite unrelated code.
- Do not paste unchanged code unless necessary.
- Preserve the model input convention: [x_scaled, t_scaled].
- Preserve model output as log_N unless explicitly asked to change it.
- Preserve dtype/device consistency.
- Preserve PyTorch-only differentiable loss paths.
- Do not detach inside the PDE loss path unless explicitly justified.
- Do not change biological equations while solving a training-loop or documentation issue.
- Distinguish fixed-grid mizer/TMB validation code from continuous/off-grid PDE residual code.
- Identify assumptions and unresolved uncertainties directly.
- If a design decision is made, propose an ADR update.
- At the end of a material change, propose updates to docs/ai_context/01_CURRENT_STATE.md and docs/ai_context/08_EXPERIMENT_LOG.jsonl.

Task:
[insert task here]
```
