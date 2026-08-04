Project context: @docs/PROJECT.md   ← read this when you need product context
Current state:   @docs/STATUS.md    ← features, TODO, known gaps, decisions log

## Hard rules (PROJECT.md §14) — enforce mechanically, never by discipline

1. **Never estimate, extrapolate, or synthesize a measurement value.** A failed run
   is recorded as a failure. An unavailable field is `null` plus a reason string.
   One hallucinated number is worse than a hundred missing ones.
2. **n≥5 runs and reported variance**, or the record is invalid.
3. **Measurements are immutable.** Never `UPDATE`. Insert with a new `harness_version`.
4. **Measure end-to-end**, including framework overhead and lowering time.

## Conventions

- Package layout: `src/edgefit/`, `uv` for everything (`uv run …`, `uv sync`).
- Runtime deps stay thin and torch-free; anything needed only to *produce* an
  artifact goes in the `export` extra (keeps the Tier-3 self-hosted runner shippable).
- Tests that touch real hardware are marked `device` and excluded from the fast suite.
- Update `docs/STATUS.md` as part of the change, not afterwards.
