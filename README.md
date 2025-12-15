# AI→Human Communication: Concise Briefs

- Goal: test whether structured, budgeted prompts improve AI-to-human clarity/trust for dialogue summaries.
- Key results (n=5 DialogSum examples, `gpt-4o-mini`): length ↓43% (87.6→49.8 words), ROUGE-L +0.03, LLM judge prefers concise brief 80% of the time, readability unchanged; HelpSteer verbosity correlates with length (ρ≈0.63).
- Takeaway: simple scaffolding (3 bullets + TL;DR + “Uncertainties”) is a low-effort way to reduce overload without losing salient content.

## Reproduce
1. `uv venv` (already created) and `source .venv/bin/activate`
2. Ensure `OPENAI_API_KEY` or `OPENROUTER_API_KEY` is available (env var or export in `~/.bashrc`).
3. Install deps: `uv sync`
4. Run experiment (reuses cached outputs if present): `python src/run_experiments.py`
   - Raw outputs: `results/model_outputs.jsonl`
   - Metrics: `results/metrics.json`
   - Plot: `results/plots/lengths.png`

## Files
- `planning.md`: research plan
- `src/run_experiments.py`: prompt comparison + metrics
- `results/`: outputs, metrics, plots
- `REPORT.md`: full report and analysis
