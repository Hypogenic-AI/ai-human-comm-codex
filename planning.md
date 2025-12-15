## Research Question
How can we make AI-to-human research communication more concise and trust-building without losing essential coverage?

## Background and Motivation
- LLMs often default to verbose, exhaustive outputs that overload users. Literature (SteerLM, HelpSteer, rationale-augmented DPO, Instructive Dialogue Summarization) shows controllable verbosity and instruction-driven summarization improve perceived helpfulness and clarity.
- Practical need: joining an ongoing research project or skimming many papers fast; users need dense sources distilled into small, trustworthy digests with surface area for spot-checking.

## Hypothesis Decomposition
1. Explicit conciseness scaffolding (budget + structure) reduces output length versus generic prompts while preserving salient content.
2. Adding trust cues (flagging uncertainty/assumptions) improves perceived usefulness/clarity relative to plain concise summaries of similar length.
3. Attribute-conditioned signals from verbosity labels (HelpSteer) correlate with shorter human-preferred outputs and can inform word budgets.

## Proposed Methodology

### Approach
Use a small, real-LLM evaluation comparing baseline summarization prompts to a concise, structured “research digest” prompt. Measure length/readability, coverage vs references, and LLM-judge preferences. Complement with a quick analysis of HelpSteer verbosity labels to set sensible budgets.

### Experimental Steps
1. **Data sanity**: Load DialogSum subset; inspect length stats; ensure fields present. Load HelpSteer sample; inspect verbosity label distribution and correlation with response length to set target word budgets.
2. **Prompt design**: Baseline = generic “summarize the dialogue.” Proposed = budgeted digest (e.g., 3 bullets + 1-line TL;DR + explicit “unknowns/assumptions”) with 80–100 word target.
3. **LLM runs**: On 30–40 DialogSum examples, generate summaries with GPT-4.1 (or GPT-5 if available) for both prompts at same temperature/seed for fairness.
4. **Automatic metrics**: Compute length, Flesch reading ease, ROUGE-L against gold summaries. Record coverage vs verbosity trade-offs.
5. **LLM-judge evaluation**: Pairwise ask a judge model to pick which output is clearer/trust-enhancing under a concise-readability rubric.
6. **Analysis**: Paired comparisons (length, ROUGE-L); bootstrap/Wilcoxon for ROUGE and preference proportions; inspect failure cases qualitatively.
7. **Reporting**: Summarize metrics, include example outputs, note limitations/costs, and propose next steps (e.g., preference fine-tuning with HelpSteer).

### Baselines
- **Baseline prompt**: Plain zero-shot summarization (“Summarize the dialogue.”).
- **Proposed prompt**: Structured digest with word budget and trust cues.
- Optional observational baseline: HelpSteer verbosity label vs length correlation to justify budget choice.

### Evaluation Metrics
- Length (words), compression ratio (dialogue words → summary words).
- Readability: Flesch reading ease.
- ROUGE-L against reference summaries (DialogSum).
- Pairwise LLM-judge preference for clarity/trustfulness.
- Verbosity-label correlation (HelpSteer): Pearson/Spearman between verbosity score and response length.

### Statistical Analysis Plan
- Paired tests (Wilcoxon signed-rank) for ROUGE-L and readability between prompts.
- Bootstrap confidence intervals for preference win rates and mean length differences.
- Report p-values and 95% CIs; check normality heuristically (length distributions).

## Expected Outcomes
- Proposed structured prompt yields shorter summaries with comparable ROUGE-L and higher judged clarity/trust.
- Verbosity labels negatively correlate with length, supporting use as guidance.
- If ROUGE drops substantially, conclude coverage cost and refine prompt.

## Timeline and Milestones
- 0:45h planning + data checks (Phase 1–2).
- 1:15h implementation + prompt runs (Phase 3–4).
- 0:45h analysis/plots (Phase 5).
- 0:30h documentation (Phase 6).

## Potential Challenges
- API key/rate limits or cost: keep sample small (30–40 items) and cache outputs.
- Variance from temperature: fix temp and seeds; run single-shot due to budget.
- LLM-judge bias: use symmetric pairwise rubric and randomize order.
- ROUGE may undervalue concise outputs; mitigate with qualitative examples.

## Success Criteria
- Executed real-LLM comparison with logged prompts and outputs.
- Demonstrated statistically supported length reduction without major coverage loss.
- Clear documentation in REPORT.md with reproducible scripts and saved results.
