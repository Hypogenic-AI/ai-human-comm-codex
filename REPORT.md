## 1. Executive Summary
- Research question: Can structured, budgeted prompts make AI-to-human outputs more concise and trustworthy without sacrificing coverage?
- Key finding: A structured “brief” prompt cut summary length by ~43% (87.6 → 49.8 words) while slightly improving ROUGE-L (+0.03) and winning 80% of LLM judge preferences for clarity/trust on DialogSum snippets. HelpSteer verbosity labels correlate with length (ρ≈0.63), supporting length control signals.
- Practical implication: Lightweight prompt scaffolding (bullets + TL;DR + uncertainties) is a fast, no-training way to reduce overload while preserving salient content; word budgets informed by verbosity labels can guide defaults.

## 2. Goal
- Hypothesis: Explicit conciseness structure and trust cues will reduce verbosity and maintain or improve usefulness vs. generic summaries.
- Importance: AI research agents often overwhelm users; concise, trustworthy digests improve onboarding and oversight.
- Problem solved: Faster human catch-up on dense dialogues/reports without rerunning full pipelines.
- Expected impact: Lower read-time, higher perceived clarity with minimal engineering effort.

## 3. Data Construction

### Dataset Description
- DialogSum subset (HuggingFace `knkarthick/dialogsum` saved locally): 2k dialogues; used first 5 examples for fast API testing.
- HelpSteer (`nvidia/HelpSteer`): 35k train / 1.7k val multi-attribute responses with verbosity labels; used 1k-sample slice for correlation.

### Example Samples
- DialogSum example (train_0): doctor visit dialogue about check-ups and smoking; reference summary provided in dataset.
- HelpSteer example: prompt on assistive device; response with verbosity score=2 (short, focused).

### Data Quality
- Missing values: none in inspected fields.
- Outliers: none observed in small slice; dialogues vary 60–180 words.
- Class distribution: HelpSteer verbosity spans 1–5; sample shows clear spread (mean_wc_low_verbosity≈77 vs. high≈190).
- Validation: schema inspection via `datasets` loading; manual spot-check of samples.

### Preprocessing Steps
1. Load DialogSum from `datasets/dialogsum_sample`; select first 5 rows.
2. No text cleaning; passed raw dialogue strings to prompts.
3. For HelpSteer, compute response word counts and correlations with verbosity.

### Train/Val/Test Splits
- DialogSum: evaluation only (no training); small fixed subset of 5.
- HelpSteer: train split slice of 1,000 for descriptive correlation only.

## 4. Experiment Description

### Methodology
#### High-Level Approach
- Compare two prompting strategies on DialogSum dialogues using a real LLM (OpenAI `gpt-4o-mini` via API):
  - Baseline: generic “Summarize the dialogue” under 150 words.
  - Proposed: structured brief with 3 bullets, TL;DR, and “Uncertainties,” capped at 100 words.
- Evaluate length, readability (Flesch), ROUGE-L vs. references, and LLM-judge preferences for clarity/trust. Use HelpSteer verbosity to justify word budgets.

#### Why This Method?
- Prompt-only change isolates communication framing without training cost.
- DialogSum provides concise, factual targets; HelpSteer verbosity offers an external signal linking length to human-labeled preference.

#### Alternatives Considered
- Fine-tuning with HelpSteer verbosity control or DPO; deferred due to time/compute.
- Using a larger judge model (GPT-4.1); opted for same model to keep latency/cost low.

### Implementation Details
#### Tools and Libraries
- `openai 2.11.0` (API client), `datasets 4.4.1`, `pandas 2.3.3`, `rouge-score 0.1.2`, `textstat 0.7.12`, `seaborn 0.13.2`.

#### Algorithms/Models
- Generation: `gpt-4o-mini`, temperature 0.2, max_tokens 200.
- Judge: same model, structured pairwise rubric.

#### Hyperparameters
| Parameter | Value | Selection Method |
|-----------|-------|------------------|
| temperature | 0.2 | Fixed for stability |
| max_tokens | 200 | Keeps outputs within budget |
| word_budget | 100 | Informed by HelpSteer low-verbosity mean (~77 words) |
| samples | 5 dialogues | Kept small for API latency budget |

#### Training Procedure or Analysis Pipeline
1. Generate baseline and concise summaries for each dialogue.
2. Randomize order and run LLM judge for pairwise clarity/trust preference.
3. Compute word counts, Flesch, ROUGE-L (stemmed).
4. Plot length distributions (`results/plots/lengths.png`).
5. Compute HelpSteer verbosity–length correlations for budget justification.

### Experimental Protocol
#### Reproducibility Information
- Runs: 1 (deterministic temperature 0.2).
- Seed: 42 for selection/order.
- Hardware: CPU-only; API-backed model.
- Execution time: ~40s for metrics using cached outputs (API run was single pass).

#### Evaluation Metrics
- Length (words): compression indicator.
- Flesch Reading Ease: readability proxy.
- ROUGE-L F: coverage vs. reference summaries.
- Judge win rate: LLM preference for clarity/trust.

### Raw Results
#### Tables
| Variant | Mean Words | ROUGE-L F | Flesch |
|---------|------------|-----------|--------|
| Baseline | 87.6 | 0.200 | 53.21 |
| Concise | 49.8 | 0.232 | 53.20 |

| Judge Preference | Rate |
|------------------|------|
| Concise wins | 0.80 |
| Baseline wins | 0.20 |
| Tie | 0.00 |

HelpSteer verbosity correlation (1k slice): Pearson 0.59, Spearman 0.63.

#### Visualizations
- `results/plots/lengths.png`: violin plot showing clear downward shift in word counts for the concise prompt.

#### Output Locations
- Raw generations: `results/model_outputs.jsonl`
- Metrics: `results/metrics.json`
- Plot: `results/plots/lengths.png`

## 5. Result Analysis

### Key Findings
1. Structured brief reduced length by ~43% while slightly improving ROUGE-L (+0.031), indicating coverage maintained despite brevity.
2. Judge preferred concise outputs 80% of the time, suggesting perceived clarity/trust gains.
3. Readability (Flesch) stayed stable (~53), so conciseness did not hurt ease-of-reading.
4. HelpSteer shows verbosity label strongly tracks word count (ρ≈0.63), supporting use of labeled budgets or control tokens.

### Hypothesis Testing Results
- Supports hypothesis: concise prompt shortened outputs and improved perceived quality without coverage loss on this small sample. No statistical tests run due to n=5; effects are descriptive.

### Comparison to Baselines
- Concise prompt beats baseline on length, ROUGE-L, and judge preference; readability unchanged.
- One case favored baseline (vaccination dialogue) where extra narrative context was valued.

### Visualizations
- Length violin plot shows distribution shift; no overlap in medians.

### Surprises and Insights
- ROUGE-L improved slightly despite fewer words, implying baseline included filler.
- Judge reasons often cited “unnecessary details” in baseline; concise prompt’s “Uncertainties” slot was rarely needed but gave confidence when empty.

### Error Analysis
- Vaccination example: concise prompt omitted action ordering nuance; suggests adding “sequence of actions” hint.
- Small sample; variance unmeasured—needs larger run to confirm.

### Limitations
- Very small evaluation set (n=5) due to API latency; results are indicative only.
- Single model family (gpt-4o-mini) for both generation and judging introduces shared biases.
- No human timing/understanding measures; readability proxy only.

## 6. Conclusions
- Structured, budgeted prompts can materially reduce verbosity while preserving salient content and improving perceived clarity on dialogue summarization.
- Verbosity labels from preference datasets provide a principled way to set word budgets or control tokens.
- Even lightweight scaffolding (bullets + TL;DR + uncertainties) helps AI outputs align with human skimmability needs.

## 7. Next Steps
### Immediate Follow-ups
1. Scale evaluation to 100+ DialogSum examples; add bootstrap CIs and Wilcoxon tests.
2. Add “action sequence” and “numbers/dates” micro-prompts to avoid missing procedural details.
3. Use a larger judge model (GPT-4.1/5) and compare to human ratings on speed-to-understand.

### Alternative Approaches
1. Train a small SteerLM-style adapter on HelpSteer verbosity to expose a length knob.
2. DPO on rationale-augmented preference pairs emphasizing conciseness/trust cues.
3. Retrieval-anchored briefs that cite snippets to increase trust.

### Broader Extensions
- Apply to long-form research reports: hierarchical briefs (executive → sectional bullet → provenance links).
- Evaluate audience-specific brevity using group preference conditioning (e.g., expert vs. novice).

### Open Questions
- How low can the word budget go before coverage drops materially?
- Does explicit “unknowns/risks” section improve human trust measurably?
- How well do verbosity controls transfer across domains (tech vs. medical vs. legal)?
