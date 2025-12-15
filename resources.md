# Resources Catalog

## Summary
Collected six papers on controllable/helpfulness-oriented alignment, three datasets (two preference-focused, one summarization), and two codebases (TRL for alignment training; InstructDS for concise dialogue summarization).

## Papers
Total papers downloaded: 6

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| SteerLM: Attribute Conditioned SFT as an Alternative to RLHF | Dong et al. | 2023 | `papers/2310.05344_steerlm_attribute_conditioned_sft.pdf` | Attribute-conditioned SFT for controllable helpfulness/verbosity |
| HELPSTEER: Multi-attribute Helpfulness Dataset for STEERLM | Wang et al. | 2023 | `papers/2311.09528_helpsteer_multi_attribute_helpfulness.pdf` | 37k multi-attribute preference dataset incl. verbosity |
| Data-Centric Human Preference with Rationales for Direct Preference Alignment | Just et al. | 2024 | `papers/2407.14477_direct_preference_alignment_rationales.pdf` | Rationales improve DPO efficiency and clarity |
| Group Preference Optimization: Few-Shot Alignment of LLMs | Zhao et al. | 2024 | `papers/2310.11523_group_preference_optimization.pdf` | Few-shot group-specific preference steering |
| Instructive Dialogue Summarization with Query Aggregations | Wang et al. | 2023 | `papers/2310.10981_instructive_dialogue_summarization.pdf` | Instructional dialogue summarization with word-budget control |
| Training language models to follow instructions with human feedback | Ouyang et al. | 2022 | `papers/2203.02155_instructgpt_human_feedback.pdf` | RLHF (InstructGPT) baseline |

See `papers/README.md` for more detail.

## Datasets
Total datasets downloaded: 3 (two preference samples, one summarization sample)

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| HelpSteer | HuggingFace `nvidia/HelpSteer` | 35k train / 1.7k val | Multi-attribute preference (incl. verbosity) | `datasets/helpsteer` | Full dataset; sample in `samples/` |
| HH-RLHF (subset) | HuggingFace `Anthropic/hh-rlhf` | 2k subset (full 160k) | Chosen vs rejected conversations | `datasets/hh-rlhf_sample` | Fast subset; instructions for full download |
| DialogSum (subset) | HuggingFace `knkarthick/dialogsum` | 2k subset (full 12.4k+) | Dialogue summarization | `datasets/dialogsum_sample` | Concise summary benchmark; sample provided |

See `datasets/README.md` for download and loading instructions.

## Code Repositories
Total repositories cloned: 2

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| TRL | https://github.com/huggingface/trl | RLHF/DPO/ORPO toolkit for LLM alignment | `code/trl/` | Use for PPO/DPO baselines on preference datasets |
| InstructDS | https://github.com/BinWang28/InstructDS | Instructional dialogue summarization | `code/instructds/` | Recipes for query-aware, word-budgeted summaries |

See `code/README.md` for details.

## Resource Gathering Notes
- **Search Strategy**: Queried arXiv for steerable alignment, preference optimization, and concise summarization; chose papers with explicit verbosity/attribute control or audience adaptation. Selected datasets from HuggingFace aligned to preference learning and dialogue summarization. Cloned codebases providing alignment training (TRL) and concise summarization recipes (InstructDS).
- **Selection Criteria**: Recent (2022–2024) papers with open data/code; explicit treatment of verbosity/conciseness or human preference alignment; datasets with usable licenses and manageable size; codebases with active maintenance.
- **Challenges Encountered**: ArXiv keyword search for “chain-of-density” returned unrelated results; pivoted to steerable alignment and preference datasets where verbosity labels are explicit.
- **Gaps and Workarounds**: Few public datasets directly annotate conciseness preferences; HelpSteer verbosity label used as proxy. Rationales dataset requires checking for released code to replicate; noted in literature review.

## Recommendations for Experiment Design
1. **Primary dataset(s)**: HelpSteer (attribute labels for verbosity/helpfulness); DialogSum for concise summarization evaluation; HH-RLHF subset for general preference alignment baseline.
2. **Baseline methods**: TRL PPO/DPO on HelpSteer+HH-RLHF; attribute-conditioned SFT (SteerLM-style) using HelpSteer verbosity tokens; vanilla instruction-tuned model without verbosity control.
3. **Evaluation metrics**: Preference win-rate focusing on brevity vs completeness; ROUGE with length budgets on DialogSum; verbosity/readability scores; small human study on time-to-understand.
4. **Code to adapt/reuse**: TRL training scripts for PPO/DPO; InstructDS pipeline for generating query-aware, length-controlled summaries; NeMo-Aligner/SteerLM recipes (not cloned but referenced) for attribute-conditioned tokens.
