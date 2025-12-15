# Literature Review

## Research Area Overview
Work on AI-to-human communication for LLMs focuses on aligning outputs with human preferences for helpfulness, faithfulness, and brevity. Recent papers replace or simplify RLHF with attribute-conditioned supervision, augment preference data with rationales, and personalize alignment to audience groups. Summarization and instruction-tuned dialogue models provide concrete techniques for producing concise, user-focused responses.

## Key Papers

### SteerLM: Attribute Conditioned SFT as an (User-Steerable) Alternative to RLHF (arXiv:2310.05344)
- **Authors**: Yi Dong, Zhilin Wang, Makesh Narsimhan Sreedhar, Xianchao Wu, Oleksii Kuchaiev  
- **Year**: 2023 | **Source**: arXiv
- **Key Contribution**: Attribute-conditioned supervised fine-tuning that exposes knobs (helpfulness, toxicity, humor, verbosity) at inference time, avoiding RLHF complexity.
- **Methodology**: Supervised fine-tuning on multi-attribute data; conditioning on attribute control tokens to steer generations.
- **Datasets Used**: Open-source multi-attribute datasets (includes HelpSteer); evaluations on MT-Bench and human prefs.
- **Results**: Matches or beats RLHF baselines on preference judgments while enabling explicit attribute control.
- **Code Available**: NVIDIA NeMo-Aligner (SteerLM recipes).
- **Relevance**: Demonstrates controllable verbosity/conciseness without RL infrastructure.

### HELPSTEER: Multi-attribute Helpfulness Dataset for STEERLM (arXiv:2311.09528)
- **Authors**: Zhilin Wang, Yi Dong, Jiaqi Zeng, Virginia Adams, Makesh Narsimhan Sreedhar, et al.  
- **Year**: 2023 | **Source**: arXiv
- **Key Contribution**: 37k-sample preference dataset with per-response correctness, coherence, complexity, and verbosity labels.
- **Methodology**: Human annotations on multiple attributes; used to train SteerLM for attribute control.
- **Datasets Used**: Releases HelpSteer dataset (CC-BY-4.0).
- **Results**: Llama2-70B + SteerLM on HelpSteer hits 7.54 MT-Bench, outperforming open models without teacher data.
- **Code Available**: Data on HuggingFace; SteerLM training scripts in NeMo-Aligner.
- **Relevance**: Directly targets verbosity bias; supplies data to teach conciseness vs overlong responses.

### Data-Centric Human Preference with Rationales for Direct Preference Alignment (arXiv:2407.14477)
- **Authors**: Hoang Anh Just, Ming Jin, Anit Sahu, Huy Phan, Ruoxi Jia  
- **Year**: 2024 | **Source**: arXiv
- **Key Contribution**: Adds machine-generated rationales to preference pairs to reduce ambiguity and improve DPO-style training efficiency.
- **Methodology**: Augments preference data with rationales, trains under DPO and other preference losses; analyzes convergence and robustness.
- **Datasets Used**: Existing preference sets enriched with rationales.
- **Results**: Faster convergence and higher final preference alignment versus baseline DPO without rationales.
- **Code Available**: Noted framework, likely released with paper (check GitHub).
- **Relevance**: Rationales explicitly encode why brevity/clarity is preferred, improving learnability of concise communication.

### Group Preference Optimization: Few-Shot Alignment of Large Language Models (arXiv:2310.11523)
- **Authors**: Siyan Zhao, John Dang, Aditya Grover  
- **Year**: 2024 (ICLR) | **Source**: arXiv
- **Key Contribution**: Meta-learns an in-context transformer to adapt outputs to specific group preferences with few examples.
- **Methodology**: Adds a separate transformer module trained via meta-learning across groups; few-shot preference conditioning at inference.
- **Datasets Used**: Grouped opinion datasets across demographics/countries/users.
- **Results**: Outperforms prompt steering and fine-tuning baselines on group alignment with fewer labeled preferences.
- **Code Available**: Paper repo (check author GitHub).
- **Relevance**: Supports audience-specific conciseness levels (e.g., expert vs layperson) with minimal data.

### Instructive Dialogue Summarization with Query Aggregations (arXiv:2310.10981)
- **Authors**: Bin Wang, Zhengyuan Liu, Nancy F. Chen  
- **Year**: 2023 | **Source**: arXiv
- **Key Contribution**: Instruction-tuned dialogue summarization capable of answering specific user queries and adhering to word budgets.
- **Methodology**: Synthesizes query-summary pairs via summary-anchored query generation, filtering, and query-based summarization; trains a unified InstructDS model.
- **Datasets Used**: Three dialogue summarization datasets with generated instructions; evaluated on multiple dialogue summarization/QA sets.
- **Results**: Beats larger baselines on ROUGE and human faithfulness, with controllable summary length.
- **Code Available**: GitHub (BinWang28/InstructDS).
- **Relevance**: Practical pathway to concise, user-tailored dialogue summaries.

### Training language models to follow instructions with human feedback (arXiv:2203.02155)
- **Authors**: Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, et al.  
- **Year**: 2022 | **Source**: arXiv
- **Key Contribution**: InstructGPT—SFT plus RLHF from human preference rankings; foundational alignment method.
- **Methodology**: Collect SFT demonstrations, train reward model on preference pairs, optimize policy via PPO.
- **Datasets Used**: API prompts with human-written demonstrations and ranked completions.
- **Results**: 1.3B InstructGPT preferred over 175B GPT-3 baseline; improved truthfulness and reduced toxic/verbose outputs.
- **Code Available**: Algorithms reproduced in TRL/Open-source RLHF libraries.
- **Relevance**: Establishes RLHF baseline for helpfulness/conciseness against which newer methods compare.

## Common Methodologies
- Attribute-conditioned SFT (SteerLM, HelpSteer) to expose knobs for verbosity/helpfulness without RL.
- Preference optimization (RLHF/PPO, DPO variants) using human comparisons; extended with rationales.
- Personalization via meta-learning or control modules (GPO) to adapt to group-specific communication norms.
- Instruction-tuned summarization with synthetic instructions and query aggregation to control length and focus (InstructDS).

## Standard Baselines
- RLHF (InstructGPT-style PPO) and DPO for preference alignment.
- Supervised instruction tuning on generic instruction datasets (e.g., Alpaca-style) as weak baselines.
- Uncontrolled summarization baselines: BART/CNNDM, GPT-style prompts without length control.

## Evaluation Metrics
- Human/auto preference scores (MT-Bench, pairwise win-rate).
- Summarization metrics (ROUGE, BLEU) with length constraints; faithfulness/QA-based factuality.
- Calibration of style attributes: verbosity scores, toxicity, coherence, correctness.
- Task-specific QA for dialog summaries (response correctness to user queries).

## Datasets in the Literature
- HelpSteer (multi-attribute helpfulness incl. verbosity) — used by SteerLM.
- HH-RLHF / similar human preference datasets — RLHF baselines.
- Dialogue summarization corpora (DialogSum, SAMSum, synthesized query-summary triples) — InstructDS.
- Group-specific preference sets for demographic adaptation — GPO.

## Gaps and Opportunities
- Few open datasets label conciseness explicitly beyond verbosity proxies; need richer brevity/focus signals.
- Limited audience-specific brevity controls (expert vs novice); GPO hints at solutions but more domains needed.
- Rationales for preferences are underused; could combine with attribute-conditioned SFT for clearer learning signals.
- Metrics often ignore user effort (time-to-comprehend); lightweight human studies or readability scores could fill this gap.

## Recommendations for Our Experiment
- **Recommended datasets**: HelpSteer for attribute-conditioned conciseness; HH-RLHF (subset) for general helpfulness/harmfulness trade-offs; DialogSum for concise dialogue summaries.  
- **Recommended baselines**: InstructGPT-style PPO or TRL DPO; SteerLM-style attribute-conditioned SFT; vanilla instruction-tuned model without conciseness control.  
- **Recommended metrics**: Preference win-rate on conciseness-focused pairwise evals; ROUGE with length budgets for summarization; verbosity/readability scores; simple human eval on speed-to-understand.  
- **Methodological considerations**: Start with attribute tokens controlling verbosity (from HelpSteer/SteerLM), optionally fine-tune with rationale-augmented DPO to encode why concise replies are preferred. Consider group-specific adapters if targeting different user segments.
