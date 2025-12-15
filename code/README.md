# Cloned Repositories

## Repo 1: TRL (Transformer Reinforcement Learning)
- **URL**: https://github.com/huggingface/trl
- **Purpose**: RLHF/DPO/ORPO training utilities for LLM alignment; baseline for preference optimization and length/verbosity control experiments.
- **Location**: `code/trl/`
- **Key Files**: `examples/scripts/` (ppo, dpo training), `trl/trainer/` implementations.
- **Notes**: Supports HF transformers; good starting point for InstructGPT-style PPO or DPO on HelpSteer/HH-RLHF.

## Repo 2: InstructDS
- **URL**: https://github.com/BinWang28/InstructDS
- **Purpose**: Instruction-based dialogue summarization (query-aware, controllable length) as in arXiv:2310.10981.
- **Location**: `code/instructds/`
- **Key Files**: `README.md` for data synthesis steps, `src/` for training/inference scripts.
- **Notes**: Provides recipes for generating query-anchored summaries and evaluating concise dialogue outputs.
