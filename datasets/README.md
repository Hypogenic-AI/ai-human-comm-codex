# Downloaded Datasets

This directory holds local copies or samples for alignment-focused communication experiments. Data files are excluded from git; use the instructions below to reproduce.

## Dataset 1: HelpSteer (multi-attribute helpfulness)
- **Source**: HuggingFace `nvidia/HelpSteer` (CC-BY-4.0)
- **Size**: ~35k train / 1.7k val examples; multi-attribute labels (helpfulness, correctness, coherence, complexity, verbosity)
- **Format**: HuggingFace Dataset saved to `datasets/helpsteer`
- **Task**: Multi-attribute preference modeling; steerable response generation
- **License**: CC-BY-4.0

### Download Instructions
Using HuggingFace Datasets:
```python
from datasets import load_dataset
ds = load_dataset("nvidia/HelpSteer")
ds.save_to_disk("datasets/helpsteer")
```

### Sample Data
See `datasets/helpsteer/samples/sample.json` (20 examples).

### Notes
- Verbosity label is directly useful for conciseness control.
- Pairs well with SteerLM-style attribute tokens.

## Dataset 2: HH-RLHF (subset)
- **Source**: HuggingFace `Anthropic/hh-rlhf`
- **Size**: Local subset of 2k examples (`train[:2000]`); full dataset ~160k train, 8.5k test
- **Format**: HuggingFace Dataset saved to `datasets/hh-rlhf_sample`
- **Task**: Chosen vs rejected conversations for helpful/harmless alignment
- **License**: Anthropic terms on HuggingFace card

### Download Instructions
Subset (fast):
```python
from datasets import load_dataset
ds = load_dataset("Anthropic/hh-rlhf", split="train[:2000]")
ds.save_to_disk("datasets/hh-rlhf_sample")
```
Full dataset:
```python
ds_full = load_dataset("Anthropic/hh-rlhf")
ds_full.save_to_disk("datasets/hh-rlhf_full")
```

### Sample Data
See `datasets/hh-rlhf_sample/samples/sample.json`.

### Notes
- Use for general preference/harmlessness; can craft conciseness-focused comparisons during eval.

## Dataset 3: DialogSum (subset)
- **Source**: HuggingFace `knkarthick/dialogsum`
- **Size**: Local subset of 2k dialogues; full dataset has 12.4k train / 500 val / 1.5k test
- **Format**: HuggingFace Dataset saved to `datasets/dialogsum_sample`
- **Task**: Dialogue summarization (faithful, concise summaries)
- **License**: As specified on dataset card (research use)

### Download Instructions
Subset (fast):
```python
from datasets import load_dataset
ds = load_dataset("knkarthick/dialogsum", split="train[:2000]")
ds.save_to_disk("datasets/dialogsum_sample")
```
Full dataset:
```python
ds_full = load_dataset("knkarthick/dialogsum")
ds_full.save_to_disk("datasets/dialogsum")
```

### Sample Data
See `datasets/dialogsum_sample/samples/sample.json`.

### Notes
- Useful for evaluating concise summaries and query-aware reporting.
