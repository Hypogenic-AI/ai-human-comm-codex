"""
LLM-based experiment on concise AI->human communication.

Steps:
- Load DialogSum subset and HelpSteer for verbosity cues.
- Generate summaries with a baseline prompt vs a structured concise prompt.
- Judge clarity/trust with an LLM.
- Compute automatic metrics and save outputs/figures.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from datasets import Dataset, load_from_disk
from matplotlib import pyplot as plt
from openai import OpenAI
from rouge_score import rouge_scorer
from textstat import textstat

RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
RAW_OUTPUTS = RESULTS_DIR / "model_outputs.jsonl"
METRICS_PATH = RESULTS_DIR / "metrics.json"

MODEL_NAME = os.environ.get("OPENROUTER_MODEL", "gpt-4o-mini")
BASE_URL = os.environ.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
TEMPERATURE = 0.2
MAX_TOKENS = 200
NUM_EXAMPLES = 5
WORD_BUDGET = 100
SEED = 42


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_key_from_bashrc() -> str | None:
    bashrc = Path.home() / ".bashrc"
    if not bashrc.exists():
        return None
    text = bashrc.read_text()
    for name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        match = re.search(rf'export {name}="([^"]+)"', text)
        if match:
            return match.group(1)
    return None


def get_client() -> OpenAI:
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY") or load_key_from_bashrc()
    if not key:
        raise RuntimeError("Missing OPENROUTER_API_KEY/OPENAI_API_KEY for API access.")
    base_url = BASE_URL if key.startswith("sk-or-") else None
    return OpenAI(api_key=key, base_url=base_url, timeout=30)


def call_model(
    client: OpenAI,
    messages: List[Dict[str, str]],
    model: str = MODEL_NAME,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    retries: int = 3,
    backoff: float = 2.0,
) -> Tuple[str, Dict[str, Any]]:
    """Call the chat model with simple retry."""
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content, resp.usage.model_dump()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"Model call failed after {retries} attempts: {last_error}")


def baseline_messages(dialogue: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a helpful research assistant who writes faithful, fluent summaries."
                " Keep the summary concise but cover key actions and decisions."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Dialogue:\n{dialogue}\n\nTask: Summarize the dialogue for a busy reader."
                " Aim for under 150 words; no bullet formatting required."
            ),
        },
    ]


def concise_messages(dialogue: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You write research handoff briefs that maximize clarity and trust with minimal words."
                " Follow the requested structure exactly."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Dialogue:\n{dialogue}\n\nWrite a concise brief with at most {WORD_BUDGET} words:\n"
                "- Three bullet takeaways (facts/decisions only)\n"
                "- TL;DR: one short sentence\n"
                "- Uncertainties: call out missing info or risks in one bullet; use 'None noted' if clear\n"
                "Avoid filler, keep factual, prefer readable phrasing a human can skim in 20 seconds."
            ),
        },
    ]


def judge_messages(dialogue: str, summary_a: str, summary_b: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are evaluating which summary is better for a human who wants a quick, trustworthy brief."
                " Prefer the option that is clearer, more concise, preserves key facts, and flags uncertainties."
                " Respond with 'A' or 'B' plus one short reason."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Dialogue:\n{dialogue}\n\nSummary A:\n{summary_a}\n\nSummary B:\n{summary_b}\n\n"
                "Which summary better balances conciseness with clarity and trust? Answer with 'A' or 'B' and a 1-line reason."
            ),
        },
    ]


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


def compute_metrics(
    scorer: rouge_scorer.RougeScorer, reference: str, candidate: str
) -> Dict[str, float]:
    rouge_l = scorer.score(reference, candidate)["rougeL"]
    return {
        "rouge_l_f": rouge_l.fmeasure,
        "words": word_count(candidate),
        "flesch": textstat.flesch_reading_ease(candidate),
    }


@dataclass
class ExampleResult:
    example_id: str
    dialogue: str
    reference: str
    baseline: str
    concise: str
    judge_winner: str
    judge_reason: str
    baseline_usage: Dict[str, Any]
    concise_usage: Dict[str, Any]
    judge_usage: Dict[str, Any]
    baseline_metrics: Dict[str, float]
    concise_metrics: Dict[str, float]


def load_dialogsum_subset(n: int) -> Dataset:
    ds = load_from_disk("datasets/dialogsum_sample")
    if n < len(ds):
        return ds.select(range(n))
    return ds


def helpsteer_verbosity_correlation(sample_size: int = 1000) -> Dict[str, float]:
    ds = load_from_disk("datasets/helpsteer")["train"]
    if sample_size < len(ds):
        ds = ds.select(range(sample_size))
    df = ds.to_pandas()
    df["word_count"] = df["response"].str.split().apply(len)
    wc_rank = df["word_count"].rank()
    verb_rank = df["verbosity"].rank()
    return {
        "pearson": float(df["word_count"].corr(df["verbosity"])),
        "spearman": float(wc_rank.corr(verb_rank)),
        "mean_wc_low_verbosity": float(df[df["verbosity"] <= 2]["word_count"].mean()),
        "mean_wc_high_verbosity": float(df[df["verbosity"] >= 4]["word_count"].mean()),
    }


def run_generation(client: OpenAI, data: Dataset) -> List[ExampleResult]:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    results: List[ExampleResult] = []
    token_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for row in data:
        base_content, base_usage = call_model(client, baseline_messages(row["dialogue"]))
        concise_content, concise_usage = call_model(client, concise_messages(row["dialogue"]))

        # Randomize order for judge to reduce position bias
        if random.random() < 0.5:
            first, second = base_content, concise_content
            first_label = "baseline"
        else:
            first, second = concise_content, base_content
            first_label = "concise"

        judge_content, judge_usage = call_model(
            client, judge_messages(row["dialogue"], first, second), max_tokens=64
        )

        winner_label = parse_judge_winner(judge_content, first_label)
        reason = judge_content.strip()

        base_metrics = compute_metrics(scorer, row["summary"], base_content)
        concise_metrics = compute_metrics(scorer, row["summary"], concise_content)

        results.append(
            ExampleResult(
                example_id=row["id"],
                dialogue=row["dialogue"],
                reference=row["summary"],
                baseline=base_content,
                concise=concise_content,
                judge_winner=winner_label,
                judge_reason=reason,
                baseline_usage=base_usage,
                concise_usage=concise_usage,
                judge_usage=judge_usage,
                baseline_metrics=base_metrics,
                concise_metrics=concise_metrics,
            )
        )

        for usage in (base_usage, concise_usage, judge_usage):
            for k in token_totals:
                token_totals[k] += usage.get(k, 0)

    print(f"Total tokens used: {token_totals['total_tokens']}")
    return results


def parse_judge_winner(response: str, first_label: str) -> str:
    text = response.strip().upper()
    winner = "tie"
    if "A" in text and "B" not in text:
        winner = "baseline" if first_label == "baseline" else "concise"
    elif "B" in text and "A" not in text:
        winner = "concise" if first_label == "baseline" else "baseline"
    elif text.startswith("A"):
        winner = "baseline" if first_label == "baseline" else "concise"
    elif text.startswith("B"):
        winner = "concise" if first_label == "baseline" else "baseline"
    return winner


def save_jsonl(path: Path, rows: List[ExampleResult]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row)) + "\n")


def aggregate_metrics(rows: List[ExampleResult]) -> Dict[str, Any]:
    base_lengths = [r.baseline_metrics["words"] for r in rows]
    concise_lengths = [r.concise_metrics["words"] for r in rows]
    base_rouge = [r.baseline_metrics["rouge_l_f"] for r in rows]
    concise_rouge = [r.concise_metrics["rouge_l_f"] for r in rows]
    base_flesch = [r.baseline_metrics["flesch"] for r in rows]
    concise_flesch = [r.concise_metrics["flesch"] for r in rows]
    winners = [r.judge_winner for r in rows]

    return {
        "n": len(rows),
        "length_mean": {"baseline": float(np.mean(base_lengths)), "concise": float(np.mean(concise_lengths))},
        "rouge_l_mean": {"baseline": float(np.mean(base_rouge)), "concise": float(np.mean(concise_rouge))},
        "flesch_mean": {"baseline": float(np.mean(base_flesch)), "concise": float(np.mean(concise_flesch))},
        "judge_win_rate": {
            "concise": winners.count("concise") / len(winners),
            "baseline": winners.count("baseline") / len(winners),
            "tie": winners.count("tie") / len(winners),
        },
    }


def plot_lengths(rows: List[ExampleResult]) -> None:
    data = []
    for r in rows:
        data.append({"variant": "baseline", "words": r.baseline_metrics["words"]})
        data.append({"variant": "concise", "words": r.concise_metrics["words"]})
    df = pd.DataFrame(data)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 4))
    sns.violinplot(data=df, x="variant", y="words", inner="quartile", palette="pastel")
    plt.title("Summary Length Distribution")
    plt.ylabel("Words")
    plt.xlabel("Prompt variant")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "lengths.png", dpi=200)
    plt.close()


def main() -> None:
    pwd = Path.cwd()
    if pwd.name != "ai-human-comm-codex":
        raise SystemExit(f"Run from workspace root; current dir: {pwd}")

    set_seed(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)
    client = get_client()

    dialogsum_subset = load_dialogsum_subset(NUM_EXAMPLES)
    if RAW_OUTPUTS.exists():
        print("Reusing existing outputs from disk.")
        rows = [ExampleResult(**json.loads(line)) for line in RAW_OUTPUTS.read_text().splitlines() if line.strip()]
    else:
        rows = run_generation(client, dialogsum_subset)
        save_jsonl(RAW_OUTPUTS, rows)

    metrics = aggregate_metrics(rows)
    metrics["helpsteer_verbosity"] = helpsteer_verbosity_correlation()
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    plot_lengths(rows)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
