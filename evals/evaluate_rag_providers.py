#!/usr/bin/env python3
"""Evaluate the centralized Latin RAG pipeline across Ollama and OpenRouter."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "models/rag_openrouter/testing/LatinSentenceTestDatav2.json"
DEFAULT_OUTPUT = ROOT / "output/rag_provider_evaluation"
sys.path.insert(0, str(ROOT))


def expected_label(value: str) -> str:
    label = value.upper()
    if "NEGATIVE" in label:
        return "negative"
    if "POSITIVE" in label:
        return "positive"
    return "neutral"


def load_cases(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data if isinstance(data, list) else data.get("test_cases", [])
    if not isinstance(cases, list) or not cases:
        raise SystemExit(f"No test cases found in {path}")
    return cases


def completed_keys(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, int]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            keys.add((str(row["provider"]), int(row["index"])))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return keys


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


async def evaluate(args: argparse.Namespace) -> int:
    from src.app.server_fast import _analyze_with_model, _analyze_with_openrouter

    cases = load_cases(args.cases)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    done = completed_keys(results_path) if args.resume else set()
    auth_header = f"Bearer {os.environ.get('OPENROUTER_API_KEY', '')}"

    if args.provider in {"both", "openrouter"} and auth_header == "Bearer ":
        raise SystemExit("OPENROUTER_API_KEY is required for the OpenRouter evaluation")

    providers = ["ollama", "openrouter"] if args.provider == "both" else [args.provider]
    total = len(cases) * len(providers)
    completed = len(done.intersection({(p, i) for p in providers for i in range(1, len(cases) + 1)}))

    for provider in providers:
        for index, case in enumerate(cases, start=1):
            if (provider, index) in done:
                continue
            sentence = str(case.get("sentence") or "")
            expected = expected_label(str(case.get("expected_sentiment") or ""))
            started = time.perf_counter()
            try:
                if provider == "ollama":
                    result = await _analyze_with_model(
                        sentence,
                        args.ollama_model,
                        "ollama",
                        options={
                            "temperature": 0.0,
                            "top_p": 0.9,
                            "num_predict": args.max_tokens,
                        },
                        include_lexicon_priors=True,
                    )
                    model = args.ollama_model
                else:
                    result = await _analyze_with_openrouter(
                        sentence,
                        openrouter_model=args.openrouter_model,
                        auth_header=auth_header,
                        options={
                            "temperature": 0.0,
                            "top_p": 0.9,
                            "num_predict": args.max_tokens,
                        },
                        include_lexicon_priors=True,
                    )
                    model = args.openrouter_model
                predicted = str(result.get("label") or "neutral")
                row = {
                    "provider": provider,
                    "model": model,
                    "index": index,
                    "sentence": sentence,
                    "expected": expected,
                    "predicted": predicted,
                    "correct": predicted == expected,
                    "latency_seconds": round(time.perf_counter() - started, 3),
                    "rag_enabled": bool((result.get("rag") or {}).get("enabled")),
                    "result": result,
                }
            except Exception as exc:
                row = {
                    "provider": provider,
                    "model": args.ollama_model if provider == "ollama" else args.openrouter_model,
                    "index": index,
                    "sentence": sentence,
                    "expected": expected,
                    "predicted": "error",
                    "correct": False,
                    "latency_seconds": round(time.perf_counter() - started, 3),
                    "rag_enabled": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            append_jsonl(results_path, row)
            completed += 1
            print(
                f"[{completed}/{total}] {provider} case {index}/{len(cases)}: "
                f"{row['predicted']} (expected {expected})",
                flush=True,
            )

    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
    latest = {(row["provider"], row["index"]): row for row in rows}
    summary: dict[str, Any] = {"cases": str(args.cases), "case_count": len(cases), "providers": {}}
    for provider in providers:
        selected = [latest[(provider, i)] for i in range(1, len(cases) + 1) if (provider, i) in latest]
        correct = sum(bool(row.get("correct")) for row in selected)
        summary["providers"][provider] = {
            "model": args.ollama_model if provider == "ollama" else args.openrouter_model,
            "completed": len(selected),
            "correct": correct,
            "accuracy": correct / len(selected) if selected else 0.0,
            "rag_enabled": sum(bool(row.get("rag_enabled")) for row in selected),
            "errors": sum("error" in row for row in selected),
            "predictions": dict(Counter(str(row.get("predicted")) for row in selected)),
            "elapsed_seconds": round(sum(float(row.get("latency_seconds", 0)) for row in selected), 3),
        }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Results: {results_path}")
    print(f"Summary: {summary_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--provider", choices=("both", "ollama", "openrouter"), default="both")
    parser.add_argument("--ollama-model", default="latin-sentiment-llama31-5class")
    parser.add_argument("--openrouter-model", default="meta-llama/llama-3.1-8b-instruct")
    parser.add_argument("--max-tokens", type=int, default=12)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.set_defaults(resume=True)
    return parser.parse_args()


def main() -> int:
    load_dotenv(ROOT / ".env")
    os.environ.setdefault("CLTK_DATA", str(ROOT / "cltk_data"))
    return asyncio.run(evaluate(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
