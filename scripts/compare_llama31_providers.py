#!/usr/bin/env python3
"""Run the same prompt against Llama 3.1 8B on Ollama and OpenRouter."""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import requests
from dotenv import load_dotenv


DEFAULT_PROMPT = (
    "Classify the sentiment of this Latin sentence as POSITIVE, NEGATIVE, or "
    "NEUTRAL. Respond with only the label: Roma gaudet."
)


def _post_json(url: str, *, payload: dict[str, Any], headers: dict[str, str] | None = None,
               timeout: float = 180.0) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json(), time.perf_counter() - started


def run_ollama(prompt: str, model: str, base_url: str, timeout: float) -> dict[str, Any]:
    data, latency = _post_json(
        f"{base_url.rstrip('/')}/api/generate",
        payload={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 32},
        },
        timeout=timeout,
    )
    return {
        "provider": "ollama",
        "model": model,
        "response": (data.get("response") or "").strip(),
        "latency_seconds": round(latency, 3),
    }


def run_openrouter(prompt: str, model: str, api_url: str, api_key: str,
                   timeout: float) -> dict[str, Any]:
    data, latency = _post_json(
        api_url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "http://localhost",
            "X-Title": "AI in Classics provider comparison",
        },
        payload={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 32,
        },
        timeout=timeout,
    )
    choices = data.get("choices") or []
    content = choices[0].get("message", {}).get("content", "") if choices else ""
    return {
        "provider": "openrouter",
        "model": model,
        "response": str(content).strip(),
        "latency_seconds": round(latency, 3),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--ollama-model", default="llama3.1:8b")
    parser.add_argument(
        "--openrouter-model", default="meta-llama/llama-3.1-8b-instruct"
    )
    parser.add_argument("--provider", choices=("both", "ollama", "openrouter"), default="both")
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()
    results: list[dict[str, Any]] = []

    if args.provider in {"both", "ollama"}:
        results.append(
            run_ollama(
                args.prompt,
                args.ollama_model,
                os.getenv("OLLAMA_URL", os.getenv("OLLAMA_HOST", "http://localhost:11434")),
                args.timeout,
            )
        )

    if args.provider in {"both", "openrouter"}:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit("OPENROUTER_API_KEY is required for the OpenRouter run")
        results.append(
            run_openrouter(
                args.prompt,
                args.openrouter_model,
                os.getenv(
                    "OPENROUTER_API_URL",
                    "https://openrouter.ai/api/v1/chat/completions",
                ),
                api_key,
                args.timeout,
            )
        )

    print(json.dumps({"prompt": args.prompt, "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
