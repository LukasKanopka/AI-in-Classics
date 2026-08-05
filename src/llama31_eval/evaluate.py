"""Evaluate llama3.1:8b on the Latin sentiment evaluation set."""

import json
import re
from pathlib import Path

import ollama


MODEL = "llama3.1:8b"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_FILE = PROJECT_ROOT / "data" / "bert_data" / "eval2.jsonl"
OUTPUT_FILE = Path(__file__).with_name("eval_results.json")
LABELS = [0, 1, 2]


def predict(sentence: str) -> int:
    prompt = f"""Classify the sentiment of this Latin sentence.

Output exactly one number and nothing else:
0 = negative
1 = positive
2 = neutral

Sentence: {sentence}
Number:"""

    response = ollama.generate(
        model=MODEL,
        prompt=prompt,
        options={"temperature": 0},
    )
    text = response["response"].strip()
    match = re.fullmatch(r"[012]", text)
    if not match:
        raise ValueError(f"Expected 0, 1, or 2; model returned {text!r}")
    return int(text)


def main() -> None:
    with EVAL_FILE.open(encoding="utf-8") as file:
        samples = [json.loads(line) for line in file if line.strip()]

    confusion_matrix = [[0 for _ in LABELS] for _ in LABELS]
    correct = 0

    for index, sample in enumerate(samples, start=1):
        actual = int(sample["label"])
        predicted = predict(sample["sentence"])
        confusion_matrix[actual][predicted] += 1
        correct += predicted == actual
        print(f"{index}/{len(samples)}  actual={actual} predicted={predicted}")

    results = {
        "model": MODEL,
        "total_examples": len(samples),
        "correct": correct,
        "accuracy": correct / len(samples) if samples else 0.0,
        "labels": {"0": "negative", "1": "positive", "2": "neutral"},
        "confusion_matrix": {
            "orientation": "rows are actual labels; columns are predicted labels",
            "label_order": LABELS,
            "values": confusion_matrix,
        },
    }

    OUTPUT_FILE.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Results written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
