# scripts/eval_multilingual_sentiment.py

from pathlib import Path
import json

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

MODEL_PATH = "cardiffnlp/" "bert-base-multilingual-cased-sentiment-multilingual"

DATA_PATH = Path("data/bert_data/eval2.jsonl")
OUTPUT_PATH = Path("output/evaluations/" "multilingual_sentiment_eval.txt")

BATCH_SIZE = 32
MAX_LENGTH = 128

# Your dataset's label ordering.
DATASET_ID_TO_LABEL = {
    0: "negative",
    1: "positive",
    2: "neutral",
}

DATASET_LABEL_TO_ID = {
    label: label_id for label_id, label in DATASET_ID_TO_LABEL.items()
}


def normalize_label(label: str) -> str:
    normalized = label.strip().lower()

    if "neg" in normalized:
        return "negative"

    if "neu" in normalized:
        return "neutral"

    if "pos" in normalized:
        return "positive"

    raise ValueError(f"Unknown sentiment label: {label}")


def load_samples(path: Path) -> list[dict]:
    samples = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            sample = json.loads(line)

            if "sentence" not in sample or "label" not in sample:
                raise ValueError(f"Missing sentence or label on line {line_number}")

            samples.append(sample)

    if not samples:
        raise ValueError(f"No samples found in {path}")

    return samples


def create_model_to_dataset_mapping(
    model: AutoModelForSequenceClassification,
) -> dict[int, int]:
    mapping = {}

    for model_label_id, model_label_name in model.config.id2label.items():
        model_label_id = int(model_label_id)
        normalized_name = normalize_label(model_label_name)

        mapping[model_label_id] = DATASET_LABEL_TO_ID[normalized_name]

    expected_labels = {"negative", "positive", "neutral"}
    configured_labels = {
        normalize_label(label) for label in model.config.id2label.values()
    }

    if configured_labels != expected_labels:
        raise ValueError(
            "The selected model does not have the expected "
            f"three sentiment labels: {model.config.id2label}"
        )

    return mapping


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Model: {MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Do not pass num_labels here. The checkpoint already contains
    # its trained three-class classification head.
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    model.to(device)
    model.eval()

    print(f"Model labels: {model.config.id2label}")

    model_to_dataset_id = create_model_to_dataset_mapping(model)

    print("Model-to-dataset label mapping: " f"{model_to_dataset_id}")

    samples = load_samples(DATA_PATH)

    sentences = [sample["sentence"] for sample in samples]
    actual_labels = np.asarray([int(sample["label"]) for sample in samples])

    predicted_labels = []
    prediction_confidences = []

    with torch.inference_mode():
        for start in range(0, len(sentences), BATCH_SIZE):
            batch_sentences = sentences[start : start + BATCH_SIZE]

            inputs = tokenizer(
                batch_sentences,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )

            inputs = {key: value.to(device) for key, value in inputs.items()}

            outputs = model(**inputs)
            probabilities = torch.softmax(
                outputs.logits,
                dim=-1,
            )

            model_predictions = probabilities.argmax(dim=-1)
            confidences = probabilities.max(dim=-1).values

            for model_prediction in model_predictions.tolist():
                predicted_labels.append(model_to_dataset_id[model_prediction])

            prediction_confidences.extend(confidences.cpu().tolist())

    predicted_labels = np.asarray(predicted_labels)

    accuracy = accuracy_score(
        actual_labels,
        predicted_labels,
    )
    macro_f1 = f1_score(
        actual_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        actual_labels,
        predicted_labels,
        labels=[0, 1, 2],
        target_names=["NEG", "POS", "NEU"],
        digits=4,
        zero_division=0,
    )

    matrix = confusion_matrix(
        actual_labels,
        predicted_labels,
        labels=[0, 1, 2],
    )

    result_lines = [
        f"Model: {MODEL_PATH}",
        f"Samples: {len(samples)}",
        f"Model labels: {model.config.id2label}",
        f"Accuracy: {accuracy:.4f}",
        f"Macro F1: {macro_f1:.4f}",
        "",
        "Classification report:",
        report,
        "Confusion matrix:",
        "Rows: actual NEG, POS, NEU",
        "Columns: predicted NEG, POS, NEU",
        str(matrix),
        "",
        "Predictions:",
    ]

    for sample, actual, predicted, confidence in zip(
        samples,
        actual_labels,
        predicted_labels,
        prediction_confidences,
        strict=True,
    ):
        status = "CORRECT" if actual == predicted else "WRONG"

        result_lines.append(
            f"{status} | "
            f"actual={DATASET_ID_TO_LABEL[int(actual)]} | "
            f"predicted={DATASET_ID_TO_LABEL[int(predicted)]} | "
            f"confidence={confidence:.4f} | "
            f"{sample['sentence']}"
        )

    output_text = "\n".join(result_lines)

    print()
    print(output_text)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    OUTPUT_PATH.write_text(
        output_text,
        encoding="utf-8",
    )

    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
