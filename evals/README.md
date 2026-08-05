# Evaluations

- `evaluate_rag_providers.py`: unified 70-case RAG comparison across Ollama and
  OpenRouter. It uses the production pipeline and writes resumable results under
  `output/rag_provider_evaluation/` by default.
- `evaluate_legacy_rag_v31_5classes.py`: repaired reproduction of the historical
  five-class CLTK + LiLa + custom Ollama evaluation. Its timestamped CSV and JSON
  artifacts remain beside the historical artifacts under
  `models/rag_v31_5classes/testing/`.
- `llama31_eval/evaluate.py`: separate non-RAG base `llama3.1:8b` evaluation over
  `data/bert_data/eval2.jsonl`.

Run the two RAG evaluations from the repository root:

```bash
python evals/evaluate_rag_providers.py
python evals/evaluate_legacy_rag_v31_5classes.py
```
