#!/usr/bin/env python3
"""Run the repaired legacy five-class CLTK + LiLa + Ollama RAG evaluation."""

from __future__ import annotations

import os
import runpy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_EVALUATOR = (
    PROJECT_ROOT / "models/rag_v31_5classes/testing/test.py"
)


def main() -> None:
    os.environ.setdefault("CLTK_DATA", str(PROJECT_ROOT / "cltk_data"))
    runpy.run_path(str(LEGACY_EVALUATOR), run_name="__main__")


if __name__ == "__main__":
    main()
