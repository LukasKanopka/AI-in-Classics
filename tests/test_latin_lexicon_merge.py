import pytest

from src.app.latin_lexicon import merge_sentiment_hits


pytestmark = pytest.mark.no_ollama


def test_merge_sentiment_hits_deduplicates_and_counts_actual_hits():
    priors = {
        "LEXICON_PRIORS": {
            "hits": [{"lemma": "malus", "score": -1.0, "count": 1}]
        }
    }
    merged = merge_sentiment_hits(
        priors,
        [
            {"lemma": "malus", "score": -1.0, "count": 1},
            {"lemma": "puer", "score": 0.0, "count": 1},
        ],
    )

    envelope = merged["LEXICON_PRIORS"]
    assert envelope["actual_hit_count"] == 2
    assert envelope["retrieval"] == "central+cltk-direct-lila"
    assert [hit["lemma"] for hit in envelope["hits"]] == ["malus", "puer"]
