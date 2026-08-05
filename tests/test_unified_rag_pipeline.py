import pytest


@pytest.mark.no_ollama
def test_sentiment_prompt_is_provider_neutral():
    from src.app.server_fast import _build_sentiment_prompt

    priors = '{"LEXICON_PRIORS":{"items":[]}}\n\n'
    prompt = _build_sentiment_prompt("Roma gaudet.", priors)

    assert priors in prompt
    assert "Roma gaudet." in prompt
    assert "ollama" not in prompt.lower()
    assert "openrouter" not in prompt.lower()
    assert "Return exactly one valid label" in prompt
    assert "VERY NEGATIVE" in prompt


@pytest.mark.no_ollama
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("VERY POSITIVE (+1)", "positive"),
        ("SOMEWHAT NEGATIVE", "negative"),
        ("NEUTRAL (0)", "neutral"),
    ],
)
def test_five_level_labels_collapse_to_api_contract(raw, expected):
    from src.app.server_fast import _parse_sentiment_label

    assert _parse_sentiment_label(raw) == expected


@pytest.mark.no_ollama
@pytest.mark.parametrize("engine", ["ollama", "openrouter"])
def test_sentiment_response_contract_matches_across_providers(engine):
    from src.app.server_fast import _normalize_sentiment_response

    parsed = {
        "label": "positive",
        "confidence": "0.8",
        "scores": {"positive": 0.8, "negative": 0.1, "neutral": 0.1},
        "translation": "Rome rejoices.",
        "analysis": {"reason": "gaudet"},
    }
    result = _normalize_sentiment_response(
        parsed,
        "raw",
        engine=engine,
        priors_json="{\"LEXICON_PRIORS\":{}}",
    )

    assert result["engine"] == engine
    assert result["rag"] == {
        "enabled": False,
        "requested": True,
        "source": None,
        "hit_count": 0,
    }
    assert result["lexicon_priors_included"] is True
    assert result["label"] == "positive"
    assert result["confidence"] == 0.8
    assert set(result["scores"]) == {"positive", "negative", "neutral"}


@pytest.mark.no_ollama
def test_response_normalizer_preserves_zero_scores():
    from src.app.server_fast import _normalize_sentiment_response

    result = _normalize_sentiment_response(
        {"label": "negative", "scores": {"positive": 0, "negative": 1, "neutral": 0}},
        "raw",
        engine="ollama",
        priors_json="",
    )

    assert result["rag"] == {
        "enabled": False,
        "requested": False,
        "source": None,
        "hit_count": 0,
    }
    assert result["scores"] == {"positive": 0.0, "negative": 1.0, "neutral": 0.0}
