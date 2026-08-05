from __future__ import annotations

import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional, Any


BASE_DIR = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def latin_lexicon_import_root() -> Optional[Path]:
    root = BASE_DIR / "src" / "Lemmatizer-LTN-LiLa"
    return root if root.exists() else None


def resolve_database_url() -> str:
    """
    Resolve DATABASE_URL in a Streamlit-free way.

    FastAPI typically relies on environment variables; we also best-effort load
    `.env` from cwd or repo root for local dev ergonomics.
    """
    try:
        from dotenv import load_dotenv  # type: ignore

        for env_path in (Path.cwd() / ".env", BASE_DIR / ".env"):
            if env_path.exists():
                load_dotenv(env_path)
                break
    except Exception:
        pass
    return (os.getenv("DATABASE_URL") or "").strip()


def make_latin_lexicon_annotator(dsn: str) -> Any:
    lex_root = latin_lexicon_import_root()
    if lex_root is None:
        raise RuntimeError(
            "Missing src/Lemmatizer-LTN-LiLa; cannot import LatinLexiconAnnotator."
        )
    if str(lex_root) not in sys.path:
        sys.path.insert(0, str(lex_root))

    from rag.latin_lexicon_annotator import (  # type: ignore
        LatinLexiconAnnotator,
        LatinLexiconAnnotatorConfig,
    )

    return LatinLexiconAnnotator(
        dsn=dsn,
        config=LatinLexiconAnnotatorConfig(top_k=12),
    )


@lru_cache(maxsize=1)
def _legacy_latin_lemmatizer() -> Any:
    """Load the CLTK lemmatizer once for the direct LiLa fallback."""
    os.environ.setdefault("CLTK_DATA", str(BASE_DIR / "cltk_data"))
    from cltk.lemmatize.lat import LatinBackoffLemmatizer

    return LatinBackoffLemmatizer()


def direct_lila_sentiment_hits(text: str, dsn: str, *, limit: int = 30) -> list[dict[str, Any]]:
    """Reproduce the high-coverage CLTK -> lila.sentiment lookup used by legacy evals."""
    import psycopg

    tokens = re.findall(r"[A-Za-z]+", text or "")
    pairs = _legacy_latin_lemmatizer().lemmatize([token.lower() for token in tokens])
    lemmas = sorted(
        {
            re.sub(r"\d+$", "", str(lemma or "").strip().lower())
            for _, lemma in pairs
            if lemma
        }
    )
    if not lemmas:
        return []

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT lemma, pos, polarity_score, has_polarity, provenance
            FROM lila.sentiment
            WHERE lemma = ANY(%s)
            """,
            (lemmas,),
        )
        rows = cur.fetchall()

    hits = [
        {
            "lemma": re.sub(r"\d+$", "", str(lemma or "").strip().lower()),
            "pos": pos,
            "score": float(score),
            "count": 1,
            "source": "cltk+lila.sentiment",
            "has_polarity": has_polarity,
            "provenance": provenance,
        }
        for lemma, pos, score, has_polarity, provenance in rows
        if score is not None
    ]
    hits.sort(key=lambda hit: (-abs(hit["score"]), hit["lemma"]))
    return hits[:limit]


def form_lookup_sentiment_hits(text: str, dsn: str, *, limit: int = 30) -> list[dict[str, Any]]:
    """Join every surface-form lemma candidate to LiLa instead of choosing one early."""
    import psycopg

    forms = sorted({token.lower() for token in re.findall(r"[A-Za-z]+", text or "")})
    if not forms:
        return []
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT COALESCE(l.lemma_nod::text, norm(s.lemma)),
                            s.pos, s.polarity_score,
                            s.has_polarity, s.provenance
            FROM public.word_lookup AS w
            LEFT JOIN public.lemmas AS l ON l.id = w.lemma_id
            JOIN lila.sentiment AS s
              ON s.id = w.sentiment_lemma_id
              OR (w.lemma_id IS NOT NULL AND norm(s.lemma) = l.lemma_nod)
            WHERE LOWER(w.form_nod) = ANY(%s)
              AND s.polarity_score IS NOT NULL
            """,
            (forms,),
        )
        rows = cur.fetchall()
    hits = [
        {
            "lemma": str(lemma or "").strip().lower(),
            "pos": pos,
            "score": float(score),
            "count": 1,
            "source": "word_lookup+lila.sentiment",
            "has_polarity": has_polarity,
            "provenance": provenance,
        }
        for lemma, pos, score, has_polarity, provenance in rows
    ]
    hits.sort(key=lambda hit: (-abs(hit["score"]), hit["lemma"]))
    return hits[:limit]


def merge_sentiment_hits(priors: dict[str, Any], extra_hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge direct LiLa hits into the compact centralized lexicon payload."""
    envelope = priors.setdefault("LEXICON_PRIORS", {})
    existing = envelope.setdefault("hits", [])
    seen = {
        (str(hit.get("lemma") or ""), float(hit.get("score") or 0.0))
        for hit in existing
        if isinstance(hit, dict)
    }
    for hit in extra_hits:
        key = (str(hit.get("lemma") or ""), float(hit.get("score") or 0.0))
        if key not in seen:
            existing.append(hit)
            seen.add(key)
    envelope["actual_hit_count"] = len(existing)
    envelope["retrieval"] = "central+cltk-direct-lila"
    return priors
