"""
Dataset loaders + BM25 retriever for Paper B.

HotpotQA (distractor) — multi-hop QA, bundled context paragraphs, no Wikipedia needed.
MuSiQue             — harder multi-hop QA, also bundled context, no download needed.

Each sample: {id, question, answer, passages: list[str]}
passages are the context paragraphs used to build the per-question BM25 index.
"""

import json
import re
from pathlib import Path
from datasets import load_dataset
from rank_bm25 import BM25Okapi


# ---------------------------------------------------------------------------
# HotpotQA
# ---------------------------------------------------------------------------

def load_hotpotqa(n: int = 300, split: str = "validation") -> list[dict]:
    """
    Load n HotpotQA distractor questions.
    Each 'context' field contains ~10 paragraphs (titles + sentences).
    We flatten them into passage strings for BM25.
    """
    ds = load_dataset("hotpot_qa", "distractor", split=split, trust_remote_code=True)
    samples = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        # Flatten context: list of [title, sentences] pairs
        passages = []
        for title, sents in zip(row["context"]["title"], row["context"]["sentences"]):
            para = f"{title}: " + " ".join(sents)
            passages.append(para)

        samples.append({
            "id":       f"hotpot_{i}",
            "question": row["question"],
            "answer":   row["answer"],
            "passages": passages,
            "dataset":  "hotpotqa",
        })
    print(f"[datasets] HotpotQA: {len(samples)} samples loaded")
    return samples


# ---------------------------------------------------------------------------
# MuSiQue
# ---------------------------------------------------------------------------

def load_musique(n: int = 200, split: str = "validation") -> list[dict]:
    """
    Load n MuSiQue questions.
    Paragraphs are in 'paragraphs' field: list of {idx, title, paragraph_text, is_supporting}.
    """
    try:
        ds = load_dataset("dgslibisey/musique", split=split, trust_remote_code=True)
    except Exception:
        # Fallback: try the allenai version
        ds = load_dataset("allenai/musique", split=split, trust_remote_code=True)

    samples = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        passages = [
            f"{p['title']}: {p['paragraph_text']}"
            for p in row.get("paragraphs", [])
        ]
        samples.append({
            "id":       f"musique_{i}",
            "question": row["question"],
            "answer":   row["answer"],
            "passages": passages,
            "dataset":  "musique",
        })
    print(f"[datasets] MuSiQue: {len(samples)} samples loaded")
    return samples


# ---------------------------------------------------------------------------
# BM25 Retriever
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def build_bm25_index(passages: list[str]) -> BM25Okapi:
    """Build a BM25 index from a list of passage strings."""
    tokenized = [_tokenize(p) for p in passages]
    return BM25Okapi(tokenized)


def retrieve(query: str, index: BM25Okapi, passages: list[str], top_k: int = 2) -> str:
    """
    Retrieve top_k passages for a query. Returns them concatenated.
    Used as the "Search" tool in the ReAct loop.
    """
    query_tokens = _tokenize(query)
    scores = index.get_scores(query_tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    retrieved = [passages[i] for i in top_indices]
    return "\n\n".join(retrieved)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def save_samples(samples: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"[datasets] Saved {len(samples)} samples → {path}")


def load_samples(path: str | Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)
