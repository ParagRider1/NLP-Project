"""
repetition_detector.py
-----------------------
Detects semantically repetitive paragraphs within a document using
sentence-transformer embeddings + cosine similarity.

Improvements over v1:
  - Group clustering: paragraph {1,3,5} all repeat the same idea shown together.
  - Sentence-level overlap: highlights which sentences drive the repetition.
  - Smarter paragraph splitting: handles bullet lists, numbered lists, etc.

Usage:
    from src.repetition_detector import RepetitionDetector
    detector = RepetitionDetector(sbert_model)
    results = detector.analyze(document_text, threshold=0.75)
"""

import re
import numpy as np
from itertools import combinations


# ─────────────────────────────────────────────────────────────────────────────
# Paragraph splitting
# ─────────────────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list:
    """Lightweight sentence splitter (no NLTK dependency)."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", text)
    return [s.strip() for s in parts if s.strip()]


def split_into_paragraphs(text: str) -> list:
    """
    Split a document into meaningful paragraphs.

    Rules:
      - Primary split on one or more blank lines.
      - Consecutive bullet/numbered list items are merged into one paragraph.
      - Paragraphs shorter than 40 characters are skipped (headings, labels, etc.).

    Returns a list of dicts:
        index     - 1-based paragraph number
        text      - full cleaned paragraph text
        preview   - first 120 chars for display
        sentences - list of individual sentences in the paragraph
    """
    raw_blocks = re.split(r"\n\s*\n", text.strip())

    # Merge adjacent list-item blocks into single paragraph
    list_item_re = re.compile(r"^\s*(?:[-*]|\d+[.):])\s+")
    merged = []
    current_list = []

    for block in raw_blocks:
        cleaned = " ".join(block.split())
        if not cleaned:
            continue
        first_line = block.strip().split("\n")[0]
        if list_item_re.match(first_line):
            current_list.append(cleaned)
        else:
            if current_list:
                merged.append(" ".join(current_list))
                current_list = []
            merged.append(cleaned)

    if current_list:
        merged.append(" ".join(current_list))

    paragraphs = []
    for p in merged:
        if len(p) < 40:
            continue
        paragraphs.append(
            {
                "index":     len(paragraphs) + 1,
                "text":      p,
                "preview":   p[:120] + ("..." if len(p) > 120 else ""),
                "sentences": _split_sentences(p),
            }
        )
    return paragraphs


# ─────────────────────────────────────────────────────────────────────────────
# Embedding helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_bi_encoder(model):
    """Return a bi-encoder regardless of whether model is cross-encoder."""
    if model.is_cross_encoder:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    return model.model


def embed_texts(model, texts: list) -> np.ndarray:
    encoder = _get_bi_encoder(model)
    return np.array(
        encoder.encode(texts, convert_to_tensor=False, show_progress_bar=False)
    )


def cosine_sim_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    normed = embeddings / norms
    return normed @ normed.T


# ─────────────────────────────────────────────────────────────────────────────
# Group clustering (Union-Find)
# ─────────────────────────────────────────────────────────────────────────────

def _cluster_pairs(pairs: list) -> list:
    """
    Union-Find clustering: merge pairs that share a paragraph index.
    E.g. pairs {(1,3), (3,5)} -> group {1,3,5}.
    Returns list of sets (each set = a group of related paragraph indices).
    """
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b, _ in pairs:
        union(a, b)

    groups = {}
    for a, b, _ in pairs:
        root = find(a)
        groups.setdefault(root, set())
        groups[root].add(a)
        groups[root].add(b)

    return list(groups.values())


# ─────────────────────────────────────────────────────────────────────────────
# Sentence-level overlap
# ─────────────────────────────────────────────────────────────────────────────

def _sentence_overlap(model, para_a: dict, para_b: dict, threshold: float = 0.70) -> list:
    """
    For two repetitive paragraphs, find which sentence pairs overlap most.
    Returns list of (sentence_from_A, sentence_from_B, score) sorted by score desc.
    """
    sents_a = para_a["sentences"]
    sents_b = para_b["sentences"]

    if not sents_a or not sents_b:
        return []

    all_sents = sents_a + sents_b
    embs = embed_texts(model, all_sents)
    na = len(sents_a)
    embs_a = embs[:na]
    embs_b = embs[na:]

    norms_a = np.linalg.norm(embs_a, axis=1, keepdims=True)
    norms_b = np.linalg.norm(embs_b, axis=1, keepdims=True)
    normed_a = embs_a / np.where(norms_a == 0, 1e-9, norms_a)
    normed_b = embs_b / np.where(norms_b == 0, 1e-9, norms_b)
    sim = normed_a @ normed_b.T  # (na, nb)

    overlaps = []
    nb = len(sents_b)
    for i in range(na):
        for j in range(nb):
            s = float(sim[i, j])
            if s >= threshold:
                overlaps.append((sents_a[i], sents_b[j], s))

    overlaps.sort(key=lambda x: -x[2])
    return overlaps


# ─────────────────────────────────────────────────────────────────────────────
# Main detector class
# ─────────────────────────────────────────────────────────────────────────────

class RepetitionDetector:
    """
    Detects semantically repetitive paragraph pairs/groups in a document.
    """

    def __init__(self, sbert_model):
        self.model = sbert_model

    def analyze(self, document_text: str, threshold: float = 0.75) -> dict:
        """
        Parameters
        ----------
        document_text : str
            Full document with paragraphs separated by blank lines.
        threshold : float
            Cosine-similarity threshold above which two paragraphs are
            considered repetitive (default 0.75).

        Returns
        -------
        dict with keys:
            paragraphs        - list of paragraph dicts
            sim_matrix        - N x N numpy array of similarities
            repetitive_pairs  - list of (i, j, score) tuples (1-based), desc order
            groups            - list of sets, each = cluster of repetitive para indices
            sentence_overlaps - dict keyed by (i,j) -> list of (sent_a, sent_b, score)
            summary           - human-readable summary string
        """
        paragraphs = split_into_paragraphs(document_text)
        n = len(paragraphs)

        if n < 2:
            return {
                "paragraphs":        paragraphs,
                "sim_matrix":        np.array([[1.0]]) if n == 1 else np.array([[]]),
                "repetitive_pairs":  [],
                "groups":            [],
                "sentence_overlaps": {},
                "summary":           "Not enough paragraphs to compare (need at least 2).",
            }

        # ── Paragraph-level embeddings ──────────────────────────────────────
        para_texts = [p["text"] for p in paragraphs]
        embeddings = embed_texts(self.model, para_texts)
        sim_matrix = cosine_sim_matrix(embeddings)

        # ── Collect repetitive pairs (upper triangle, no diagonal) ──────────
        repetitive_pairs = []
        for i, j in combinations(range(n), 2):
            score = float(sim_matrix[i, j])
            if score >= threshold:
                repetitive_pairs.append((i + 1, j + 1, score))  # 1-based

        repetitive_pairs.sort(key=lambda x: -x[2])

        # ── Cluster pairs into groups ────────────────────────────────────────
        groups = _cluster_pairs(repetitive_pairs) if repetitive_pairs else []

        # ── Sentence-level overlaps for every flagged pair ───────────────────
        para_map = {p["index"]: p for p in paragraphs}
        sentence_overlaps = {}
        for a, b, _ in repetitive_pairs:
            overlaps = _sentence_overlap(self.model, para_map[a], para_map[b])
            if overlaps:
                sentence_overlaps[(a, b)] = overlaps

        # ── Human-readable summary ───────────────────────────────────────────
        if repetitive_pairs:
            lines = [
                f"Found {len(repetitive_pairs)} repetitive paragraph pair(s) "
                f"across {len(groups)} group(s):"
            ]
            for rank, (a, b, s) in enumerate(repetitive_pairs, 1):
                severity = (
                    "near-duplicate" if s >= 0.90 else
                    "high overlap"   if s >= 0.80 else
                    "moderate overlap"
                )
                lines.append(
                    f"  #{rank}  Paragraph {a} <-> Paragraph {b}  "
                    f"(similarity {s:.3f}, {severity})"
                )
            if groups:
                lines.append("")
                lines.append("Repetition groups:")
                for g in sorted(groups, key=lambda s: min(s)):
                    sorted_g = sorted(g)
                    lines.append(
                        f"  * Paragraphs {', '.join(str(x) for x in sorted_g)} "
                        f"all express the same idea."
                    )
            summary = "\n".join(lines)
        else:
            summary = "No repetitive paragraphs detected above the threshold."

        return {
            "paragraphs":        paragraphs,
            "sim_matrix":        sim_matrix,
            "repetitive_pairs":  repetitive_pairs,
            "groups":            groups,
            "sentence_overlaps": sentence_overlaps,
            "summary":           summary,
        }