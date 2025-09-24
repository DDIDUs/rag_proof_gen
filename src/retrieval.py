# retrieval.py
import json
import numpy as np
import chromadb
from rank_bm25 import BM25Okapi
from chromadb.utils import embedding_functions
from src.config import (
    PERSIST_DIR, COLLECTION, EMBED_MODEL, DENSE_TOPK,
    JSONL_PATH  # config에 없으면 추가하세요.
)

# ---------- Dense (cosine) ----------
def _get_collection():
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return client.get_or_create_collection(name=COLLECTION, embedding_function=emb_fn)

# ---------- BM25 (lazy load, 프로세스 1회) ----------
_BM25 = None
_BM25_META = None

def _tokenize(s: str):
    # 필요시 형태소 분석기로 교체
    return s.split()

def _ensure_bm25_loaded(jsonl_path: str = JSONL_PATH):
    global _BM25, _BM25_META
    if _BM25 is not None:
        return
    tokens, meta = [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for row_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = f"{rec.get('explanation','')} {rec.get('snippet','')} {rec.get('source_file','')}"
            tokens.append(_tokenize(text))
            rec["row_idx"] = row_idx
            meta.append(rec)
    _BM25 = BM25Okapi(tokens)
    _BM25_META = meta

# ---------- Unified API ----------
def retrieve(query: str, topk: int = DENSE_TOPK, mode: str = "dense"):
    """
    mode: "dense" | "bm25"
    반환 형식 동일(id, row_idx, document, metadata, score)
    """
    if mode == "dense":
        col = _get_collection()
        res = col.query(
            query_texts=[query],
            n_results=topk,
            include=["metadatas", "documents", "distances"]
        )
        out = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for i, _id in enumerate(ids):
            dist = dists[i]
            score = 1.0 / (1.0 + dist)  # 코사인 거리 → 간단 스코어
            out.append({
                "id": _id,
                "row_idx": metas[i].get("row_idx"),
                "document": docs[i],
                "metadata": metas[i],
                "score": float(score),
                "mode": "dense"
            })
        return out

    elif mode == "bm25":
        _ensure_bm25_loaded()
        scores = _BM25.get_scores(_tokenize(query))
        order = np.argsort(scores)[::-1][:topk]
        out = []
        for idx in order:
            r = _BM25_META[idx]
            doc = f"[type={r.get('type','')}] file={r.get('source_file','')}\n{r.get('explanation','')}"
            out.append({
                "id": f"row-{idx}",
                "row_idx": r["row_idx"],
                "document": doc,  # explanation 중심
                "metadata": {
                    "snippet": r.get("snippet",""),
                    "source_file": r.get("source_file",""),
                    "type": r.get("type",""),
                    "score_meta": float(r.get("score", 0.0)),
                },
                "score": float(scores[idx]),
                "mode": "bm25"
            })
        return out

    else:
        raise ValueError("mode must be 'dense' or 'bm25'")
