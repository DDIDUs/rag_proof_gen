import json
import numpy as np
from rank_bm25 import BM25Okapi
from src.config import JSONL_PATH
from src.retrieval import retrieve

# ---- Sparse(BM25) 코퍼스 로드 (프로세스 최초 1회) ----
_bm25_tokens, _bm25_meta = [], []

with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        rec = json.loads(line)
        text = f"{rec.get('explanation','')} {rec.get('snippet','')} {rec.get('source_file','')}"
        _bm25_tokens.append(text.split())
        _bm25_meta.append(rec)
        
_bm25 = BM25Okapi(_bm25_tokens)

def _sparse_rank(query: str, k: int):
    scores = _bm25.get_scores(query.split())
    order = np.argsort(scores)[::-1][:k]
    return {int(idx): rank for rank, idx in enumerate(order)}

def search_hybrid(query: str, k_dense=50, k_sparse=50, rrf_c=60, final_n=10):
    # 1) Dense 후보
    dense_hits = retrieve(query, topk=k_dense, mode='dense')
    dense_by_row = {int(h["row_idx"]): rank for rank, h in enumerate(dense_hits) if h["row_idx"] is not None}

    # 2) Sparse 후보 (row_idx = JSONL의 행번호)
    sparse_rank = _sparse_rank(query, k=k_sparse)

    # 3) RRF 결합
    cand_rows = set(dense_by_row) | set(sparse_rank)
    scored = []
    for row in cand_rows:
        rd = dense_by_row.get(row, 10**9)
        rs = sparse_rank.get(row, 10**9)
        s = 1.0/(rd + 1 + rrf_c) + 1.0/(rs + 1 + rrf_c)
        scored.append((s, row))
    scored.sort(reverse=True)

    # 4) 최종 N개 반환 (explanation/snippet/source_file 포함)
    results = []
    for _, row in scored[:final_n]:
        rec = _bm25_meta[row]
        results.append({
            "row_idx": row,
            "explanation": rec.get("explanation",""),
            "snippet": rec.get("snippet",""),
            "source_file": rec.get("source_file",""),
            "type": rec.get("type",""),
            "score": float(rec.get("score", 0.0)),
        })
    return results
