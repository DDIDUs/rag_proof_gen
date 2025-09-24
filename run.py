# main.py
import argparse, sys
from command import *

def main():
    ap = argparse.ArgumentParser(description="RAG (Dense/BM25/Hybrid) with proof generation → JSON output")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # 색인
    p = sub.add_parser("index", help="JSONL 색인 → Chroma")
    p.add_argument("--jsonl", default=None)
    p.set_defaults(func=cmd_index)

    # retrieval
    p = sub.add_parser("retrieval", help="검색(dense|bm25) → JSON")
    p.add_argument("query", nargs="?", help="--test-jsonl 없을 때만 필요")
    p.add_argument("--mode", choices=["dense", "bm25"], default="dense")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--test-jsonl", default=None, help="테스트 파일(JSONL; {input, gt})")
    p.add_argument("--out", default=None, help="저장할 JSON 경로")
    # 생성 옵션
    p.add_argument("--gen", action="store_true")
    p.add_argument("--k", type=int, default=ANSWER_TOPK)
    p.add_argument("--backend", choices=["openai", "vllm"], default="vllm")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--temp", type=float, default=0.1)
    p.set_defaults(func=cmd_retrieval)

    # search
    p = sub.add_parser("search", help="Hybrid 검색 → JSON")
    p.add_argument("query", nargs="?", help="--test-jsonl 없을 때만 필요")
    p.add_argument("--final_n", type=int, default=10)
    p.add_argument("--test-jsonl", default=None)
    p.add_argument("--out", default=None)
    # 생성 옵션
    p.add_argument("--gen", action="store_true")
    p.add_argument("--k", type=int, default=ANSWER_TOPK)
    p.add_argument("--backend", choices=["openai", "vllm"], default="vllm")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--temp", type=float, default=0.1)
    p.set_defaults(func=cmd_search)

    args = ap.parse_args()
    if getattr(args, "cmd", None) in ("retrieval", "search"):
        has_test_jsonl = getattr(args, "test_jsonl", None)
        has_query = getattr(args, "query", None)
        if not has_test_jsonl and has_query is None:
            ap.error("query가 필요합니다(또는 --test-jsonl).")
    args.func(args)

if __name__ == "__main__":
    main()


