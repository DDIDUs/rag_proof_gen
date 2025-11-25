import json, re
from src.indexing import index_jsonl
from src.retrieval import retrieve
from src.search import search_hybrid
from src.generator import build_proof_prompt_from_examples, LemmaGenerator
from src.config import ANSWER_TOPK

# --------- 유틸 ---------
def _doc_body(h):
    doc = h.get("document", "")
    return doc.split("\n", 1)[1] if "\n" in doc else doc

def _meta(h, key):
    return (h.get("metadata") or {}).get(key)

def _hits_to_examples(hits):
    out = []
    for h in hits:
        explanation = h.get("explanation") or _meta(h, "explanation") or _doc_body(h) or ""
        snippet     = h.get("snippet")     or _meta(h, "snippet")     or ""
        out.append({
            "explanation": explanation,
            "snippet": snippet,
            "source_file": h.get("source_file") or _meta(h, "source_file") or "",
        })
    return out

def _maybe_generate(args, query_input, hits):
    if not args.gen:
        return None
    examples = _hits_to_examples(hits[:args.k])
    prompt = build_proof_prompt_from_examples(query_input, examples, max_examples=args.k)
    gen = LemmaGenerator(backend=args.backend, model=args.model, temperature=args.temp)
    return gen.generate(prompt), prompt

def _iter_test_inputs(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            q = rec.get("input")
            if q:
                yield q, rec.get("gt")

def _save_json(data, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 결과를 {out_path} 에 저장했습니다.")
    
def build_explanation_prompt_for_input(lemma_input: str) -> str:
    header = (
        "Explain about this isabelle/HOL lemma. "
        "Output must be follow the outline format below. DO NOT predict anything\n\n"
        "Output format:\n```\n"
        "Goal that the lemma aims to prove (within 2 lines):\n"
        "...\n"
        "Explanation about proof:\n"
        "...\""
    )

    return f"{header}\n\nLEMMA INPUT:\n{lemma_input}"
    
def _explain_to_query(input_text: str, backend: str, model: str, temp: float) -> str:
    prompt = build_explanation_prompt_for_input(input_text)
    gen = LemmaGenerator(backend=backend, model=model, temperature=temp)
    out = gen.generate(prompt) or ""
    m = re.search(r"```(.*?)```", out, flags=re.S)
    return m.group(1) if m else input_text

# --------- 커맨드 ---------
def cmd_index(args):
    index_jsonl(jsonl_path=args.jsonl) if args.jsonl else index_jsonl()

def cmd_retrieval(args):
    results = []
    if args.test_jsonl:
        for idx, (q, gt) in enumerate(_iter_test_inputs(args.test_jsonl), 1):
            explanation = _explain_to_query(q, args.backend, args.model, args.temp)
            hits = retrieve(explanation, topk=max(args.k, args.topk), mode=args.mode)
            proof, prompt = _maybe_generate(args, q, hits)
            results.append({
                "case": idx,
                "input": q,
                "gt": gt,
                "proof": proof,
                "prompt": prompt,
                "explanation": explanation,
                "hits": hits[:args.topk],
            })
    else:
        explanation = _explain_to_query(args.query, args.backend, args.model, args.temp)
        hits = retrieve(explanation, topk=max(args.k, args.topk), mode=args.mode)
        proof, prompt = _maybe_generate(args, args.query, hits)
        results.append({
            "input": args.query,
            "proof": proof,
            "prompt": prompt,
            "hits": hits[:args.topk],
        })

    if args.out:
        _save_json(results, args.out)
    else:
        print(json.dumps(results, ensure_ascii=False, indent=2))

def cmd_search(args):
    results = []
    if args.test_jsonl:
        for idx, (q, gt) in enumerate(_iter_test_inputs(args.test_jsonl), 1):
            hits = search_hybrid(q, final_n=max(args.k, args.final_n))
            proof, prompt = _maybe_generate(args, q, hits)
            results.append({
                "case": idx,
                "input": q,
                "gt": gt,
                "proof": proof,
                "prompt": prompt,
                "hits": hits[:args.k],
            })
    else:
        hits = search_hybrid(args.query, final_n=max(args.k, args.final_n))
        proof, prompt = _maybe_generate(args, args.query, hits)
        results.append({
            "input": args.query,
            "proof": proof,
            "prompt": prompt,
            "hits": hits[:args.k],
        })

    if args.out:
        _save_json(results, args.out)
    else:
        print(json.dumps(results, ensure_ascii=False, indent=2))