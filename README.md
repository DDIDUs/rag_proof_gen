# RAG proof generator

This script provides a simple pipeline to:
1. **Index** Isabelle/HOL lemma data (`explanation`, `snippet`) into ChromaDB.
2. **Generate explanations** for input lemmas (using an LLM).
3. **Retrieve/Search** similar examples via Dense, BM25, or Hybrid RAG.
4. **Save results as JSON** for evaluation or further processing.

---

## Requirements
- Python 3.9+
- Prepare an `api_key.json` file for API credentials:

```json
{
  "openai_key": "sk-xxxxxx",
  "vllm_url": "http://localhost:8000/v1"
}
```

##Setup
1. unzip 'data.tar.gz'
```
tar -zxvf data.tar.gz
```

2. Install dependency
```
pip install -r requirements.txt
```

3. Build ChromaDB
```
python3 run.py --jsonl data/isabelle_judge.jsonl
```

##Backend

--backend openai → use OpenAI API (openai_key from api_key.json)
--backend vllm → use a local vLLM server (vllm_url from api_key.json)

##Usage

1. Retrieval

###OpenAI
```python
python3 run.py retrieval \
    --test-jsonl test_data/lemmas_short.jsonl \
    --gen --backend openai \
    --out result/results_openai.json
```

###VLLM
```python
python3 run.py retrieval \
    --test-jsonl test_data/lemmas_short.jsonl \
    --gen --backend vllm \
    --out result/results_vllm.json
```