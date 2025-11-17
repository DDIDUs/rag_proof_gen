# RAG proof generator

This script provides a simple pipeline to:
1. **Index** Isabelle/HOL lemma data (`explanation`, `snippet`) into ChromaDB.
2. **Generate explanations** for input lemmas (using an LLM).
3. **Retrieve/Search** similar examples via Dense, BM25.
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

## Backend
```
--backend openai → use OpenAI API (openai_key from api_key.json)
--backend vllm → use a local vLLM server (vllm_url from api_key.json)
```

## Setup
### 1. unzip 'data.tar.gz'
```
tar -zxvf data.tar.gz
```

### 2. Install dependency
```
pip install -r requirements.txt
```

### 3. Build ChromaDB
```
python3 run.py index --jsonl data/isabelle_judge.jsonl
```

## Usage

### 1. Retrieval

#### OpenAI
```python
python3 run.py retrieval \
    --test-jsonl test_data/lemmas_short.jsonl \
    --gen --backend openai \
    --out result/results_openai.json
```

#### VLLM
```python
python3 run.py retrieval \
    --test-jsonl test_data/lemmas_short.jsonl \
    --gen --backend vllm \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \ 
    --out result/results_vllm.json
```

### 2. Search

#### OpenAI
```python
python3 run.py search \
    --test-jsonl test_data/lemmas_short.jsonl \
    --gen --backend openai \
    --out result/results_openai.json
```

#### VLLM
```python
python3 run.py search \
    --test-jsonl test_data/lemmas_short.jsonl \
    --gen --backend vllm \
    --model Qwen/Qwen2.5-Coder-7B-Instruct     \   # 사용 모델 이름
    --out result/results_vllm.json
```

### 3. Evaluation
```python
export PATH="../l4v/isabelle/bin:$PATH"           # isabelle 경로

python3 eval.py \
      --jsonl ../example.json \        # proof 경로
      --thy ../l4v/lib/CorresK/CorresK_Lemmas.thy \    # .thy 파일 경로
      --session CorresK \                           
      --root ../l4v/ \                                 # l4v 레포지토리 경로
      --out ../result/build_report.jsonl      # output 경로
```
