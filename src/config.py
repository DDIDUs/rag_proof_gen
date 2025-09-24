import json, os

JSONL_PATH = "./data/isabelle_judge.jsonl"
PERSIST_DIR = "chroma_store"
COLLECTION = "rag_collection"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DENSE_TOPK = 5
ANSWER_TOPK = 5

API_KEY_FILE = os.getenv("API_KEY_FILE", "api_key.json")

openai_key = None
vllm_url = None

if os.path.exists(API_KEY_FILE):
    with open(API_KEY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        openai_key = data.get("openai_key")
        vllm_url = data.get("vllm_url")
