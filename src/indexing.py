import json, uuid
import chromadb
from chromadb.utils import embedding_functions
from src.config import JSONL_PATH, PERSIST_DIR, COLLECTION, EMBED_MODEL

def build_content(rec):
    # explanation 중심 + 최소 컨텍스트(prefix)
    return f"[type={rec.get('type','')}] file={rec.get('source_file','')}\n{rec.get('explanation','')}"

def index_jsonl(jsonl_path=JSONL_PATH):
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    col = client.get_or_create_collection(name=COLLECTION, embedding_function=emb_fn)

    ids, docs, metadatas = [], [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for row_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            content = build_content(rec)
            meta = {
                "row_idx": row_idx,  # 하이브리드 매핑용
                "snippet": rec.get("snippet",""),
                "source_file": rec.get("source_file",""),
                "type": rec.get("type",""),
                "score": float(rec.get("score", 0.0)),
            }

            ids.append(str(uuid.uuid4()))
            docs.append(content)
            metadatas.append(meta)

    # 배치 업서트
    BATCH = 2048
    for i in range(0, len(ids), BATCH):
        col.add(ids=ids[i:i+BATCH], documents=docs[i:i+BATCH], metadatas=metadatas[i:i+BATCH])

    print(f"[indexing] Indexed {len(ids)} docs -> '{COLLECTION}' ({PERSIST_DIR})")