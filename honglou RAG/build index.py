# build_index.py
import json, os
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from tqdm import tqdm

# 配置
CHUNKS_PATH = "E:/AI advanced learning/RAG/honglou_chunks.json"   # 你分好段的文件
PERSONS_PATH = "E:/AI advanced learning/RAG/person.json"             # 你的人物关系文件
EMBED_MODEL = "BAAI/bge-base-zh-v1.5"     # 可替换为 "moka-ai/m3e-base" 等
INDEX_PATH = "hongloumeng.index"
META_PATH = "hongloumeng_meta.json"
PERSON_KB_PATH = "person_kb.json"         # 保存人物小知识库

# 1. 加载数据
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

persons = []
if os.path.exists(PERSONS_PATH):
    with open(PERSONS_PATH, "r", encoding="utf-8") as f:
        persons = json.load(f)
else:
    print("Warning: persons.json not found. Continuing without person KB.")

# 2. 加载 embedding 模型
print("加载 embedding 模型:", EMBED_MODEL)
embedder = SentenceTransformer(EMBED_MODEL)

# 3. 为 chunks 生成 embedding
texts = [c["content"] for c in chunks]
batch_size = 64
embs = []
for i in tqdm(range(0, len(texts), batch_size), desc="Embedding chunks"):
    batch = texts[i:i+batch_size]
    emb = embedder.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
    embs.append(emb)
embeddings = np.vstack(embs).astype("float32")
print("embeddings shape:", embeddings.shape)

# 4. 建 FAISS 索引（用 Inner Product，因为向量已归一化 => 等价余弦）
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)
faiss.write_index(index, INDEX_PATH)
print("Saved faiss index to", INDEX_PATH)

# 5. 保存 metadata（chunks + optionally person KB）
# we will add a pointer to persons KB (by name) inside metadata if relevant
meta = []
for i, c in enumerate(chunks):
    item = dict(c)  # copy
    item["_index"] = i
    meta.append(item)

with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print("Saved metadata to", META_PATH)

# 6. 将人物关系转换为简短可检索条目（person KB）
person_kb = []
for p in persons:
    # 构造描述文本（合并 description + relationships）
    rels = p.get("relationships", [])
    rel_text = "；".join([f"{r['relation']}：{r['name']}" for r in rels])
    full_text = f"{p.get('name','')}：{p.get('description','')}"
    if rel_text:
        full_text += " 关系：" + rel_text
    person_kb.append({
        "name": p.get("name"),
        "text": full_text
    })

with open(PERSON_KB_PATH, "w", encoding="utf-8") as f:
    json.dump(person_kb, f, ensure_ascii=False, indent=2)
print("Saved person KB to", PERSON_KB_PATH)

print("Index build finished. chunks:", len(chunks), "persons:", len(person_kb))
