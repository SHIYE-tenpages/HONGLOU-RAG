import json, os, time
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ============= 配置 =============
HF_MODEL_NAME = "Qwen/Qwen2-7B-Instruct"   
INDEX_PATH = "hongloumeng.index"
META_PATH = "hongloumeng_meta.json"
PERSON_KB_PATH = "person_kb.json"
COMMENT_PATH = "comment.json"   # 可选，红学评论文件 待补充
EMBED_MODEL = "BAAI/bge-base-zh-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"  # cross-encoder rerank

#加载模型
print("加载 embedding model:", EMBED_MODEL)
embedder = SentenceTransformer(EMBED_MODEL)

print("加载 faiss index & metadata...")
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

person_kb, comments = [], []
if os.path.exists(PERSON_KB_PATH):
    with open(PERSON_KB_PATH, "r", encoding="utf-8") as f:
        person_kb = json.load(f)
if os.path.exists(COMMENT_PATH):
    with open(COMMENT_PATH, "r", encoding="utf-8") as f:
        comments = json.load(f)

# reranker
print("加载 reranker:", RERANK_MODEL)
reranker = CrossEncoder(RERANK_MODEL)

# HF 生成模型
print("加载 HF 生成模型:", HF_MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(HF_MODEL_NAME, trust_remote_code=True)
device = 0 if model.device.type == "cuda" else -1
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

#keyword fallback 
def keyword_search(query, top_k=5):
    hits = []
    q_lower = query.lower()
    for i, m in enumerate(meta):
        text = (m.get("chapter_title", "") + " " + m.get("content", "")).lower()
        if any(k.lower() in text for k in query.split()):
            hits.append((i, m))
        elif q_lower in text:
            hits.append((i, m))
    seen, res = set(), []
    for idx, item in hits:
        if idx not in seen:
            res.append(item)
            seen.add(idx)
        if len(res) >= top_k:
            break
    return res

#检索函数（多通道+加权融合）
def retrieve(query, top_k=8, n_probe=50, use_keyword_fallback=True):
    qvec = embedder.encode([query], normalize_embeddings=True).astype("float32")
    D, I = index.search(qvec, n_probe)
    candidates = []
    for j, i in enumerate(I[0]):
        if i >= 0:
            candidates.append({"meta": meta[i], "score": float(D[0][j]), "source": "原文"})

    # keyword fallback
    if use_keyword_fallback and len(candidates) < top_k:
        extra_hits = keyword_search(query, top_k=top_k)
        for i, m in extra_hits:
            candidates.append({"meta": m, "score": 0.2, "source": "原文"})

    # 人物 KB
    for p in person_kb:
        if p["name"] in query:
            candidates.append({"meta": {"chapter_title": f"人物：{p['name']}", "content": p["text"]}, "score": 0.5, "source": "人物图谱"})

    # 评论 KB
    for c in comments:
        if any(w in query for w in c.get("tags", [])):
            candidates.append({"meta": {"chapter_title": f"评论", "content": c["text"]}, "score": 0.3, "source": "评论"})

    # rerank 二次排序（cross-encoder）
    pairs = [(query, c["meta"]["content"]) for c in candidates]
    scores = reranker.predict(pairs)
    for i, s in enumerate(scores):
        candidates[i]["score"] = float(s)

    # 按 score 排序 + 去重
    candidates = sorted(candidates, key=lambda x: -x["score"])
    seen, final = set(), []
    for c in candidates:
        t = c["meta"]["content"][:80]
        if t not in seen:
            final.append(c)
            seen.add(t)
        if len(final) >= top_k:
            break
    return final

#Prompt 构建
def build_prompt(question, retrieved):
    header = (
        "你是《红楼梦》研究助手。\n"
        "以下是检索到的片段，每个片段标注了编号和来源。\n"
        "请严格遵守以下要求：\n"
        "1. 回答必须基于片段内容。\n"
        "2. 每个回答后面必须注明引用的片段编号和来源。\n"
        "3. 不要编造信息。\n"
        "格式：\n"
        "答案：...\n引用：片段编号-来源\n"
    )
    ctxs = []
    for i, c in enumerate(retrieved, 1):
        ctxs.append(f"【片段{i} | 来源：{c['source']}】 章节：{c['meta'].get('chapter_title','')} \n{c['meta'].get('content','')}")
    ctx_text = "\n\n---\n\n".join(ctxs)
    prompt = header + "\n检索片段：\n" + ctx_text + f"\n\n用户问题：{question}\n请回答："
    return prompt

#生成答案
def generate_hf(prompt, max_new_tokens=256):
    out = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    return out[0]["generated_text"]

#主接口
def answer_question(question, top_k=6):
    t0 = time.time()
    retrieved = retrieve(question, top_k=top_k)
    prompt = build_prompt(question, retrieved)
    ans = generate_hf(prompt)
    t1 = time.time()
    return {
        "question": question,
        "retrieved": retrieved,
        "prompt": prompt,
        "answer": ans,
        "time": t1-t0
    }

#CLI 测试
if __name__ == "__main__":
    q = "刘姥姥首次在那个章节中登场？"
    res = answer_question(q, top_k=6)
    print("Answer:\n", res["answer"])
    print("\nRetrieved片段：")
    for i, r in enumerate(res["retrieved"], 1):
        print(f"片段{i} (score={r['score']:.4f}, 来源={r['source']}) -> {r['meta']['chapter_title']}")
        print(r['meta']['content'][:200].replace("\n"," "))
        print("-"*60)
