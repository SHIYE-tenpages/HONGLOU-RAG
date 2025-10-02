import jieba
import json

# ========= Step 1: 加载词典 =========
jieba.load_userdict("E:/AI新技术学习/RAG/红楼梦数据集/vocabularys.txt")

# 停用词表（GB2312 编码）
with open("DATA/stopword.txt", "r", encoding="gb2312", errors="ignore") as f:
    stopwords = set([line.strip() for line in f if line.strip()])

# ========= Step 2: 读取 chunks =========
with open("honglou_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# ========= Step 3: 分词 + 去停用词 =========
tokenized_chunks = []
for chunk in chunks:
    words = jieba.lcut(chunk["content"])
    words = [w for w in words if w not in stopwords and w.strip()]
    
    tokenized_chunks.append({
        "chapter_id": chunk["chapter_id"],
        "chapter_title": chunk["chapter_title"],
        "chunk_id": chunk["chunk_id"],
        "tokens": words
    })

# ========= Step 4: 保存 =========
with open("honglou_chunks_tokenized.json", "w", encoding="utf-8") as f:
    json.dump(tokenized_chunks, f, ensure_ascii=False, indent=2)

print(f"✅ 已完成分词增强，共 {len(tokenized_chunks)} 个 chunks，保存到 honglou_chunks_tokenized.json")
