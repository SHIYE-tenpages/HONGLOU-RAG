import re
import json

# ========= Step 1: 读取章节 JSON =========
with open("honglou_cleaned.json", "r", encoding="utf-8") as f:
    chapters = json.load(f)

# ========= Step 2: 句子切分函数 =========
def split_sentences(text):
    # 用中文句号、问号、感叹号切分，保留标点
    sentences = re.split(r"([。！？])", text)
    # 合并标点到句子
    merged = []
    for i in range(0, len(sentences)-1, 2):
        sentence = sentences[i].strip() + sentences[i+1].strip()
        if sentence:
            merged.append(sentence)
    # 处理可能剩下的最后一个无标点部分
    if len(sentences) % 2 != 0 and sentences[-1].strip():
        merged.append(sentences[-1].strip())
    return merged

# ========= Step 3: 拼接成 chunks =========
chunks = []
chunk_size_min = 300
chunk_size_max = 500

for chapter in chapters:
    chapter_id = chapter["chapter_id"]
    title = chapter["title"]
    sentences = split_sentences(chapter["content"])

    chunk_id = 0
    buffer = ""
    for sent in sentences:
        if len(buffer) + len(sent) <= chunk_size_max:
            buffer += sent
        else:
            # 如果 buffer 已经达到阈值，则输出一个 chunk
            if len(buffer) >= chunk_size_min:
                chunk_id += 1
                chunks.append({
                    "chapter_id": chapter_id,
                    "chapter_title": title,
                    "chunk_id": chunk_id,
                    "content": buffer
                })
                buffer = sent
            else:
                # buffer 太短时，直接加上句子避免碎片
                buffer += sent

    # 最后一块
    if buffer:
        chunk_id += 1
        chunks.append({
            "chapter_id": chapter_id,
            "chapter_title": title,
            "chunk_id": chunk_id,
            "content": buffer
        })

# ========= Step 4: 保存结果 =========
with open("honglou_chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print(f"✅ 已完成分片，共生成 {len(chunks)} 个 chunks，保存到 honglou_chunks.json")
