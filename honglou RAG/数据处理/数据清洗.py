import re
import json

# ========= Step 1: 读取原始文本 =========
with open("DATA/honglou.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# ========= Step 2: 清洗 =========
# 保留中文、数字、字母、换行，以及中文句号/问号/感叹号
cleaned_text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9。！？\n ]", " ", raw_text)
cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)   # 合并多余空格
cleaned_text = re.sub(r"\n+", "\n", cleaned_text).strip()  # 去掉多余空行

# ========= Step 3: 按章节切分 =========
pattern = r"(?m)^(第[一二三四五六七八九十百零〇两\d]+回[^\n]*)"
splits = re.split(pattern, cleaned_text)

chapters = []
chapter_id = 0

for i in range(1, len(splits), 2):
    chapter_id += 1
    title = splits[i].strip()
    content = splits[i+1].strip()
    content = re.sub(r"\s+", " ", content)  # 规范化空格

    chapters.append({
        "chapter_id": chapter_id,
        "title": title,
        "content": content
    })

# ========= Step 4: 保存为 JSON =========
with open("honglou_cleaned.json", "w", encoding="utf-8") as f:
    json.dump(chapters, f, ensure_ascii=False, indent=2)

print(f"✅ 共处理 {len(chapters)} 章，已保存到 honglou_cleaned.json")





