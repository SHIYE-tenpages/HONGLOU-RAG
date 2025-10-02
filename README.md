# 📖 红楼梦 RAG 助手

一个基于 **RAG (Retrieval-Augmented Generation)** 的《红楼梦》智能问答系统。  
支持 **原文检索**、**人物关系图谱**、**评论知识库** 三重知识源，并结合 **Cross-Encoder 重排序**，提升问答准确率。  
前端基于 **Streamlit**，同时提供人物关系的交互式可视化。

---

##  功能亮点
- **智能问答**：基于《红楼梦》原文，支持人物关系与评论补充。
- **多源知识融合**：原文 + 人物知识库 + 红学评论（待完善）。
- **Cross-Encoder Rerank**：避免纯向量检索的错误匹配。
- **人物关系图谱可视化**：交互式展示人物网络。
- **创新点**：融合多源知识、可视化、RAG pipeline 优化。


## ⚙️ 安装与运行

1. **克隆仓库并进入目录**
   ```bash
   git clone https://github.com/yourname/RAG-HonglouMeng.git
   cd RAG-HonglouMeng
2. **安装依赖**
pip install -r requirements.txt
3. **构建索引**
python build_index.py
4. **启动前端**
streamlit run app.py

然后访问：
 http://localhost:8501


## 技术栈：
FAISS
 - 向量检索

Sentence-Transformers
 - 向量表示

Transformers
 - Qwen 生成模型

Streamlit
 - 前端展示

Pyvis + NetworkX
 - 人物关系可视化

## TODO

支持更大规模的知识扩展（脂砚斋评注等）

优化 chunk 切分和检索效果

增加英文版支持

尝试更大的模型（电脑还没换目前模型能运行的很小，后续或许会更新）
