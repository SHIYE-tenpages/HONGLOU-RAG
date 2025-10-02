import streamlit as st
import networkx as nx
from pyvis.network import Network
import os, json
from rag_qa import answer_question

#Streamlit 页面配置
st.set_page_config(page_title="红楼梦 RAG 助手", layout="wide")
st.title("📖 红楼梦 RAG 助手")
st.markdown("一个基于 RAG 的《红楼梦》智能问答系统，支持原文、人物关系与评论三重知识源。")

# 问答区域
st.subheader("🔍 提问区")
q = st.text_input("请输入你的问题：", "刘姥姥首次在那个章节中登场？")

if st.button("查询"):
    with st.spinner("思考中..."):
        res = answer_question(q)

        # 显示答案
        st.subheader("📌 答案")
        st.write(res["answer"])

        # 显示检索片段
        st.subheader("📚 检索到的片段")
        for i, r in enumerate(res["retrieved"], 1):
            st.markdown(f"**片段{i}** (score={r['score']:.2f}, 来源={r['source']})")
            st.markdown(f"> {r['meta'].get('content','')[:300]} ...")

#人物关系图谱
st.subheader("👥 人物关系图谱 (Demo)")

# 加载人物关系数据
PERSON_PATH = "data/person.json"
if os.path.exists(PERSON_PATH):
    with open(PERSON_PATH, "r", encoding="utf-8") as f:
        persons = json.load(f)
else:
    persons = []

# 构建图谱
if persons:
    G = nx.Graph()
    for p in persons:
        G.add_node(p["name"])
        for r in p.get("relationships", []):
            G.add_edge(p["name"], r["name"], relation=r["relation"])

    # 用 pyvis 生成交互图
    net = Network(notebook=False, height="500px", width="100%", bgcolor="#222222", font_color="white")
    net.from_nx(G)
    net.save_graph("person_graph.html")

    st.markdown("下面是红楼梦人物关系的交互图谱：")
    st.components.v1.html(open("person_graph.html", "r", encoding="utf-8").read(), height=520)
else:
    st.info("未找到人物关系文件，请确保存在 `data/person.json`。")
