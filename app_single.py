import streamlit as st
import requests
import uuid

# FastAPI 服务的地址
API_URL = "http://localhost:8000"

st.set_page_config(page_title="企业财报 RAG 智能体", page_icon="📈", layout="wide")

# 1. 初始化唯一的用户会话 ID (Session ID)
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 侧边栏：通过 API 动态获取 Redis 中的记忆 ---
with st.sidebar:
    st.header("🧠 实时结构化记忆库")
    try:
        mem_resp = requests.get(f"{API_URL}/memory/{st.session_state.session_id}")
        memory_list = mem_resp.json().get("structured_memory", [])
        
        if not memory_list:
            st.info("记忆库目前为空。")
        else:
            for mem in memory_list:
                with st.expander(f"📚 {mem['company']} - {mem['metric']}"):
                    st.write(f"**提取数值**: `{mem['value']}`")
                    st.write(f"**物理页码**: {mem['pages']}")
                    st.caption(f"文档Hash: {mem['pdf_hash'][:10]}...")
    except Exception as e:
        st.error("无法连接到后端数据库。")

# --- 主界面 ---
st.title("📈 跨实体金融财报 RAG 智能问答系统")
st.caption("基于 FastAPI + Redis 驱动的微服务架构")

# 渲染历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("请输入疑问..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("🤖 Agent 正在后端服务器思考与检索..."):
            try:
                # 向 FastAPI 发送请求
                response = requests.post(
                    f"{API_URL}/chat",
                    json={"session_id": st.session_state.session_id, "user_input": user_input}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    final_answer = data["final_answer"]
                    
                    st.success(f"**最终答案**: {final_answer}")
                    with st.expander("👀 查看底层推演逻辑"):
                        st.info(data["reasoning"])
                        st.json(data["references"])
                        
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})
                    # 强迫 Streamlit 刷新侧边栏的新记忆
                    st.rerun() 
                else:
                    st.error(f"后端报错: {response.text}")
            except Exception as e:
                st.error(f"无法连接到后端服务: {e}")