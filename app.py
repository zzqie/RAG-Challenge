import streamlit as st
import requests
import uuid
import time

API_URL = "http://localhost:8000"
st.set_page_config(page_title="企业财报 RAG 智能体", page_icon="📈", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

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
                    st.caption(f"文档Hash: {mem['pdf_hash'][:10]}...")
    except:
        pass

# --- 主界面 ---
st.title("📈 跨实体金融财报 RAG 智能问答系统")
st.caption("基于 MQ (消息队列) 的工业级高并发架构")

# ====================================================
# 【修改点 1】：历史记录渲染逻辑升级
# 循环读取时，检查是否有推理过程，如果有就渲染折叠面板
# ====================================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # 如果是 AI 的回复，且字典里保存了推理过程，就把它画出来
        if message["role"] == "assistant" and "reasoning" in message:
            with st.expander("👀 查看底层推演逻辑"):
                st.info(message["reasoning"])
                st.json(message["references"])

if user_input := st.chat_input("请输入疑问..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        # 1. 把任务发给 FastAPI (瞬间返回 task_id)
        response = requests.post(
            f"{API_URL}/chat",
            json={"session_id": st.session_state.session_id, "user_input": user_input}
        )
        task_info = response.json()
        task_id = task_info.get("task_id")

        # 2. 核心：建立一个等待占位符，开始轮询！
        status_placeholder = st.empty()
        
        while True:
            # 向后端查询任务状态
            status_resp = requests.get(f"{API_URL}/task/{task_id}")
            task_data = status_resp.json()
            
            if task_data["status"] == "completed":
                # 任务完成，清除占位符
                status_placeholder.empty()
                
                final_answer = task_data["final_answer"]
                reasoning = task_data.get("reasoning", "无推理过程")
                refs = task_data.get("references", [])
                
                # ====================================================
                # 【修改点 2】：不仅展示，还要把完整数据存入 session_state
                # 这样下次 st.rerun() 刷新页面时，依然能画出完整的组件
                # ====================================================
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": final_answer,
                    "reasoning": reasoning,  # 保存推理过程
                    "references": refs       # 保存物理页码
                })
                
                # 触发页面全局刷新（这会重新执行上面那一块渲染历史记录的代码）
                st.rerun() 
                break
                
            elif task_data["status"] == "failed":
                status_placeholder.error(f"处理失败: {task_data.get('error')}")
                break
                
            else:
                # 任务还在队列或处理中，显示动画并睡眠 1.5 秒再问
                status_placeholder.info("⏳ 任务已入队，AI Agent 正在后台拼命推理中，请稍候...")
                time.sleep(1.5)