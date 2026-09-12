import streamlit as st
import json
import os
import glob
from pyprojroot import here
from openai import OpenAI
from src.pipeline import Pipeline, configs
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(base_url="https://api.chatanywhere.tech/v1")

# ==========================================
# 1. 核心业务逻辑与智能体工具
# ==========================================

def rewrite_and_classify_query(user_query: str, chat_history: list, memory: list) -> dict:
    """带记忆感知能力的智能路由引擎"""
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history]) if chat_history else "无"
    memory_str = json.dumps(memory, ensure_ascii=False) if memory else "无"
    
    prompt = f"""
    You are an intelligent routing agent for a financial RAG system.
    
    Chat History: {history_str}
    Structured Memory (Facts already known to the system without needing DB search): {memory_str}
    Latest User Query: {user_query}
    
    TASK:
    1. Determine if the User Query asks for a comparison between two companies.
    2. If it is a COMPARISON (e.g., A vs B) AND we already have the metric for Company A in the Structured Memory:
       - You MUST set "compare_mode" to true.
       - Extract Company A's data from memory and put it in "known_data".
       - Your "text" should be a standalone query ONLY asking to find the metric for Company B.
    3. If it is NOT a comparison, or neither company is in memory, rewrite the query to be standalone, and set "compare_mode" to false.
    4. Classify "kind" as "boolean", "number", or "name".
    
    Output strictly in JSON:
    {{
        "text": "The standalone query to run in RAG database",
        "kind": "boolean | number | name",
        "compare_mode": true or false,
        "known_data": "If compare_mode is true, describe the known company's value and source from memory, else empty string"
    }}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        response_format={ "type": "json_object" }
    )
    return json.loads(response.choices[0].message.content.strip())

def extract_to_structured_memory(query_text: str, answer_obj: dict):
    """记忆抽取器：将非结构化回答转化为结构化记忆"""
    value = answer_obj.get("value", "N/A")
    refs = answer_obj.get("references", [])
    
    # 提取物理页码
    pages = [ref.get("page_index") for ref in refs if "page_index" in ref]
    pdf_hash = refs[0].get("pdf_sha1", "Unknown") if refs else "Unknown"

    prompt = f"""
    Extract the company name and the specific metric being asked about from the following question.
    Question: {query_text}
    
    Output JSON:
    {{
        "company_name": "Name of the company",
        "metric": "Name of the metric or topic"
    }}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" }
        )
        extracted = json.loads(response.choices[0].message.content.strip())
        
        # 组装结构化记忆并保存
        memory_item = {
            "company": extracted.get("company_name", "Unknown"),
            "metric": extracted.get("metric", "Unknown"),
            "value": value,
            "pdf_hash": pdf_hash,
            "pages": pages
        }
        st.session_state.structured_memory.append(memory_item)
    except Exception as e:
        pass # 如果抽取失败，静默跳过，不影响主流程

def synthesize_comparison(user_query: str, known_data: str, rag_answer_obj: dict) -> dict:
    """记忆融合器：将记忆数据与新检索的数据拼接对比"""
    rag_val = rag_answer_obj.get("value")
    rag_reasoning = rag_answer_obj.get("reasoning_process")
    rag_refs = rag_answer_obj.get("references", [])
    
    prompt = f"""
    Answer the user's comparative question by comparing Data 1 (from our memory) and Data 2 (from new DB search).
    User Question: {user_query}
    
    Data 1 (Memory): {known_data}
    Data 2 (New Search): Value={rag_val}, Reasoning={rag_reasoning}, References={rag_refs}
    
    Output JSON:
    {{
        "final_answer": "Clear statement of which company is higher/lower/better based on the question",
        "reasoning": "Step by step comparison of Data 1 and Data 2",
        "combined_references": [
            {{"source": "Company 1 (Memory)", "info": "Hash/Page details"}},
            {{"source": "Company 2 (Database)", "info": "Hash/Page details"}}
        ]
    }}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        response_format={ "type": "json_object" }
    )
    return json.loads(response.choices[0].message.content.strip())

def run_rag_pipeline(query_data: dict) -> dict:
    """底层离线 RAG 调用"""
    root_path = here() / "data" / "test_set"
    questions_path = root_path / "questions.json"
    
    with open(questions_path, "w", encoding="utf-8") as f:
        json.dump([{"text": query_data["text"], "kind": query_data["kind"]}], f, ensure_ascii=False, indent=2)
        
    pipeline = Pipeline(root_path, run_config=configs["max_nst_o3m"])
    pipeline.process_questions()
    
    search_pattern = str(root_path / "answers*.json")
    latest_answer_file = max(glob.glob(search_pattern), key=os.path.getctime)
    
    with open(latest_answer_file, "r", encoding="utf-8") as f:
        result_data = json.load(f)
        
    return result_data.get("answers", [{}])[0]

# ==========================================
# 2. Streamlit 界面渲染
# ==========================================

st.set_page_config(page_title="企业财报 RAG 智能体", page_icon="📈", layout="wide")

# 初始化状态
if "messages" not in st.session_state:
    st.session_state.messages = []
if "structured_memory" not in st.session_state:
    st.session_state.structured_memory = []

# --- 炫酷的侧边栏：实时监控结构化记忆 ---
with st.sidebar:
    st.header("🧠 实时结构化记忆库")
    st.caption("已固化的实体知识，下次提问免检索直出！")
    if not st.session_state.structured_memory:
        st.info("记忆库目前为空。")
    else:
        # 美观地展示记忆
        for i, mem in enumerate(st.session_state.structured_memory):
            with st.expander(f"📚 {mem['company']} - {mem['metric']}"):
                st.write(f"**提取数值**: `{mem['value']}`")
                st.write(f"**物理页码**: {mem['pages']}")
                st.caption(f"文档Hash: {mem['pdf_hash'][:10]}...")

# --- 主界面 ---
st.title("📈 跨实体金融财报 RAG 智能问答系统")
st.caption("带有 Agentic Memory (记忆感知与动态路由) 机制")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("请输入疑问（如：Apple 的营收是多少？/ 它和 Holley Inc. 谁的更高？）"):
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("正在分析语境与匹配记忆库..."):
            query_data = rewrite_and_classify_query(user_input, st.session_state.messages, st.session_state.structured_memory)
            
        if query_data.get("compare_mode"):
            # 【分支 A：触发记忆融合，只检索一家公司】
            st.warning(f"⚡ **触发记忆短路机制**！已知 `{query_data['known_data']}`。系统动态截断问题，仅向底层 RAG 检索缺失实体：`{query_data['text']}`")
            
            with st.spinner("🚀 正在底层数据库中检索缺失实体..."):
                try:
                    rag_answer = run_rag_pipeline(query_data)
                    
                    with st.spinner("🧠 正在进行跨数据源(Memory + RAG)对比推理..."):
                        final_synth = synthesize_comparison(user_input, query_data["known_data"], rag_answer)
                    
                    st.success(f"**对比结果**: {final_synth.get('final_answer')}")
                    
                    with st.expander("👀 查看记忆融合与对比推理细节"):
                        st.markdown("### ⚖️ 对比逻辑推演")
                        st.info(final_synth.get("reasoning"))
                        st.markdown("### 📄 双源溯源信息")
                        st.json(final_synth.get("combined_references"))
                        
                    st.session_state.messages.append({"role": "user", "content": user_input})
                    st.session_state.messages.append({"role": "assistant", "content": final_synth.get('final_answer')})
                except Exception as e:
                    st.error(f"处理时发生错误: {str(e)}")
                    
        else:
            # 【分支 B：常规的全新检索】
            if query_data['text'] != user_input:
                st.info(f"🔄 **问题已结合语境重写**: {query_data['text']}")
                
            with st.spinner("🚀 正在触发全链路 RAG 检索 (召回 -> 父文档提取 -> O3-Mini推理)..."):
                try:
                    answer_obj = run_rag_pipeline(query_data)
                    
                    # 抽取并保存到记忆库
                    extract_to_structured_memory(query_data['text'], answer_obj)
                    
                    final_answer = answer_obj.get("value", "抱歉，未找到答案。")
                    reasoning = answer_obj.get("reasoning_process", "无推理过程。")
                    refs = answer_obj.get("references", [])
                    
                    st.success(f"**最终答案**: {final_answer}")
                    
                    with st.expander("👀 查看底层 CoT 推理与精准溯源信息"):
                        st.markdown("### 🧠 深度推理链 (Step-by-Step Analysis)")
                        st.info(reasoning)
                        
                        st.markdown("### 📄 精准引用出处 (PDR溯源)")
                        if refs:
                            for i, ref in enumerate(refs):
                                st.markdown(f"- **来源 {i+1}**: 文档 ID `{ref.get('pdf_sha1', '未知')[:8]}...` | **第 {ref.get('page_index', '未知')} 页**")
                        else:
                            st.write("未提取到具体引用页码。")
                    
                    st.session_state.messages.append({"role": "user", "content": user_input})
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})
                except Exception as e:
                    st.error(f"处理时发生底层错误: {str(e)}")