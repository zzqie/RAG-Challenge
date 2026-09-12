import json
import os
import glob
import redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from pyprojroot import here
from openai import OpenAI
from src.pipeline import Pipeline, configs
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(base_url="https://api.chatanywhere.tech/v1")

# 初始化 FastAPI 和 Redis
app = FastAPI(title="RAG Agentic API")
# 根据你的实际情况修改 Redis 配置
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# ==========================================
# 0. API 数据模型 (Pydantic)
# ==========================================
class ChatRequest(BaseModel):
    session_id: str  # 用于区分不同用户的标识
    user_input: str

class ChatResponse(BaseModel):
    final_answer: str
    reasoning: str
    references: list
    is_comparison: bool

# ==========================================
# 1. Redis 状态管理工具
# ==========================================
def get_history(session_id: str) -> list:
    history = redis_client.get(f"history:{session_id}")
    return json.loads(history) if history else []

def save_history(session_id: str, role: str, content: str):
    history = get_history(session_id)
    history.append({"role": role, "content": content})
    # 缓存过期时间设为 1 天
    redis_client.setex(f"history:{session_id}", 86400, json.dumps(history, ensure_ascii=False))

def get_memory(session_id: str) -> list:
    memory = redis_client.get(f"memory:{session_id}")
    return json.loads(memory) if memory else []

def save_memory(session_id: str, memory_item: dict):
    memory = get_memory(session_id)
    memory.append(memory_item)
    redis_client.setex(f"memory:{session_id}", 86400, json.dumps(memory, ensure_ascii=False))

# ==========================================
# 2. 核心业务逻辑 (保持原样，略作调整)
# ==========================================
def rewrite_and_classify_query(user_query: str, chat_history: list, memory: list) -> dict:
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

def extract_to_structured_memory(query_text: str, answer_obj: dict, session_id: str):
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
        
        # 组装结构化记忆并保存到 Redis
        memory_item = {
            "company": extracted.get("company_name", "Unknown"),
            "metric": extracted.get("metric", "Unknown"),
            "value": value,
            "pdf_hash": pdf_hash,
            "pages": pages
        }
        save_memory(session_id, memory_item)
    except Exception as e:
        pass # 如果抽取失败，静默跳过，不影响主流程

# (保留你原来的 synthesize_comparison 和 run_rag_pipeline)
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
# 3. FastAPI 路由端点
# ==========================================
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    session_id = request.session_id
    user_input = request.user_input

    # 1. 从 Redis 获取上下文
    history = get_history(session_id)
    memory = get_memory(session_id)

    # 2. 意图分类与改写
    query_data = rewrite_and_classify_query(user_input, history, memory)
    
    # 保存用户输入到历史记录
    save_history(session_id, "user", user_input)

    try:
        if query_data.get("compare_mode"):
            # 分支 A: 记忆短路对比
            rag_answer = run_rag_pipeline(query_data)
            final_synth = synthesize_comparison(user_input, query_data["known_data"], rag_answer)
            
            answer = final_synth.get("final_answer", "")
            reasoning = final_synth.get("reasoning", "")
            refs = final_synth.get("combined_references", [])
            
            save_history(session_id, "assistant", answer)
            return ChatResponse(final_answer=answer, reasoning=reasoning, references=refs, is_comparison=True)
            
        else:
            # 分支 B: 常规检索
            answer_obj = run_rag_pipeline(query_data)
            # 异步/同步抽取到 Redis 记忆
            extract_to_structured_memory(query_data['text'], answer_obj, session_id)
            
            answer = answer_obj.get("value", "未找到答案")
            reasoning = answer_obj.get("reasoning_process", "无")
            refs = answer_obj.get("references", [])
            
            save_history(session_id, "assistant", str(answer))
            return ChatResponse(final_answer=str(answer), reasoning=reasoning, references=refs, is_comparison=False)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/{session_id}")
async def get_memory_endpoint(session_id: str):
    """供前端轮询侧边栏记忆库的接口"""
    return {"structured_memory": get_memory(session_id)}