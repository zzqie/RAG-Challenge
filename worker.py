import json
import redis
import os
import glob
from pyprojroot import here
from openai import OpenAI
from src.pipeline import Pipeline, configs
from dotenv import load_dotenv

# 加载环境变量与初始化客户端
load_dotenv()
client = OpenAI(base_url="https://api.chatanywhere.tech/v1")
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# ==========================================
# 1. Redis 状态管理工具
# ==========================================
def get_history(session_id: str) -> list:
    history = redis_client.get(f"history:{session_id}")
    return json.loads(history) if history else []

def save_history(session_id: str, role: str, content: str):
    history = get_history(session_id)
    history.append({"role": role, "content": content})
    redis_client.setex(f"history:{session_id}", 86400, json.dumps(history, ensure_ascii=False))

def get_memory(session_id: str) -> list:
    memory = redis_client.get(f"memory:{session_id}")
    return json.loads(memory) if memory else []

def save_memory(session_id: str, memory_item: dict):
    memory = get_memory(session_id)
    memory.append(memory_item)
    redis_client.setex(f"memory:{session_id}", 86400, json.dumps(memory, ensure_ascii=False))

# ==========================================
# 2. 核心大模型与 RAG 业务逻辑 (全部搬到这里)
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
    value = answer_obj.get("value", "N/A")
    refs = answer_obj.get("references", [])
    
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
        
        memory_item = {
            "company": extracted.get("company_name", "Unknown"),
            "metric": extracted.get("metric", "Unknown"),
            "value": value,
            "pdf_hash": pdf_hash,
            "pages": pages
        }
        save_memory(session_id, memory_item)
    except Exception as e:
        pass 

def synthesize_comparison(user_query: str, known_data: str, rag_answer_obj: dict) -> dict:
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
# 3. 消费者 (Worker) 队列处理逻辑
# ==========================================
def process_task(task_payload: dict):
    task_id = task_payload["task_id"]
    session_id = task_payload["session_id"]
    user_input = task_payload["user_input"]

    print(f"🚀 [Worker] 开始处理任务: {task_id}")

    try:
        # 记录用户提问
        save_history(session_id, "user", user_input)
        
        # 获取最新状态
        history = get_history(session_id)
        memory = get_memory(session_id)

        # 意图路由
        query_data = rewrite_and_classify_query(user_input, history, memory)
        
        # 核心分支处理
        if query_data.get("compare_mode"):
            rag_answer = run_rag_pipeline(query_data)
            final_synth = synthesize_comparison(user_input, query_data["known_data"], rag_answer)
            
            final_answer = final_synth.get("final_answer", "")
            reasoning = final_synth.get("reasoning", "")
            refs = final_synth.get("combined_references", [])
        else:
            answer_obj = run_rag_pipeline(query_data)
            extract_to_structured_memory(query_data['text'], answer_obj, session_id)
            
            final_answer = answer_obj.get("value", "未找到答案")
            reasoning = answer_obj.get("reasoning_process", "无")
            refs = answer_obj.get("references", [])
            
        # 记录 AI 回答
        save_history(session_id, "assistant", str(final_answer))

        # 将结果写回 Redis，状态更新为 completed
        result_data = {
            "status": "completed",
            "final_answer": str(final_answer),
            "reasoning": reasoning,
            "references": refs
        }
        redis_client.setex(f"task_status:{task_id}", 3600, json.dumps(result_data))
        print(f"✅ [Worker] 任务完成: {task_id}")

    except Exception as e:
        print(f"❌ [Worker] 任务失败: {str(e)}")
        error_data = {"status": "failed", "error": str(e)}
        redis_client.setex(f"task_status:{task_id}", 3600, json.dumps(error_data))

def start_worker():
    print("🤖 AI Worker 启动，正在监听 Redis 消息队列 (rag_task_queue)...")
    while True:
        # BLPOP 0 表示一直阻塞等待，不占 CPU
        task = redis_client.blpop("rag_task_queue", 0)
        if task:
            queue_name, task_data = task
            task_payload = json.loads(task_data)
            process_task(task_payload)

if __name__ == "__main__":
    start_worker()