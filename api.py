# api.py
import json
import uuid
import redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="RAG Agentic API (MQ Version)")
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# API 模型
class ChatRequest(BaseModel):
    session_id: str
    user_input: str

# ==========================================
# FastAPI 路由端点
# ==========================================
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """生产者：将用户的提问作为任务丢进 Redis 消息队列"""
    task_id = str(uuid.uuid4())
    session_id = request.session_id
    user_input = request.user_input

    # 1. 构建任务负载 (Payload)
    task_payload = {
        "task_id": task_id,
        "session_id": session_id,
        "user_input": user_input
    }

    # 2. 初始化任务状态为 pending (处理中)
    redis_client.setex(f"task_status:{task_id}", 3600, json.dumps({"status": "pending"}))

    # 3. 将任务压入 Redis 队列的最右侧 (RPUSH)
    redis_client.rpush("rag_task_queue", json.dumps(task_payload))

    # 4. 0.1秒内立刻向前端返回 task_id，绝不阻塞！
    return {"task_id": task_id, "status": "pending", "message": "任务已加入队列，后台正在拼命计算中..."}


@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """前端轮询接口：检查后台 Worker 是否把任务算完了"""
    status_data = redis_client.get(f"task_status:{task_id}")
    if not status_data:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return json.loads(status_data)


@app.get("/memory/{session_id}")
async def get_memory_endpoint(session_id: str):
    """获取侧边栏结构化记忆"""
    memory = redis_client.get(f"memory:{session_id}")
    return {"structured_memory": json.loads(memory) if memory else []}