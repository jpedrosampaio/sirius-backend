from fastapi import FastAPI, APIRouter, HTTPException, File, UploadFile, Form, Cookie, Response, Request
from fastapi.responses import FileResponse, StreamingResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict, EmailStr
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone, timedelta
import bcrypt
from emergentintegrations.llm.chat import LlmChat, UserMessage
import io
import json

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

app = FastAPI()
api_router = APIRouter(prefix="/api")

class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    user_id: str
    email: str
    name: str
    picture: Optional[str] = None
    xp: int = 0
    rank: str = "Recruta"
    created_at: datetime

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Task(BaseModel):
    model_config = ConfigDict(extra="ignore")
    task_id: str
    user_id: str
    title: str
    description: Optional[str] = None
    completed: bool = False
    date: str
    priority: str = "medium"
    xp_reward: int = 10
    recurrence: str = "none"
    is_template: bool = True
    created_at: datetime

class TaskCreate(BaseModel):
    title: str
    description: Optional[str] = None
    date: str
    priority: str = "medium"
    recurrence: str = "none"

class Budget(BaseModel):
    model_config = ConfigDict(extra="ignore")
    budget_id: str
    user_id: str
    category: str
    limit: float
    spent: float = 0
    month: str
    budget_type: str = "fixed"
    percentage: Optional[float] = None
    created_at: datetime

class BudgetCreate(BaseModel):
    category: str
    limit: float
    month: str
    budget_type: str = "fixed"
    percentage: Optional[float] = None

class ChatMessageCreate(BaseModel):
    content: str

class Goal(BaseModel):
    model_config = ConfigDict(extra="ignore")
    goal_id: str
    user_id: str
    title: str
    description: Optional[str] = None
    target_date: str
    progress: float = 0
    sprint_duration: int = 60
    daily_checks: List[str] = []
    sprints: List[Dict[str, Any]] = []
    created_at: datetime

async def get_current_user(authorization: Optional[str] = None, session_token: Optional[str] = Cookie(None)) -> User:
    token = session_token or (authorization.replace("Bearer ", "") if authorization else None)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    session = await db.user_sessions.find_one({"session_token": token}, {"_id": 0})
    if not session:
        raise HTTPException(status_code=401, detail="Invalid session")
    
    expires_at = session["expires_at"]
    if isinstance(expires_at, str):
        expires_at = datetime.fromisoformat(expires_at)
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail="Session expired")
    
    user_doc = await db.users.find_one({"user_id": session["user_id"]}, {"_id": 0})
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found")
    
    if isinstance(user_doc['created_at'], str):
        user_doc['created_at'] = datetime.fromisoformat(user_doc['created_at'])
    
    return User(**user_doc)

@api_router.get("/")
async def root():
    return {"message": "Sirius API - Discipline is Destiny"}

@api_router.post("/auth/register")
async def register(user_data: UserCreate, response: Response):
    existing = await db.users.find_one({"email": user_data.email}, {"_id": 0})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = bcrypt.hashpw(user_data.password.encode('utf-8'), bcrypt.gensalt())
    user_id = f"user_{uuid.uuid4().hex[:12]}"
    
    user_doc = {
        "user_id": user_id,
        "email": user_data.email,
        "name": user_data.name,
        "password": hashed_password.decode('utf-8'),
        "picture": None,
        "xp": 0,
        "rank": "Recruta",
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    await db.users.insert_one(user_doc)
    
    session_token = f"session_{uuid.uuid4().hex}"
    session_doc = {
        "user_id": user_id,
        "session_token": session_token,
        "expires_at": (datetime.now(timezone.utc) + timedelta(days=7)).isoformat(),
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    await db.user_sessions.insert_one(session_doc)
    
    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=True,
        secure=True,
        samesite="none",
        path="/",
        max_age=7*24*60*60
    )
    
    return {"session_token": session_token, "user": {"user_id": user_id, "email": user_data.email, "name": user_data.name}}

@api_router.post("/auth/login")
async def login(credentials: UserLogin, response: Response):
    user_doc = await db.users.find_one({"email": credentials.email}, {"_id": 0})
    if not user_doc:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if not bcrypt.checkpw(credentials.password.encode('utf-8'), user_doc['password'].encode('utf-8')):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    session_token = f"session_{uuid.uuid4().hex}"
    session_doc = {
        "user_id": user_doc["user_id"],
        "session_token": session_token,
        "expires_at": (datetime.now(timezone.utc) + timedelta(days=7)).isoformat(),
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    await db.user_sessions.insert_one(session_doc)
    
    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=True,
        secure=True,
        samesite="none",
        path="/",
        max_age=7*24*60*60
    )
    
    return {"session_token": session_token, "user": {"user_id": user_doc["user_id"], "email": user_doc["email"], "name": user_doc["name"]}}

@api_router.post("/tasks")
async def create_task(request: Request, task_data: TaskCreate, session_token: Optional[str] = Cookie(None)):
    auth_header = request.headers.get("Authorization")
    user = await get_current_user(authorization=auth_header, session_token=session_token)
    
    task_id = f"task_{uuid.uuid4().hex[:12]}"
    task_doc = {
        "task_id": task_id,
        "user_id": user.user_id,
        "title": task_data.title,
        "description": task_data.description,
        "completed": False,
        "date": task_data.date,
        "priority": task_data.priority,
        "xp_reward": 10 if task_data.priority == "low" else 20 if task_data.priority == "medium" else 30,
        "recurrence": task_data.recurrence,
        "is_template": True,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    await db.tasks.insert_one(task_doc)
    task_doc['created_at'] = datetime.fromisoformat(task_doc['created_at'])
    return Task(**task_doc)

@api_router.post("/goals/{goal_id}/check")
async def check_goal_day(request: Request, goal_id: str, date: str, session_token: Optional[str] = Cookie(None)):
    auth_header = request.headers.get("Authorization")
    user = await get_current_user(authorization=auth_header, session_token=session_token)
    
    goal = await db.goals.find_one({"goal_id": goal_id, "user_id": user.user_id}, {"_id": 0})
    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")
    
    daily_checks = goal.get('daily_checks', [])
    was_checked = date in daily_checks
    
    if was_checked:
        daily_checks.remove(date)
        xp_change = 0
    else:
        daily_checks.append(date)
        xp_change = 5
        new_xp = user.xp + xp_change
        new_rank = calculate_rank(new_xp)
        await db.users.update_one({"user_id": user.user_id}, {"$set": {"xp": new_xp, "rank": new_rank}})
    
    await db.goals.update_one(
        {"goal_id": goal_id},
        {"$set": {"daily_checks": daily_checks}}
    )
    
    if xp_change > 0:
        return {"message": "Day checked", "xp_earned": xp_change, "new_xp": user.xp + xp_change}
    else:
        return {"message": "Day unchecked", "xp_earned": 0}

@api_router.post("/chat/send")
async def send_chat_message(request: Request, message_data: ChatMessageCreate, session_token: Optional[str] = Cookie(None)):
    auth_header = request.headers.get("Authorization")
    user = await get_current_user(authorization=auth_header, session_token=session_token)
    
    content = message_data.content
    
    message_id = f"msg_{uuid.uuid4().hex[:12]}"
    user_message = {
        "message_id": message_id,
        "user_id": user.user_id,
        "role": "user",
        "content": content,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    await db.chat_messages.insert_one(user_message)
    
    prompt = f"""Você é um assistente financeiro integrado ao app Sirius. O usuário enviou a seguinte mensagem: "{content}".
    
Analise a mensagem e identifique se há informações sobre transações financeiras (receitas ou despesas).
Se houver, extraia:
- tipo: "income" ou "expense"
- valor (número)
- categoria
- descrição

Responda em formato JSON se for uma transação, ou uma mensagem de texto amigável caso contrário.

Exemplo de resposta JSON:
{{"type": "expense", "amount": 50.0, "category": "alimentação", "description": "almoço no restaurante"}}
"""
    
    try:
        llm_key = os.getenv("EMERGENT_LLM_KEY", "")
        chat = LlmChat(
            api_key=llm_key,
            session_id=f"chat_{user.user_id}",
            system_message="Você é um assistente financeiro do Sirius."
        ).with_model("openai", "gpt-5.2")
        
        response = await chat.send_message(UserMessage(text=prompt))
        
        ai_message_id = f"msg_{uuid.uuid4().hex[:12]}"
        ai_message = {
            "message_id": ai_message_id,
            "user_id": user.user_id,
            "role": "assistant",
            "content": response,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        transaction_data = None
        try:
            transaction_data = json.loads(response)
            if "type" in transaction_data and "amount" in transaction_data:
                transaction_id = f"trans_{uuid.uuid4().hex[:12]}"
                transaction_doc = {
                    "transaction_id": transaction_id,
                    "user_id": user.user_id,
                    "type": transaction_data["type"],
                    "amount": float(transaction_data["amount"]),
                    "category": transaction_data.get("category", "outros"),
                    "description": transaction_data.get("description", ""),
                    "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    "created_at": datetime.now(timezone.utc).isoformat()
                }
                await db.transactions.insert_one(transaction_doc)
                ai_message["transaction_data"] = transaction_data
        except:
            pass
        
        await db.chat_messages.insert_one(ai_message)
        
        user_message['created_at'] = datetime.fromisoformat(user_message['created_at'])
        ai_message['created_at'] = datetime.fromisoformat(ai_message['created_at'])
        
        return {"user_message": user_message, "ai_message": ai_message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/reports/{report_id}/download")
async def download_report(report_id: str, request: Request, session_token: Optional[str] = Cookie(None)):
    auth_header = request.headers.get("Authorization")
    user = await get_current_user(authorization=auth_header, session_token=session_token)
    
    report = await db.reports.find_one({"report_id": report_id, "user_id": user.user_id}, {"_id": 0})
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    content = f"""SIRIUS - RELATÓRIO {report['type'].upper()}
Período: {report['period']}
Gerado em: {report['created_at']}

{'='*60}
DADOS DO PERÍODO
{'='*60}

Tarefas: {report['data']['tasks_completed']}/{report['data']['tasks']}
Hábitos: {report['data']['total_habits_completions']} completações
Receitas: R$ {report['data']['income']:.2f}
Despesas: R$ {report['data']['expenses']:.2f}
Metas: {report['data']['goals']} total
Progresso Médio: {report['data']['goals_progress']:.1f}%

{'='*60}
INSIGHTS E SUGESTÕES
{'='*60}

{report['insights']}
"""
    
    buffer = io.BytesIO(content.encode('utf-8'))
    buffer.seek(0)
    
    return StreamingResponse(
        buffer,
        media_type="text/plain",
        headers={"Content-Disposition": f"attachment; filename=sirius_relatorio_{report_id}.txt"}
    )

@api_router.get("/finance/stats")
async def get_finance_stats(request: Request, month: Optional[str] = None, session_token: Optional[str] = Cookie(None)):
    auth_header = request.headers.get("Authorization")
    user = await get_current_user(authorization=auth_header, session_token=session_token)
    
    if not month:
        month = datetime.now(timezone.utc).strftime("%Y-%m")
    
    transactions = await db.transactions.find(
        {"user_id": user.user_id, "date": {"$regex": f"^{month}"}},
        {"_id": 0}
    ).to_list(1000)
    
    categories_expense = {}
    categories_income = {}
    
    for t in transactions:
        if t['type'] == 'expense':
            categories_expense[t['category']] = categories_expense.get(t['category'], 0) + t['amount']
        else:
            categories_income[t['category']] = categories_income.get(t['category'], 0) + t['amount']
    
    return {
        "expense_by_category": categories_expense,
        "income_by_category": categories_income,
        "total_expense": sum(categories_expense.values()),
        "total_income": sum(categories_income.values())
    }

def calculate_rank(xp: int) -> str:
    ranks = [
        (0, "Recruta"),
        (100, "Soldado"),
        (300, "Cabo"),
        (600, "Sargento"),
        (1000, "Tenente"),
        (1500, "Capitão"),
        (2200, "Major"),
        (3000, "Coronel"),
        (4000, "General")
    ]
    for threshold, rank in reversed(ranks):
        if xp >= threshold:
            return rank
    return "Recruta"

app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()