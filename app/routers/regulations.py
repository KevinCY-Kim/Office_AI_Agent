# backend.py
from fastapi import FastAPI, Request, APIRouter, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from typing_extensions import Annotated
import uvicorn
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# regulation_service.py 경로 추가
sys.path.append("/home/alpaco/cjw/final/web/Office_AI_Agent/app/services")

import gc
import torch
gc.collect()
with torch.cuda.device(1):
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
from app.services import regulations_service 
agent = regulations_service.RAGAgent(
    chunks_path="/home/alpaco/cjw/final/web/Office_AI_Agent/report/standard_flattened.json"
)

# 요청 형식 정의
class QueryRequest(BaseModel):
    query: str

# 루트 페이지 - 챗봇 웹페이지 제공
@router.get("/regulations", response_class=HTMLResponse)
async def regulations(request: Request):
    return templates.TemplateResponse("regulations.html", {"request": request})
    
    
@router.get("/ask")
def ask_get():
    return {"message": "POST 요청으로 질문을 보내세요."}    

# API 엔드포인트
@router.post("/ask")
def ask_question(req: QueryRequest):
    ans = agent.answer(
        req.query,
        top_k=10,
        score_threshold=0.45,
        max_return=3,
        generate_answer=True,
        keyword_bonus=0.2,
        gen_max_new_tokens=2000
    )
    
    # numpy 타입을 Python 기본 타입으로 변환하여 JSON 직렬화 문제 해결
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy scalar types
            return obj.item()
        else:
            return obj
    
    return convert_numpy_types(ans)

