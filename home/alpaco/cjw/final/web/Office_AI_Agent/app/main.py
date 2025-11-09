import sys
from pathlib import Path
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

# app 폴더 인식 가능하게 경로 추가
sys.path.append(str(Path(__file__).resolve().parent.parent))

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

from app.routers import auth, document_generator, regulations, notices, pages, statistics

# FastAPI 애플리케이션 생성
app = FastAPI(title="Office AI Agent System")

# 정적 파일 마운트 (절대 경로 사용)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Jinja2 템플릿 설정 (절대 경로 사용 - 위에서 이미 설정됨)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 라우터 등록
app.include_router(pages.router, tags=["Pages"])
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(document_generator.router, prefix="/document-generator", tags=["Document Generator"])
app.include_router(regulations.router, tags=["Regulations"])
app.include_router(notices.router, prefix="/notices", tags=["Notices"])
app.include_router(statistics.router, prefix="/statistics", tags=["Statistics"])

# 루트 경로
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



# ✅ uvicorn을 코드 내에서 직접 실행
if __name__ == "__main__":
    uvicorn.run(
        app,          # 모듈 경로
        host="127.0.0.1",        # 로컬호스트
        port=8001,               # 기본 포트
    )