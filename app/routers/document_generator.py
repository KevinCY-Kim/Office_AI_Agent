import os, shutil
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing_extensions import Annotated
from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
router = APIRouter()
templates = Jinja2Templates(directory="templates")
import fitz
from docx import Document

@router.get("/")
async def generate_document(request : Request): 
    return templates.TemplateResponse("document.html", context={'request' : request}) 


# app = FastAPI()

# # ======== CORS 설정 ========


UPLOAD_DIR = "/home/alpaco/cjw/final/web/Office_AI_Agent/app/uploads"
OUTPUT_DIR = "/home/alpaco/cjw/final/web/Office_AI_Agent/app/outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ======== 파일 업로드 포함 문서 생성 ========
@router.get("/generate_with_file")
async def generate_with_file_get():
    return {"message": "POST 요청으로 문서를 생성하세요."} 


@router.post("/generate_with_file")
async def generate_with_file(
    project_name: str = Form(...),
    depart_name: str = Form(...),
    project_no: str = Form(...),
    period: str = Form(...),
    budget: str = Form(...),
    file: UploadFile = File(None)
):
    try:
        uploaded_path = None
        if file:
            uploaded_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(uploaded_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        # 문서 생성
        # import gc
        # import torch
        # gc.collect()
        # with torch.cuda.device(1):
        #     torch.cuda.empty_cache()
        #     torch.cuda.ipc_collect()
        # from routers.RND import render_doc
        # render_doc(project_name, depart_name, project_no, period, budget)

        output_path = os.path.join(OUTPUT_DIR, "RND_Report.docx")
        if os.path.exists("RND_Report.docx"):
            os.replace("RND_Report.docx", output_path)

        return JSONResponse({
            "message": "✅ 문서 생성 완료" + (f" (첨부파일: {file.filename})" if file else ""),
            "file_path": output_path,
            "uploaded_file": uploaded_path,
            "status": "success"
        })

    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)


@router.get("/download")
async def download_report():
    path = os.path.join(OUTPUT_DIR, "RND_Report.docx")
    if not os.path.exists(path):
        return JSONResponse({"error": "문서가 존재하지 않습니다."}, status_code=404)
    return FileResponse(path, filename="RND_Report.docx")

from jinja2 import Template

@router.get("/doc_page", response_class=HTMLResponse)
async def doc_page(request: Request):
    doc = Document(OUTPUT_DIR+"/RND_Report.docx")
    html_content = "<h2>📄 문서 페이지 내용</h2>"
    for pages in doc.paragraphs:
        for page in pages.runs:
            html_content += "<br>" + page.text + "</br>" 
    return HTMLResponse(content=html_content)
