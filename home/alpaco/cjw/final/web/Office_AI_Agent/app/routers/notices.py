from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing_extensions import Annotated
from fastapi import Form
router = APIRouter()
templates = Jinja2Templates(directory="templates")

@router.get("/")
async def get_notices(request : Request): 
    return templates.TemplateResponse("notice.html", context={'request' : request}) 
