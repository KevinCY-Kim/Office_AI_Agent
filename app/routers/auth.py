from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing_extensions import Annotated
from fastapi import Form
router = APIRouter()
templates = Jinja2Templates(directory="templates")

@router.post("/login")
async def login_post(request : Request, id : Annotated[str, Form()] , pwd : Annotated[str, Form()]): 
    print(id, pwd) 
    

@router.post("/register")
async def register_post(request : Request, id : Annotated[str, Form()] , pwd : Annotated[str, Form()], email : Annotated[str, Form()] , phone : Annotated[str, Form()], office : Annotated[str, Form()]):
    print(id, pwd) 
    