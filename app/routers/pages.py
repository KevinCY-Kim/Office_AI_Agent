from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import Optional, List
from typing import TypedDict
from typing_extensions import Annotated
import operator
from fastapi import Form

router = APIRouter()
templates = Jinja2Templates(directory="templates")

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("auth/login.html", {"request": request})

@router.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("auth/register.html", {"request": request})

@router.get("/comfy", response_class=HTMLResponse)
async def comfy(request: Request):
    return templates.TemplateResponse("comfy.html", {"request": request})

@router.get("/comfy_prompt", response_class=HTMLResponse)
async def comfy_prompt(request: Request):
    return templates.TemplateResponse("comfy_prompt.html", {"request": request})

#comfy 관련 코드====================================================================


import torch

from langchain_community.llms import Ollama  
from langchain_core.prompts import PromptTemplate 
from langgraph.graph import StateGraph, END  
from typing import TypedDict, Annotated, List, Dict
import operator
import pandas as pd  

import sqlite3
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnableLambda

style = ""
scale = 8.5
DB_PATH = "/home/alpaco/cjw/final/web/Office_AI_Agent/app/db/prompts.db"
TABLE   = "prompts"
CHROMA_DIR = "/home/alpaco/cjw/final/web/Office_AI_Agent/app/db/chroma_prompts"

POSTER_HINTS = [
    "포스터", "배경", "배경만", "백그라운드", "레이아웃", "테두리",
    "보더", "프레임", "패턴", "추상", "그래픽", "벽지", "poster", "background"
]
MODEL_HINTS = [
    "모델", "인물", "여성", "남성", "남자", "여자", "사람", "뷰티", "face", "portrait", "model"
]

def detect_kind_heuristic(q: str) -> str:
    global scale, style
    ql = q.lower()
    if any(h.lower() in ql for h in POSTER_HINTS):
        style =", A clean and minimal poster design, centered layout, white background, simple modern sans-serif font, high contrast text, balanced whitespace, soft shadows, subtle gradients, elegant composition, professional look, flat design style, no clutter, design for information clarity, only poster background"
        return "poster"
    if any(h.lower() in ql for h in MODEL_HINTS):
        scale = 10.0
        style =", a stunning, beautiful Korean, elegant facial features, flawless skin, V-line jaw, straight black hair, natural makeup, graceful pose, cinematic lighting, ultra-realistic, high-definition photo, DSLR, looking slightly off-camera, wearing fashionable modern Korean style clothing, serene expression, 1other"
        return "model"
    return "poster"

def fetch_prompts_from_sql(kind: str, limit: int | None = None) -> list[str]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    if limit and limit > 0:
        cur.execute(f"SELECT text FROM {TABLE} WHERE kind=? AND text IS NOT NULL LIMIT ?;", (kind, limit))
    else:
        cur.execute(f"SELECT text FROM {TABLE} WHERE kind=? AND text IS NOT NULL;", (kind,))
    rows = cur.fetchall()
    conn.close()
    return [r[0].strip() for r in rows if r[0] and str(r[0]).strip()]


emb = OllamaEmbeddings(model="nomic-embed-text") 
vectordb = Chroma(
    embedding_function=emb,
    collection_name="prompt_tags",
    persist_directory=CHROMA_DIR
)
retriever = vectordb.as_retriever(search_kwargs={"k": 8}) 

class AgentState(TypedDict):   
    query: Annotated[List[str], operator.add]
    kind: Annotated[List[str], operator.add] 
    lore: Annotated[List[str], operator.add] 
    tags_list: Annotated[List[str], operator.add] 
    result: str  
    result2: str  


llm = Ollama(model="gemma3:4b") 

def retrieve_agent(state: AgentState):
    q = state["query"][0]
    kind = detect_kind_heuristic(q)
    docs = vectordb.similarity_search(q, k=12, filter={"kind": kind})
    vec_hits = [d.page_content.strip() for d in docs if d.page_content and d.page_content.strip()]
    sql_hits = fetch_prompts_from_sql(kind, limit=50)
    merged = list(dict.fromkeys(vec_hits + sql_hits))
    MAX_ITEMS = 200
    merged = merged[:MAX_ITEMS]
    tags_list = ", ".join(merged)
    return {**state, "kind": [kind, kind], "tags_list": [tags_list, tags_list]}



extractor_prompt = PromptTemplate.from_template("""
                                                사용자의 요구 사항에 따라 내용을 증가시켜 주십시오.
                                                오직 요구사항을 증가시키는데 초점을 맞춰서 설명하세요. 또한 당신은
                                                한국어로 설명하여야만 합니다.
                                                요구: {query}
                                                """) 


def extractor_agent(state: AgentState):  
    chain = extractor_prompt | llm
    lore = chain.invoke({
                    "query": state["query"]
                })  
    return {**state, "lore": [lore.strip(), lore.strip()]}  

result_prompt = PromptTemplate.from_template("""
                                                Extract sentences highly relevant to the requirements 
                                                and generate Positive Positive prompts for creating illustrations and backgrounds. When generating Positive prompts, reinforce them by adding sentences and words suitable for the requirements.
                                                Separate Positive prompts with commas and write them in English.
                                                Exclude all words except “Positive prompt”.
                                                Additionally, since only the poster background will be drawn, the Positive prompt must not contain the words “text” or “string”. That is, the Positive prompt must not contain any characters.
                                                Requirements: {lore}
                                                tag list: {tags_list}
                                                """) 


def result_agent(state: AgentState):  
    chain = result_prompt | llm 
    tags = chain.invoke({
                    "lore": state["lore"],
                    "tags_list": state["tags_list"]
                })  
    return {**state, "result": tags.strip()}  

result_prompt2 = PromptTemplate.from_template("""
                                                Extract sentences highly relevant to the requirements
                                                and generate Negative prompts for creating illustrations and backgrounds. When generating Negative prompts, reinforce them by adding sentences and words suitable for the requirements.
                                                Separate Negative prompts with commas and write them in English.
                                                Exclude all words except “Negative prompt”.
                                                Additionally, since only the poster background will be drawn, the Negative prompt must not contain the words “text” or “string”. That is, the Negative prompt must not contain any characters.
                                                Requirements: {lore}
                                                tag list: {tags_list}
                                                """) 


def result_agent2(state: AgentState):  
    chain = result_prompt2 | llm 
    tags = chain.invoke({
                    "lore": state["lore"],
                    "tags_list": state["tags_list"]
                })  
    return {**state, "result2": tags.strip()}

from langgraph.graph import StateGraph  

graph = StateGraph(AgentState) 
graph.add_node("retrieve", retrieve_agent)
graph.add_node("extractor", extractor_agent)
graph.add_node("result", result_agent)
graph.add_node("result2", result_agent2)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "extractor")
# graph.set_entry_point("extractor")
graph.add_edge("extractor", "result")
graph.add_edge("extractor", "result2")
graph.add_edge("result", END)
graph.add_edge("result2", END)

@router.post("/comfy_prompt") 
def comfy_post(request : Request, prompt : Annotated[str, Form()]): 
    app = graph.compile() 
    app.get_graph().print_ascii()  
    result1 = app.invoke({"query": [prompt, prompt]}) 
    return templates.TemplateResponse("comfy_prompt.html", {"request": request, "po_pro" : result1["result"]+style, "ne_pro" : result1["result2"]+", NSFW"})

