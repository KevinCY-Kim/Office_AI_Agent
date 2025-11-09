# Office AI Agent (FastAPI 기반)

## 🚀 프로젝트 실행 방법

1.  **가상환경 생성 및 활성화**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```

2.  **필요 패키지 설치**
    ```bash
    pip install -r requirements.txt
    ```

3.  **FastAPI 서버 실행**
    ```bash
    uvicorn app.main:app --reload
    ```
    서버 실행 후 브라우저에서 `http://127.0.0.1:8001`으로 접속하세요.

4.  **폴더 트리구조**
    ```bash
    Office_AI_Agent_System/
    ├── .env
    ├── .gitignore
    ├── README.md
    ├── requirements.txt
    ├── run.sh                      # uvicorn 실행 스크립트
    │
    ├── app/
    │   ├── main.py                 # FastAPI 진입점 (uvicorn app.main:app)
    │   ├── core/                   # 핵심 설정 및 초기화
    │   │   ├── config.py           # 환경변수, DB, 모델 로드
    │   │   ├── security.py         # JWT 토큰, 인증 관련
    │   │   └── logger.py           # 로깅 설정
    │   │
    │   ├── models/                 # 데이터베이스 ORM 모델 (SQLAlchemy or Tortoise)
    │   │   ├── user.py             # 사용자 (회원가입, 로그인)
    │   │   ├── document.py         # 문서생성 관련 데이터
    │   │   ├── regulation.py       # 사내규정 데이터
    │   │   └── notice.py           # 공지 게시글 모델
    │   │
    │   ├── schemas/                # Pydantic 스키마
    │   │   ├── user_schema.py
    │   │   ├── document_schema.py
    │   │   ├── regulation_schema.py
    │   │   └── notice_schema.py
    │   │
    │   ├── services/               # 비즈니스 로직
    │   │   ├── document_service.py # 문서 생성기 (LLM 연동)
    │   │   ├── regulation_service.py # 규정 검색 (RAG / embedding)
    │   │   ├── notice_service.py   # 공지 이미지 생성 (ComfyUI 연동)
    │   │   └── user_service.py     # 회원가입, 로그인, 인증
    │   │
    │   ├── routers/                # API 라우팅
    │   │   ├── auth_router.py      # 로그인/회원가입
    │   │   ├── document_router.py  # 문서생성기 챗봇
    │   │   ├── regulation_router.py # 사내규정검색 챗봇
    │   │   ├── notice_router.py    # 공지게시판 챗봇
    │   │   └── main_router.py      # 홈 및 공통 페이지 라우팅
    │   │
    │   ├── templates/              # Jinja2 템플릿 (EJS 대체)
    │   │   ├── base.html           # 모든 페이지의 공통 상단/하단
    │   │   ├── layout/
    │   │   │   ├── top.html        # 탑 및 로그인/로그아웃 상태 반영
    │   │   │   └── bottom.html     # footer, 링크
    │   │   ├── index.html          # 메인 페이지
    │   │   ├── login.html          # 로그인
    │   │   ├── register.html       # 회원가입
    │   │   ├── document.html       # 문서 생성기
    │   │   ├── regulation.html     # 사내규정 검색
    │   │   ├── notice.html         # 공지 목록
    │   │   ├── notice_detail.html  # 공지 상세
    │   │   └── error.html          # 에러 페이지
    │   │
    │   ├── static/                 # 정적 자원 (Tailwind + JS)
    │   │   ├── css/
    │   │   │   └── main.css
    │   │   ├── js/
    │   │   │   ├── main.js
    │   │   │   ├── document.js     # 문서생성기 인터랙션
    │   │   │   ├── regulation.js   # 규정검색 HTMX 요청
    │   │   │   └── notice.js       # 공지 관리
    │   │   └── img/
    │   │       └── logo.png
    │   │
    │   └── db/
    │       └── session.py          # DB 연결 세션 (SQLAlchemy)
    │
    ├── scripts/                    # 데이터 초기화 / 관리 스크립트
    │   ├── init_db.py
    │   ├── load_regulations.py
    │   └── sync_notices.py
    │
    ├── logs/
    │   └── app.log
    │
    └── docker/
        ├── Dockerfile # 배포방법 추후 선정
        └── nginx.conf # 배포방법 추후 선정
    ```