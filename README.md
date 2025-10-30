# Office AI Agent

## 📝 프로젝트 개요
- Office 문서 자동화, 사내 규정 검색, 통계/보고서 작성, 공지/문서 이미지 자동 생성 등 기업 업무의 AI 기반 자동화 및 효율화를 목표로 하는 FastAPI 기반 시스템
- 실제 현업에 맞춘 **모듈화 구조**, 데이터·보안·UI/UX 표준 엄격 준수

## 🛠️ 기술 스택 및 주요 라이브러리
- Python 3.10+, FastAPI, Jinja2, Uvicorn, SQLAlchemy, Pydantic
- 기타: numpy, pandas, docx, custom GPT/Embedding, HTMX 등

## 📁 폴더/파일 구조 (주요 파일 중심, 실제 경로 기준 정정)
```
Office_AI_Agent/
├── app/
│   ├── main.py                # FastAPI 진입점
│   ├── core/
│   │   ├── config.py
│   │   └── security.py
│   ├── db/
│   │   ├── database.py
│   │   └── models.py
│   ├── routers/
│   │   ├── auth.py
│   │   ├── document_generator.py
│   │   ├── regulations.py
│   │   ├── notices.py
│   │   └── statistics.py
│   ├── services/
│   │   ├── auth_service.py
│   │   ├── doc_generator_service.py
│   │   ├── notice_service.py
│   │   └── regulations_service.py
│   ├── schemas/
│   │   └── user.py
│   ├── templates/            # Jinja2 템플릿
│   ├── static/               # 정적 리소스
│   └── data/                 # 업로드/출력/DB파일 등
├── chunking/  parsing/  report/ (AI 문서/리포트 파싱 및 분할)
├── scripts/
├── tests/
└── README.md  requirements.txt
```

## 🚦 설치 및 실행 방법
1. **가상환경 생성 및 활성화**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```
2. **필수 패키지 설치**
    ```bash
    pip install -r requirements.txt
    ```
3. **FastAPI 서버 실행**
    ```bash
    uvicorn app.main:app --reload --port 8001
    ```
    *→ 브라우저에서 http://localhost:8001 접속*

## ✨ 주요 기능
- **AI 문서 생성:** LLM 기반 자동 문서/보고서 생성 (Docx 등)
- **사내 규정 RAG 검색:** 문서 임베딩 · RAG로 사내 규정, 문서 내용 추출·검색
- **공지/문서 자동 이미지화:** 텍스트→이미지 생성 (ComfyUI 등 외부 연동)
- **통계/보고서 시각화:** 업로드 파일 데이터 통계 분석 및 시각화
- **JWT 기반 인증**(회원/권한별 관리), 업로드/다운로드 등 부가 관리 기능

## 🏗️ 설계/코딩/보안 원칙
- **SOLID 원칙**, 단일 책임 분리: 한 클래스·모듈은 하나의 역할만 담당
- **SQL 인젝션 방지**: DB Query는 반드시 ORM 또는 Prepared Statement 사용, Raw 문자열 금지
- **상태 관리:** 컴포넌트 상태 Flux/Redux 또는 Context-API 패턴 엄격 적용
- **API/입력검증:** 모든 외부 입력은 유효성 검증 & 상세 예외 핸들링/로깅 기본
- **재사용/중복방지:** 신규 기능 작성 전 기존 UI 컴포넌트(/src/components/ui), 유틸(/src/utils) 사전 확인
- **네이밍/코딩규칙:** 의도 명확·일관된 네이밍, 작은 변경세트 지향(기존 패턴 최대 활용)
- **보안/유저 피드백:** try-catch → 예측가능한 사용자 안내와 서버 로깅 체계 구현

## 🔗 참고 문서 및 가이드
- [FastAPI 공식문서](https://fastapi.tiangolo.com/ko/)
- [SQLAlchemy ORM 사용법](https://docs.sqlalchemy.org/)
- [Pydantic 문서](https://docs.pydantic.dev/)
- [Flux 패턴 가이드](https://facebook.github.io/flux/docs/in-depth-overview/)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [문서 자동화/파싱 관련 튜토리얼](https://realpython.com/python-docx/)

---
### 문의 및 협업
- 추가 문의/이슈는 [GitHub Issue] 또는 팀 내 지정 채널 이용