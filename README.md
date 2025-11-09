# Office AI Agent (FastAPI 기반)

## 1. 프로젝트 소개
- **Office AI Agent**는 FastAPI 기반의 R&D 문서 자동생성, 사내 규정 질의·챗봇, 통계 분석, 공지 관리 등 AI 기반 사무자동화를 지원하는 통합 Web 시스템입니다.
- AI/LLM, RAG, 통계분석, 사내규정 관리 등 다양한 업무 프로세스를 일관되게 자동화합니다.

## 2. 주요 기능 요약
- **R&D 문서 자동 생성:** 입력값 및 데이터 기반 보고서 문서 자동 작성/다운로드 지원.
- **사내 규정 질의 챗봇:** RAG, 임베딩 활용 실시간 사내 규정 검색/해설/답변.
- **통계 분석 자동화:** 파일 업로드/통계 분석(T-test, ANOVA, Correlation 등), 시각화 제공.
- **공지 및 인증:** 공지 게시 및 회원 관리(로그인/회원가입/인증/JWT)

## 3. 기술 스택 및 주요 라이브러리
- **Backend:** FastAPI, Uvicorn, Jinja2, Python-dotenv, Passlib(bcrypt), SQLAlchemy, Pydantic
- **AI/RAG:** transformers, sentence-transformers, scikit-learn, PyTorch 등
- **프론트/정적 자원:** Jinja2 HTML Templates, Tailwind CSS, JS, HTMX
- **기타:** Dockerfile(옵션), 로그, 각종 데이터 처리 스크립트
- [FastAPI 공식문서](https://fastapi.tiangolo.com/ko/) / [SQLAlchemy](https://docs.sqlalchemy.org/ko/latest/) / [Jinja2](https://jinja.palletsprojects.com/ko/latest/)

## 4. 폴더 구조(핵심)
```bash
Office_AI_Agent/
├── README.md
├── requirements.txt
├── app/
│   ├── main.py                # FastAPI 앱 진입점
│   ├── routers/               # 모든 API 라우트
│   ├── services/              # 비즈니스 로직 및 AI 서비스
│   ├── schemas/               # Pydantic/모델 스키마
│   ├── templates/             # Jinja2 HTML 템플릿
│   ├── static/                # CSS/JS/이미지 등
│   ├── db/                    # DB 세션 관리 등
│   ├── ...
├── data/, report/, scripts/, tests/    # 참고 데이터·통계·실행 스크립트·테스트
├── Dockerfile, run.sh, .env 예시 (옵션)
```

## 5. 환경 구성 및 설치
1. Python 3.9+ 및 venv 권장
2. 가상환경 생성 및 진입
   ```bash
   python -m venv venv
   source venv/bin/activate  # (Windows: venv\Scripts\activate)
   ```
3. 라이브러리 설치
   ```bash
   pip install -r requirements.txt
   ```
4. (선택) .env 파일 설정, DB 경로/환경 지정

## 6. 실행 방법
```bash
uvicorn app.main:app --reload
```
- 기본 접속 URL: http://127.0.0.1:8001/
- 포트 변경 가능, main.py 참고

## 7. 주요 엔드포인트/화면(예시)
- `/` : 메인화면(Jinja2)
- `/auth/login`, `/auth/register` : 로그인·회원가입
- `/document-generator/` : 문서 생성 챗봇·파일 첨부 가능
- `/regulations/` : 사내 규정 챗봇/검색
- `/notices/` : 사내 공지/공지작성
- `/statistics/`, `/statistics/stats` : 통계 분석 및 시각화

## 8. 기능별 사용법 (예시)
- 문서 생성: 필수 정보 입력 + (선택) 파일 첨부 → 자동 보고서(docx) 생성 및 다운로드
- 규정 챗봇: 질의 입력 → RAG 기반 검색/FAQ/답변 제공
- 통계 분석: 데이터 파일(CSV/XLSX) 업로드 → 변수/분석 방식 선택 → 결과 표/그래프로 출력

## 9. 데이터/모델 구조 참고
- Pydantic 스키마 및 주요 DB 모델(예, User/Document/Notice)
- 상세 구조: `app/schemas/`, `app/db/` 참고

## 10. 운영 및 배포(옵션)
- Dockerfile, nginx.conf, run.sh 등 운영 자동화 파일 예시 포함(커스텀 필요)
- logs/ 폴더: 앱 실행 로그 저장 위치
- 실제 배포 방식 확정 전, 개발/테스트 환경에서 우선적으로 활용 권장
- FastAPI 운영 참고([링크](https://fastapi.tiangolo.com/deployment/))

## 11. 오류 처리/보안
- 모든 외부 입력에 Pydantic validation, 오류 시 JSON/HTML 피드백 제공
- JWT 등 인증 체계 적용, 로그 기록
- 세부 보안 설정은 운영환경 확정 시 보완 필요

## 12. 기여 및 유지보수 가이드
- 함수·클래스·파일명은 명확하게, 기존 컨벤션 준수
- 확장 시 UI/유틸 재활용 적극 검토 (`/components/`, `/utils/` 활용 권장)
- 기능별 모듈화와 SOLID 원칙(단일책임 등) 유지
- 장애/개선사항은 issues 또는 별도 문서로 관리

## 13. 참고자료 및 링크
- [FastAPI 공식 카테고리](https://fastapi.tiangolo.com/ko/)
- [SQLAlchemy](https://docs.sqlalchemy.org/ko/latest/)
- [Jinja2](https://jinja.palletsprojects.com/ko/latest/)
- [Tailwind CSS](https://tailwindcss.com/)
- [Pandas](https://pandas.pydata.org/)
- (이외 추가 자료 및 샘플 데이터/문서 경로 필요 시 내용 보강)

---

**본 문서는 실제 코드/구조/사용 가이드, 최신 기술 기준을 최대한 반영하였으며, 운영환경이 확정되는 경우 배포·보안에 관한 상세 안내를 추가 작성하세요.**