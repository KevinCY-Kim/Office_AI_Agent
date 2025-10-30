from fastapi import APIRouter, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import File
from typing_extensions import Annotated
from fastapi import Form
router = APIRouter()
templates = Jinja2Templates(directory="templates")
import pandas as pd
import numpy as np
import os
from scipy.stats import shapiro, pearsonr, spearmanr, f_oneway, ttest_ind
from fastapi.responses import StreamingResponse
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import pingouin as pg
from sklearn.preprocessing import LabelEncoder

@router.get("/statistics")
async def statistics(request : Request): 
    return templates.TemplateResponse("statistics.html", context={'request' : request}) 

BASE_DIR = "/home/alpaco/cjw/final/web/Office_AI_Agent/app"
STATIC_DIR = os.path.join(BASE_DIR, ".", "static")        # ✅ app 밖 static/
TEMPLATE_DIR = os.path.join(BASE_DIR, ".", "templates")    # ✅ app 밖 templates/
UPLOAD_PATH = os.path.join(BASE_DIR, ".", "data", "uploads")

os.makedirs(UPLOAD_PATH, exist_ok=True)

# ------------------------------------------------------
# 4️⃣ Static / Templates 등록
# ------------------------------------------------------
router.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATE_DIR)

# ------------------------------------------------------
# 5️⃣ 기본 페이지
# ------------------------------------------------------
@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """기본 홈 화면 렌더링"""
    return templates.TemplateResponse("statistics.html", {"request": request})


# ------------------------------------------------------
# 6️⃣ 파일 업로드
# ------------------------------------------------------
@router.get("/submit")
async def analyze_data_get2():
    return {"submit" : "post 페이지"}

@router.post("/submit")
async def upload_file(file: UploadFile = File(...)):
    """파일 업로드 및 미리보기"""
    try:
        ext = os.path.splitext(file.filename)[-1].lower()
        if ext not in [".csv", ".xlsx", ".xls"]:
            return JSONResponse({"error": "CSV, XLSX만 허용됩니다."}, status_code=400)

        filepath = os.path.join(UPLOAD_PATH, file.filename)
        with open(filepath, "wb") as f:
            f.write(await file.read())

        df = pd.read_csv(filepath) if ext == ".csv" else pd.read_excel(filepath)
        preview = df.head(5).to_dict(orient="records")

        return {
            "filename": file.filename,
            "rows": len(df),
            "columns": list(df.columns),
            "preview": preview,
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ------------------------------------------------------
# 7️⃣ 통계 분석 (/stats)
# ------------------------------------------------------

@router.get("/stats")
def analyze_data_get(request: Request):
    return templates.TemplateResponse("sta_page.html", {"request": request})


# def visualize_ttest(group1, group2, group_names=("Group 1", "Group 2"), value_label="Sales"):
#     """두 그룹의 분포와 평균 차이를 시각화"""
#     data = pd.DataFrame({
#         value_label: np.concatenate([group1, group2]),
#         "Group": [group_names[0]] * len(group1) + [group_names[1]] * len(group2)
#     })
#     plt.figure(figsize=(8,5))
#     sns.violinplot(x="Group", y=value_label, data=data, inner="box", palette="Set2")
#     sns.swarmplot(x="Group", y=value_label, data=data, color=".3", alpha=0.5)
#     plt.title(f"{group_names[0]} vs {group_names[1]} — t-test 분포 비교", fontsize=13)
#     plt.ylabel(value_label)
#     plt.xlabel("")
#     buf = BytesIO()
#     plt.savefig(buf, format="png")
#     buf.seek(0)
#     return StreamingResponse(buf, media_type="image/png")

def visualze_Pearson(group1, group2, value, method_val, files : File):
    filepath = os.path.join(UPLOAD_PATH, files.filename)

    # ---- 2️⃣ 파일 읽기 ----
    if files.filename.endswith(".csv"):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)
    mood_group = (df[df["Product Class"] == "Mood Stabilizers"].groupby(value)["Sales"].sum().reset_index(name="mood_sales"))
    pain_group = (df[df["Product Class"] == "Analgesics"].groupby(value)["Sales"].sum().reset_index(name="pain_sales"))
    merged = pd.merge(mood_group, pain_group, on=value, how="outer")
    merged = merged.fillna(0) # 판매량이 없는 지역은 0 으로 간주 (결측치 대체)

    # 4️. Pearson, Spearman 상관계수 계산
    pearson_corr, pearson_p = pearsonr(merged["mood_sales"], merged["pain_sales"])
    spearman_corr, spearman_p = spearmanr(merged["mood_sales"], merged["pain_sales"])

    print(f"Pearson 상관계수 (r): {pearson_corr:.4f}, P-value: {pearson_p:.4f}")
    print(f"Spearman 상관계수 (ρ): {spearman_corr:.4f}, P-value: {spearman_p:.4f}")

    alpha = 0.05

    def interpret_correlation(corr, p_value, method):
        abs_corr = abs(corr)
        if abs_corr >= 0.7:
            strength = "매우 강함"
        elif abs_corr >= 0.5:
            strength = "강함"
        elif abs_corr >= 0.3:
            strength = "보통"
        else:
            strength = "약함 또는 없음"

        if p_value < alpha:
            significance = "통계적으로 유의미함"
            if corr > 0:
                direction = "양의 상관관계 (Mood ↑, Pain ↑)"
            elif corr < 0:
                direction = "음의 상관관계 (Mood ↑, Pain ↓)"
            else:
                direction = "상관관계 없음"
        else:
            significance = "통계적으로 유의미하지 않음"
            direction = "상관관계 없음"

        result = (f"[{method}] 값: {corr:.4f}, 유의성: {significance}, "
                f"강도: {strength}, 방향: {direction}")
        return result

    print(interpret_correlation(pearson_corr, pearson_p, method_val))
    print(interpret_correlation(spearman_corr, spearman_p, method_val))
    # 5. 그래프
    plt.figure(figsize=(9,7))
    sns.regplot(x="mood_sales", y="pain_sales", data=merged, scatter_kws={"alpha":0.6})
    plt.title(f"Mood Stabilizers vs Analgesics Sales\n"
            f"Pearson r={pearson_corr:.2f}, Spearman ρ={spearman_corr:.2f}")
    plt.xlabel("Mood Stabilizer Sales")
    plt.ylabel("Analgesic Sales")
    plt.grid(True, linestyle="--", alpha=0.4)

    for i in range(len(merged)):
        plt.text(merged['mood_sales'][i], merged['pain_sales'][i], merged['City'][i],
                fontsize=8, ha='right')

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


# def visualze_anova(group1, group2, group_names=("Group 1", "Group 2"), value_label="Sales"):
#     data_dict = {
#     'Product Class': np.random.choice(class_list, 500),
#     'Sales': np.concatenate([
#         np.random.normal(loc=10000, scale=3000, size=150),
#         np.random.normal(loc=12000, scale=4000, size=150),
#         np.random.normal(loc=9000, scale=2500, size=100),
#         np.random.normal(loc=15000, scale=5000, size=100)
#     ]).clip(min=1000)}

#     df = pd.DataFrame(data_dict)
#     # 시각화
#     plt.figure(figsize=(10,6))
#     sns.boxplot(x='Product Class', y='Sales', data=df, palette='Set2')
#     sns.stripplot(x='Product Class', y='Sales', data=df, color='black', alpha=0.5, size=3, jitter=True)

#     mean_sales = df.groupby('Product Class')['Sales'].mean().reset_index()
#     plt.scatter(mean_sales['Product Class'], mean_sales['Sales'], color='red', marker='D', s=100, label='Mean')
#     anova_type = "Welch’s One-way ANOVA"
#     plt.title(f"{anova_type} (p={p_main:.4f})")
#     plt.xlabel("Product Class")
#     plt.ylabel("Sales")
#     plt.legend()
#     plt.grid(axis='y', linestyle='--')
#     buf = BytesIO()
#     plt.savefig(buf, format="png")
#     buf.seek(0)
#     return StreamingResponse(buf, media_type="image/png")
    

@router.post("/stats", response_class=HTMLResponse)
async def analyze_data(
    file: UploadFile = File(...),
    analysis_type: str = Form(...),
    value_col: str = Form(None),
    alpha: float = Form(0.05),
    group_col_ttest: str = Form(None),
    group1: str = Form(None),
    group2: str = Form(None),
    group_col_one: str = Form(None),
    group_col_two: str = Form(None),
    factors_two: str = Form(None),
    within: str = Form(None),
    subject: str = Form(None),
    var1: str = Form(None),
    var2: str = Form(None),
    method: str = Form(None),
):
    print("📥 /stats 라우트 호출됨")

    try:
        # ---- 1️⃣ 파일 저장 ----
        filepath = os.path.join(UPLOAD_PATH, file.filename)
        with open(filepath, "wb") as f:
            f.write(await file.read())

        # ---- 2️⃣ 파일 읽기 ----
        if file.filename.endswith(".csv"):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)

        print("📁 파일명:", file.filename)
        print("📊 분석유형:", analysis_type)
        print("🎯 Value 변수:", value_col)
        print("α (유의수준):", alpha)

        result = {}

        # ---- 3️⃣ 분석 수행 ----
        if analysis_type == "ttest" and group_col_ttest and value_col:
            unique = df[group_col_ttest].dropna().unique()
            if len(unique) >= 2:
                g1 = df[df[group_col_ttest] == unique[0]][value_col]
                g2 = df[df[group_col_ttest] == unique[1]][value_col]
                print("g1",g1)
                print("g2",g2)
                t, p = ttest_ind(g1, g2, equal_var=False)
                result = {"분석유형": "t-test", "t값": round(t, 3), "p값": round(p, 5)}
                # return visualize_ttest(round(t, 3), round(p, 5), group_names=unique, value_label=value_col)

        elif analysis_type == "anova_one" and group_col_one and value_col:
            groups = [g[value_col].dropna() for _, g in df.groupby(group_col_one)]
            f, p = f_oneway(*groups)
            result = {"분석유형": "One-way ANOVA", "F값": round(f, 3), "p값": round(p, 5)}
            # return visualze_anova(file)

        elif analysis_type == "correlation" and var1 and var2:
            print(var1, var2)
            x, y = df[var1].dropna(), df[var2].dropna()
            p1, p2 = shapiro(x[:5000]).pvalue, shapiro(y[:5000]).pvalue
            method = "Pearson" if (p1 >= alpha and p2 >= alpha) else "Spearman"
            r, p = (pearsonr(x, y) if method == "Pearson" else spearmanr(x, y))
            result = {
                "분석유형": "Correlation",
                "방법": method,
                "상관계수(r)": round(r, 4),
                "p값": round(p, 5),
                "정규성검정": f"p1={round(p1,5)}, p2={round(p2,5)}",
            }
            return visualze_Pearson(var1, var2, value_col, method, file)
        else:
            result = {"오류": "지원되지 않는 분석 유형이거나 인자가 부족합니다."}

        # # ---- 4️⃣ HTML 형식으로 반환 ----
        # html_content = "<h3>📊 분석 결과</h3>"
        # html_content += "<table border='1' cellpadding='6' style='border-collapse:collapse;'>"
        # for key, value in result.items():
        #     html_content += f"<tr><th style='background:#f0f0f0;text-align:left;padding:5px;'>{key}</th><td style='padding:5px;'>{value}</td></tr>"
        # html_content += "</table>"

        # return HTMLResponse(content=html_content)

    except Exception as e:
        error_html = f"""
        <div style='color:red;'>
            <h3>❌ 오류 발생</h3>
            <p>{str(e)}</p>
        </div>
        """
        return HTMLResponse(content=error_html)