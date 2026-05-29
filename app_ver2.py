"""
마케팅 기여도 분석 도구 (v3)
- 일간데이터 헤더: 2행 그룹헤더 + 컬럼명 (header_rows=2)
- 컬럼 위치 (검증 완료):
    A=날짜, B=브랜드, C=제품
    O=메타제외 매출, L=메타제외 판매수량
    T=1~3위조회수(바이럴), X=숏폼조회수
- 사이다: 별도 시트에서 도메인 lag으로 일자 귀속
    E=브랜드, F=제품, G=게시일자, I=채널, K=URL, L=조회수
- lag 정책:
    바이럴: lag 없음 (당일)
    숏폼: 당일 + 직전 6일 (총 7일)
    사이다: URL 도메인별 (cafe/kin.naver=0, blog.naver/pann.nate=1, 그 외=제외)
- 오가닉: 매출=브랜드별, 수량=제품별 (단일/요일별/시즌보정/추세/추세+요일 자동 선택)
- 가중치 0 매체는 엑셀/화면에 0으로 표시 ('-' 사용 금지)
- 탭 5개: 매출 분석 / 수량 분석 / 데이터 개수 / 용어 정의 / 산정 기준
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from io import BytesIO
import unicodedata
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 페이지 설정
# ==========================================
st.set_page_config(
    page_title="마케팅 기여도 분석",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 색상
BG = "#0F1419"
SURFACE = "#1A2332"
SURFACE_2 = "#2A3548"
BORDER = "#3D4A5F"
TEXT = "#F1F5F9"
TEXT_DIM = "#CBD5E1"
TEXT_MUTED = "#94A3B8"
NAVY = "#1E3A5F"
NAVY_LIGHT = "#3B5B85"
ACCENT = "#A5C9E8"
ACCENT_DEEP = "#5B8FBF"
HIGHLIGHT_BG = "#1E3A5F"
SUCCESS_BG = "#264870"

MIN_DATA_DAYS = 30
MIN_SOLO_DAYS = 20

MEDIA_ORDER = ['바이럴조회수', '숏폼조회수', '사이다조회수']
SHORTFORM_LAG_MAX = 6

MEDIA_LAG_POLICY = {
    '바이럴조회수': 'none',
    '숏폼조회수': 'fixed6',
    '사이다조회수': 'preshift',
}


def cida_domain_lag(url):
    u = str(url).lower()
    if 'cafe' in u:
        return 0
    if 'kin.naver' in u:
        return 0
    if 'blog.naver' in u:
        return 1
    if 'pann.nate' in u:
        return 1
    return None


# CSS
st.markdown(f"""
<style>
    .stApp {{ background-color: {BG}; }}
    .stApp, .stApp * {{ color: {TEXT} !important; }}
    h1, h2, h3, h4, h5, h6 {{ color: {TEXT} !important; font-weight: 600; }}
    .stMarkdown h4 {{
        font-size: 1.05rem !important;
        margin-top: 1.2rem !important;
        margin-bottom: 0.6rem !important;
        color: {ACCENT} !important;
    }}
    section[data-testid="stSidebar"] {{
        background-color: {SURFACE};
        border-right: 1px solid {BORDER};
    }}
    section[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}
    section[data-testid="stSidebar"] [data-testid="stCodeBlock"],
    section[data-testid="stSidebar"] pre {{
        background-color: {BG} !important;
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 14px !important;
        margin: 8px 0;
    }}
    section[data-testid="stSidebar"] [data-testid="stCodeBlock"] code,
    section[data-testid="stSidebar"] pre code,
    section[data-testid="stSidebar"] [data-testid="stCodeBlock"] * {{
        background-color: transparent !important;
        color: {TEXT} !important;
        font-size: 0.82rem !important;
        line-height: 1.8 !important;
        white-space: pre !important;
    }}
    [data-testid="stMetric"] {{
        background-color: {SURFACE};
        padding: 18px 20px;
        border-radius: 10px;
        border: 1px solid {BORDER};
    }}
    [data-testid="stMetric"] [data-testid="stMetricLabel"] {{
        color: {TEXT_MUTED} !important;
        font-size: 0.85rem !important;
    }}
    [data-testid="stMetricValue"] {{
        color: {TEXT} !important;
        font-size: 1.8rem !important;
        font-weight: 600 !important;
    }}
    .stButton > button {{
        background-color: {ACCENT};
        color: {BG} !important;
        border: 1px solid {ACCENT};
        font-weight: 700;
        padding: 12px 32px;
        border-radius: 8px;
        font-size: 1rem;
    }}
    .stButton > button:hover {{ background-color: {TEXT}; border-color: {TEXT}; color: {BG} !important; }}
    .stButton > button * {{ color: {BG} !important; }}
    .stDownloadButton > button {{
        background-color: {ACCENT};
        color: {BG} !important;
        border: 1px solid {ACCENT};
        font-weight: 700;
        padding: 12px 32px;
        border-radius: 8px;
        font-size: 1rem;
    }}
    .stDownloadButton > button:hover {{ background-color: {TEXT}; border-color: {TEXT}; color: {BG} !important; }}
    .stDownloadButton > button * {{ color: {BG} !important; }}
    [data-testid="stFileUploader"] {{
        background-color: {NAVY};
        padding: 24px;
        border-radius: 10px;
        border: 1px dashed {NAVY_LIGHT};
    }}
    [data-testid="stFileUploader"] *,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] div {{ color: {TEXT} !important; }}
    [data-testid="stFileUploader"] button {{
        background-color: {ACCENT} !important;
        color: {BG} !important;
        border: 1px solid {ACCENT} !important;
        font-weight: 600;
    }}
    [data-testid="stFileUploader"] button * {{ color: {BG} !important; }}
    [data-testid="stFileUploader"] button:hover {{ background-color: {TEXT} !important; border-color: {TEXT} !important; }}
    [data-testid="stFileUploaderFile"],
    [data-testid="stFileUploaderFileData"],
    [data-testid="stFileUploaderFileName"] {{
        background-color: {SURFACE_2} !important;
        color: {TEXT} !important;
    }}
    [data-testid="stFileUploaderFile"] *,
    [data-testid="stFileUploaderFileData"] *,
    [data-testid="stFileUploaderFileName"] * {{
        color: {TEXT} !important;
        opacity: 1 !important;
    }}
    [data-testid="stFileUploaderFile"] svg,
    [data-testid="stFileUploaderDeleteBtn"] svg {{
        fill: {TEXT} !important;
        color: {TEXT} !important;
        opacity: 1 !important;
    }}
    [data-testid="stFileUploaderDeleteBtn"] {{ background-color: transparent !important; }}
    [data-testid="stFileUploaderDeleteBtn"]:hover {{ background-color: {NAVY_LIGHT} !important; }}
    [data-testid="stFileUploaderDropzoneInstructions"] {{ display: none !important; }}
    [data-testid="stFileUploaderDropzone"] {{
        justify-content: center !important;
        padding: 12px !important;
    }}
    [data-testid="stElementToolbar"],
    [data-testid="stElementToolbarButton"],
    [data-testid="stElementToolbarButtonGroup"],
    [data-testid="stDataFrameToolbar"] {{
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }}
    [data-testid="stElementToolbar"] button,
    [data-testid="stElementToolbarButton"] button,
    [data-testid="stElementToolbarButtonGroup"] button,
    [data-testid="stDataFrameToolbar"] button {{
        background-color: {SURFACE_2} !important;
        border: none !important;
        color: {TEXT} !important;
        width: 32px !important;
        height: 32px !important;
        padding: 6px !important;
        margin: 2px !important;
        border-radius: 50% !important;
        opacity: 0.7 !important;
    }}
    [data-testid="stElementToolbar"] button:hover,
    [data-testid="stElementToolbarButton"] button:hover,
    [data-testid="stElementToolbarButtonGroup"] button:hover,
    [data-testid="stDataFrameToolbar"] button:hover {{
        background-color: {NAVY_LIGHT} !important;
        opacity: 1 !important;
    }}
    [data-testid="stElementToolbar"] svg,
    [data-testid="stElementToolbarButton"] svg,
    [data-testid="stElementToolbarButtonGroup"] svg,
    [data-testid="stDataFrameToolbar"] svg {{
        filter: brightness(0) invert(1) !important;
        opacity: 1 !important;
    }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {BORDER}; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent !important;
        color: {TEXT_MUTED} !important;
        font-weight: 500;
        padding: 12px 20px;
        border: none;
        border-bottom: 2px solid transparent;
        border-radius: 0;
    }}
    .stTabs [data-baseweb="tab"]:hover {{ color: {TEXT} !important; }}
    .stTabs [aria-selected="true"] {{
        color: {ACCENT} !important;
        border-bottom: 2px solid {ACCENT} !important;
    }}
    [data-testid="stDataFrame"] {{
        background-color: {SURFACE};
        border-radius: 10px;
        border: 1px solid {BORDER};
    }}
    div[data-testid="stAlert"] {{
        background-color: transparent !important;
        border: none !important;
        padding: 0 !important;
        box-shadow: none !important;
    }}
    div[data-testid="stAlertContainer"],
    [data-baseweb="notification"],
    div[data-testid="stAlert"] > div {{
        background-color: {SUCCESS_BG} !important;
        border: 1px solid {NAVY_LIGHT} !important;
        border-left: 4px solid {ACCENT} !important;
        border-radius: 8px !important;
        color: {TEXT} !important;
    }}
    div[data-testid="stAlert"] *,
    div[data-testid="stAlertContainer"] *,
    [data-baseweb="notification"] * {{
        color: {TEXT} !important;
        background-color: transparent !important;
    }}
    div[data-testid="stAlert"] svg {{ fill: {ACCENT} !important; }}
    [data-testid="stCaptionContainer"] * {{ color: {TEXT_MUTED} !important; }}
    [data-testid="stExpander"] {{
        background-color: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 8px;
    }}
    [data-testid="stExpander"] summary {{ color: {TEXT} !important; }}
    hr {{ border-color: {BORDER} !important; }}
    .stMarkdown table {{
        background-color: {SURFACE};
        border-collapse: collapse;
        border-radius: 8px;
        overflow: hidden;
    }}
    .stMarkdown th {{
        background-color: {NAVY};
        color: {TEXT} !important;
        font-weight: 600;
        padding: 10px 14px;
        border: 1px solid {BORDER};
    }}
    .stMarkdown td {{
        color: {TEXT} !important;
        padding: 10px 14px;
        border: 1px solid {BORDER};
        background-color: {SURFACE};
    }}
    .stMarkdown code {{
        background-color: {SURFACE_2} !important;
        color: {ACCENT} !important;
        padding: 2px 8px !important;
        border-radius: 4px;
        font-size: 0.85rem;
        border: 1px solid {BORDER};
    }}
    .stMarkdown pre {{
        background-color: {SURFACE_2} !important;
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 14px 18px !important;
    }}
    .stMarkdown pre code {{
        background-color: transparent !important;
        color: {TEXT} !important;
        border: none !important;
        padding: 0 !important;
    }}
    .stMarkdown pre * {{ color: {TEXT} !important; }}
    [data-testid="stCodeBlock"] {{
        background-color: {SURFACE_2} !important;
        border: 1px solid {BORDER};
        border-radius: 8px;
    }}
    [data-testid="stCodeBlock"] * {{ color: {TEXT} !important; background-color: transparent !important; }}
    .header-brand-tag {{
        display: inline-block;
        background-color: {NAVY};
        color: {TEXT} !important;
        font-weight: 600;
        font-size: 0.8rem;
        padding: 4px 12px;
        border-radius: 12px;
        letter-spacing: 0.5px;
    }}
    [data-testid="stStatusWidget"] {{ display: none !important; }}
    .stDeployButton {{ display: none !important; }}
    footer {{ visibility: hidden !important; }}
    header[data-testid="stHeader"] {{ background-color: {BG} !important; }}
    header[data-testid="stHeader"] * {{ color: {TEXT} !important; }}
    header[data-testid="stHeader"] svg {{ fill: {TEXT} !important; }}
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"],
    button[kind="header"] {{ background-color: {SURFACE_2} !important; color: {TEXT} !important; }}
    [data-testid="stSidebarCollapseButton"] svg,
    [data-testid="collapsedControl"] svg,
    button[kind="header"] svg {{ fill: {TEXT} !important; }}
    [data-testid="stMain"], section.main, .main {{ padding-bottom: 80px !important; }}
    .app-footer-sticky {{
        position: fixed;
        bottom: 0; left: 0; right: 0;
        background-color: {SURFACE};
        border-top: 1px solid {BORDER};
        padding: 10px 28px;
        font-size: 0.82rem;
        color: {TEXT_DIM} !important;
        z-index: 100;
        text-align: center;
    }}
    .app-footer-sticky span {{ color: {TEXT_DIM} !important; margin: 0 10px; }}
    .app-footer-sticky strong {{ color: {TEXT} !important; }}
    .app-footer-sticky .sep {{ color: {BORDER} !important; margin: 0 8px; }}
</style>
""", unsafe_allow_html=True)

# 헤더
st.markdown(f"<div style='margin-bottom: 8px;'><span class='header-brand-tag'>온누리커뮤니케이션</span></div>", unsafe_allow_html=True)
st.markdown("<h1 style='margin-top: 0; margin-bottom: 4px;'>마케팅 기여도 분석</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='color:{TEXT_MUTED} !important; margin-top:0;'>청정일 베이스라인 + 매체별 회귀 가중치 산출</p>", unsafe_allow_html=True)

st.markdown("""
<div class="app-footer-sticky">
    <span><strong>마케팅 기여도 분석 도구</strong></span>
    <span class="sep">·</span>
    <span>2026-05-28</span>
    <span class="sep">·</span>
    <span>박시은</span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# 사이드바
st.sidebar.header("분석 정보")

DEFAULT_BRAND_FEATURES = {
    '란시노': ['바이럴조회수', '숏폼조회수', '사이다조회수'],
    '라쉼': ['바이럴조회수', '숏폼조회수', '사이다조회수'],
    '뉴트리딥': ['바이럴조회수', '숏폼조회수', '사이다조회수'],
    '포벰브': ['바이럴조회수', '숏폼조회수', '사이다조회수'],
}

st.sidebar.markdown("**매체별 lag 정책**")
st.sidebar.caption(
    "바이럴(1~3위): lag 없음 (당일만)\n"
    "숏폼: 당일 + 직전 6일\n"
    "사이다: URL 도메인별\n"
    "  cafe/kin.naver → 당일\n"
    "  blog.naver/pann.nate → +1일\n"
    "  그 외/빈 URL → 제외"
)
st.sidebar.markdown("**오가닉 산출**")
st.sidebar.caption(
    "단일/요일별/시즌보정/추세/추세+요일 5가지를 회귀 R²로 비교해 최적 자동 선택"
)
st.sidebar.markdown("**기준값 (고정)**")
st.sidebar.caption("최소 데이터 30일 / 매체 단독일 최소 20개")

st.sidebar.markdown("---")
st.sidebar.markdown("**일간데이터 컬럼 (위치 고정)**")
st.sidebar.code(
    "A 날짜\nB 브랜드\nC 제품\nK 메타제외 매출\nL 메타제외 판매수량\nT 1~3위 조회수(바이럴)\nX 숏폼 조회수",
    language=None
)
st.sidebar.markdown("**사이다 시트 컬럼**")
st.sidebar.code(
    "E 브랜드\nF 제품\nG 게시일자\nI 채널\nK URL\nL 조회수",
    language=None
)


# ==========================================
# 파일 업로드
# ==========================================
uploaded_file = st.file_uploader(
    "일간데이터 업로드 (xlsx)",
    type=["xlsx"],
    key="main_data"
)
uploaded_cida = st.file_uploader(
    "사이다 시트 업로드 (필수, xlsx)",
    type=["xlsx"],
    key="cida_raw",
    help="사이다 조회수는 이 시트의 URL 도메인별 lag으로 일자 귀속됩니다."
)


def col_letter_to_idx(letter):
    letter = letter.upper()
    idx = 0
    for ch in letter:
        idx = idx * 26 + (ord(ch) - ord('A') + 1)
    return idx - 1


# 컬럼 매핑 (정확한 위치)
# 헤더 구조: 행0=그룹헤더(병합셀, 한 칸 밀림) + 행1=컬럼명, 데이터=행2부터
DAILY_COLMAP = {
    '날짜': 'A',           # 0
    '브랜드': 'B',          # 1
    '제품': 'C',           # 2
    '메타제외판매수량': 'L',  # 11
    '메타제외매출': 'K',     # 10  (← 이전 O(14)는 마케팅매출이라 잘못)
    '바이럴조회수': 'T',     # 19  (1~3위 조회수)
    '숏폼조회수': 'X',      # 23  (숏폼 상세/조회수)
}
CIDA_COLMAP = {
    '브랜드': 'E', '제품': 'F', '게시일자': 'G', '채널': 'I', 'url': 'K', '조회수': 'L',
}


def read_by_position(file, header_rows, col_map, sheet_name=None):
    xl = pd.ExcelFile(file)
    all_sheets = xl.sheet_names
    used = None
    if sheet_name is not None:
        if sheet_name in all_sheets:
            used = sheet_name
        else:
            tgt = unicodedata.normalize('NFC', str(sheet_name)).strip()
            cands = []
            for s in all_sheets:
                sn = unicodedata.normalize('NFC', str(s)).strip()
                if sn == tgt or tgt in sn or sn in tgt:
                    cands.append(s)
            cands.sort(key=lambda s: ('삭제' in str(s), len(str(s))))
            if cands:
                used = cands[0]
    if used is None:
        used = all_sheets[0]
    raw = pd.read_excel(xl, sheet_name=used, header=None)
    raw = raw.iloc[header_rows:].reset_index(drop=True)
    out = pd.DataFrame()
    max_idx = raw.shape[1] - 1
    for name, letter in col_map.items():
        ci = col_letter_to_idx(letter)
        if ci > max_idx:
            out[name] = np.nan
        else:
            out[name] = raw.iloc[:, ci].values
    return out, used, all_sheets


if uploaded_file is None:
    st.info("일간데이터 파일을 업로드하세요 (xlsx).")
    st.stop()
if uploaded_cida is None:
    st.info("사이다 시트는 필수입니다.")
    st.stop()

try:
    df, used_sheet, all_sheets = read_by_position(
        uploaded_file, header_rows=2, col_map=DAILY_COLMAP, sheet_name='일간데이터'
    )
    if used_sheet != '일간데이터':
        st.warning(f"'일간데이터' 시트를 찾지 못해 '{used_sheet}' 시트를 사용. 전체 시트: {all_sheets}")
except Exception as e:
    st.error(f"일간데이터 읽기 실패: {e}")
    st.stop()

df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
df = df.dropna(subset=['날짜', '브랜드', '제품'])
df['브랜드'] = df['브랜드'].astype(str).str.strip()
df['제품'] = df['제품'].astype(str).str.strip()
for c in ['메타제외매출', '메타제외판매수량', '바이럴조회수', '숏폼조회수']:
    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

df_products = df[df['제품'] != '전체'].sort_values(by=['브랜드', '제품', '날짜']).copy()
if len(df_products) == 0:
    st.error("'전체'를 제외한 제품 행이 없습니다.")
    st.stop()

# 데이터 읽기 진단 (사용자가 컬럼 매핑 확인용)
with st.expander("📊 데이터 읽기 진단 — 컬럼 매핑 확인", expanded=(df['메타제외매출'].sum() == 0)):
    diag_cols = st.columns(2)
    with diag_cols[0]:
        st.markdown("**컬럼별 합계** (0이면 위치 오류)")
        st.write({
            '메타제외매출 (K열)': f"{df['메타제외매출'].sum():,.0f}",
            '메타제외판매수량 (L열)': f"{df['메타제외판매수량'].sum():,.0f}",
            '바이럴조회수 (T열)': f"{df['바이럴조회수'].sum():,.0f}",
            '숏폼조회수 (X열)': f"{df['숏폼조회수'].sum():,.0f}",
        })
    with diag_cols[1]:
        st.markdown("**제품 단위 (전체 제외 후)**")
        st.write({
            '행수': f"{len(df_products):,}",
            '브랜드 종류': list(df_products['브랜드'].unique())[:5],
            '날짜 범위': f"{df_products['날짜'].min().strftime('%Y-%m-%d')} ~ {df_products['날짜'].max().strftime('%Y-%m-%d')}",
        })

    if df['메타제외매출'].sum() == 0:
        st.error(
            "**메타제외매출 합계 = 0** — K열에 매출 데이터가 없습니다.\n\n"
            "현재 코드는 시트 구조 B (2행 헤더, K열=매출) 기준입니다.\n"
            "업로드한 파일이 이전 구조 (1행 헤더, I열=매출)일 가능성이 큽니다.\n\n"
            "확인:\n"
            "1. 헤더가 2행인지 (0행=그룹헤더, 1행=컬럼명)\n"
            "2. K열(11번째)이 '메타제외/매출'인지\n"
            "3. 데이터는 2행부터 시작하는지"
        )

# 사이다
try:
    df_cida, used_cida_sheet, cida_all = read_by_position(
        uploaded_cida, header_rows=2, col_map=CIDA_COLMAP, sheet_name='기획사이다_raw'
    )
except Exception as e:
    st.error(f"사이다 시트 읽기 실패: {e}")
    st.stop()

df_cida['게시일자'] = pd.to_datetime(df_cida['게시일자'], errors='coerce')
df_cida = df_cida.dropna(subset=['게시일자', '브랜드', '제품'])
df_cida['브랜드'] = df_cida['브랜드'].astype(str).str.strip()
df_cida['제품'] = df_cida['제품'].astype(str).str.strip()
df_cida['채널'] = df_cida['채널'].astype(str).str.strip()
df_cida['url'] = df_cida['url'].astype(str).str.strip()
df_cida['조회수'] = pd.to_numeric(df_cida['조회수'], errors='coerce')

df_cida['_lag'] = df_cida['url'].map(cida_domain_lag)
n_total_cida = len(df_cida)
n_excluded_domain = int(df_cida['_lag'].isna().sum())
df_cida_valid = df_cida.dropna(subset=['_lag']).copy()
df_cida_valid['_lag'] = df_cida_valid['_lag'].astype(int)

# (제품×채널) 평균 보정
empty_views = df_cida_valid['조회수'].isna() | (df_cida_valid['조회수'] == 0)
valid_for_avg = df_cida_valid[~empty_views]
combo_avg = valid_for_avg.groupby(['제품', '채널'])['조회수'].mean()

df_cida_valid['조회수_보정'] = df_cida_valid['조회수'].fillna(0.0).astype(float)
n_corrected = 0
n_skip_combo = 0
missed_combos = []
for idx in df_cida_valid[empty_views].index:
    p = df_cida_valid.at[idx, '제품']
    ch = df_cida_valid.at[idx, '채널']
    key = (p, ch)
    if key in combo_avg.index and not pd.isna(combo_avg[key]):
        df_cida_valid.at[idx, '조회수_보정'] = float(combo_avg[key])
        n_corrected += 1
    else:
        n_skip_combo += 1
        missed_combos.append(f"{p}/{ch}")

df_cida_valid['영향일'] = df_cida_valid['게시일자'] + pd.to_timedelta(df_cida_valid['_lag'], unit='D')

cida_by_product = df_cida_valid.groupby(['영향일', '브랜드', '제품'], as_index=False)['조회수_보정'].sum()
cida_by_product.columns = ['날짜', '브랜드', '제품', '사이다조회수']
cida_by_brand = df_cida_valid.groupby(['영향일', '브랜드'], as_index=False)['조회수_보정'].sum()
cida_by_brand.columns = ['날짜', '브랜드', '사이다조회수']

cida_info = {
    'total': n_total_cida,
    'excluded_domain': n_excluded_domain,
    'corrected': n_corrected,
    'skipped_combo': n_skip_combo,
    'missed_combos': sorted(set(missed_combos)),
    'total_views': int(df_cida_valid['조회수_보정'].sum()),
    'lag_dist': df_cida_valid['_lag'].value_counts().to_dict(),
}

# 제품/브랜드 일자 테이블
df_prod = df_products.groupby(['날짜', '브랜드', '제품'], as_index=False).agg({
    '메타제외매출': 'sum',
    '메타제외판매수량': 'sum',
    '바이럴조회수': 'sum',
    '숏폼조회수': 'sum',
})
df_prod = df_prod.merge(cida_by_product, on=['날짜', '브랜드', '제품'], how='left')
df_prod['사이다조회수'] = df_prod['사이다조회수'].fillna(0)

df_brand = df_products.groupby(['날짜', '브랜드'], as_index=False).agg({
    '메타제외매출': 'sum',
    '메타제외판매수량': 'sum',
    '바이럴조회수': 'sum',
    '숏폼조회수': 'sum',
})
df_brand = df_brand.merge(cida_by_brand, on=['날짜', '브랜드'], how='left')
df_brand['사이다조회수'] = df_brand['사이다조회수'].fillna(0)

for tbl in (df_brand, df_prod):
    tbl['요일'] = tbl['날짜'].dt.day_name()
    tbl['청정일'] = (tbl[MEDIA_ORDER].sum(axis=1) == 0)

# 메트릭
col1, col2, col3, col4 = st.columns(4)
col1.metric("일간데이터 행수(제품)", f"{len(df_products):,}")
col2.metric("브랜드 단위 행수", f"{len(df_brand):,}")
col3.metric("제품 단위 행수", f"{len(df_prod):,}")
col4.metric("사이다 유효행", f"{len(df_cida_valid):,}")

latest_date = df_brand['날짜'].max()
earliest_date = df_brand['날짜'].min()
st.caption(f"데이터 기간: {earliest_date.strftime('%Y-%m-%d')} ~ {latest_date.strftime('%Y-%m-%d')}")

lag_dist_str = ', '.join(f"lag{k}={v}건" for k, v in sorted(cida_info['lag_dist'].items()))
st.info(
    f"사이다 처리 — 전체 {cida_info['total']}건 / 도메인 매칭 {len(df_cida_valid)}건 / "
    f"제외 {cida_info['excluded_domain']}건 / 결측 보정 {cida_info['corrected']}건 / "
    f"총 조회수 {cida_info['total_views']:,}회 [{lag_dist_str}]"
)
if cida_info['skipped_combo'] > 0:
    with st.expander(f"보정 못 한 행 {cida_info['skipped_combo']}건", expanded=False):
        st.caption("(제품, 채널) 조합으로 조회수가 한 건도 없어 평균 산출 불가.")
        st.code('\n'.join(cida_info['missed_combos'][:30]))

btn_col, info_col = st.columns([1, 4])
with btn_col:
    run_analysis = st.button("분석 실행", type="primary", use_container_width=True)
with info_col:
    result_placeholder = st.empty()

if not run_analysis:
    st.stop()


# ==========================================
# 오가닉 베이스라인 (매출/수량 공용)
# ==========================================
def compute_organic_baseline(group, value_col, min_trend_clean=10):
    weekday_names = ['월', '화', '수', '목', '금', '토', '일']
    clean = group[group['청정일']].copy()
    clean['_v'] = clean[value_col].clip(lower=0)
    n_clean = len(clean)
    out = {
        'overall': 0.0,
        'by_wd': {d: 0.0 for d in range(7)},
        'by_wd_n': {d: 0 for d in range(7)},
        'by_wd_src': {d: '-' for d in range(7)},
        'trend_pred': None,
        'trend_slope': 0.0,
        'trend_r2': 0.0,
        'trend_n': n_clean,
        'n_clean': n_clean,
        'wd_cover': 0,
        'days_since': None,
        'dist_grade': '-',
        'recency_grade': '-',
    }
    if n_clean == 0:
        return out
    overall = clean['_v'].mean()
    out['overall'] = overall
    out['wd_cover'] = clean['요일'].nunique()
    latest_clean = clean['날짜'].max()
    latest_data = group['날짜'].max()
    out['days_since'] = (latest_data - latest_clean).days
    clean['요일num'] = clean['날짜'].dt.dayofweek
    cnt = clean.groupby('요일num').size().to_dict()
    avg = clean.groupby('요일num')['_v'].mean().to_dict()
    MIN_WD = 5
    for d in range(7):
        n = cnt.get(d, 0)
        v = avg.get(d, np.nan)
        out['by_wd_n'][d] = int(n)
        if n >= MIN_WD and not pd.isna(v):
            out['by_wd'][d] = round(v, 1)
            out['by_wd_src'][d] = '요일별'
        else:
            out['by_wd'][d] = round(overall, 1)
            out['by_wd_src'][d] = '전체평균'
    wc = out['wd_cover']
    out['dist_grade'] = 'HIGH' if wc == 7 else ('MEDIUM' if wc >= 5 else 'LOW')
    ds = out['days_since']
    out['recency_grade'] = 'HIGH' if ds <= 30 else ('MEDIUM' if ds <= 90 else 'LOW')

    if n_clean >= min_trend_clean:
        base_date = group['날짜'].min()
        ct = (clean['날짜'] - base_date).dt.days.values.reshape(-1, 1)
        cy = clean['_v'].values
        if len(np.unique(ct)) >= 2:
            tm = LinearRegression().fit(ct, cy)
            allt = ((group['날짜'] - base_date).dt.days).values.reshape(-1, 1)
            pred = np.clip(tm.predict(allt), 0, None)
            out['trend_pred'] = pd.Series(pred, index=group.index)
            out['trend_slope'] = float(tm.coef_[0])
            out['trend_r2'] = float(tm.score(ct, cy))
    return out


def build_media_features(g, features_base):
    gg = g.copy()
    media_cols = {}
    for f in features_base:
        cols = [f]
        if MEDIA_LAG_POLICY.get(f) == 'fixed6':
            for lag in range(1, SHORTFORM_LAG_MAX + 1):
                c = f'{f}_lag{lag}'
                gg[c] = gg[f].shift(lag)
                cols.append(c)
        media_cols[f] = cols
    return gg, media_cols


def fit_r2(g, y_col, x_cols):
    reg = g[[y_col] + x_cols].dropna()
    if len(reg) < MIN_SOLO_DAYS:
        return -1, None, 0
    m = LinearRegression(fit_intercept=True)
    m.fit(reg[x_cols], reg[y_col])
    return m.score(reg[x_cols], reg[y_col]), m, len(reg)


# ==========================================
# 브랜드 분석 (매출)
# ==========================================
def analyze_brand(group, features_base, brand_label):
    group = group.copy().sort_values('날짜').reset_index(drop=True)
    weekday_names = ['월', '화', '수', '목', '금', '토', '일']

    result = {
        '브랜드': brand_label,
        '청정일수': 0, '청정일_요일커버': 0, '오가닉_분포등급': '-',
        '최근청정일_경과(일)': None, '오가닉_최신성등급': '-',
        '오가닉매출': 0, '오가닉매출_최근': 0,
        '오가닉매출_월': 0, '오가닉매출_화': 0, '오가닉매출_수': 0, '오가닉매출_목': 0,
        '오가닉매출_금': 0, '오가닉매출_토': 0, '오가닉매출_일': 0,
        '청정일_월': 0, '청정일_화': 0, '청정일_수': 0, '청정일_목': 0,
        '청정일_금': 0, '청정일_토': 0, '청정일_일': 0,
        '오가닉모드': '-',
        '추세_기울기(월)': 0, '추세_R²': 0,
        '기저매출_7일vs30일(%)': 0,
        '시즌보정': '-', '요일패턴_최대': '-', '월패턴_최대': '-',
        '회귀샘플수': 0, '전체R²': 0,
    }
    # 매체 컬럼 기본값 0
    for m in MEDIA_ORDER:
        result[f'{m}_가중치'] = 0
        result[f'{m}_최적lag'] = 0
        result[f'{m}_단독R²'] = 0
        result[f'{m}_단독일수'] = 0
    result['분석상태'] = '대기'

    if len(group) < MIN_DATA_DAYS:
        result['분석상태'] = f'데이터부족({len(group)}일)'
        return result
    if group['메타제외매출'].sum() == 0:
        result['분석상태'] = '매출 0'
        return result

    base = compute_organic_baseline(group, '메타제외매출')
    if base['n_clean'] == 0:
        result['분석상태'] = '청정일 없음'
        return result

    organic_avg = base['overall']
    organic_by_wd = base['by_wd']
    result.update({
        '청정일수': base['n_clean'],
        '오가닉매출': round(organic_avg, 0),
        '청정일_요일커버': base['wd_cover'],
        '오가닉_분포등급': base['dist_grade'],
        '최근청정일_경과(일)': base['days_since'],
        '오가닉_최신성등급': base['recency_grade'],
    })
    if base['trend_pred'] is not None:
        result['오가닉매출_최근'] = round(float(base['trend_pred'].iloc[-1]), 0)
    else:
        result['오가닉매출_최근'] = round(organic_avg, 0)
    for d in range(7):
        result[f'오가닉매출_{weekday_names[d]}'] = organic_by_wd[d]
        result[f'청정일_{weekday_names[d]}'] = base['by_wd_n'][d]

    # 기저매출 변동
    group['매출_음수제거'] = group['메타제외매출'].clip(lower=0)
    group['평균_7일'] = group['매출_음수제거'].rolling(7, min_periods=3).mean().shift(1)
    group['평균_30일'] = group['매출_음수제거'].rolling(30, min_periods=10).mean().shift(1)
    recent = group.dropna(subset=['평균_7일', '평균_30일']).tail(30)
    if len(recent) > 0 and recent['평균_30일'].mean() > 0:
        dev = ((recent['평균_7일'].mean() - recent['평균_30일'].mean()) / recent['평균_30일'].mean()) * 100
    else:
        dev = 0
    result['기저매출_7일vs30일(%)'] = round(dev, 1)

    # Y 후보
    group['요일num'] = group['날짜'].dt.dayofweek
    group['월'] = group['날짜'].dt.month
    group['Y_raw'] = group['메타제외매출'] - organic_avg
    group['Y_wd'] = group['메타제외매출'] - group['요일num'].map(organic_by_wd).fillna(organic_avg)

    has_trend = base['trend_pred'] is not None
    if has_trend:
        trend_pred = base['trend_pred'].reindex(group.index)
        group['Y_trend'] = group['메타제외매출'] - trend_pred.values
        clean_t = group[group['청정일']].copy()
        clean_t['_resid'] = clean_t['메타제외매출'].clip(lower=0) - trend_pred.reindex(clean_t.index).values
        clean_t['요일num'] = clean_t['날짜'].dt.dayofweek
        wd_resid_cnt = clean_t.groupby('요일num').size().to_dict()
        wd_resid = clean_t.groupby('요일num')['_resid'].mean().to_dict()
        wd_resid_safe = {d: (wd_resid.get(d, 0.0) if wd_resid_cnt.get(d, 0) >= 5 and not pd.isna(wd_resid.get(d, np.nan)) else 0.0) for d in range(7)}
        trend_wd_organic = trend_pred.values + group['요일num'].map(wd_resid_safe).fillna(0.0).values
        group['Y_trend_wd'] = group['메타제외매출'] - np.clip(trend_wd_organic, 0, None)

    apply_season = False
    season_info = {'weekday_coef': {}, 'month_coef': {}}
    if base['n_clean'] >= 14:
        clean = group[group['청정일']].copy()
        clean['_v'] = clean['메타제외매출'].clip(lower=0)
        clean['요일num'] = clean['날짜'].dt.dayofweek
        clean['월'] = clean['날짜'].dt.month
        oavg = clean['_v'].mean()
        if oavg > 0:
            MIN_S = 5
            wd_cnt = clean.groupby('요일num').size().to_dict()
            mo_cnt = clean.groupby('월').size().to_dict()
            wd_avg = (clean.groupby('요일num')['_v'].mean() / oavg).to_dict()
            mo_avg = (clean.groupby('월')['_v'].mean() / oavg).to_dict()

            def safe_coef(coef_dict, count_dict, full_range, lo=0.6, hi=1.7):
                r = {}
                for k in full_range:
                    n = count_dict.get(k, 0)
                    if n < MIN_S:
                        r[k] = 1.0
                    else:
                        v = coef_dict.get(k, 1.0)
                        r[k] = 1.0 if pd.isna(v) else max(lo, min(hi, v))
                return r

            wd_coef = safe_coef(wd_avg, wd_cnt, range(7))
            mo_coef = safe_coef(mo_avg, mo_cnt, range(1, 13))
            group['시즌계수'] = group['요일num'].map(wd_coef).fillna(1.0) * group['월'].map(mo_coef).fillna(1.0)
            group['조정매출'] = group['메타제외매출'] / group['시즌계수']
            clean['시즌계수'] = clean['요일num'].map(wd_coef).fillna(1.0) * clean['월'].map(mo_coef).fillna(1.0)
            organic_adj = (clean['_v'] / clean['시즌계수']).mean()
            group['Y_adj'] = group['조정매출'] - organic_adj
            apply_season = True
            season_info['weekday_coef'] = wd_coef
            season_info['month_coef'] = mo_coef

    def quick_r2(y_col):
        gg, mc = build_media_features(group, features_base)
        all_x = [c for cols in mc.values() for c in cols]
        return fit_r2(gg, y_col, all_x)[0]

    r2_단일 = quick_r2('Y_raw')
    r2_요일별 = quick_r2('Y_wd')
    r2_시즌 = quick_r2('Y_adj') if apply_season else -999
    r2_추세 = quick_r2('Y_trend') if has_trend else -999
    r2_추세요일 = quick_r2('Y_trend_wd') if has_trend else -999

    candidates = [('단일', r2_단일, 'Y_raw'), ('요일별', r2_요일별, 'Y_wd')]
    if apply_season:
        candidates.append(('시즌보정', r2_시즌, 'Y_adj'))
    if has_trend:
        candidates.append(('추세', r2_추세, 'Y_trend'))
        candidates.append(('추세+요일', r2_추세요일, 'Y_trend_wd'))
    best_mode, best_r2, best_y = max(candidates, key=lambda x: x[1])
    group['Y'] = group[best_y]
    result['오가닉모드'] = f"{best_mode} (R²={best_r2:.3f})"

    if has_trend:
        result['추세_기울기(월)'] = round(base['trend_slope'] * 30, 0)
        result['추세_R²'] = round(base['trend_r2'], 3)

    if apply_season:
        wmax = max(season_info['weekday_coef'], key=season_info['weekday_coef'].get)
        mmax = max(season_info['month_coef'], key=season_info['month_coef'].get)
        tag = '적용' if best_mode == '시즌보정' else '미채택'
        result['시즌보정'] = f"{tag} (단일 {r2_단일:.3f}/요일별 {r2_요일별:.3f}/시즌 {r2_시즌:.3f})"
        result['요일패턴_최대'] = f"{weekday_names[wmax]} ({season_info['weekday_coef'][wmax]:.2f})"
        result['월패턴_최대'] = f"{mmax}월 ({season_info['month_coef'][mmax]:.2f})"
    else:
        result['시즌보정'] = f"미적용 (청정일 {base['n_clean']}개<14)" if base['n_clean'] < 14 else '미적용'

    # 매체별 가중치 (최적lag 표기)
    gg, media_cols = build_media_features(group, features_base)
    for f in features_base:
        if MEDIA_LAG_POLICY.get(f) == 'none':
            result[f'{f}_최적lag'] = 0
        elif MEDIA_LAG_POLICY.get(f) == 'fixed6':
            result[f'{f}_최적lag'] = '0~6'
        else:
            result[f'{f}_최적lag'] = '도메인'

    all_x = [c for cols in media_cols.values() for c in cols]
    reg = gg[['Y'] + all_x].dropna()
    if len(reg) < MIN_SOLO_DAYS:
        result['회귀샘플수'] = len(reg)
        result['분석상태'] = f'회귀샘플 부족 ({len(reg)})'
        return result

    model = LinearRegression(fit_intercept=True)
    model.fit(reg[all_x], reg['Y'])
    coefs = dict(zip(all_x, model.coef_))
    result['전체R²'] = round(model.score(reg[all_x], reg['Y']), 3)
    result['회귀샘플수'] = len(reg)

    for f in features_base:
        total = sum(coefs.get(c, 0) for c in media_cols[f])
        result[f'{f}_가중치'] = max(0, round(total, 2))

    # 단독R²
    for f in features_base:
        others = [m for m in features_base if m != f]
        mask = (reg[f] > 0)
        for om in others:
            mask = mask & (reg[om] == 0)
        solo = reg[mask]
        result[f'{f}_단독일수'] = len(solo)
        if len(solo) < MIN_SOLO_DAYS:
            result[f'{f}_단독R²'] = 0
            continue
        sc = media_cols[f]
        ms = LinearRegression(fit_intercept=True)
        ms.fit(solo[sc], solo['Y'])
        result[f'{f}_단독R²'] = round(ms.score(solo[sc], solo['Y']), 3)

    result['분석상태'] = '분석성공'
    return result


# ==========================================
# 제품별 오가닉수량
# ==========================================
def analyze_product_qty(group, brand_label, product_label):
    weekday_names = ['월', '화', '수', '목', '금', '토', '일']
    res = {
        '브랜드': brand_label, '제품': product_label,
        '청정일수': 0,
        '오가닉수량': 0, '오가닉수량_최근': 0,
        '오가닉수량_월': 0, '오가닉수량_화': 0, '오가닉수량_수': 0, '오가닉수량_목': 0,
        '오가닉수량_금': 0, '오가닉수량_토': 0, '오가닉수량_일': 0,
        '청정일_월': 0, '청정일_화': 0, '청정일_수': 0, '청정일_목': 0,
        '청정일_금': 0, '청정일_토': 0, '청정일_일': 0,
        '추세_기울기(월)': 0,
        '오가닉_분포등급': '-', '최근청정일_경과(일)': None, '오가닉_최신성등급': '-',
        '상태': '대기',
    }
    if len(group) < MIN_DATA_DAYS:
        res['상태'] = f'데이터부족({len(group)}일)'
    base = compute_organic_baseline(group, '메타제외판매수량')
    if base['n_clean'] == 0:
        res['상태'] = '청정일 없음'
        return res
    res.update({
        '청정일수': base['n_clean'],
        '오가닉수량': round(base['overall'], 2),
        '오가닉_분포등급': base['dist_grade'],
        '최근청정일_경과(일)': base['days_since'],
        '오가닉_최신성등급': base['recency_grade'],
    })
    if base['trend_pred'] is not None:
        res['오가닉수량_최근'] = round(float(base['trend_pred'].iloc[-1]), 2)
        res['추세_기울기(월)'] = round(base['trend_slope'] * 30, 2)
    else:
        res['오가닉수량_최근'] = round(base['overall'], 2)
    for d in range(7):
        res[f'오가닉수량_{weekday_names[d]}'] = base['by_wd'][d]
        res[f'청정일_{weekday_names[d]}'] = base['by_wd_n'][d]
    if res['상태'] == '대기':
        res['상태'] = '산출완료'
    return res


# ==========================================
# 실행
# ==========================================
results = []
progress = st.progress(0)
status = st.empty()

all_brands = list(df_brand.groupby('브랜드'))
for idx, (brand, group) in enumerate(all_brands):
    progress.progress((idx + 1) / max(1, len(all_brands)))
    status.text(f"브랜드 분석: {brand}")
    if brand not in DEFAULT_BRAND_FEATURES:
        continue
    results.append(analyze_brand(group, DEFAULT_BRAND_FEATURES[brand], brand))
progress.empty()
status.empty()

if len(results) == 0:
    st.warning("브랜드 분석 결과 없음.")
    st.stop()

result_df = pd.DataFrame(results)

qty_results = []
for (brand, product), g in df_prod.groupby(['브랜드', '제품']):
    if brand not in DEFAULT_BRAND_FEATURES:
        continue
    qty_results.append(analyze_product_qty(g.sort_values('날짜').reset_index(drop=True), brand, product))
qty_df = pd.DataFrame(qty_results) if qty_results else pd.DataFrame()

success_n = int((result_df['분석상태'] == '분석성공').sum())
result_placeholder.success(
    f"분석 완료 — 브랜드 {len(result_df)}개 / 성공 {success_n}개 · 제품 오가닉수량 {len(qty_df)}개"
)


# ==========================================
# 스타일
# ==========================================
def color_grade(val):
    return f'background-color: {HIGHLIGHT_BG}; color: {TEXT}; font-weight: 600' if val == 'HIGH' else ''


def color_r2(val):
    if val in ('-', None) or (isinstance(val, float) and pd.isna(val)):
        return ''
    try:
        v = float(val)
    except Exception:
        return ''
    return f'background-color: {HIGHLIGHT_BG}; color: {TEXT}; font-weight: 600' if v >= 0.5 else ''


def color_status(val):
    return f'background-color: {HIGHLIGHT_BG}; color: {TEXT}; font-weight: 600' if val in ('분석성공', '산출완료') else ''


# ==========================================
# 탭 5개
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["매출 분석", "수량 분석", "데이터 개수", "용어 정의", "산정 기준"]
)

# ---- 매출 분석 ----
with tab1:
    st.subheader("브랜드별 매출 분석")
    st.caption(
        "각 브랜드의 오가닉매출(요일별, 추세 반영), 매체별 가중치, 신뢰도(R²). "
        "오가닉모드 컬럼: 단일/요일별/시즌보정/추세/추세+요일 중 R² 가장 높은 것 자동 채택."
    )

    sales_cols = [
        '브랜드',
        '청정일수', '오가닉_분포등급', '오가닉_최신성등급',
        '오가닉매출', '오가닉매출_최근',
        '오가닉매출_월', '오가닉매출_화', '오가닉매출_수', '오가닉매출_목',
        '오가닉매출_금', '오가닉매출_토', '오가닉매출_일',
        '오가닉모드', '추세_기울기(월)', '추세_R²',
        '전체R²',
    ]
    for m in MEDIA_ORDER:
        sales_cols += [f'{m}_가중치', f'{m}_최적lag', f'{m}_단독R²']
    sales_cols.append('분석상태')
    sales_cols = [c for c in sales_cols if c in result_df.columns]
    
    s = result_df[sales_cols].style
    s = s.map(color_grade, subset=[c for c in ['오가닉_분포등급', '오가닉_최신성등급'] if c in sales_cols])
    s = s.map(color_r2, subset=[c for c in sales_cols if 'R²' in c])
    s = s.map(color_status, subset=['분석상태'])
    st.dataframe(s, use_container_width=True, height=400)

# ---- 수량 분석 ----
with tab2:
    st.subheader("제품별 수량 분석")
    st.caption(
        "각 제품의 오가닉수량(메타제외 판매수량 기준 청정일 평균, 요일별, 추세 반영). "
        "마케팅 영향 없는 기본 판매량을 의미합니다."
    )
    if len(qty_df) > 0:
        qty_cols = [
            '브랜드', '제품',
            '청정일수', '오가닉_분포등급', '오가닉_최신성등급',
            '오가닉수량', '오가닉수량_최근',
            '오가닉수량_월', '오가닉수량_화', '오가닉수량_수', '오가닉수량_목',
            '오가닉수량_금', '오가닉수량_토', '오가닉수량_일',
            '추세_기울기(월)', '상태',
        ]
        qty_cols = [c for c in qty_cols if c in qty_df.columns]
        sq = qty_df[qty_cols].style
        sq = sq.map(color_grade, subset=[c for c in ['오가닉_분포등급', '오가닉_최신성등급'] if c in qty_cols])
        sq = sq.map(color_status, subset=['상태'])
        st.dataframe(sq, use_container_width=True, height=600)
    else:
        st.warning("수량 분석 결과 없음.")

# ---- 데이터 개수 ----
with tab3:
    st.subheader("데이터 개수 / 최신성")
    st.caption(
        "각 분석의 신뢰도 판단용. 청정일이 요일별로 충분히 분포되어 있는지, "
        "최근 청정일이 얼마나 가까운지 확인."
    )

    st.markdown("#### 매출 분석 — 브랜드별 청정일 데이터")
    cnt_cols = [
        '브랜드', '청정일수', '청정일_요일커버',
        '청정일_월', '청정일_화', '청정일_수', '청정일_목',
        '청정일_금', '청정일_토', '청정일_일',
        '오가닉_분포등급', '최근청정일_경과(일)', '오가닉_최신성등급',
        '회귀샘플수',
    ]
    cnt_cols = [c for c in cnt_cols if c in result_df.columns]
    sc = result_df[cnt_cols].style
    sc = sc.map(color_grade, subset=[c for c in ['오가닉_분포등급', '오가닉_최신성등급'] if c in cnt_cols])
    st.dataframe(sc, use_container_width=True, height=300)

    st.markdown("#### 수량 분석 — 제품별 청정일 데이터")
    if len(qty_df) > 0:
        qcnt_cols = [
            '브랜드', '제품', '청정일수',
            '청정일_월', '청정일_화', '청정일_수', '청정일_목',
            '청정일_금', '청정일_토', '청정일_일',
            '오가닉_분포등급', '최근청정일_경과(일)', '오가닉_최신성등급',
        ]
        qcnt_cols = [c for c in qcnt_cols if c in qty_df.columns]
        sqc = qty_df[qcnt_cols].style
        sqc = sqc.map(color_grade, subset=[c for c in ['오가닉_분포등급', '오가닉_최신성등급'] if c in qcnt_cols])
        st.dataframe(sqc, use_container_width=True, height=500)
    else:
        st.warning("수량 분석 데이터 없음.")

# ---- 용어 정의 ----
with tab4:
    st.subheader("용어 정의")
    glossary_df = pd.DataFrame([
        ['오가닉매출', '광고/마케팅 없이 발생하는 매출 (브랜드별)', '청정일 메타제외매출 평균'],
        ['오가닉수량', '광고/마케팅 없이 발생하는 판매수량 (제품별)', '청정일 메타제외판매수량 평균'],
        ['청정일', '바이럴·숏폼·사이다 조회수가 모두 0인 날', '오가닉 추정 기준'],
        ['마케팅기여매출', '당일매출 - 오가닉매출', '양수만 측정'],
        ['콘텐츠기여매출', '마케팅기여매출 × 채널비중 × 채널내 콘텐츠조회수비중', ''],
        ['마케팅추정ROAS', '콘텐츠기여매출 / 비용', ''],
        ['전체R²', '모든 매체 동시 학습 회귀 설명력', '0.5↑ 신뢰 가능'],
        ['단독R²', '해당 매체만 단독 진행된 날 회귀 설명력', f'단독일 {MIN_SOLO_DAYS}개↑'],
        ['오가닉매출_최근', '추세선이 예측한 최신 시점 오가닉매출', '시간 흐름 반영'],
        ['추세_기울기(월)', '오가닉이 한 달에 변하는 양', '+ 성장, − 감소'],
    ], columns=['용어', '정의', '비고'])
    st.dataframe(glossary_df, use_container_width=True, hide_index=True, height=400)

    st.markdown("### 등급 기준")
    st.markdown("""
| 등급 항목 | HIGH | MEDIUM | LOW |
|---|---|---|---|
| 오가닉_분포등급 | 7요일 모두 청정일 있음 | 5~6요일 | 4요일 이하 |
| 오가닉_최신성등급 | 최근 청정일 30일 이내 | 31~90일 | 90일 초과 |
| R² (회귀 설명력) | 0.5 이상 | 0.3 ~ 0.5 | 0.3 미만 |
""")

# ---- 산정 기준 ----
with tab5:
    st.subheader("산정 기준")
    st.markdown(f"""
#### 1. 분석 단위
- **매출 분석 (오가닉매출 + 회귀 가중치)**: 브랜드별 (같은 브랜드 모든 제품 일자 합산)
- **수량 분석 (오가닉수량)**: 제품별 (제품 단위 청정일 평균)
- 모든 산출은 **메타광고를 제외한** 매출/수량 기준 (메타제외매출, 메타제외판매수량)

#### 2. 컬럼 위치 매핑 (고정)
일간데이터는 2행 그룹헤더 구조 → 컬럼명만으로는 식별 어려움 → 엑셀 열 위치로 직접 매핑.

| 항목 | 열 | 그룹 + 컬럼명 |
|---|---|---|
| 날짜 | A | 날짜 |
| 브랜드 | B | 구분/브랜드 |
| 제품 | C | 구분/제품 |
| 메타제외 매출 | K | 메타제외/매출 |
| 메타제외 판매수량 | L | 메타제외/판매수량 |
| 바이럴 조회수 | T | 바이럴 상세/1~3위 조회수 |
| 숏폼 조회수 | X | 숏폼 상세/조회수 |

사이다 조회수는 일간데이터에서 안 쓰고 사이다 시트에서 산출.

#### 3. 사이다 시트
| 항목 | 열 |
|---|---|
| 브랜드 | E |
| 제품 | F |
| 게시일자 | G |
| 채널 | I |
| URL | K |
| 조회수 | L |

#### 4. 매체별 lag 정책
| 매체 | lag | 회귀 반영 |
|---|---|---|
| 바이럴 | 없음 (당일) | 당일값만 X 변수 |
| 숏폼 | 당일 + 직전 6일 (총 7일) | 당일 + lag1~6 X 변수 |
| 사이다 | **URL 도메인별** | 게시일자+lag로 일자 귀속 후 당일값 |

**사이다 도메인 규칙**:
- `cafe` 포함 → 0 (당일)
- `kin.naver` 포함 → 0
- `blog.naver` 포함 → 1
- `pann.nate` 포함 → 1
- 그 외 / 빈 URL → **제외**

#### 5. 사이다 결측 보정
- URL 매칭된 행 중 조회수 비었거나 0이면 → (제품×채널) 평균으로 추정
- (제품×채널) 조합 없으면 보정 없이 0

#### 6. 오가닉 산출 — 5가지 모드 자동 비교

**5가지 베이스라인** 각각에 대해 매출 회귀를 돌려 **R² 가장 높은 모드를 자동 채택**.

| 모드 | Y 공식 | 특징 |
|---|---|---|
| 단일 | `매출 - 청정일 평균` | 가장 단순 |
| 요일별 | `매출 - 그날 요일별 평균` | 요일 효과 반영 |
| 시즌보정 | `(매출 ÷ 요일계수×월계수) - 오가닉` | 비례 정규화 |
| **추세** | `매출 - (청정일 시간추세 예측값)` | **시간 흐름 반영** |
| **추세+요일** | `매출 - (추세 + 요일잔차)` | 추세 + 요일 |

**추세 반영 방식**:
1. 청정일 데이터로 `매출 ~ 경과일` 선형회귀 학습
2. 각 날짜의 추세선 예측값 = 그날의 오가닉
3. 이렇게 하면 시간이 지남에 따라 오가닉이 자연 성장/감소하는 효과 반영
4. `오가닉매출_최근` 컬럼 = 최신 시점의 추세 예측 오가닉 (참고용)

**적용 조건**:
- 요일별: 요일 표본 5건 이상
- 시즌보정: 청정일 14개 이상
- 추세: 청정일 10개 이상
- 미달 시 해당 모드는 후보에서 제외

#### 7. 회귀 + 가중치 산출
- Y = (선택된 오가닉 모드의 매출 잔차)
- X = 매체별 당일 (+숏폼 lag1~6) 조회수
- 가중치 = 해당 매체 당일+lag 계수 합 (음수면 0)
- 전체R² = 모든 매체 동시 다중회귀 설명력
- 매체별 단독R² = 그 매체만 진행된 날 데이터로 측정 (단독일 {MIN_SOLO_DAYS}개 이상)

#### 8. 결과 해석 가이드
- R² ≥ 0.5: 가중치 신뢰 가능 → 시트 적용
- R² 0.3~0.5: 참고 + 정성 판단
- R² < 0.3: 단순 조회수 비율 대체 권장

가중치 0: 해당 매체가 매출과 유의한 관계 없음 (또는 음의 관계) → 시트에서 그대로 0으로 사용 (다른 매체 점수만으로 비중 산출).
""")

# ==========================================
# 엑셀 다운로드
# ==========================================
buffer = BytesIO()
with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
    result_df.to_excel(writer, sheet_name='매출분석_브랜드별', index=False)
    if len(qty_df) > 0:
        qty_df.to_excel(writer, sheet_name='수량분석_제품별', index=False)
    pd.DataFrame([
        ['오가닉매출', '광고/마케팅 없이 발생하는 매출 (브랜드별)', '청정일 메타제외매출 평균'],
        ['오가닉수량', '광고/마케팅 없이 발생하는 판매수량 (제품별)', '청정일 메타제외판매수량 평균'],
        ['청정일', '바이럴·숏폼·사이다 조회수가 모두 0인 날', ''],
        ['마케팅기여매출', '당일매출 - 오가닉매출', '양수만 측정'],
    ], columns=['용어', '정의', '비고']).to_excel(writer, sheet_name='용어집', index=False)

st.download_button(
    label="결과 엑셀 다운로드 (.xlsx)",
    data=buffer.getvalue(),
    file_name="마케팅기여도분석_결과.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
st.markdown("---")
