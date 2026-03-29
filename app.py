# ============================================================
#  ML Dashboard — Production-Ready Streamlit Application
#  Author : Senior Python Developer / Data Scientist
#  Version: 1.0.0
# ============================================================

import io
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  CUSTOM CSS  —  clean, friendly light theme
# ─────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800&family=Nunito+Sans:wght@400;600&display=swap');

/* ── Palette ── */
:root {
    --bg:          #f0f4ff;
    --bg-sidebar:  #ffffff;
    --bg-card:     #ffffff;
    --bg-card2:    #f7f9ff;
    --accent:      #4f6ef7;
    --accent-dk:   #3a57e8;
    --accent-g:    #22c9a5;
    --accent-r:    #ff6b6b;
    --accent-y:    #ffb84d;
    --text:        #1e2a45;
    --text-muted:  #7a869a;
    --border:      #dde4f5;
    --shadow-sm:   0 2px 10px rgba(79,110,247,.08);
    --shadow-md:   0 6px 28px rgba(79,110,247,.13);
    --radius:      16px;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Main content padding ── */
.block-container {
    padding: 2rem 2.5rem 3rem !important;
    max-width: 1280px !important;
}

/* ══════════════════════════════════
   SIDEBAR
══════════════════════════════════ */
[data-testid="stSidebar"] {
    background: var(--bg-sidebar) !important;
    border-right: 2px solid var(--border) !important;
    box-shadow: 4px 0 20px rgba(79,110,247,.07) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] .stRadio > div { gap: 4px !important; }
[data-testid="stSidebar"] .stRadio label {
    display: flex !important;
    align-items: center !important;
    gap: .55rem !important;
    font-size: .95rem !important;
    font-weight: 600 !important;
    padding: .6rem .9rem !important;
    border-radius: 12px !important;
    cursor: pointer !important;
    transition: background .18s, color .18s !important;
    color: var(--text-muted) !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: #eef1ff !important;
    color: var(--accent) !important;
}
[data-testid="stSidebar"] .stRadio label[data-checked="true"] {
    background: linear-gradient(90deg,#eef1ff,#f0f4ff) !important;
    color: var(--accent) !important;
    border-left: 3px solid var(--accent) !important;
}

/* ══════════════════════════════════
   METRIC TILES
══════════════════════════════════ */
[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1.1rem 1.4rem !important;
    box-shadow: var(--shadow-sm) !important;
    transition: box-shadow .2s, transform .2s !important;
}
[data-testid="metric-container"]:hover {
    box-shadow: var(--shadow-md) !important;
    transform: translateY(-2px) !important;
}
[data-testid="metric-container"] label {
    color: var(--text-muted) !important;
    font-size: .72rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: .09em !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-size: 1.9rem !important;
    font-weight: 800 !important;
}

/* ══════════════════════════════════
   HEADINGS
══════════════════════════════════ */
h1 {
    font-size: 2rem !important;
    font-weight: 800 !important;
    color: var(--text) !important;
    letter-spacing: -.02em !important;
}
h2 {
    font-size: 1.45rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
}
h3, h4, h5 {
    font-size: 1rem !important;
    font-weight: 700 !important;
    color: var(--text-muted) !important;
}

/* ══════════════════════════════════
   CARDS
══════════════════════════════════ */
.kard {
    background: var(--bg-card);
    border: 1.5px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow-sm);
    transition: box-shadow .2s, transform .2s;
}
.kard:hover { box-shadow: var(--shadow-md); transform: translateY(-2px); }
.kard-label {
    font-size: .65rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .13em;
    color: var(--accent);
    margin-bottom: .45rem;
}

/* feature tile on Home */
.feat-tile {
    background: var(--bg-card);
    border: 1.5px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem 1.5rem;
    height: 155px;
    box-shadow: var(--shadow-sm);
    transition: box-shadow .22s, transform .22s;
    cursor: default;
}
.feat-tile:hover { box-shadow: var(--shadow-md); transform: translateY(-3px); }
.feat-icon { font-size: 1.9rem; margin-bottom: .5rem; }
.feat-name { font-size: .95rem; font-weight: 700; color: var(--text); margin-bottom: .3rem; }
.feat-desc { font-size: .8rem; color: var(--text-muted); line-height: 1.5; }

/* step row on Home */
.step-row {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    background: var(--bg-card);
    border: 1.5px solid var(--border);
    border-radius: 14px;
    padding: .95rem 1.2rem;
    margin-bottom: .6rem;
    box-shadow: var(--shadow-sm);
}
.step-num {
    background: var(--accent);
    color: #fff;
    font-weight: 800;
    border-radius: 10px;
    padding: .2rem .7rem;
    font-size: .9rem;
    flex-shrink: 0;
    margin-top: .05rem;
}
.step-body-title { font-weight: 700; font-size: .95rem; margin-bottom: .15rem; color: var(--text); }
.step-body-desc  { color: var(--text-muted); font-size: .83rem; }

/* status pills in sidebar */
.pill {
    border-radius: 12px;
    padding: .8rem 1rem;
    font-size: .82rem;
    margin-top: .6rem;
}
.pill-green { background:#eafaf5; border:1.5px solid #a8edda; }
.pill-red   { background:#fff2f2; border:1.5px solid #ffd0d0; }
.pill-blue  { background:#eef1ff; border:1.5px solid #c5ceff; }
.pill-title { font-weight: 700; margin-bottom: .25rem; }
.pill-sub   { color: var(--text-muted); }

/* page header banner */
.page-header {
    background: linear-gradient(115deg, #4f6ef7 0%, #7c9cff 100%);
    border-radius: var(--radius);
    padding: 1.4rem 1.8rem;
    margin-bottom: 1.6rem;
    box-shadow: 0 6px 24px rgba(79,110,247,.22);
}
.page-header h2 { color: #fff !important; margin: 0 !important; font-size: 1.45rem !important; }
.page-header p  { color: rgba(255,255,255,.82) !important; margin: .3rem 0 0 !important; font-size: .88rem !important; }

/* ══════════════════════════════════
   BUTTONS
══════════════════════════════════ */
.stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, #7c9cff 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: .6rem 1.8rem !important;
    font-weight: 700 !important;
    font-size: .9rem !important;
    letter-spacing: .02em !important;
    box-shadow: 0 4px 14px rgba(79,110,247,.35) !important;
    transition: transform .15s, box-shadow .15s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 7px 20px rgba(79,110,247,.45) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

.stDownloadButton > button {
    background: linear-gradient(135deg, var(--accent-g) 0%, #19b899 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: .6rem 1.8rem !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 14px rgba(34,201,165,.3) !important;
}

/* ══════════════════════════════════
   INPUTS
══════════════════════════════════ */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    border: 1.5px solid var(--border) !important;
    border-radius: 12px !important;
    background: var(--bg-card2) !important;
    font-size: .92rem !important;
}
[data-testid="stSlider"] { padding: .2rem 0 !important; }
.stNumberInput input {
    border: 1.5px solid var(--border) !important;
    border-radius: 12px !important;
    background: var(--bg-card2) !important;
    font-size: .92rem !important;
}
[data-testid="stFileUploader"] {
    border: 2px dashed var(--border) !important;
    border-radius: 14px !important;
    background: var(--bg-card2) !important;
    padding: 1.2rem !important;
    transition: border-color .2s !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }

/* ══════════════════════════════════
   TABS
══════════════════════════════════ */
[data-testid="stTabs"] [role="tab"] {
    font-weight: 700 !important;
    font-size: .88rem !important;
    border-radius: 10px 10px 0 0 !important;
    padding: .5rem 1.1rem !important;
    color: var(--text-muted) !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 3px solid var(--accent) !important;
}

/* ══════════════════════════════════
   DATAFRAME / TABLE
══════════════════════════════════ */
[data-testid="stDataFrame"] {
    border-radius: 14px !important;
    border: 1.5px solid var(--border) !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ══════════════════════════════════
   ALERTS
══════════════════════════════════ */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    font-weight: 600 !important;
    border-left-width: 4px !important;
}

/* ══════════════════════════════════
   EXPANDER
══════════════════════════════════ */
[data-testid="stExpander"] {
    border: 1.5px solid var(--border) !important;
    border-radius: 14px !important;
    background: var(--bg-card) !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ══════════════════════════════════
   HR DIVIDER
══════════════════════════════════ */
hr { border-color: var(--border) !important; margin: 1.4rem 0 !important; }

/* ══════════════════════════════════
   PLOTLY
══════════════════════════════════ */
.js-plotly-plot .plotly { border-radius: 14px !important; overflow: hidden !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  SESSION STATE INITIALISATION
# ─────────────────────────────────────────────────────────────
def init_session_state():
    defaults = {
        "raw_df": None,
        "processed_df": None,
        "preprocessing_log": [],
        "label_encoders": {},
        "model": None,
        "model_name": None,
        "target_col": None,
        "feature_cols": [],
        "accuracy": None,
        "predictions": None,
        "prediction_df": None,
        "train_report": None,
        "conf_matrix": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ─────────────────────────────────────────────────────────────
#  HELPER: CARD WRAPPER
# ─────────────────────────────────────────────────────────────
def card(title: str, content_fn, *args, **kwargs):
    """Render a styled card with a heading and body content."""
    with st.container():
        st.markdown(f'<div class="card"><div class="card-title">{title}</div>', unsafe_allow_html=True)
        content_fn(*args, **kwargs)
        st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────
def load_dataset(uploaded_file) -> pd.DataFrame | None:
    """Parse CSV or Excel file into a DataFrame."""
    try:
        ext = uploaded_file.name.split(".")[-1].lower()
        if ext == "csv":
            return pd.read_csv(uploaded_file)
        elif ext in ("xls", "xlsx"):
            return pd.read_excel(uploaded_file)
        else:
            st.error("❌ Unsupported format. Please upload a **CSV** or **Excel** file.")
            return None
    except Exception as exc:
        st.error(f"❌ Failed to load file: {exc}")
        return None


# ─────────────────────────────────────────────────────────────
#  PREPROCESSING
# ─────────────────────────────────────────────────────────────
def preprocess_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict]:
    """
    Automatic preprocessing pipeline:
      1. Fill numeric NaN with column mean.
      2. Fill categorical NaN with column mode.
      3. Label-encode object columns.
    Returns (processed_df, log_messages, label_encoders).
    """
    df = df.copy()
    log: list[str] = []
    encoders: dict[str, LabelEncoder] = {}

    # ── Missing value imputation ──────────────────────────────
    numeric_cols   = df.select_dtypes(include=[np.number]).columns.tolist()
    categoric_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in numeric_cols:
        missing = df[col].isna().sum()
        if missing:
            df[col].fillna(df[col].mean(), inplace=True)
            log.append(f"🔢 **{col}**: filled {missing} missing numeric value(s) with mean ({df[col].mean():.4f})")

    for col in categoric_cols:
        missing = df[col].isna().sum()
        if missing:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            log.append(f"🔤 **{col}**: filled {missing} missing categorical value(s) with mode ('{mode_val}')")

    # ── Label encoding ────────────────────────────────────────
    for col in categoric_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        log.append(f"🏷️ **{col}**: label-encoded ({len(le.classes_)} unique classes → integers)")

    if not log:
        log.append("✅ Dataset is already clean — no preprocessing steps were needed.")

    return df, log, encoders


# ─────────────────────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────────────────────
def train_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_name: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Train chosen sklearn classifier.
    Returns dict with model, accuracy, classification report, confusion matrix.
    """
    X = df[feature_cols]
    y = df[target_col]

    # Only stratify when every class has ≥ 2 samples; otherwise sklearn raises ValueError
    min_class_count = y.value_counts().min()
    use_stratify = y.nunique() > 1 and min_class_count >= 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if use_stratify else None
    )

    if model_name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=random_state, n_jobs=-1)
    else:  # Logistic Regression
        clf = LogisticRegression(max_iter=1000, random_state=random_state)

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    acc    = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    cm     = confusion_matrix(y_test, preds)

    return {"model": clf, "accuracy": acc, "report": report, "cm": cm}


# ─────────────────────────────────────────────────────────────
#  PREDICTION
# ─────────────────────────────────────────────────────────────
def predict(model, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Run inference on a DataFrame and return predicted labels."""
    X = df[feature_cols]
    return model.predict(X)


# ─────────────────────────────────────────────────────────────
#  PLOTLY THEME HELPER
# ─────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#f7f9ff",
    font=dict(family="Nunito, sans-serif", color="#1e2a45", size=13),
    margin=dict(l=20, r=20, t=50, b=20),
    xaxis=dict(gridcolor="#dde4f5", linecolor="#dde4f5", showgrid=True),
    yaxis=dict(gridcolor="#dde4f5", linecolor="#dde4f5", showgrid=True),
    colorway=["#4f6ef7", "#22c9a5", "#ff6b6b", "#ffb84d", "#a29bfe"],
    title_font=dict(size=15, color="#1e2a45", family="Nunito"),
)

COLOR_SCALE = [
    [0.0, "#eef1ff"],
    [0.4, "#7c9cff"],
    [0.7, "#4f6ef7"],
    [1.0, "#22c9a5"],
]


# ─────────────────────────────────────────────────────────────
#  SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────
def render_sidebar() -> str:
    with st.sidebar:
        # ── Brand ─────────────────────────────────────────────
        st.markdown(
            """
            <div style='text-align:center;padding:1.4rem 0 1rem'>
                <div style='font-size:2.8rem;line-height:1'>🤖</div>
                <div style='font-size:1.25rem;font-weight:800;color:#1e2a45;margin-top:.4rem'>ML Dashboard</div>
                <div style='font-size:.75rem;color:#7a869a;margin-top:.2rem;font-weight:600'>
                    Production Suite v1.0
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<hr style='margin:.2rem 0 .9rem'/>", unsafe_allow_html=True)

        # ── Navigation ─────────────────────────────────────────
        pages = [
            "🏠  Home",
            "📂  Upload Dataset",
            "🔍  Data Analysis",
            "🧠  Model Training",
            "🎯  Prediction",
            "📊  Results",
        ]
        selected = st.radio("Navigation", pages, label_visibility="collapsed")

        # ── Dataset status ─────────────────────────────────────
        st.markdown("<hr style='margin:.9rem 0'/>", unsafe_allow_html=True)

        if st.session_state.raw_df is not None:
            r, c = st.session_state.raw_df.shape
            st.markdown(
                f"""
                <div class='pill pill-green'>
                    <div class='pill-title' style='color:#13a678'>✅ Dataset Loaded</div>
                    <div class='pill-sub'>{r:,} rows &nbsp;·&nbsp; {c} columns</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class='pill pill-red'>
                    <div class='pill-title' style='color:#e05252'>⚠️ No Dataset</div>
                    <div class='pill-sub'>Upload a CSV or Excel file.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if st.session_state.model is not None:
            st.markdown(
                f"""
                <div class='pill pill-blue'>
                    <div class='pill-title' style='color:#4f6ef7'>🧠 Model Ready</div>
                    <div class='pill-sub'>{st.session_state.model_name}
                         &nbsp;·&nbsp; Acc {st.session_state.accuracy:.1%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Reset ──────────────────────────────────────────────
        st.markdown("<div style='height:1rem'/>", unsafe_allow_html=True)
        if st.button("🔄  Reset Session", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            init_session_state()
            st.rerun()

        # ── Footer ─────────────────────────────────────────────
        st.markdown(
            """
            <div style='position:fixed;bottom:1.2rem;left:0;width:260px;
                        text-align:center;font-size:.72rem;color:#b0bbd4;font-weight:600'>
                Built with ❤️ using Streamlit
            </div>
            """,
            unsafe_allow_html=True,
        )

    return selected.split("  ")[-1]


# ─────────────────────────────────────────────────────────────
#  PAGE: HOME
# ─────────────────────────────────────────────────────────────
def page_home():
    # ── Hero banner ───────────────────────────────────────────
    st.markdown(
        """
        <div class='page-header'>
            <h2>👋 Welcome to ML Dashboard</h2>
            <p>An end-to-end machine learning pipeline — upload data, preprocess,
               train models, predict, and export results, all in one place.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Feature tiles ─────────────────────────────────────────
    features = [
        ("📂", "Upload Dataset",   "CSV & Excel support with instant preview and session storage."),
        ("⚙️", "Auto Preprocess",  "Smart imputation (mean/mode) + label encoding with full audit log."),
        ("📊", "Data Analysis",    "Interactive Plotly charts: scatter, histogram, heatmap."),
        ("🧠", "Model Training",   "Random Forest & Logistic Regression with accuracy metrics."),
        ("🎯", "Prediction",       "Run inference on full or new datasets with one click."),
        ("📥", "Export Results",   "Download prediction results as a ready-to-use CSV."),
    ]
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div class='feat-tile'>
                    <div class='feat-icon'>{icon}</div>
                    <div class='feat-name'>{title}</div>
                    <div class='feat-desc'>{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:1.2rem'/>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Quick-start guide ─────────────────────────────────────
    st.markdown(
        "<div style='font-size:1.1rem;font-weight:800;margin-bottom:.8rem;color:#1e2a45'>"
        "🚀 Quick-Start Guide</div>",
        unsafe_allow_html=True,
    )
    steps = [
        ("1", "Upload Dataset",   "Go to **Upload Dataset** and load a CSV or Excel file."),
        ("2", "Analyse Data",     "Explore your data with interactive charts in **Data Analysis**."),
        ("3", "Train a Model",    "Pick target, features, and algorithm in **Model Training**."),
        ("4", "Get Predictions",  "Run inference on new data via **Prediction**."),
        ("5", "Download Results", "Review and export your predictions from **Results**."),
    ]
    for num, title, desc in steps:
        st.markdown(
            f"""
            <div class='step-row'>
                <div class='step-num'>{num}</div>
                <div>
                    <div class='step-body-title'>{title}</div>
                    <div class='step-body-desc'>{desc}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────
#  PAGE: UPLOAD DATASET
# ─────────────────────────────────────────────────────────────
def page_upload():
    st.markdown(
        """
        <div class='page-header'>
            <h2>📂 Upload Dataset</h2>
            <p>Supported formats: CSV · XLS · XLSX — stored in session for all pipeline steps.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Drop your file here or click to browse",
        type=["csv", "xls", "xlsx"],
        label_visibility="collapsed",
    )

    if uploaded:
        df = load_dataset(uploaded)
        if df is not None:
            st.session_state.raw_df = df
            # Reset downstream state on new upload
            for key in ("processed_df", "preprocessing_log", "label_encoders",
                        "model", "model_name", "target_col", "feature_cols",
                        "accuracy", "predictions", "prediction_df", "train_report", "conf_matrix"):
                st.session_state[key] = [] if key in ("preprocessing_log", "feature_cols") else None

            st.success(f"✅ **{uploaded.name}** loaded successfully — {df.shape[0]:,} rows × {df.shape[1]} columns")

    df = st.session_state.raw_df
    if df is None:
        st.info("👆 Upload a dataset to get started.")
        return

    # ── Summary metrics ───────────────────────────────────────
    st.markdown("#### Dataset Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows",          f"{df.shape[0]:,}")
    m2.metric("Columns",       df.shape[1])
    m3.metric("Missing Values",int(df.isna().sum().sum()))
    m4.metric("Numeric Cols",  len(df.select_dtypes(include=np.number).columns))

    st.markdown("<div style='height:.5rem'/>", unsafe_allow_html=True)

    # ── Preview ───────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📋 Preview (Head)", "📝 Data Types", "📈 Stats"])

    with tab1:
        st.dataframe(df.head(10), use_container_width=True, height=320)

    with tab2:
        dtype_df = pd.DataFrame({
            "Column": df.columns,
            "Dtype":  df.dtypes.astype(str).values,
            "Non-Null": df.notna().sum().values,
            "Null": df.isna().sum().values,
            "Unique": df.nunique().values,
        })
        st.dataframe(dtype_df, use_container_width=True, height=320)

    with tab3:
        st.dataframe(df.describe(include="all").T, use_container_width=True, height=320)

    # ── Preprocessing button ──────────────────────────────────
    st.markdown("<div style='height:.5rem'/>", unsafe_allow_html=True)
    st.markdown("#### ⚙️ Preprocess Dataset")
    st.caption("Fills missing values and label-encodes categorical columns — required before training.")

    if st.button("▶ Run Preprocessing"):
        with st.spinner("Processing …"):
            proc_df, log, encoders = preprocess_dataframe(df)
            st.session_state.processed_df      = proc_df
            st.session_state.preprocessing_log = log
            st.session_state.label_encoders    = encoders
        st.success("✅ Preprocessing complete!")

    if st.session_state.processed_df is not None:
        st.markdown("#### 📋 Preprocessing Log")
        with st.expander("View steps applied", expanded=True):
            for step in st.session_state.preprocessing_log:
                st.markdown(f"- {step}")
        st.markdown("**Processed Dataset Preview**")
        st.dataframe(st.session_state.processed_df.head(8), use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  PAGE: DATA ANALYSIS
# ─────────────────────────────────────────────────────────────
def page_analysis():
    st.markdown(
        """
        <div class='page-header'>
            <h2>🔍 Data Analysis</h2>
            <p>Explore distributions, relationships, and correlations with interactive Plotly charts.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = st.session_state.raw_df
    if df is None:
        st.warning("⚠️ Please upload a dataset first.")
        return

    # ── KPI metrics ───────────────────────────────────────────
    rows, cols_n = df.shape
    total_missing = int(df.isna().sum().sum())
    dup_rows      = int(df.duplicated().sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Rows",     f"{rows:,}")
    k2.metric("Total Columns",  cols_n)
    k3.metric("Missing Values", total_missing,
              delta=f"-{total_missing}" if total_missing else None,
              delta_color="inverse")
    k4.metric("Duplicate Rows", dup_rows,
              delta=f"-{dup_rows}" if dup_rows else None,
              delta_color="inverse")

    st.markdown("<div style='height:.5rem'/>", unsafe_allow_html=True)

    # ── Full dataset table ────────────────────────────────────
    with st.expander("📋 Full Dataset (scrollable)", expanded=False):
        st.dataframe(df, use_container_width=True, height=400)

    st.markdown("---")
    st.markdown("#### 📊 Interactive Visualisations")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    all_cols     = df.columns.tolist()

    if len(numeric_cols) < 2:
        st.info("Need at least 2 numeric columns for charts.")
        return

    tab_scatter, tab_hist, tab_heat = st.tabs(["🔵 Scatter Plot", "📊 Histogram", "🌡️ Correlation Heatmap"])

    # ── Scatter ───────────────────────────────────────────────
    with tab_scatter:
        c1, c2, c3 = st.columns(3)
        x_col   = c1.selectbox("X Axis",    numeric_cols, index=0, key="sc_x")
        y_col   = c2.selectbox("Y Axis",    numeric_cols, index=min(1, len(numeric_cols)-1), key="sc_y")
        col_by  = c3.selectbox("Color By",  ["None"] + all_cols, key="sc_color")

        color_arg = None if col_by == "None" else col_by
        fig = px.scatter(
            df, x=x_col, y=y_col, color=color_arg,
            title=f"{x_col} vs {y_col}",
            color_continuous_scale="Viridis",
            opacity=0.75,
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    # ── Histogram ─────────────────────────────────────────────
    with tab_hist:
        h1, h2 = st.columns(2)
        hist_col  = h1.selectbox("Column",    numeric_cols, key="hist_col")
        bin_count = h2.slider("Bins", 5, 100, 30, key="hist_bins")

        fig = px.histogram(
            df, x=hist_col, nbins=bin_count,
            title=f"Distribution of {hist_col}",
            color_discrete_sequence=["#6c63ff"],
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    # ── Correlation heatmap ───────────────────────────────────
    with tab_heat:
        corr = df[numeric_cols].corr()
        fig  = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                colorscale=COLOR_SCALE,
                text=np.round(corr.values, 2),
                texttemplate="%{text}",
                hoverongaps=False,
                showscale=True,
            )
        )
        fig.update_layout(title="Correlation Matrix", height=520, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  PAGE: MODEL TRAINING
# ─────────────────────────────────────────────────────────────
def page_training():
    st.markdown(
        """
        <div class='page-header'>
            <h2>🧠 Model Training</h2>
            <p>Select your target, choose features, pick an algorithm, and train with one click.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    proc_df = st.session_state.processed_df
    if proc_df is None:
        st.warning("⚠️ Preprocess your dataset first (Upload Dataset → Run Preprocessing).")
        return

    cols_all = proc_df.columns.tolist()

    # ── Feature / target selection ────────────────────────────
    st.markdown("#### 🎛️ Feature Selection")
    s1, s2 = st.columns(2)
    target_col   = s1.selectbox("Target Column (Y)", cols_all, key="train_target")
    feature_cols = s2.multiselect(
        "Feature Columns (X)",
        [c for c in cols_all if c != target_col],
        default=[c for c in cols_all if c != target_col][:min(6, len(cols_all)-1)],
        key="train_features",
    )

    st.markdown("---")
    st.markdown("#### ⚙️ Training Configuration")
    cfg1, cfg2, cfg3 = st.columns(3)
    model_name  = cfg1.selectbox("Algorithm",  ["Random Forest", "Logistic Regression"], key="model_select")
    test_size   = cfg2.slider("Test Split (%)", 10, 40, 20, step=5, key="test_split") / 100
    random_seed = cfg3.number_input("Random Seed", value=42, step=1, key="rand_seed")

    st.markdown("<div style='height:.3rem'/>", unsafe_allow_html=True)

    # ── Validation ────────────────────────────────────────────
    if not feature_cols:
        st.error("❌ Select at least one feature column.")
        return
    if target_col in feature_cols:
        st.error("❌ Target column must not be in the feature list.")
        return

    # ── Train ─────────────────────────────────────────────────
    if st.button("🚀 Train Model", use_container_width=False):
        with st.spinner(f"Training {model_name} …"):
            result = train_model(
                proc_df, feature_cols, target_col,
                model_name, test_size, int(random_seed)
            )

        st.session_state.model       = result["model"]
        st.session_state.model_name  = model_name
        st.session_state.accuracy    = result["accuracy"]
        st.session_state.target_col  = target_col
        st.session_state.feature_cols = feature_cols
        st.session_state.train_report = result["report"]
        st.session_state.conf_matrix  = result["cm"]

        st.success(f"✅ {model_name} trained successfully!")

    # ── Results ───────────────────────────────────────────────
    if st.session_state.model is None:
        return

    st.markdown("---")
    st.markdown("#### 📈 Training Results")

    a1, a2, a3 = st.columns(3)
    a1.metric("Accuracy",   f"{st.session_state.accuracy:.2%}")
    a2.metric("Algorithm",  st.session_state.model_name)
    a3.metric("Test Split", f"{int(test_size*100)}%")

    # Classification report table
    report_df = (
        pd.DataFrame(st.session_state.train_report)
        .T.drop(["accuracy", "macro avg", "weighted avg"], errors="ignore")
        .round(3)
    )
    st.markdown("##### Classification Report")
    st.dataframe(report_df, use_container_width=True)

    # Confusion matrix
    cm  = st.session_state.conf_matrix
    fig = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale=COLOR_SCALE,
        title="Confusion Matrix",
        labels=dict(x="Predicted", y="Actual"),
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Feature importance (RF only)
    if st.session_state.model_name == "Random Forest":
        st.markdown("##### 🌲 Feature Importance")
        importances = st.session_state.model.feature_importances_
        imp_df = (
            pd.DataFrame({"Feature": st.session_state.feature_cols, "Importance": importances})
            .sort_values("Importance", ascending=True)
        )
        fig2 = px.bar(
            imp_df, x="Importance", y="Feature", orientation="h",
            title="Feature Importance", color="Importance",
            color_continuous_scale=COLOR_SCALE,
        )
        fig2.update_layout(**PLOTLY_LAYOUT, height=max(300, len(imp_df) * 32 + 80))
        st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  PAGE: PREDICTION
# ─────────────────────────────────────────────────────────────
def page_prediction():
    st.markdown(
        """
        <div class='page-header'>
            <h2>🎯 Prediction</h2>
            <p>Run inference on the training dataset or upload a new file for batch predictions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.model is None:
        st.warning("⚠️ Train a model first (Model Training page).")
        return

    pred_mode = st.radio(
        "Prediction Source",
        ["Use Trained Dataset", "Upload New Dataset"],
        horizontal=True,
    )

    if pred_mode == "Use Trained Dataset":
        pred_df = st.session_state.processed_df
        if pred_df is None:
            st.warning("⚠️ Processed dataset not found.")
            return
        st.caption(f"Dataset: {pred_df.shape[0]:,} rows × {pred_df.shape[1]} columns")

    else:  # Upload new
        new_file = st.file_uploader("Upload prediction file (CSV / Excel)", type=["csv", "xls", "xlsx"])
        if new_file is None:
            st.info("Upload a file to predict on.")
            return

        raw_new = load_dataset(new_file)
        if raw_new is None:
            return

        # Preprocess new file
        with st.spinner("Preprocessing new dataset …"):
            pred_df, _, _ = preprocess_dataframe(raw_new)

        st.success(f"✅ New dataset loaded: {pred_df.shape[0]:,} rows")

    # Validate features
    missing_feats = [f for f in st.session_state.feature_cols if f not in pred_df.columns]
    if missing_feats:
        st.error(f"❌ Columns missing in prediction data: {missing_feats}")
        return

    if st.button("🔮 Run Prediction", use_container_width=False):
        with st.spinner("Running inference …"):
            preds = predict(st.session_state.model, pred_df, st.session_state.feature_cols)

        out_df = pred_df.copy()
        out_df["Prediction"] = preds
        st.session_state.predictions  = preds
        st.session_state.prediction_df = out_df

        st.success(f"✅ Prediction complete on {len(preds):,} samples.")

    if st.session_state.prediction_df is not None:
        st.markdown("---")
        st.markdown("#### 🔍 Prediction Preview")

        val_counts = pd.Series(st.session_state.predictions).value_counts().reset_index()
        val_counts.columns = ["Class", "Count"]
        fig = px.bar(
            val_counts, x="Class", y="Count",
            title="Predicted Class Distribution",
            color="Count",
            color_continuous_scale=COLOR_SCALE,
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=360)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(st.session_state.prediction_df.head(20), use_container_width=True, height=340)


# ─────────────────────────────────────────────────────────────
#  PAGE: RESULTS
# ─────────────────────────────────────────────────────────────
def page_results():
    st.markdown(
        """
        <div class='page-header'>
            <h2>📊 Results</h2>
            <p>Review predictions, inspect class distributions, and download results as CSV.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pred_df = st.session_state.prediction_df
    if pred_df is None:
        st.warning("⚠️ No predictions available yet. Run predictions first.")
        return

    # ── Summary ───────────────────────────────────────────────
    preds = st.session_state.predictions
    r1, r2, r3 = st.columns(3)
    r1.metric("Total Predictions", f"{len(preds):,}")
    r2.metric("Unique Classes",    int(pd.Series(preds).nunique()))
    r3.metric("Model Used",        st.session_state.model_name)

    st.markdown("---")

    # ── Full results table ────────────────────────────────────
    st.markdown("#### 📋 Full Results Table")
    st.dataframe(pred_df, use_container_width=True, height=420)

    # ── Download ──────────────────────────────────────────────
    csv_buffer = io.StringIO()
    pred_df.to_csv(csv_buffer, index=False)
    st.markdown("<div style='height:.5rem'/>", unsafe_allow_html=True)
    st.download_button(
        label="⬇️ Download Results as CSV",
        data=csv_buffer.getvalue().encode("utf-8"),
        file_name="ml_dashboard_predictions.csv",
        mime="text/csv",
        use_container_width=False,
    )

    # ── Distribution chart ────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📊 Prediction Distribution")
    vc = pd.Series(preds).value_counts().reset_index()
    vc.columns = ["Class", "Count"]
    vc["Percentage"] = (vc["Count"] / vc["Count"].sum() * 100).round(1)

    fig = px.pie(
        vc, values="Count", names="Class",
        title="Class Distribution of Predictions",
        color_discrete_sequence=["#4f6ef7", "#22c9a5", "#ff6b6b", "#ffb84d", "#a29bfe"],
        hole=0.4,
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(vc, use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  MAIN ROUTER
# ─────────────────────────────────────────────────────────────
def main():
    page = render_sidebar()

    if page == "Home":
        page_home()
    elif page == "Upload Dataset":
        page_upload()
    elif page == "Data Analysis":
        page_analysis()
    elif page == "Model Training":
        page_training()
    elif page == "Prediction":
        page_prediction()
    elif page == "Results":
        page_results()
    else:
        page_home()


if __name__ == "__main__":
    main()