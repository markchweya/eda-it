# app.py
# =============================================================================
# EDA — A single-file Streamlit app for Exploratory Data Analysis (+ optional ML metrics)
#
# NOTES (what you'll get)
# - Upload dataset (CSV / Excel / Parquet)
# - View + profile: shape, types, summary stats, unique counts, memory
# - Data quality checks: missing values, duplicates, basic fixes (optional)
# - Visual EDA: univariate + bivariate plots, correlation heatmap, outlier scan
# - Optional modeling tab: train/test split + F1/accuracy/precision/recall (+ ROC-AUC when possible)
# - Export a lightweight EDA report (Markdown)
#
# Run:
#   pip install streamlit pandas numpy plotly scikit-learn openpyxl pyarrow
#   streamlit run app.py
# =============================================================================

import io
import textwrap
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC


# =============================================================================
# NOTES (Step 1) — App look & layout
# - Light colors + soft cards
# - Sidebar navigation
# - Session state to hold dataset + user choices
# =============================================================================

st.set_page_config(
    page_title="EDA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

LIGHT_CSS = """
<style>
:root {
  --bg: #f7f9fc;
  --panel: #ffffff;
  --text: #0f172a;
  --muted: #475569;
  --border: #e5e7eb;
  --accent: #3b82f6;
  --accent2: #22c55e;
  --warn: #f59e0b;
  --danger: #ef4444;
  --shadow: 0 10px 30px rgba(2, 8, 23, 0.06);
  --radius: 18px;
}

html, body, [class*="css"]  {
  background: var(--bg) !important;
  color: var(--text) !important;
}

.block-container {
  padding-top: 1.1rem;
  padding-bottom: 1.6rem;
  max-width: 1400px;
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #ffffff 0%, #f3f6ff 100%) !important;
  border-right: 1px solid var(--border);
}

.card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.0rem 1.1rem;
  box-shadow: var(--shadow);
}

.pill {
  display: inline-block;
  padding: 0.22rem 0.6rem;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: #f8fafc;
  color: var(--muted);
  font-size: 0.85rem;
  margin-right: 0.35rem;
}

.kpi {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 0.9rem 1.0rem;
  box-shadow: var(--shadow);
}
.kpi .label {
  color: var(--muted);
  font-size: 0.85rem;
  margin-bottom: 0.2rem;
}
.kpi .value {
  font-size: 1.35rem;
  font-weight: 700;
}

.small-note {
  color: var(--muted);
  font-size: 0.92rem;
}
.hr {
  height: 1px;
  background: var(--border);
  margin: 0.7rem 0;
}

.stButton>button {
  border-radius: 12px;
  border: 1px solid var(--border);
  background: white;
  box-shadow: 0 6px 16px rgba(2,8,23,0.06);
}
.stDownloadButton>button {
  border-radius: 12px;
}
</style>
"""
st.markdown(LIGHT_CSS, unsafe_allow_html=True)


# =============================================================================
# NOTES (Step 2) — Helpers (robust reading, profiling, safe plotting)
# =============================================================================

@dataclass
class DatasetInfo:
    rows: int
    cols: int
    mem_mb: float
    n_numeric: int
    n_categorical: int
    n_datetime: int
    n_bool: int


def _format_int(n: int) -> str:
    return f"{n:,}"


def _infer_datetime_columns(df: pd.DataFrame) -> list[str]:
    dt_cols = []
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            dt_cols.append(c)
    return dt_cols


def basic_info(df: pd.DataFrame) -> DatasetInfo:
    mem_mb = float(df.memory_usage(deep=True).sum() / (1024**2))
    n_numeric = int(sum(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns))
    n_bool = int(sum(pd.api.types.is_bool_dtype(df[c]) for c in df.columns))
    n_datetime = int(sum(pd.api.types.is_datetime64_any_dtype(df[c]) for c in df.columns))
    n_categorical = int(len(df.columns) - n_numeric - n_bool - n_datetime)
    return DatasetInfo(
        rows=int(df.shape[0]),
        cols=int(df.shape[1]),
        mem_mb=mem_mb,
        n_numeric=n_numeric,
        n_categorical=n_categorical,
        n_datetime=n_datetime,
        n_bool=n_bool,
    )


@st.cache_data(show_spinner=False)
def read_dataset(file_bytes: bytes, filename: str) -> pd.DataFrame:
    name = filename.lower()
    buf = io.BytesIO(file_bytes)

    if name.endswith(".csv"):
        # Try common encodings safely
        for enc in ["utf-8", "utf-8-sig", "latin-1"]:
            try:
                buf.seek(0)
                return pd.read_csv(buf, encoding=enc)
            except Exception:
                continue
        buf.seek(0)
        return pd.read_csv(buf, encoding_errors="ignore")

    if name.endswith(".xlsx") or name.endswith(".xls"):
        buf.seek(0)
        return pd.read_excel(buf)

    if name.endswith(".parquet"):
        buf.seek(0)
        return pd.read_parquet(buf)

    # Fall back (try CSV)
    buf.seek(0)
    return pd.read_csv(buf, encoding_errors="ignore")


def sample_df(df: pd.DataFrame, max_rows: int = 30_000, seed: int = 42) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.sample(max_rows, random_state=seed)


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum()
    pct = (miss / len(df)) * 100 if len(df) else 0
    out = pd.DataFrame({"missing": miss, "missing_%": pct}).sort_values("missing", ascending=False)
    out["missing_%"] = out["missing_%"].round(2)
    return out


def duplicates_count(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    return int(df.duplicated().sum())


def top_correlations(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return pd.DataFrame(columns=["var_1", "var_2", "corr_abs", "corr"])
    corr = num.corr(numeric_only=True)
    corr_vals = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().reset_index()
    corr_vals.columns = ["var_1", "var_2", "corr"]
    corr_vals["corr_abs"] = corr_vals["corr"].abs()
    corr_vals = corr_vals.sort_values("corr_abs", ascending=False).head(k)
    return corr_vals[["var_1", "var_2", "corr_abs", "corr"]].reset_index(drop=True)


def iqr_outlier_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = df[c].dropna()
        if s.empty:
            continue
        q1, q3 = np.percentile(s, 25), np.percentile(s, 75)
        iqr = q3 - q1
        if iqr == 0:
            rows.append((c, 0, q1, q3, 0.0))
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((df[c] < lo) | (df[c] > hi)).sum())
        rows.append((c, n_out, lo, hi, float(n_out / len(df) * 100)))
    out = pd.DataFrame(rows, columns=["column", "outliers", "lower_bound", "upper_bound", "outliers_%"])
    if out.empty:
        return out
    out["outliers_%"] = out["outliers_%"].round(2)
    return out.sort_values("outliers", ascending=False).reset_index(drop=True)


# =============================================================================
# NOTES (Step 3) — Sidebar: navigation + global controls
# =============================================================================

st.sidebar.markdown("## 📊 EDA")
st.sidebar.markdown(
    "<div class='small-note'>Upload a dataset and explore it fast — with clean visuals and optional ML metrics.</div>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("<div class='hr'></div>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigate",
    [
        "1) Upload & Overview",
        "2) Data Quality",
        "3) Univariate EDA",
        "4) Bivariate EDA",
        "5) Correlation & Outliers",
        "6) Modeling (Optional)",
        "7) Export Report",
    ],
)

st.sidebar.markdown("<div class='hr'></div>", unsafe_allow_html=True)
max_plot_rows = st.sidebar.slider("Max rows used for plots (sampling)", 2_000, 200_000, 30_000, step=1_000)
seed = st.sidebar.number_input("Sampling seed", min_value=0, value=42, step=1)

st.sidebar.markdown("<div class='hr'></div>", unsafe_allow_html=True)
st.sidebar.markdown("**Tip:** Keep plots fast by sampling big datasets.")


# =============================================================================
# NOTES (Step 4) — Session state storage
# =============================================================================

if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
if "df_work" not in st.session_state:
    st.session_state.df_work = None
if "fixes" not in st.session_state:
    st.session_state.fixes = {"drop_duplicates": False, "missing_strategy": "None"}


def require_df() -> bool:
    if st.session_state.df_work is None:
        st.markdown("<div class='card'>Upload a dataset first (go to <b>1) Upload & Overview</b>).</div>", unsafe_allow_html=True)
        return False
    return True


# =============================================================================
# NOTES (Step 5) — Page 1: Upload & Overview
# =============================================================================

if page == "1) Upload & Overview":
    st.markdown("### 1) Upload & Overview")
    st.markdown(
        "<div class='card'>"
        "<span class='pill'>Upload</span>"
        "<span class='pill'>Preview</span>"
        "<span class='pill'>Types</span>"
        "<span class='pill'>Summary</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader("Upload your dataset", type=["csv", "xlsx", "xls", "parquet"])

    colA, colB = st.columns([1.2, 1])

    with colA:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Notes**")
        st.markdown(
            "- CSV/Excel/Parquet supported.\n"
            "- After upload, you’ll work on a *copy* of the data (so you can apply fixes safely).\n"
            "- For very large files, plots will use a sample for speed."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with colB:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Quick controls**")
        reset = st.button("Reset session (clear dataset)")
        if reset:
            st.session_state.df_raw = None
            st.session_state.df_work = None
            st.session_state.fixes = {"drop_duplicates": False, "missing_strategy": "None"}
            st.success("Session reset. Upload again.")
        st.markdown("</div>", unsafe_allow_html=True)

    if uploaded is not None:
        try:
            df = read_dataset(uploaded.getvalue(), uploaded.name)

            # Light touch: attempt datetime parsing for obvious columns
            df2 = df.copy()
            for c in df2.columns:
                if df2[c].dtype == "object":
                    # only attempt if it looks like dates (cheap heuristic)
                    sample_vals = df2[c].dropna().astype(str).head(25)
                    if len(sample_vals) and sample_vals.str.contains(r"\d{4}|\d{2}[/\-]\d{2}", regex=True).mean() > 0.6:
                        try:
                            df2[c] = pd.to_datetime(df2[c], errors="ignore")
                        except Exception:
                            pass

            st.session_state.df_raw = df2
            st.session_state.df_work = df2.copy()

            info = basic_info(st.session_state.df_work)

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.markdown(f"<div class='kpi'><div class='label'>Rows</div><div class='value'>{_format_int(info.rows)}</div></div>", unsafe_allow_html=True)
            with k2:
                st.markdown(f"<div class='kpi'><div class='label'>Columns</div><div class='value'>{_format_int(info.cols)}</div></div>", unsafe_allow_html=True)
            with k3:
                st.markdown(f"<div class='kpi'><div class='label'>Memory</div><div class='value'>{info.mem_mb:.2f} MB</div></div>", unsafe_allow_html=True)
            with k4:
                st.markdown(
                    f"<div class='kpi'><div class='label'>Types</div>"
                    f"<div class='value'>{info.n_numeric} num · {info.n_categorical} cat</div></div>",
                    unsafe_allow_html=True,
                )

            st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

            tab1, tab2, tab3 = st.tabs(["Preview", "Column Profile", "Summary Stats"])

            with tab1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("**Dataset preview**")
                st.dataframe(st.session_state.df_work, use_container_width=True, height=420)
                st.markdown("</div>", unsafe_allow_html=True)

            with tab2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                dtypes = st.session_state.df_work.dtypes.astype(str)
                nunique = st.session_state.df_work.nunique(dropna=True)
                miss = st.session_state.df_work.isna().sum()
                prof = pd.DataFrame(
                    {
                        "dtype": dtypes,
                        "unique": nunique,
                        "missing": miss,
                        "missing_%": (miss / len(st.session_state.df_work) * 100).round(2) if len(st.session_state.df_work) else 0,
                    }
                ).sort_values(["missing", "unique"], ascending=False)
                st.dataframe(prof, use_container_width=True, height=420)
                st.markdown("</div>", unsafe_allow_html=True)

            with tab3:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("**Numeric summary**")
                num = st.session_state.df_work.select_dtypes(include=[np.number])
                if num.shape[1] == 0:
                    st.info("No numeric columns found.")
                else:
                    st.dataframe(num.describe().T, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Could not read file: {e}")


# =============================================================================
# NOTES (Step 6) — Page 2: Data Quality (missing values, duplicates, simple fixes)
# =============================================================================

elif page == "2) Data Quality":
    st.markdown("### 2) Data Quality")
    if not require_df():
        st.stop()

    df = st.session_state.df_work

    st.markdown(
        "<div class='card'>"
        "<span class='pill'>Missing</span>"
        "<span class='pill'>Duplicates</span>"
        "<span class='pill'>Fixes</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.05, 0.95])

    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Missing values by column**")
        mt = missing_table(df)
        st.dataframe(mt, use_container_width=True, height=420)

        # Missing plot (top N)
        top_n = st.slider("Show top N columns (by missing)", 5, min(50, len(df.columns)), min(20, len(df.columns)))
        mt_top = mt.head(top_n).reset_index().rename(columns={"index": "column"})
        fig = px.bar(mt_top, x="column", y="missing", hover_data=["missing_%"])
        fig.update_layout(margin=dict(l=10, r=10, t=35, b=10), xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Duplicates**")
        dup = duplicates_count(df)
        st.markdown(f"- Duplicate rows: **{_format_int(dup)}**")
        if dup > 0:
            st.markdown("<div class='small-note'>You can drop duplicates below (applies to the working copy).</div>", unsafe_allow_html=True)
        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

        st.markdown("**Quick fixes (optional)**")
        drop_dups = st.checkbox("Drop duplicate rows", value=st.session_state.fixes["drop_duplicates"])
        missing_strategy = st.selectbox(
            "Missing value handling (working copy)",
            ["None", "Drop rows with any missing", "Fill numeric with median + categorical with mode"],
            index=["None", "Drop rows with any missing", "Fill numeric with median + categorical with mode"].index(
                st.session_state.fixes.get("missing_strategy", "None")
            ),
        )

        apply = st.button("Apply fixes to working copy")
        if apply:
            new_df = df.copy()

            if drop_dups:
                new_df = new_df.drop_duplicates()

            if missing_strategy == "Drop rows with any missing":
                new_df = new_df.dropna(axis=0, how="any")

            elif missing_strategy == "Fill numeric with median + categorical with mode":
                num_cols = new_df.select_dtypes(include=[np.number]).columns.tolist()
                cat_cols = [c for c in new_df.columns if c not in num_cols]

                for c in num_cols:
                    med = new_df[c].median()
                    new_df[c] = new_df[c].fillna(med)

                for c in cat_cols:
                    if new_df[c].isna().any():
                        mode = new_df[c].mode(dropna=True)
                        fill = mode.iloc[0] if len(mode) else "Unknown"
                        new_df[c] = new_df[c].fillna(fill)

            st.session_state.df_work = new_df
            st.session_state.fixes = {"drop_duplicates": drop_dups, "missing_strategy": missing_strategy}
            st.success("Fixes applied to working copy.")
            st.rerun()

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        st.markdown("**Column type suggestions**")
        st.markdown(
            "<div class='small-note'>If a date column is still 'object', consider cleaning it in pandas before upload, "
            "or adjust parsing rules in the upload step.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# NOTES (Step 7) — Page 3: Univariate EDA (distributions for one variable)
# =============================================================================

elif page == "3) Univariate EDA":
    st.markdown("### 3) Univariate EDA")
    if not require_df():
        st.stop()

    df = st.session_state.df_work
    dfp = sample_df(df, max_rows=max_plot_rows, seed=seed)

    st.markdown(
        "<div class='card'>"
        "<span class='pill'>Distributions</span>"
        "<span class='pill'>Counts</span>"
        "<span class='pill'>Boxplots</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([0.9, 1.1])

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Pick a column**")
        col = st.selectbox("Column", df.columns.tolist())
        st.markdown("<div class='small-note'>Plots use a sample if the dataset is large.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        series = dfp[col]

        if pd.api.types.is_numeric_dtype(series):
            st.markdown("**Numeric distribution**")
            bins = st.slider("Bins", 10, 120, 40)
            fig = px.histogram(dfp, x=col, nbins=bins, marginal="box")
            fig.update_layout(margin=dict(l=10, r=10, t=35, b=10))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Quick stats**")
            stats = pd.DataFrame(
                {
                    "stat": ["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
                    "value": [
                        series.count(),
                        series.mean(),
                        series.std(),
                        series.min(),
                        series.quantile(0.25),
                        series.quantile(0.50),
                        series.quantile(0.75),
                        series.max(),
                    ],
                }
            )
            st.dataframe(stats, use_container_width=True, hide_index=True)

        elif pd.api.types.is_datetime64_any_dtype(series):
            st.markdown("**Datetime distribution**")
            # Show counts by day/week/month
            gran = st.selectbox("Granularity", ["Day", "Week", "Month"])
            s = pd.to_datetime(series, errors="coerce").dropna()
            if s.empty:
                st.info("No valid datetime values to plot.")
            else:
                if gran == "Day":
                    grp = s.dt.date.value_counts().sort_index()
                elif gran == "Week":
                    grp = s.dt.to_period("W").astype(str).value_counts().sort_index()
                else:
                    grp = s.dt.to_period("M").astype(str).value_counts().sort_index()
                plot_df = grp.reset_index()
                plot_df.columns = ["period", "count"]
                fig = px.line(plot_df, x="period", y="count", markers=True)
                fig.update_layout(margin=dict(l=10, r=10, t=35, b=10), xaxis_tickangle=-35)
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.markdown("**Categorical counts**")
            top_k = st.slider("Top categories to show", 5, 50, 20)
            vc = series.astype(str).value_counts(dropna=False).head(top_k)
            plot_df = vc.reset_index()
            plot_df.columns = ["category", "count"]
            fig = px.bar(plot_df, x="category", y="count")
            fig.update_layout(margin=dict(l=10, r=10, t=35, b=10), xaxis_tickangle=-35)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Most frequent values**")
            st.dataframe(plot_df, use_container_width=True, height=260)

        st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# NOTES (Step 8) — Page 4: Bivariate EDA (relationships between two variables)
# =============================================================================

elif page == "4) Bivariate EDA":
    st.markdown("### 4) Bivariate EDA")
    if not require_df():
        st.stop()

    df = st.session_state.df_work
    dfp = sample_df(df, max_rows=max_plot_rows, seed=seed)

    st.markdown(
        "<div class='card'>"
        "<span class='pill'>Scatter</span>"
        "<span class='pill'>Box</span>"
        "<span class='pill'>Bar</span>"
        "<span class='pill'>Trend</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([0.95, 1.05])

    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        cols = df.columns.tolist()
        x = st.selectbox("X (feature)", cols, index=0)
        y = st.selectbox("Y (feature)", cols, index=min(1, len(cols)-1))

        color = st.selectbox("Color (optional)", ["None"] + cols)
        facet = st.selectbox("Facet (optional)", ["None"] + cols)

        plot_type = st.selectbox(
            "Plot type",
            [
                "Auto (recommended)",
                "Scatter",
                "Line (sorted by X)",
                "Box (Y by X)",
                "Violin (Y by X)",
                "Bar (mean Y by X)",
            ],
        )

        st.markdown("<div class='small-note'>If the plot looks weird, try swapping X and Y or choose a different plot type.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        c_arg = None if color == "None" else color
        f_arg = None if facet == "None" else facet

        x_is_num = pd.api.types.is_numeric_dtype(dfp[x])
        y_is_num = pd.api.types.is_numeric_dtype(dfp[y])

        auto = plot_type.startswith("Auto")

        try:
            if auto:
                if x_is_num and y_is_num:
                    fig = px.scatter(dfp, x=x, y=y, color=c_arg, trendline="ols" if len(dfp) <= 50_000 else None)
                elif (not x_is_num) and y_is_num:
                    fig = px.box(dfp, x=x, y=y, color=c_arg)
                elif x_is_num and (not y_is_num):
                    # For numeric X and categorical Y, show box of X by Y (swap)
                    fig = px.box(dfp, x=y, y=x, color=c_arg)
                else:
                    # both categorical -> counts heatmap
                    ct = pd.crosstab(dfp[x].astype(str), dfp[y].astype(str)).reset_index().melt(id_vars=[x], var_name=y, value_name="count")
                    fig = px.density_heatmap(ct, x=x, y=y, z="count", histfunc="sum")

            elif plot_type == "Scatter":
                fig = px.scatter(dfp, x=x, y=y, color=c_arg)

            elif plot_type == "Line (sorted by X)":
                tmp = dfp[[x, y] + ([c_arg] if c_arg else [])].copy()
                if not x_is_num:
                    st.warning("Line plot works best when X is numeric or datetime. Trying anyway.")
                tmp = tmp.sort_values(by=x)
                fig = px.line(tmp, x=x, y=y, color=c_arg)

            elif plot_type == "Box (Y by X)":
                fig = px.box(dfp, x=x, y=y, color=c_arg)

            elif plot_type == "Violin (Y by X)":
                fig = px.violin(dfp, x=x, y=y, color=c_arg, box=True, points="outliers")

            else:  # Bar mean
                tmp = dfp[[x, y]].copy()
                tmp = tmp.dropna()
                if tmp.empty:
                    st.info("No data to plot after dropping missing values.")
                    st.stop()
                grp = tmp.groupby(x)[y].mean().reset_index().sort_values(by=y, ascending=False).head(50)
                fig = px.bar(grp, x=x, y=y)

            if f_arg:
                # Simple facet strategy: only facet columns if not too many unique values
                nun = dfp[f_arg].nunique(dropna=True)
                if nun <= 12:
                    fig = px.scatter(dfp, x=x, y=y, color=c_arg, facet_col=f_arg) if (x_is_num and y_is_num) else fig
                else:
                    st.info(f"Facet skipped: '{f_arg}' has {nun} unique values (too many).")

            fig.update_layout(margin=dict(l=10, r=10, t=35, b=10))
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Could not build plot: {e}")

        st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# NOTES (Step 9) — Page 5: Correlation & Outliers
# =============================================================================

elif page == "5) Correlation & Outliers":
    st.markdown("### 5) Correlation & Outliers")
    if not require_df():
        st.stop()

    df = st.session_state.df_work
    dfp = sample_df(df, max_rows=max_plot_rows, seed=seed)

    st.markdown(
        "<div class='card'>"
        "<span class='pill'>Correlation</span>"
        "<span class='pill'>Top pairs</span>"
        "<span class='pill'>IQR outliers</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1.1, 0.9])

    with c1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Correlation heatmap (numeric columns)**")
        num = dfp.select_dtypes(include=[np.number])
        if num.shape[1] < 2:
            st.info("Need at least 2 numeric columns for correlation.")
        else:
            method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"])
            corr = num.corr(method=method, numeric_only=True)

            fig = px.imshow(
                corr,
                text_auto=False,
                aspect="auto",
                zmin=-1,
                zmax=1,
            )
            fig.update_layout(margin=dict(l=10, r=10, t=35, b=10))
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Top correlated pairs**")
        topk = st.slider("How many pairs", 5, 30, 10)
        tc = top_correlations(dfp, k=topk)
        if tc.empty:
            st.info("Not enough numeric columns.")
        else:
            st.dataframe(tc, use_container_width=True, height=260)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Outlier scan (IQR method)**")
        num_cols = dfp.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            st.info("No numeric columns found.")
        else:
            chosen = st.multiselect("Columns to scan", num_cols, default=num_cols[: min(6, len(num_cols))])
            out_df = iqr_outlier_summary(dfp, chosen)
            if out_df.empty:
                st.info("No outlier summary available.")
            else:
                st.dataframe(out_df, use_container_width=True, height=260)
        st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# NOTES (Step 10) — Page 6: Modeling (Optional)
# - This is NOT full ML workflow; it’s a quick sanity check for metrics like F1.
# - Uses a clean preprocessing pipeline (impute + one-hot + (optional) scaling).
# =============================================================================

elif page == "6) Modeling (Optional)":
    st.markdown("### 6) Modeling (Optional)")
    if not require_df():
        st.stop()

    df = st.session_state.df_work

    st.markdown(
        "<div class='card'>"
        "<span class='pill'>Train/Test</span>"
        "<span class='pill'>F1</span>"
        "<span class='pill'>Confusion Matrix</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**Notes**")
    st.markdown(
        "- Pick a target column.\n"
        "- App detects classification vs regression (you can override).\n"
        "- Computes F1/Accuracy/Precision/Recall for classification; basic regression metrics aren’t shown by default here.\n"
        "- Everything runs on the working copy (after fixes)."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    cols = df.columns.tolist()
    if len(cols) < 2:
        st.info("Need at least 2 columns to run modeling.")
        st.stop()

    left, right = st.columns([0.95, 1.05])

    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        target = st.selectbox("Target column", cols)

        # Feature selection default: everything except target
        feature_candidates = [c for c in cols if c != target]
        features = st.multiselect("Feature columns", feature_candidates, default=feature_candidates)

        if not features:
            st.warning("Select at least one feature.")
            st.stop()

        # Auto problem type
        y = df[target]
        unique_y = y.dropna().nunique()
        is_numeric_y = pd.api.types.is_numeric_dtype(y)

        auto_problem = "Regression"
        if (not is_numeric_y) or (unique_y <= 20):
            auto_problem = "Classification"

        problem_type = st.selectbox("Problem type", ["Classification", "Regression"], index=0 if auto_problem == "Classification" else 1)

        test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random state", min_value=0, value=42, step=1)

        scale_numeric = st.checkbox("Scale numeric features (recommended for Logistic/SVM)", value=True)

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        if problem_type == "Classification":
            model_name = st.selectbox(
                "Model",
                [
                    "Logistic Regression",
                    "Random Forest (Classifier)",
                    "Gradient Boosting (Classifier)",
                    "SVM (Classifier)",
                ],
            )
        else:
            model_name = st.selectbox(
                "Model",
                [
                    "Linear Regression",
                    "Random Forest (Regressor)",
                    "Gradient Boosting (Regressor)",
                ],
            )

        run = st.button("Train & Evaluate")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        if not run:
            st.info("Configure the options on the left, then click **Train & Evaluate**.")
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

        work = df[features + [target]].copy()
        work = work.dropna(subset=[target])  # keep target present
        if work.empty:
            st.error("No rows left after dropping missing target values.")
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

        X = work[features]
        y = work[target]

        # Split
        stratify = y if (problem_type == "Classification" and y.nunique() > 1) else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=float(test_size),
            random_state=int(random_state),
            stratify=stratify,
        )

        # Preprocessing
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in X.columns if c not in num_cols]

        numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        numeric_transformer = Pipeline(steps=numeric_steps)

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, num_cols),
                ("cat", categorical_transformer, cat_cols),
            ]
        )

        # Model selection
        if problem_type == "Classification":
            if model_name == "Logistic Regression":
                model = LogisticRegression(max_iter=2000, n_jobs=None)
            elif model_name == "Random Forest (Classifier)":
                model = RandomForestClassifier(n_estimators=300, random_state=int(random_state))
            elif model_name == "Gradient Boosting (Classifier)":
                model = GradientBoostingClassifier(random_state=int(random_state))
            else:
                model = SVC(probability=True, random_state=int(random_state))
        else:
            if model_name == "Linear Regression":
                model = LinearRegression()
            elif model_name == "Random Forest (Regressor)":
                model = RandomForestRegressor(n_estimators=400, random_state=int(random_state))
            else:
                model = GradientBoostingRegressor(random_state=int(random_state))

        clf = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

        try:
            clf.fit(X_train, y_train)
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

        # Evaluate
        y_pred = clf.predict(X_test)

        if problem_type == "Classification":
            # Handle labels that are strings / categories
            avg = st.selectbox("F1 averaging", ["weighted", "macro", "micro"], index=0)

            f1 = f1_score(y_test, y_pred, average=avg, zero_division=0)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average=avg, zero_division=0)
            rec = recall_score(y_test, y_pred, average=avg, zero_division=0)

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.markdown(f"<div class='kpi'><div class='label'>F1 ({avg})</div><div class='value'>{f1:.3f}</div></div>", unsafe_allow_html=True)
            with k2:
                st.markdown(f"<div class='kpi'><div class='label'>Accuracy</div><div class='value'>{acc:.3f}</div></div>", unsafe_allow_html=True)
            with k3:
                st.markdown(f"<div class='kpi'><div class='label'>Precision ({avg})</div><div class='value'>{prec:.3f}</div></div>", unsafe_allow_html=True)
            with k4:
                st.markdown(f"<div class='kpi'><div class='label'>Recall ({avg})</div><div class='value'>{rec:.3f}</div></div>", unsafe_allow_html=True)

            st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

            # Confusion matrix
            labels = sorted(pd.Series(y_test).dropna().unique().tolist())
            cm = confusion_matrix(y_test, y_pred, labels=labels)
            cm_fig = px.imshow(cm, x=labels, y=labels, text_auto=True, aspect="auto")
            cm_fig.update_layout(title="Confusion Matrix", margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(cm_fig, use_container_width=True)

            # ROC-AUC if binary and model can do proba
            try:
                if len(labels) == 2 and hasattr(clf.named_steps["model"], "predict_proba"):
                    proba = clf.predict_proba(X_test)[:, 1]
                    # Need numeric y for roc_auc_score; map labels
                    y_bin = (pd.Series(y_test) == labels[1]).astype(int)
                    auc = roc_auc_score(y_bin, proba)
                    st.markdown(f"**ROC-AUC (binary):** `{auc:.3f}`")
            except Exception:
                pass

        else:
            # Minimal regression feedback (still useful for EDA sanity)
            # We’ll show MAE + RMSE + R2 quickly.
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            mae = mean_absolute_error(y_test, y_pred)
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            r2 = r2_score(y_test, y_pred)

            k1, k2, k3 = st.columns(3)
            with k1:
                st.markdown(f"<div class='kpi'><div class='label'>MAE</div><div class='value'>{mae:.3f}</div></div>", unsafe_allow_html=True)
            with k2:
                st.markdown(f"<div class='kpi'><div class='label'>RMSE</div><div class='value'>{rmse:.3f}</div></div>", unsafe_allow_html=True)
            with k3:
                st.markdown(f"<div class='kpi'><div class='label'>R²</div><div class='value'>{r2:.3f}</div></div>", unsafe_allow_html=True)

            st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
            # Pred vs actual
            plot_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred}).dropna()
            fig = px.scatter(plot_df, x="Actual", y="Predicted", trendline="ols" if len(plot_df) <= 50_000 else None)
            fig.update_layout(title="Predicted vs Actual", margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# NOTES (Step 11) — Page 7: Export report (Markdown)
# - Simple + clean, easy to paste into docs
# =============================================================================

elif page == "7) Export Report":
    st.markdown("### 7) Export Report")
    if not require_df():
        st.stop()

    df = st.session_state.df_work
    dfp = sample_df(df, max_rows=max_plot_rows, seed=seed)
    info = basic_info(df)

    st.markdown(
        "<div class='card'>"
        "<span class='pill'>Markdown</span>"
        "<span class='pill'>Summary</span>"
        "<span class='pill'>Quality</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    title = st.text_input("Report title", value="EDA Report")
    author = st.text_input("Author (optional)", value="")
    include_corr = st.checkbox("Include top correlations", value=True)
    include_missing = st.checkbox("Include missingness table (top 25)", value=True)

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    # Build report content
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    dup = duplicates_count(df)
    miss = missing_table(df)

    report_lines = []
    report_lines.append(f"# {title}")
    report_lines.append("")
    report_lines.append(f"- Generated: **{now}**")
    if author.strip():
        report_lines.append(f"- Author: **{author.strip()}**")
    report_lines.append("")
    report_lines.append("## Dataset overview")
    report_lines.append(f"- Rows: **{_format_int(info.rows)}**")
    report_lines.append(f"- Columns: **{_format_int(info.cols)}**")
    report_lines.append(f"- Memory: **{info.mem_mb:.2f} MB**")
    report_lines.append(f"- Types: **{info.n_numeric} numeric**, **{info.n_categorical} categorical**, **{info.n_datetime} datetime**, **{info.n_bool} boolean**")
    report_lines.append("")
    report_lines.append("## Data quality")
    report_lines.append(f"- Duplicate rows: **{_format_int(dup)}**")
    total_missing = int(df.isna().sum().sum())
    report_lines.append(f"- Total missing cells: **{_format_int(total_missing)}**")
    report_lines.append("")

    if include_missing:
        report_lines.append("### Missing values (top 25 columns)")
        top = miss.head(25).copy()
        top = top.reset_index().rename(columns={"index": "column"})
        report_lines.append(top.to_markdown(index=False))
        report_lines.append("")

    if include_corr:
        report_lines.append("## Top correlations (numeric)")
        tc = top_correlations(dfp, k=10)
        if tc.empty:
            report_lines.append("_Not enough numeric columns to compute correlations._")
        else:
            report_lines.append(tc.to_markdown(index=False))
        report_lines.append("")

    # Quick “notable columns” section
    report_lines.append("## Notable columns")
    dtype_df = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str).values})
    report_lines.append(dtype_df.to_markdown(index=False))
    report_lines.append("")

    report_md = "\n".join(report_lines)

    st.markdown("**Preview**")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(report_md)
    st.markdown("</div>", unsafe_allow_html=True)

    st.download_button(
        "Download report (Markdown)",
        data=report_md.encode("utf-8"),
        file_name="eda_report.md",
        mime="text/markdown",
    )
    st.markdown("</div>", unsafe_allow_html=True)
