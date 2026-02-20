# marketing_dashboard_app_styled_yellow.py
# Marketing / Campaign Pipeline Dashboard (Yellow + White premium theme)
# -----------------------------------------------------------------------------
# Keeps your existing logic (cache + session_state + KPIs + tables),
# but upgrades the UI:
# - Yellow/white premium theme with clean typography
# - Custom KPI cards (instead of default st.metric) for a "dashboard" look
# - Better spacing, section headers, and card containers
# - Tables shown inside cards (still in one row)

from __future__ import annotations

import streamlit as st
import pandas as pd
from datetime import date
import io
import hashlib

# --- Streamlit config ---
st.set_page_config(page_title="Marketing Dashboard", layout="wide")

# -----------------------
# Yellow + White Theme (CSS)
# -----------------------
st.markdown(
    """
<style>
:root{
  --bg1:#FFFDF5;
  --bg2:#FFF2CC;
  --card:#FFFFFF;
  --txt:#111827;
  --muted:#6B7280;
  --line:rgba(17,24,39,.10);
  --accent:#FBBF24;      /* amber-400 */
  --accent2:#F59E0B;     /* amber-500 */
  --accentSoft:rgba(251,191,36,.22);
  --shadow:0 14px 30px rgba(17,24,39,.08);
  --radius:18px;
}

.stApp{
  background:
    radial-gradient(1200px circle at 8% 0%, rgba(251,191,36,.35), transparent 55%),
    radial-gradient(900px circle at 92% 12%, rgba(245,158,11,.20), transparent 55%),
    linear-gradient(180deg, var(--bg2) 0%, var(--bg1) 45%, #FFFFFF 100%);
  color: var(--txt);
}
html, body, [class*="css"]{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }

.block-container { padding-top: 1.1rem; padding-bottom: 1.3rem; }

section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(251,191,36,.18), rgba(255,255,255,.85));
  border-right: 1px solid var(--line);
}

.hero{
  display:flex; justify-content:space-between; align-items:flex-start;
  background: linear-gradient(90deg, rgba(251,191,36,.32), rgba(255,255,255,.85));
  border: 1px solid rgba(251,191,36,.45);
  border-radius: var(--radius);
  padding: 16px 18px;
  box-shadow: var(--shadow);
  margin-bottom: 12px;
}
.hero h1{ margin:0; font-size: 22px; font-weight: 800; letter-spacing: .2px; }
.hero p{ margin:6px 0 0 0; color: var(--muted); font-size: 13px; }
.badge{
  display:inline-flex; align-items:center; gap:8px;
  padding: 8px 12px;
  border-radius: 999px;
  border: 1px solid rgba(251,191,36,.55);
  background: rgba(255,255,255,.80);
  color: var(--txt);
  font-size: 12px;
  font-weight: 700;
}

.card{
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 14px 14px;
}
.card-title{
  font-size: 12px;
  letter-spacing: .12em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 8px;
}
.kpi-grid{
  display:grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
}
.kpi{
  border: 1px solid rgba(251,191,36,.35);
  border-radius: 16px;
  background: linear-gradient(180deg, rgba(251,191,36,.12), rgba(255,255,255,.95));
  padding: 12px 12px;
}
.kpi .label{
  font-size: 12px;
  color: var(--muted);
  letter-spacing: .04em;
  margin-bottom: 6px;
}
.kpi .value{
  font-size: 26px;
  font-weight: 900;
  color: var(--txt);
  line-height: 1;
}
.kpi .hint{
  font-size: 12px;
  color: var(--muted);
  margin-top: 6px;
}

hr{
  border: none;
  height: 1px;
  background: rgba(17,24,39,.10);
  margin: 12px 0;
}

.stButton button, .stDownloadButton button{
  border-radius: 14px !important;
  border: 1px solid rgba(245,158,11,.55) !important;
  background: linear-gradient(180deg, rgba(251,191,36,.90), rgba(245,158,11,.90)) !important;
  color: #111827 !important;
  font-weight: 800 !important;
}
.stButton button:hover, .stDownloadButton button:hover{
  filter: brightness(1.03);
}

div[data-testid="stDataFrame"]{
  border: 1px solid var(--line);
  border-radius: 16px;
  overflow: hidden;
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------
# Helpers (unchanged)
# -----------------------
def safe_series(df: pd.DataFrame, col: str, dtype=None):
    if df is None:
        return pd.Series(dtype=dtype)
    if col in df.columns:
        return df[col]
    return pd.Series([pd.NA] * len(df), index=df.index, dtype=dtype)

def to_datetime_inplace(df: pd.DataFrame, cols, dayfirst: bool = False):
    if df is None:
        return
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=dayfirst)

def to_numeric_inplace(df: pd.DataFrame, cols):
    if df is None:
        return
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def parse_range(r):
    if isinstance(r, (list, tuple)) and len(r) == 2:
        return r[0], r[1]
    return None, None

# ---- Fixed categories & normalizer ----
FIXED_CATEGORIES = [
    "Video", "Presentation", "Document", "Copy",
    "Branding", "Client Servicing", "Campaign",
    "Initiatives", "Not filled Properly",
]
_CANON = {c.lower(): c for c in FIXED_CATEGORIES}

def normalize_category(x: object) -> str:
    s = "" if pd.isna(x) else str(x).strip()
    if not s:
        return "Not filled Properly"
    key = s.lower()
    return _CANON.get(key, "Not filled Properly")

def kpi_card(label: str, value: int | float, hint: str = ""):
    st.markdown(
        f"""
<div class="kpi">
  <div class="label">{label}</div>
  <div class="value">{int(value):,}</div>
  <div class="hint">{hint}</div>
</div>
""",
        unsafe_allow_html=True,
    )

# -----------------------
# Cached loaders / compute (unchanged)
# -----------------------
@st.cache_data(show_spinner=False)
def load_excel_cached(file_bytes: bytes):
    xl = pd.ExcelFile(io.BytesIO(file_bytes))

    def read_sheet_safe(name: str, usecols=None):
        if name not in xl.sheet_names:
            return None
        try:
            return xl.parse(name, usecols=usecols)
        except Exception:
            return xl.parse(name)

    def read_sheet_by_aliases(names, usecols=None):
        for n in names:
            if n in xl.sheet_names:
                return read_sheet_safe(n, usecols=usecols)
        return None

    df_task = read_sheet_safe(
        "Task Tracker 2026",
        usecols=[
            "Completion Date", "Requested Date", "No of Collaterals", "Tasks",
            "Category", "Status", "Client", "Sector"
        ],
    )

    df_sales_leads = read_sheet_safe("Sales Leads Generated", usecols=["Lead Date", "Status"])
    df_brand_leads = read_sheet_safe("Brand Leads Generated", usecols=["Lead Date", "Status"])
    df_partner_leads = read_sheet_safe("Partner Leads Generated", usecols=["Lead Date"])

    df_campaigns_inits = read_sheet_by_aliases(
        ["Campaigns and Initiatives", "Campaigns & Initiatives", "Campaigns & initiatives"],
        usecols=["Start Date", "Category"]
    )

    to_datetime_inplace(df_task, ["Completion Date", "Requested Date"])
    to_datetime_inplace(df_sales_leads, ["Lead Date"], dayfirst=True)
    to_datetime_inplace(df_brand_leads, ["Lead Date"], dayfirst=True)
    to_datetime_inplace(df_partner_leads, ["Lead Date"], dayfirst=True)
    to_datetime_inplace(df_campaigns_inits, ["Start Date"], dayfirst=True)

    to_numeric_inplace(df_task, ["No of Collaterals", "Tasks"])

    return df_task, df_sales_leads, df_brand_leads, df_partner_leads, df_campaigns_inits

def compute_dashboard_payload(df_task, df_sales_leads, df_brand_leads, df_partner_leads, df_campaigns_inits, start_date, end_date):
    ts_start = pd.Timestamp(start_date)
    ts_end = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

    done_statuses = {"completed", "design completed", "copy completed"}

    # -------- Filter Task Tracker --------
    if df_task is not None and not df_task.empty:
        comp = safe_series(df_task, "Completion Date")
        req = safe_series(df_task, "Requested Date")

        filtered_task_completed = df_task.loc[comp.between(ts_start, ts_end, inclusive="both")]
        filtered_task_requested = df_task.loc[req.between(ts_start, ts_end, inclusive="both")]
    else:
        filtered_task_completed = pd.DataFrame()
        filtered_task_requested = pd.DataFrame()

    # -------- KPIs from Task Tracker --------
    noc = pd.to_numeric(safe_series(filtered_task_completed, "No of Collaterals", dtype="float"), errors="coerce").fillna(0)

    collateral_shipped = int(noc.sum())
    cat_norm_completed = safe_series(filtered_task_completed, "Category").map(normalize_category)
    content_pieces_shipped = int(noc[cat_norm_completed.eq("Copy")].sum())

    requests_received = int(len(filtered_task_requested))
    status_req = (
        safe_series(filtered_task_requested, "Status")
        .astype("string").fillna("").str.strip().str.lower()
    )
    pipeline_pending = int((~status_req.isin(done_statuses)).sum())

    # -------- Sales Leads KPIs --------
    new_leads_generated = 0
    conversions = 0
    marketing_leads_in_process = 0

    if df_sales_leads is not None and not df_sales_leads.empty:
        lead_dt = safe_series(df_sales_leads, "Lead Date")
        status_sales = (
            safe_series(df_sales_leads, "Status")
            .astype("string").fillna("").str.strip().str.lower()
        )

        in_selected_range = lead_dt.between(ts_start, ts_end, inclusive="both")
        new_leads_generated = int(in_selected_range.sum())

        conversions = int((in_selected_range & status_sales.eq("closed")).sum())

        upto_selected_end = lead_dt.le(ts_end)
        marketing_leads_in_process = int((upto_selected_end & ~status_sales.eq("closed")).sum())

    # -------- Brand Leads KPIs --------
    brand_leads_generated = 0
    brand_conversion_cycle = 0

    if df_brand_leads is not None and not df_brand_leads.empty:
        brand_lead_dt = safe_series(df_brand_leads, "Lead Date")
        brand_status = (
            safe_series(df_brand_leads, "Status")
            .astype("string").fillna("").str.strip().str.lower()
        )

        brand_in_selected_range = brand_lead_dt.between(ts_start, ts_end, inclusive="both")
        brand_leads_generated = int(brand_in_selected_range.sum())
        brand_conversion_cycle = int((brand_in_selected_range & brand_status.eq("closed")).sum())

    # -------- Partner Leads --------
    partner_leads_generated = 0
    if df_partner_leads is not None and not df_partner_leads.empty:
        p_dt = safe_series(df_partner_leads, "Lead Date")
        partner_leads_generated = int(p_dt.between(ts_start, ts_end, inclusive="both").sum())

    # -------- Campaigns & Initiatives KPIs --------
    campaigns_created = 0
    initiatives_taken = 0

    if df_campaigns_inits is not None and not df_campaigns_inits.empty:
        start_dt = safe_series(df_campaigns_inits, "Start Date")
        cat = (
            safe_series(df_campaigns_inits, "Category")
            .astype("string").fillna("").str.strip().str.lower()
        )

        in_selected_range = start_dt.between(ts_start, ts_end, inclusive="both")
        campaigns_created = int((in_selected_range & cat.str.contains("campaign", na=False)).sum())
        initiatives_taken = int((in_selected_range & cat.str.contains("initiative", na=False)).sum())

    # -------- Tables --------
    collateral_delivery_df = pd.DataFrame({"Category": FIXED_CATEGORIES, "Total Collateral": 0})
    pipeline_by_category_df = pd.DataFrame({"Category": FIXED_CATEGORIES, "Open Pipeline Tasks": 0})
    top_clients_df = pd.DataFrame({"Client": pd.Series(dtype="string"), "Total Collateral": pd.Series(dtype="int64")})
    top_sectors_df = pd.DataFrame({"Sector": pd.Series(dtype="string"), "Total Collateral": pd.Series(dtype="int64")})

    if not filtered_task_completed.empty and "No of Collaterals" in filtered_task_completed.columns:
        tmp = filtered_task_completed[["Category", "No of Collaterals"]].copy()
        tmp["No of Collaterals"] = pd.to_numeric(tmp["No of Collaterals"], errors="coerce").fillna(0)
        tmp["Category"] = safe_series(tmp, "Category").map(normalize_category)
        collateral_delivery_df = (
            tmp.groupby("Category", as_index=True)["No of Collaterals"]
               .sum()
               .reindex(FIXED_CATEGORIES, fill_value=0)
               .rename("Total Collateral")
               .reset_index()
        )
        collateral_delivery_df["Total Collateral"] = collateral_delivery_df["Total Collateral"].round(0).astype("int64")

    if not filtered_task_requested.empty:
        cols = [c for c in ["Category", "Status"] if c in filtered_task_requested.columns]
        tmp = filtered_task_requested[cols].copy()
        status = (
            safe_series(tmp, "Status")
            .astype("string").fillna("").str.strip().str.lower()
        )
        open_mask = ~status.isin(done_statuses)
        tmp = tmp.loc[open_mask]
        tmp["Category"] = safe_series(tmp, "Category").map(normalize_category)
        pipeline_by_category_df = (
            tmp.groupby("Category", as_index=True)
               .size()
               .reindex(FIXED_CATEGORIES, fill_value=0)
               .rename("Open Pipeline Tasks")
               .reset_index()
        )
        pipeline_by_category_df["Open Pipeline Tasks"] = pipeline_by_category_df["Open Pipeline Tasks"].astype("int64")

    if not filtered_task_completed.empty and "No of Collaterals" in filtered_task_completed.columns:
        cols = [c for c in ["No of Collaterals", "Client", "Sector"] if c in filtered_task_completed.columns]
        base = filtered_task_completed[cols].copy()
        base["No of Collaterals"] = pd.to_numeric(base["No of Collaterals"], errors="coerce").fillna(0)

        if "Client" in base.columns:
            base["Client"] = base["Client"].astype("string").fillna("").str.strip().replace("", "Unknown")
            top_clients_df = (
                base.groupby("Client", as_index=False)["No of Collaterals"]
                    .sum()
                    .rename(columns={"No of Collaterals": "Total Collateral"})
                    .sort_values("Total Collateral", ascending=False)
                    .head(7)
            )
            top_clients_df["Total Collateral"] = top_clients_df["Total Collateral"].round(0).astype("int64")

        if "Sector" in base.columns:
            base["Sector"] = base["Sector"].astype("string").fillna("").str.strip().replace("", "Unknown")
            top_sectors_df = (
                base.groupby("Sector", as_index=False)["No of Collaterals"]
                    .sum()
                    .rename(columns={"No of Collaterals": "Total Collateral"})
                    .sort_values("Total Collateral", ascending=False)
                    .head(7)
            )
            top_sectors_df["Total Collateral"] = top_sectors_df["Total Collateral"].round(0).astype("int64")

    return {
        "collateral_shipped": collateral_shipped,
        "content_pieces_shipped": content_pieces_shipped,
        "partner_leads_generated": partner_leads_generated,
        "requests_received": requests_received,
        "new_leads_generated": new_leads_generated,
        "brand_leads_generated": brand_leads_generated,
        "pipeline_pending": pipeline_pending,
        "marketing_leads_in_process": marketing_leads_in_process,
        "brand_conversion_cycle": brand_conversion_cycle,
        "campaigns_created": campaigns_created,
        "conversions": conversions,
        "initiatives_taken": initiatives_taken,
        "collateral_delivery_df": collateral_delivery_df,
        "pipeline_by_category_df": pipeline_by_category_df,
        "top_clients_df": top_clients_df,
        "top_sectors_df": top_sectors_df,
    }

# -----------------------
# Sidebar controls
# -----------------------
with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Upload dashboard Excel file", type=["xlsx"], accept_multiple_files=False)

# -----------------------
# Session state init
# -----------------------
if "file_sig" not in st.session_state:
    st.session_state.file_sig = None
if "loaded_data" not in st.session_state:
    st.session_state.loaded_data = None
if "range_key" not in st.session_state:
    st.session_state.range_key = None
if "payload" not in st.session_state:
    st.session_state.payload = None

# -----------------------
# Default placeholder values
# -----------------------
collateral_shipped = 0
content_pieces_shipped = 0
partner_leads_generated = 0
requests_received = 0
new_leads_generated = 0
brand_leads_generated = 0
pipeline_pending = 0
marketing_leads_in_process = 0
brand_conversion_cycle = 0
campaigns_created = 0
conversions = 0
initiatives_taken = 0

collateral_delivery_df = pd.DataFrame({"Category": FIXED_CATEGORIES, "Total Collateral": 0})
pipeline_by_category_df = pd.DataFrame({"Category": FIXED_CATEGORIES, "Open Pipeline Tasks": 0})
top_clients_df = pd.DataFrame({"Client": pd.Series(dtype="string"), "Total Collateral": pd.Series(dtype="int64")})
top_sectors_df = pd.DataFrame({"Sector": pd.Series(dtype="string"), "Total Collateral": pd.Series(dtype="int64")})

date_range = None
start_date = None
end_date = None

# -----------------------
# Load Excel if uploaded + compute payload (cached + session_state)
# -----------------------
if uploaded_file:
    try:
        file_bytes = uploaded_file.getvalue()
        file_sig = hashlib.md5(file_bytes).hexdigest()

        if st.session_state.file_sig != file_sig:
            st.session_state.file_sig = file_sig
            st.session_state.loaded_data = load_excel_cached(file_bytes)
            st.session_state.range_key = None
            st.session_state.payload = None

        with st.sidebar:
            st.markdown("---")
            date_range = st.date_input("Select Date Range", value=[date.today(), date.today()], key="date_range")

        start_date, end_date = parse_range(date_range)

        if start_date and end_date:
            current_range_key = f"{start_date}_{end_date}"
            if st.session_state.range_key != current_range_key or st.session_state.payload is None:
                df_task, df_sales_leads, df_brand_leads, df_partner_leads, df_campaigns_inits = st.session_state.loaded_data
                st.session_state.payload = compute_dashboard_payload(
                    df_task, df_sales_leads, df_brand_leads, df_partner_leads, df_campaigns_inits,
                    start_date, end_date
                )
                st.session_state.range_key = current_range_key

            p = st.session_state.payload
            collateral_shipped = p["collateral_shipped"]
            content_pieces_shipped = p["content_pieces_shipped"]
            partner_leads_generated = p["partner_leads_generated"]
            requests_received = p["requests_received"]
            new_leads_generated = p["new_leads_generated"]
            brand_leads_generated = p["brand_leads_generated"]
            pipeline_pending = p["pipeline_pending"]
            marketing_leads_in_process = p["marketing_leads_in_process"]
            brand_conversion_cycle = p["brand_conversion_cycle"]
            campaigns_created = p["campaigns_created"]
            conversions = p["conversions"]
            initiatives_taken = p["initiatives_taken"]
            collateral_delivery_df = p["collateral_delivery_df"]
            pipeline_by_category_df = p["pipeline_by_category_df"]
            top_clients_df = p["top_clients_df"]
            top_sectors_df = p["top_sectors_df"]

    except Exception as e:
        st.warning(f"Failed to parse uploaded file. Error: {e}")

# -----------------------
# Hero header
# -----------------------
range_txt = "Upload file + pick date range"
if start_date and end_date:
    range_txt = f"{start_date} → {end_date}"

st.markdown(
    f"""
<div class="hero">
  <div>
    <h1>📣 Campaign Pipeline Dashboard</h1>
    <p>One view for shipments, leads, pipeline health, and initiatives — clean & fast.</p>
  </div>
  <div class="badge">🗓️ {range_txt}</div>
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------
# KPI metrics layout (premium cards)
# -----------------------
st.markdown('<div class="card"><div class="card-title">Key Metrics</div>', unsafe_allow_html=True)

r1, r2, r3, r4 = st.columns(4)

with r1:
    st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
    kpi_card("Collaterals shipped", collateral_shipped)
    kpi_card("Content pieces shipped", content_pieces_shipped)
    kpi_card("Partner leads generated", partner_leads_generated)
    st.markdown('</div>', unsafe_allow_html=True)

with r2:
    st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
    kpi_card("Requests received", requests_received)
    kpi_card("New leads generated", new_leads_generated)
    kpi_card("Brand leads generated", brand_leads_generated)
    st.markdown('</div>', unsafe_allow_html=True)

with r3:
    st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
    kpi_card("Pipeline pending", pipeline_pending)
    kpi_card("Marketing leads in process", marketing_leads_in_process)
    kpi_card("Brand leads in conversion", brand_conversion_cycle)
    st.markdown('</div>', unsafe_allow_html=True)

with r4:
    st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
    kpi_card("Campaigns created", campaigns_created)
    kpi_card("Conversions", conversions)
    kpi_card("Initiatives taken", initiatives_taken)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# -----------------------
# Tables (ALL IN ONE ROW) in cards
# -----------------------
st.markdown('<div class="card"><div class="card-title">Breakdowns</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("#### Collateral Delivery by Category")
    st.dataframe(collateral_delivery_df.reset_index(drop=True), use_container_width=True, hide_index=True)

with c2:
    st.markdown("#### Pipeline Tasks by Category")
    st.dataframe(pipeline_by_category_df.reset_index(drop=True), use_container_width=True, hide_index=True)

with c3:
    st.markdown("#### Top 7 Clients (by collaterals)")
    st.dataframe(top_clients_df.reset_index(drop=True), use_container_width=True, hide_index=True)

with c4:
    st.markdown("#### Top 7 Sectors (by collaterals)")
    st.dataframe(top_sectors_df.reset_index(drop=True), use_container_width=True, hide_index=True)

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------
# Friendly empty state
# -----------------------
if not uploaded_file:
    st.info("👈 Upload your Excel file from the sidebar to populate this dashboard.")
elif not (start_date and end_date):
    st.warning("Select a valid **from date** and **end date** in the sidebar to load the dashboard.")
