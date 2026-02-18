import streamlit as st
import pandas as pd
from datetime import date
import io
import hashlib

# --- Streamlit config ---
st.set_page_config(page_title="Dashboard", layout="wide")

# -----------------------
# Helpers
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

# -----------------------
# Cached loaders / compute
# -----------------------
@st.cache_data(show_spinner=False)
def load_excel_cached(file_bytes: bytes):
    """
    Parse Excel once per file content.
    Returns cleaned dataframes for all required sheets.
    """
    xl = pd.ExcelFile(io.BytesIO(file_bytes))

    def read_sheet_safe(name: str, usecols=None):
        if name not in xl.sheet_names:
            return None
        try:
            return xl.parse(name, usecols=usecols)
        except Exception:
            # fallback if usecols mismatch
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

    # Parse dates once (DD/MM/YY safe for lead/start dates)
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
        lead_dt = safe_series(df_sales_leads, "Lead Date")  # already parsed
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
        brand_lead_dt = safe_series(df_brand_leads, "Lead Date")  # already parsed
        brand_status = (
            safe_series(df_brand_leads, "Status")
            .astype("string").fillna("").str.strip().str.lower()
        )

        brand_in_selected_range = brand_lead_dt.between(ts_start, ts_end, inclusive="both")
        brand_leads_generated = int(brand_in_selected_range.sum())
        brand_conversion_cycle = int((brand_in_selected_range & brand_status.eq("closed")).sum())

    # -------- Partner Leads (count entries in range) --------
    partner_leads_generated = 0
    if df_partner_leads is not None and not df_partner_leads.empty:
        p_dt = safe_series(df_partner_leads, "Lead Date")  # already parsed
        partner_leads_generated = int(p_dt.between(ts_start, ts_end, inclusive="both").sum())

    # -------- Campaigns & Initiatives KPIs --------
    campaigns_created = 0
    initiatives_taken = 0

    if df_campaigns_inits is not None and not df_campaigns_inits.empty:
        start_dt = safe_series(df_campaigns_inits, "Start Date")  # already parsed
        cat = (
            safe_series(df_campaigns_inits, "Category")
            .astype("string").fillna("").str.strip().str.lower()
        )

        in_selected_range = start_dt.between(ts_start, ts_end, inclusive="both")
        campaigns_created = int((in_selected_range & cat.str.contains("campaign", na=False)).sum())
        initiatives_taken = int((in_selected_range & cat.str.contains("initiative", na=False)).sum())  # covers initiative/initiatives

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
    uploaded_file = st.file_uploader(
        "Upload dashboard Excel file", type=["xlsx"], accept_multiple_files=False
    )

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

# -----------------------
# Load Excel if uploaded + compute payload (cached + session_state)
# -----------------------
if uploaded_file:
    try:
        file_bytes = uploaded_file.getvalue()
        file_sig = hashlib.md5(file_bytes).hexdigest()

        # Reload only when uploaded file content changes
        if st.session_state.file_sig != file_sig:
            st.session_state.file_sig = file_sig
            st.session_state.loaded_data = load_excel_cached(file_bytes)
            st.session_state.range_key = None
            st.session_state.payload = None

        # Ask for date range after file is ready
        with st.sidebar:
            date_range = st.date_input(
                "Select Date Range",
                value=[date.today(), date.today()],
                key="date_range",
            )

        start_date, end_date = parse_range(date_range)
        if not start_date or not end_date:
            st.warning("Select **from date** and **end date** in the sidebar to load the dashboard.")
        else:
            st.header("Campaign Pipeline Dashboard")
            st.markdown("---")

            current_range_key = f"{start_date}_{end_date}"

            # Recompute only when date range changes
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
        st.warning(f"Failed to parse uploaded file, showing placeholder data. Error: {e}")

# -----------------------
# KPI metrics layout
# -----------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Number of Collaterals shipped", int(collateral_shipped))
    st.metric("Number of Content pieces shipped", int(content_pieces_shipped))
    st.metric("Partner Leads Generated", int(partner_leads_generated))
with col2:
    st.metric("Number of Requests received", int(requests_received))
    st.metric("New leads generated", int(new_leads_generated))
    st.metric("Brand leads Generated", int(brand_leads_generated))
with col3:
    st.metric("Total Pipeline Pending", int(pipeline_pending))
    st.metric("Marketing Leads in Process", int(marketing_leads_in_process))
    st.metric("Brand leads in conversion cycle", int(brand_conversion_cycle))
with col4:
    st.metric("New Campaigns Created", int(campaigns_created))
    st.metric("Conversions", int(conversions))
    st.metric("New Initiatives taken", int(initiatives_taken))

# -----------------------
# Tables (ALL IN ONE ROW) using st.table
# -----------------------
st.markdown("---")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.subheader("Collateral Delivery by Category type")
    st.table(collateral_delivery_df.reset_index(drop=True))

with c2:
    st.subheader("Pipeline of Tasks by Category")
    st.table(pipeline_by_category_df.reset_index(drop=True))

with c3:
    st.subheader("Top 7 clients based on quantity of collaterals")
    st.table(top_clients_df.reset_index(drop=True))

with c4:
    st.subheader("Top 7 Sectors on Quantity of Collateral")
    st.table(top_sectors_df.reset_index(drop=True))
