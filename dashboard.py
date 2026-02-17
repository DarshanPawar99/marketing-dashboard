import streamlit as st
import pandas as pd
from datetime import date

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

def to_datetime_inplace(df: pd.DataFrame, cols):
    if df is None:
        return
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

def to_numeric_inplace(df: pd.DataFrame, cols):
    if df is None:
        return
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def load_excel(file):
    return pd.ExcelFile(file)

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

def parse_range(r):
    if isinstance(r, (list, tuple)) and len(r) == 2:
        return r[0], r[1]
    return None, None

# -----------------------
# Sidebar controls
# -----------------------
with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader(
        "Upload dashboard Excel file", type=["xlsx"], accept_multiple_files=False
    )

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
top_clients_df = pd.DataFrame({"Client": pd.Series(dtype='string'), "Total Collateral": pd.Series(dtype='float')})
top_sectors_df = pd.DataFrame({"Sector": pd.Series(dtype='string'), "Total Collateral": pd.Series(dtype='float')})

# -----------------------
# Load Excel if uploaded
# -----------------------
if uploaded_file:
    try:
        xl = load_excel(uploaded_file)

        def read_sheet(name: str):
            return xl.parse(name) if name in xl.sheet_names else None

        df_task = read_sheet("Task Tracker 2026")
        df_sales_leads = read_sheet("Sales Leads Generated")
        df_brand_leads = read_sheet("Brand Leads Generated")
        df_partner_leads = read_sheet("Partner Leads Generated")

        # Ensure datetime and numeric types
        to_datetime_inplace(df_task, ["Completion Date", "Requested Date"])
        to_datetime_inplace(df_sales_leads, ["Lead Date"])
        to_datetime_inplace(df_brand_leads, ["Lead Date"])
        to_datetime_inplace(df_partner_leads, ["Lead Date"])

        to_numeric_inplace(df_task, ["No of Collaterals", "Tasks"])

        # Ask for date range AFTER successful upload
        with st.sidebar:
            date_range = st.date_input(
                "Select Completion Date Range",
                value=[date.today(), date.today()],
                key="date_range",
            )

        start_date, end_date = parse_range(date_range)
        if not start_date or not end_date:
            st.warning("Select **from date** and **end date** in the sidebar to load the dashboard.")
        else:
            st.header(f"Campaign Pipeline Dashboard")
            st.markdown("---")

            ts_start = pd.Timestamp(start_date)
            ts_end = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

            # --- Filter data ---
            if df_task is not None:
                comp = safe_series(df_task, "Completion Date")
                mask_completion = comp.between(ts_start, ts_end, inclusive="both")
                filtered_task_completed = df_task.loc[mask_completion]
            else:
                filtered_task_completed = pd.DataFrame()

            # KPIs
            noc = safe_series(filtered_task_completed, "No of Collaterals", dtype="float")
            noc = pd.to_numeric(noc, errors="coerce")
            collateral_shipped = float(noc.sum()) if not noc.empty else 0
            cat_norm_completed = safe_series(filtered_task_completed, "Category").map(normalize_category)
            content_pieces_shipped = int(noc[cat_norm_completed == "Copy"].sum()) if not noc.empty else 0

            if df_task is not None:
                req = safe_series(df_task, "Requested Date")
                mask_requested = req.between(ts_start, ts_end, inclusive="both")
                filtered_task_requested = df_task.loc[mask_requested]
            else:
                filtered_task_requested = pd.DataFrame()

            requests_received = int(len(filtered_task_requested))
            status_req = safe_series(filtered_task_requested, "Status").fillna("")
            pipeline_pending = int((~status_req.isin(["Completed", "Design Completed", "Copy Completed"])).sum())

            def count_leads(df):
                if df is None or df.empty:
                    return 0
                ld = safe_series(df, "Lead Date")
                return int(ld.between(ts_start, ts_end, inclusive="both").sum())

            new_leads_generated = count_leads(df_sales_leads)
            brand_leads_generated = count_leads(df_brand_leads)
            partner_leads_generated = count_leads(df_partner_leads)

            rcat_norm = safe_series(filtered_task_requested, "Category").map(normalize_category).fillna("")
            campaigns_created = int((rcat_norm == "Campaign").sum())
            initiatives_taken = int((rcat_norm == "Initiatives").sum())

            # Tables
            if 'No of Collaterals' in filtered_task_completed.columns:
                tmp = filtered_task_completed.copy()
                tmp['No of Collaterals'] = pd.to_numeric(tmp['No of Collaterals'], errors='coerce').fillna(0)
                tmp['Category'] = safe_series(tmp, 'Category').map(normalize_category)
                collateral_delivery_df = (
                    tmp.groupby('Category', as_index=True)['No of Collaterals']
                       .sum()
                       .reindex(FIXED_CATEGORIES, fill_value=0)
                       .rename('Total Collateral')
                       .reset_index()
                )

            if not filtered_task_requested.empty:
                tmp = filtered_task_requested.copy()
                status = safe_series(tmp, "Status").fillna("")
                open_mask = ~status.isin(["Completed", "Design Completed", "Copy Completed"])
                tmp = tmp.loc[open_mask]
                tmp['Category'] = safe_series(tmp, 'Category').map(normalize_category)
                pipeline_by_category_df = (
                    tmp.groupby('Category', as_index=True)
                       .size()
                       .reindex(FIXED_CATEGORIES, fill_value=0)
                       .rename('Open Pipeline Tasks')
                       .reset_index()
                )

            if not filtered_task_completed.empty and 'No of Collaterals' in filtered_task_completed.columns:
                if 'Client' in filtered_task_completed.columns:
                    tmp = filtered_task_completed.copy()
                    tmp['No of Collaterals'] = pd.to_numeric(tmp['No of Collaterals'], errors='coerce').fillna(0)
                    tmp['Client'] = tmp['Client'].astype('string').fillna('').str.strip().replace('', 'Unknown')
                    top_clients_df = (
                        tmp.groupby('Client', as_index=False)['No of Collaterals']
                           .sum()
                           .rename(columns={'No of Collaterals': 'Total Collateral'})
                           .sort_values('Total Collateral', ascending=False)
                           .head(7)
                    )

                if 'Sector' in filtered_task_completed.columns:
                    tmp = filtered_task_completed.copy()
                    tmp['No of Collaterals'] = pd.to_numeric(tmp['No of Collaterals'], errors='coerce').fillna(0)
                    tmp['Sector'] = tmp['Sector'].astype('string').fillna('').str.strip().replace('', 'Unknown')
                    top_sectors_df = (
                        tmp.groupby('Sector', as_index=False)['No of Collaterals']
                           .sum()
                           .rename(columns={'No of Collaterals': 'Total Collateral'})
                           .sort_values('Total Collateral', ascending=False)
                           .head(7)
                    )

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
    st.table(top_sectors_df)







