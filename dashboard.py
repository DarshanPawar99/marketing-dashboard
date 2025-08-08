import streamlit as st
import pandas as pd
from datetime import date
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

# --- Streamlit config ---
st.set_page_config(page_title="Marketing Performance & Pipeline Dashboard", layout="wide")

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

def apply_cross_filters(df_completed: pd.DataFrame, df_requested: pd.DataFrame, filters: dict):
    """Apply table-driven filters (Category/Client/Sector) to completed/requested frames."""
    def _apply(df):
        if df is None or df.empty:
            return df
        out = df.copy()

        # Category
        if 'Category' in out.columns:
            out['_norm_cat'] = safe_series(out, 'Category').map(normalize_category)
            if filters.get('Category'):
                out = out.loc[out['_norm_cat'] == filters['Category']]

        # Client (strict)
        if filters.get('Client') and 'Client' in out.columns:
            out = out.loc[out['Client'].astype('string').str.strip().replace('', 'Unknown') == filters['Client']]

        # Sector (strict)
        if filters.get('Sector') and 'Sector' in out.columns:
            out = out.loc[out['Sector'].astype('string').str.strip().replace('', 'Unknown') == filters['Sector']]

        return out

    return _apply(df_completed), _apply(df_requested)

def build_aggrid(df: pd.DataFrame, key: str, selectable=True, height=280):
    """Render a grid and return selected row (dict) or None."""
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(resizable=True, sortable=True, filter=True)
    if selectable:
        gb.configure_selection(selection_mode='single', use_checkbox=True)
    else:
        gb.configure_selection(selection_mode='none')
    go = gb.build()
    grid = AgGrid(
        df,
        gridOptions=go,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        fit_columns_on_grid_load=True,
        height=height,
        allow_unsafe_jscode=True,
        key=key,
    )
    sel = grid.get("selected_rows", [])
    return sel[0] if sel else None

# -----------------------
# Sidebar controls
# -----------------------
with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader(
        "Upload dashboard Excel file", type=["xlsx"], accept_multiple_files=False
    )

# -----------------------
# Session filter state (table-driven)
# -----------------------
if "xf" not in st.session_state:
    st.session_state.xf = {"Category": None, "Client": None, "Sector": None}

def set_filter(which, value):
    st.session_state.xf[which] = value if value else None

# -----------------------
# Defaults / placeholders
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

# Placeholder table data
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

        df_task = read_sheet("Task Tracker 2025")
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
            st.header("Campaign Pipeline Dashboard")
            st.markdown("---")

            ts_start = pd.Timestamp(start_date)
            ts_end = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

            # --- Base frames by date ---
            if df_task is not None:
                comp = safe_series(df_task, "Completion Date")
                mask_completion = comp.between(ts_start, ts_end, inclusive="both")
                filtered_task_completed_base = df_task.loc[mask_completion].copy()

                req = safe_series(df_task, "Requested Date")
                mask_requested = req.between(ts_start, ts_end, inclusive="both")
                filtered_task_requested_base = df_task.loc[mask_requested].copy()
            else:
                filtered_task_completed_base = pd.DataFrame()
                filtered_task_requested_base = pd.DataFrame()

            # Normalize key columns for strict filters
            if "Client" in filtered_task_completed_base.columns:
                filtered_task_completed_base["Client"] = (
                    filtered_task_completed_base["Client"].astype("string").fillna("").str.strip().replace("", "Unknown")
                )
            if "Sector" in filtered_task_completed_base.columns:
                filtered_task_completed_base["Sector"] = (
                    filtered_task_completed_base["Sector"].astype("string").fillna("").str.strip().replace("", "Unknown")
                )
            if "Client" in filtered_task_requested_base.columns:
                filtered_task_requested_base["Client"] = (
                    filtered_task_requested_base["Client"].astype("string").fillna("").str.strip().replace("", "Unknown")
                )
            if "Sector" in filtered_task_requested_base.columns:
                filtered_task_requested_base["Sector"] = (
                    filtered_task_requested_base["Sector"].astype("string").fillna("").str.strip().replace("", "Unknown")
                )

            # ---------- Build grouped tables from CURRENT filters ----------
            def build_tables(df_completed, df_requested, xf):
                # Apply cross-filters to the frames
                cdf, rdf = apply_cross_filters(df_completed, df_requested, xf)

                # Collateral Delivery by Category (completed)
                if not cdf.empty and 'No of Collaterals' in cdf.columns:
                    tmp = cdf.copy()
                    tmp['No of Collaterals'] = pd.to_numeric(tmp['No of Collaterals'], errors='coerce').fillna(0)
                    tmp['Category'] = safe_series(tmp, 'Category').map(normalize_category)
                    t1 = (tmp.groupby('Category', as_index=True)['No of Collaterals']
                            .sum()
                            .reindex(FIXED_CATEGORIES, fill_value=0)
                            .rename('Total Collateral')
                            .reset_index())
                else:
                    t1 = pd.DataFrame({"Category": FIXED_CATEGORIES, "Total Collateral": 0})

                # Pipeline of Tasks by Category (requested window, open only)
                if not rdf.empty:
                    tmp = rdf.copy()
                    status = safe_series(tmp, "Status").fillna("")
                    open_mask = ~status.isin(["Completed", "Design Completed", "Copy Completed"])
                    tmp = tmp.loc[open_mask]
                    tmp['Category'] = safe_series(tmp, 'Category').map(normalize_category)
                    t2 = (tmp.groupby('Category', as_index=True)
                            .size()
                            .reindex(FIXED_CATEGORIES, fill_value=0)
                            .rename('Open Pipeline Tasks')
                            .reset_index())
                else:
                    t2 = pd.DataFrame({"Category": FIXED_CATEGORIES, "Open Pipeline Tasks": 0})

                # Top 7 Clients (completed)
                if not cdf.empty and 'No of Collaterals' in cdf.columns and 'Client' in cdf.columns:
                    tmp = cdf.copy()
                    tmp['No of Collaterals'] = pd.to_numeric(tmp['No of Collaterals'], errors='coerce').fillna(0)
                    t3 = (tmp.groupby('Client', as_index=False)['No of Collaterals']
                            .sum()
                            .rename(columns={'No of Collaterals': 'Total Collateral'})
                            .sort_values('Total Collateral', ascending=False)
                            .head(7))
                else:
                    t3 = pd.DataFrame({"Client": [], "Total Collateral": []})

                # Top 7 Sectors (completed)
                if not cdf.empty and 'No of Collaterals' in cdf.columns and 'Sector' in cdf.columns:
                    tmp = cdf.copy()
                    tmp['No of Collaterals'] = pd.to_numeric(tmp['No of Collaterals'], errors='coerce').fillna(0)
                    t4 = (tmp.groupby('Sector', as_index=False)['No of Collaterals']
                            .sum()
                            .rename(columns={'No of Collaterals': 'Total Collateral'})
                            .sort_values('Total Collateral', ascending=False)
                            .head(7))
                else:
                    t4 = pd.DataFrame({"Sector": [], "Total Collateral": []})

                return t1, t2, t3, t4, cdf, rdf

            # Build current tables for display and capture selections
            collateral_delivery_df, pipeline_by_category_df, top_clients_df, top_sectors_df, cdf_all, rdf_all = \
                build_tables(filtered_task_completed_base, filtered_task_requested_base, st.session_state.xf)

            # -----------------------
            # TABLES (interactive) — set filters by clicking rows
            # -----------------------
            t1, t2 = st.columns(2)
            with t1:
                st.subheader("Collateral Delivery by Category type")
                sel = build_aggrid(collateral_delivery_df, key="grid_cat")
                if sel:
                    set_filter("Category", sel.get("Category"))
            with t2:
                st.subheader("Pipeline of Tasks by Category")
                sel = build_aggrid(pipeline_by_category_df, key="grid_pipe")
                if sel:
                    set_filter("Category", sel.get("Category"))  # pipeline also sets Category filter

            b1, b2 = st.columns(2)
            with b1:
                st.subheader("Top 7 clients based on quantity of collaterals")
                sel = build_aggrid(top_clients_df, key="grid_client")
                if sel:
                    set_filter("Client", sel.get("Client"))
            with b2:
                st.subheader("Top 7 Sectors on Quantity of Collateral")
                sel = build_aggrid(top_sectors_df, key="grid_sector")
                if sel:
                    set_filter("Sector", sel.get("Sector"))

            c1, c2, c3 = st.columns([1,1,2])
            with c1:
                if st.button("Clear filters"):
                    set_filter("Category", None); set_filter("Client", None); set_filter("Sector", None)
                    st.experimental_rerun()
            with c2:
                st.write("**Active filters:**")
                st.write({k: v for k, v in st.session_state.xf.items() if v})

            # -----------------------
            # KPIs — recompute from filtered frames
            # -----------------------
            cdf, rdf = apply_cross_filters(filtered_task_completed_base, filtered_task_requested_base, st.session_state.xf)

            noc = safe_series(cdf, "No of Collaterals", dtype="float")
            noc = pd.to_numeric(noc, errors="coerce")
            collateral_shipped = float(noc.sum()) if not noc.empty else 0

            cat_norm_completed = safe_series(cdf, "Category").map(normalize_category)
            content_pieces_shipped = int(noc[cat_norm_completed == "Copy"].sum()) if not noc.empty else 0

            requests_received = int(len(rdf))
            status_req = safe_series(rdf, "Status").fillna("")
            pipeline_pending = int((~status_req.isin(["Completed", "Design Completed", "Copy Completed"])).sum())

            def count_leads(df):
                if df is None or df.empty:
                    return 0
                ld = safe_series(df, "Lead Date")
                return int(ld.between(pd.Timestamp(start_date), pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)).sum())

            new_leads_generated = count_leads(df_sales_leads)
            brand_leads_generated = count_leads(df_brand_leads)
            partner_leads_generated = count_leads(df_partner_leads)

            rcat_norm = safe_series(rdf, "Category").map(normalize_category).fillna("")
            campaigns_created = int((rcat_norm == "Campaign").sum())
            initiatives_taken = int((rcat_norm == "Initiatives").sum())

            st.markdown("---")

            # KPI metrics layout (now reflects table selections)
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

    except Exception as e:
        st.warning(f"Failed to parse uploaded file, showing placeholder data. Error: {e}")
else:
    st.info("Upload an Excel file in the sidebar to begin.")
