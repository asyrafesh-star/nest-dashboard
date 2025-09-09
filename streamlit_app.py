import io
from typing import Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Business Analytics Dashboard", layout="wide")

# =========================
# Helpers & header mapping
# =========================

# Map many possible header variants -> canonical keys (to match your HTML version)
HEADER_MAP: Dict[str, List[str]] = {
    "companyName": ["COMPANY_NAME", "Company Name", "COMPANY"],
    "state": ["STATE", "Negeri", "NEGERI", "State"],
    "district": ["DISTRICT", "Daerah", "DAERAH", "District"],
    "registrationNumber": ["REGISTRATION_NUMBER_SSM", "SSM", "Registration Number"],
    "businessType": ["BUSINESS_TYPE", "TYPE_OF_BUSINESS", "Business Type"],
    "industry": ["INDUSTRY", "SECTOR", "Industry"],
    "basket": ["BASKET"],
    "group": ["GROUP"],
    "ethnicity": ["ETHNICITY"],

    # Revenues
    "baselineRevenue2023": ["BASELINE_REVENUE_2023", "Revenue 2023 Baseline"],
    "baselineRevenue2024": ["BASELINE_REVENUE_2024", "Revenue 2024 Baseline"],
    "projection2024": ["REVENUE_2024_PROJECTION", "Projection 2024"],
    "actual2024": ["REVENUE_2024_UPDATE", "ACTUAL_2024", "Actual 2024"],
    "projection2025": ["REVENUE_2025_PROJECTION", "Projection 2025"],

    # Jobs / manpower
    "baselineJobCreation": ["BASELINE_MANPOWER_2024", "BASELINE_MANPOWER_2023", "Baseline Job Creation"],
    "projectionJobCreation2024": ["MANPOWER_2024_PROJECTION", "GROWTH_JOB_CREATION_2024"],
    "actualJobCreation": ["ACTUAL_GROWTH_JOB_CREATION_2024", "MANPOWER_2024_UPDATE"],
    "projectionJobCreation2025": ["MANPOWER_2025_PROJECTION"],

    # Status & achievements
    "status": ["STATUS", "MD_STATUS_COMPANIES", "Status"],
    "salesAchieved": ["SELF_DECLARATION_2024"],
    "jobsAchieved": ["PERCENTAGE_JOB_CREATION_2024", "JOBS_ACHIEVED_2024"],
    # Optional sales growth % to infer achievements
    "growthSalesPct2024": ["PERCENTAGE_SALES_2024", "% SALES 2024", "PERC_SALES_2024"],
    "growthSalesRM2024": ["GROWTH_SALES_2024"],
}

CANON_COLS = list(HEADER_MAP.keys())

def _to_num(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return 0.0
    try:
        return float(str(v).replace(",", "").strip())
    except Exception:
        return 0.0

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # create lowercase->original map for easy matching
    lowered = {c.lower().strip(): c for c in df.columns}
    def pick(aliases):
        for a in aliases:
            c = lowered.get(a.lower().strip())
            if c is not None:
                return c
        return None

    out = pd.DataFrame()
    for canon, aliases in HEADER_MAP.items():
        src = pick(aliases)
        if src is not None:
            out[canon] = df[src]
        else:
            out[canon] = pd.NA

    # Coerce numerics
    for c in [
        "baselineRevenue2023", "baselineRevenue2024", "projection2024", "actual2024", "projection2025",
        "baselineJobCreation", "projectionJobCreation2024", "actualJobCreation", "projectionJobCreation2025",
        "growthSalesPct2024", "growthSalesRM2024",
    ]:
        out[c] = out[c].map(_to_num)

    # Strings (clean)
    for c in ["companyName", "state", "district", "registrationNumber", "businessType", "industry", "basket", "group", "ethnicity", "status"]:
        out[c] = out[c].astype("string").fillna("").str.strip()

    # Achievements inference (sales)
    sales_decl = out["salesAchieved"].astype("string").str.lower().fillna("")
    sales_flag = sales_decl.isin(["achieved", "yes", "y", "true", "1"])
    sales_flag = sales_flag | (out["growthSalesRM2024"] > 0) | (out["growthSalesPct2024"] > 0)
    out["salesAchieved"] = sales_flag

    # Achievements inference (jobs): prefer explicit %, else positive actualJobCreation
    jobs_pct_present = ~pd.isna(out["jobsAchieved"]) & (out["jobsAchieved"].astype("string") != "")
    # If there is a numeric percentage column mapped into jobsAchieved by mistake, try to coerce
    try:
        jobs_pct_numeric = pd.to_numeric(out["jobsAchieved"], errors="coerce").fillna(0)
    except Exception:
        jobs_pct_numeric = pd.Series([0] * len(out))
    out["jobsAchieved"] = (jobs_pct_numeric > 0) | (out["actualJobCreation"] > 0)

    # Better "actualJobCreation": if empty but we have manpower baseline & update, compute delta
    need_delta = (out["actualJobCreation"] == 0) & (out["baselineJobCreation"].notna())
    # Try to locate 2024 update column from original df if not already mapped
    mp_update_candidates = ["MANPOWER_2024_UPDATE", "Manpower 2024 Update"]
    for cand in mp_update_candidates:
        if cand in df.columns:
            delta = df[cand].map(_to_num) - out["baselineJobCreation"].map(_to_num)
            out.loc[need_delta, "actualJobCreation"] = delta[need_delta]
            break

    # Defaults
    out["status"] = out["status"].replace({"<NA>": ""}).fillna("")
    out.loc[out["status"] == "", "status"] = "Active"

    return out

def make_template_bytes() -> bytes:
    # Build an Excel template with canonical headers + sample rows
    template_cols = [
        "Company Name","State","District","Registration Number","Business Type","Industry","Basket","Group","Ethnicity",
        "Baseline Revenue 2023","Baseline Revenue 2024","Projection 2024","Actual 2024","Projection 2025",
        "Baseline Job Creation","Projection Job Creation 2024","Actual Job Creation","Projection Job Creation 2025",
        "Status","Sales Achieved 2024","Jobs Achieved 2024"
    ]
    sample = [
        ["Tech Solutions Sdn Bhd","Selangor","Petaling Jaya","SSM123456","SME","Technology","A","Group A","Malay",
         2100000,2300000,2500000,2650000,2800000,40,45,48,52,"Active","Achieved","Achieved"],
        ["Manufacturing Plus","Penang","Georgetown","SSM987654","MNC","Manufacturing","B","Group B","Chinese",
         7500000,8000000,8200000,8500000,9000000,110,120,125,135,"Active","Achieved","Achieved"],
    ]
    df = pd.DataFrame(sample, columns=template_cols)
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="NEST Template")
    return bio.getvalue()

# =========================
# Sidebar: data IO
# =========================
st.sidebar.title("📊 Data Management")

# Download Template
st.sidebar.download_button(
    "📥 Download Excel Template",
    data=make_template_bytes(),
    file_name="NEST_Template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    help="Template with correct columns & sample rows"
)

uploaded = st.sidebar.file_uploader("📁 Upload Excel/CSV", type=["xlsx", "xls", "csv"], help="Upload Book2 or any dataset")
page_size = st.sidebar.selectbox("Rows per page", [10, 20, 50, 100], index=1)

@st.cache_data(show_spinner=False)
def load_any(file) -> pd.DataFrame:
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file, engine="openpyxl")

# Seed sample (so the UI is not empty if no file yet)
seed = pd.DataFrame([
    dict(
        COMPANY_NAME="Tech Solutions Sdn Bhd", STATE="Selangor", DISTRICT="Petaling Jaya",
        REGISTRATION_NUMBER_SSM="SSM123456", BUSINESS_TYPE="SME", INDUSTRY="Technology",
        BASKET="A", GROUP="Group A", ETHNICITY="Malay",
        BASELINE_REVENUE_2023=2_100_000, BASELINE_REVENUE_2024=2_300_000,
        REVENUE_2024_PROJECTION=2_500_000, REVENUE_2024_UPDATE=2_650_000, REVENUE_2025_PROJECTION=2_800_000,
        BASELINE_MANPOWER_2024=40, MANPOWER_2024_PROJECTION=45, ACTUAL_GROWTH_JOB_CREATION_2024=48, MANPOWER_2025_PROJECTION=52,
        STATUS="Active", SELF_DECLARATION_2024="Achieved"
    ),
    dict(
        COMPANY_NAME="Manufacturing Plus", STATE="Penang", DISTRICT="Georgetown",
        REGISTRATION_NUMBER_SSM="SSM987654", BUSINESS_TYPE="MNC", INDUSTRY="Manufacturing",
        BASKET="B", GROUP="Group B", ETHNICITY="Chinese",
        BASELINE_REVENUE_2023=7_500_000, BASELINE_REVENUE_2024=8_000_000,
        REVENUE_2024_PROJECTION=8_200_000, REVENUE_2024_UPDATE=8_500_000, REVENUE_2025_PROJECTION=9_000_000,
        BASELINE_MANPOWER_2024=110, MANPOWER_2024_PROJECTION=120, ACTUAL_GROWTH_JOB_CREATION_2024=125, MANPOWER_2025_PROJECTION=135,
        STATUS="Active", SELF_DECLARATION_2024="Achieved"
    ),
    dict(
        COMPANY_NAME="Retail Express", STATE="Kuala Lumpur", DISTRICT="Bukit Bintang",
        REGISTRATION_NUMBER_SSM="SSM456789", BUSINESS_TYPE="SME", INDUSTRY="Retail",
        BASKET="C", GROUP="Group A", ETHNICITY="Indian",
        BASELINE_REVENUE_2023=1_900_000, BASELINE_REVENUE_2024=1_800_000,
        REVENUE_2024_PROJECTION=2_000_000, REVENUE_2024_UPDATE=1_750_000, REVENUE_2025_PROJECTION=1_900_000,
        BASELINE_MANPOWER_2024=35, MANPOWER_2024_PROJECTION=32, ACTUAL_GROWTH_JOB_CREATION_2024=30, MANPOWER_2025_PROJECTION=33,
        STATUS="Review", SELF_DECLARATION_2024="Not"
    ),
])

if uploaded is not None:
    raw_df = load_any(uploaded)
else:
    raw_df = seed

df = normalize_columns(raw_df)

# =========================
# Title & filters
# =========================
st.title("🌐 Business Analytics Dashboard")

with st.expander("ℹ️ How this works", expanded=False):
    st.write(
        "- Upload **Excel/CSV**. We normalize your column names (like `STATE`, `INDUSTRY`, "
        "`REVENUE_2024_UPDATE`, `ACTUAL_GROWTH_JOB_CREATION_2024`, etc.) into a standard schema.\n"
        "- Filters & charts update live.\n"
        "- Download filtered data as CSV/Excel."
    )

# Filters
c1, c2, c3, c4 = st.columns(4)
states = sorted([s for s in df["state"].dropna().unique() if str(s).strip() != ""])
industries = sorted([s for s in df["industry"].dropna().unique() if str(s).strip() != ""])
btypes = sorted([s for s in df["businessType"].dropna().unique() if str(s).strip() != ""])
groups = sorted([s for s in df["group"].dropna().unique() if str(s).strip() != ""])

sel_state = c1.selectbox("State", ["All"] + states, index=0)
sel_ind = c2.selectbox("Industry", ["All"] + industries, index=0)
sel_bt = c3.selectbox("Business Type", ["All"] + btypes, index=0)
sel_grp = c4.selectbox("Group", ["All"] + groups, index=0)

fdf = df.copy()
if sel_state != "All":
    fdf = fdf[fdf["state"] == sel_state]
if sel_ind != "All":
    fdf = fdf[fdf["industry"] == sel_ind]
if sel_bt != "All":
    fdf = fdf[fdf["businessType"] == sel_bt]
if sel_grp != "All":
    fdf = fdf[fdf["group"] == sel_grp]

# =========================
# KPIs
# =========================
def human_rm(n: float) -> str:
    if n >= 1e9:
        return f"RM {n/1e9:.2f}B"
    if n >= 1e6:
        return f"RM {n/1e6:.1f}M"
    if n >= 1e3:
        return f"RM {n/1e3:.1f}K"
    return f"RM {n:,.0f}"

total_companies = len(fdf)
total_revenue = float((fdf["actual2024"].replace(pd.NA, 0).map(_to_num)) \
                      .where(lambda s: s > 0, other=fdf["baselineRevenue2024"].map(_to_num)).sum())
total_jobs = float(fdf["actualJobCreation"].map(_to_num).sum())
baseline_rev = float(fdf["baselineRevenue2024"].map(_to_num).sum())
growth_pct = ((total_revenue - baseline_rev) / baseline_rev * 100) if baseline_rev > 0 else 0.0

m1, m2, m3, m4 = st.columns(4)
m1.metric("🏢 Total Companies", f"{total_companies:,}")
m2.metric("💰 Total Revenue 2024", human_rm(total_revenue))
m3.metric("👥 Jobs Created (2024)", f"{int(total_jobs):,}")
m4.metric("📈 Growth Rate", f"{growth_pct:.1f}%")

st.markdown("---")

# =========================
# Charts
# =========================
cc1, cc2 = st.columns(2)

if not fdf.empty:
    # Revenue by State
    rev_state = (
        fdf.assign(
            revenue=fdf["actual2024"].where(fdf["actual2024"] > 0, fdf["baselineRevenue2024"]).map(_to_num)
        )
        .groupby("state", as_index=False)["revenue"].sum()
        .sort_values("revenue", ascending=False)
    )
    if not rev_state.empty:
        fig1 = px.bar(rev_state, x="state", y="revenue", title="Revenue by State (Actual 2024; fallback Baseline)")
        cc1.plotly_chart(fig1, width="stretch")

    # Jobs by State
    jobs_state = (
        fdf.assign(jobs=fdf["actualJobCreation"].map(_to_num))
        .groupby("state", as_index=False)["jobs"].sum()
        .sort_values("jobs", ascending=False)
    )
    if not jobs_state.empty:
        fig2 = px.bar(jobs_state, x="state", y="jobs", title="Job Creation by State (2024)")
        cc2.plotly_chart(fig2, width="stretch")

cc3, cc4, cc5 = st.columns(3)

# Industry distribution (company counts)
if not fdf.empty and "industry" in fdf:
    ind_count = fdf.groupby("industry", as_index=False).size().rename(columns={"size": "companies"})
    if not ind_count.empty:
        fig3 = px.pie(ind_count, names="industry", values="companies", title="Industry Distribution")
        cc3.plotly_chart(fig3, width="stretch")

# State distribution (company counts)
if not fdf.empty and "state" in fdf:
    st_count = fdf.groupby("state", as_index=False).size().rename(columns={"size": "companies"})
    if not st_count.empty:
        fig4 = px.pie(st_count, names="state", values="companies", title="State Distribution")
        cc4.plotly_chart(fig4, width="stretch")

# Achievements donut (Sales + Jobs)
if not fdf.empty:
    sales_ach = int(fdf["salesAchieved"].sum())
    sales_not = int(len(fdf) - sales_ach)
    jobs_ach = int(fdf["jobsAchieved"].sum())
    jobs_not = int(len(fdf) - jobs_ach)

    ach_df = pd.DataFrame(
        {"Metric": ["Sales Achieved", "Sales Not", "Jobs Achieved", "Jobs Not"],
         "Count": [sales_ach, sales_not, jobs_ach, jobs_not]}
    )
    fig5 = px.pie(ach_df, names="Metric", values="Count", hole=0.5, title="Achievements (Sales / Jobs)")
    cc5.plotly_chart(fig5, width="stretch")

st.markdown("---")

# =========================
# Search + pagination table
# =========================
st.subheader("📋 Company Data Table")

if "table_page" not in st.session_state:
    st.session_state.table_page = 1

q = st.text_input("Search (company / any field)", placeholder="Type and press Enter…")
if q:
    qlow = q.strip().lower()
    mask = fdf.apply(lambda r: any(qlow in str(v).lower() for v in r.values), axis=1)
    tdf = fdf[mask].copy()
else:
    tdf = fdf.copy()

total_rows = len(tdf)
max_page = max(1, (total_rows + page_size - 1) // page_size)
st.session_state.table_page = min(st.session_state.table_page, max_page)

col_prev, col_info, col_next = st.columns([1, 3, 1])
with col_prev:
    if st.button("◀ Previous", use_container_width=True) and st.session_state.table_page > 1:
        st.session_state.table_page -= 1
with col_info:
    st.write(f"Page **{st.session_state.table_page}** of **{max_page}** — Showing **{total_rows}** rows")
with col_next:
    if st.button("Next ▶", use_container_width=True) and st.session_state.table_page < max_page:
        st.session_state.table_page += 1

start = (st.session_state.table_page - 1) * page_size
end = start + page_size

show_cols = [
    "companyName", "state", "industry", "businessType",
    "baselineRevenue2024", "actual2024", "actualJobCreation", "status"
]
present_cols = [c for c in show_cols if c in tdf.columns]
display_df = tdf[present_cols].copy()
display_df.rename(columns={
    "companyName": "Company",
    "state": "State",
    "industry": "Industry",
    "businessType": "Type",
    "baselineRevenue2024": "Baseline Revenue 2024 (RM)",
    "actual2024": "Actual 2024 (RM)",
    "actualJobCreation": "Jobs Created 2024",
    "status": "Status",
}, inplace=True)

st.dataframe(display_df.iloc[start:end], use_container_width=True)

# =========================
# Downloads for filtered table
# =========================
st.markdown("#### ⬇️ Export filtered data")
csv_bytes = tdf.to_csv(index=False).encode("utf-8")
st.download_button("Export CSV", data=csv_bytes, file_name="dashboard_filtered.csv", mime="text/csv")

bio = io.BytesIO()
with pd.ExcelWriter(bio, engine="openpyxl") as xw:
    tdf.to_excel(xw, index=False, sheet_name="Filtered")
st.download_button(
    "Export Excel",
    data=bio.getvalue(),
    file_name="dashboard_filtered.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
