import io
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="NEST DASHBOARD 2024", layout="wide")

# --------------------------------------------------------------------------------------
# THEME SWITCHER (Dark / Light)
# --------------------------------------------------------------------------------------
THEMES = {
    "Dark": {
        "PRIMARY_BG": "#0c2340",
        "PANEL_BG":   "#0d2a5a",
        "CARD_BG":    "#0f326e",
        "TEXT":       "#eaf2ff",
        "ACCENT":     "#ffcc00",
        "ACCENT_RED": "#e74c3c",
        "ACCENT_GREEN": "#27ae60",
        "ACCENT_BLUE":  "#1f78ff",
        "ACCENT_PURPLE":"#9b59b6",
        "plotly_template": "plotly_dark",
        "grid_color": "rgba(255,255,255,.08)",
        "border_color": "rgba(255,255,255,.06)",
        "shadow": "0 6px 20px rgba(0,0,0,.35)",
    },
    "Light": {
        "PRIMARY_BG": "#f7f9fc",
        "PANEL_BG":   "#ffffff",
        "CARD_BG":    "#ffffff",
        "TEXT":       "#000000",
        "ACCENT":     "#7c3aed",
        "ACCENT_RED": "#dc2626",
        "ACCENT_GREEN": "#16a34a",
        "ACCENT_BLUE":  "#2563eb",
        "ACCENT_PURPLE":"#9333ea",
        "plotly_template": "plotly_white",
        "grid_color": "rgba(0,0,0,.08)",
        "border_color": "rgba(0,0,0,.08)",
        "shadow": "0 6px 18px rgba(2,6,23,.06)",
    },
}

with st.sidebar:
    theme_choice = st.selectbox("🎨 Theme", list(THEMES.keys()), index=0)

T = THEMES[theme_choice]
PRIMARY_BG   = T["PRIMARY_BG"]
PANEL_BG     = T["PANEL_BG"]
CARD_BG      = T["CARD_BG"]
TEXT         = T["TEXT"]
ACCENT       = T["ACCENT"]
ACCENT_RED   = T["ACCENT_RED"]
ACCENT_GREEN = T["ACCENT_GREEN"]
ACCENT_BLUE  = T["ACCENT_BLUE"]
ACCENT_PURPLE= T["ACCENT_PURPLE"]
PLOTLY_TPL   = T["plotly_template"]
GRID_COLOR   = T["grid_color"]
BORDER_COLOR = T["border_color"]
SHADOW       = T["shadow"]

# Colorful gradients used when Light theme is active
COLORFUL_GRADS = [
    "linear-gradient(135deg,#6366f1,#a78bfa)",  # indigo → purple
    "linear-gradient(135deg,#f59e0b,#fb7185)",  # amber → rose
    "linear-gradient(135deg,#06b6d4,#22c55e)",  # cyan → green
    "linear-gradient(135deg,#ef4444,#f97316)",  # red → orange
    "linear-gradient(135deg,#10b981,#3b82f6)",  # emerald → blue
]

# Global CSS (brace-escaped)
st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {PRIMARY_BG}; }}
    .stSidebar {{ background-color: { '#ffffff' if theme_choice == 'Light' else PRIMARY_BG }; }}
    .block-container {{ padding-top: .8rem; padding-bottom: 2rem; }}
    .st-plotly-chart {{ background-color: { '#ffffff' if theme_choice == 'Light' else PANEL_BG }; }}
    h1,h2,h3,h4,h5,p,span,div,li,th,td,label {{ color: {TEXT} !important; }}
    .dash-card {{
        background: {CARD_BG if theme_choice == 'Dark' else COLORFUL_GRADS[0]};
        border: 1px solid {BORDER_COLOR};
        border-radius: 12px; padding: 14px 16px; box-shadow: {SHADOW};
    }}
    .kpi-num {{ font-weight: 800; font-size: 36px; color: {ACCENT if theme_choice == 'Dark' else '#000000'}; }}
    .kpi-label {{ opacity:.9; font-size: 12px; letter-spacing: .08em; color: {TEXT} !important; }}
    .kpi-stack {{ text-align:center; color: {TEXT} !important; }}
    .headline {{ font-size:28px; font-weight:900; letter-spacing:.04em; }}
    .right-panel {{ background: {PANEL_BG}; border-radius: 12px; border:1px solid {BORDER_COLOR}; padding:16px; }}
    .pill {{
        background: rgba(0,0,0,.06); padding:4px 8px; border-radius:999px;
        font-size:11px; border:1px solid {BORDER_COLOR};
        color:{TEXT};
    }}
    .highlight-card {{
        border-radius: 12px; padding: 14px 16px;
        border: 1px solid {BORDER_COLOR}; box-shadow: {SHADOW}; height:100%;
    }}
    .highlight-title {{ font-size:12px; letter-spacing:.08em; opacity:.9; }}
    .highlight-big {{ font-size:28px; font-weight:900; }}
    .highlight-sub {{ font-size:12px; opacity:.8; }}
    .bullets li {{ margin:2px 0; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------------------
st.markdown(
    """
    <div style="background:rgba(0,0,0,.08);border-radius:12px;border:1px solid rgba(0,0,0,.06);
         padding:14px 18px;margin-bottom:12px;">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div>
                <div class="headline">NEST DASHBOARD 2024</div>
                <div><span>STATE</span> &nbsp;|&nbsp; <span>INDUSTRY</span> &nbsp;|&nbsp; <span>GROUP</span></div>
            </div>
            <div><span class="pill">Overall</span></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
HEADER_MAP: Dict[str, List[str]] = {
    "company": ["COMPANY_NAME","Company Name","COMPANY"],
    "state": ["STATE","Negeri"],
    "industry": ["INDUSTRY","Sector"],
    "group": ["GROUP"],
    "tc_member": ["TC_MEMBER","KAM","PIC","TC"],
    "rev_baseline_prev": ["BASELINE_REVENUE_2023","Revenue 2023"],
    "rev_baseline": ["BASELINE_REVENUE_2024","Revenue 2024 Baseline"],
    "rev_projection": ["REVENUE_2024_PROJECTION","Projection 2024"],
    "rev_actual": ["REVENUE_2024_UPDATE","Actual 2024","ACTUAL_2024"],
    "jobs_baseline": ["BASELINE_MANPOWER_2024"],
    "jobs_projection": ["MANPOWER_2024_PROJECTION"],
    "jobs_update": ["MANPOWER_2024_UPDATE"],
    "jobs_actual_growth": ["ACTUAL_GROWTH_JOB_CREATION_2024"],
    "sale_status": ["SELF_DECLARATION_2024","SALE_STATUS"],
    "job_status": ["PERCENTAGE_JOB_CREATION_2024","JOB_STATUS"],
}

def pick_col(df: pd.DataFrame, aliases: List[str]):
    lower = {c.lower().strip(): c for c in df.columns}
    for a in aliases:
        c = lower.get(a.lower().strip())
        if c:
            return c
    return None

def num(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 0.0
    try:
        return float(str(v).replace(",", "").strip())
    except Exception:
        return 0.0

@st.cache_data(show_spinner=False)
def load_any(file):
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file, engine="openpyxl")

# --------------------------------------------------------------------------------------
# Sidebar: Template + Upload
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📦 Data")
    upl = st.file_uploader("Upload Excel/CSV", type=["xlsx","xls","csv"])

    def template_bytes():
        cols = [
            "COMPANY_NAME","STATE","INDUSTRY","GROUP","TC_MEMBER",
            "BASELINE_REVENUE_2023","BASELINE_REVENUE_2024","REVENUE_2024_PROJECTION","REVENUE_2024_UPDATE",
            "BASELINE_MANPOWER_2024","MANPOWER_2024_PROJECTION","MANPOWER_2024_UPDATE","ACTUAL_GROWTH_JOB_CREATION_2024",
            "SELF_DECLARATION_2024","PERCENTAGE_JOB_CREATION_2024"
        ]
        sample = [
            ["Tech Solutions","SELANGOR","SERVICES","3A","DATIN SHARIDEE",
             1800000,2300000,2500000,2650000,40,45,48,8,"Achieved",25],
            ["BestMart","PULAU PINANG","MANUFACTURING","2","EN FAUZAN",
             1200000,1500000,1650000,1600000,20,25,26,6,"Not Achieved",10],
        ]
        bio = io.BytesIO()
        pd.DataFrame(sample, columns=cols).to_excel(bio, index=False, sheet_name="NEST Data")
        return bio.getvalue()

    st.download_button(
        "📥 Download Excel Template",
        data=template_bytes(),
        file_name="NEST_Template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

# --------------------------------------------------------------------------------------
# Load data (demo fallback)
# --------------------------------------------------------------------------------------
if upl is not None:
    raw = load_any(upl)
else:
    raw = pd.DataFrame(
        [
            ["Tech Solutions","SELANGOR","SERVICES","3A","DATIN SHARIDEE",1800000,2300000,2500000,2650000,40,45,48,8,"Achieved",25],
            ["BestMart","PULAU PINANG","MANUFACTURING","2","EN FAUZAN",1200000,1500000,1650000,1600000,20,25,26,6,"Not Achieved",10],
            ["Swift Bridge","PERAK","AGRIBUSINESS","1","PN SURIALA", 800000,1100000,1250000,1490000,10,13,17,9,"Achieved",40],
            ["HealthCare Ideal","PULAU PINANG","SERVICES","3A","EM FIRDAUS",2000000,2600000,3000000,6240000,60,75,119,59,"Achieved",65],
        ],
        columns=[
            "COMPANY_NAME","STATE","INDUSTRY","GROUP","TC_MEMBER",
            "BASELINE_REVENUE_2023","BASELINE_REVENUE_2024","REVENUE_2024_PROJECTION","REVENUE_2024_UPDATE",
            "BASELINE_MANPOWER_2024","MANPOWER_2024_PROJECTION","MANPOWER_2024_UPDATE","ACTUAL_GROWTH_JOB_CREATION_2024",
            "SELF_DECLARATION_2024","PERCENTAGE_JOB_CREATION_2024",
        ],
    )

# --------------------------------------------------------------------------------------
# Normalize columns
# --------------------------------------------------------------------------------------
norm = pd.DataFrame()
for key, aliases in HEADER_MAP.items():
    col = pick_col(raw, aliases)
    norm[key] = raw[col] if col is not None else pd.NA

for c in ["company","state","industry","group","tc_member","sale_status"]:
    norm[c] = norm[c].astype("string").fillna("")

for c in ["rev_baseline_prev","rev_baseline","rev_projection","rev_actual",
          "jobs_baseline","jobs_projection","jobs_update","jobs_actual_growth","job_status"]:
    if c in norm:
        norm[c] = norm[c].apply(num)

if (norm["jobs_actual_growth"].fillna(0).sum() == 0) and ("jobs_update" in norm) and ("jobs_baseline" in norm):
    norm["jobs_actual_growth"] = (norm["jobs_update"].apply(num) - norm["jobs_baseline"].apply(num)).clip(lower=0)

norm["growth_pct"] = np.where(
    norm["rev_baseline"].apply(num) > 0,
    (norm["rev_actual"].apply(num) - norm["rev_baseline"].apply(num)) / norm["rev_baseline"].apply(num) * 100,
    0,
)

# --------------------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------------------
fc1, fc2, fc3, fc4 = st.columns([1.5,1.5,1.2,1.2])
with fc1:
    states = sorted(norm["state"].dropna().unique().tolist())
    f_state = st.multiselect("STATE", states, default=states)
with fc2:
    inds = sorted(norm["industry"].dropna().unique().tolist())
    f_ind = st.multiselect("INDUSTRY", inds, default=inds)
with fc3:
    grps = sorted(norm["group"].dropna().unique().tolist())
    f_grp = st.multiselect("GROUP", grps, default=grps[:3] if len(grps)>3 else grps)
with fc4:
    tcs = sorted(norm["tc_member"].dropna().unique().tolist())
    f_tc = st.multiselect("TC MEMBER", tcs, default=tcs)

filt = norm.copy()
if f_state: filt = filt[filt["state"].isin(f_state)]
if f_ind:   filt = filt[filt["industry"].isin(f_ind)]
if f_grp:   filt = filt[filt["group"].isin(f_grp)]
if f_tc:    filt = filt[filt["tc_member"].isin(f_tc)]

# --------------------------------------------------------------------------------------
# KPI row
# --------------------------------------------------------------------------------------
k1, k2, k3, k4, k5, k6 = st.columns([1.1,1.6,1.6,1.6,1.4,1.6])

def kpi_card(container, label, value_html, grad_idx=0):
    bg = CARD_BG if theme_choice == 'Dark' else COLORFUL_GRADS[grad_idx % len(COLORFUL_GRADS)]
    with container:
        st.markdown(
            f"""
            <div class="dash-card kpi-stack" style="background:{bg}">
                <div class="kpi-label">{label}</div>
                <div class="kpi-num">{value_html}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

total_company = len(filt)
base_sales = float(filt["rev_baseline"].apply(num).sum())
base_jobs = int(filt["jobs_baseline"].apply(num).sum())
total_growth_sale = float((filt["rev_actual"].apply(num) - filt["rev_baseline"].apply(num)).sum())
avg_growth_pct = float(filt["growth_pct"].replace([np.inf, -np.inf], 0).fillna(0).mean())
total_new_jobs = int(filt["jobs_actual_growth"].apply(num).sum())
avg_job_pct = float(
    np.where(filt["jobs_baseline"].apply(num) > 0,
             (filt["jobs_actual_growth"].apply(num) / np.maximum(filt["jobs_baseline"].apply(num),1)) * 100,
             0).mean()
)

kpi_card(k1, "TOTAL COMPANY", f"{total_company:,}", grad_idx=0)
kpi_card(k2, "BASELINES<br/><span style='font-size:11px;opacity:.8'>SALES & JOBS</span>",
         f"<span style='font-size:36px'>RM{base_sales:,.0f}</span><br/><span style='font-size:36px'>{base_jobs:,}</span>", grad_idx=1)
kpi_card(k3, "TOTAL GROWTH SALE", f"RM{total_growth_sale:,.0f}", grad_idx=2)
kpi_card(k4, "AVERAGE GROWTH SALE %", f"{avg_growth_pct:.0f}%", grad_idx=3)
kpi_card(k5, "TOTAL NEW JOBS CREATION", f"{total_new_jobs:,}", grad_idx=4)
kpi_card(k6, "AVERAGE JOB CREATION %", f"{avg_job_pct:.0f}%", grad_idx=0)

st.markdown("<br/>", unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# Plotly helpers (theme-aware)
# --------------------------------------------------------------------------------------
def grouped_bar(df, x, series, title):
    fig = go.Figure()
    for col, nm, colr in series:
        fig.add_trace(go.Bar(x=df[x], y=df[col], name=nm, marker_color=colr))
    fig.update_layout(
        template=PLOTLY_TPL,
        barmode="group", height=340, margin=dict(l=10,r=10,t=40,b=10),
        plot_bgcolor=PANEL_BG, paper_bgcolor=PANEL_BG,
        font=dict(color='#000000' if theme_choice == 'Light' else TEXT, size=12),
        title=dict(text=title, x=0.02, y=0.98, font=dict(size=14, color='#000000' if theme_choice == 'Light' else TEXT))
    )
    fig.update_xaxes(
        showgrid=False, zeroline=False,
        tickfont=dict(color='#000000' if theme_choice == 'Light' else TEXT)
    )
    fig.update_yaxes(
        gridcolor=GRID_COLOR,
        tickfont=dict(color='#000000' if theme_choice == 'Light' else TEXT),
        title_font=dict(color='#000000' if theme_choice == 'Light' else TEXT)
    )
    return fig

def donut(labels, values, title):
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.58,
                           textinfo="value+percent", sort=False))
    fig.update_traces(
        marker=dict(line=dict(color=PANEL_BG, width=2)),
        textfont=dict(color='#000000' if theme_choice == 'Light' else TEXT)
    )
    fig.update_layout(
        template=PLOTLY_TPL,
        height=320, margin=dict(l=10,r=10,t=40,b=10),
        plot_bgcolor=PANEL_BG, paper_bgcolor=PANEL_BG,
        font=dict(color='#000000' if theme_choice == 'Light' else TEXT),
        title=dict(text=title, x=0.5, y=0.98, font=dict(size=14, color='#000000' if theme_choice == 'Light' else TEXT))
    )
    return fig

# --------------------------------------------------------------------------------------
# Sales by State / Industry / Group
# --------------------------------------------------------------------------------------
cA1, cA2, cA3 = st.columns([1.3,1.3,1.1])
if not filt.empty:
    by_state = (filt.groupby("state", as_index=False)
                .agg(baseline=("rev_baseline", lambda s: sum(map(num, s))),
                     projection=("rev_projection", lambda s: sum(map(num, s))),
                     actual=("rev_actual", lambda s: sum(map(num, s)))))
    cA1.plotly_chart(
        grouped_bar(by_state, "state",
                    [("baseline","BASELINE", ACCENT_BLUE),
                     ("projection","PROJECTION", ACCENT_PURPLE),
                     ("actual","ACTUAL", ACCENT_RED)],
                    "SALES BY STATE"),
        use_container_width=True
    )

    by_ind = (filt.groupby("industry", as_index=False)
              .agg(baseline=("rev_baseline", lambda s: sum(map(num, s))),
                   projection=("rev_projection", lambda s: sum(map(num, s))),
                   actual=("rev_actual", lambda s: sum(map(num, s)))))
    cA2.plotly_chart(
        grouped_bar(by_ind, "industry",
                    [("baseline","BASELINE", ACCENT_BLUE),
                     ("projection","PROJECTION", ACCENT_PURPLE),
                     ("actual","ACTUAL", ACCENT_RED)],
                    "SALES BY INDUSTRY"),
        use_container_width=True
    )

    by_grp = (filt.groupby("group", as_index=False)
              .agg(baseline=("rev_baseline", lambda s: sum(map(num, s))),
                   projection=("rev_projection", lambda s: sum(map(num, s))),
                   actual=("rev_actual", lambda s: sum(map(num, s)))))
    cA3.plotly_chart(
        grouped_bar(by_grp, "group",
                    [("baseline","BASELINE", ACCENT_BLUE),
                     ("projection","PROJECTION", ACCENT_PURPLE),
                     ("actual","ACTUAL", ACCENT_RED)],
                    "SALES BY GROUP"),
        use_container_width=True
    )

# --------------------------------------------------------------------------------------
# Sales/Jobs growth status & jobs breakdown
# --------------------------------------------------------------------------------------
cB1, cB2, cB3 = st.columns([1.0,1.3,1.3])
achieved = (filt["sale_status"].fillna("").str.lower().str.contains("achiev")).sum()
not_achieved = (filt["sale_status"].fillna("").str.lower().str.contains("not")).sum()
new_company = max(total_company - achieved - not_achieved, 0)
cB1.plotly_chart(donut(["Achieved","Not Achieved","New Company"],
                       [achieved, not_achieved, new_company], "SALES GROWTH STATUS"),
                 use_container_width=True)

by_state_jobs = (filt.groupby("state", as_index=False)
                 .agg(baseline=("jobs_baseline", lambda s: sum(map(num, s))),
                      projection=("jobs_projection", lambda s: sum(map(num, s))),
                      actual=("jobs_actual_growth", lambda s: sum(map(num, s)))))
cB2.plotly_chart(
    grouped_bar(by_state_jobs, "state",
                [("baseline","BASELINE", ACCENT_BLUE),
                 ("projection","PROJECTION", ACCENT_PURPLE),
                 ("actual","ACTUAL", ACCENT_RED)],
                "JOB CREATION BY STATE"),
    use_container_width=True
)

by_grp_jobs = (filt.groupby("group", as_index=False)
               .agg(baseline=("jobs_baseline", lambda s: sum(map(num, s))),
                    projection=("jobs_projection", lambda s: sum(map(num, s))),
                    actual=("jobs_actual_growth", lambda s: sum(map(num, s)))))
cB3.plotly_chart(
    grouped_bar(by_grp_jobs, "group",
                [("baseline","BASELINE", ACCENT_BLUE),
                 ("projection","PROJECTION", ACCENT_PURPLE),
                 ("actual","ACTUAL", ACCENT_RED)],
                "JOB CREATION BY GROUP"),
    use_container_width=True
)

# --------------------------------------------------------------------------------------
# Jobs growth donut + performance by state
# --------------------------------------------------------------------------------------
cC1, cC2 = st.columns([1.0,2.0])
job_ach = int((filt["jobs_actual_growth"].apply(num) > 0).sum())
job_not = int(total_company - job_ach)
cC1.plotly_chart(donut(["ACHIEVED","NOT ACHIEVED"], [job_ach, job_not], "JOBS GROWTH STATUS"),
                 use_container_width=True)

perf_state = (filt.assign(pos=(filt["growth_pct"]>0).astype(int))
              .groupby("state", as_index=False)
              .agg(positive=("pos","sum"))
              .sort_values("positive", ascending=False))
if not perf_state.empty:
    fig_perf = px.bar(perf_state, x="state", y="positive", title="OVERALL PERFORMANCE BY STATE")
    fig_perf.update_layout(
        template=PLOTLY_TPL, height=340, margin=dict(l=10,r=10,t=40,b=10),
        plot_bgcolor=PANEL_BG, paper_bgcolor=PANEL_BG,
        font=dict(color='#000000' if theme_choice == 'Light' else TEXT)
    )
    fig_perf.update_xaxes(
        showgrid=False,
        tickfont=dict(color='#000000' if theme_choice == 'Light' else TEXT)
    )
    fig_perf.update_yaxes(
        gridcolor=GRID_COLOR,
        tickfont=dict(color='#000000' if theme_choice == 'Light' else TEXT),
        title_font=dict(color='#000000' if theme_choice == 'Light' else TEXT)
    )
    cC2.plotly_chart(fig_perf, use_container_width=True)

st.markdown("<br/>", unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# UPGRADE PACK: Projection pies, stacked performance, Top-5 lists
# --------------------------------------------------------------------------------------
def stacked_perf(df: pd.DataFrame, by: str, title: str):
    tmp = (
        df.assign(
            Achieved=df["sale_status"].fillna("").str.lower().str.contains("achiev"),
            NotAchieved=df["sale_status"].fillna("").str.lower().str.contains("not"),
        )
        .groupby(by, as_index=False)
        .agg(
            Achieved=("Achieved", "sum"),
            NotAchieved=("NotAchieved", "sum"),
            Total=("sale_status", "count"),
        )
    )
    tmp["NewCompany"] = (tmp["Total"] - tmp["Achieved"] - tmp["NotAchieved"]).clip(lower=0)

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Achieved", x=tmp[by], y=tmp["Achieved"], marker_color=ACCENT_BLUE))
    fig.add_trace(go.Bar(name="Not Achieved", x=tmp[by], y=tmp["NotAchieved"], marker_color=ACCENT_RED))
    fig.add_trace(go.Bar(name="New Company", x=tmp[by], y=tmp["NewCompany"], marker_color=ACCENT_PURPLE))
    fig.update_layout(
        template=PLOTLY_TPL, barmode="stack", height=350, title=title,
        plot_bgcolor=PANEL_BG, paper_bgcolor=PANEL_BG, margin=dict(l=10,r=10,t=40,b=10),
        font=dict(color='#000000' if theme_choice == 'Light' else TEXT),
        legend=dict(orientation="h", y=1.1, font=dict(color='#000000' if theme_choice == 'Light' else TEXT))
    )
    fig.update_yaxes(
        gridcolor=GRID_COLOR,
        tickfont=dict(color='#000000' if theme_choice == 'Light' else TEXT),
        title_font=dict(color='#000000' if theme_choice == 'Light' else TEXT)
    )
    fig.update_xaxes(
        tickfont=dict(color='#000000' if theme_choice == 'Light' else TEXT)
    )
    return fig

p1, p2, p3 = st.columns(3)
total_bl = float(filt["rev_baseline"].apply(num).sum())
total_pr = float(filt["rev_projection"].apply(num).sum())
total_ac = float(filt["rev_actual"].apply(num).sum())
proj_fig = donut(["Baseline","Projection","Actual"], [total_bl,total_pr,total_ac], "PROJECTION")
p1.plotly_chart(proj_fig, use_container_width=True)

a_cnt = int(filt["sale_status"].fillna("").str.lower().str.contains("achiev").sum())
n_cnt = int(filt["sale_status"].fillna("").str.lower().str.contains("not").sum())
c_cnt = max(len(filt) - a_cnt - n_cnt, 0)
perf_df = pd.DataFrame({"Status":["Achieved","Not Achieved","New Company"], "Count":[a_cnt,n_cnt,c_cnt]})
perf_fig = px.bar(perf_df, x="Status", y="Count", text="Count",
                  color="Status",
                  color_discrete_map={"Achieved":ACCENT_BLUE,"Not Achieved":ACCENT_RED,"New Company":ACCENT_PURPLE})
perf_fig.update_traces(
    textposition="outside",
    textfont=dict(color='#000000' if theme_choice == 'Light' else TEXT)
)
perf_fig.update_layout(
    template=PLOTLY_TPL, height=320, title="OVERALL PERFORMANCE",
    plot_bgcolor=PANEL_BG, paper_bgcolor=PANEL_BG,
    font=dict(color='#000000' if theme_choice == 'Light' else TEXT),
    margin=dict(l=10,r=10,t=40,b=10)
)
perf_fig.update_xaxes(
    tickfont=dict(color='#000000' if theme_choice == 'Light' else TEXT)
)
perf_fig.update_yaxes(
    gridcolor=GRID_COLOR,
    tickfont=dict(color='#000000' if theme_choice == 'Light' else TEXT),
    title_font=dict(color='#000000' if theme_choice == 'Light' else TEXT)
)
p2.plotly_chart(perf_fig, use_container_width=True)

p3.plotly_chart(stacked_perf(filt, "state", "OVERALL PERFORMANCE BY STATE"), use_container_width=True)

q1, q2 = st.columns(2)
q1.plotly_chart(stacked_perf(filt, "industry", "OVERALL PERFORMANCE BY INDUSTRY"), use_container_width=True)
q2.plotly_chart(stacked_perf(filt, "group", "OVERALL PERFORMANCE BY GROUP"), use_container_width=True)

st.markdown("<br/>", unsafe_allow_html=True)

r1, r2, r3 = st.columns(3)
jb = float(filt["jobs_baseline"].apply(num).sum())
jp = float(filt["jobs_projection"].apply(num).sum())
ja = float(filt["jobs_actual_growth"].apply(num).sum())
jobs_fig = donut(["Base Jobs","Proj Jobs","Actual New Jobs"], [jb,jp,ja], "PROJECTION (JOBS)")
r1.plotly_chart(jobs_fig, use_container_width=True)

jobs_perf = pd.DataFrame({
    "Status": ["Achieved","Not Achieved"],
    "Count": [int((filt["jobs_actual_growth"].apply(num)>0).sum()),
              int(len(filt) - (filt["jobs_actual_growth"].apply(num)>0).sum())]
})
jp_fig = px.bar(jobs_perf, x="Status", y="Count", text="Count",
                color="Status",
                color_discrete_map={"Achieved":ACCENT_GREEN,"Not Achieved":ACCENT_RED})
jp_fig.update_traces(
    textposition="outside",
    textfont=dict(color='#000000' if theme_choice == 'Light' else TEXT)
)
jp_fig.update_layout(
    template=PLOTLY_TPL, height=320, title="OVERALL PERFORMANCE (JOBS)",
    plot_bgcolor=PANEL_BG, paper_bgcolor=PANEL_BG,
    margin=dict(l=10,r=10,t=40,b=10),
    font=dict(color='#000000' if theme_choice == 'Light' else TEXT)
)
jp_fig.update_xaxes(
    tickfont=dict(color='#000000' if theme_choice == 'Light' else TEXT)
)
jp_fig.update_yaxes(
    gridcolor=GRID_COLOR,
    tickfont=dict(color='#000000' if theme_choice == 'Light' else TEXT),
    title_font=dict(color='#000000' if theme_choice == 'Light' else TEXT)
)
r2.plotly_chart(jp_fig, use_container_width=True)
r3.plotly_chart(stacked_perf(filt, "state", "OVERALL PERFORMANCE BY STATE (JOBS)"), use_container_width=True)

st.markdown("<br/>", unsafe_allow_html=True)

tA, tB, tC = st.columns(3)
top5_sales = filt.sort_values("rev_actual", key=lambda s: s.apply(num), ascending=False).head(5).copy()
top5_sales["ACTUAL RM"] = top5_sales["rev_actual"].apply(num).map(lambda v: f"RM {v:,.0f}")
top5_sales = top5_sales.rename(columns={"company":"COMPANY","state":"STATE"})
with tA:
    st.markdown("**TOP 5 COMPANIES (SALES)**")
    st.dataframe(top5_sales[["COMPANY","STATE","ACTUAL RM"]], hide_index=True, use_container_width=True)

gdf = filt.copy(); gdf["GROWTH %"] = gdf["growth_pct"].round(1)
top5_gpct = gdf.sort_values("GROWTH %", ascending=False).head(5).rename(columns={"company":"COMPANY","state":"STATE"})
with tB:
    st.markdown("**TOP 5 GROWTH SALES COMPANIES**")
    st.dataframe(top5_gpct[["COMPANY","STATE","GROWTH %"]], hide_index=True, use_container_width=True)

jdf = filt.copy(); jdf["NEW JOBS"] = jdf["jobs_actual_growth"].apply(num).astype(int)
top5_jobs = jdf.sort_values("NEW JOBS", ascending=False).head(5).rename(columns={"company":"COMPANY","state":"STATE"})
with tC:
    st.markdown("**TOP 5 GROWTH JOBS**")
    st.dataframe(top5_jobs[["COMPANY","STATE","NEW JOBS"]], hide_index=True, use_container_width=True)

# --------------------------------------------------------------------------------------
# Bottom Highlights (Top Ranked / Highest …) — horizontal row
# --------------------------------------------------------------------------------------
def compute_highlights(df: pd.DataFrame):
    if df.empty:
        return dict(
            top_ranked=[],
            high_growth_sale=("—", 0.0),
            high_sales=("—", 0.0),
            high_job_growth=("—", 0.0),
            high_new_jobs=("—", 0),
        )

    tmp_growth = df.assign(g=(df["rev_actual"].apply(num) - df["rev_baseline"].apply(num)))
    top_ranked = tmp_growth.sort_values("g", ascending=False)["company"].dropna().astype(str).head(5).tolist()

    pct_df = df.copy()
    pct_df["pct"] = np.where(
        pct_df["rev_baseline"].apply(num) > 0,
        (pct_df["rev_actual"].apply(num) - pct_df["rev_baseline"].apply(num)) / pct_df["rev_baseline"].apply(num) * 100,
        -np.inf,
    )
    high_growth_row = pct_df.sort_values("pct", ascending=False).head(1)
    high_growth_sale = (str(high_growth_row.iloc[0]["company"]), float(high_growth_row.iloc[0]["pct"])) if not high_growth_row.empty else ("—", 0.0)

    hs = df.sort_values("rev_actual", key=lambda s: s.apply(num), ascending=False).head(1)
    high_sales = (str(hs.iloc[0]["company"]), float(num(hs.iloc[0]["rev_actual"]))) if not hs.empty else ("—", 0.0)

    jg = df.copy()
    jg["job_pct"] = np.where(
        jg["jobs_baseline"].apply(num) > 0,
        jg["jobs_actual_growth"].apply(num) / np.maximum(jg["jobs_baseline"].apply(num), 1) * 100,
        -np.inf,
    )
    jg_row = jg.sort_values("job_pct", ascending=False).head(1)
    high_job_growth = (str(jg_row.iloc[0]["company"]), float(jg_row.iloc[0]["job_pct"])) if not jg_row.empty else ("—", 0.0)

    nj = df.sort_values("jobs_actual_growth", key=lambda s: s.apply(num), ascending=False).head(1)
    high_new_jobs = (str(nj.iloc[0]["company"]), int(num(nj.iloc[0]["jobs_actual_growth"]))) if not nj.empty else ("—", 0)

    return dict(
        top_ranked=top_ranked[:5],
        high_growth_sale=high_growth_sale,
        high_sales=high_sales,
        high_job_growth=high_job_growth,
        high_new_jobs=high_new_jobs,
    )

H = compute_highlights(filt)
st.markdown("<br/>", unsafe_allow_html=True)
st.markdown("### 🔝 Highlights")

cols = st.columns(5)

def card_bg(idx: int):
    if theme_choice == "Light":
        return COLORFUL_GRADS[idx % len(COLORFUL_GRADS)]
    return PANEL_BG

with cols[0]:
    bg = card_bg(0)
    st.markdown(
        f"""
        <div class="highlight-card" style="background:{bg}">
            <div class="highlight-title">TOP RANKED COMPANIES</div>
            <div style="height:8px"></div>
            <ul class="bullets">
                {"".join([f"<li><b>{name}</b></li>" for name in H["top_ranked"]])}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with cols[1]:
    bg = card_bg(1)
    comp, pct = H["high_growth_sale"]
    st.markdown(
        f"""
        <div class="highlight-card" style="background:{bg}">
            <div class="highlight-title">HIGHEST GROWTH SALE</div>
            <div style="height:6px"></div>
            <div><b>{comp}</b></div>
            <div class="highlight-big">{pct:.0f}%</div>
            <div class="highlight-sub">vs baseline</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with cols[2]:
    bg = card_bg(2)
    comp, amt = H["high_sales"]
    st.markdown(
        f"""
        <div class="highlight-card" style="background:{bg}">
            <div class="highlight-title">HIGHEST SALES</div>
            <div style="height:6px"></div>
            <div><b>{comp}</b></div>
            <div class="highlight-big">RM {amt:,.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with cols[3]:
    bg = card_bg(3)
    comp, pct = H["high_job_growth"]
    st.markdown(
        f"""
        <div class="highlight-card" style="background:{bg}">
            <div class="highlight-title">HIGHEST GROWTH JOB CREATION</div>
            <div style="height:6px"></div>
            <div><b>{comp}</b></div>
            <div class="highlight-big">{pct:.0f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with cols[4]:
    bg = card_bg(4)
    comp, jobs = H["high_new_jobs"]
    st.markdown(
        f"""
        <div class="highlight-card" style="background:{bg}">
            <div class="highlight-title">HIGHEST NEW JOB CREATION</div>
            <div style="height:6px"></div>
            <div><b>{comp}</b></div>
            <div class="highlight-big">{jobs:,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------------------
# Data preview & exports
# --------------------------------------------------------------------------------------
st.markdown("<br/>", unsafe_allow_html=True)
st.markdown("#### 📋 Data Preview")
show_cols = ["company","state","industry","group","rev_baseline","rev_projection","rev_actual",
             "jobs_baseline","jobs_projection","jobs_actual_growth","growth_pct"]
df_show = filt[show_cols].rename(columns={
    "company":"COMPANY","state":"STATE","industry":"INDUSTRY","group":"GROUP",
    "rev_baseline":"BASELINE RM","rev_projection":"PROJECTION RM","rev_actual":"ACTUAL RM",
    "jobs_baseline":"BASE JOBS","jobs_projection":"PROJ JOBS","jobs_actual_growth":"ACTUAL NEW JOBS",
    "growth_pct":"GROWTH %"
})
st.dataframe(df_show, hide_index=True, use_container_width=True)

cdown1, cdown2 = st.columns(2)
csv_bytes = df_show.to_csv(index=False).encode("utf-8")
cdown1.download_button("Export (CSV)", data=csv_bytes, file_name="nest_dashboard_filtered.csv",
                       mime="text/csv", use_container_width=True)

bio = io.BytesIO()
with pd.ExcelWriter(bio, engine="openpyxl") as xw:
    df_show.to_excel(xw, index=False, sheet_name="Filtered")
cdown2.download_button("Export (Excel)", data=bio.getvalue(),
                       file_name="nest_dashboard_filtered.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True)
