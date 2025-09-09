import streamlit as st
import pandas as pd
import plotly.express as px

# =====================
# Theme Toggle
# =====================
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

is_dark = st.session_state.theme == "dark"

# =====================
# Colors
# =====================
if is_dark:
    PRIMARY_BG = "#0a1633"
    CARD_BG = "#162b55"
    TEXT = "#ffffff"
    CARD_COLORS = [CARD_BG, CARD_BG, CARD_BG, CARD_BG]
else:
    PRIMARY_BG = "#f9fafb"
    CARD_BG = "#ffffff"
    TEXT = "#111827"
    # colourful for white mode
    CARD_COLORS = ["#ffeaa7", "#fab1a0", "#81ecec", "#a29bfe"]

ACCENT = "#f39c12"
ACCENT_GREEN = "#27ae60"
ACCENT_BLUE = "#2980b9"
ACCENT_PURPLE = "#9b59b6"
ACCENT_RED = "#e74c3c"

# =====================
# Page Config
# =====================
st.set_page_config(page_title="Business Analytics Dashboard", layout="wide")

# =====================
# CSS
# =====================
st.markdown(
    f"""
    <style>
    body {{
        background:{PRIMARY_BG};
        color:{TEXT};
    }}
    .dash-card {{
        border-radius:12px;
        padding:18px;
        margin-bottom:18px;
        text-align:center;
    }}
    .metric-label {{
        font-size:13px; opacity:.85; margin-bottom:4px;
    }}
    .metric-value {{
        font-size:28px; font-weight:700;
    }}
    .chip {{
        padding:4px 10px; border-radius:14px; font-size:12px; font-weight:600;
    }}
    .chip-up {{ background: rgba(39,174,96,.15); color:{ACCENT_GREEN}; }}
    .chip-down {{ background: rgba(231,76,60,.15); color:{ACCENT_RED}; }}
    .kpi-baselines .section-label {{
        font-size:12px; letter-spacing:.08em; opacity:.9;
    }}
    .kpi-baselines .row-label {{
        font-size:12px; letter-spacing:.06em; opacity:.85; margin-top:6px;
    }}
    .kpi-baselines .big-num {{
        font-size:32px; font-weight:800; line-height:1.1; margin-top:2px;
        color:{ACCENT};
    }}
    .kpi-baselines .unit {{
        font-size:14px; opacity:.9; margin-left:6px;
        color:{TEXT};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================
# Example Data
# =====================
data = {
    "Company": ["HealthCare Ideal", "Swift Bridge", "Tech Solutions", "BestMart"],
    "Revenue2024": [6240000, 4200000, 3800000, 3600000],
    "Jobs": [130, 90, 70, 59],
    "Growth%": [140, 98, 75, 60]
}
df = pd.DataFrame(data)

# =====================
# KPI Card Functions
# =====================
def kpi_card(container, label, value, chip=None, chip_type="up", bg=None):
    bg_color = bg if bg else CARD_BG
    with container:
        st.markdown(
            f"""
            <div class="dash-card" style="background:{bg_color}">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                {f'<span class="chip chip-{chip_type}">{chip}</span>' if chip else ""}
            </div>
            """,
            unsafe_allow_html=True,
        )

def kpi_baselines(container, sales_amount: float, jobs_count: int, bg=None):
    bg_color = bg if bg else CARD_BG
    with container:
        st.markdown(
            f"""
            <div class="dash-card kpi-baselines" style="background:{bg_color}">
                <div class="section-label">BASELINES</div>

                <div class="row-label">SALES</div>
                <div class="big-num">RM{sales_amount:,.0f}</div>

                <div class="row-label">JOBS</div>
                <div class="big-num">{jobs_count:,}<span class="unit">Person</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =====================
# Header
# =====================
col1, col2 = st.columns([6,1])
with col1:
    st.markdown(f"<h2 style='color:{TEXT}'>📊 Business Analytics Dashboard</h2>", unsafe_allow_html=True)
with col2:
    st.button("🌓 Toggle Theme", on_click=toggle_theme)

# =====================
# KPI Section
# =====================
k1, k2, k3, k4 = st.columns(4)
kpi_card(k1, "TOTAL COMPANIES", "1,247", "+12%", "up", CARD_COLORS[0])
kpi_baselines(k2, 7_500_000, 130, CARD_COLORS[1])
kpi_card(k3, "HIGHEST SALES", "RM6,240,000", "HealthCare Ideal", "up", CARD_COLORS[2])
kpi_card(k4, "HIGHEST GROWTH JOB CREATION", "98%", "HealthCare Ideal", "up", CARD_COLORS[3])

# =====================
# Charts
# =====================
c1, c2 = st.columns(2)
with c1:
    fig = px.bar(df, x="Company", y="Revenue2024", text_auto=".2s", color="Company")
    st.plotly_chart(fig, use_container_width=True)
with c2:
    fig2 = px.pie(df, values="Jobs", names="Company", hole=0.4)
    st.plotly_chart(fig2, use_container_width=True)

# =====================
# Data Table
# =====================
st.markdown("#### 📋 Company Data")
st.dataframe(df, use_container_width=True)

# =====================
# Bottom Highlights
# =====================
st.markdown("#### 🏆 Highlights")
b1, b2, b3, b4 = st.columns(4)
kpi_card(b1, "TOP RANKED COMPANY", "HealthCare Ideal", bg=CARD_COLORS[0])
kpi_card(b2, "HIGHEST GROWTH SALE", "140% vs baseline", bg=CARD_COLORS[1])
kpi_card(b3, "HIGHEST SALES", "RM6,240,000", bg=CARD_COLORS[2])
kpi_card(b4, "HIGHEST NEW JOB CREATION", "59 Jobs", bg=CARD_COLORS[3])
