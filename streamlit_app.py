# ======================= UPGRADE PACK ===========================
st.markdown("<br/>", unsafe_allow_html=True)

# ---------- helpers ----------
def status_count(series: pd.Series):
    s = series.fillna("").str.lower()
    ach = int(s.str.contains("achiev").sum())
    notach = int(s.str.contains("not").sum())
    newc = max(len(series) - ach - notach, 0)
    return ach, notach, newc

def stacked_perf(df: pd.DataFrame, by: str, title: str):
    ach, notach, newc = "Achieved", "Not Achieved", "New Company"
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
    fig.add_trace(go.Bar(name=ach, x=tmp[by], y=tmp["Achieved"], marker_color=ACCENT_BLUE))
    fig.add_trace(go.Bar(name=notach, x=tmp[by], y=tmp["NotAchieved"], marker_color=ACCENT_RED))
    fig.add_trace(go.Bar(name=newc, x=tmp[by], y=tmp["NewCompany"], marker_color=ACCENT_PURPLE))
    fig.update_layout(
        barmode="stack", height=350, title=title, plot_bgcolor=PANEL_BG, paper_bgcolor=PANEL_BG,
        margin=dict(l=10,r=10,t=40,b=10), font=dict(color=TEXT), legend=dict(orientation="h", y=1.1)
    )
    fig.update_yaxes(gridcolor="rgba(255,255,255,.08)")
    return fig

def top5_table(df: pd.DataFrame, cols, title: str):
    st.markdown(f"**{title}**")
    st.dataframe(df[cols].reset_index(drop=True), hide_index=True, use_container_width=True)

# ---------- projection pie ----------
p1, p2, p3 = st.columns([1.0, 1.0, 1.0])

# “Projection” pie = share of Baseline vs Projection vs Actual for current filter
total_bl = float(filt["rev_baseline"].apply(num).sum())
total_pr = float(filt["rev_projection"].apply(num).sum())
total_ac = float(filt["rev_actual"].apply(num).sum())

proj_fig = go.Figure(
    go.Pie(
        labels=["Baseline", "Projection", "Actual"],
        values=[total_bl, total_pr, total_ac],
        hole=0.55,
        textinfo="label+percent",
    )
)
proj_fig.update_traces(marker=dict(colors=[ACCENT_BLUE, ACCENT_PURPLE, ACCENT_RED],
                                   line=dict(color=PANEL_BG, width=2)))
proj_fig.update_layout(
    title="PROJECTION", height=320, plot_bgcolor=PANEL_BG, paper_bgcolor=PANEL_BG,
    margin=dict(l=10,r=10,t=40,b=10), font=dict(color=TEXT)
)
p1.plotly_chart(proj_fig, use_container_width=True)

# ---------- overall performance small bars (global) ----------
# counts of Achieved / Not / New for current filter
a, n, cnew = status_count(filt["sale_status"])
perf_df = pd.DataFrame({"Status": ["Achieved","Not Achieved","New Company"], "Count":[a,n,cnew]})
perf_fig = px.bar(perf_df, x="Status", y="Count", text="Count",
                  color="Status",
                  color_discrete_map={"Achieved":ACCENT_BLUE,"Not Achieved":ACCENT_RED,"New Company":ACCENT_PURPLE})
perf_fig.update_traces(textposition="outside")
perf_fig.update_layout(height=320, title="OVERALL PERFORMANCE",
                       plot_bgcolor=PANEL_BG, paper_bgcolor=PANEL_BG,
                       font=dict(color=TEXT), margin=dict(l=10,r=10,t=40,b=10))
p2.plotly_chart(perf_fig, use_container_width=True)

# ---------- state stacked performance ----------
p3.plotly_chart(stacked_perf(filt, "state", "OVERALL PERFORMANCE BY STATE"), use_container_width=True)

# ---------- row: stacked by INDUSTRY and GROUP ----------
q1, q2 = st.columns(2)
q1.plotly_chart(stacked_perf(filt, "industry", "OVERALL PERFORMANCE BY INDUSTRY"), use_container_width=True)
q2.plotly_chart(stacked_perf(filt, "group", "OVERALL PERFORMANCE BY GROUP"), use_container_width=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ---------- second projection row (jobs) ----------
r1, r2, r3 = st.columns(3)

# jobs projection pie = baseline vs projection vs actual new jobs
jb = float(filt["jobs_baseline"].apply(num).sum())
jp = float(filt["jobs_projection"].apply(num).sum())
ja = float(filt["jobs_actual_growth"].apply(num).sum())
jobs_fig = go.Figure(go.Pie(labels=["Base Jobs", "Proj Jobs", "Actual New Jobs"],
                            values=[jb, jp, ja], hole=0.55, textinfo="label+percent"))
jobs_fig.update_traces(marker=dict(colors=[ACCENT_BLUE, ACCENT_PURPLE, ACCENT_GREEN],
                                   line=dict(color=PANEL_BG, width=2)))
jobs_fig.update_layout(title="PROJECTION (JOBS)", height=320,
                       plot_bgcolor=PANEL_BG, paper_bgcolor=PANEL_BG,
                       margin=dict(l=10,r=10,t=40,b=10), font=dict(color=TEXT))
r1.plotly_chart(jobs_fig, use_container_width=True)

# performance mini bars (again, jobs achieved = actual>0)
jobs_perf = pd.DataFrame({
    "Status": ["Achieved", "Not Achieved"],
    "Count": [int((filt["jobs_actual_growth"].apply(num)>0).sum()),
              int(len(filt) - (filt["jobs_actual_growth"].apply(num)>0).sum())]
})
jp_fig = px.bar(jobs_perf, x="Status", y="Count", text="Count",
                color="Status",
                color_discrete_map={"Achieved":ACCENT_GREEN,"Not Achieved":ACCENT_RED})
jp_fig.update_traces(textposition="outside")
jp_fig.update_layout(height=320, title="OVERALL PERFORMANCE (JOBS)",
                     plot_bgcolor=PANEL_BG, paper_bgcolor=PANEL_BG,
                     margin=dict(l=10,r=10,t=40,b=10), font=dict(color=TEXT))
r2.plotly_chart(jp_fig, use_container_width=True)

# state stacked performance for JOBS (uses same sale_status buckets as proxy)
r3.plotly_chart(stacked_perf(filt, "state", "OVERALL PERFORMANCE BY STATE (JOBS)"),
                use_container_width=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ---------- RIGHT RAIL style "Top 5" (rendered full width in 3 columns) ----------
tA, tB, tC = st.columns(3)

# Top 5 Companies by Actual Sales
top5_sales = filt.sort_values("rev_actual", key=lambda s: s.apply(num), ascending=False).head(5).copy()
top5_sales["ACTUAL RM"] = top5_sales["rev_actual"].apply(num).map(lambda v: f"RM {v:,.0f}")
top5_sales = top5_sales.rename(columns={"company":"COMPANY","state":"STATE"})
with tA:
    top5_table(top5_sales[["COMPANY","STATE","ACTUAL RM"]], ["COMPANY","STATE","ACTUAL RM"], "TOP 5 COMPANIES (SALES)")

# Top 5 Growth Sales %
gdf = filt.copy()
gdf["GROWTH %"] = gdf["growth_pct"].round(1)
top5_gpct = gdf.sort_values("GROWTH %", ascending=False).head(5).rename(columns={"company":"COMPANY","state":"STATE"})
with tB:
    top5_table(top5_gpct[["COMPANY","STATE","GROWTH %"]], ["COMPANY","STATE","GROWTH %"], "TOP 5 GROWTH SALES COMPANIES")

# Top 5 Growth Jobs (absolute new jobs)
jdf = filt.copy()
jdf["NEW JOBS"] = jdf["jobs_actual_growth"].apply(num).astype(int)
top5_jobs = jdf.sort_values("NEW JOBS", ascending=False).head(5).rename(columns={"company":"COMPANY","state":"STATE"})
with tC:
    top5_table(top5_jobs[["COMPANY","STATE","NEW JOBS"]], ["COMPANY","STATE","NEW JOBS"], "TOP 5 GROWTH JOBS")
# ======================= /UPGRADE PACK ==========================
