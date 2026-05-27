"""
Streamlit Dashboard — Energy Burden Index
Problem Statement 1 — Theme 3: Community Energy, Equity and Sustainability
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import os

st.set_page_config(
    page_title="Energy Burden Index · Toronto",
    page_icon="⚡",
    layout="wide"
)

# ── DATA (same as ebi_model.py — run that first or inline here) ───────────────

@st.cache_data
def load_data():
    csv_path = "outputs/ebi_scores.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    # Fallback: rebuild inline if CSV not found
    data = {
        "neighbourhood": [
            "Regent Park", "Jane & Finch", "Malvern", "Weston",
            "Flemingdon Park", "Rexdale", "Thorncliffe Park",
            "Scarborough Village", "North York Centre", "Etobicoke Centre",
            "Don Valley Village", "Annex", "Forest Hill", "High Park",
            "Leaside", "Mount Pleasant", "Parkdale", "Moss Park",
            "Lawrence Heights", "Agincourt"
        ],
        "median_income": [
            28000,31000,33000,34000,30000,35000,29000,
            38000,47000,52000,56000,61000,95000,82000,
            110000,88000,42000,26000,32000,37000
        ],
        "renter_pct": [
            82,74,68,71,78,66,81,58,55,49,44,67,28,40,22,38,72,88,76,61
        ],
        "avg_building_year": [
            1968,1971,1978,1965,1972,1974,1970,1980,1985,1988,
            1990,1955,1952,1960,1958,1975,1962,1960,1969,1982
        ],
        "avg_energy_spend_monthly": [
            210,195,188,202,198,185,205,175,162,155,
            148,170,130,142,125,145,190,220,200,178
        ],
        "lat": [
            43.6604,43.7376,43.7765,43.7002,43.7127,43.7234,43.7089,
            43.7523,43.7615,43.6435,43.7754,43.6677,43.6995,43.6532,
            43.7070,43.6987,43.6396,43.6574,43.7248,43.7894
        ],
        "lon": [
            -79.3598,-79.5072,-79.2318,-79.5148,-79.3388,-79.5651,-79.3367,
            -79.2341,-79.4130,-79.5626,-79.3398,-79.3988,-79.4124,-79.4638,
            -79.3630,-79.3876,-79.4341,-79.3656,-79.4588,-79.2710
        ]
    }
    df = pd.DataFrame(data)
    df["building_age"] = 2026 - df["avg_building_year"]
    df["monthly_income"] = df["median_income"] / 12
    df["energy_burden_ratio"] = df["avg_energy_spend_monthly"] / df["monthly_income"]

    scaler = MinMaxScaler()
    features = ["energy_burden_ratio","renter_pct","building_age","avg_energy_spend_monthly"]
    df_norm = pd.DataFrame(scaler.fit_transform(df[features]), columns=[f+"_norm" for f in features])
    WEIGHTS = {"energy_burden_ratio_norm":0.40,"renter_pct_norm":0.25,
               "building_age_norm":0.20,"avg_energy_spend_monthly_norm":0.15}
    df["ebi_score"] = sum(df_norm[col]*w for col,w in WEIGHTS.items()) * 100
    df["ebi_score"] = df["ebi_score"].round(1)

    def risk_tier(s):
        if s>=75: return "Critical"
        if s>=55: return "High"
        if s>=35: return "Moderate"
        return "Low"
    df["risk_tier"] = df["ebi_score"].apply(risk_tier)
    df["incentive_gap"] = (df["renter_pct"]>65) & (df["ebi_score"]>55)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(df_norm[list(WEIGHTS.keys())].values)
    ranked = df.groupby("cluster")["ebi_score"].mean().sort_values(ascending=False)
    labels = ["Cluster A — Highest burden","Cluster B — High burden",
              "Cluster C — Moderate burden","Cluster D — Lower burden"]
    df["cluster_label"] = df["cluster"].map({int(i):l for i,l in zip(ranked.index,labels)})
    return df

df = load_data()

# ── SIDEBAR FILTERS ───────────────────────────────────────────────────────────

st.sidebar.markdown("## ⚡ Energy Burden Index")
st.sidebar.title("⚡ EBI Dashboard")
st.sidebar.markdown("**Filters**")

tier_filter = st.sidebar.multiselect(
    "Risk tier", ["Critical","High","Moderate","Low"],
    default=["Critical","High","Moderate","Low"]
)
renter_min = st.sidebar.slider("Min renter share (%)", 0, 100, 0)
gap_only = st.sidebar.checkbox("Show incentive gap neighbourhoods only", False)

filtered = df[df["risk_tier"].isin(tier_filter) & (df["renter_pct"] >= renter_min)]
if gap_only:
    filtered = filtered[filtered["incentive_gap"]]

# ── HEADER ────────────────────────────────────────────────────────────────────

st.title("⚡ Toronto Energy Burden Index")
st.markdown("Identifying high-burden, high-renter neighbourhoods where energy incentives fall short.")
st.divider()

# ── METRIC CARDS ──────────────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
c1.metric("Critical zones", len(df[df["risk_tier"]=="Critical"]), help="EBI score ≥ 75")
c2.metric("High renter zones", len(df[df["renter_pct"]>65]), help="> 65% renters")
c3.metric("Incentive gap", f"{len(df[df['incentive_gap']])} hoods", help="High burden + high renter")
c4.metric("Avg EBI score", f"{df['ebi_score'].mean():.1f}/100")

st.divider()

# ── MAP + BAR CHART ───────────────────────────────────────────────────────────

col1, col2 = st.columns([1.4, 1])

with col1:
    st.subheader("Neighbourhood map")
    color_map = {"Critical":"#E24B4A","High":"#EF9F27","Moderate":"#97C459","Low":"#5DCAA5"}
    fig_map = px.scatter_mapbox(
        filtered, lat="lat", lon="lon",
        size="ebi_score", color="risk_tier",
        color_discrete_map=color_map,
        hover_name="neighbourhood",
        hover_data={"ebi_score":True,"renter_pct":True,"median_income":True,
                    "risk_tier":True,"lat":False,"lon":False},
        size_max=28, zoom=10,
        mapbox_style="carto-positron",
        labels={"risk_tier":"Risk tier","ebi_score":"EBI score","renter_pct":"Renter %","median_income":"Median income"}
    )
    fig_map.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=420)
    st.plotly_chart(fig_map, use_container_width=True)

with col2:
    st.subheader("Top 10 highest burden")
    top10 = filtered.sort_values("ebi_score", ascending=False).head(10)
    fig_bar = px.bar(
        top10, x="ebi_score", y="neighbourhood",
        orientation="h", color="risk_tier",
        color_discrete_map=color_map,
        text="ebi_score",
        labels={"ebi_score":"EBI score","neighbourhood":"","risk_tier":"Risk tier"}
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(
        height=420, margin=dict(l=0,r=20,t=10,b=0),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
        xaxis=dict(range=[0,105])
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ── CLUSTER ANALYSIS ──────────────────────────────────────────────────────────

st.subheader("K-Means cluster analysis")
col3, col4 = st.columns(2)

with col3:
    fig_scatter = px.scatter(
        filtered, x="median_income", y="ebi_score",
        color="cluster_label", size="renter_pct",
        hover_name="neighbourhood",
        labels={"median_income":"Median income ($)","ebi_score":"EBI score",
                "cluster_label":"Cluster","renter_pct":"Renter %"},
        title="Income vs EBI score (bubble size = renter %)"
    )
    fig_scatter.update_layout(height=340, margin=dict(t=40,b=0))
    st.plotly_chart(fig_scatter, use_container_width=True)

with col4:
    cluster_summary = df.groupby("cluster_label").agg(
        neighbourhoods=("neighbourhood","count"),
        avg_ebi=("ebi_score","mean"),
        avg_renter=("renter_pct","mean"),
        avg_income=("median_income","mean")
    ).round(1).reset_index()
    cluster_summary.columns = ["Cluster","Neighbourhoods","Avg EBI","Avg Renter %","Avg Income ($)"]
    cluster_summary = cluster_summary.sort_values("Avg EBI", ascending=False)
    st.markdown("**Cluster summary**")
    st.dataframe(cluster_summary, use_container_width=True, hide_index=True)

st.divider()

# ── DATA TABLE ────────────────────────────────────────────────────────────────

st.subheader("Full neighbourhood data")
display_cols = ["neighbourhood","ebi_score","risk_tier","renter_pct",
                "median_income","avg_building_year","avg_energy_spend_monthly",
                "incentive_gap","cluster_label"]
st.dataframe(
    filtered[display_cols].sort_values("ebi_score", ascending=False)
    .rename(columns={
        "neighbourhood":"Neighbourhood","ebi_score":"EBI Score",
        "risk_tier":"Risk Tier","renter_pct":"Renter %",
        "median_income":"Median Income","avg_building_year":"Avg Building Year",
        "avg_energy_spend_monthly":"Monthly Energy Spend ($)",
        "incentive_gap":"Incentive Gap","cluster_label":"Cluster"
    }),
    use_container_width=True, hide_index=True
)

st.caption("Data: Simulated from Toronto Neighbourhood Profiles, Statistics Canada Income Data, CMHC. For demo purposes.")
