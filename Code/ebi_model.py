"""
Energy Burden Index (EBI) Model
Problem Statement 1 — Theme 3: Community Energy, Equity and Sustainability
Hackathon 2026

Calculates EBI scores for Toronto neighbourhoods using:
  - Median household income
  - Renter percentage
  - Building age
  - Estimated energy spend

Then runs K-Means clustering and exports a Folium choropleth map.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import folium
from folium.plugins import MarkerCluster
import json
import os

# ── 1. SIMULATED DATASET ──────────────────────────────────────────────────────
# Replace with real Toronto Open Data / Statistics Canada data when available
# Columns match what you'd get merging:
#   - Toronto Neighbourhood Profiles (open.toronto.ca)
#   - Statistics Canada Income Data
#   - CMHC Rental Market Data

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
        28000, 31000, 33000, 34000, 30000, 35000, 29000,
        38000, 47000, 52000, 56000, 61000, 95000, 82000,
        110000, 88000, 42000, 26000, 32000, 37000
    ],
    "renter_pct": [
        82, 74, 68, 71, 78, 66, 81,
        58, 55, 49, 44, 67, 28, 40,
        22, 38, 72, 88, 76, 61
    ],
    "avg_building_year": [
        1968, 1971, 1978, 1965, 1972, 1974, 1970,
        1980, 1985, 1988, 1990, 1955, 1952, 1960,
        1958, 1975, 1962, 1960, 1969, 1982
    ],
    "avg_energy_spend_monthly": [
        210, 195, 188, 202, 198, 185, 205,
        175, 162, 155, 148, 170, 130, 142,
        125, 145, 190, 220, 200, 178
    ],
    # Lat/lon for Folium markers (approximate neighbourhood centroids)
    "lat": [
        43.6604, 43.7376, 43.7765, 43.7002,
        43.7127, 43.7234, 43.7089,
        43.7523, 43.7615, 43.6435,
        43.7754, 43.6677, 43.6995, 43.6532,
        43.7070, 43.6987, 43.6396, 43.6574,
        43.7248, 43.7894
    ],
    "lon": [
        -79.3598, -79.5072, -79.2318, -79.5148,
        -79.3388, -79.5651, -79.3367,
        -79.2341, -79.4130, -79.5626,
        -79.3398, -79.3988, -79.4124, -79.4638,
        -79.3630, -79.3876, -79.4341, -79.3656,
        -79.4588, -79.2710
    ]
}

df = pd.DataFrame(data)

# ── 2. FEATURE ENGINEERING ────────────────────────────────────────────────────

# Building age (years old from 2026)
df["building_age"] = 2026 - df["avg_building_year"]

# Energy burden ratio = monthly energy spend / monthly income
df["monthly_income"] = df["median_income"] / 12
df["energy_burden_ratio"] = df["avg_energy_spend_monthly"] / df["monthly_income"]

# ── 3. EBI SCORING ────────────────────────────────────────────────────────────
# Weighted index (all components normalized 0–1, higher = worse burden)
# Weights chosen based on literature on energy poverty drivers:
#   income burden   40% — most direct measure of affordability stress
#   renter share    25% — proxy for incentive misalignment
#   building age    20% — older buildings = less efficient
#   energy spend    15% — absolute cost pressure

scaler = MinMaxScaler()

features = ["energy_burden_ratio", "renter_pct", "building_age", "avg_energy_spend_monthly"]
df_norm = pd.DataFrame(
    scaler.fit_transform(df[features]),
    columns=[f + "_norm" for f in features]
)

WEIGHTS = {
    "energy_burden_ratio_norm": 0.40,
    "renter_pct_norm":          0.25,
    "building_age_norm":        0.20,
    "avg_energy_spend_monthly_norm": 0.15
}

df["ebi_score"] = sum(df_norm[col] * w for col, w in WEIGHTS.items()) * 100
df["ebi_score"] = df["ebi_score"].round(1)

# Risk tier
def risk_tier(score):
    if score >= 75: return "Critical"
    if score >= 55: return "High"
    if score >= 35: return "Moderate"
    return "Low"

df["risk_tier"] = df["ebi_score"].apply(risk_tier)

# Incentive gap flag (high renter + high burden = programs likely miss them)
df["incentive_gap"] = (df["renter_pct"] > 65) & (df["ebi_score"] > 55)

# ── 4. K-MEANS CLUSTERING ────────────────────────────────────────────────────
# Groups neighbourhoods into 4 clusters for targeted program design

cluster_features = df_norm[list(WEIGHTS.keys())].values
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(cluster_features)

cluster_labels = {
    df.groupby("cluster")["ebi_score"].mean().idxmax(): "Cluster A — Highest burden",
}
# Auto-label remaining clusters by mean EBI rank
ranked = df.groupby("cluster")["ebi_score"].mean().sort_values(ascending=False)
labels = ["Cluster A — Highest burden", "Cluster B — High burden",
          "Cluster C — Moderate burden", "Cluster D — Lower burden"]
cluster_map = {int(idx): label for idx, label in zip(ranked.index, labels)}
df["cluster_label"] = df["cluster"].map(cluster_map)

# ── 5. FOLIUM CHOROPLETH MAP ─────────────────────────────────────────────────

def ebi_color(score):
    if score >= 75: return "#E24B4A"
    if score >= 55: return "#EF9F27"
    if score >= 35: return "#97C459"
    return "#5DCAA5"

m = folium.Map(location=[43.7184, -79.3776], zoom_start=11, tiles="CartoDB positron")

# Add neighbourhood markers
for _, row in df.iterrows():
    color = ebi_color(row["ebi_score"])
    gap_icon = " ⚠" if row["incentive_gap"] else ""

    popup_html = f"""
    <div style="font-family:sans-serif;min-width:220px;font-size:13px;">
      <b style="font-size:15px">{row['neighbourhood']}</b><br>
      <hr style="margin:4px 0">
      <b>EBI Score:</b> {row['ebi_score']}/100<br>
      <b>Risk Tier:</b> <span style="color:{color}">{row['risk_tier']}</span><br>
      <b>Renter Share:</b> {row['renter_pct']}%<br>
      <b>Median Income:</b> ${row['median_income']:,}<br>
      <b>Avg Building Age:</b> {row['avg_building_year']}s<br>
      <b>Monthly Energy Spend:</b> ${row['avg_energy_spend_monthly']}<br>
      <b>Cluster:</b> {row['cluster_label']}<br>
      {'<b style="color:#A32D2D">⚠ Incentive gap — renter programs needed</b>' if row['incentive_gap'] else ''}
    </div>
    """

    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=8 + (row["ebi_score"] / 100) * 14,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.75,
        popup=folium.Popup(popup_html, max_width=260),
        tooltip=f"{row['neighbourhood']} — EBI {row['ebi_score']}"
    ).add_to(m)

# Legend
legend_html = """
<div style="position:fixed;bottom:30px;left:30px;z-index:1000;
     background:white;padding:12px 16px;border-radius:8px;
     border:1px solid #ddd;font-family:sans-serif;font-size:13px;">
  <b>Energy Burden Index</b><br><br>
  <span style="background:#E24B4A;display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:6px"></span>Critical (≥75)<br>
  <span style="background:#EF9F27;display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:6px"></span>High (55–74)<br>
  <span style="background:#97C459;display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:6px"></span>Moderate (35–54)<br>
  <span style="background:#5DCAA5;display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:6px"></span>Low (&lt;35)<br><br>
  <span style="color:#A32D2D">⚠</span> = Incentive gap detected
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# Save map
os.makedirs("outputs", exist_ok=True)
m.save("outputs/ebi_map.html")
print("Map saved to outputs/ebi_map.html")

# ── 6. EXPORT DATA ────────────────────────────────────────────────────────────
df.to_csv("outputs/ebi_scores.csv", index=False)
print("Data exported to outputs/ebi_scores.csv")

# ── 7. SUMMARY PRINT ─────────────────────────────────────────────────────────
print("\n── Energy Burden Index Summary ──")
print(df[["neighbourhood", "ebi_score", "risk_tier", "cluster_label", "incentive_gap"]]
      .sort_values("ebi_score", ascending=False)
      .to_string(index=False))

print("\n── Cluster Centres (mean EBI by cluster) ──")
print(df.groupby("cluster_label")["ebi_score"].mean().sort_values(ascending=False).round(1))
