# ⚡ Energy Burden Index — Toronto

**Hackathon 2026 · Theme 3: Community Energy, Equity and Sustainability**  
**Problem Statement 1: Energy Burden & Renter Incentive Gap**

---

## Problem

Low-income renters in Toronto pay a disproportionate share of their income on energy — yet most utility incentive programs target property owners. Because renters pay the bills but don't control building upgrades, they are systematically excluded from energy efficiency programs.

## Our Solution

We built an **Energy Burden Index (EBI)** that combines four key factors to score every Toronto neighbourhood:

| Factor | Weight | Why |
|--------|--------|-----|
| Energy burden ratio (spend / income) | 40% | Most direct measure of affordability stress |
| Renter share | 25% | Proxy for incentive misalignment |
| Building age | 20% | Older buildings = less efficient |
| Absolute energy spend | 15% | Cost pressure indicator |

We then use **K-Means clustering** to group neighbourhoods into four tiers for targeted program design, and flag **incentive gaps** where high-renter, high-burden communities are being missed by owner-only programs.

---

## Features

- **EBI scoring model** — weighted index normalized 0–100
- **K-Means clustering** — 4 clusters for targeted outreach
- **Folium choropleth map** — interactive HTML map of all neighbourhoods
- **Streamlit dashboard** — live filters, bar charts, scatter plots, full data table
- **Incentive gap detection** — flags neighbourhoods where renter programs are needed

---

## Project Structure

```
energy_burden/
├── ebi_model.py        # EBI scoring + K-Means + Folium map
├── app.py              # Streamlit dashboard
├── requirements.txt    # Python dependencies
├── outputs/
│   ├── ebi_scores.csv  # Generated scores (run ebi_model.py first)
│   └── ebi_map.html    # Interactive Folium map
└── README.md
```

---

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/energy-burden-index.git
cd energy-burden-index

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the EBI model (generates outputs/)
python ebi_model.py

# 4. Launch the Streamlit dashboard
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## Data Sources

- [Toronto Neighbourhood Profiles](https://open.toronto.ca/)
- [Statistics Canada Income Data](https://www150.statcan.gc.ca/n1/en/type/data)
- [Peel Region Demographics](https://opendata.peelregion.ca/)
- [Canadian Index of Social Vulnerability](https://www150.statcan.gc.ca/n1/en/catalogue/45200001)

> **Note:** Current dataset is simulated for demo purposes, structured to match the above sources. Replace `data` dict in `ebi_model.py` with real CSV imports when available.

---

## Team

- **Abishek** — Data pipeline, EBI model, K-Means clustering, Folium map, GitHub
- **Abishek 2** — Streamlit dashboard, slide deck, video presentation

---

## License

MIT
