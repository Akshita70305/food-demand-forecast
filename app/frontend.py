import streamlit as st
import requests
import pandas as pd
import numpy as np
import datetime

# ── Page config ──
st.set_page_config(
    page_title="Cumin Price Forecaster",
    page_icon="🌿",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000"

# ── Helper functions ──
def get_season_flags(month):
    is_harvest = 1 if month in [2, 3, 4] else 0
    is_lean    = 1 if month in [5, 6, 7, 8] else 0
    return is_harvest, is_lean

def call_predict(payload):
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

def price_level(price):
    if price < 15000:
        return "🟢 Low", "#2E7D32"
    elif price < 25000:
        return "🟡 Moderate", "#F57F17"
    else:
        return "🔴 High", "#C62828"

# ── Header ──
st.title("🌿 Cumin (Jeera) Price Forecaster")
st.markdown("Predict cumin modal price across Indian markets using ML — powered by MLflow + FastAPI")
st.divider()

# ── API Status ──
try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    info   = requests.get(f"{API_URL}/model-info", timeout=3).json()
    st.success(f"✅ API Online  |  Model: **{info['model_name']} v{info['model_version']}**  |  Target: {info['target']}")
except:
    st.error("❌ API is not reachable. Make sure FastAPI is running on port 8000.")
    st.stop()

st.divider()

# ── Sidebar — Input Features ──
st.sidebar.title("📋 Input Features")
st.sidebar.markdown("Adjust values to predict cumin price")

# Date inputs
st.sidebar.subheader("📅 Date")
selected_date = st.sidebar.date_input("Select Date", value=datetime.date.today())
year       = selected_date.year
month      = selected_date.month
day_of_year = selected_date.timetuple().tm_yday
day_of_week = selected_date.weekday()
is_harvest_season, is_lean_season = get_season_flags(month)

st.sidebar.info(
    f"**Month:** {selected_date.strftime('%B')}  \n"
    f"**Day of Year:** {day_of_year}  \n"
    f"**Harvest Season:** {'Yes' if is_harvest_season else 'No'}  \n"
    f"**Lean Season:** {'Yes' if is_lean_season else 'No'}"
)

# Market inputs
st.sidebar.subheader("🏪 Market Activity")
arrivals    = st.sidebar.slider("Arrivals (MT)", min_value=1.0, max_value=500.0, value=45.0, step=0.5)
num_markets = st.sidebar.slider("Active Markets", min_value=1, max_value=50, value=12)

# Lag price inputs
st.sidebar.subheader("💰 Recent Prices (Rs/Quintal)")
price_lag_7  = st.sidebar.number_input("Price 7 days ago",  min_value=1000, max_value=80000, value=19500, step=100)
price_lag_14 = st.sidebar.number_input("Price 14 days ago", min_value=1000, max_value=80000, value=19200, step=100)
price_lag_30 = st.sidebar.number_input("Price 30 days ago", min_value=1000, max_value=80000, value=18800, step=100)

# Lag arrivals
st.sidebar.subheader("📦 Recent Arrivals")
arrivals_lag_7 = st.sidebar.slider("Arrivals 7 days ago (MT)", min_value=1.0, max_value=500.0, value=42.0, step=0.5)

# Rolling averages — auto computed
price_roll_7    = round((price_lag_7 + price_lag_14) / 2, 2)
price_roll_30   = round((price_lag_7 + price_lag_14 + price_lag_30) / 3, 2)
arrivals_roll_7 = round((arrivals + arrivals_lag_7) / 2, 2)

st.sidebar.markdown("---")
st.sidebar.caption(f"7-day avg price (auto): ₹{price_roll_7:,.0f}")
st.sidebar.caption(f"30-day avg price (auto): ₹{price_roll_30:,.0f}")
st.sidebar.caption(f"7-day avg arrivals (auto): {arrivals_roll_7} MT")

# ── Build payload ──
payload = {
    "year": year, "month": month,
    "day_of_year": day_of_year, "day_of_week": day_of_week,
    "is_harvest_season": is_harvest_season,
    "is_lean_season": is_lean_season,
    "arrivals": arrivals, "num_markets": num_markets,
    "price_lag_7": float(price_lag_7),
    "price_lag_14": float(price_lag_14),
    "price_lag_30": float(price_lag_30),
    "arrivals_lag_7": float(arrivals_lag_7),
    "price_roll_7": price_roll_7,
    "price_roll_30": price_roll_30,
    "arrivals_roll_7": arrivals_roll_7
}

# ── Main Panel ──
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("🔮 Prediction")
    if st.button("⚡ Predict Price", type="primary", use_container_width=True):
        with st.spinner("Calling model..."):
            result = call_predict(payload)
        if result:
            price = result["predicted_price"]
            level, color = price_level(price)
            st.markdown(f"""
            <div style='background:{color}22;border-left:6px solid {color};
                        padding:20px;border-radius:8px;margin:10px 0'>
                <h2 style='color:{color};margin:0'>₹{price:,.2f}</h2>
                <p style='margin:4px 0;font-size:16px'>per Quintal &nbsp;|&nbsp; {level}</p>
                <p style='margin:4px 0;color:#666;font-size:13px'>
                    Model: {result['model_name']} v{result['model_version']}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Price context
            st.markdown("**Price Context:**")
            ctx_col1, ctx_col2, ctx_col3 = st.columns(3)
            ctx_col1.metric("vs 7-day ago",  f"₹{price:,.0f}", f"{price - price_lag_7:+,.0f}")
            ctx_col2.metric("vs 14-day ago", f"₹{price:,.0f}", f"{price - price_lag_14:+,.0f}")
            ctx_col3.metric("vs 30-day ago", f"₹{price:,.0f}", f"{price - price_lag_30:+,.0f}")
    else:
        st.info("👈 Adjust inputs in the sidebar and click **Predict Price**")

with col2:
    st.subheader("📊 Input Summary")
    summary_data = {
        "Feature": [
            "Date", "Month", "Harvest Season", "Lean Season",
            "Arrivals (MT)", "Active Markets",
            "Price Lag 7d", "Price Lag 14d", "Price Lag 30d",
            "7-day Avg Price", "30-day Avg Price"
        ],
        "Value": [
            str(selected_date), selected_date.strftime("%B"),
            "Yes" if is_harvest_season else "No",
            "Yes" if is_lean_season else "No",
            f"{arrivals} MT", num_markets,
            f"₹{price_lag_7:,}", f"₹{price_lag_14:,}", f"₹{price_lag_30:,}",
            f"₹{price_roll_7:,.0f}", f"₹{price_roll_30:,.0f}"
        ]
    }
    st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)

st.divider()

# ── Price Trend Chart ──
st.subheader("📈 Price Trend Simulation")
st.caption("Simulated trend based on your lag inputs — adjust lag prices in sidebar to see different scenarios")

months = ["7 days ago", "14 days ago", "30 days ago", "Today (predicted)"]
prices = [price_lag_7, price_lag_14, price_lag_30, 0]

try:
    result_chart = call_predict(payload)
    if result_chart:
        prices[-1] = result_chart["predicted_price"]
except:
    prices[-1] = price_lag_7

chart_df = pd.DataFrame({"Period": months, "Price (Rs/Quintal)": prices})
st.line_chart(chart_df.set_index("Period"), use_container_width=True)

# ── Footer ──
st.divider()
st.caption("Food Demand Forecasting MLOps Pipeline · Akshita · Phase 8 — Streamlit Frontend")