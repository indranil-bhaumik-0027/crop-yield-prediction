import streamlit as st
import pandas as pd
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌾",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("🌾 Crop Yield Prediction System")
st.markdown("Predict estimated crop yield using agricultural factors.")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("crop_yield.csv")
    return df

try:
    df = load_data()

except Exception as e:
    st.error(f"Dataset Error: {e}")
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.header("📥 Enter Agricultural Data")

# Dropdown options
crop_list = sorted(df["Crop"].unique())
state_list = sorted(df["State"].unique())
season_list = sorted(df["Season"].unique())

# ---------------- INPUT SECTION ----------------
col1, col2 = st.columns(2)

with col1:
    crop = st.selectbox("🌱 Select Crop", crop_list)

    state = st.selectbox("📍 Select State", state_list)

    season = st.selectbox("☀️ Select Season", season_list)

    area = st.number_input(
        "🌾 Area (Hectares)",
        min_value=1.0,
        value=1000.0
    )

with col2:
    production = st.number_input(
        "🏭 Production (Tonnes)",
        min_value=0.0,
        value=5000.0
    )

    rainfall = st.number_input(
        "🌧️ Annual Rainfall (mm)",
        min_value=0.0,
        value=1000.0
    )

    fertilizer = st.number_input(
        "🧪 Fertilizer Usage (kg)",
        min_value=0.0,
        value=200000.0
    )

    pesticide = st.number_input(
        "☠️ Pesticide Usage (kg)",
        min_value=0.0,
        value=500.0
    )

# ---------------- SIMPLE PREDICTION ----------------
if st.button("🔍 Predict Crop Yield"):

    try:
        # Simple custom prediction formula
        estimated_yield = (
            (production / area)
            + (rainfall * 0.002)
            + (fertilizer * 0.00001)
            - (pesticide * 0.0005)
        )

        estimated_yield = max(estimated_yield, 0)

        st.markdown("---")

        st.success(
            f"🌟 Estimated Crop Yield: {estimated_yield:.2f} Quintal/Hectare"
        )

        # Performance Category
        if estimated_yield > 80:
            st.info("Excellent Yield Prediction 🌟")

        elif estimated_yield > 50:
            st.info("Good Yield Prediction ✅")

        else:
            st.warning("Low Yield Prediction ⚠️")

        # Display Summary
        st.subheader("📊 Input Summary")

        summary = pd.DataFrame({
            "Parameter": [
                "Crop",
                "State",
                "Season",
                "Area",
                "Production",
                "Rainfall",
                "Fertilizer",
                "Pesticide"
            ],
            "Value": [
                crop,
                state,
                season,
                area,
                production,
                rainfall,
                fertilizer,
                pesticide
            ]
        })

        st.dataframe(summary, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction Error: {e}")
