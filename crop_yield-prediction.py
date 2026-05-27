import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("🌾 Crop Yield Prediction System")
st.markdown("Predict crop yield using agricultural and environmental factors.")

# ---------------- LOAD DATA & TRAIN MODEL ----------------
@st.cache_resource
def load_and_train_model():
    try:
        # Load dataset
        df = pd.read_csv("crop_yield.csv")

        # Remove unwanted column if exists
        df = df.drop(columns=["Crop_Year"], errors="ignore")

        # Label Encoders
        crop_encoder = LabelEncoder()
        state_encoder = LabelEncoder()
        season_encoder = LabelEncoder()

        # Save original names for dropdown
        crop_names = sorted(df["Crop"].unique())
        state_names = sorted(df["State"].unique())
        season_names = sorted(df["Season"].unique())

        # Encode categorical columns
        df["Crop"] = crop_encoder.fit_transform(df["Crop"])
        df["State"] = state_encoder.fit_transform(df["State"])
        df["Season"] = season_encoder.fit_transform(df["Season"])

        # Features and Target
        X = df.drop("Yield", axis=1)
        y = df["Yield"]

        # Train Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # Train Model
        model = KNeighborsRegressor(n_neighbors=5)
        model.fit(X_train, y_train)

        return (
            model,
            crop_encoder,
            state_encoder,
            season_encoder,
            crop_names,
            state_names,
            season_names,
            X.columns
        )

    except Exception as e:
        st.error(f"Error Loading Dataset or Training Model: {e}")
        st.stop()


# Load everything
(
    model,
    crop_encoder,
    state_encoder,
    season_encoder,
    crop_names,
    state_names,
    season_names,
    feature_columns
) = load_and_train_model()

# ---------------- SIDEBAR ----------------
st.sidebar.header("📥 Enter Crop Information")

# ---------------- INPUT SECTION ----------------
col1, col2 = st.columns(2)

with col1:
    selected_crop = st.selectbox("🌱 Select Crop", crop_names)

    selected_state = st.selectbox("📍 Select State", state_names)

    selected_season = st.selectbox("☀️ Select Season", season_names)

    area = st.number_input(
        "🌾 Area (Hectares)",
        min_value=0.0,
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

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Crop Yield"):

    try:
        # Create Input DataFrame
        input_data = pd.DataFrame({
            "Crop": [crop_encoder.transform([selected_crop])[0]],
            "Season": [season_encoder.transform([selected_season])[0]],
            "State": [state_encoder.transform([selected_state])[0]],
            "Area": [area],
            "Production": [production],
            "Annual_Rainfall": [rainfall],
            "Fertilizer": [fertilizer],
            "Pesticide": [pesticide]
        })

        # Match training column order
        input_data = input_data[feature_columns]

        # Prediction
        prediction = model.predict(input_data)

        # ---------------- OUTPUT ----------------
        st.markdown("---")

        st.success(
            f"🌟 Estimated Crop Yield: {float(prediction[0]):.2f} Quintal/Hectare"
        )

        st.info(
            "Prediction is based on rainfall, fertilizer, pesticide, "
            "production, area, crop type, season, and state."
        )

    except Exception as e:
        st.error(f"Prediction Error: {e}")
