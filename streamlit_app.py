import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="UIDAI Biometric Risk Prediction System",
    layout="wide"
)

st.title("UIDAI Biometric Risk Prediction System")
st.markdown("---")

# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    ["Risk Prediction", "Data Analysis", "About"]
)

# ---------------- LOAD MODELS ----------------
@st.cache_data
def load_models():
    model_files = {
        "model": "risk_prediction_model.pkl",
        "state": "state_encoder.pkl",
        "district": "district_encoder.pkl",
        "risk": "risk_encoder.pkl"
    }

    for name, path in model_files.items():
        if not os.path.exists(path):
            st.error(f"Missing model file: {path}")
            st.stop()

    model = joblib.load(model_files["model"])
    le_state = joblib.load(model_files["state"])
    le_district = joblib.load(model_files["district"])
    le_risk = joblib.load(model_files["risk"])

    return model, le_state, le_district, le_risk

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    files = [
        "api_data_aadhar_biometric_0_500000.csv",
        "api_data_aadhar_biometric_500000_1000000.csv",
        "api_data_aadhar_biometric_1000000_1500000.csv",
        "api_data_aadhar_biometric_1500000_1861108.csv"
    ]

    for f in files:
        if not os.path.exists(f):
            st.error(f"Missing data file: {f}")
            st.stop()

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # Clean columns
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    # ---- SAFE COLUMN HANDLING ----
    required_cols = ["bio_age_0_5", "bio_age_5_17"]
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Required column missing: {col}")
            st.write("Available columns:", df.columns.tolist())
            st.stop()

    df["total_bio"] = df["bio_age_5_17"] + df["bio_age_17_"]

    # Date handling
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)

    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["weekday"] = df["date"].dt.weekday

    # Risk labeling
    def usage_risk(x):
        if x < 500:
            return "Low"
        elif x < 1000:
            return "Medium"
        return "High"

    df["risk_level"] = df["total_bio"].apply(usage_risk)
    return df

# ---------------- PAGE TRANSITION ----------------
if "last_page" not in st.session_state:
    st.session_state.last_page = page

if st.session_state.last_page != page:
    with st.spinner("Switching page..."):
        time.sleep(0.3)
    st.session_state.last_page = page

# ================= RISK PREDICTION =================
if page == "Risk Prediction":
    st.header("🔮 Risk Level Prediction")

    model, le_state, le_district, le_risk = load_models()
    df = load_data()

    states = sorted(df["state"].unique().tolist())

    col1, col2 = st.columns(2)

    with col1:
        state = st.selectbox("Select State", states)

        districts_raw = (
            df[df["state"] == state]["district"]
            .dropna()
            .unique()
            .tolist()
        )
        districts_clean = [d for d in districts_raw if d not in ("?", "NA", "Unknown")]
        districts = ["Select District"] + sorted(districts_clean)

        district = st.selectbox("Select District", districts)

        pincode = st.text_input(
            "Enter Pincode",
            max_chars=6,
            placeholder="6-digit pincode"
        )

    with col2:
        selected_date = st.date_input("Select Date", datetime.now())
        day = selected_date.day
        month = selected_date.month
        weekday = selected_date.weekday()

    if st.button("Predict Risk Level", type="primary"):
        if not pincode.isdigit() or len(pincode) != 6:
            st.error("Pincode must be a valid 6-digit number.")
        elif district == "Select District":
            st.error("Please select a valid district.")
        else:
            try:
                state_enc = le_state.transform([state])[0]
                district_enc = le_district.transform([district])[0]

                input_data = [[state_enc, district_enc, int(pincode), day, month, weekday]]
                pred = model.predict(input_data)[0]
                risk = le_risk.inverse_transform([pred])[0]

                st.success("Prediction Complete")

                if risk == "Low":
                    st.success("🟢 Predicted Risk Level: **Low**")
                elif risk == "Medium":
                    st.warning("🟡 Predicted Risk Level: **Medium**")
                else:
                    st.error("🔴 Predicted Risk Level: **High**")

                with st.expander("View Input Summary"):
                    st.markdown(f"""
                    **State:** {state}  
                    **District:** {district}  
                    **Pincode:** {pincode}  
                    **Date:** {selected_date}
                    """)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ================= DATA ANALYSIS =================
elif page == "Data Analysis":
    st.header("📊 Data Analysis Dashboard")

    df = load_data()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Total Biometric Transactions", f"{df['total_bio'].sum():,}")
    col3.metric("Unique States", df["state"].nunique())
    col4.metric("Unique Districts", df["district"].nunique())

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["State Analysis", "District Analysis", "Monthly Trends", "Risk Distribution"]
    )

    with tab1:
        s = df.groupby("state")["total_bio"].sum().nlargest(10)
        fig, ax = plt.subplots()
        s.plot(kind="barh", ax=ax)
        st.pyplot(fig)

    with tab2:
        d = df.groupby("district")["total_bio"].sum().nlargest(10)
        fig, ax = plt.subplots()
        d.plot(kind="barh", ax=ax)
        st.pyplot(fig)

    with tab3:
        m = df.groupby("month")["total_bio"].sum()
        fig, ax = plt.subplots()
        m.plot(marker="o", ax=ax)
        st.pyplot(fig)

    with tab4:
        fig, ax = plt.subplots()
        df["risk_level"].value_counts().plot(
            kind="pie",
            autopct="%1.1f%%",
            ax=ax
        )
        ax.set_ylabel("")
        st.pyplot(fig)

# ================= ABOUT =================
elif page == "About":
    st.header("ℹ️ About This Application")

    st.markdown("""
**UIDAI Biometric Risk Prediction System**

This application predicts biometric usage risk levels using machine learning
and provides interactive data analysis dashboards.

**Technology Stack**
- Streamlit
- Scikit-learn
- Pandas
- Matplotlib & Seaborn

**Risk Levels**
- Low
- Medium
- High
""")
