import os
import streamlit as st
import pandas as pd
import joblib

# Setting Page Configuration
st.set_page_config(
    page_title="Hyderabad House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# Load the model
@st.cache_resource
def load_model():
    model_path = os.path.join("models", "best_model.pkl")
    return joblib.load(model_path)

with st.spinner("Loading machine learning model..."):
    model = load_model()

# Load dataset
@st.cache_data
def load_data():
    candidates = [
        os.path.join("data", "Hyderbad_House_price.csv"),
        os.path.join("data", "Hyderabad_House_Price.csv"),
        os.path.join("data", "Hyderabad_House_price.csv"),
        os.path.join("data", "hyderabad_house_price.csv")
    ]
    for path in candidates:
        if os.path.exists(path):
            return pd.read_csv(path)
    if os.path.exists("data"):
        for f in os.listdir("data"):
            if f.lower().endswith(".csv"):
                return pd.read_csv(os.path.join("data", f))
    raise FileNotFoundError("House price dataset CSV not found in data/ folder")

with st.spinner("Loading property data..."):
    df = load_data()

# Creating BHK feature if missing
if "bhk" not in df.columns:
    df["bhk"] = (
        df["title"]
        .str.extract(r"(\d+)\s*BHK", expand=False)
        .astype(float)
    )
    # Handle RK properties
    df.loc[
        df["title"].str.contains("RK", case=False, na=False),
        "bhk"
    ] = 1

# Creating Property Type feature if missing
def get_property_type(title):
    title = str(title).lower()
    if "plot" in title:
        return "Residential Plot"
    elif "villa" in title:
        return "Villa"
    elif "independent house" in title:
        return "Independent House"
    elif "independent floor" in title:
        return "Independent Floor"
    elif "studio" in title:
        return "Studio Apartment"
    elif "apartment" in title:
        return "Apartment"
    else:
        return "Other"

if "property_type" not in df.columns:
    df["property_type"] = df["title"].apply(get_property_type)

# App Header
st.title("🏠 Hyderabad House Price Prediction")
st.markdown(
    "Estimate property prices across various localities in Hyderabad using Machine Learning."
)
st.divider()

# User Inputs
st.subheader("Property Specifications")

col1, col2 = st.columns(2)

with col1:
    location = st.selectbox(
        "📍 Location",
        sorted(df["location"].dropna().unique())
    )
    
    area = st.number_input(
        "📐 Area (sq. ft)",
        min_value=100,
        max_value=20000,
        value=1200,
        step=50
    )
    
    bhk = st.number_input(
        "🛏️ BHK",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )

with col2:
    property_type = st.selectbox(
        "🏢 Property Type",
        sorted(df["property_type"].dropna().unique())
    )
    
    building_status = st.selectbox(
        "🏗️ Building Status",
        sorted(df["building_status"].dropna().unique())
    )

st.divider()

# Prediction Button
if st.button("🔮 Predict House Price", use_container_width=True, type="primary"):
    input_data = pd.DataFrame({
        "location": [location],
        "area_insqft": [area],
        "building_status": [building_status],
        "bhk": [bhk],
        "property_type": [property_type]
    })

    with st.spinner("Estimating price..."):
        prediction = model.predict(input_data)[0]

    st.success("✅ Prediction Completed!")
    
    # Formatted display
    st.subheader("Estimated Property Valuation")
    
    if prediction >= 100:
        crores = prediction / 100
        st.metric(
            label="Estimated Price",
            value=f"₹ {crores:.2f} Cr",
            delta=f"₹ {prediction:.2f} Lakhs"
        )
    else:
        st.metric(
            label="Estimated Price",
            value=f"₹ {prediction:.2f} Lakhs"
        )
    
    st.caption("ℹ️ *Price estimates are based on historical real estate market data in Hyderabad.*")
