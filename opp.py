import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration
st.set_page_config(
    page_title="AI Property Valuation Pro", 
    page_icon="🏠", 
    layout="wide"
)

# 2. Load Model Artifacts
@st.cache_resource
def load_model():
    return joblib.load('property_valuation_model.pkl')

artifacts = load_model()
model = artifacts['model']
features = artifacts['features']

# 3. Clean, High-Contrast UI Styling
st.markdown("""
<style>
    /* Main container styling */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Card containers */
    .report-card {
        background-color: #1a1c24;
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #2e3440;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    }
    
    .price-display {
        font-size: 2.4rem;
        font-weight: 700;
        color: #4ade80;
        margin: 10px 0;
    }
    
    .bound-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        margin-top: 16px;
        padding-top: 16px;
        border-top: 1px solid #2e3440;
    }

    /* Custom Button */
    div.stButton > button:first-child {
        background-color: #22c55e !important;
        color: #052e16 !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.6rem 1.2rem !important;
        width: 100%;
        transition: all 0.2s ease;
    }
    
    div.stButton > button:first-child:hover {
        background-color: #16a34a !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# 4. Header Section
st.title("🏠 Advanced Property Valuation Engine")
st.caption("Deploying localized machine learning architectures for high-precision real estate appraisal.")
st.divider()

# 5. UI Input & Prediction Layout
left_panel, right_panel = st.columns([3, 2], gap="large")

with left_panel:
    st.subheader("📋 Property Specifications")
    
    with st.expander("📐 Dimensions & Core Layout", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            area_sqft = st.number_input("Living Area (sqft)", min_value=500, max_value=10000, value=3500, step=50)
            bedrooms = st.radio("Bedrooms", options=[1, 2, 3, 4, 5, 6, 7, 8], index=2, horizontal=True)
            bathrooms = st.radio("Bathrooms", options=[1, 2, 3, 4, 5, 6], index=2, horizontal=True)
        with c2:
            lot_size = st.number_input("Total Lot Size (sqft)", min_value=int(area_sqft), max_value=100000, value=max(5000, int(area_sqft)), step=100)
            floors = st.selectbox("Number of Floors", options=[1, 2, 3, 4, 5], index=1)
            fireplaces = st.slider("Fireplaces Count", min_value=0, max_value=5, value=1)

    with st.expander("🛠️ Age & Structural Quality", expanded=True):
        c3, c4 = st.columns(2)
        with c3:
            property_age = st.number_input("Property Age (Years)", min_value=0, max_value=200, value=5)
            construction_quality = st.slider("Construction Build Quality (1-10)", min_value=1, max_value=10, value=7)
        with c4:
            renovation_status = st.slider("Renovation Tier (0=Original, 10=Modern)", min_value=0, max_value=10, value=5)
            is_duplex = st.toggle("Is this a Duplex property?", value=False)
            property_type_Duplex = 1 if is_duplex else 0

    with st.expander("📍 Location & Environmental Amenities", expanded=True):
        c5, c6 = st.columns(2)
        with c5:
            neighborhood_score = st.slider("Neighborhood Rating (1-10)", min_value=1, max_value=10, value=8)
            school_rating = st.slider("Public School Quality (1-10)", min_value=1, max_value=10, value=7)
        with c6:
            water_supply_score = st.slider("Water Infra Reliability (1-10)", min_value=1, max_value=10, value=9)
            green_space_index = st.slider("Parks & Greenery Proximity (1-10)", min_value=1, max_value=10, value=8)

with right_panel:
    st.subheader("📊 Valuation Output")
    st.info("Adjust the specification parameters on the left and trigger the engine to generate an assessment.")

    if st.button("Run Prediction Model", type="primary"):
        input_data = {
            'area_sqft': area_sqft,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'floors': floors,
            'property_age': property_age,
            'renovation_status': renovation_status,
            'lot_size': lot_size,
            'neighborhood_score': neighborhood_score,
            'school_rating': school_rating,
            'parking_availability': 1,
            'construction_quality': construction_quality,
            'water_supply_score': water_supply_score,
            'green_space_index': green_space_index,
            'Fireplaces': fireplaces,
            'property_type_Duplex': property_type_Duplex
        }
        
        input_df = pd.DataFrame([input_data])[features]
        log_pred = model.predict(input_df)
        base_price = np.expm1(log_pred)[0]

        mae_value = 18541.37
        lower_bound = max(0, base_price - mae_value)
        upper_bound = base_price + mae_value

        st.markdown(f"""
        <div class="report-card">
            <h4 style="color: #4ade80; margin: 0 0 8px 0; text-transform: uppercase; letter-spacing: 0.05em;">Valuation Report Generated</h4>
            <span style="color: #94a3b8; font-size: 0.85rem;">POINT ESTIMATE MARKET VALUE</span>
            <div class="price-display">${base_price:,.2f}</div>
            <div class="bound-container">
                <div>
                    <span style="color: #94a3b8; font-size: 0.75rem;">CONSERVATIVE (-MAE)</span>
                    <div style="font-size: 1.15rem; font-weight: 600; color: #f87171; margin-top: 4px;">${lower_bound:,.2f}</div>
                </div>
                <div>
                    <span style="color: #94a3b8; font-size: 0.75rem;">OPTIMISTIC (+MAE)</span>
                    <div style="font-size: 1.15rem; font-weight: 600; color: #60a5fa; margin-top: 4px;">${upper_bound:,.2f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.caption("⚠️ Core valuation metrics scale dynamically inside standard structural confidence bands.")
