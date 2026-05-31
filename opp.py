import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load the exported model artifacts
@st.cache_resource
def load_model():
    return joblib.load('property_valuation_model.pkl')

artifacts = load_model()
model = artifacts['model']
features = artifacts['features']

# 2. Web Page Configuration & Custom Styling
st.set_page_config(page_title="AI Property Valuation Pro", page_icon="🏠", layout="wide")

# Custom CSS to elevate UI appearance
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%;
        background-color: #2E7D32 !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        padding: 0.6rem 1rem !important;
        border: none !important;
    }
    .report-card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border-left: 5px solid #2E7D32;
    }
    </style>
""", unsafe_allow_html=True)

# App Header
st.title("🏠 Advanced Property Valuation Engine")
st.markdown("##### Deploying localized machine learning architectures for high-precision real estate appraisal.")
st.divider()

# 3. Structural Layout: Two major functional zones
left_panel, right_panel = st.columns([3, 2], gap="large")

with left_panel:
    st.markdown("### 📋 Property Specifications")
    
    # Grouping features into intuitive expandable UI sections
    with st.expander("📐 Dimensions & Core Layout", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            area_sqft = st.number_input("Living Area (sqft)", min_value=500, max_value=10000, value=3500, step=50)
            bedrooms = st.segmented_control("Bedrooms", options=[1, 2, 3, 4, 5, 6, 7, 8], default=3)
            bathrooms = st.segmented_control("Bathrooms", options=[1, 2, 3, 4, 5, 6], default=3)
        with c2:
            # Smart validation context mapping
            lot_size = st.number_input("Total Lot Size (sqft)", min_value=int(area_sqft), max_value=100000, value=max(5000, int(area_sqft)), step=100)
            floors = st.selectbox("Number of Floors", options=[1, 2, 3, 4, 5], index=1)
            fireplaces = st.slider("Fireplaces Count", min_value=0, max_value=5, value=1)

    with st.expander("🛠️ Age & Structural Quality", expanded=True):
        c3, c4 = st.columns(2)
        with c3:
            property_age = st.number_input("Property Age (Years)", min_value=0, max_value=200, value=5)
            construction_quality = st.slider("Construction Build Quality", min_value=1, max_value=10, value=7, help="Material & structural integrity score")
        with c4:
            renovation_status = st.slider("Renovation Tier", min_value=0, max_value=10, value=5, help="0 = Original Condition, 10 = Fully Modernized")
            is_duplex = st.toggle("Is this a Duplex property?", value=False)
            property_type_Duplex = 1 if is_duplex else 0

    with st.expander("📍 Location & Environmental Amenities", expanded=True):
        c5, c6 = st.columns(2)
        with c5:
            neighborhood_score = st.slider("Neighborhood Rating", min_value=1, max_value=10, value=8)
            school_rating = st.slider("Public School Quality Index", min_value=1, max_value=10, value=7)
        with c6:
            water_supply_score = st.slider("Water Infra Reliability", min_value=1, max_value=10, value=9)
            green_space_index = st.slider("Parks & Greenery Proximity", min_value=1, max_value=10, value=8)

with right_panel:
    st.markdown("### 📊 Valuation Output")
    st.info("Adjust the specification parameters on the left and trigger the engine to generate an assessment.")
    
    # Trigger valuation
    if st.button("Run Prediction Model", type="primary"):
        # Map values directly to training format
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
            'parking_availability': 1,  # Kept intact from original specifications
            'construction_quality': construction_quality,
            'water_supply_score': water_supply_score,
            'green_space_index': green_space_index,
            'Fireplaces': fireplaces,
            'property_type_Duplex': property_type_Duplex
        }
        
        # Enforce correct structural ordering for feature pipeline
        input_df = pd.DataFrame([input_data])[features]
        
        # Inverse log transform log1p predictions back to true dollar amounts
        log_pred = model.predict(input_df)
        base_price = np.expm1(log_pred)[0]
        
        # Calculate evaluation margins using model's exact baseline MAE
        mae_value = 18541.37
        lower_bound = max(0, base_price - mae_value)
        upper_bound = base_price + mae_value
        
        # Clean custom HTML wrapper presentation for outputs
        st.markdown(f"""
            <div class="report-card">
                <h3 style="color: #2E7D32; margin-top:0;">VALUATION REPORT GENERATED</h3>
                <hr style="margin: 10px 0;">
                <p style="margin-bottom:2px; color:#555; font-size:14px;">POINT ESTIMATE MARKET VALUE</p>
                <h1 style="color:#111; margin-top:0; font-size:42px;">${base_price:,.2f}</h1>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 20px;">
                    <div>
                        <p style="margin:0; color:#666; font-size:12px;">CONSERVATIVE BOUND (-MAE)</p>
                        <strong style="font-size:18px; color:#c62828;">${lower_bound:,.2f}</strong>
                    </div>
                    <div>
                        <p style="margin:0; color:#666; font-size:12px;">OPTIMISTIC BOUND (+MAE)</p>
                        <strong style="font-size:18px; color:#1565c0;">${upper_bound:,.2f}</strong>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Visual contextual aids
        st.caption("⚠️ Core valuation metrics scale dynamically inside standard structural confidence bands.")