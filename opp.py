import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration
st.set_page_config(
    page_title="Advanced ML Property Valuation Pro",
    page_icon="🏠",
    layout="wide"
)

# --- Google Analytics 4 Tracking ---
GA_TRACKING_ID = "G-K6EGDWJ6D1"

ga_snippet = f"""
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id={GA_TRACKING_ID}"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){{dataLayer.push(arguments);}}
  gtag('js', new Date());
  gtag('config', '{GA_TRACKING_ID}', {{
    'cookie_flags': 'SameSite=None;Secure',
    'send_page_view': true
  }});
</script>
"""

# Must have non-zero dimensions so the browser does not throttle the script execution
components.html(ga_snippet, height=1, width=1)

# 2. Load Model Artifacts
@st.cache_resource
def load_model():
    return joblib.load('property_valuation_model.pkl')

artifacts = load_model()
model = artifacts['model']
features = artifacts['features']

# USD to INR conversion rate helper
USD_TO_INR = 83.50

def format_inr(amount):
    if amount >= 10000000:
        return f"₹{amount / 10000000:.2f} Cr"
    elif amount >= 100000:
        return f"₹{amount / 100000:.2f} Lakh"
    return f"₹{amount:,.0f}"

# 3. Clean UI Styling
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .report-card {
        background-color: #1a1c24;
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #2e3440;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    }
    .price-usd {
        font-size: 2.2rem;
        font-weight: 700;
        color: #4ade80;
        margin: 4px 0 0 0;
    }
    .price-inr {
        font-size: 1.4rem;
        font-weight: 600;
        color: #facc15;
        margin-bottom: 12px;
    }
    .bound-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        margin-top: 16px;
        padding-top: 16px;
        border-top: 1px solid #2e3440;
    }
    div.stButton > button:first-child {
        background-color: #22c55e !important;
        color: #052e16 !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.6rem 1.2rem !important;
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        background-color: #16a34a !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# 4. Header Section
st.title("🏠 Advanced ML Property Valuation Engine")
st.caption("Deploying localized machine learning architectures for high-precision real estate appraisal.")
st.divider()

# 5. UI Layout
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
        base_price_usd = np.expm1(log_pred)[0]
        mae_value = 18541.37
        
        lower_bound_usd = max(0, base_price_usd - mae_value)
        upper_bound_usd = base_price_usd + mae_value
        
        # Calculate INR values
        base_price_inr = base_price_usd * USD_TO_INR
        lower_bound_inr = lower_bound_usd * USD_TO_INR
        upper_bound_inr = upper_bound_usd * USD_TO_INR
        
        st.markdown(f"""
        <div class="report-card">
            <h4 style="color: #4ade80; margin: 0 0 8px 0; text-transform: uppercase; letter-spacing: 0.05em;">Valuation Report Generated</h4>
            <span style="color: #94a3b8; font-size: 0.85rem;">POINT ESTIMATE MARKET VALUE</span>
            <div class="price-usd">${base_price_usd:,.2f}</div>
            <div class="price-inr">({format_inr(base_price_inr)})</div>
            <div class="bound-container">
                <div>
                    <span style="color: #94a3b8; font-size: 0.75rem;">CONSERVATIVE (-MAE)</span>
                    <div style="font-size: 1.1rem; font-weight: 600; color: #f87171; margin-top: 4px;">${lower_bound_usd:,.2f}</div>
                    <div style="font-size: 0.85rem; color: #fca5a5;">{format_inr(lower_bound_inr)}</div>
                </div>
                <div>
                    <span style="color: #94a3b8; font-size: 0.75rem;">OPTIMISTIC (+MAE)</span>
                    <div style="font-size: 1.1rem; font-weight: 600; color: #60a5fa; margin-top: 4px;">${upper_bound_usd:,.2f}</div>
                    <div style="font-size: 0.85rem; color: #93c5fd;">{format_inr(upper_bound_inr)}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("⚠️ Conversions use a reference standard rate (1 USD ≈ ₹83.50).")
