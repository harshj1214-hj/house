import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_gtag import st_gtag

# 1. Page Configuration
st.set_page_config(
    page_title="Advanced ML Property Valuation Pro",
    page_icon="🏠",
    layout="wide"
)

# 2. Fire Google Analytics Event
st_gtag(
    id="gtag_send_event",
    event_name="page_view",
    params={
        "page_title": "AI Property Valuation Pro",
        "page_location": "https://www.jainovation.xyz",
        "send_to": "G-K6EGDWJ6D1"
    }
)
