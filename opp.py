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
    'send_page_view': true
  }});
</script>
"""

components.html(ga_snippet, height=0, width=0)
