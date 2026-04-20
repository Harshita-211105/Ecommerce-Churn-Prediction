import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import os

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# -------------------------------
# CUSTOM UI
# -------------------------------
st.markdown("""
<style>

/* ---------- GLOBAL ---------- */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #0f172a;
    color: #e2e8f0;
}

/* ---------- MAIN CONTAINER ---------- */
.main {
    padding: 20px;
    animation: fadeIn 0.6s ease-in-out;
}

/* ---------- ANIMATION ---------- */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px);}
    to { opacity: 1; transform: translateY(0);}
}

/* ---------- CARD DESIGN ---------- */
.card {
    background: linear-gradient(145deg, #1e293b, #0f172a);
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 20px;
    border: 1px solid #1e293b;
    box-shadow: 0 6px 25px rgba(0,0,0,0.4);
    transition: all 0.3s ease;
}

.card:hover {
    transform: translateY(-6px);
    box-shadow: 0 12px 35px rgba(0,0,0,0.6);
}

/* ---------- HEADINGS ---------- */
.section-title {
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 12px;
    color: #38bdf8;
    letter-spacing: 0.5px;
}

/* ---------- BUTTON ---------- */
.stButton>button {
    background: linear-gradient(135deg, #38bdf8, #6366f1);
    color: white;
    border-radius: 12px;
    padding: 10px 22px;
    font-weight: 600;
    border: none;
    transition: all 0.25s ease;
}

.stButton>button:hover {
    transform: scale(1.06);
    box-shadow: 0 6px 20px rgba(99,102,241,0.5);
}

/* ---------- SIDEBAR ---------- */
section[data-testid="stSidebar"] {
    background: #020617;
    border-right: 1px solid #1e293b;
}

/* ---------- INPUTS ---------- */
.stSlider, .stSelectbox {
    padding-top: 5px;
}

/* ---------- DATAFRAME ---------- */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* ---------- IMAGE STYLING ---------- */
img {
    border-radius: 12px;
    transition: 0.3s;
}

img:hover {
    transform: scale(1.02);
}

/* ---------- CAPTION ---------- */
.stCaption {
    font-size: 13px;
    color: #94a3b8;
}

/* ---------- SUCCESS / ERROR ---------- */
.stAlert {
    border-radius: 12px;
}

/* ---------- TITLE ---------- */
h1 {
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: 0.5px;
}

/* ---------- SCROLLBAR ---------- */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-thumb {
    background: #334155;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD MODEL
# -------------------------------
try:
    model = pickle.load(open("churn_model.pkl", "rb"))
    model_features = model.feature_names_in_
except:
    model = None
    model_features = []

# -------------------------------
# LOAD DATASET
# -------------------------------
try:
    df = pd.read_csv("your_dataset.csv")
except:
    df = pd.DataFrame()

# -------------------------------
# FEATURE ALIGNMENT
# -------------------------------
def align_features(df, required_features):
    for col in required_features:
        if col not in df.columns:
            df[col] = 0
    return df[required_features]

# -------------------------------
# SIDEBAR INPUT
# -------------------------------
st.sidebar.header("📝 Customer Inputs")

def user_input():
    data = {}

    fields = {
        "Age": (18, 70, 30),
        "Membership_Years": (0, 10, 2),
        "Login_Frequency": (0, 50, 10),
        "Session_Duration_Avg": (1, 60, 15),
        "Pages_Per_Session": (1, 20, 5),
        "Cart_Abandonment_Rate": (0.0, 1.0, 0.3),
        "Wishlist_Items": (0, 50, 5),
        "Email_Open_Rate": (0.0, 1.0, 0.5),
        "Customer_Service_Calls": (0, 20, 2),
        "Product_Reviews_Written": (0, 50, 3),
        "Social_Media_Engagement_Score": (0, 100, 50),
        "Mobile_App_Usage": (0, 100, 40),
        "Payment_Method_Diversity": (1, 5, 2),
        "Lifetime_Value": (100, 10000, 2000),
        "Credit_Balance": (0, 5000, 500)
    }

    for field, (min_val, max_val, default) in fields.items():
        data[field] = st.sidebar.slider(field, min_val, max_val, default)

    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    data["Gender"] = 1 if gender == "Male" else 0

    return pd.DataFrame([data])

input_df = user_input()

# -------------------------------
# TITLE
# -------------------------------
st.title("📊 Customer Churn Prediction Dashboard")

# -------------------------------
# 1. INTRO PARAGRAPH
# -------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📌 Why Churn Prediction Matters</div>', unsafe_allow_html=True)

st.write("""
Customer churn prediction is a critical task for businesses because retaining existing customers is often far more cost-effective than acquiring new ones. By analyzing user behavior, engagement patterns, and transaction history, organizations can identify early warning signs that indicate a customer may stop using their service. This enables companies to take proactive steps such as personalized offers, improved customer support, or targeted engagement strategies to retain valuable users. Accurate churn prediction not only improves customer satisfaction but also helps in maximizing long-term revenue and building sustainable business growth.
""")

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# 2. USER INPUT DISPLAY
# -------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📝 Customer Input Data</div>', unsafe_allow_html=True)

st.dataframe(input_df, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# 3. PREDICTION
# -------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🔮 Prediction Result</div>', unsafe_allow_html=True)

if st.button("Predict Churn"):
    if model is None:
        st.warning("⚠️ Model not loaded")
    else:
        try:
            aligned_df = align_features(input_df.copy(), model_features)
            prediction = model.predict(aligned_df)[0]
            prob = model.predict_proba(aligned_df)[0][1]

            if prediction == 1:
                st.error(f"⚠️ High Risk of Churn ({prob:.2f})")
            else:
                st.success(f"✅ Customer Likely to Stay ({prob:.2f})")

        except Exception as e:
            st.error(f"Error: {e}")

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# 5. INSIGHT IMAGES WITH EXPLANATIONS
# -------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📷 Visual Insights & Interpretation</div>', unsafe_allow_html=True)

image_files = ["Visualisations/insight1.png", "Visualisations/insight2.png", "Visualisations/insight3.png"]

col1, col2, col3 = st.columns(3)

# ---- IMAGE 1 ----
with col1:
    if os.path.exists(image_files[0]):
        img = Image.open(image_files[0])
        st.image(img, use_container_width=True)
        st.markdown("**Customer Churn Distribution**")
        st.caption("""
The dataset shows a clear imbalance where most customers are retained and fewer customers churn.  
This indicates the need for careful model handling to avoid bias toward predicting non-churn.
""")
    else:
        st.info("insight1.png not found")

# ---- IMAGE 2 ----
with col2:
    if os.path.exists(image_files[1]):
        img = Image.open(image_files[1])
        st.image(img, use_container_width=True)
        st.markdown("**Feature Correlation Heatmap**")
        st.caption("""
User engagement features such as login frequency and session duration are strongly correlated with each other.  
Churn is negatively associated with engagement and positively linked to factors like cart abandonment and service calls.
""")
    else:
        st.info("insight2.png not found")

# ---- IMAGE 3 ----
with col3:
    if os.path.exists(image_files[2]):
        img = Image.open(image_files[2])
        st.image(img, use_container_width=True)
        st.markdown("**Churn by Country**")
        st.caption("""
Higher churn counts in countries like the USA and UK are mainly due to larger customer bases.  
The churn ratio remains fairly consistent across countries, indicating behavior matters more than location.
""")
    else:
        st.info("insight3.png not found")

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# 6. FINAL INSIGHTS TEXT
# -------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📖 Key Insights</div>', unsafe_allow_html=True)

st.write("""
The visualizations highlight that customer engagement plays a major role in retention. Users who interact more frequently with the platform, spend more time per session, and show consistent activity are significantly less likely to churn. On the other hand, higher cart abandonment rates and lower interaction metrics are strong indicators of potential churn. These insights can help businesses design targeted strategies to improve customer retention.
""")

st.markdown('</div>', unsafe_allow_html=True)