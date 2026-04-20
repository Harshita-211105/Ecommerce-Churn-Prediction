import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)


st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# UI
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

# Load Model
try:
    model = pickle.load(open(os.path.join(ROOT_DIR, "churn_model.pkl"), "rb"))
    model_features = pickle.load(open(os.path.join(ROOT_DIR, "model_columns.pkl"), "rb"))
    st.sidebar.success("Model loaded successfully")
except Exception as e:
    model = None
    model_features = []
    st.sidebar.error(f"Model loading error: {e}")

# Load Dataset
try:
    df = pd.read_csv(os.path.join(ROOT_DIR, "ecommerce_customer_churn_dataset.csv"))
except Exception as e:
    df = pd.DataFrame()
    st.sidebar.error(f"Dataset loading error: {e}")

# Feature Alignment
def align_features(input_df, required_features):
    input_df = pd.get_dummies(input_df, drop_first=True)

    for col in required_features:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[required_features]
    return input_df

# Input
st.sidebar.header("📝 Customer Inputs")

def user_input():
    data = {}

    data["Age"] = st.sidebar.slider("Age", 18, 75, 30)
    data["Membership_Years"] = st.sidebar.slider("Membership Years", 0.0, 10.0, 2.0)
    data["Login_Frequency"] = st.sidebar.slider("Login Frequency", 0.0, 50.0, 10.0)
    data["Session_Duration_Avg"] = st.sidebar.slider("Session Duration Avg", 1.0, 80.0, 20.0)
    data["Pages_Per_Session"] = st.sidebar.slider("Pages Per Session", 1.0, 25.0, 6.0)
    data["Cart_Abandonment_Rate"] = st.sidebar.slider("Cart Abandonment Rate", 0.0, 100.0, 40.0)
    data["Wishlist_Items"] = st.sidebar.slider("Wishlist Items", 0.0, 30.0, 4.0)
    data["Total_Purchases"] = st.sidebar.slider("Total Purchases", 0.0, 130.0, 12.0)
    data["Average_Order_Value"] = st.sidebar.slider("Average Order Value", 20.0, 10000.0, 120.0)
    data["Days_Since_Last_Purchase"] = st.sidebar.slider("Days Since Last Purchase", 0.0, 300.0, 20.0)
    data["Discount_Usage_Rate"] = st.sidebar.slider("Discount Usage Rate", 0.0, 100.0, 40.0)
    data["Returns_Rate"] = st.sidebar.slider("Returns Rate", 0.0, 100.0, 5.0)
    data["Email_Open_Rate"] = st.sidebar.slider("Email Open Rate", 0.0, 100.0, 20.0)
    data["Customer_Service_Calls"] = st.sidebar.slider("Customer Service Calls", 0.0, 25.0, 5.0)
    data["Product_Reviews_Written"] = st.sidebar.slider("Product Reviews Written", 0.0, 25.0, 2.0)
    data["Social_Media_Engagement_Score"] = st.sidebar.slider("Social Media Engagement Score", 0.0, 100.0, 30.0)
    data["Mobile_App_Usage"] = st.sidebar.slider("Mobile App Usage", 0.0, 65.0, 20.0)
    data["Payment_Method_Diversity"] = st.sidebar.slider("Payment Method Diversity", 1.0, 5.0, 2.0)
    data["Lifetime_Value"] = st.sidebar.slider("Lifetime Value", 0.0, 9000.0, 1200.0)
    data["Credit_Balance"] = st.sidebar.slider("Credit Balance", 0.0, 7200.0, 1800.0)

    data["Gender"] = st.sidebar.selectbox("Gender", ["Male", "Female"])
    data["Country"] = st.sidebar.selectbox("Country", ["France", "UK", "Canada", "USA", "India", "Japan", "Germany", "Australia"])
    data["City"] = st.sidebar.text_input("City", "New York")
    data["Signup_Quarter"] = st.sidebar.selectbox("Signup Quarter", ["Q1", "Q2", "Q3", "Q4"])

    return pd.DataFrame([data])

input_df = user_input()

st.title("📊 Customer Churn Prediction Dashboard")

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📌 Why Churn Prediction Matters</div>', unsafe_allow_html=True)

st.write("""
Customer churn prediction is a critical task for businesses because retaining existing customers is often far more cost-effective than acquiring new ones. By analyzing user behavior, engagement patterns, and transaction history, organizations can identify early warning signs that indicate a customer may stop using their service. This enables companies to take proactive steps such as personalized offers, improved customer support, or targeted engagement strategies to retain valuable users. Accurate churn prediction not only improves customer satisfaction but also helps in maximizing long-term revenue and building sustainable business growth.
""")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📝 Customer Input Data</div>', unsafe_allow_html=True)

st.dataframe(input_df, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)


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
                st.error(f"⚠️ High Risk of Churn ({prob*100:.2f}%)")
            else:
                st.success(f"✅ Customer Likely to Stay ({prob*100:.2f}%)")

            st.write(f"Churn Probability: {prob*100:.2f}%")
            st.progress(float(prob))

            if prob < 0.30:
                risk_level = "Low Risk"
            elif prob < 0.60:
                risk_level = "Medium Risk"
            else:
                risk_level = "High Risk"

            st.markdown(f"### Risk Level: {risk_level}")

            if prob >= 0.60:
                st.warning("Recommended Action: Offer discount, re-engagement email, and support follow-up.")
            elif prob >= 0.30:
                st.info("Recommended Action: Send personalised offers and monitor engagement.")
            else:
                st.success("Recommended Action: Customer is stable. Continue loyalty strategies.")

        except Exception as e:
            st.error(f"Error: {e}")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📈 Customer vs Average Comparison</div>', unsafe_allow_html=True)

compare_cols = [
    "Login_Frequency",
    "Session_Duration_Avg",
    "Pages_Per_Session",
    "Cart_Abandonment_Rate",
    "Email_Open_Rate",
    "Customer_Service_Calls"
]

if not df.empty:
    avg_values = df[compare_cols].mean()
    user_values = input_df[compare_cols].iloc[0]

    compare_df = pd.DataFrame({
        "Feature": compare_cols,
        "Customer": user_values.values,
        "Average": avg_values.values
    })

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(compare_df))
    width = 0.35

    ax.bar(x - width/2, compare_df["Customer"], width, label="Customer")
    ax.bar(x + width/2, compare_df["Average"], width, label="Average")

    ax.set_xticks(x)
    ax.set_xticklabels(compare_df["Feature"], rotation=45)
    ax.set_title("Customer vs Dataset Average")
    ax.legend()

    st.pyplot(fig)

st.markdown('</div>', unsafe_allow_html=True)


st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📖 Key Insights</div>', unsafe_allow_html=True)

st.write("""
The visualizations highlight that customer engagement plays a major role in retention. Users who interact more frequently with the platform, spend more time per session, and show consistent activity are significantly less likely to churn. On the other hand, higher cart abandonment rates and lower interaction metrics are strong indicators of potential churn. These insights can help businesses design targeted strategies to improve customer retention.
""")

st.markdown('</div>', unsafe_allow_html=True)