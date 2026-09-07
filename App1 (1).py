import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =========================================================
# PAGE
# =========================================================
st.set_page_config(
    page_title="BigMart Sales AI",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# LOAD TRAINED MODEL + SCALER
# =========================================================
@st.cache_resource
def load_model():
    with open("bigmart_knn_model.pkl", "rb") as f:
        saved = pickle.load(f)
    return saved["model"], saved["scaler"]

model, scaler = load_model()

# =========================================================
# OPTIONS FROM THE TRAINING DATASET
# =========================================================
ITEM_TYPES = [
    "Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables",
    "Household", "Baking Goods", "Snack Foods", "Frozen Foods",
    "Breakfast", "Health and Hygiene", "Hard Drinks", "Canned",
    "Breads", "Starchy Foods", "Others", "Seafood"
]

OUTLETS = [
    "OUT049", "OUT018", "OUT010", "OUT013", "OUT027",
    "OUT045", "OUT017", "OUT046", "OUT035", "OUT019"
]

OUTLET_TYPES = [
    "Grocery Store",
    "Supermarket Type1",
    "Supermarket Type2",
    "Supermarket Type3"
]

FEATURES = list(scaler.feature_names_in_)

# =========================================================
# MODERN UI
# =========================================================
st.markdown("""
<style>
    .stApp {
        background: #f5f7fb;
    }

    #MainMenu, footer, header {
        visibility: hidden;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #172554, #1e3a8a 55%, #312e81);
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    .hero {
        background: linear-gradient(115deg, #2563eb, #4f46e5 45%, #7c3aed 72%, #db2777);
        border-radius: 22px;
        padding: 30px 34px;
        color: white;
        box-shadow: 0 12px 30px rgba(37,99,235,.18);
        margin-bottom: 22px;
    }

    .hero-grid {
        display: grid;
        grid-template-columns: 1.15fr .85fr;
        gap: 24px;
        align-items: center;
    }

    .hero h1 {
        margin: 0;
        font-size: 38px;
        line-height: 1.08;
        font-weight: 900;
    }

    .hero h1 span {
        color: #fde047;
    }

    .hero p {
        margin: 10px 0 0;
        color: #eef2ff;
        font-size: 14px;
    }

    .hero-cards {
        display: flex;
        justify-content: flex-end;
        gap: 10px;
        flex-wrap: wrap;
    }

    .hero-card {
        min-width: 125px;
        padding: 14px;
        border-radius: 14px;
        background: rgba(255,255,255,.12);
        border: 1px solid rgba(255,255,255,.25);
    }

    .hero-card .emoji {
        font-size: 22px;
    }

    .hero-card b {
        display: block;
        margin-top: 4px;
        font-size: 12px;
    }

    .hero-card small {
        color: #e0e7ff;
        font-size: 10px;
    }

    .section {
        background: white;
        border: 1px solid #e3e8f1;
        border-radius: 17px;
        padding: 19px;
        box-shadow: 0 6px 18px rgba(15,23,42,.05);
        margin-bottom: 14px;
    }

    .section-title {
        color: #17356f;
        font-size: 19px;
        font-weight: 850;
    }

    .section-sub {
        color: #64748b;
        font-size: 12px;
        margin-top: 3px;
    }

    .line {
        width: 90px;
        height: 3px;
        border-radius: 8px;
        background: linear-gradient(90deg,#2563eb,#7c3aed);
        margin-top: 12px;
    }

    div[data-testid="stTextInput"] label,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stSelectbox"] label {
        color: #075fd6 !important;
        font-weight: 750 !important;
        font-size: 13px !important;
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input {
        color: #172033 !important;
        background: #fff !important;
        border: 1px solid #d5deec !important;
        border-radius: 9px !important;
    }

    div[data-baseweb="select"] > div {
        background: #fff !important;
        color: #172033 !important;
        border-color: #d5deec !important;
        border-radius: 9px !important;
    }

    .ready {
        text-align: center;
        padding: 40px 15px;
        border-radius: 16px;
        background: linear-gradient(135deg,#eff6ff,#f5f3ff);
        border: 1px solid #dbeafe;
    }

    .ready .big {
        font-size: 38px;
    }

    .ready h3 {
        color: #243b72;
        margin: 8px 0 5px;
    }

    .ready p {
        color: #64748b;
        font-size: 12px;
    }

    .sales-card {
        text-align: center;
        padding: 25px 18px;
        border-radius: 17px;
        background: linear-gradient(135deg,#ecfdf5,#f0fdf4);
        border: 1px solid #86efac;
        box-shadow: 0 8px 24px rgba(16,185,129,.08);
    }

    .sales-label {
        color: #047857;
        font-size: 12px;
        font-weight: 850;
        letter-spacing: .06em;
    }

    .sales-value {
        color: #047857 !important;
        font-size: 34px;
        font-weight: 950;
        margin-top: 7px;
    }

    .sales-note {
        color: #475569;
        font-size: 11px;
        margin-top: 4px;
    }

    .mini {
        border-radius: 13px;
        padding: 15px;
        border: 1px solid;
        min-height: 70px;
    }

    .mini-blue {
        background: #eff6ff;
        border-color: #bfdbfe;
    }

    .mini-purple {
        background: #f5f3ff;
        border-color: #ddd6fe;
    }

    .mini-label {
        color: #64748b;
        font-size: 10px;
        font-weight: 850;
        letter-spacing: .05em;
    }

    .mini-value {
        color: #17356f !important;
        font-size: 17px;
        font-weight: 900;
        margin-top: 5px;
    }

    .tip {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 13px;
        padding: 13px;
        color: #7c2d12;
        font-size: 12px;
        margin-top: 13px;
    }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 11px;
        padding: 18px;
    }

    div.stButton > button {
        min-height: 46px;
        border-radius: 10px;
        font-weight: 850;
    }

    @media(max-width:900px) {
        .hero-grid { grid-template-columns: 1fr; }
        .hero-cards { justify-content: flex-start; }
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# HERO
# =========================================================
st.markdown("""
<div class="hero">
    <div class="hero-grid">
        <div>
            <h1>🛒 BigMart Sales<br><span>Prediction System</span></h1>
            <p>AI-powered outlet sales estimation using KNN Regression.</p>
        </div>
        <div class="hero-cards">
            <div class="hero-card">
                <div class="emoji">🤖</div>
                <b>KNN Regression</b>
                <small>Machine Learning Model</small>
            </div>
            <div class="hero-card">
                <div class="emoji">📊</div>
                <b>35 Features</b>
                <small>Processed Model Inputs</small>
            </div>
            <div class="hero-card">
                <div class="emoji">⚡</div>
                <b>Instant Result</b>
                <small>Real-time Estimation</small>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 🛒 BigMart AI")
    st.caption("Outlet sales prediction dashboard")
    st.divider()

    st.markdown("### 🤖 Model")
    st.markdown("**KNN Regressor**")

    st.markdown("### ⚙️ Pipeline")
    st.markdown(
        "- 35 processed features\n"
        "- MinMaxScaler\n"
        "- KNN Regression\n"
        "- Real-time prediction"
    )

    st.divider()
    st.info(
        "Enter product and outlet details, then click "
        "**Predict Outlet Sales**."
    )

# =========================================================
# MAIN COLUMNS
# =========================================================
left, right = st.columns([1.35, 1], gap="large")

# =========================================================
# INPUTS
# =========================================================
with left:
    st.markdown("""
    <div class="section">
        <div class="section-title">📦 Product & Outlet Information</div>
        <div class="section-sub">Enter the details used by the trained KNN model.</div>
        <div class="line"></div>
    </div>
    """, unsafe_allow_html=True)

    a, b = st.columns(2)

    with a:
        item_weight = st.number_input(
            "⚖️ Item Weight", min_value=4.555, max_value=21.35,
            value=12.6, step=0.01
        )
        fat = st.selectbox("🥗 Item Fat Content", ["Low Fat", "Regular"])
        visibility = st.number_input(
            "👁️ Item Visibility", min_value=0.0, max_value=0.3284,
            value=0.054, step=0.001, format="%.4f"
        )
        mrp = st.number_input(
            "💰 Item MRP", min_value=31.29, max_value=266.89,
            value=143.01, step=0.01
        )
        year = st.selectbox(
            "📅 Outlet Establishment Year",
            [1985, 1987, 1997, 1998, 1999, 2002, 2004, 2007, 2009],
            index=4
        )

    with b:
        item_type = st.selectbox("🏷️ Item Type", ITEM_TYPES)
        outlet_id = st.selectbox("🏪 Outlet Identifier", OUTLETS)
        outlet_size = st.selectbox(
            "📐 Outlet Size", ["Small", "Medium", "High"], index=1
        )
        location = st.selectbox(
            "📍 Outlet Location", ["Tier 1", "Tier 2", "Tier 3"], index=2
        )
        outlet_type = st.selectbox("🏬 Outlet Type", OUTLET_TYPES)

    st.markdown("<br>", unsafe_allow_html=True)
    predict = st.button(
        "🚀 Predict Outlet Sales",
        type="primary",
        use_container_width=True
    )

# =========================================================
# RESULT
# =========================================================
with right:
    st.markdown("""
    <div class="section">
        <div class="section-title">🎯 Prediction Result</div>
        <div class="section-sub">Estimated Item Outlet Sales</div>
        <div class="line"></div>
    </div>
    """, unsafe_allow_html=True)

    if predict:
        # Build exactly the 35 columns used by the saved scaler/model.
        row = {feature: 0.0 for feature in FEATURES}

        # Numeric / encoded features from the notebook.
        row["Item_Weight"] = item_weight
        row["Item_Fat_Content"] = 0 if fat == "Low Fat" else 1
        row["Item_Visibility"] = visibility
        row["Item_MRP"] = mrp
        row["Outlet_Establishment_Year"] = year
        row["Outlet_Size"] = {"Small": 0, "Medium": 1, "High": 2}[outlet_size]
        row["Outlet_Location_Type"] = {"Tier 1": 0, "Tier 2": 1, "Tier 3": 2}[location]

        # Item_Type dummy encoding: Dairy is the drop_first baseline.
        if item_type != "Dairy":
            key = f"Item_Type_{item_type}"
            if key in row:
                row[key] = 1.0

        # Outlet identifier: all 10 identifiers were one-hot encoded.
        outlet_key = f"Outlet_Identifier_{outlet_id}"
        if outlet_key in row:
            row[outlet_key] = 1.0

        # Outlet type: Grocery Store is the drop_first baseline.
        if outlet_type != "Grocery Store":
            key = f"Outlet_Type_{outlet_type}"
            if key in row:
                row[key] = 1.0

        input_df = pd.DataFrame([row], columns=FEATURES)

        try:
            scaled = scaler.transform(input_df)
            prediction = float(model.predict(scaled)[0])

            st.markdown(f"""
            <div class="sales-card">
                <div class="sales-label">ESTIMATED OUTLET SALES</div>
                <div class="sales-value">₹ {prediction:,.2f}</div>
                <div class="sales-note">
                    Predicted using the trained KNN Regression model
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="tip">
                💡 <b>Note:</b> This is a Machine Learning estimate.
                Actual sales can vary with demand, location, product pricing,
                seasonality and other business factors.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            r1, r2 = st.columns(2)
            with r1:
                st.markdown(f"""
                <div class="mini mini-blue">
                    <div class="mini-label">🏷️ ITEM TYPE</div>
                    <div class="mini-value">{item_type}</div>
                </div>
                """, unsafe_allow_html=True)

            with r2:
                st.markdown(f"""
                <div class="mini mini-purple">
                    <div class="mini-label">🏪 OUTLET</div>
                    <div class="mini-value">{outlet_id}</div>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("📋 View Processed Model Input"):
                st.dataframe(input_df, use_container_width=True)

        except Exception as e:
            st.error("❌ Prediction failed")
            st.exception(e)

    else:
        st.markdown("""
        <div class="ready">
            <div class="big">🛒</div>
            <h3>Ready for Prediction</h3>
            <p>Enter product and outlet details on the left and click<br>
            <b>Predict Outlet Sales</b> to estimate sales.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        r1, r2 = st.columns(2)
        with r1:
            st.markdown("""
            <div class="mini mini-blue">
                <div style="font-size:23px;">🤖</div>
                <div class="mini-value">KNN Regression</div>
                <div class="mini-label">TRAINED ML MODEL</div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown("""
            <div class="mini mini-purple">
                <div style="font-size:23px;">📊</div>
                <div class="mini-value">35 Features</div>
                <div class="mini-label">PROCESSED INPUTS</div>
            </div>
            """, unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer">
    🛒 <b>BigMart Sales Prediction System</b>
    &nbsp;•&nbsp; Streamlit & Scikit-Learn
    &nbsp;•&nbsp; KNN Regression
</div>
""", unsafe_allow_html=True)
