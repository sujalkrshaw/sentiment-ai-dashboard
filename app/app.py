import streamlit as st
import sys
import os
import pickle
import requests
import plotly.graph_objects as go
import pandas as pd

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AI Sentiment Intelligence", page_icon="🤖", layout="wide")

# -------------------------------
# PREMIUM CSS
# -------------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f172a, #020617);
}
.glass {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 25px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.title {
    font-size: 42px;
    font-weight: 700;
    color: white;
    text-align: center;
}
.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 20px;
}
.stButton button {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    border-radius: 10px;
    color: white;
    font-weight: bold;
    height: 50px;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# PATH FIX
# -------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# -------------------------------
# IMPORT PREPROCESS
# -------------------------------
try:
    from src.preprocess import clean_text
except:
    st.error("❌ Missing src/preprocess.py")
    st.stop()

# -------------------------------
# LOAD MODEL (fallback)
# -------------------------------
model = None
model_path = os.path.join(BASE_DIR, "model.pkl")

if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

# -------------------------------
# SESSION
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------
# HEADER
# -------------------------------
st.markdown("<div class='title'>🤖 AI Sentiment Intelligence System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Analyze emotions in text with AI</div>", unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)

# -------------------------------
# LAYOUT
# -------------------------------
col1, col2 = st.columns([2, 1])

# -------------------------------
# INPUT
# -------------------------------
with col1:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    st.subheader("✍️ Enter Text")
    text = st.text_area("", height=150)

    analyze = st.button("🚀 Analyze")

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# RESULT
# -------------------------------
with col2:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    st.subheader("📊 Result")

    prediction = None
    confidence = 0

    if not analyze:
        st.info("Enter text and click Analyze")

    if analyze and text.strip():

        with st.spinner("Analyzing..."):

            processed = clean_text(text)

            try:
                res = requests.post(
                    "http://127.0.0.1:8000/predict",
                    json={"text": processed},
                    timeout=2
                )
                data = res.json()
                prediction = data["prediction"]
                confidence = data["confidence"]

            except:
                if model is None:
                    st.error("❌ No API & no model found")
                    st.stop()

                prediction = model.predict([processed])[0]
                proba = model.predict_proba([processed])[0]
                confidence = max(proba)

        # Premium Result Card
        if prediction == 1:
            st.markdown(f"""
            <div style='padding:15px;border-radius:10px;
            background:linear-gradient(90deg,#16a34a,#22c55e);color:white'>
            😊 Positive Sentiment
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='padding:15px;border-radius:10px;
            background:linear-gradient(90deg,#dc2626,#ef4444);color:white'>
            😞 Negative Sentiment
            </div>
            """, unsafe_allow_html=True)

        # Confidence Indicator
        if confidence > 0.75:
            st.success(f"High Confidence: {confidence:.2f}")
        elif confidence > 0.5:
            st.warning(f"Medium Confidence: {confidence:.2f}")
        else:
            st.error(f"Low Confidence: {confidence:.2f}")

        # Save history
        st.session_state.history.append({
            "text": text,
            "result": "Positive" if prediction == 1 else "Negative",
            "confidence": round(confidence, 2)
        })

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# CHART
# -------------------------------
if analyze and text.strip() and prediction is not None:

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    st.subheader("📈 Confidence Distribution")

    if prediction == 1:
        values = [confidence, 1 - confidence]
    else:
        values = [1 - confidence, confidence]

    fig = go.Figure(data=[
        go.Bar(
            x=["Positive", "Negative"],
            y=values,
            text=[f"{v:.2f}" for v in values],
            textposition='auto'
        )
    ])

    fig.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# HISTORY
# -------------------------------
if st.session_state.history:

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    st.subheader("🕘 Recent Activity")

    df = pd.DataFrame(st.session_state.history[-5:])
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "📥 Download Report",
        df.to_csv(index=False),
        "report.csv"
    )

    if st.button("🧹 Clear History"):
        st.session_state.history = []

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("""
<hr>
<p style='text-align:center;color:gray'>
Built with ❤️ using Streamlit + FastAPI
</p>
""", unsafe_allow_html=True)