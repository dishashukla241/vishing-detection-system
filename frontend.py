import streamlit as st
from predict import predict
import json
import os

st.set_page_config(page_title="Voice Phishing Detection", layout="centered")

# -------------------------
# LOAD METRICS (SAFE)
# -------------------------
metrics = None
if os.path.exists("models/metrics.json"):
    with open("models/metrics.json") as f:
        metrics = json.load(f)

# -------------------------
# SESSION STATE
# -------------------------
if "page" not in st.session_state:
    st.session_state.page = "upload"
if "score" not in st.session_state:
    st.session_state.score = None
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "verdict" not in st.session_state:
    st.session_state.verdict = ""

# -------------------------
# CSS
# -------------------------
st.markdown("""
<style>
.stApp { background-color: #0d0b3d; }

.title {
    text-align: center;
    font-size: 40px;
    font-weight: 700;
    color: white;
    margin-bottom: 30px;
}

.center {
    text-align: center;
}

.metric-box {
    background-color: #14124f;
    padding: 10px;
    border-radius: 10px;
    text-align: center;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ------------------ PAGE 1 ------------------
if st.session_state.page == "upload":

    st.markdown('<div class="title">Voice Phishing Detection</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

    if st.button("Analyze"):
        if uploaded_file:
            result = predict(uploaded_file)

            st.session_state.score = result["score"] / 100
            st.session_state.transcript = result["transcript"]
            st.session_state.verdict = result["verdict"]

            st.session_state.page = "result"
            st.rerun()
        else:
            st.warning("Upload a file first")

# ------------------ PAGE 2 ------------------
elif st.session_state.page == "result":

    percent = int(st.session_state.score * 100)

    # Risk levels
    if percent < 25:
        color = "#00ff9f"
        label = "Safe"
    elif percent < 50:
        color = "#ffe600"
        label = "Caution"
    elif percent < 75:
        color = "#ff8c00"
        label = "Suspicious"
    else:
        color = "#ff3b3b"
        label = "Likely Scam"

    st.markdown('<div class="title">Analysis Result</div>', unsafe_allow_html=True)

    # ------------------ GAUGE ------------------
    st.markdown(f"""
<div style="display:flex; justify-content:center; align-items:center; margin-top:30px;">
    <div style="
        width:260px;
        height:260px;
        border-radius:50%;
        background: conic-gradient(
            #00ff9f 0%,
            #ffe600 25%,
            #ff8c00 50%,
            #ff3b3b 75%,
            #ff3b3b {percent}%,
            #2a276a {percent}% 100%
        );
        display:flex;
        align-items:center;
        justify-content:center;
        box-shadow: 0 0 20px {color}55;
    ">
        <div style="
            width:190px;
            height:190px;
            border-radius:50%;
            background:#14124f;
            display:flex;
            flex-direction:column;
            align-items:center;
            justify-content:center;
            color:white;
        ">
            <div style="font-size:36px; font-weight:700;">
                {percent}%
            </div>
            <div style="font-size:14px; opacity:0.7;">
                {label}
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

    # ------------------ VERDICT ------------------
    st.markdown(
        f"<h3 style='text-align:center; color:{color}; margin-top:30px;'>⚠️ {label}</h3>",
        unsafe_allow_html=True
    )

    # ------------------ TRANSCRIPT ------------------
 

    # ------------------ METRICS ------------------
    if metrics:
        st.markdown("###  Model Performance")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Audio")
            st.write(f"Accuracy: {metrics['audio']['accuracy']:.2f}")
            st.write(f"Precision: {metrics['audio']['precision']:.2f}")
            st.write(f"Recall: {metrics['audio']['recall']:.2f}")
            st.write(f"F1: {metrics['audio']['f1']:.2f}")

        with col2:
            st.markdown("#### Text")
            st.write(f"Accuracy: {metrics['text']['accuracy']:.2f}")
            st.write(f"Precision: {metrics['text']['precision']:.2f}")
            st.write(f"Recall: {metrics['text']['recall']:.2f}")
            st.write(f"F1: {metrics['text']['f1']:.2f}")

        with col3:
            st.markdown("#### Fusion")
            st.write(f"Accuracy: {metrics['fusion']['accuracy']:.2f}")
            st.write(f"Precision: {metrics['fusion']['precision']:.2f}")
            st.write(f"Recall: {metrics['fusion']['recall']:.2f}")
            st.write(f"F1: {metrics['fusion']['f1']:.2f}")

    # ------------------ LEGEND ------------------
    st.markdown("""
    <div style='margin-top:30px; text-align:center;'>

    <span style='color:#00ff9f;'>●</span> Safe  
    &nbsp;&nbsp;
    <span style='color:#ffe600;'>●</span> Caution  
    &nbsp;&nbsp;
    <span style='color:#ff8c00;'>●</span> Suspicious  
    &nbsp;&nbsp;
    <span style='color:#ff3b3b;'>●</span> Likely Scam  

    </div>
    """, unsafe_allow_html=True)

    # ------------------ BACK BUTTON ------------------
    st.markdown("<div class='center'>", unsafe_allow_html=True)
    if st.button("Back"):
        st.session_state.page = "upload"
        st.session_state.score = None
        st.session_state.transcript = ""
        st.session_state.verdict = ""
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)