import streamlit as st

st.set_page_config(page_title="Voice Phishing Detection", layout="centered")

# --- SESSION STATE ---
if "page" not in st.session_state:
    st.session_state.page = "upload"
if "score" not in st.session_state:
    st.session_state.score = None

# --- CSS ---
st.markdown("""
<style>
.stApp { background-color: #0d0b3d; }

.title {
    text-align: center;
    font-size: 36px;
    font-weight: 600;
    color: white;
    margin-bottom: 30px;
}

[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed #6c63ff !important;
    border-radius: 15px !important;
    padding: 60px !important;
    background-color: #14124f !important;
    text-align: center;
}

[data-testid="stFileUploaderDropzone"] > div {
    border: none !important;
    background: transparent !important;
}

[data-testid="stFileUploader"] label {
    color: #cfcfff !important;
    font-size: 16px;
}

.stButton>button {
    border-radius: 12px;
    padding: 14px 24px;
    background-color: #1a185f;
    color: white;
    border: 1px solid #6c63ff;
}

.stButton>button:hover {
    background-color: #6c63ff;
}
</style>
""", unsafe_allow_html=True)

# --- PAGE 1 ---
if st.session_state.page == "upload":

    st.markdown('<div class="title">Voice Phishing Detection</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload your audio file (.wav)",
        type=["wav"]
    )

    if st.button("Analyze for Vishing"):
        if uploaded_file:
            st.session_state.score = 0.87  # replace with model
            st.session_state.page = "result"
            st.rerun()
        else:
            st.warning("Upload a WAV file first.")


# --- PAGE 2 ---
# --- PAGE 2 ---
elif st.session_state.page == "result":

    st.markdown('<div class="title">Analysis Result</div>', unsafe_allow_html=True)

    score = st.session_state.score
    percent = int(score * 100)

    # --- Risk logic ---
    if percent < 25:
        color = "#00ff9f"
        label = "Low Risk"
        heading = "No Immediate Threat Detected"
        message = "The call appears safe."
    elif percent < 50:
        color = "#ffe600"
        label = "Moderate Risk"
        heading = "Suspicious Activity Detected"
        message = "Stay alert while engaging."
    elif percent < 75:
        color = "#ff8c00"
        label = "High Risk"
        heading = "Potential Fraud Detected"
        message = "Proceed with caution."
    else:
        color = "#ff3b3b"
        label = "Critical"
        heading = "High-Risk Scam Identified"
        message = "Avoid engaging with the caller."

    # --- Layout (center + push legend right) ---
    left, center, right = st.columns([1, 2, 1.3])

    # --- CENTER CONTENT ---
    with center:

        # --- Ring (subtle glow) ---
        st.markdown(f"""
<div style="display:flex; justify-content:center; align-items:center; margin-top:20px;">
    <div style="
        width:240px;
        height:240px;
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
        box-shadow: 0 0 18px {color}66;
    ">
        <div style="
            width:180px;
            height:180px;
            border-radius:50%;
            background:#14124f;
            display:flex;
            flex-direction:column;
            align-items:center;
            justify-content:center;
            color:white;
            box-shadow: inset 0 0 12px #00000088;
        ">
            <div style="font-size:34px; font-weight:700;">
                {percent}%
            </div>
            <div style="font-size:14px; opacity:0.7;">
                {label}
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

        # --- CLEAN MESSAGE (NO HTML BLOCK BUG) ---
        st.markdown(
            f"<h3 style='text-align:center; color:{color}; margin-top:30px;'>{heading}</h3>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<p style='text-align:center; font-size:20px; color:#e0e0ff; margin-bottom:60px;'>"
            f"This call is <b>{percent}% likely to be a scam</b>. {message}"
            f"</p>",
            unsafe_allow_html=True
        )

        # --- BUTTON ---
        if st.button("Analyze Another File"):
            st.session_state.page = "upload"
            st.session_state.score = None
            st.rerun()

# --- RIGHT LEGEND ---
    with right:
        st.markdown(
            "<div style='margin-top:120px; margin-left:40px;'></div>",
            unsafe_allow_html=True
        )

        st.markdown("<span style='color:#00ff9f; font-size:22px;'>●</span> Low Risk", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("<span style='color:#ffe600; font-size:22px;'>●</span> Moderate Risk", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("<span style='color:#ff8c00; font-size:22px;'>●</span> High Risk", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("<span style='color:#ff3b3b; font-size:22px;'>●</span> Critical", unsafe_allow_html=True)