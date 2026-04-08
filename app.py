import io

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from gradcam import generate_gradcam, overlay_heatmap
from resnet_model import get_resnet_model


st.set_page_config(
    page_title="AI Medical Imaging Assistant",
    page_icon=":hospital:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DEVICE = torch.device("cpu")
PNEUMONIA_THRESHOLD = 0.7
CLASS_NAMES = ["Normal", "Pneumonia"]
MODEL_METRICS = {
    "Accuracy": "80.45%",
    "Recall": "99.49%",
    "F1 Score": "86.41%",
}

PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


@st.cache_resource
def load_model():
    model = get_resnet_model()
    state_dict = torch.load("model.pth", map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def predict_image(model, image):
    input_tensor = PREPROCESS(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    pneumonia_probability = probabilities[1].item()
    normal_probability = probabilities[0].item()
    predicted_index = 1 if pneumonia_probability >= PNEUMONIA_THRESHOLD else 0

    return {
        "label": CLASS_NAMES[predicted_index],
        "normal_probability": normal_probability,
        "pneumonia_probability": pneumonia_probability,
    }


def build_report_text(prediction):
    predicted_label = prediction["label"].upper()
    normal_score = prediction["normal_probability"] * 100
    pneumonia_score = prediction["pneumonia_probability"] * 100
    selected_confidence = (
        prediction["pneumonia_probability"]
        if prediction["label"] == "Pneumonia"
        else prediction["normal_probability"]
    ) * 100
    return f"""AI Medical Imaging Assistant
Explainable Pneumonia Detection Report

Prediction: {predicted_label}
Confidence: {selected_confidence:.2f}%
Normal Confidence: {normal_score:.2f}%
Pneumonia Confidence: {pneumonia_score:.2f}%
Pneumonia Threshold: {PNEUMONIA_THRESHOLD:.2f}

Reference Model Metrics
Accuracy: {MODEL_METRICS['Accuracy']}
Recall: {MODEL_METRICS['Recall']}
F1 Score: {MODEL_METRICS['F1 Score']}

Interpretation Note
Highlighted Grad-CAM regions indicate areas that influenced the model's prediction.
This tool is intended for educational and research support only.
"""


def render_metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probability_bar(label, value, accent_class):
    width = max(6, min(100, int(round(value * 100))))
    st.markdown(
        f"""
        <div class="prob-row">
            <div class="prob-header">
                <span>{label}</span>
                <span>{value * 100:.2f}%</span>
            </div>
            <div class="prob-track">
                <div class="prob-fill {accent_class}" style="width: {width}%;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(86, 136, 255, 0.18), transparent 30%),
            radial-gradient(circle at top right, rgba(30, 215, 176, 0.14), transparent 28%),
            linear-gradient(135deg, #07111f 0%, #0c1628 52%, #111827 100%);
        color: #f3f7ff;
    }
    .block-container {
        max-width: 1280px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .hero-card, .glass-card, .result-card, .footer-card, .upload-shell {
        background: rgba(10, 18, 32, 0.62);
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 24px 60px rgba(0, 0, 0, 0.28);
        backdrop-filter: blur(14px);
        border-radius: 24px;
    }
    .hero-card {
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.2rem;
    }
    .eyebrow {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: rgba(120, 158, 255, 0.12);
        border: 1px solid rgba(120, 158, 255, 0.24);
        color: #c8d7ff;
        font-size: 0.8rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .hero-title {
        margin: 0.9rem 0 0.35rem 0;
        font-size: 2.65rem;
        line-height: 1.05;
        font-weight: 700;
        letter-spacing: -0.04em;
        color: #f8fbff;
    }
    .hero-subtitle {
        margin: 0;
        max-width: 760px;
        color: #a8b7d3;
        font-size: 1.02rem;
        line-height: 1.7;
    }
    .hero-stats {
        display: flex;
        gap: 0.85rem;
        flex-wrap: wrap;
        margin-top: 1.2rem;
    }
    .hero-stat {
        padding: 0.75rem 0.95rem;
        min-width: 150px;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .hero-stat-label {
        color: #8fa1c1;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .hero-stat-value {
        color: #ffffff;
        font-size: 1.2rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #f8fbff;
        margin-bottom: 0.75rem;
    }
    .upload-shell, .glass-card, .result-card {
        padding: 1.25rem;
        min-height: 100%;
    }
    .glass-card {
        margin-top: 1rem;
    }
    .result-card {
        padding: 1.35rem;
        margin-bottom: 1rem;
    }
    .result-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
        border-radius: 999px;
        padding: 0.45rem 0.8rem;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.9rem;
    }
    .chip-normal {
        background: rgba(42, 193, 123, 0.14);
        border: 1px solid rgba(42, 193, 123, 0.28);
        color: #86efac;
    }
    .chip-pneumonia {
        background: rgba(255, 91, 91, 0.16);
        border: 1px solid rgba(255, 91, 91, 0.28);
        color: #fda4af;
    }
    .result-label {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        margin: 0;
        color: #ffffff;
    }
    .result-caption {
        margin-top: 0.35rem;
        color: #97a6c3;
        font-size: 0.96rem;
        line-height: 1.6;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        margin-bottom: 0.8rem;
    }
    .metric-label {
        color: #95a6c6;
        font-size: 0.88rem;
    }
    .metric-value {
        color: #ffffff;
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 0.15rem;
    }
    .prob-row {
        margin-bottom: 1rem;
    }
    .prob-header {
        display: flex;
        justify-content: space-between;
        color: #d7e2f5;
        font-size: 0.92rem;
        margin-bottom: 0.4rem;
    }
    .prob-track {
        width: 100%;
        height: 10px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.08);
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        border-radius: 999px;
    }
    .prob-blue {
        background: linear-gradient(90deg, #6ea8ff 0%, #8dd6ff 100%);
    }
    .prob-red {
        background: linear-gradient(90deg, #ff7b7b 0%, #ffb38a 100%);
    }
    .helper-copy {
        color: #9fb0cd;
        line-height: 1.7;
        font-size: 0.95rem;
    }
    .footer-card {
        margin-top: 1.4rem;
        padding: 1rem 1.2rem;
        text-align: center;
        color: #8fa1c1;
        font-size: 0.9rem;
    }
    [data-testid="stFileUploader"] {
        background: transparent;
        border: 0;
        padding: 0;
    }
    [data-testid="stFileUploader"] section {
        border-radius: 20px;
        border: 1px dashed rgba(255, 255, 255, 0.18);
        background: linear-gradient(180deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.02) 100%);
        padding: 1.25rem 1rem;
    }
    [data-testid="stImage"] img {
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .status-note {
        padding: 0.95rem 1rem;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.07);
        color: #cbd8ef;
        line-height: 1.65;
    }
    div.stDownloadButton > button {
        width: 100%;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: linear-gradient(90deg, rgba(110, 168, 255, 0.18), rgba(33, 208, 176, 0.14));
        color: #f8fbff;
        font-weight: 600;
        padding: 0.75rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <span class="eyebrow">Healthcare AI Platform</span>
        <h1 class="hero-title">AI Medical Imaging Assistant</h1>
        <p class="hero-subtitle">
            Explainable Pneumonia Detection System for chest X-rays. Upload an image to receive a
            model prediction, confidence profile, and Grad-CAM visualization that highlights the
            regions influencing the decision.
        </p>
        <div class="hero-stats">
            <div class="hero-stat">
                <div class="hero-stat-label">Screening Recall</div>
                <div class="hero-stat-value">99.49%</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-label">Held-out Accuracy</div>
                <div class="hero-stat-value">80.45%</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-label">Explainability</div>
                <div class="hero-stat-value">Grad-CAM Ready</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

model_load_error = None
try:
    model = load_model()
except Exception as exc:
    model = None
    model_load_error = str(exc)

left_col, right_col = st.columns([1.05, 1], gap="large")

with left_col:
    st.markdown('<div class="upload-shell">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Upload Chest X-ray</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload a chest X-ray image",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
        help="Supported formats: PNG, JPG, JPEG",
    )

    preview_image = None
    if uploaded_file is not None:
        try:
            preview_image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
            st.image(preview_image, caption="Uploaded X-ray Preview", use_container_width=True)
        except Exception:
            preview_image = None
            st.warning("The uploaded file could not be read as an image. Please try a PNG or JPG file.")
    else:
        st.markdown(
            """
            <div class="status-note">
                Upload a frontal chest X-ray to begin analysis. The app will show the original scan,
                prediction confidence, and an explainability overlay generated from Grad-CAM.
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Prediction Summary</div>', unsafe_allow_html=True)

    if model_load_error:
        st.error(f"Model loading failed: {model_load_error}")
    elif preview_image is None:
        st.markdown(
            """
            <div class="status-note">
                No image uploaded yet. Once an X-ray is provided, the platform will return a
                structured prediction with confidence scores and explainability support.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        with st.spinner("Analyzing X-ray and generating explainability map..."):
            prediction = predict_image(model, preview_image)
            cam, gradcam_source = generate_gradcam(model, preview_image, PREPROCESS, DEVICE)
            overlay = overlay_heatmap(cam, gradcam_source)

        is_pneumonia = prediction["label"] == "Pneumonia"
        chip_class = "chip-pneumonia" if is_pneumonia else "chip-normal"
        chip_icon = "Red Flag" if is_pneumonia else "Green Status"
        confidence = (
            prediction["pneumonia_probability"]
            if is_pneumonia
            else prediction["normal_probability"]
        )
        st.markdown(
            f"""
            <div class="result-chip {chip_class}">{chip_icon}</div>
            <div class="result-label">{prediction["label"].upper()}</div>
            <div class="result-caption">
                Confidence score: <strong>{confidence * 100:.2f}%</strong><br/>
                Threshold for pneumonia flagging: <strong>{PNEUMONIA_THRESHOLD:.2f}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_probability_bar("Normal Confidence", prediction["normal_probability"], "prob-blue")
        render_probability_bar("Pneumonia Confidence", prediction["pneumonia_probability"], "prob-red")

        explainability_enabled = st.toggle("Show Explainability", value=True)
        if explainability_enabled:
            cam_col1, cam_col2 = st.columns(2, gap="medium")
            with cam_col1:
                st.image(preview_image, caption="Original X-ray", use_container_width=True)
            with cam_col2:
                st.image(overlay[:, :, ::-1], caption="Grad-CAM Overlay", use_container_width=True)

        report_text = build_report_text(prediction)
        st.download_button(
            "Download Prediction Report",
            data=report_text,
            file_name="ai_medical_imaging_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

        st.markdown(
            """
            <div class="helper-copy">
                Highlighted regions indicate areas influencing the model's decision. Grad-CAM is
                intended to support interpretation, not replace clinical review.
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
    metric_cols = st.columns(3, gap="small")
    for column, (label, value) in zip(metric_cols, MODEL_METRICS.items()):
        with column:
            render_metric_card(label, value)
    st.markdown(
        """
        <div class="helper-copy">
            These metrics reflect the current trained model snapshot on the held-out test set and
            provide a quick reference for expected screening performance.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="footer-card">
        For educational and research use only. Predictions should be reviewed by qualified clinicians
        and never used as the sole basis for diagnosis.
    </div>
    """,
    unsafe_allow_html=True,
)
