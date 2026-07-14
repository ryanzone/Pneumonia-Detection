"""
AI Chest X-Ray Classifier -- Production Streamlit Application
Architecture: ResNet18 fine-tuned binary classifier (Normal / Pneumonia)
"""

import os
import io
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image, ImageEnhance, UnidentifiedImageError

from model_arch import PneumoniaModel, get_test_transform

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
APP_TITLE: str = "AI Chest X-Ray Classifier"
APP_SUBTITLE: str = "Deep-learning diagnostic support powered by ResNet18"
MODEL_PATH: str = "best_model.pth"
SAMPLES_DIR: str = "samples"
CLASS_NAMES: list[str] = ["Normal", "Pneumonia"]
MAX_UPLOAD_MB: int = 10
MAX_HISTORY: int = 50
DEBUG: bool = False

SAMPLE_IMAGES: dict[str, str] = {
    "Normal  --  IM-0049-0001": "samples/IM-0049-0001.jpeg",
    "Normal  --  IM-0050-0001": "samples/IM-0050-0001.jpeg",
    "Normal  --  IM-0059-0001": "samples/IM-0059-0001.jpeg",
    "Normal  --  IM-0061-0001": "samples/IM-0061-0001.jpeg",
    "Pneumonia  --  person38_virus_84": "samples/person38_virus_84.jpeg",
    "Pneumonia  --  person39_virus_85": "samples/person39_virus_85.jpeg",
    "Pneumonia  --  person40_virus_87": "samples/person40_virus_87.jpeg",
    "Pneumonia  --  person41_virus_88": "samples/person41_virus_88.jpeg",
}

# =========================
# Color Palette
# =========================
ACCENT = "#58A6FF"
ACCENT_LIGHT = "rgba(88,166,255,0.15)"

SUCCESS_COLOR = "#3FB950"
DANGER_COLOR = "#F85149"

NEUTRAL_BG = "#0D1117"
CARD_BG = "#161B22"
SIDEBAR_BG = "#010409"

BORDER_COLOR = "#30363D"

TEXT_PRIMARY = "#F0F6FC"
TEXT_SECONDARY = "#C9D1D9"
TEXT_MUTED = "rgba(201,209,217,0.60)"

INPUT_BG = "#21262D"
HOVER_BG = "#30363D"
DISABLED_BG = "#161B22"

SHADOW = "rgba(0,0,0,0.40)"

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>X</text></svg>",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
def inject_custom_css() -> None:
    """Inject custom CSS to style the Streamlit app as a modern medical dashboard."""
    css_content = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');

        /* ---------- Global ---------- */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: __TEXT_PRIMARY__ !important;
        }
        .stApp {
            background-color: __NEUTRAL_BG__;
        }
        
        /* Force markdown text colors */
        .stMarkdown p, .stMarkdown li, .stMarkdown span {
            color: __TEXT_SECONDARY__ !important;
        }
        .stMarkdown strong {
            color: __TEXT_PRIMARY__ !important;
        }

        h1, h2, h3, h4, h5, h6, .app-header h1, .section-header {
            font-family: 'Outfit', sans-serif !important;
            color: __TEXT_PRIMARY__ !important;
        }

        /* ---------- Hide Streamlit chrome ---------- */
        #MainMenu, footer, header {visibility: hidden;}

        /* ---------- Kill Streamlit default purple/blue everywhere ---------- */
        .stSlider [data-baseweb="slider"] div[role="slider"] {
            background-color: __ACCENT__ !important;
            border-color: __ACCENT__ !important;
            color: #FFFFFF !important;
        }
        .stSlider [data-baseweb="slider"] div[data-testid="stTickBar"] > div {
            background-color: __ACCENT__ !important;
        }
        .stSlider div[data-baseweb="slider"] > div > div > div[role="progressbar"] {
            background-color: __ACCENT__ !important;
        }
        .stSlider div[data-baseweb="slider"] > div > div {
            background-color: __INPUT_BG__ !important;
        }
        .stSlider div[data-baseweb="slider"] > div > div > div {
            background-color: __ACCENT__ !important;
        }
        .stSlider, .stSlider * {
            color: __TEXT_PRIMARY__ !important;
        }
        .stSlider [data-testid="stThumbValue"] {
            font-weight: 700;
        }

        /* File uploader */
        [data-testid="stFileUploader"] label {
            color: __TEXT_PRIMARY__ !important;
            font-weight: 500;
        }
        [data-testid="stFileUploader"] section {
            border-color: __BORDER_COLOR__ !important;
            border-style: dashed !important;
            border-width: 2px !important;
            border-radius: 12px !important;
            background-color: __NEUTRAL_BG__ !important;
            transition: all 0.2s ease !important;
            padding: 1.5rem !important;
        }
        [data-testid="stFileUploader"] section:hover {
            border-color: __ACCENT__ !important;
            background-color: rgba(79, 70, 229, 0.02) !important;
        }

        /* Select box */
        .stSelectbox [data-baseweb="select"] > div {
            border-color: __BORDER_COLOR__ !important;
            background-color: __CARD_BG__ !important;
            border-radius: 10px !important;
        }
        .stSelectbox [data-baseweb="select"] > div:focus-within {
            border-color: __ACCENT__ !important;
            box-shadow: 0 0 0 1px __ACCENT__ !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: none !important;
            background-color: __NEUTRAL_BG__;
            padding: 6px;
            border-radius: 12px;
            margin-bottom: 1.25rem;
        }
        .stTabs [data-baseweb="tab"], .stTabs button[role="tab"], .stTabs [data-testid="stTab"] {
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            font-size: 0.82rem;
            padding: 0.5rem 1.25rem;
            color: __TEXT_SECONDARY__ !important;
            border: 1px solid transparent !important;
            border-radius: 8px !important;
            background-color: transparent !important;
            margin-bottom: 0;
            transition: all 0.2s ease;
        }
        .stTabs [data-baseweb="tab"] *, .stTabs button[role="tab"] *, .stTabs [data-testid="stTab"] * {
            color: __TEXT_SECONDARY__ !important;
        }
        .stTabs [aria-selected="true"], .stTabs button[role="tab"][aria-selected="true"], .stTabs [data-testid="stTab"][aria-selected="true"] {
            color: __ACCENT__ !important;
            background-color: __CARD_BG__ !important;
            border-color: __BORDER_COLOR__ !important;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05) !important;
        }
        .stTabs [aria-selected="true"] *, .stTabs button[role="tab"][aria-selected="true"] *, .stTabs [data-testid="stTab"][aria-selected="true"] * {
            color: __ACCENT__ !important;
        }
        .stTabs [data-baseweb="tab-highlight"] {
            display: none !important;
        }
        .stTabs [data-baseweb="tab-border"] {
            display: none !important;
        }

        /* Buttons */
        .stButton > button {
            font-family: 'Inter', sans-serif;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.82rem;
            transition: all 0.2s ease;
            border: 1px solid __BORDER_COLOR__;
            color: __TEXT_PRIMARY__;
            background-color: __CARD_BG__;
            padding: 0.5rem 1rem;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        .stButton > button:hover {
            border-color: __ACCENT__;
            background-color: __ACCENT__;
            color: #FFFFFF;
            box-shadow: 0 4px 10px rgba(79, 70, 229, 0.2);
        }
        .stButton > button:active,
        .stButton > button:focus {
            border-color: __ACCENT__ !important;
            color: __TEXT_PRIMARY__ !important;
            box-shadow: none !important;
            background-color: __ACCENT_LIGHT__ !important;
        }

        .stDownloadButton > button {
            font-family: 'Inter', sans-serif;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.82rem;
            background-color: __ACCENT__;
            color: #FFFFFF;
            border: 1px solid __ACCENT__;
            padding: 0.5rem 1rem;
            transition: all 0.2s ease;
            box-shadow: 0 2px 4px rgba(79, 70, 229, 0.15);
        }
        .stDownloadButton > button:hover {
            background-color: #4338CA;
            border-color: #4338CA;
            color: #FFFFFF;
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
        }

        /* Dataframe */
        [data-testid="stDataFrame"] {
            border: 1px solid __BORDER_COLOR__;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        }

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {
            background-color: __SIDEBAR_BG__;
            border-right: 1px solid __BORDER_COLOR__;
        }
        section[data-testid="stSidebar"] .stMarkdown h2 {
            font-size: 0.75rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: __TEXT_SECONDARY__;
            margin-top: 1.75rem;
            margin-bottom: 0.75rem;
            border-bottom: none !important;
        }

        /* ---------- Card Utility ---------- */
        .card {
            background: __CARD_BG__;
            border: 1px solid rgba(226, 232, 240, 0.8);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 10px 15px -3px rgba(15, 23, 42, 0.04), 0 4px 6px -4px rgba(15, 23, 42, 0.04);
        }

        /* ---------- Header ---------- */
        .app-header {
            text-align: center;
            padding: 2.25rem 1.5rem 1.75rem;
            background: linear-gradient(135deg, rgba(79, 70, 229, 0.04) 0%, rgba(59, 130, 246, 0.04) 100%);
            border-radius: 20px;
            margin-bottom: 2rem;
            border: 1px solid rgba(79, 70, 229, 0.06);
            box-shadow: inset 0 0 12px rgba(255, 255, 255, 0.5);
        }
        .app-header h1 {
            font-size: 2.5rem;
            font-weight: 800;
            color: __TEXT_PRIMARY__;
            margin-bottom: 0.5rem;
            letter-spacing: -0.03em;
            background: linear-gradient(to right, __ACCENT__, #3B82F6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .app-header p {
            font-size: 1.05rem;
            color: __TEXT_SECONDARY__;
            margin: 0;
            font-weight: 400;
        }

        /* ---------- Status Pills ---------- */
        .status-row {
            display: flex;
            justify-content: center;
            gap: 0.75rem;
            flex-wrap: wrap;
            margin-top: 1.25rem;
            margin-bottom: 0.25rem;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: __ACCENT_LIGHT__;
            border: 1px solid rgba(79, 70, 229, 0.06);
            border-radius: 999px;
            padding: 0.4rem 1rem;
            font-size: 0.78rem;
            font-weight: 600;
            color: __TEXT_PRIMARY__;
            box-shadow: 0 1px 2px rgba(0,0,0,0.02);
        }
        .status-pill .dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
        }
        .dot-green { background: __SUCCESS_COLOR__; }
        .dot-neutral { background: #94A3B8; }
        .dot-amber { background: #F59E0B; }
        .dot-blue { background: __ACCENT__; }

        /* ---------- Prediction Card ---------- */
        .prediction-card {
            text-align: center;
            padding: 2.25rem 1.5rem;
            border-radius: 18px;
            margin-bottom: 1.25rem;
            position: relative;
            overflow: hidden;
            box-shadow: 0 10px 25px -5px rgba(0,0,0,0.02);
        }
        .prediction-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 4px;
        }
        .prediction-card.normal {
            background: linear-gradient(185deg, rgba(159, 192, 216, 0.1) 0%, __CARD_BG__ 100%);
            border: 1px solid rgba(159, 192, 216, 0.25);
        }
        .prediction-card.normal::before {
            background: __SUCCESS_COLOR__;
        }
        .prediction-card.pneumonia {
            background: linear-gradient(185deg, rgba(204, 0, 0, 0.1) 0%, __CARD_BG__ 100%);
            border: 1px solid rgba(204, 0, 0, 0.25);
        }
        .prediction-card.pneumonia::before {
            background: __DANGER_COLOR__;
        }
        .prediction-label {
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: __TEXT_SECONDARY__;
            margin-bottom: 0.35rem;
        }
        .prediction-value {
            font-size: 2.25rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin-bottom: 0.25rem;
        }
        .prediction-card.normal .prediction-value { color: __SUCCESS_COLOR__; }
        .prediction-card.pneumonia .prediction-value { color: __DANGER_COLOR__; }

        .confidence-pct {
            font-size: 3.5rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            margin: 0.25rem 0;
        }
        .confidence-pct.normal { color: __SUCCESS_COLOR__; }
        .confidence-pct.pneumonia { color: __DANGER_COLOR__; }

        .conf-sublabel {
            font-size: 0.75rem;
            color: __TEXT_SECONDARY__;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .meta-row {
            display: flex;
            justify-content: center;
            gap: 1.25rem;
            margin-top: 1.25rem;
            flex-wrap: wrap;
            border-top: 1px solid __BORDER_COLOR__;
            padding-top: 1rem;
        }
        .meta-item {
            font-size: 0.78rem;
            color: __TEXT_SECONDARY__;
        }
        .meta-item strong {
            color: __TEXT_PRIMARY__;
        }

        /* ---------- Upload area ---------- */
        .upload-section {
            text-align: center;
            margin-bottom: 1.5rem;
        }

        /* ---------- Section headers ---------- */
        .section-header {
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: __TEXT_SECONDARY__;
            margin-bottom: 1.25rem;
            padding-left: 10px;
            border-left: 3px solid __ACCENT__;
            border-bottom: none !important;
        }

        /* ---------- Image captions ---------- */
        .img-caption {
            text-align: center;
            font-size: 0.8rem;
            color: __TEXT_SECONDARY__;
            margin-top: 0.75rem;
            font-weight: 500;
        }

        /* ---------- Disclaimer ---------- */
        .disclaimer {
            background: __CARD_BG__;
            border: 1px solid __BORDER_COLOR__;
            border-left: 4px solid #94A3B8;
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            margin-top: 2.5rem;
            font-size: 0.8rem;
            color: __TEXT_SECONDARY__;
            line-height: 1.6;
        }
        .disclaimer strong {
            display: block;
            margin-bottom: 0.35rem;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: __TEXT_PRIMARY__;
        }

        /* ---------- Error card ---------- */
        .error-card {
            background: rgba(204, 0, 0, 0.1);
            border: 1px solid rgba(204, 0, 0, 0.2);
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            color: __DANGER_COLOR__;
            font-size: 0.88rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 4px 6px -1px rgba(204, 0, 0, 0.05);
        }
        .error-card strong {
            display: block;
            margin-bottom: 0.35rem;
            font-weight: 700;
        }

        /* ---------- Sidebar metric card ---------- */
        .sidebar-metric {
            background: __NEUTRAL_BG__;
            border-radius: 10px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.6rem;
            border: 1px solid rgba(226, 232, 240, 0.6);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .sidebar-metric .label {
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: __TEXT_SECONDARY__;
        }
        .sidebar-metric .value {
            font-size: 0.88rem;
            font-weight: 700;
            color: __TEXT_PRIMARY__;
            margin-top: 0;
        }

        /* ---------- Plotly chart container ---------- */
        .stPlotlyChart {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(226, 232, 240, 0.4);
        }
        </style>
        """
    css_content = css_content.replace("__ACCENT__", ACCENT)
    css_content = css_content.replace("__ACCENT_LIGHT__", ACCENT_LIGHT)
    css_content = css_content.replace("__SUCCESS_COLOR__", SUCCESS_COLOR)
    css_content = css_content.replace("__DANGER_COLOR__", DANGER_COLOR)
    css_content = css_content.replace("__NEUTRAL_BG__", NEUTRAL_BG)
    css_content = css_content.replace("__CARD_BG__", CARD_BG)
    css_content = css_content.replace("__SIDEBAR_BG__", SIDEBAR_BG)
    css_content = css_content.replace("__BORDER_COLOR__", BORDER_COLOR)
    css_content = css_content.replace("__TEXT_PRIMARY__", TEXT_PRIMARY)
    css_content = css_content.replace("__TEXT_SECONDARY__", TEXT_SECONDARY)
    css_content = css_content.replace("__TEXT_MUTED__", TEXT_MUTED)
    css_content = css_content.replace("__INPUT_BG__", INPUT_BG)
    css_content = css_content.replace("__HOVER_BG__", HOVER_BG)
    css_content = css_content.replace("__DISABLED_BG__", DISABLED_BG)
    css_content = css_content.replace("__SHADOW__", SHADOW)
    
    st.markdown(css_content, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_path: str, device: torch.device) -> Optional[PneumoniaModel]:
    """Load the trained PneumoniaModel from disk."""
    if not Path(model_path).is_file():
        return None
    try:
        model = PneumoniaModel(num_classes=len(CLASS_NAMES))
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        return model
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_transform():
    """Return the cached test-time transform pipeline."""
    return get_test_transform()


@st.cache_data(show_spinner=False)
def load_sample_image(path: str) -> Optional[Image.Image]:
    """Load and cache a sample image from disk."""
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    """Select the best available compute device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_gpu_info(device: torch.device) -> dict[str, str]:
    """Return GPU metadata when available."""
    info: dict[str, str] = {"device": device.type.upper()}
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        mem_alloc = torch.cuda.memory_allocated(0) / (1024**3)
        info["vram_total"] = f"{mem_total:.1f} GB"
        info["vram_used"] = f"{mem_alloc:.2f} GB"
    return info


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def apply_enhancements(
    img: Image.Image,
    brightness: float,
    contrast: float,
    sharpness: float,
) -> Image.Image:
    """Return an enhanced copy of *img*. Original is never mutated."""
    out = img.copy()
    if brightness != 1.0:
        out = ImageEnhance.Brightness(out).enhance(brightness)
    if contrast != 1.0:
        out = ImageEnhance.Contrast(out).enhance(contrast)
    if sharpness != 1.0:
        out = ImageEnhance.Sharpness(out).enhance(sharpness)
    return out


def validate_uploaded_file(uploaded_file) -> tuple[bool, str]:
    """Validate an uploaded file before opening as an image."""
    if uploaded_file is None:
        return False, "No file provided."
    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        return False, f"File exceeds the {MAX_UPLOAD_MB} MB limit ({size_mb:.1f} MB)."
    return True, ""


def open_image_safely(source) -> tuple[Optional[Image.Image], str]:
    """Attempt to open an image; return (image, error_message)."""
    try:
        img = Image.open(source).convert("RGB")
        img.load()
        return img, ""
    except UnidentifiedImageError:
        return None, "The file does not appear to be a valid image."
    except Exception as exc:
        return None, f"Could not read image: {exc}"


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_inference(
    model: PneumoniaModel,
    image: Image.Image,
    transform,
    device: torch.device,
) -> tuple[str, float, list[float], float]:
    """
    Run model inference on the ORIGINAL image.

    Returns:
        prediction: class name
        confidence: probability of predicted class
        probabilities: list of class probabilities
        latency_ms: inference time in milliseconds
    """
    tensor = transform(image).unsqueeze(0).to(device)
    start = time.perf_counter()
    try:
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA out of memory. The image could not be processed on GPU. "
                "Try restarting the application or using a smaller image."
            ) from exc
        raise
    latency_ms = (time.perf_counter() - start) * 1000
    idx = probs.argmax().item()
    prediction = CLASS_NAMES[idx]
    confidence = probs[idx].item()
    probabilities = [p.item() for p in probs]
    return prediction, confidence, probabilities, latency_ms


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_markdown_report(
    filename: str,
    prediction: str,
    confidence: float,
    probabilities: list[float],
    latency_ms: float,
    device: torch.device,
) -> str:
    """Generate a Markdown diagnostic report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gpu_label = torch.cuda.get_device_name(0) if device.type == "cuda" else "N/A"
    lines = [
        "# Diagnostic Classification Report",
        "",
        f"Generated: {now}",
        "",
        "## Case Details",
        f"- Filename: {filename}",
        f"- Model: ResNet18 (PneumoniaModel)",
        f"- Device: {device.type.upper()}",
        f"- GPU: {gpu_label}",
        f"- Inference Latency: {latency_ms:.1f} ms",
        "",
        "## Classification Result",
        f"- Diagnosis: {prediction}",
        f"- Confidence: {confidence * 100:.2f}%",
        "",
        "## Class Probabilities",
    ]
    for i, name in enumerate(CLASS_NAMES):
        lines.append(f"- {name}: {probabilities[i] * 100:.2f}%")
    lines += [
        "",
        "---",
        "",
        "Disclaimer: This report is generated by a prototype deep-learning system "
        "intended for research and educational purposes only. It must not be used "
        "as a substitute for professional medical diagnosis.",
    ]
    return "\n".join(lines)


def generate_plaintext_report(
    filename: str,
    prediction: str,
    confidence: float,
    probabilities: list[float],
    latency_ms: float,
    device: torch.device,
) -> str:
    """Generate a plain-text diagnostic report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gpu_label = torch.cuda.get_device_name(0) if device.type == "cuda" else "N/A"
    sep = "=" * 52
    lines = [
        sep,
        "  DIAGNOSTIC CLASSIFICATION REPORT",
        sep,
        f"  Generated : {now}",
        "",
        "  CASE DETAILS",
        f"  Filename  : {filename}",
        f"  Model     : ResNet18 (PneumoniaModel)",
        f"  Device    : {device.type.upper()}",
        f"  GPU       : {gpu_label}",
        f"  Latency   : {latency_ms:.1f} ms",
        "",
        "  CLASSIFICATION RESULT",
        f"  Diagnosis : {prediction}",
        f"  Confidence: {confidence * 100:.2f}%",
        "",
        "  CLASS PROBABILITIES",
    ]
    for i, name in enumerate(CLASS_NAMES):
        lines.append(f"  {name:12s}: {probabilities[i] * 100:.2f}%")
    lines += [
        "",
        sep,
        "  DISCLAIMER",
        "  This report is generated by a prototype deep-learning",
        "  system for research and educational purposes only.",
        "  It must not be used for clinical diagnosis.",
        sep,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------
def build_probability_chart(probabilities: list[float], prediction: str) -> go.Figure:
    """Build a horizontal bar chart of class probabilities."""
    colors = []
    for i, name in enumerate(CLASS_NAMES):
        if name == prediction:
            colors.append(SUCCESS_COLOR if name == "Normal" else DANGER_COLOR)
        else:
            colors.append("#CBD5E1")

    fig = go.Figure(
        go.Bar(
            x=[p * 100 for p in probabilities],
            y=CLASS_NAMES,
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(width=0),
                cornerradius=6,
            ),
            text=[f"{p * 100:.1f}%" for p in probabilities],
            textposition="outside",
            textfont=dict(family="Inter", size=13, weight=600),
            hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis=dict(
            range=[0, 110],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            fixedrange=True,
        ),
        yaxis=dict(
            autorange="reversed",
            showgrid=False,
            zeroline=False,
            tickfont=dict(family="Inter", size=13, weight=600, color=TEXT_PRIMARY),
            fixedrange=True,
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=40, t=10, b=10),
        height=120,
        bargap=0.35,
    )
    return fig


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------
def render_header(device: torch.device, latency_ms: Optional[float] = None) -> None:
    """Render the page header with status pills."""
    model_label = "ResNet18"
    device_label = device.type.upper()
    latency_label = f"{latency_ms:.0f} ms" if latency_ms else "--"

    if device.type == "cuda":
        dot_cls = "dot-green"
    else:
        dot_cls = "dot-amber"

    st.markdown(
        f"""
        <div class="app-header">
            <h1>{APP_TITLE}</h1>
            <p>{APP_SUBTITLE}</p>
            <div class="status-row">
                <span class="status-pill">
                    <span class="dot {dot_cls}"></span> Device: {device_label}
                </span>
                <span class="status-pill">
                    <span class="dot dot-blue"></span> Model: {model_label}
                </span>
                <span class="status-pill">
                    <span class="dot dot-blue"></span> Inference: {latency_label}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_card(
    prediction: str,
    confidence: float,
    latency_ms: float,
    device: torch.device,
) -> None:
    """Render the large centered prediction card."""
    css_cls = "normal" if prediction == "Normal" else "pneumonia"
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    st.markdown(
        f"""
        <div class="card prediction-card {css_cls}">
            <div class="prediction-label">Diagnosis</div>
            <div class="prediction-value">{prediction.upper()}</div>
            <div class="confidence-pct {css_cls}">{confidence * 100:.1f}%</div>
            <div style="font-size:0.8rem;color:{TEXT_SECONDARY};">Confidence</div>
            <div class="meta-row">
                <span class="meta-item"><strong>{latency_ms:.1f} ms</strong> latency</span>
                <span class="meta-item"><strong>{device.type.upper()}</strong> device</span>
                <span class="meta-item"><strong>{gpu_name}</strong></span>
                <span class="meta-item"><strong>ResNet18</strong> model</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_metrics_group(title: str, metrics: dict[str, str]) -> None:
    """Render a group of metrics in a single styled card block."""
    items_html = ""
    for idx, (label, val) in enumerate(metrics.items()):
        border_style = "border-bottom: 1px solid rgba(226, 232, 240, 0.4); padding-bottom: 0.6rem; margin-bottom: 0.6rem;" if idx < len(metrics) - 1 else ""
        items_html += f"""
        <div style="display:flex; justify-content:space-between; align-items:center; {border_style}">
            <span style="font-size:0.7rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; color:{TEXT_SECONDARY};">{label}</span>
            <span style="font-size:0.8rem; font-weight:700; color:{TEXT_PRIMARY};">{val}</span>
        </div>
        """
    st.markdown(
        f"""
        <div class="card" style="padding: 1rem 1.25rem; margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.02); border-radius: 12px;">
            <div style="font-size: 0.72rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em; color: {TEXT_SECONDARY}; margin-bottom: 0.75rem; border-bottom: 1px solid rgba(226, 232, 240, 0.6); padding-bottom: 0.35rem;">
                {title}
            </div>
            {items_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(text: str) -> None:
    """Render a styled section header."""
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


def render_error(title: str, detail: str, show_trace: bool = False) -> None:
    """Render a professional error card."""
    trace_html = ""
    if show_trace and DEBUG:
        trace_html = (
            f"<pre style='font-size:0.75rem;margin-top:0.75rem;white-space:pre-wrap;'>"
            f"{traceback.format_exc()}</pre>"
        )
    st.markdown(
        f"""
        <div class="error-card">
            <strong>{title}</strong>
            {detail}{trace_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer() -> None:
    """Render the bottom disclaimer banner."""
    st.markdown(
        """
        <div class="disclaimer">
            <strong>Medical Disclaimer</strong>
            This application is a research prototype and is intended for educational and
            demonstration purposes only. It must not be used as a substitute for professional
            medical advice, diagnosis, or treatment. Always consult a qualified healthcare
            provider for clinical decision-making.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    """Initialize all session-state keys."""
    if "history" not in st.session_state:
        st.session_state.history = []
    if "selected_sample" not in st.session_state:
        st.session_state.selected_sample = None


def add_history_entry(
    filename: str,
    prediction: str,
    confidence: float,
    latency_ms: float,
    device_label: str,
) -> None:
    """Append a classification entry to session history (deduplicates consecutive runs)."""
    entry = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Filename": filename,
        "Prediction": prediction,
        "Confidence": f"{confidence * 100:.2f}%",
        "Latency (ms)": f"{latency_ms:.1f}",
        "Device": device_label,
    }
    history: list[dict] = st.session_state.history
    if history and history[-1]["Filename"] == filename and history[-1]["Prediction"] == prediction:
        return
    history.append(entry)
    if len(history) > MAX_HISTORY:
        st.session_state.history = history[-MAX_HISTORY:]


def get_history_df() -> pd.DataFrame:
    """Return the classification history as a DataFrame."""
    if not st.session_state.history:
        return pd.DataFrame()
    return pd.DataFrame(st.session_state.history)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar(device: torch.device, model_loaded: bool) -> None:
    """Build the sidebar: device info, GPU info, model info, clear history, about."""
    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding: 0.5rem 0; margin-bottom: 1.5rem; text-align: center;">
                <span style="font-family: 'Outfit', sans-serif; font-size: 1.15rem; font-weight: 800; color: {ACCENT}; letter-spacing: -0.01em;">
                    CLINICAL DASHBOARD
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        gpu_info = get_gpu_info(device)
        hardware_metrics = {
            "Compute Device": gpu_info["device"]
        }
        if device.type == "cuda":
            hardware_metrics["GPU Name"] = gpu_info.get("gpu_name", "N/A")
            hardware_metrics["Total VRAM"] = gpu_info.get("vram_total", "N/A")
            hardware_metrics["VRAM in Use"] = gpu_info.get("vram_used", "N/A")
        else:
            hardware_metrics["GPU Status"] = "No GPU / Running on CPU"
        
        render_sidebar_metrics_group("Hardware Details", hardware_metrics)

        model_metrics = {}
        if model_loaded:
            model_metrics["Architecture"] = "ResNet18"
            model_metrics["Classes"] = " / ".join(CLASS_NAMES)
            model_metrics["Weights File"] = MODEL_PATH
        else:
            model_metrics["Status"] = "Model not loaded"
            
        render_sidebar_metrics_group("Model Details", model_metrics)

        count = len(st.session_state.history)
        history_metrics = {
            "Recorded Entries": str(count)
        }
        render_sidebar_metrics_group("Session Statistics", history_metrics)

        if count > 0:
            if st.button("Clear History", use_container_width=True):
                st.session_state.history = []
                st.rerun()

        st.markdown(
            f"""
            <div class="card" style="padding: 1rem; border-radius: 12px; font-size: 0.76rem; color: {TEXT_SECONDARY}; line-height: 1.6; background-color: rgba(226, 232, 240, 0.2); border: 1px solid rgba(226, 232, 240, 0.4);">
                <strong style="color: {TEXT_PRIMARY}; display: block; margin-bottom: 0.25rem;">AI Chest X-Ray Classifier</strong>
                Built with Streamlit & PyTorch<br>
                Model: ResNet18 Binary Classifier<br>
                Diagnostic Support System v1.0.0
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry point for the Streamlit application."""
    inject_custom_css()
    init_session_state()

    device = get_device()
    transform = load_transform()
    model = load_model(MODEL_PATH, device)
    model_loaded = model is not None

    render_sidebar(device, model_loaded)

    # ---- Header ----
    render_header(device)

    # ---- Model validation ----
    if not model_loaded:
        render_error(
            "Model Not Found",
            f"The model weights file <strong>{MODEL_PATH}</strong> could not be loaded. "
            "Please ensure the file exists in the application directory and is a valid PyTorch checkpoint.",
        )
        st.stop()

    # ---- Upload section ----
    st.markdown('<div class="card upload-section">', unsafe_allow_html=True)
    render_section_header("Upload or Select an X-Ray Image")

    upload_col, sample_col = st.columns([1, 1], gap="large")

    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload a chest X-ray",
            type=["jpg", "jpeg", "png"],
            help=f"Supported formats: JPG, JPEG, PNG. Max size: {MAX_UPLOAD_MB} MB.",
            label_visibility="collapsed",
        )

    with sample_col:
        available_samples = {
            name: path for name, path in SAMPLE_IMAGES.items() if Path(path).is_file()
        }
        if available_samples:
            selected_sample_name = st.selectbox(
                "Or choose a sample image",
                options=["-- Select a sample --"] + list(available_samples.keys()),
                label_visibility="collapsed",
            )
        else:
            selected_sample_name = None
            st.caption("No sample images found in the samples/ directory.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Resolve image source ----
    image: Optional[Image.Image] = None
    filename: str = ""

    if uploaded_file is not None:
        valid, err_msg = validate_uploaded_file(uploaded_file)
        if not valid:
            render_error("Upload Validation Failed", err_msg)
        else:
            image, err_msg = open_image_safely(uploaded_file)
            if image is None:
                render_error("Invalid Image", err_msg)
            else:
                filename = uploaded_file.name
    elif (
        selected_sample_name
        and selected_sample_name != "-- Select a sample --"
        and available_samples
    ):
        path = available_samples[selected_sample_name]
        image = load_sample_image(path)
        if image is None:
            render_error(
                "Sample Image Error",
                f"Could not load sample image at <strong>{path}</strong>.",
            )
        else:
            filename = Path(path).name

    if image is None:
        st.markdown(
            f"""
            <div class="card" style="text-align:center;padding:3rem 1rem;color:{TEXT_SECONDARY};">
                Upload a chest X-ray image or select a sample above to begin analysis.
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_disclaimer()
        return

    # ---- Run inference on ORIGINAL image ----
    try:
        prediction, confidence, probabilities, latency_ms = run_inference(
            model, image, transform, device
        )
    except RuntimeError as exc:
        render_error("Inference Failed", str(exc), show_trace=True)
        return
    except Exception as exc:
        render_error(
            "Unexpected Error",
            "An unexpected error occurred during inference. "
            "Please try again or use a different image.",
            show_trace=True,
        )
        return

    # ---- Update header with real latency ----
    # (The header was already rendered; we update status pills via the prediction card)

    # ---- Image viewer (side-by-side) ----
    img_left, img_right = st.columns(2, gap="medium")
    with img_left:
        left_placeholder = st.empty()
    with img_right:
        right_placeholder = st.empty()

    # ---- Enhancement sliders (below images, for VISUALIZATION ONLY) ----
    st.markdown('<div class="card">', unsafe_allow_html=True)
    render_section_header("Image Enhancement (Visualization Only)")
    enh_c1, enh_c2, enh_c3, enh_c4 = st.columns([1, 1, 1, 0.5])
    with enh_c1:
        brightness_val = st.slider("Brightness", 0.5, 2.0, 1.0, 0.05, key="brightness")
    with enh_c2:
        contrast_val = st.slider("Contrast", 0.5, 2.0, 1.0, 0.05, key="contrast")
    with enh_c3:
        sharpness_val = st.slider("Sharpness", 0.5, 3.0, 1.0, 0.05, key="sharpness")
    with enh_c4:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Reset", use_container_width=True):
            for k in ["brightness", "contrast", "sharpness"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    enhanced = apply_enhancements(image, brightness_val, contrast_val, sharpness_val)

    # Populate image placeholders
    with left_placeholder.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        render_section_header("Original Image")
        st.image(image, use_container_width=True)
        st.markdown(f'<div class="img-caption">{filename}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right_placeholder.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        render_section_header("Enhanced Image")
        st.image(enhanced, use_container_width=True)
        has_edits = brightness_val != 1.0 or contrast_val != 1.0 or sharpness_val != 1.0
        adj_label = "Adjusted" if has_edits else "No adjustments applied"
        st.markdown(f'<div class="img-caption">{adj_label}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- Record history ----
    add_history_entry(filename, prediction, confidence, latency_ms, device.type.upper())

    # ---- Dashboard Layout (Diagnosis on Left, Detailed Diagnostics on Right) ----
    dash_left, dash_right = st.columns([5, 7], gap="medium")

    with dash_left:
        # Prediction card
        render_prediction_card(prediction, confidence, latency_ms, device)

        # Probability chart
        st.markdown('<div class="card">', unsafe_allow_html=True)
        render_section_header("Class Probabilities")
        fig = build_probability_chart(probabilities, prediction)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with dash_right:
        # Tabs: Analysis / History / About Model
        tab_analysis, tab_history, tab_about = st.tabs(["Analysis", "History", "About Model"])

        with tab_analysis:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            render_section_header("Diagnostic Reports")
            md_report = generate_markdown_report(
                filename, prediction, confidence, probabilities, latency_ms, device
            )
            txt_report = generate_plaintext_report(
                filename, prediction, confidence, probabilities, latency_ms, device
            )
            dl_c1, dl_c2 = st.columns(2)
            with dl_c1:
                st.download_button(
                    label="Download Markdown Report",
                    data=md_report,
                    file_name=f"report_{Path(filename).stem}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with dl_c2:
                st.download_button(
                    label="Download Plain Text Report",
                    data=txt_report,
                    file_name=f"report_{Path(filename).stem}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        with tab_history:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            render_section_header("Classification History")
            df = get_history_df()
            if df.empty:
                st.markdown(
                    f'<p style="color:{TEXT_SECONDARY};font-size:0.9rem;">No classifications recorded yet.</p>',
                    unsafe_allow_html=True,
                )
            else:
                st.dataframe(df, use_container_width=True, hide_index=True)
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="Download History as CSV",
                    data=csv_data,
                    file_name="classification_history.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        with tab_about:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            render_section_header("Model Information")
            about_c1, about_c2 = st.columns(2)
            with about_c1:
                st.markdown(
                    f"""
                    **Architecture:** ResNet18 (torchvision)

                    **Task:** Binary classification -- Normal vs. Pneumonia

                    **Input:** 224 x 224 RGB chest X-ray image

                    **Normalization:** ImageNet mean and std
                    """,
                )
            with about_c2:
                st.markdown(
                    f"""
                    **Training:** Fine-tuned Layer 4 and FC head

                    **Loss:** Weighted Cross-Entropy (class-balanced)

                    **Optimizer:** Adam (lr = 1e-4)

                    **Weights File:** {MODEL_PATH}
                    """,
                )
            st.markdown("</div>", unsafe_allow_html=True)

    # ---- Disclaimer ----
    render_disclaimer()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()