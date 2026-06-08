"""LitterBot AI - YOLOv8 detection + EfficientNetV2B0 (TFLite) classification.

Pipeline per uploaded image:
  1. YOLOv8n proposes object bounding boxes (class-agnostic for our purposes).
  2. Each cropped box is run through the EfficientNetV2-B0 TFLite classifier
     trained on the 9 RealWaste categories.
  3. An annotated copy of the image is rendered with one colored rectangle per
     piece of trash, color-coded by which bin it belongs in.
  4. A per-detection breakdown lists the predicted category, confidence, and
     disposal tip for each box.

If YOLO finds no objects, the classifier falls back to the whole image.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

MODEL_PATH = Path(__file__).resolve().parent / "models" / "best_litterbot_model.tflite"
YOLO_WEIGHTS = Path(__file__).resolve().parent / "yolov8n.pt"
IMG_SIZE = 224
TOP_K = 3

# YOLO detection settings.
YOLO_CONF = 0.25     # minimum object confidence
YOLO_IOU = 0.45      # NMS IoU threshold
YOLO_IMGSZ = 640     # YOLO inference size
MAX_DETECTIONS = 12  # cap to keep the UI readable

CLASS_NAMES = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Miscellaneous Trash",
    "Paper",
    "Plastic",
    "Textile Trash",
    "Vegetation",
]

DISPOSAL_TIPS = {
    "Cardboard":           "Flatten and place in paper/cardboard recycling.",
    "Food Organics":       "Compost where available, otherwise general waste.",
    "Glass":               "Rinse and place in glass recycling.",
    "Metal":               "Rinse cans and place in metal recycling.",
    "Miscellaneous Trash": "General waste, not recyclable.",
    "Paper":               "Place clean paper in paper recycling.",
    "Plastic":             "Check the resin code; rinse and recycle if accepted locally.",
    "Textile Trash":       "Donate if usable; otherwise textile recycling or general waste.",
    "Vegetation":          "Compost or green-waste bin.",
}

BIN_FOR = {
    "Cardboard":           ("Recycling",       "#1565c0"),
    "Food Organics":       ("Compost",         "#2e7d32"),
    "Glass":               ("Recycling",       "#1565c0"),
    "Metal":               ("Recycling",       "#1565c0"),
    "Miscellaneous Trash": ("Landfill",        "#616161"),
    "Paper":               ("Recycling",       "#1565c0"),
    "Plastic":             ("Recycling",       "#1565c0"),
    "Textile Trash":       ("Donate/Special",  "#6a1b9a"),
    "Vegetation":          ("Compost",         "#2e7d32"),
}

# COCO classes (YOLOv8 default) that are almost never trash and produce noisy
# crops if included. Filtered out before classification.
COCO_SKIP = {"person"}


# ============================================================
# Model loading
# ============================================================

@st.cache_resource(show_spinner="Loading classifier...")
def load_interpreter():
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tflite_runtime.interpreter import Interpreter

    interp = Interpreter(model_path=str(MODEL_PATH))
    interp.allocate_tensors()
    return interp


@st.cache_resource(show_spinner="Loading YOLO detector...")
def load_yolo():
    from ultralytics import YOLO
    # Use the weights committed beside the app if present; otherwise ultralytics
    # downloads them on first use.
    return YOLO(str(YOLO_WEIGHTS) if YOLO_WEIGHTS.exists() else "yolov8n.pt")


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


# ============================================================
# Inference
# ============================================================

def classify_crop(interp, crop: Image.Image) -> np.ndarray:
    """Run the TFLite classifier on a single PIL crop -> length-9 prob vector."""
    arr = np.asarray(
        crop.convert("RGB").resize((IMG_SIZE, IMG_SIZE)),
        dtype=np.float32,
    )[None, ...]  # EfficientNetV2 expects raw [0, 255]

    input_details = interp.get_input_details()
    output_details = interp.get_output_details()
    interp.set_tensor(input_details[0]["index"], arr)
    interp.invoke()
    probs = interp.get_tensor(output_details[0]["index"])[0]

    probs = np.clip(probs, 0.0, 1.0)
    s = probs.sum()
    return probs / s if s > 0 else probs


def detect_boxes(yolo, img: Image.Image) -> list[dict]:
    """Run YOLO on the full image and return a list of {box, yolo_label}."""
    results = yolo(
        np.array(img),
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        imgsz=YOLO_IMGSZ,
        verbose=False,
    )[0]

    detections: list[dict] = []
    if results.boxes is None or len(results.boxes) == 0:
        return detections

    xyxy = results.boxes.xyxy.cpu().numpy()
    cls_ids = results.boxes.cls.cpu().numpy().astype(int)
    confs = results.boxes.conf.cpu().numpy()
    names = results.names  # dict id -> name

    # Sort by confidence desc, then cap.
    order = np.argsort(-confs)
    for i in order:
        label = names.get(int(cls_ids[i]), "object")
        if label in COCO_SKIP:
            continue
        x1, y1, x2, y2 = xyxy[i].tolist()
        detections.append({
            "box": (float(x1), float(y1), float(x2), float(y2)),
            "yolo_label": label,
            "yolo_conf": float(confs[i]),
        })
        if len(detections) >= MAX_DETECTIONS:
            break
    return detections


def annotate(img: Image.Image, detections: list[dict]) -> Image.Image:
    """Draw a colored rectangle + label for every detection."""
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated, "RGBA")
    # Scale font with image size so labels are readable on big and small uploads.
    font_size = max(14, int(min(annotated.size) * 0.025))
    font = _load_font(font_size)

    for i, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det["box"]
        color = det["color"]
        rgb = _hex_to_rgb(color)
        # Box (thick outline).
        line_w = max(3, int(min(annotated.size) * 0.005))
        draw.rectangle([x1, y1, x2, y2], outline=rgb, width=line_w)

        label = f"{i}. {det['class']} ({det['conf'] * 100:.0f}%)"
        # Measure text for background pill.
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        pad = max(4, font_size // 4)
        ly1 = max(0, y1 - th - 2 * pad)
        lx2 = min(annotated.size[0], x1 + tw + 2 * pad)
        draw.rectangle([x1, ly1, lx2, ly1 + th + 2 * pad], fill=rgb + (230,))
        draw.text((x1 + pad, ly1 + pad), label, fill=(255, 255, 255), font=font)

    return annotated


# ============================================================
# UI
# ============================================================

st.set_page_config(
    page_title="LitterBot AI",
    page_icon="\u267b\ufe0f",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    .main { padding: 2rem; }
    .title-container { text-align: center; margin-bottom: 2rem; }
    .verdict {
        padding: 1.25rem 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
        background: #f8fafc;
        border-left: 6px solid #1565c0;
    }
    .verdict h3 { margin: 0 0 0.25rem 0; }
    .verdict .bin-pill {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        color: white;
        font-size: 0.85rem;
        font-weight: 600;
        margin-left: 0.5rem;
        vertical-align: middle;
    }
    .verdict .bin-note {
        margin-top: 0.5rem;
        color: #475569;
        font-size: 0.95rem;
    }

    .leon-breadcrumb {
        font-size: 0.875rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.4rem;
    }
    .leon-breadcrumb a { color: inherit; text-decoration: none; }
    .leon-breadcrumb a:hover { color: #111827; text-decoration: underline; }
    .leon-sep { color: #9ca3af; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<nav class="leon-breadcrumb">
  <span>\u2190</span>
  <a href="https://leonzhao.dev/">leonzhao.dev</a>
  <span class="leon-sep">/</span>
  <a href="https://leonzhao.dev/ai/">AI</a>
  <span class="leon-sep">/</span>
  <a href="https://leonzhao.dev/ai/litterbot/">LitterBot AI</a>
</nav>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="title-container">
    <h1>\u267b\ufe0f LitterBot AI</h1>
    <p>Upload a photo. YOLOv8 finds every piece of trash and the EfficientNetV2-B0 classifier tells you which of 9 bins each one belongs in.</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### About")
    st.info(
        "YOLOv8n proposes object bounding boxes, then EfficientNetV2-B0 "
        "(fine-tuned on the ~4.8k-image RealWaste dataset) classifies each "
        "crop into one of 9 waste categories. Classifier validation accuracy "
        "is around 87.6%. Everything runs on CPU."
    )
    st.markdown(
        """
- Supports JPG, PNG, BMP, WebP
- Detects up to 12 objects per image
- Confidence threshold: 0.25
"""
    )
    st.markdown("**Classes**")
    st.markdown("\n".join(f"- {c}" for c in CLASS_NAMES))
    st.markdown("---")
    st.markdown("**apps.leonzhao.dev/litterbot**")

if not MODEL_PATH.exists():
    st.error(f"Model file not found: `{MODEL_PATH}`")
    st.stop()

try:
    interp = load_interpreter()
    yolo = load_yolo()
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load models: {exc}")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    st.markdown("### Step 1: Upload Image(s)")
    uploaded_files = st.file_uploader(
        "Choose image files",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
        accept_multiple_files=True,
    )
with col2:
    st.markdown("### Step 2: View the Results")

if not uploaded_files:
    col2.info("Upload one or more images to get started.")
else:
    for index, uploaded_file in enumerate(uploaded_files, start=1):
        try:
            image_bytes = uploaded_file.read()
            if len(image_bytes) > 50 * 1024 * 1024:
                col2.error(f"{uploaded_file.name}: image too large (max 50MB)")
                continue

            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            col1.markdown(f"#### Image {index}: {uploaded_file.name}")

            with st.spinner(f"Detecting objects in {uploaded_file.name}..."):
                detections = detect_boxes(yolo, img)

            # Classify each detected crop. If none, fall back to the whole image.
            if not detections:
                detections = [{
                    "box": (0.0, 0.0, float(img.size[0]), float(img.size[1])),
                    "yolo_label": "whole image",
                    "yolo_conf": 1.0,
                }]
                fallback = True
            else:
                fallback = False

            with st.spinner(f"Classifying {len(detections)} crop(s)..."):
                for det in detections:
                    x1, y1, x2, y2 = det["box"]
                    crop = img.crop((x1, y1, x2, y2))
                    probs = classify_crop(interp, crop)
                    top_idx = np.argsort(probs)[::-1][:TOP_K]
                    top = int(top_idx[0])
                    det["probs"] = probs
                    det["top_idx"] = top_idx
                    det["class"] = CLASS_NAMES[top]
                    det["conf"] = float(probs[top])
                    bin_name, bin_color = BIN_FOR[det["class"]]
                    det["bin"] = bin_name
                    det["color"] = bin_color

            annotated = annotate(img, detections)
            col1.image(annotated, use_column_width=True,
                       caption=("Whole image classified (no objects detected)"
                                if fallback else f"{len(detections)} object(s) detected"))

            if fallback:
                col2.warning(
                    f"YOLO did not find distinct objects in **{uploaded_file.name}** "
                    f"so the whole image was classified instead."
                )

            for det in detections:
                bin_color = det["color"]
                col2.markdown(
                    f"""
                    <div class="verdict" style="border-left-color: {bin_color};">
                        <h3>#{detections.index(det) + 1} {det['class']}
                            <span class="bin-pill" style="background:{bin_color};">{det['bin']}</span>
                        </h3>
                        <div style="color:#475569; font-size:0.9rem;">
                            {det['conf'] * 100:.1f}% confidence
                            &middot; detected as <em>{det['yolo_label']}</em> ({det['yolo_conf'] * 100:.0f}%)
                        </div>
                        <div class="bin-note">{DISPOSAL_TIPS[det['class']]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                with col2.expander("Top-3 classifier predictions", expanded=False):
                    for i in det["top_idx"]:
                        name = CLASS_NAMES[int(i)]
                        conf = float(det["probs"][int(i)])
                        st.progress(conf, text=f"{name}: {conf * 100:.1f}%")

            if index < len(uploaded_files):
                col2.markdown("---")

        except Exception as exc:  # noqa: BLE001
            col2.error(f"{uploaded_file.name}: {str(exc)[:200]}")

st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>LitterBot AI &middot; YOLOv8n + EfficientNetV2-B0 &middot; TFLite + Streamlit</p>
</div>
""",
    unsafe_allow_html=True,
)
