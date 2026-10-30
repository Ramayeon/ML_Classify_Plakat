import csv
import html
import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model


# =====================================================================
# 1. PAGE CONFIGURATION  (must be the first Streamlit call)
# =====================================================================

st.set_page_config(
    page_title="BettaVision",
    page_icon="🐟",
    layout="wide"
)


# =====================================================================
# 2. DESIGN SYSTEM  (all colors live in :root, no hex inside the HTML)
# =====================================================================

st.markdown("""
<style>

/* ---------- Color tokens ---------- */
:root {
    --primary: #174C46;
    --primary-light: #2F6F5E;
    --secondary: #E6F1EC;
    --accent: #C2542D;
    --accent-hover: #A8441F;
    --background: #F4F7F4;
    --surface: #FFFFFF;
    --text: #243C36;
    --muted: #5A6B64;
    --border: #DCE7E0;

    --ok: #1F7A4D;      --ok-bg: #E3F4EA;
    --warn: #8A5A00;    --warn-bg: #FFF1D6;
    --danger: #A63A2B;  --danger-bg: #FBE5E1;
}

/* ---------- Base ---------- */
html, body, .stApp {
    font-family: "Inter", "Segoe UI", "Noto Sans Thai", sans-serif;
}

.stApp {
    background: var(--background);
    color: var(--text);
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3, h4 {
    color: var(--primary) !important;
    letter-spacing: -0.4px;
}

p, label {
    color: var(--text);
}

hr {
    border-color: var(--border);
    margin: 22px 0;
}

/* ---------- Hero ---------- */
.hero {
    background: linear-gradient(120deg, #174C46 0%, #28695B 65%, #2F6F5E 100%);
    border-radius: 24px;
    padding: 34px 36px;
    margin-bottom: 26px;
    color: #FFFFFF;
    box-shadow: 0 10px 28px rgba(23, 76, 70, 0.12);
    position: relative;
    overflow: hidden;
}

.hero::after {
    content: "";
    position: absolute;
    width: 190px;
    height: 190px;
    border: 28px solid rgba(255, 255, 255, 0.07);
    border-radius: 50%;
    right: -35px;
    top: -70px;
}

.hero-title {
    color: #FFFFFF;
    font-size: clamp(30px, 4vw, 43px);
    font-weight: 800;
    letter-spacing: -1.2px;
    margin: 0;
}

.hero-description {
    color: #E0EEE7;
    font-size: 15px;
    line-height: 1.8;
    margin-top: 10px;
    max-width: 620px;
}

/* ---------- Headings and text ---------- */
.section-title {
    color: var(--primary);
    font-size: 22px;
    font-weight: 800;
    margin-top: 10px;
    margin-bottom: 4px;
}

.section-description {
    color: var(--muted);
    font-size: 14px;
    margin-bottom: 16px;
}

.step-title {
    color: var(--primary);
    font-size: 15px;
    font-weight: 750;
    margin-bottom: 2px;
}

.step-hint {
    color: var(--muted);
    font-size: 13px;
    margin-bottom: 10px;
}

.card-title {
    color: var(--primary);
    font-size: 17px;
    font-weight: 750;
    margin-bottom: 10px;
}

.card-label {
    color: var(--muted);
    font-size: 13px;
    font-weight: 600;
}

.file-name {
    color: var(--muted);
    font-size: 13px;
    padding-top: 8px;
    overflow-wrap: anywhere;
}

.file-name.center {
    color: var(--primary);
    font-weight: 650;
    text-align: center;
    min-height: 42px;
}

.status-ready {
    color: var(--primary-light);
    font-size: 14px;
    font-weight: 650;
}

.status-wait {
    color: var(--muted);
    font-size: 14px;
}

/* ---------- Result content ---------- */
.result-name {
    color: var(--primary);
    font-size: 30px;
    font-weight: 800;
    margin-top: 6px;
    line-height: 1.3;
    overflow-wrap: anywhere;
}

.result-name.small {
    font-size: 23px;
}

.result-sub {
    color: var(--muted);
    font-size: 13px;
    margin-top: 6px;
}

/* ---------- Badges ---------- */
.badge {
    display: inline-block;
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 700;
    margin-top: 8px;
}

.badge-class  { background: var(--secondary); color: var(--primary); }
.badge-ok     { background: var(--ok-bg);     color: var(--ok); }
.badge-warn   { background: var(--warn-bg);   color: var(--warn); }
.badge-danger { background: var(--danger-bg); color: var(--danger); }

/* ---------- Confidence ---------- */
.conf-number {
    font-size: 30px;
    font-weight: 800;
    margin-top: 18px;
}

.conf-caption {
    color: var(--muted);
    font-size: 13px;
    margin-bottom: 8px;
}

.conf-track {
    width: 100%;
    height: 9px;
    background: var(--secondary);
    border-radius: 20px;
    overflow: hidden;
}

.conf-fill {
    height: 9px;
    border-radius: 20px;
}

.ok     { color: var(--ok); }
.warn   { color: var(--warn); }
.danger { color: var(--danger); }

.conf-fill.ok     { background: var(--ok); }
.conf-fill.warn   { background: var(--warn); }
.conf-fill.danger { background: var(--danger); }

/* ---------- Containers ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 18px;
}

[data-testid="stExpander"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    overflow: hidden;
}

[data-testid="stExpander"] summary {
    color: var(--primary);
    font-weight: 650;
}

[data-testid="stAlert"] {
    border-radius: 12px;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}

/* ---------- Radio ---------- */
.stRadio > div {
    gap: 10px;
}

.stRadio label {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 8px 14px;
}

.stRadio label:has(input:checked) {
    background: var(--secondary);
    border-color: var(--primary-light);
}

/* ---------- Uploader ---------- */
[data-testid="stFileUploader"] {
    background: var(--surface);
    border: 1px dashed #A8C6B5;
    border-radius: 16px;
    padding: 15px;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--primary-light);
}

/* ---------- Buttons ---------- */
.stButton > button {
    background: var(--surface);
    color: var(--primary);
    border: 1px solid var(--primary-light);
    border-radius: 12px;
    min-height: 44px;
    font-size: 15px;
    font-weight: 700;
    transition: background 0.2s ease, border-color 0.2s ease;
}

.stButton > button:hover {
    background: var(--secondary);
    border-color: var(--primary);
    color: var(--primary);
}

.stButton > button[data-testid="stBaseButton-primary"] {
    background: var(--accent);
    border-color: var(--accent);
    color: #FFFFFF;
    min-height: 46px;
    box-shadow: 0 4px 10px rgba(194, 84, 45, 0.18);
}

.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: var(--accent-hover);
    border-color: var(--accent-hover);
    color: #FFFFFF;
}

.stButton > button[data-testid="stBaseButton-primary"]:disabled {
    background: #E3E9E5;
    border-color: #E3E9E5;
    color: #7C8A84;
    box-shadow: none;
    cursor: not-allowed;
}

.stDownloadButton > button {
    background: var(--primary);
    color: #FFFFFF;
    border: none;
    border-radius: 12px;
    min-height: 44px;
    font-weight: 650;
}

.stDownloadButton > button:hover {
    background: var(--primary-light);
    color: #FFFFFF;
}

.stButton > button:focus-visible,
.stDownloadButton > button:focus-visible {
    outline: 3px solid rgba(47, 111, 94, 0.35);
    outline-offset: 2px;
}

/* ---------- Mobile ---------- */
@media (max-width: 640px) {
    .block-container {
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .hero {
        padding: 25px 22px;
        border-radius: 18px;
    }

    .result-name {
        font-size: 24px;
    }
}

/* ---------- Reduced motion ---------- */
@media (prefers-reduced-motion: reduce) {
    * {
        transition: none !important;
    }
}

</style>
""", unsafe_allow_html=True)


# =====================================================================
# 3. MODEL CONFIGURATION
# =====================================================================

MODEL_PATH = Path(__file__).parent / "betta4_model_inceptionV3(latest).h5"

CLASS_INFO = [
    {"code": "CT", "name": "Crowntail",  "subtitle": "Crown-shaped fins"},
    {"code": "DT", "name": "Doubletail", "subtitle": "Two tail lobes"},
    {"code": "HM", "name": "Halfmoon",   "subtitle": "Half-moon tail"},
    {"code": "PK", "name": "Plakat",     "subtitle": "Short-finned type"},
    {"code": "RT", "name": "Rosetail",   "subtitle": "Rose-like tail"},
    {"code": "VT", "name": "Veiltail",   "subtitle": "Flowing tail"},
]

# The order must match the class order used when the model was trained
CLASS_NAMES = [item["code"] for item in CLASS_INFO]
CLASS_LOOKUP = {item["code"]: item for item in CLASS_INFO}

# Confidence thresholds (%). Tune these with your own validation results.
HIGH_CONFIDENCE = 85
MEDIUM_CONFIDENCE = 60

ALLOWED_TYPES = ["jpg", "jpeg", "png", "webp"]
RESULT_KEYS = [
    "betta_results",
    "betta_predictions",
    "betta_files",
    "betta_failed_files",
]


# =====================================================================
# 4. LOAD MODEL
# =====================================================================

@st.cache_resource
def load_betta_model():
    return load_model(str(MODEL_PATH), compile=False)


try:
    model = load_betta_model()
except Exception as error:
    st.error("Could not load the model.")
    st.code(str(error))
    st.info(
        "Check that the model file is in the same folder as this app "
        "and that TensorFlow is installed."
    )
    st.stop()


# =====================================================================
# 5. CHECK MODEL INPUT AND OUTPUT
# =====================================================================

try:
    input_shape = model.input_shape
    output_shape = model.output_shape

    if isinstance(input_shape, list):
        if len(input_shape) != 1:
            st.error("This model has several inputs, which this app does not support.")
            st.stop()
        input_shape = input_shape[0]

    if isinstance(output_shape, list):
        if len(output_shape) != 1:
            st.error("This model has several outputs, which this app does not support.")
            st.stop()
        output_shape = output_shape[0]

    if len(input_shape) != 4:
        st.error("The model input shape is not valid for images.")
        st.stop()

    if input_shape[-1] != 3:
        st.error("The model must accept RGB images with 3 channels.")
        st.stop()

    if output_shape[-1] != len(CLASS_NAMES):
        st.error(
            f"The model predicts {output_shape[-1]} classes, "
            f"but {len(CLASS_NAMES)} class names are defined."
        )
        st.stop()

    image_height = input_shape[1] or 224
    image_width = input_shape[2] or 224

except Exception as error:
    st.error("Could not check the model structure.")
    st.code(str(error))
    st.stop()


# =====================================================================
# 6. IMAGE PREPROCESSING AND PREDICTION
# =====================================================================

def preprocess_image(uploaded_file):
    """Resize and scale an image to match the model input."""
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((image_width, image_height))
    image_array = np.array(image, dtype=np.float32)

    # Use this only if the model was trained with pixels divided by 255
    return image_array / 255.0


def predict_images(image_arrays):
    """Predict several images in one batch."""
    batch = np.stack(image_arrays, axis=0)
    predictions = model.predict(batch, verbose=0)

    results = []

    for prediction in predictions:
        predicted_index = int(np.argmax(prediction))
        predicted_code = CLASS_NAMES[predicted_index]
        predicted_info = CLASS_LOOKUP[predicted_code]

        probabilities = {
            CLASS_LOOKUP[code]["name"]: round(float(prediction[index]) * 100, 2)
            for index, code in enumerate(CLASS_NAMES)
        }

        results.append({
            "code": predicted_code,
            "name": predicted_info["name"],
            "subtitle": predicted_info["subtitle"],
            "confidence": float(prediction[predicted_index]) * 100,
            "probabilities": probabilities,
        })

    return results


# =====================================================================
# 7. UI HELPERS
# =====================================================================

def section(title, description=""):
    """Section heading that uses the CSS classes, so styling stays in one place."""
    markup = f'<div class="section-title">{title}</div>'
    if description:
        markup += f'<div class="section-description">{description}</div>'
    st.markdown(markup, unsafe_allow_html=True)


def confidence_level(percent):
    """Return a readable label and a CSS level (ok / warn / danger)."""
    if percent >= HIGH_CONFIDENCE:
        return "High confidence", "ok"
    if percent >= MEDIUM_CONFIDENCE:
        return "Moderate, double-check", "warn"
    return "Low, try a clearer photo", "danger"


def confidence_block(percent):
    """Confidence number, colored bar and level badge as one HTML string."""
    percent = max(0.0, min(100.0, percent))
    label, level = confidence_level(percent)

    return (
        f'<div class="conf-number {level}">{percent:.2f}%</div>'
        f'<div class="conf-caption">Model confidence</div>'
        f'<div class="conf-track">'
        f'<div class="conf-fill {level}" style="width:{percent:.2f}%"></div>'
        f'</div>'
        f'<span class="badge badge-{level}">{label}</span>'
    )


def probability_table(prediction):
    """All class probabilities, highest first."""
    rows = sorted(
        (
            {"Variety": name, "Probability (%)": value}
            for name, value in prediction["probabilities"].items()
        ),
        key=lambda row: row["Probability (%)"],
        reverse=True,
    )

    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={
            "Variety": st.column_config.TextColumn("Fish variety"),
            "Probability (%)": st.column_config.ProgressColumn(
                "Probability",
                format="%.2f%%",
                min_value=0,
                max_value=100,
            ),
        },
    )


def show_image(uploaded_file, **kwargs):
    """Show an uploaded image, or a warning if it cannot be opened."""
    try:
        st.image(Image.open(uploaded_file).convert("RGB"), **kwargs)
    except Exception:
        st.warning(f"Could not display {uploaded_file.name}.")


def clear_results():
    for key in RESULT_KEYS:
        st.session_state.pop(key, None)


# =====================================================================
# 8. HERO
# =====================================================================

st.markdown("""
<div class="hero">
    <div class="hero-title">🐟 BettaVision</div>
    <div class="hero-description">
        Identify betta fish varieties from a photo.
        Upload one image or a whole batch and get the predicted variety
        with a confidence score in seconds.
    </div>
</div>
""", unsafe_allow_html=True)


# =====================================================================
# 9. START CLASSIFICATION
# =====================================================================

section("Start classification", "Choose a mode, then upload your betta fish images.")

with st.container(border=True):

    mode_col, upload_col = st.columns([0.85, 1.65], gap="large")

    with mode_col:
        st.markdown(
            '<div class="step-title">Step 1: Analysis mode</div>'
            '<div class="step-hint">Choose how to analyze images.</div>',
            unsafe_allow_html=True,
        )

        analysis_mode = st.radio(
            "Choose analysis mode",
            ["Single Image", "Multiple Images"],
            label_visibility="collapsed",
            key="analysis_mode",
        )

        if analysis_mode == "Single Image":
            st.caption("Analyze one fish at a time.")
        else:
            st.caption("Analyze several images together.")

    with upload_col:
        st.markdown(
            '<div class="step-title">Step 2: Upload images</div>'
            '<div class="step-hint">JPG, JPEG, PNG or WEBP. '
            'A side view with the whole fish visible works best.</div>',
            unsafe_allow_html=True,
        )

        if analysis_mode == "Single Image":
            single_file = st.file_uploader(
                "Upload a betta fish image",
                type=ALLOWED_TYPES,
                accept_multiple_files=False,
                key="single_uploader",
                label_visibility="collapsed",
            )
            files_to_analyze = [single_file] if single_file is not None else []
        else:
            files_to_analyze = st.file_uploader(
                "Upload betta fish images",
                type=ALLOWED_TYPES,
                accept_multiple_files=True,
                key="multiple_uploader",
                label_visibility="collapsed",
            ) or []

    st.divider()

    status_col, button_col = st.columns([1.5, 1], vertical_alignment="center")

    with status_col:
        if files_to_analyze:
            st.markdown(
                f'<div class="status-ready">✓ {len(files_to_analyze)} image(s) ready to analyze</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status-wait">Add at least one image to start.</div>',
                unsafe_allow_html=True,
            )

    with button_col:
        analyze_clicked = st.button(
            "Analyze image" if analysis_mode == "Single Image" else "Analyze all images",
            type="primary",
            width="stretch",
            disabled=not files_to_analyze,
            key="analyze_button",
        )


# =====================================================================
# 10. UPLOADED IMAGE PREVIEW  (collapses once results are shown)
# =====================================================================

if files_to_analyze:

    section("Uploaded images", f"{len(files_to_analyze)} selected")

    with st.expander(
        "Preview",
        expanded="betta_results" not in st.session_state,
    ):

        if analysis_mode == "Single Image":
            show_image(
                files_to_analyze[0],
                caption=files_to_analyze[0].name,
                width=420,
            )

        else:
            IMAGES_PER_ROW = 3

            for start in range(0, len(files_to_analyze), IMAGES_PER_ROW):
                row_files = files_to_analyze[start:start + IMAGES_PER_ROW]
                columns = st.columns(IMAGES_PER_ROW, gap="medium")

                for column, uploaded_file in zip(columns, row_files):
                    with column:
                        with st.container(border=True):
                            show_image(uploaded_file, width="stretch")
                            st.markdown(
                                f'<div class="file-name center">{html.escape(uploaded_file.name)}</div>',
                                unsafe_allow_html=True,
                            )


# =====================================================================
# 11. ANALYZE IMAGES
# =====================================================================

if files_to_analyze and analyze_clicked:

    image_arrays = []
    valid_files = []
    failed_files = []

    with st.spinner("Analyzing images..."):

        for uploaded_file in files_to_analyze:
            try:
                image_arrays.append(preprocess_image(uploaded_file))
                valid_files.append(uploaded_file)
            except Exception as error:
                failed_files.append({
                    "Filename": uploaded_file.name,
                    "Error": str(error),
                })

        if not image_arrays:
            st.error("None of the selected images could be analyzed.")
            st.stop()

        try:
            predictions = predict_images(image_arrays)
        except Exception as error:
            st.error("Something went wrong during analysis.")
            st.code(str(error))
            st.stop()

    result_rows = [
        {
            "Filename": uploaded_file.name,
            "Predicted Variety": prediction["name"],
            "Class Code": prediction["code"],
            "Confidence (%)": round(prediction["confidence"], 2),
        }
        for uploaded_file, prediction in zip(valid_files, predictions)
    ]

    st.session_state["betta_results"] = result_rows
    st.session_state["betta_predictions"] = predictions
    st.session_state["betta_files"] = valid_files
    st.session_state["betta_failed_files"] = failed_files

    st.toast("Analysis complete")


# =====================================================================
# 12. ANALYSIS RESULTS
# =====================================================================

if "betta_results" in st.session_state:

    result_rows = st.session_state["betta_results"]
    predictions = st.session_state["betta_predictions"]
    result_files = st.session_state["betta_files"]

    st.divider()

    title_col, clear_col = st.columns([4, 1], vertical_alignment="center")

    with title_col:
        section("Analysis results", f"Analyzed {len(result_rows)} image(s).")

    with clear_col:
        if st.button("Clear results", key="clear_results", width="stretch"):
            clear_results()
            st.rerun()

    # -----------------------------------------------------------------
    # SINGLE IMAGE RESULT
    # -----------------------------------------------------------------

    if len(result_rows) == 1:

        result = result_rows[0]
        prediction = predictions[0]
        _, level = confidence_level(prediction["confidence"])

        col_image, col_result = st.columns(2, gap="large")

        with col_image:
            st.markdown('<div class="card-title">Fish image</div>', unsafe_allow_html=True)

            with st.container(border=True):
                # Read from session state so the image stays visible
                # even if the uploader is cleared after analysis.
                show_image(result_files[0], width="stretch")
                st.markdown(
                    f'<div class="file-name">{html.escape(result["Filename"])}</div>',
                    unsafe_allow_html=True,
                )

        with col_result:
            st.markdown('<div class="card-title">Prediction</div>', unsafe_allow_html=True)

            with st.container(border=True):
                st.markdown(
                    '<div class="card-label">Predicted variety</div>'
                    f'<div class="result-name">{prediction["name"]}</div>'
                    f'<span class="badge badge-class">Class {prediction["code"]}</span>'
                    f'{confidence_block(prediction["confidence"])}'
                    f'<div class="result-sub">{prediction["subtitle"]}</div>',
                    unsafe_allow_html=True,
                )

        if level == "danger":
            st.warning(
                "The model is not confident about this image. "
                "Try a sharper photo with the whole fish visible, "
                "taken from the side against a plain background."
            )

        section("Class probabilities", "How the model scored every variety.")
        probability_table(prediction)

    # -----------------------------------------------------------------
    # MULTIPLE IMAGE RESULTS
    # -----------------------------------------------------------------

    else:

        section("Individual results", "One card per image, with a summary table first.")

        with st.container(border=True):
            st.markdown('<div class="card-title">Prediction summary</div>', unsafe_allow_html=True)

            st.dataframe(
                pd.DataFrame(result_rows),
                hide_index=True,
                width="stretch",
                column_config={
                    "Filename": st.column_config.TextColumn("Image filename", width="large"),
                    "Predicted Variety": st.column_config.TextColumn("Predicted variety", width="medium"),
                    "Class Code": st.column_config.TextColumn("Class code", width="small"),
                    "Confidence (%)": st.column_config.ProgressColumn(
                        "Confidence",
                        format="%.2f%%",
                        min_value=0,
                        max_value=100,
                    ),
                },
            )

        st.markdown('<div class="card-title">Detailed analysis</div>', unsafe_allow_html=True)

        for index, (result, prediction) in enumerate(zip(result_rows, predictions)):

            with st.container(border=True):

                head_col, badge_col = st.columns([3, 1], vertical_alignment="center")

                with head_col:
                    st.markdown(
                        f'<div class="card-title" style="margin-bottom:2px">Image {index + 1}</div>'
                        f'<div class="file-name" style="padding-top:0">{html.escape(result["Filename"])}</div>',
                        unsafe_allow_html=True,
                    )

                with badge_col:
                    st.markdown(
                        f'<div style="text-align:right">'
                        f'<span class="badge badge-class">Class {prediction["code"]}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                st.divider()

                image_col, detail_col = st.columns([1, 1.35], gap="large")

                with image_col:
                    if index < len(result_files):
                        show_image(result_files[index], width="stretch")

                with detail_col:
                    st.markdown(
                        '<div class="card-label">Predicted variety</div>'
                        f'<div class="result-name small">{prediction["name"]}</div>'
                        f'<div class="result-sub">{prediction["subtitle"]}</div>'
                        f'{confidence_block(prediction["confidence"])}',
                        unsafe_allow_html=True,
                    )

                with st.expander("View all class probabilities"):
                    probability_table(prediction)

    # -----------------------------------------------------------------
    # EXPORT
    # -----------------------------------------------------------------

    st.divider()
    section("Export results", "Save the table above as a CSV file.")

    csv_buffer = io.StringIO()
    writer = csv.DictWriter(
        csv_buffer,
        fieldnames=["Filename", "Predicted Variety", "Class Code", "Confidence (%)"],
    )
    writer.writeheader()
    writer.writerows(result_rows)

    st.download_button(
        label="Download results as CSV",
        data=csv_buffer.getvalue().encode("utf-8-sig"),
        file_name="betta_classification_results.csv",
        mime="text/csv",
        width="stretch",
    )

    # -----------------------------------------------------------------
    # FILES THAT FAILED
    # -----------------------------------------------------------------

    failed_files = st.session_state.get("betta_failed_files", [])

    if failed_files:
        st.warning(f"{len(failed_files)} image(s) could not be processed.")
        st.dataframe(pd.DataFrame(failed_files), hide_index=True, width="stretch")


# =====================================================================
# 13. ABOUT
# =====================================================================

st.divider()

with st.expander("About BettaVision"):
    st.write(
        "BettaVision predicts betta fish varieties from photos "
        "using a trained deep learning model."
    )

    st.write("Supported classes:")

    for item in CLASS_INFO:
        st.write(f"- **{item['name']} ({item['code']})**: {item['subtitle']}")

    st.caption(f"Model input size: {image_width} × {image_height} pixels")
    