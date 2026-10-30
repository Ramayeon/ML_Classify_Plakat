
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
import csv
import io


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="BettaVision",
    page_icon="🐟",
    layout="wide"
)



# ==================================================
# 2. MODERN UI DESIGN
# ==================================================

st.markdown("""
<style>

/* ---------- Color system ---------- */
:root {
    --primary: #174C46;
    --primary-light: #397B69;
    --secondary: #EAF4EF;
    --accent: #E58B5A;
    --accent-hover: #CF7547;
    --background: #F6F8F5;
    --surface: #FFFFFF;
    --text: #243C36;
    --muted: #74847D;
    --border: #DFE9E2;
}

/* ---------- App background ---------- */
.stApp {
    background: var(--background);
    color: var(--text);
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* ---------- Typography ---------- */
html, body, [class*="css"] {
    font-family: "Inter", "Segoe UI", sans-serif;
}

h1, h2, h3 {
    color: var(--primary) !important;
    letter-spacing: -0.5px;
}

p, label {
    color: var(--text);
}

/* ---------- Hero header ---------- */
.hero {
    background: linear-gradient(
        120deg,
        #174C46 0%,
        #28695B 65%,
        #397B69 100%
    );
    border-radius: 24px;
    padding: 34px 36px;
    margin-bottom: 25px;
    color: white;
    box-shadow: 0 10px 28px rgba(23, 76, 70, 0.12);
    position: relative;
    overflow: hidden;
}

.hero::after {
    content: "";
    position: absolute;
    width: 190px;
    height: 190px;
    border: 28px solid rgba(255,255,255,0.07);
    border-radius: 50%;
    right: -35px;
    top: -70px;
}

.hero-label {
    color: #D2E8DC;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
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

/* ---------- Section headings ---------- */
.section-title {
    color: var(--primary);
    font-size: 23px;
    font-weight: 750;
    margin-top: 12px;
    margin-bottom: 5px;
}

.section-description {
    color: var(--muted);
    font-size: 14px;
    margin-bottom: 18px;
}

/* ---------- Cards ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 18px;
}

.result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 23px;
    margin-bottom: 16px;
    box-shadow: 0 5px 18px rgba(28, 64, 49, 0.04);
}

.result-label {
    color: var(--muted);
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.1px;
}

.result-name {
    color: var(--primary);
    font-size: 28px;
    font-weight: 800;
    margin-top: 8px;
    line-height: 1.3;
}

.class-badge {
    display: inline-block;
    background: var(--secondary);
    color: var(--primary);
    border-radius: 7px;
    padding: 5px 10px;
    font-size: 12px;
    font-weight: 700;
    margin-top: 9px;
}

.confidence-number {
    color: var(--primary-light);
    font-size: 30px;
    font-weight: 800;
    margin-top: 18px;
}

.confidence-caption {
    color: var(--muted);
    font-size: 13px;
    margin-top: 2px;
}

/* ---------- Buttons ---------- */
.stButton > button {
    background: var(--accent);
    color: #FFFFFF;
    border: 1px solid var(--accent);
    border-radius: 12px;
    min-height: 46px;
    font-size: 15px;
    font-weight: 700;
    transition: all 0.2s ease;
    box-shadow: 0 4px 10px rgba(229, 139, 90, 0.15);
}

.stButton > button:hover {
    background: var(--accent-hover);
    border-color: var(--accent-hover);
    color: #FFFFFF;
    transform: translateY(-1px);
}

.stButton > button:focus {
    color: #FFFFFF;
    border-color: var(--accent-hover);
}

/* Download button */
.stDownloadButton > button {
    background: var(--primary);
    color: white;
    border: none;
    border-radius: 12px;
    min-height: 44px;
    font-weight: 650;
}

.stDownloadButton > button:hover {
    background: var(--primary-light);
    color: white;
}

/* ---------- Radio selection ---------- */
.stRadio > div {
    gap: 12px;
}

.stRadio label {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 8px 14px;
}

/* ---------- Uploader ---------- */
[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 1px dashed #A8C6B5;
    border-radius: 16px;
    padding: 15px;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--primary-light);
}

/* ---------- Data table ---------- */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}

/* ---------- Expanders ---------- */
[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 14px;
    overflow: hidden;
}

[data-testid="stExpander"] summary {
    color: var(--primary);
    font-weight: 650;
}

/* ---------- Progress ---------- */
.stProgress > div > div > div > div {
    background: var(--primary-light);
}

/* ---------- Dividers ---------- */
hr {
    border-color: var(--border);
    margin-top: 24px;
    margin-bottom: 24px;
}

/* ---------- Alerts ---------- */
[data-testid="stAlert"] {
    border-radius: 12px;
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

    .hero-title {
        font-size: 30px;
    }

    .result-name {
        font-size: 23px;
    }
}

</style>
""", unsafe_allow_html=True)



# ==================================================
# 3. MODEL CONFIGURATION
# ==================================================

MODEL_PATH = "betta4_model_inceptionV3(latest).h5"

CLASS_INFO = [
    {
        "code": "CT",
        "name": "Crowntail",
        "subtitle": "Crown-shaped fins"
    },
    {
        "code": "DT",
        "name": "Doubletail",
        "subtitle": "Two tail lobes"
    },
    {
        "code": "HM",
        "name": "Halfmoon",
        "subtitle": "Half-moon tail"
    },
    {
        "code": "PK",
        "name": "Plakat",
        "subtitle": "Short-finned type"
    },
    {
        "code": "RT",
        "name": "Rosetail",
        "subtitle": "Rose-like tail"
    },
    {
        "code": "VT",
        "name": "Veiltail",
        "subtitle": "Flowing tail"
    }
]

# ลำดับนี้ต้องตรงกับลำดับคลาสที่ใช้ฝึกโมเดล
CLASS_NAMES = [item["code"] for item in CLASS_INFO]

CLASS_LOOKUP = {
    item["code"]: item
    for item in CLASS_INFO
}


# ==================================================
# 4. LOAD MODEL
# ==================================================

@st.cache_resource
def load_betta_model():
    return load_model(MODEL_PATH, compile=False)


try:
    model = load_betta_model()
except Exception as error:
    st.error("ไม่สามารถโหลดโมเดลได้")
    st.code(str(error))
    st.info(
        "ตรวจสอบว่าไฟล์โมเดลอยู่ในโฟลเดอร์เดียวกับ "
        "palkut_app.py และติดตั้ง TensorFlow แล้ว"
    )
    st.stop()


# ==================================================
# 5. CHECK MODEL INPUT AND OUTPUT
# ==================================================

try:
    input_shape = model.input_shape
    output_shape = model.output_shape

    # รองรับโมเดลที่มี input/output เพียงชุดเดียว
    if isinstance(input_shape, list):
        if len(input_shape) != 1:
            st.error("โมเดลนี้มีหลาย input ซึ่งโค้ดนี้ไม่รองรับ")
            st.stop()
        input_shape = input_shape[0]

    if isinstance(output_shape, list):
        if len(output_shape) != 1:
            st.error("โมเดลนี้มีหลาย output ซึ่งโค้ดนี้ไม่รองรับ")
            st.stop()
        output_shape = output_shape[0]

    if len(input_shape) != 4:
        st.error("รูปแบบ input ของโมเดลไม่ถูกต้อง")
        st.stop()

    if input_shape[-1] != 3:
        st.error("โมเดลต้องรับภาพสี RGB จำนวน 3 channels")
        st.stop()

    if output_shape[-1] != len(CLASS_NAMES):
        st.error(
            f"โมเดลทำนายได้ {output_shape[-1]} คลาส "
            f"แต่กำหนดชื่อไว้ {len(CLASS_NAMES)} คลาส"
        )
        st.stop()

    image_height = input_shape[1] or 224
    image_width = input_shape[2] or 224

except Exception as error:
    st.error("เกิดปัญหาในการตรวจสอบโครงสร้างโมเดล")
    st.code(str(error))
    st.stop()


# ==================================================
# 6. IMAGE PREPROCESSING
# ==================================================

def preprocess_image(uploaded_file):
    """
    แปลงภาพให้มีขนาดและรูปแบบตรงกับ input ของโมเดล
    """

    image = Image.open(uploaded_file).convert("RGB")

    image = image.resize(
        (image_width, image_height)
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    # ใช้เมื่อโมเดลฝึกด้วยการหาร pixel ด้วย 255
    image_array = image_array / 255.0

    return image_array


# ==================================================
# 7. PREDICTION FUNCTION
# ==================================================

def predict_images(image_arrays):
    """
    รับภาพหลายภาพแล้วทำนายพร้อมกันใน batch เดียว
    """

    batch = np.stack(image_arrays, axis=0)

    predictions = model.predict(
        batch,
        verbose=0
    )

    results = []

    for prediction in predictions:
        predicted_index = int(np.argmax(prediction))
        predicted_code = CLASS_NAMES[predicted_index]
        predicted_info = CLASS_LOOKUP[predicted_code]

        confidence = float(prediction[predicted_index])

        probabilities = {
            CLASS_LOOKUP[code]["name"]: round(
                float(prediction[index]) * 100, 2
            )
            for index, code in enumerate(CLASS_NAMES)
        }

        results.append({
            "code": predicted_code,
            "name": predicted_info["name"],
            "subtitle": predicted_info["subtitle"],
            "confidence": confidence * 100,
            "probabilities": probabilities
        })

    return results



# ==================================================
# 8. HERO HEADER
# ==================================================

st.markdown("""
<div class="hero">
    <div class="hero-label">AI · IMAGE CLASSIFICATION</div>
    <div class="hero-title">🐟 BettaVision</div>
    <div class="hero-description">
        Discover the beauty of betta fish with AI.
        Upload one image or analyze multiple fish images
        to identify their varieties in seconds.
    </div>
</div>
""", unsafe_allow_html=True)


# ==================================================
# 9. START CLASSIFICATION - BALANCED LAYOUT
# ==================================================

st.markdown("""
<div style="
    color: #174C46;
    font-size: 24px;
    font-weight: 800;
    margin-top: 10px;
    margin-bottom: 5px;
">
    Start Classification
</div>
<div style="
    color: #74847D;
    font-size: 14px;
    margin-bottom: 18px;
">
    Choose a mode and upload your betta fish images.
</div>
""", unsafe_allow_html=True)

# ใช้การ์ดเดียวครอบคลุมตัวเลือกและพื้นที่อัปโหลด
with st.container(border=True):

    # แบ่งเป็นสองคอลัมน์ให้สมส่วน
    mode_col, upload_col = st.columns(
        [0.85, 1.65],
        gap="large"
    )

    with mode_col:
        st.markdown("""
        <div style="
            color: #174C46;
            font-size: 15px;
            font-weight: 750;
            margin-bottom: 8px;
        ">
            01 &nbsp; Analysis Mode
        </div>
        <div style="
            color: #74847D;
            font-size: 12px;
            margin-bottom: 10px;
        ">
            Choose how to analyze images.
        </div>
        """, unsafe_allow_html=True)

        analysis_mode = st.radio(
            "Choose analysis mode",
            ["Single Image", "Multiple Images"],
            horizontal=False,
            label_visibility="collapsed",
            key="analysis_mode"
        )

        if analysis_mode == "Single Image":
            st.caption("Analyze one fish at a time.")
        else:
            st.caption("Analyze several images together.")

    with upload_col:
        st.markdown("""
        <div style="
            color: #174C46;
            font-size: 15px;
            font-weight: 750;
            margin-bottom: 8px;
        ">
            02 &nbsp; Upload Images
        </div>
        <div style="
            color: #74847D;
            font-size: 12px;
            margin-bottom: 10px;
        ">
            Supported formats: JPG, JPEG, PNG, WEBP
        </div>
        """, unsafe_allow_html=True)

        if analysis_mode == "Single Image":
            uploaded_files = st.file_uploader(
                "Upload a betta fish image",
                type=["jpg", "jpeg", "png", "webp"],
                accept_multiple_files=False,
                key="single_uploader",
                label_visibility="collapsed"
            )

            if uploaded_files is not None:
                files_to_analyze = [uploaded_files]
            else:
                files_to_analyze = []

        else:
            uploaded_files = st.file_uploader(
                "Upload betta fish images",
                type=["jpg", "jpeg", "png", "webp"],
                accept_multiple_files=True,
                key="multiple_uploader",
                label_visibility="collapsed"
            )

            files_to_analyze = uploaded_files or []

        if files_to_analyze:
            st.success(
                f"{len(files_to_analyze)} image(s) selected"
            )
        else:
            st.caption(
                "Select an image or drag and drop files here."
            )

    # เส้นแบ่งส่วนล่าง
    st.markdown(
        "<hr style='margin: 20px 0 14px 0;'>",
        unsafe_allow_html=True
    )

    # แสดงสถานะและปุ่มให้อยู่ในแนวเดียวกัน
    status_col, button_col = st.columns(
        [1.5, 1],
        vertical_alignment="center"
    )

    with status_col:
        if files_to_analyze:
            st.markdown(
                f"""
                <div style="
                    color: #397B69;
                    font-size: 13px;
                    font-weight: 650;
                ">
                    ✓ Ready to analyze
                    &nbsp; · &nbsp;
                    {len(files_to_analyze)} image(s)
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown("""
                <div style="
                    color: #74847D;
                    font-size: 13px;
                ">
                    Waiting for images...
                </div>
            """, unsafe_allow_html=True)

    with button_col:
        st.markdown("""
        <style>
        div[data-testid="stVerticalBlock"] 
        div[data-testid="stButton"] button {
            width: 100%;
        }
        </style>
        """, unsafe_allow_html=True)

        analyze_clicked = st.button(
            "Analyze Fish Image"
            if analysis_mode == "Single Image"
            else "Analyze All Images",
            type="primary",
            use_container_width=True,
            disabled=not bool(files_to_analyze),
            key="analyze_button"
        )



# ==================================================
# 10. DISPLAY UPLOADED IMAGES
# ==================================================

if files_to_analyze:

    st.markdown("""
        <div style="
            color: #174C46;
            font-size: 21px;
            font-weight: 750;
            margin-top: 18px;
            margin-bottom: 5px;
        ">
            Uploaded Images
        </div>
        <div style="
            color: #74847D;
            font-size: 13px;
            margin-bottom: 16px;
        ">
            Preview of the selected betta fish images.
        </div>
    """, unsafe_allow_html=True)

    st.caption(f"{len(files_to_analyze)} image(s) selected")

    # แสดงภาพเดียวแบบใหญ่
    if analysis_mode == "Single Image":

        uploaded_file = files_to_analyze[0]

        try:
            preview_image = Image.open(
                uploaded_file
            ).convert("RGB")

            st.image(
                preview_image,
                caption=uploaded_file.name,
                width=420
            )

        except Exception:
            st.warning(
                f"Unable to display {uploaded_file.name}"
            )

    # แสดงหลายภาพในกริด 3 คอลัมน์
    else:

        IMAGES_PER_ROW = 3

        for start in range(
            0,
            len(files_to_analyze),
            IMAGES_PER_ROW
        ):

            row_files = files_to_analyze[
                start:start + IMAGES_PER_ROW
            ]

            columns = st.columns(
                IMAGES_PER_ROW,
                gap="medium"
            )

            for column, uploaded_file in zip(
                columns,
                row_files
            ):

                with column:

                    with st.container(border=True):

                        try:
                            preview_image = Image.open(
                                uploaded_file
                            ).convert("RGB")

                            # ปรับเฉพาะภาพตัวอย่างที่แสดงบนเว็บ
                            # ไม่แก้ไขไฟล์ภาพต้นฉบับ
                            preview_image.thumbnail(
                                (420, 280),
                                Image.Resampling.LANCZOS
                            )

                            st.image(
                                preview_image,
                                use_container_width=True
                            )

                            st.markdown(
                                f"""
                                <div style="
                                    color: #174C46;
                                    font-size: 13px;
                                    font-weight: 650;
                                    text-align: center;
                                    overflow-wrap: anywhere;
                                    padding: 8px 2px 3px 2px;
                                    min-height: 42px;
                                ">
                                    {uploaded_file.name}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        except Exception:
                            st.warning(
                                f"Unable to display "
                                f"{uploaded_file.name}"
                            )




# ==================================================
# 11. ANALYZE IMAGES
# ==================================================

if files_to_analyze and analyze_clicked:

    image_arrays = []
    valid_files = []
    failed_files = []

    # ----------------------------------------------
    # PREPROCESS ALL IMAGES
    # ----------------------------------------------

    with st.spinner("Preparing images..."):

        for uploaded_file in files_to_analyze:
            try:
                image_array = preprocess_image(uploaded_file)

                image_arrays.append(image_array)
                valid_files.append(uploaded_file)

            except Exception as error:
                failed_files.append({
                    "Filename": uploaded_file.name,
                    "Error": str(error)
                })

    if not image_arrays:
        st.error("ไม่มีภาพที่สามารถนำไปวิเคราะห์ได้")
        st.stop()

    # ----------------------------------------------
    # BATCH PREDICTION
    # ----------------------------------------------

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.write(
            f"Analyzing {len(image_arrays)} image(s)..."
        )

        predictions = predict_images(image_arrays)

        progress_bar.progress(100)
        status_text.success("Analysis completed!")

    except Exception as error:
        st.error("เกิดข้อผิดพลาดระหว่างการวิเคราะห์")
        st.code(str(error))
        st.stop()

    # ----------------------------------------------
    # SAVE RESULTS
    # ----------------------------------------------

    result_rows = []

    for uploaded_file, prediction in zip(
        valid_files,
        predictions
    ):
        result_rows.append({
            "Filename": uploaded_file.name,
            "Predicted Variety": prediction["name"],
            "Class Code": prediction["code"],
            "Confidence (%)": round(
                prediction["confidence"], 2
            )
        })

    st.session_state["betta_results"] = result_rows
    st.session_state["betta_predictions"] = predictions

    # เก็บไฟล์ไว้ใช้แสดงภาพ โดยไม่แก้ไขภาพต้นฉบับ
    st.session_state["betta_files"] = valid_files

    st.session_state["betta_failed_files"] = failed_files


# ==================================================
# 12. DISPLAY ANALYSIS RESULTS
# ==================================================

if "betta_results" in st.session_state:

    result_rows = st.session_state["betta_results"]
    predictions = st.session_state["betta_predictions"]
    result_filenames = st.session_state["betta_files"]

    st.divider()
    st.header("Analysis Results")

    st.write(
        f"Successfully analyzed **{len(result_rows)}** image(s)."
    )


    # ----------------------------------------------
    # SINGLE IMAGE RESULT - MODERN UI
    # ----------------------------------------------

    if len(result_rows) == 1:

        result = result_rows[0]
        prediction = predictions[0]

        col_image, col_result = st.columns(
            [1, 1],
            gap="large"
        )

        # ------------------------------------------
        # LEFT: FISH IMAGE CARD
        # ------------------------------------------

        with col_image:

            current_file = next(
                (
                    file for file in files_to_analyze
                    if file.name == result["Filename"]
                ),
                None
            )

            st.markdown("""
                <div style="
                    color: #174C46;
                    font-size: 18px;
                    font-weight: 750;
                    margin-bottom: 12px;
                ">
                    🐟 Fish Image
                </div>
            """, unsafe_allow_html=True)

            with st.container(border=True):

                if current_file is not None:
                    try:
                        st.image(
                            Image.open(current_file).convert("RGB"),
                            use_container_width=True
                        )
                    except Exception:
                        st.warning("Unable to display this image.")

                st.markdown(
                    f"""
                    <div style="
                        color: #74847D;
                        font-size: 13px;
                        padding-top: 8px;
                        overflow-wrap: anywhere;
                    ">
                        📁 {result["Filename"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # ------------------------------------------
        # RIGHT: PREDICTION RESULT CARD
        # ------------------------------------------

        with col_result:

            confidence = prediction["confidence"]
            confidence_bar = max(0, min(100, confidence))

            st.markdown("""
                <div style="
                    color: #174C46;
                    font-size: 18px;
                    font-weight: 750;
                    margin-bottom: 12px;
                ">
                    ✨ AI Prediction
                </div>
            """, unsafe_allow_html=True)

            with st.container(border=True):

                st.markdown("""
                    <div style="
                        color: #74847D;
                        font-size: 12px;
                        font-weight: 700;
                        letter-spacing: 1.4px;
                        text-transform: uppercase;
                    ">
                        Predicted Variety
                    </div>
                """, unsafe_allow_html=True)

                st.markdown(
                    f"""
                    <div style="
                        color: #174C46;
                        font-size: 30px;
                        font-weight: 800;
                        margin-top: 8px;
                        line-height: 1.3;
                    ">
                        {prediction["name"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <div style="
                        display: inline-block;
                        background: #EAF4EF;
                        color: #174C46;
                        padding: 5px 12px;
                        border-radius: 8px;
                        font-size: 12px;
                        font-weight: 700;
                        margin-top: 8px;
                    ">
                        CLASS {prediction["code"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <div style="
                        color: #397B69;
                        font-size: 29px;
                        font-weight: 800;
                        margin-top: 24px;
                    ">
                        {confidence:.2f}%
                    </div>
                    <div style="
                        color: #74847D;
                        font-size: 13px;
                        margin-bottom: 10px;
                    ">
                        Model Confidence
                    </div>
                    <div style="
                        width: 100%;
                        height: 9px;
                        background: #EAF4EF;
                        border-radius: 20px;
                        overflow: hidden;
                    ">
                        <div style="
                            width: {confidence_bar}%;
                            height: 9px;
                            background: #397B69;
                            border-radius: 20px;
                        "></div>
                    </div>
                    <div style="
                        color: #74847D;
                        font-size: 13px;
                        margin-top: 14px;
                    ">
                        {prediction["subtitle"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # ------------------------------------------
        # CLASS PROBABILITIES
        # ------------------------------------------

        st.markdown("""
            <div style="
                color: #174C46;
                font-size: 21px;
                font-weight: 750;
                margin-top: 26px;
                margin-bottom: 5px;
            ">
                📊 Class Probabilities
            </div>
            <div style="
                color: #74847D;
                font-size: 13px;
                margin-bottom: 14px;
            ">
                Compare the model's output across all fish varieties.
            </div>
        """, unsafe_allow_html=True)

        probability_rows = [
            {
                "Variety": name,
                "Probability (%)": probability
            }
            for name, probability
            in prediction["probabilities"].items()
        ]

        probability_rows.sort(
            key=lambda item: item["Probability (%)"],
            reverse=True
        )

        # แสดงผลเป็นตาราง
        st.dataframe(
            pd.DataFrame(probability_rows),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Variety": st.column_config.TextColumn(
                    "Fish Variety"
                ),
                "Probability (%)": st.column_config.ProgressColumn(
                    "Probability",
                    format="%.2f%%",
                    min_value=0,
                    max_value=100
                )
            }
        )


    # ----------------------------------------------
    # MULTIPLE IMAGE RESULTS - CONSISTENT UI
    # ----------------------------------------------

    else:
        st.markdown("""
            <div style="
                color: #174C46;
                font-size: 24px;
                font-weight: 800;
                margin-top: 12px;
            ">
                Individual Results
            </div>
            <div style="
                color: #74847D;
                font-size: 14px;
                margin-bottom: 18px;
            ">
                Each image is displayed in a consistent card layout.
            </div>
        """, unsafe_allow_html=True)

        # สรุปผลลัพธ์ทั้งหมด
        with st.container(border=True):
            st.markdown("#### 📊 Prediction Summary")

            st.dataframe(
                pd.DataFrame(result_rows),
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Filename": st.column_config.TextColumn(
                        "Image Filename",
                        width="large"
                    ),
                    "Predicted Variety": st.column_config.TextColumn(
                        "Predicted Variety",
                        width="medium"
                    ),
                    "Class Code": st.column_config.TextColumn(
                        "Class Code",
                        width="small"
                    ),
                    "Confidence (%)": st.column_config.ProgressColumn(
                        "Confidence",
                        format="%.2f%%",
                        min_value=0,
                        max_value=100
                    )
                }
            )

        st.markdown("#### 🐟 Detailed Analysis")

        # แสดงการ์ดแต่ละภาพในรูปแบบเดียวกัน
        for index, (result, prediction) in enumerate(
            zip(result_rows, predictions)
        ):
            with st.container(border=True):

                # หัวการ์ด: ลำดับและชื่อไฟล์
                head_col, badge_col = st.columns([3, 1])

                with head_col:
                    st.markdown(
                        f"""
                        <div style="
                            color: #174C46;
                            font-size: 17px;
                            font-weight: 750;
                            overflow-wrap: anywhere;
                        ">
                            Image {index + 1}
                        </div>
                        <div style="
                            color: #74847D;
                            font-size: 12px;
                            margin-top: 4px;
                            overflow-wrap: anywhere;
                        ">
                            {result["Filename"]}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with badge_col:
                    st.markdown(
                        f"""
                        <div style="
                            background: #EAF4EF;
                            color: #174C46;
                            text-align: center;
                            padding: 7px 8px;
                            border-radius: 9px;
                            font-weight: 750;
                            font-size: 12px;
                        ">
                            {prediction["code"]}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown(
                    "<hr style='margin: 14px 0;'>",
                    unsafe_allow_html=True
                )

                # เนื้อหาการ์ด: ภาพและผลทำนาย
                image_col, detail_col = st.columns(
                    [1, 1.35],
                    gap="large"
                )

                with image_col:
                    matched_file = (
                        st.session_state["betta_files"][index]
                        if index < len(st.session_state["betta_files"])
                        else None
                    )

                    if matched_file is not None:
                        try:
                            fish_image = Image.open(
                                matched_file
                            ).convert("RGB")

                            # ใช้ขนาดภาพแสดงผลเท่ากันทุกการ์ด
                            st.image(
                                fish_image,
                                width="stretch"
                            )

                        except Exception:
                            st.warning("Image preview unavailable")

                with detail_col:
                    st.markdown(
                        """
                        <div style="
                            color: #74847D;
                            font-size: 11px;
                            font-weight: 700;
                            letter-spacing: 1px;
                        ">
                            PREDICTED VARIETY
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f"""
                        <div style="
                            color: #174C46;
                            font-size: 23px;
                            font-weight: 800;
                            margin-top: 5px;
                            overflow-wrap: anywhere;
                        ">
                            {prediction["name"]}
                        </div>
                        <div style="
                            color: #74847D;
                            font-size: 13px;
                            margin-top: 5px;
                            min-height: 36px;
                        ">
                            {prediction["subtitle"]}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    confidence = max(
                        0,
                        min(100, prediction["confidence"])
                    )

                    st.markdown(
                        f"""
                        <div style="
                            color: #397B69;
                            font-size: 25px;
                            font-weight: 800;
                            margin-top: 12px;
                        ">
                            {confidence:.2f}%
                        </div>
                        <div style="
                            color: #74847D;
                            font-size: 12px;
                            margin-bottom: 8px;
                        ">
                            Model Confidence
                        </div>
                        <div style="
                            width: 100%;
                            height: 8px;
                            background: #EAF4EF;
                            border-radius: 20px;
                            overflow: hidden;
                        ">
                            <div style="
                                width: {confidence}%;
                                height: 8px;
                                background: #397B69;
                                border-radius: 20px;
                            "></div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # รายละเอียดความน่าจะเป็น
                with st.expander("View all class probabilities"):
                    probability_rows = [
                        {
                            "Variety": name,
                            "Probability (%)": probability
                        }
                        for name, probability
                        in prediction["probabilities"].items()
                    ]

                    probability_rows.sort(
                        key=lambda item: item["Probability (%)"],
                        reverse=True
                    )

                    st.dataframe(
                        pd.DataFrame(probability_rows),
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "Variety": st.column_config.TextColumn(
                                "Fish Variety"
                            ),
                            "Probability (%)":
                                st.column_config.ProgressColumn(
                                    "Probability",
                                    format="%.2f%%",
                                    min_value=0,
                                    max_value=100
                                )
                        }
                    )

    # ----------------------------------------------
    # DOWNLOAD CSV
    # ----------------------------------------------

    st.divider()
    st.subheader("Export Results")

    csv_buffer = io.StringIO()

    writer = csv.DictWriter(
        csv_buffer,
        fieldnames=[
            "Filename",
            "Predicted Variety",
            "Class Code",
            "Confidence (%)"
        ]
    )

    writer.writeheader()
    writer.writerows(result_rows)

    csv_data = csv_buffer.getvalue().encode("utf-8-sig")

    st.download_button(
        label="Download Results as CSV",
        data=csv_data,
        file_name="betta_classification_results.csv",
        mime="text/csv",
        use_container_width=True
    )

    # ----------------------------------------------
    # SHOW FILE ERRORS
    # ----------------------------------------------

    failed_files = st.session_state.get(
        "betta_failed_files", []
    )

    if failed_files:
        st.warning(
            f"{len(failed_files)} image(s) could not be processed."
        )

        st.dataframe(
            pd.DataFrame(failed_files),
            hide_index=True,
            use_container_width=True
        )


# ==================================================
# 13. ABOUT
# ==================================================

st.divider()

with st.expander("About BettaVision"):
    st.write(
        "BettaVision is an image classification application "
        "designed to predict betta fish varieties using a "
        "trained deep learning model."
    )

    st.write("Supported classes:")

    for item in CLASS_INFO:
        st.write(
            f"- **{item['name']} ({item['code']})**: "
            f"{item['subtitle']}"
        )

    st.caption(
        f"Model input size: {image_width} × {image_height} pixels"
    )
