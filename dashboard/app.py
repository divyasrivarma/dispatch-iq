import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import cv2
import easyocr
import tempfile
import streamlit as st
from ultralytics import YOLO


MODEL_PATH = "models/gunny_bag_detector.pt"
OUTPUT_DIR = "outputs/dashboard_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

st.set_page_config(
    page_title="Dispatch IQ",
    layout="wide"
)


# -------------------- CSS --------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #eef4fb !important;
        color: #1e293b !important;
    }

    .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: #1e293b !important;
    }

    .main-header {
        background-color: #ffffff;
        border: 1px solid #d7e3f2;
        border-radius: 18px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.08);
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #0f172a !important;
        margin-bottom: 8px;
    }

    .main-subtitle {
        font-size: 17px;
        color: #475569 !important;
    }

    .section-card {
        background-color: #ffffff;
        border: 1px solid #d7e3f2;
        border-radius: 18px;
        padding: 26px;
        margin-top: 22px;
        margin-bottom: 22px;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.08);
    }

    .result-card {
        background-color: #ffffff;
        border: 1px solid #d7e3f2;
        border-radius: 18px;
        padding: 26px;
        margin-top: 22px;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.08);
    }

    .success-box {
        background-color: #dcfce7;
        color: #166534 !important;
        border: 1px solid #86efac;
        padding: 16px;
        border-radius: 12px;
        font-weight: 700;
        margin-top: 12px;
        margin-bottom: 12px;
    }

    .alert-box {
        background-color: #fee2e2;
        color: #991b1b !important;
        border: 1px solid #fca5a5;
        padding: 16px;
        border-radius: 12px;
        font-weight: 700;
        margin-top: 12px;
        margin-bottom: 12px;
    }

    .warning-box {
        background-color: #fef3c7;
        color: #92400e !important;
        border: 1px solid #fcd34d;
        padding: 16px;
        border-radius: 12px;
        font-weight: 700;
        margin-top: 12px;
        margin-bottom: 12px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: #ffffff;
        border: 1px solid #d7e3f2;
        border-radius: 16px;
        padding: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #f1f5f9;
        border-radius: 12px;
        padding: 12px 24px;
        color: #1e293b !important;
        font-weight: 700;
    }

    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: #ffffff !important;
    }

    .stButton > button {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.65rem 1.2rem !important;
        font-weight: 700 !important;
    }

    .stButton > button:hover {
        background-color: #1d4ed8 !important;
        color: #ffffff !important;
    }

    .stDownloadButton > button {
        background-color: #059669 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.65rem 1.2rem !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetric"] {
        background-color: #f8fafc;
        border: 1px solid #d7e3f2;
        padding: 18px;
        border-radius: 14px;
    }

    [data-testid="stFileUploader"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 14px !important;
        padding: 14px !important;
    }

    [data-testid="stFileUploader"] section {
        background-color: #f8fafc !important;
        border: 2px dashed #94a3b8 !important;
        border-radius: 12px !important;
    }

    [data-testid="stFileUploader"] button {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        border: none !important;
    }

    [data-testid="stFileUploader"] small {
        color: #475569 !important;
    }

    [data-testid="stFileUploader"] div {
        color: #1e293b !important;
    }

    [data-testid="stCameraInput"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 14px !important;
        padding: 14px !important;
    }

    [data-testid="stCameraInput"] button {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        border: none !important;
        font-weight: 700 !important;
    }

    [data-testid="stCameraInput"] video {
        border-radius: 14px !important;
    }

    video {
        border-radius: 14px;
    }

    img {
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -------------------- HEADER --------------------
st.markdown(
    """
    <div class="main-header">
        <div class="main-title">Dispatch IQ - Gunny Bag Label Monitoring</div>
        <div class="main-subtitle">
            Detect gunny bags, read label text using OCR, and alert when no label text is found.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# -------------------- SESSION STATE KEYS --------------------
if "image_uploader_key" not in st.session_state:
    st.session_state.image_uploader_key = 0

if "video_uploader_key" not in st.session_state:
    st.session_state.video_uploader_key = 0

if "camera_key" not in st.session_state:
    st.session_state.camera_key = 0


# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


@st.cache_resource
def load_ocr():
    return easyocr.Reader(["en"], gpu=False)


model = load_model()
reader = load_ocr()


# -------------------- IMAGE FUNCTION --------------------
def detect_and_read_image(image_path, output_name):
    image = cv2.imread(image_path)

    if image is None:
        return None, {
            "bag_count": 0,
            "status": "Error: Could not read image",
            "texts": []
        }

    results = model(image)[0]

    bag_count = 0
    extracted_texts = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        conf = float(box.conf[0])

        if class_name == "gunny_bag" and conf > 0.3:
            bag_count += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            h, w = image.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            bag_crop = image[y1:y2, x1:x2]

            if bag_crop.size > 0:
                ocr_result = reader.readtext(bag_crop)
                text = " ".join([item[1] for item in ocr_result])

                if text.strip():
                    extracted_texts.append(text)

            cv2.rectangle(image, (x1, y1), (x2, y2), (22, 163, 74), 2)

            cv2.putText(
                image,
                f"gunny_bag {conf:.2f}",
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (22, 163, 74),
                2,
            )

    if bag_count == 0:
        status = "No Gunny Bag Found"
    elif len(extracted_texts) == 0:
        status = "ALERT: No Label/Text Found"
    else:
        status = "Label/Text Found"

    cv2.putText(
        image,
        status,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255) if "ALERT" in status else (22, 163, 74),
        2,
    )

    output_path = os.path.join(OUTPUT_DIR, output_name)
    cv2.imwrite(output_path, image)

    return output_path, {
        "bag_count": bag_count,
        "status": status,
        "texts": extracted_texts
    }


# -------------------- VIDEO FUNCTION --------------------
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None, {
            "status": "Error: Could not open video",
            "total_frames": 0,
            "processed_frames": 0,
            "total_bags": 0
        }

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps == 0:
        fps = 20

    output_path = os.path.join(OUTPUT_DIR, "video_result.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 0
    total_bags = 0
    frame_id = 0

    progress_bar = st.progress(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_id += 1

        if frame_id % 5 == 0:
            results = model(frame)[0]
            frame_bag_count = 0

            for box in results.boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                conf = float(box.conf[0])

                if class_name == "gunny_bag" and conf > 0.3:
                    frame_bag_count += 1
                    total_bags += 1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (22, 163, 74), 2)

                    cv2.putText(
                        frame,
                        f"gunny_bag {conf:.2f}",
                        (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (22, 163, 74),
                        2,
                    )

            cv2.putText(
                frame,
                f"Bags: {frame_bag_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (22, 163, 74),
                2,
            )

            processed_frames += 1

        out.write(frame)

        if total_frames > 0:
            progress_bar.progress(min(frame_id / total_frames, 1.0))

    cap.release()
    out.release()

    return output_path, {
        "status": "Video Processed Successfully",
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "total_bags": total_bags
    }


# -------------------- RESULT DISPLAY --------------------
def show_image_results(output_path, result):
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)

    st.subheader("Detection Result")

    col1, col2, col3 = st.columns(3)

    col1.metric("Gunny Bags Detected", result["bag_count"])
    col2.metric("Status", result["status"])
    col3.metric("OCR Text Count", len(result["texts"]))

    if "ALERT" in result["status"]:
        st.markdown(f"<div class='alert-box'>{result['status']}</div>", unsafe_allow_html=True)
    elif result["status"] == "No Gunny Bag Found":
        st.markdown(f"<div class='warning-box'>{result['status']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='success-box'>{result['status']}</div>", unsafe_allow_html=True)

    st.subheader("Extracted Text")

    if result["texts"]:
        for i, text in enumerate(result["texts"], start=1):
            st.write(f"Text {i}: {text}")
    else:
        st.write("No readable text found.")

    if output_path:
        st.subheader("Output Image")
        st.image(output_path, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# -------------------- TABS --------------------
tab1, tab2, tab3 = st.tabs(
    ["Image Upload", "Video Upload", "Web Camera"]
)


# -------------------- IMAGE UPLOAD --------------------
with tab1:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)

    st.subheader("Image Upload")
    st.write("Upload a JPG image for gunny bag detection and OCR.")

    uploaded_image = st.file_uploader(
        "Choose JPG image",
        type=["jpg"],
        key=f"image_upload_{st.session_state.image_uploader_key}"
    )

    if st.button("Upload Another Image", key="reset_image_button"):
        st.session_state.image_uploader_key += 1
        st.rerun()

    if uploaded_image is not None:
        temp_dir = tempfile.mkdtemp()
        image_path = os.path.join(temp_dir, uploaded_image.name)

        with open(image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())

        st.image(image_path, caption="Uploaded Image", use_container_width=True)

        if st.button("Run Image Detection and OCR", key="image_detect_button"):
            with st.spinner("Detecting gunny bags and reading text..."):
                output_path, result = detect_and_read_image(
                    image_path,
                    "image_result.jpg"
                )

            show_image_results(output_path, result)

    st.markdown("</div>", unsafe_allow_html=True)


# -------------------- VIDEO UPLOAD --------------------
with tab2:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)

    st.subheader("Video Upload")
    st.write("Upload an MP4 video for gunny bag detection.")

    uploaded_video = st.file_uploader(
        "Choose MP4 video",
        type=["mp4"],
        key=f"video_upload_{st.session_state.video_uploader_key}"
    )

    if st.button("Upload Another Video", key="reset_video_button"):
        st.session_state.video_uploader_key += 1
        st.rerun()

    if uploaded_video is not None:
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, uploaded_video.name)

        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        st.video(video_path)

        if st.button("Run Video Detection", key="video_detect_button"):
            with st.spinner("Processing video. This may take time on CPU."):
                output_video_path, video_result = process_video(video_path)

            st.markdown("<div class='result-card'>", unsafe_allow_html=True)

            st.subheader("Video Detection Result")

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Frames", video_result["total_frames"])
            col2.metric("Processed Frames", video_result["processed_frames"])
            col3.metric("Total Bag Detections", video_result["total_bags"])

            st.markdown(
                f"<div class='success-box'>{video_result['status']}</div>",
                unsafe_allow_html=True
            )

            if output_video_path:
                st.subheader("Processed Video")
                st.video(output_video_path)

                with open(output_video_path, "rb") as f:
                    st.download_button(
                        label="Download Processed Video",
                        data=f,
                        file_name="dispatch_iq_video_result.mp4",
                        mime="video/mp4"
                    )

            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# -------------------- WEB CAMERA --------------------
with tab3:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)

    st.subheader("Web Camera")
    st.write("Step 1: Take a photo. Step 2: Run detection on the captured image.")

    camera_image = st.camera_input(
        "Capture image from camera",
        key=f"camera_input_{st.session_state.camera_key}"
    )

    if st.button("Take Another Photo", key="reset_camera_button"):
        st.session_state.camera_key += 1
        st.rerun()

    if camera_image is not None:
        temp_dir = tempfile.mkdtemp()
        camera_path = os.path.join(temp_dir, "camera_capture.jpg")

        with open(camera_path, "wb") as f:
            f.write(camera_image.getbuffer())

        st.image(camera_path, caption="Captured Image", use_container_width=True)

        st.success("Image captured successfully.")

        if st.button("Run Camera Detection and OCR", key="camera_detect_button"):
            with st.spinner("Detecting gunny bags and reading text..."):
                output_path, result = detect_and_read_image(
                    camera_path,
                    "camera_result.jpg"
                )

            show_image_results(output_path, result)
    else:
        st.info("Take a photo to run camera detection.")

    st.markdown("</div>", unsafe_allow_html=True)
