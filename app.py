import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO
import av
import cv2
import time
import os


st.set_page_config(
    page_title="Live Object Detection & Tracing",
    layout="wide",
    initial_sidebar_state="expanded"
)

SAVE_DIR = "frames"
os.makedirs(SAVE_DIR, exist_ok=True)

st.markdown("""
<style>

.stApp {
    background-color: #f5f0ff;
}

[data-testid="stSidebar"] {
    background-color: #ede4ff;
    border-right: none;
}

[data-testid="stSidebar"] * {
    color: #4b2e83 !important;
}

.control-box {
    background: transparent;
    border: none;
    padding: 0;
    margin-bottom: 10px;
    box-shadow: none;
}

.main-title {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
    color: #4b2e83;
    margin-bottom: 20px;
}

.stSelectbox > div > div {
    background-color: white !important;
    border: 1px solid #cdb4ff !important;
    border-radius: 8px !important;
}

video {
    border-radius: 12px !important;
    border: 2px solid #cdb4ff !important;
}

.status-box {
    background: transparent;
    color: #4b2e83;
    text-align: center;
    font-weight: bold;
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = "yolov8n.pt"

    if not os.path.exists(model_path):
        YOLO(model_path)  # force fresh download

    return YOLO(model_path, task="detect")

model = load_model()
class_names = list(model.names.values())

with st.sidebar:

    st.markdown("""
    <h2 style='text-align:center; color:#4b2e83;'>
        Settings
    </h2>
    """, unsafe_allow_html=True)

    st.markdown('<div class="control-box">', unsafe_allow_html=True)
    save_frames = st.checkbox("💾Save Detected Frames", False)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="control-box">', unsafe_allow_html=True)
    enable_alert = st.toggle("🚨 Enable Alert", True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="control-box">', unsafe_allow_html=True)
    target_object = st.selectbox(
        " Target Object",
        class_names,
        index=0
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="main-title">
    Live Object Detection & Tracing
</div>
""", unsafe_allow_html=True)

class VideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.frame_count = 0

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))

        results = model.track(img, persist=True, verbose=False)

        object_counts = {}
        target_detected = False

        for r in results:

            if r.boxes is not None:

                for box in r.boxes:

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]

                    object_counts[label] = object_counts.get(label, 0) + 1

                    is_target = (label == target_object)

                    color = (0, 0, 255) if is_target else (170, 120, 255)

                    if is_target:
                        target_detected = True

                    cv2.rectangle(
                        img,
                        (x1, y1),
                        (x2, y2),
                        color,
                        3
                    )

                    cv2.putText(
                        img,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2
                    )

        if enable_alert and target_detected:

            alert_text = f"🚨 ALERT: {target_object.upper()} DETECTED"

            cv2.rectangle(
                img,
                (80, 10),
                (560, 60),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                img,
                alert_text,
                (100, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                3
            )

        y = 80

        for obj, cnt in object_counts.items():

            cv2.putText(
                img,
                f"{obj}: {cnt}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            y += 30

        if save_frames and self.frame_count % 30 == 0:

            filename = f"frame_{int(time.time())}.jpg"

            cv2.imwrite(
                os.path.join(SAVE_DIR, filename),
                img
            )

        self.frame_count += 1

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )

webrtc_streamer(
    key="live-detection",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)
