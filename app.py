import io
import os
import time
from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import torch
from ultralytics import YOLO


@st.cache_resource
def load_model(weights_path: str):
    return YOLO(weights_path)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def get_device() -> int | str:
    # device=0 forces GPU if CUDA is available; otherwise use CPU
    return 0 if torch.cuda.is_available() else "cpu"


st.set_page_config(page_title="Crack Detection (YOLOv8)", page_icon="🧱", layout="centered")
st.title("🧱 Concrete Crack Detection (YOLOv8) — Offline Deployment")

project_dir = Path(__file__).resolve().parent
weights_path = project_dir / "weights" / "best.pt"
outputs_dir = project_dir / "outputs"
ensure_dir(outputs_dir)

if not weights_path.exists():
    st.error(f"Model weights not found: {weights_path}")
    st.stop()

# --- Sidebar (show GPU status early) ---
device = get_device()
st.sidebar.header("System")
st.sidebar.write("torch:", torch.__version__)
st.sidebar.write("cuda available:", torch.cuda.is_available())
st.sidebar.write("device used:", device)
if torch.cuda.is_available():
    st.sidebar.write("gpu:", torch.cuda.get_device_name(0))

# --- Helper function to place near the top or inside the script ---
def convert_video_to_h264(input_path, output_path):
    """
    Converts video to H.264 (browser compatible) using FFmpeg.
    Returns True if successful, False if FFmpeg is missing.
    """
    command = [
        "ffmpeg", "-y",             # Overwrite output file
        "-i", str(input_path),      # Input file
        "-vcodec", "libx264",       # H.264 Video Codec (Browser Friendly)
        "-acodec", "aac",           # AAC Audio Codec
        str(output_path)
    ]
    try:
        # Run ffmpeg command
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

# Load model
model = load_model(str(weights_path))

mode = st.radio("Choose input type", ["Image", "Video"], horizontal=True)

conf = st.slider("Confidence threshold", 0.05, 0.90, 0.25, 0.05)
imgsz = st.selectbox("Image size (imgsz)", [320, 480, 640, 768, 960, 1024], index=2)

st.caption("Runs fully offline on your laptop. Upload input → YOLOv8 inference → results shown.")

# -----------------------
# IMAGE MODE
# -----------------------
if mode == "Image":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        st.info("Upload an image to run detection.")
        st.stop()

    img_bytes = uploaded.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_np = np.array(pil_img)

    st.subheader("Input Image")
    st.image(pil_img, use_container_width=True)

    with st.spinner("Running detection..."):
        t0 = time.time()
        results = model.predict(
            source=img_np,
            conf=conf,
            imgsz=imgsz,
            device=device,      # ✅ force GPU if available
            verbose=False
        )
        dt = time.time() - t0

    r0 = results[0]
    annotated_rgb = r0.plot()[:, :, ::-1]  # BGR -> RGB

    st.subheader("Detection Result")
    st.image(annotated_rgb, use_container_width=True)

    n_boxes = 0 if r0.boxes is None else len(r0.boxes)
    st.success(f"Detected {n_boxes} crack(s) | Time: {dt:.2f}s")

# -----------------------
# VIDEO MODE
# -----------------------
else:
    uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

    if uploaded is None:
        st.info("Upload a short video (10–20 seconds recommended).")
        st.stop()

    temp_in = outputs_dir / "temp_input_video.mp4"
    with open(temp_in, "wb") as f:
        f.write(uploaded.read())

    st.subheader("Processing Video (Ultralytics pipeline)")
    run_name = f"st_video_{int(time.time())}"
    run_dir = outputs_dir / run_name
    ensure_dir(run_dir)

    with st.spinner("Running YOLO on video..."):
        t0 = time.time()
        # Run YOLO inference
        _ = model.predict(
            source=str(temp_in),
            conf=conf,
            imgsz=imgsz,
            device=device,
            save=True,
            project=str(outputs_dir),
            name=run_name,
            exist_ok=True,
            verbose=False
        )
        dt = time.time() - t0

    # Find the YOLO output file
    saved_videos = list(run_dir.glob("*.mp4")) + list(run_dir.glob("*.avi")) + list(run_dir.glob("*.mov")) + list(run_dir.glob("*.mkv"))
    if not saved_videos:
        st.error(f"Could not find output video in: {run_dir}")
        st.stop()
    
    raw_output_path = saved_videos[0]
    
    # --- NEW: CONVERSION STEP ---
    converted_path = run_dir / "browser_compatible_out.mp4"
    
    with st.spinner("Converting video for browser playback..."):
        success = convert_video_to_h264(raw_output_path, converted_path)
    
    st.success(f"✅ Done in {dt:.1f}s")

    # Display Video
    st.subheader("Output Video (Annotated)")
    
    if success:
        # If conversion worked, show the converted H.264 video
        st.video(converted_path.read_bytes())
        final_path = converted_path
    else:
        # Fallback if FFmpeg is missing: Show error but allow download
        st.warning("⚠️ Video processed, but FFmpeg was not found to convert it for browser playback. You can still download and play it in VLC.")
        final_path = raw_output_path

    # Download Button
    with open(final_path, "rb") as f:
        st.download_button(
            "⬇️ Download annotated video",
            data=f,
            file_name="detected_cracks.mp4",
            mime="video/mp4"
        )

    # Cleanup input
    try:
        os.remove(temp_in)
    except:
        pass
