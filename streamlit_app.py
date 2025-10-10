"""
streamlit_app.py

Streamlit UI for:
- uploading labeled face images into dataset/<name>/
- building embeddings (calls build_embeddings from create_embeddings.py)
- live webcam recognition using saved embeddings (displayed in Streamlit)

Notes:
- Run locally (Streamlit + OpenCV access to local webcam)
- Ensure create_embeddings.build_embeddings and utils are in the same project path
"""

import os
import time
import glob
import uuid
from pathlib import Path
from typing import Tuple

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import joblib
import torch

# Import your existing functions
# Assumes create_embeddings.py exposes build_embeddings()
# and utils.py contains required helper functions used for realtime recognition.
from create_embeddings import build_embeddings
from utils import (
    mediapipe_face_landmarks,
    upper_face_bbox,
    face_bbox_from_landmarks,
    crop_region,
    preprocess_face_crop,
    load_facenet,
    embedding_from_image,
    cosine_similarity,
)

# Constants
DATASET_DIR = "dataset"
EMBED_PATH = "models/embeddings.joblib"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.55  # adjust as needed

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

st.set_page_config(page_title="Masked Face Labeller & Live Recognizer", layout="wide")

st.title("📸 Live Data Labelling & Recognition (Streamlit)")

# --- Sidebar / Controls ---
st.sidebar.header("Actions")

# uploader / label
st.header("1) Upload and label images")
col1, col2 = st.columns([1, 1])
with col1:
    name = st.text_input("Person name (label)", key="name_input")
    uploaded_files = st.file_uploader(
        "Upload images (multiple)", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )
    save_button = st.button("Save sample(s) to dataset")

with col2:
    st.write("Preview uploaded images")
    if uploaded_files:
        for f in uploaded_files:
            img = Image.open(f)
            st.image(img, width=160)
    else:
        st.info("No images uploaded yet")

# Save logic: store uploaded images into dataset/<name>/
if save_button:
    if not name:
        st.error("Please enter a person name before saving.")
    elif not uploaded_files:
        st.error("Please upload at least one image to save.")
    else:
        dest_dir = os.path.join(DATASET_DIR, name)
        os.makedirs(dest_dir, exist_ok=True)
        saved = 0
        for f in uploaded_files:
            try:
                pil = Image.open(f).convert("RGB")
                filename = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.jpg"
                dest_path = os.path.join(dest_dir, filename)
                pil.save(dest_path, quality=95)
                saved += 1
            except Exception as e:
                st.warning(f"Failed to save {f.name}: {e}")
        st.success(f"Saved {saved} images to `{dest_dir}`")

st.markdown("---")

# --- Build embeddings (train) ---
st.header("2) Build embeddings (train)")
st.write("After you uploaded and saved all images for each person, click the button below to compute embeddings.")
build_btn = st.button("📦 Build Embeddings (run build_embeddings)")

if build_btn:
    with st.spinner("Building embeddings (this may take some time)..."):
        try:
            # call your existing function; it will save models/embeddings.joblib
            build_embeddings(dataset_dir=DATASET_DIR, out_path=EMBED_PATH)
            st.success(f"Embeddings built and saved to `{EMBED_PATH}`")
        except Exception as e:
            st.error(f"Error while building embeddings: {e}")

st.markdown("---")

# --- Live recognition area ---
st.header("3) Live recognition (webcam)")
st.write(
    "Start webcam to test recognition. Embeddings file must exist (`models/embeddings.joblib`). "
    "This uses the same matching logic as your recognize_realtime.py but streams frames to Streamlit."
)

col_live = st.columns([1, 3])
with col_live[0]:
    start_live = st.button("▶️ Start Live Recognition")
    stop_live = st.button("⏹ Stop Live Recognition")
    similarity_threshold = st.slider("Similarity threshold", 0.30, 0.90, THRESHOLD, 0.01)

with col_live[1]:
    # placeholder for image
    image_placeholder = st.empty()
    info_placeholder = st.empty()

# Session state for run control
if "live_running" not in st.session_state:
    st.session_state.live_running = False

if start_live:
    # only start when embeddings exist
    if not Path(EMBED_PATH).exists():
        st.error(f"Embeddings not found at `{EMBED_PATH}`. Run Build Embeddings first.")
    else:
        st.session_state.live_running = True

if stop_live:
    st.session_state.live_running = False

# Live loop
if st.session_state.live_running:
    # load db & model
    try:
        db = joblib.load(EMBED_PATH)
        names = db.get("names", [])
        embs = db.get("embeddings", np.empty((0, 512)))
    except Exception as e:
        st.error(f"Failed to load embeddings: {e}")
        st.session_state.live_running = False
        names, embs = [], np.empty((0, 512))

    if len(names) == 0:
        st.error("No names in DB. Add images and build embeddings first.")
        st.session_state.live_running = False
    else:
        # initialize facenet
        model = load_facenet(device=DEVICE)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Unable to open webcam. Make sure no other app is using it.")
            st.session_state.live_running = False
        else:
            info_placeholder.info("Press Stop to end the stream.")
            try:
                while st.session_state.live_running:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # process frame (detect landmarks)
                    lm = mediapipe_face_landmarks(frame, static_mode=False)
                    label = "No Face"
                    conf = 0.0
                    box = None
                    if lm is not None:
                        bbox_upper = upper_face_bbox(lm, expand=1.25, img_shape=frame.shape)
                        crop = crop_region(frame, bbox_upper)
                        if crop is None or crop.size == 0:
                            bbox = face_bbox_from_landmarks(lm, expand=1.2, img_shape=frame.shape)
                            crop = crop_region(frame, bbox)
                            box = bbox
                        else:
                            box = bbox_upper
                        if crop is not None and crop.size != 0:
                            proc = preprocess_face_crop(crop, size=160)
                            emb = embedding_from_image(proc, model, device=DEVICE)
                            sims = [cosine_similarity(emb, db_emb) for db_emb in embs]
                            best_idx = int(np.argmax(sims))
                            best_sim = sims[best_idx]
                            if best_sim >= similarity_threshold:
                                label = names[best_idx]
                                conf = best_sim
                            else:
                                label = "Unknown"
                                conf = best_sim
                    # draw box & label on frame for preview
                    if box is not None:
                        x1, y1, x2, y2 = box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            f"{label} {conf:.2f}",
                            (max(0, x1), max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2,
                        )
                    else:
                        cv2.putText(frame, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

                    # convert BGR->RGB for Streamlit
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

                    # small sleep
                    if not st.session_state.live_running:
                        break
                    time.sleep(0.01)

            except Exception as e:
                st.error(f"Error in live loop: {e}")
            finally:
                cap.release()
                st.session_state.live_running = False
                info_placeholder.info("Stream stopped.")
else:
    image_placeholder.empty()
    info_placeholder.empty()

st.markdown("---")
st.write("💡 Tips:")
st.write(
    "- Capture multiple images per person with different angles/lighting.\n"
    "- For masked faces prefer upper-face captures (eyes/forehead) — `upper_face_bbox` is used.\n"
    "- Tune the similarity threshold slider to balance false accepts vs false rejects.\n"
)

st.caption("Built with your existing utils.py and create_embeddings.py. Run locally — Streamlit cannot access remote webcams.")
