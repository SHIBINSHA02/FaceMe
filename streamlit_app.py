# streamlit_app.py
import os
import time
import uuid
from pathlib import Path

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import joblib
import torch

from utils import (
    mediapipe_face_landmarks,
    face_bbox_from_landmarks,
    crop_region,
    preprocess_face_crop,
    load_facenet,
    embedding_from_image,
    cosine_similarity,
)
from create_embeddings import build_embeddings

DATASET_DIR = "dataset"
EMBED_PATH = "models/embeddings.joblib"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

st.set_page_config(page_title="FaceMe - Full Face Recognition", layout="wide")
st.title("FaceMe — Full Face Recognition System")

# ---------------- Upload or Capture
st.header("1) Add samples")
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload images (manual)")
    name_upload = st.text_input("Person name (for upload)", key="upload_name")
    uploaded_files = st.file_uploader("Select images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if st.button("Save uploaded images"):
        if not name_upload:
            st.error("Enter a name first.")
        elif not uploaded_files:
            st.error("Upload at least one image.")
        else:
            dest = os.path.join(DATASET_DIR, name_upload)
            os.makedirs(dest, exist_ok=True)
            saved = 0
            for f in uploaded_files:
                try:
                    img = Image.open(f).convert("RGB")
                    fname = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.jpg"
                    img.save(os.path.join(dest, fname), quality=95)
                    saved += 1
                except Exception as e:
                    st.warning(f"Failed to save {f.name}: {e}")
            st.success(f"Saved {saved} images to {dest}")

with col2:
    st.subheader("Capture from webcam (recommended)")
    capture_name = st.text_input("Person name (for capture)", key="capture_name")
    num_samples = st.slider("Samples to capture", 5, 60, 25)
    expansion_factor = st.slider("Face crop expansion", 1.1, 1.8, 1.35, 0.05)
    capture_btn = st.button("Start capture (webcam)")

    if capture_btn:
        if not capture_name:
            st.error("Enter a name to capture.")
        else:
            dest = os.path.join(DATASET_DIR, capture_name)
            os.makedirs(dest, exist_ok=True)
            cap = cv2.VideoCapture(0)
            stframe = st.empty()
            saved = 0
            last_saved_time = 0
            st.info("Move your head slightly and make different expressions. Press Stop to abort.")
            try:
                while saved < num_samples:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to read from webcam.")
                        break
                    
                    # Detect full face landmarks
                    lm = mediapipe_face_landmarks(frame, static_mode=False)
                    if lm is not None:
                        # Get FULL FACE bounding box
                        bbox = face_bbox_from_landmarks(lm, expand=expansion_factor, img_shape=frame.shape)
                        crop = crop_region(frame, bbox)
                        
                        if crop is not None and crop.size != 0:
                            # Save every 0.4s for variation
                            now = time.time()
                            if now - last_saved_time > 0.4:
                                fname = f"{int(now)}_{uuid.uuid4().hex[:8]}.jpg"
                                path = os.path.join(dest, fname)
                                # Save full face crop
                                cv2.imwrite(path, crop)
                                saved += 1
                                last_saved_time = now
                            
                            # Draw bounding box
                            x1, y1, x2, y2 = bbox
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"Saved {saved}/{num_samples}", (10, 30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                            cv2.putText(frame, "Full Face", (10, 60), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    stframe.image(frame_rgb, channels="RGB", use_container_width=True)
                    
                st.success(f"✅ Captured {saved} full face images for {capture_name}")
            except Exception as e:
                st.error(f"Error capturing: {e}")
            finally:
                cap.release()

st.markdown("---")

# ---------------- Build embeddings
st.header("2) Build embeddings (Train)")
st.write("Compute embeddings from full face images in `dataset/` and save to `models/embeddings.joblib`.")
col_train = st.columns([2, 1])
with col_train[0]:
    use_masked = st.checkbox("Use masked-finetuned FaceNet if available", value=False)
    build_btn = st.button("Build Embeddings (Train Model)")
with col_train[1]:
    st.write("")

if build_btn:
    with st.spinner("Building embeddings from full face images... (GPU recommended)"):
        try:
            out = build_embeddings(dataset_dir=DATASET_DIR, out_path=EMBED_PATH, masked=use_masked)
            if out is not None:
                st.success(f"✅ Embeddings saved to {out}")
                # Load and show stats
                db = joblib.load(EMBED_PATH)
                names = db.get("names", [])
                embs = db.get("embeddings", [])
                
                st.info(f"📊 Trained on {len(names)} people: {', '.join(names)}")
                
                if len(names) >= 2:
                    # Compute similarity stats
                    sims = []
                    for i in range(len(embs)):
                        for j in range(i+1, len(embs)):
                            sims.append(cosine_similarity(embs[i], embs[j]))
                    if sims:
                        mean_sim = float(np.mean(sims))
                        max_sim = float(np.max(sims))
                        min_sim = float(np.min(sims))
                        suggested = max(0.45, mean_sim * 0.85)
                        st.info(f"📈 Similarity stats - Min: {min_sim:.3f}, Mean: {mean_sim:.3f}, Max: {max_sim:.3f}")
                        st.info(f"💡 Suggested threshold: {suggested:.2f}")
            else:
                st.error("❌ Failed to build embeddings. Check console for details.")
        except Exception as e:
            st.error(f"Failed to build embeddings: {e}")

st.markdown("---")

# ---------------- Live recognition
st.header("3) Live Full Face Recognition")
col_a, col_b = st.columns([1, 3])
with col_a:
    start = st.button("▶️ Start Live Recognition")
    stop = st.button("⏹️ Stop Live")
    live_threshold = st.slider("Similarity threshold", 0.30, 0.90, 0.55, 0.01)
    live_expansion = st.slider("Face crop expansion (live)", 1.1, 1.8, 1.35, 0.05)
    live_use_masked = st.checkbox("Use masked model", value=False)
    show_landmarks = st.checkbox("Show face landmarks", value=False)
with col_b:
    live_placeholder = st.empty()
    info_placeholder = st.empty()

if "running" not in st.session_state:
    st.session_state.running = False

if start:
    if not Path(EMBED_PATH).exists():
        st.error("❌ Embeddings not found — build embeddings first.")
    else:
        st.session_state.running = True

if stop:
    st.session_state.running = False

if st.session_state.running:
    # Load database & model
    try:
        db = joblib.load(EMBED_PATH)
        names = db.get("names", [])
        embs = db.get("embeddings", [])
    except Exception as e:
        st.error("Failed to load embeddings: " + str(e))
        st.session_state.running = False
        names, embs = [], []

    if len(names) == 0:
        st.error("❌ No identities in DB. Add samples and rebuild embeddings.")
        st.session_state.running = False
    else:
        model = load_facenet(device=DEVICE, masked=live_use_masked)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Cannot open webcam.")
            st.session_state.running = False
        else:
            info_placeholder.info(f"🎥 Live recognition active | Threshold: {live_threshold} | People: {len(names)}")
            frame_count = 0
            try:
                while st.session_state.running:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    
                    # Detect FULL FACE landmarks
                    lm = mediapipe_face_landmarks(frame, static_mode=False)
                    label = "No Face Detected"
                    conf = 0.0
                    box = None
                    
                    if lm is not None:
                        # Get FULL FACE bounding box
                        bbox = face_bbox_from_landmarks(lm, expand=live_expansion, img_shape=frame.shape)
                        crop = crop_region(frame, bbox)
                        
                        if crop is not None and crop.size != 0:
                            # Preprocess and get embedding
                            proc = preprocess_face_crop(crop, size=160)
                            if proc is not None:
                                emb = embedding_from_image(proc, model, device=DEVICE)
                                if emb is not None:
                                    # Compare with database
                                    sims = [cosine_similarity(emb, e) for e in embs]
                                    best_idx = int(np.argmax(sims)) if len(sims) > 0 else -1
                                    best_sim = sims[best_idx] if best_idx >= 0 else 0.0
                                    
                                    if best_sim >= live_threshold:
                                        label = names[best_idx]
                                        conf = best_sim
                                    else:
                                        label = "Unknown Person"
                                        conf = best_sim
                                    
                                    box = bbox
                        
                        # Draw landmarks if enabled
                        if show_landmarks and lm is not None:
                            for pt in lm:
                                cv2.circle(frame, tuple(pt), 1, (0, 255, 255), -1)
                    
                    # Draw bounding box and label
                    if box is not None:
                        x1, y1, x2, y2 = box
                        
                        # Color based on recognition
                        if label == "Unknown Person":
                            color = (0, 165, 255)  # Orange
                        elif label == "No Face Detected":
                            color = (0, 0, 255)  # Red
                        else:
                            color = (0, 255, 0)  # Green
                        
                        # Draw box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                        
                        # Prepare text
                        text = f"{label} ({conf:.2f})"
                        font_scale = 0.8
                        thickness = 2
                        
                        # Get text size for background
                        (text_width, text_height), baseline = cv2.getTextSize(
                            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                        )
                        
                        # Draw text background
                        cv2.rectangle(frame, (x1, y1 - text_height - 10), 
                                    (x1 + text_width + 10, y1), color, -1)
                        
                        # Draw text
                        cv2.putText(frame, text, (x1 + 5, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
                        
                        # Draw "FULL FACE" indicator
                        cv2.putText(frame, "FULL FACE MODE", (10, frame.shape[0] - 20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    else:
                        cv2.putText(frame, label, (20, 40), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    
                    # Show frame count
                    cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Display frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    live_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                    
                    if not st.session_state.running:
                        break
                    time.sleep(0.02)
                    
            except Exception as e:
                st.error("Live loop error: " + str(e))
            finally:
                cap.release()
                st.session_state.running = False
                info_placeholder.success(f"✅ Recognition stopped. Processed {frame_count} frames.")
else:
    live_placeholder.empty()
    info_placeholder.empty()

st.markdown("---")
st.write("### 📋 Tips for Best Results:")
st.write("- **Capture 25-40 samples** per person with different angles and expressions")
st.write("- **Ensure good lighting** - avoid shadows on face")
st.write("- **Vary head positions** - slight left/right/up/down rotations")
st.write("- **Keep face centered** during capture")
st.write("- **Adjust expansion factor** if face crops are too tight or loose")
st.write("- **Full face mode** captures entire face including chin, forehead, and ears")
st.caption("⚠️ Note: Streamlit must run on the machine with the webcam. Remote access won't work for webcam features.")