# create_embeddings.py
# create_embeddings.py
import os
import glob
import cv2
import numpy as np
import joblib
from utils import mediapipe_face_landmarks, face_bbox_from_landmarks, upper_face_bbox, crop_region, preprocess_face_crop, load_facenet, embedding_from_image

import torch

DATASET_DIR = "dataset"
OUT_PATH = "models/embeddings.joblib"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs("models", exist_ok=True)

def build_embeddings(dataset_dir=DATASET_DIR, out_path=OUT_PATH):
    model = load_facenet(device=DEVICE)
    names = []
    embeddings = []

    for person_dir in sorted(os.listdir(dataset_dir)):
        person_path = os.path.join(dataset_dir, person_dir)
        if not os.path.isdir(person_path):
            continue
        print("Processing person:", person_dir)
        person_embs = []
        image_files = glob.glob(os.path.join(person_path, "*"))
        for imgf in image_files:
            img = cv2.imread(imgf)
            if img is None:
                continue
            # get landmarks
            lm = mediapipe_face_landmarks(img, static_mode=True)
            if lm is None:
                # try simple face detection fallback (use whole image or skip)
                print("  no landmarks for", imgf, "skipping")
                continue
            # attempt upper-face crop first (better for masked)
            bbox_upper = upper_face_bbox(lm, expand=1.3, img_shape=img.shape)
            crop = crop_region(img, bbox_upper)
            if crop is None or crop.size == 0:
                # fallback to full face bbox
                bbox = face_bbox_from_landmarks(lm, expand=1.2, img_shape=img.shape)
                crop = crop_region(img, bbox)
                if crop is None:
                    continue
            proc = preprocess_face_crop(crop, size=160)
            emb = embedding_from_image(proc, model, device=DEVICE)
            person_embs.append(emb)
        if len(person_embs) == 0:
            print("  warning: no embeddings for", person_dir)
            continue
        # average the embeddings for this person
        avg_emb = np.mean(np.stack(person_embs, axis=0), axis=0)
        avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-10)
        names.append(person_dir)
        embeddings.append(avg_emb)
    # Save
    db = {"names": names, "embeddings": np.stack(embeddings) if embeddings else np.empty((0,512))}
    joblib.dump(db, out_path)
    print("Saved embeddings to", out_path)

if __name__ == "__main__":
    build_embeddings()
