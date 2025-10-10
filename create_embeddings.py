# create_embeddings.py
import os
import glob
import cv2
import numpy as np
import joblib
import torch
from utils import (
    mediapipe_face_landmarks,
    face_bbox_from_landmarks,
    crop_region,
    preprocess_face_crop,
    load_facenet,
    embedding_from_image
)

DATASET_DIR = "dataset"
OUT_PATH = "models/embeddings.joblib"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs("models", exist_ok=True)

def build_embeddings(dataset_dir=DATASET_DIR, out_path=OUT_PATH, masked=True):
    """
    Build embeddings for all people in `dataset_dir` and save to `out_path`.
    Returns the output path on success, None on failure.
    """
    print(f"Device: {DEVICE}")
    print(f"Using masked model: {masked}")
    
    # Load FaceNet model
    model = load_facenet(device=DEVICE, masked=masked)
    names, embeddings = [], []

    person_dirs = [d for d in sorted(os.listdir(dataset_dir)) 
                   if os.path.isdir(os.path.join(dataset_dir, d))]
    
    if not person_dirs:
        print(f"❌ No person directories found in {dataset_dir}")
        return None

    for person_dir in person_dirs:
        person_path = os.path.join(dataset_dir, person_dir)
        print(f"\nProcessing person: {person_dir}")
        person_embs = []

        image_files = glob.glob(os.path.join(person_path, "*.jpg")) + \
                      glob.glob(os.path.join(person_path, "*.jpeg")) + \
                      glob.glob(os.path.join(person_path, "*.png"))
        
        if not image_files:
            print(f"  ⚠️ No images found for {person_dir}")
            continue

        successful_images = 0
        for imgf in image_files:
            try:
                img = cv2.imread(imgf)
                if img is None:
                    print(f"  ⚠️ Failed to read image {imgf}")
                    continue

                # Detect landmarks
                lm = mediapipe_face_landmarks(img, static_mode=True)
                if lm is None:
                    print(f"  ⚠️ No landmarks detected in {os.path.basename(imgf)}")
                    continue

                # Get full-face bbox
                bbox = face_bbox_from_landmarks(lm, expand=1.25, img_shape=img.shape)
                crop = crop_region(img, bbox)
                
                if crop is None or crop.size == 0:
                    print(f"  ⚠️ Failed to crop face in {os.path.basename(imgf)}")
                    continue

                proc = preprocess_face_crop(crop, size=160)
                if proc is None:
                    print(f"  ⚠️ Preprocessing failed for {os.path.basename(imgf)}")
                    continue

                emb = embedding_from_image(proc, model, device=DEVICE)
                
                if emb is not None and len(emb) > 0:
                    # Ensure embedding is properly normalized
                    norm = np.linalg.norm(emb)
                    if norm > 1e-10:
                        emb = emb / norm
                        person_embs.append(emb)
                        successful_images += 1
                else:
                    print(f"  ⚠️ Embedding failed for {os.path.basename(imgf)}")
                    
            except Exception as e:
                print(f"  ⚠️ Error processing {os.path.basename(imgf)}: {str(e)}")
                continue

        if not person_embs:
            print(f"  ⚠️ No valid embeddings for {person_dir}, skipping")
            continue

        # Average embeddings and normalize
        avg_emb = np.mean(np.stack(person_embs, axis=0), axis=0)
        norm = np.linalg.norm(avg_emb)
        if norm > 1e-10:
            avg_emb = avg_emb / norm
        else:
            print(f"  ⚠️ Invalid average embedding for {person_dir}, skipping")
            continue

        names.append(person_dir)
        embeddings.append(avg_emb)
        print(f"  ✅ Added embeddings for {person_dir} ({successful_images}/{len(image_files)} images)")

    if not embeddings:
        print("❌ No embeddings generated. Check your images and landmarks.")
        return None

    # Stack embeddings into numpy array
    embeddings_array = np.stack(embeddings, axis=0)
    
    db = {
        "names": names,
        "embeddings": embeddings_array
    }

    try:
        joblib.dump(db, out_path)
        print(f"\n✅ Saved embeddings to {out_path}")
        print(f"  Total people: {len(names)}")
        print(f"  Embedding shape: {embeddings_array.shape}")
        print(f"  Names: {names}")
        return out_path
    except Exception as e:
        print(f"❌ Failed to save embeddings: {str(e)}")
        return None

if __name__ == "__main__":
    result = build_embeddings(dataset_dir=DATASET_DIR, out_path=OUT_PATH, masked=False)
    if result is None:
        print("\n❌ Embedding creation failed!")
        exit(1)
    else:
        print(f"\n✅ Success! Embeddings saved to {result}")