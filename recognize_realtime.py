# recognize_realtime.py
# recognize_realtime.py
import cv2
import numpy as np
import joblib
import time
import torch

from utils import mediapipe_face_landmarks, upper_face_bbox, face_bbox_from_landmarks, crop_region, preprocess_face_crop, load_facenet, embedding_from_image, cosine_similarity

EMBED_PATH = "models/embeddings.joblib"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
THRESHOLD = 0.55   # cosine similarity threshold; tune empirically (0.5-0.7)
FONT = cv2.FONT_HERSHEY_SIMPLEX

def load_db(path=EMBED_PATH):
    data = joblib.load(path)
    return data.get("names", []), data.get("embeddings", np.empty((0,512)))

def main():
    names, embs = load_db(EMBED_PATH)
    if len(names) == 0:
        print("No names in DB. Run create_embeddings.py first.")
        return
    model = load_facenet(device=DEVICE)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # detect landmarks
        lm = mediapipe_face_landmarks(frame, static_mode=False)
        label = "No Face"
        conf = 0.0
        box = None
        if lm is not None:
            # Check if mask likely present by checking if mouth landmarks are occluded?
            # Simpler heuristic: compute lower-face area brightness / edges — but for now always try upper-face crop.
            bbox_upper = upper_face_bbox(lm, expand=1.25, img_shape=frame.shape)
            crop = crop_region(frame, bbox_upper)
            if crop is None or crop.size == 0:
                bbox = face_bbox_from_landmarks(lm, expand=1.2, img_shape=frame.shape)
                crop = crop_region(frame, bbox)
                box = bbox
            else:
                box = bbox_upper
            proc = preprocess_face_crop(crop, size=160)
            emb = embedding_from_image(proc, model, device=DEVICE)
            # compute similarities to DB
            sims = [cosine_similarity(emb, db_emb) for db_emb in embs]
            best_idx = int(np.argmax(sims))
            best_sim = sims[best_idx]
            if best_sim >= THRESHOLD:
                label = names[best_idx]
                conf = best_sim
            else:
                label = "Unknown"
                conf = best_sim
        # draw box and label
        if box is not None:
            x1,y1,x2,y2 = box
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10), FONT, 0.8, (0,255,0), 2)
        else:
            cv2.putText(frame, label, (20,30), FONT, 1.0, (0,0,255), 2)

        cv2.imshow("Masked Face Recognizer", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
