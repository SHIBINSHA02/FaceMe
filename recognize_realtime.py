# recognize_realtime.py - FULL FACE VERSION
import cv2
import numpy as np
import joblib
import torch
import os
from utils import (
    mediapipe_face_landmarks,
    face_bbox_from_landmarks,
    crop_region,
    preprocess_face_crop,
    load_facenet,
    embedding_from_image,
    cosine_similarity
)

EMBED_PATH = "models/embeddings.joblib"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
THRESHOLD = 0.55
EXPANSION = 1.35  # Full face expansion factor
FONT = cv2.FONT_HERSHEY_SIMPLEX
BUFFER_SIZE = 5   # Smooth recognition over 5 frames

def load_db(path=EMBED_PATH):
    """Load embeddings database."""
    if not os.path.exists(path):
        print(f"❌ Embeddings file not found at {path}")
        print("   Run: python create_embeddings.py")
        return [], np.empty((0, 512))
    
    try:
        data = joblib.load(path)
        names = data.get("names", [])
        embeddings = data.get("embeddings", np.empty((0, 512)))
        
        if len(names) != len(embeddings):
            print(f"⚠️  Database mismatch: {len(names)} names, {len(embeddings)} embeddings")
            return [], np.empty((0, 512))
        
        print("="*60)
        print("DATABASE LOADED")
        print("="*60)
        print(f"✅ Loaded {len(names)} identities")
        print(f"   People: {', '.join(names)}")
        print(f"   Embedding shape: {embeddings.shape}")
        print("="*60 + "\n")
        
        return names, embeddings
    except Exception as e:
        print(f"❌ Error loading embeddings: {str(e)}")
        return [], np.empty((0, 512))

def draw_face_info(frame, box, label, conf, frame_num):
    """Draw bounding box and information on frame."""
    x1, y1, x2, y2 = box
    
    # Determine color based on recognition
    if label == "Unknown":
        color = (0, 165, 255)  # Orange
        box_thickness = 2
    elif label == "No Face":
        color = (0, 0, 255)  # Red
        box_thickness = 2
    else:
        color = (0, 255, 0)  # Green
        box_thickness = 3
    
    # Draw main bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
    
    # Draw corner markers for better visibility
    corner_len = 20
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 4)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 4)
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 4)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 4)
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 4)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 4)
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 4)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 4)
    
    # Prepare label text
    if label in ["Unknown", "No Face"]:
        text = label
    else:
        text = f"{label} ({conf:.2%})"
    
    # Text styling
    font_scale = 0.9
    thickness = 2
    
    # Get text dimensions
    (text_w, text_h), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
    
    # Draw text background
    padding = 10
    cv2.rectangle(frame, 
                 (x1, y1 - text_h - padding * 2), 
                 (x1 + text_w + padding * 2, y1), 
                 color, -1)
    
    # Draw text
    cv2.putText(frame, text, (x1 + padding, y1 - padding), 
               FONT, font_scale, (255, 255, 255), thickness)
    
    # Draw "FULL FACE" indicator
    cv2.putText(frame, "FULL FACE MODE", (10, frame.shape[0] - 50), 
               FONT, 0.7, (255, 255, 0), 2)
    
    # Draw frame counter
    cv2.putText(frame, f"Frame: {frame_num}", (10, 30), 
               FONT, 0.6, (255, 255, 255), 2)
    
    return frame

def main():
    print("\n🚀 Starting Full Face Recognition System\n")
    
    # Load database
    names, embs = load_db(EMBED_PATH)
    if len(names) == 0:
        print("❌ Cannot start: No identities in database")
        print("   1. Add images to dataset/PersonName/")
        print("   2. Run: python create_embeddings.py")
        return

    # Load model
    print(f"Loading FaceNet model on {DEVICE}...")
    model = load_facenet(device=DEVICE, masked=False)
    print("✅ Model loaded\n")

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    # Set webcam properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("="*60)
    print("🎥 LIVE RECOGNITION ACTIVE")
    print("="*60)
    print(f"Recognition threshold: {THRESHOLD}")
    print(f"Expansion factor: {EXPANSION}")
    print(f"Buffer size: {BUFFER_SIZE} frames")
    print(f"Device: {DEVICE}")
    print("\nControls:")
    print("  'q' or ESC - Quit")
    print("  's' - Save current frame")
    print("  'l' - Toggle landmark display")
    print("="*60 + "\n")
    
    embedding_buffer = []
    frame_count = 0
    recognized_count = 0
    show_landmarks = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️  Failed to read frame")
                break

            frame_count += 1
            
            # Detect FULL FACE landmarks
            lm = mediapipe_face_landmarks(frame, static_mode=False, refine_landmarks=True)
            label = "No Face"
            conf = 0.0
            box = None

            if lm is not None:
                # Get FULL FACE bounding box
                bbox = face_bbox_from_landmarks(lm, expand=EXPANSION, img_shape=frame.shape)
                crop = crop_region(frame, bbox)

                if crop is not None and crop.size != 0:
                    # Preprocess full face crop
                    proc = preprocess_face_crop(crop, size=160)
                    
                    if proc is not None:
                        # Generate embedding
                        emb = embedding_from_image(proc, model, device=DEVICE)
                        
                        if emb is not None and len(emb) > 0:
                            # Normalize embedding
                            norm = np.linalg.norm(emb)
                            if norm > 1e-10:
                                emb = emb / norm
                                embedding_buffer.append(emb)
                                
                                # Maintain buffer size
                                if len(embedding_buffer) > BUFFER_SIZE:
                                    embedding_buffer.pop(0)

                                # Average embeddings for stability
                                avg_emb = np.mean(np.stack(embedding_buffer, axis=0), axis=0)
                                avg_norm = np.linalg.norm(avg_emb)
                                
                                if avg_norm > 1e-10:
                                    avg_emb = avg_emb / avg_norm

                                    # Compare with all database embeddings
                                    sims = [cosine_similarity(avg_emb, e) for e in embs]
                                    
                                    if len(sims) > 0:
                                        best_idx = int(np.argmax(sims))
                                        best_sim = sims[best_idx]

                                        if best_sim >= THRESHOLD:
                                            label = names[best_idx]
                                            conf = best_sim
                                            recognized_count += 1
                                        else:
                                            label = "Unknown"
                                            conf = best_sim

                                        box = bbox
                
                # Draw landmarks if enabled
                if show_landmarks and lm is not None:
                    for idx, pt in enumerate(lm):
                        # Different colors for different face regions
                        if idx < 17:  # Jaw
                            color = (255, 0, 0)
                        elif idx < 27:  # Eyebrows
                            color = (0, 255, 0)
                        elif idx < 36:  # Nose
                            color = (0, 0, 255)
                        elif idx < 48:  # Eyes
                            color = (255, 255, 0)
                        else:  # Mouth
                            color = (255, 0, 255)
                        cv2.circle(frame, tuple(pt), 2, color, -1)
            else:
                # Reset buffer if no face detected
                if len(embedding_buffer) > 0:
                    embedding_buffer = []

            # Draw visualization
            if box is not None:
                frame = draw_face_info(frame, box, label, conf, frame_count)
            else:
                # No face detected
                cv2.putText(frame, "No Face Detected", (20, 50), 
                           FONT, 1.2, (0, 0, 255), 3)
                cv2.putText(frame, "Position your face in frame", (20, 90), 
                           FONT, 0.7, (0, 0, 255), 2)
            
            # Display recognition stats
            if frame_count > 0:
                recognition_rate = (recognized_count / frame_count) * 100
                stats_text = f"Recognition Rate: {recognition_rate:.1f}%"
                cv2.putText(frame, stats_text, (10, frame.shape[0] - 20), 
                           FONT, 0.6, (255, 255, 255), 2)

            # Show frame
            cv2.imshow("Full Face Recognition", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27 or key == ord('q'):  # ESC or 'q'
                print("\n⏹️  Stopping recognition...")
                break
            elif key == ord('s'):  # Save frame
                filename = f"capture_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Frame saved as {filename}")
            elif key == ord('l'):  # Toggle landmarks
                show_landmarks = not show_landmarks
                print(f"👁️  Landmarks: {'ON' if show_landmarks else 'OFF'}")

    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during recognition: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*60)
        print("SESSION SUMMARY")
        print("="*60)
        print(f"Total frames processed: {frame_count}")
        print(f"Recognized frames: {recognized_count}")
        if frame_count > 0:
            print(f"Recognition rate: {(recognized_count/frame_count)*100:.1f}%")
        print("="*60)
        print("✅ Recognition stopped successfully")

if __name__ == "__main__":
    main()