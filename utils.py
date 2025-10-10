# utils.py - FULL FACE OPTIMIZED
import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
import mediapipe as mp
from PIL import Image, ExifTags

mp_face_mesh = mp.solutions.face_mesh

# ------------------ FaceNet Loader ------------------
def load_facenet(device='cpu', masked=False, masked_path="models/facenet_masked.pth"):
    """
    Load InceptionResnetV1 (FaceNet). 
    If masked==True and masked_path exists, load fine-tuned weights.
    Default: pretrained vggface2 weights.
    """
    model = InceptionResnetV1(pretrained='vggface2').eval()
    
    if masked and os.path.exists(masked_path):
        try:
            state = torch.load(masked_path, map_location=device)
            
            # Handle different checkpoint formats
            if isinstance(state, dict):
                if 'state_dict' in state:
                    sd = state['state_dict']
                elif 'model_state_dict' in state:
                    sd = state['model_state_dict']
                else:
                    sd = state
            else:
                sd = state
                
            model.load_state_dict(sd, strict=False)
            print(f"✅ Loaded masked FaceNet weights from {masked_path}")
        except Exception as e:
            print(f"⚠️  Failed to load masked weights: {e}")
            print(f"   Using base vggface2 weights instead")
    else:
        if masked:
            print(f"⚠️  Masked weights not found at {masked_path}")
        print(f"✅ Using base vggface2 FaceNet model")
    
    return model.to(device)

# ------------------ EXIF Orientation Fix ------------------
def load_corrected_image(path, max_size=1920):
    """
    Load image and correct EXIF orientation.
    Resize if larger than max_size.
    """
    img = Image.open(path)
    
    # Handle EXIF orientation
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        
        exif = img._getexif()
        if exif is not None:
            orientation_val = exif.get(orientation, None)
            if orientation_val == 3:
                img = img.rotate(180, expand=True)
            elif orientation_val == 6:
                img = img.rotate(270, expand=True)
            elif orientation_val == 8:
                img = img.rotate(90, expand=True)
    except (AttributeError, KeyError, TypeError):
        pass

    # Resize if too large
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# ------------------ Mediapipe Landmarks ------------------
def mediapipe_face_landmarks(image, static_mode=True, max_faces=1, refine_landmarks=True):
    """
    Detect face landmarks using MediaPipe Face Mesh.
    Returns numpy array of shape (N, 2) with pixel coordinates, or None if no face.
    
    Optimized for FULL FACE detection with lower confidence thresholds.
    """
    with mp_face_mesh.FaceMesh(
        static_image_mode=static_mode,
        max_num_faces=max_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=0.3,  # Lower for better detection
        min_tracking_confidence=0.3
    ) as face_mesh:
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None
        
        # Get first face landmarks
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = image.shape[:2]
        
        # Convert normalized coordinates to pixel coordinates
        pts = np.array([
            (int(p.x * w), int(p.y * h)) 
            for p in landmarks
        ], dtype=np.int32)
        
        return pts

# ------------------ FULL FACE Bounding Box ------------------
def face_bbox_from_landmarks(landmarks, expand=1.35, img_shape=None):
    """
    Calculate FULL FACE bounding box from landmarks.
    
    Args:
        landmarks: numpy array of shape (N, 2)
        expand: expansion factor (1.35 captures full face including forehead and chin)
        img_shape: tuple (height, width, channels) to clip coordinates
    
    Returns:
        tuple (x1, y1, x2, y2) - bounding box coordinates
    """
    if landmarks is None or len(landmarks) == 0:
        if img_shape is not None:
            return 0, 0, img_shape[1], img_shape[0]
        return 0, 0, 0, 0
    
    xs = landmarks[:, 0]
    ys = landmarks[:, 1]
    
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    
    width = x_max - x_min
    height = y_max - y_min
    
    if width <= 0 or height <= 0:
        if img_shape is not None:
            return 0, 0, img_shape[1], img_shape[0]
        return 0, 0, 0, 0
    
    # Calculate center
    cx = x_min + width // 2
    cy = y_min + height // 2
    
    # Expand bounding box
    new_width = int(width * expand)
    new_height = int(height * expand)
    
    # Calculate new coordinates
    x1 = cx - new_width // 2
    y1 = cy - new_height // 2
    x2 = x1 + new_width
    y2 = y1 + new_height
    
    # Clip to image boundaries
    if img_shape is not None:
        H, W = img_shape[:2]
        x1 = max(0, min(int(x1), W - 1))
        y1 = max(0, min(int(y1), H - 1))
        x2 = max(1, min(int(x2), W))
        y2 = max(1, min(int(y2), H))
    else:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    return x1, y1, x2, y2

def upper_face_bbox(landmarks, expand=1.3, img_shape=None):
    """
    Calculate upper face (eyes/forehead) bounding box.
    Useful for masked face recognition.
    
    Falls back to full face if upper region detection fails.
    """
    # Key landmarks for upper face region
    # Eyes, eyebrows, nose bridge, forehead
    upper_indices = [
        # Right eyebrow
        33, 7, 163, 144, 145, 153, 154, 155,
        # Left eyebrow  
        263, 249, 390, 373, 374, 380, 381, 382,
        # Nose bridge and upper nose
        6, 197, 195, 168, 122, 351,
        # Forehead region
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172
    ]
    
    # Filter valid indices
    valid_indices = [i for i in upper_indices if i < landmarks.shape[0]]
    
    if len(valid_indices) == 0:
        return face_bbox_from_landmarks(landmarks, expand, img_shape)
    
    upper_landmarks = landmarks[valid_indices]
    return face_bbox_from_landmarks(upper_landmarks, expand, img_shape)

# ------------------ Crop & Preprocess ------------------
def crop_region(img, bbox):
    """
    Crop image region defined by bounding box.
    
    Args:
        img: input image (numpy array)
        bbox: tuple (x1, y1, x2, y2)
    
    Returns:
        Cropped image or None if invalid
    """
    x1, y1, x2, y2 = bbox
    H, W = img.shape[:2]
    
    # Ensure valid coordinates
    x1 = max(0, min(x1, W - 1))
    x2 = max(1, min(x2, W))
    y1 = max(0, min(y1, H - 1))
    y2 = max(1, min(y2, H))
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    return img[y1:y2, x1:x2].copy()

def preprocess_face_crop(crop, size=160):
    """
    Preprocess face crop for FaceNet input.
    
    Steps:
    1. Convert to RGB
    2. Resize to 160x160
    3. Apply CLAHE for better contrast
    4. Return normalized RGB
    
    Args:
        crop: face crop (BGR numpy array)
        size: target size (default 160 for FaceNet)
    
    Returns:
        Preprocessed RGB image or None
    """
    if crop is None or crop.size == 0:
        return None
    
    try:
        # Convert BGR to RGB
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
        
        # Apply CLAHE for contrast enhancement
        # Convert to YCrCb color space
        ycrcb = cv2.cvtColor(resized, cv2.COLOR_RGB2YCrCb)
        
        # Apply CLAHE to Y channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        
        return enhanced
    except Exception as e:
        print(f"⚠️  Preprocessing error: {e}")
        return None

# ------------------ Embedding Generation ------------------
def embedding_from_image(img_rgb, facenet_model, device='cpu'):
    """
    Generate embedding from preprocessed face image.
    
    Args:
        img_rgb: preprocessed RGB image (160x160)
        facenet_model: loaded FaceNet model
        device: 'cpu' or 'cuda'
    
    Returns:
        Normalized embedding vector (512-d) or None
    """
    if img_rgb is None:
        return None
    
    try:
        # Normalize to [-1, 1] range
        img_normalized = img_rgb.astype(np.float32) / 255.0
        img_normalized = (img_normalized - 0.5) / 0.5
        
        # Convert to tensor: (H, W, C) -> (C, H, W) -> (1, C, H, W)
        tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).float()
        tensor = tensor.to(device)
        
        # Generate embedding
        with torch.no_grad():
            embedding = facenet_model(tensor)
        
        # Convert to numpy and normalize
        emb = embedding.cpu().numpy().reshape(-1)
        norm = np.linalg.norm(emb)
        
        if norm > 1e-10:
            emb = emb / norm
        else:
            return None
        
        return emb
    except Exception as e:
        print(f"⚠️  Embedding generation error: {e}")
        return None

# ------------------ Similarity Computation ------------------
def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors.
    
    Returns:
        Float similarity score between -1 and 1 (1 = identical)
    """
    if a is None or b is None:
        return -1.0
    
    # Ensure normalization
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a < 1e-10 or norm_b < 1e-10:
        return -1.0
    
    a_normalized = a / norm_a
    b_normalized = b / norm_b
    
    similarity = float(np.dot(a_normalized, b_normalized))
    
    # Clip to valid range
    return np.clip(similarity, -1.0, 1.0)

def euclidean_distance(a, b):
    """
    Compute Euclidean distance between two embedding vectors.
    Lower distance = more similar.
    """
    if a is None or b is None:
        return float('inf')
    
    return float(np.linalg.norm(a - b))

# ------------------ Face Quality Assessment ------------------
def assess_face_quality(crop, landmarks=None):
    """
    Assess quality of face crop for recognition.
    
    Returns:
        dict with quality metrics
    """
    if crop is None or crop.size == 0:
        return {"valid": False, "reason": "Empty crop"}
    
    h, w = crop.shape[:2]
    
    # Check minimum size
    if h < 80 or w < 80:
        return {"valid": False, "reason": "Too small", "size": (w, h)}
    
    # Check aspect ratio
    aspect_ratio = w / h
    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
        return {"valid": False, "reason": "Invalid aspect ratio", "aspect_ratio": aspect_ratio}
    
    # Check brightness
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    mean_brightness = np.mean(gray)
    
    if mean_brightness < 30:
        return {"valid": False, "reason": "Too dark", "brightness": mean_brightness}
    if mean_brightness > 225:
        return {"valid": False, "reason": "Too bright", "brightness": mean_brightness}
    
    # Check blur (Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if laplacian_var < 50:
        return {"valid": False, "reason": "Too blurry", "sharpness": laplacian_var}
    
    return {
        "valid": True,
        "size": (w, h),
        "aspect_ratio": aspect_ratio,
        "brightness": mean_brightness,
        "sharpness": laplacian_var
    }