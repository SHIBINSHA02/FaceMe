# utils.py
# utils.py
import cv2
import numpy as np
import mediapipe as mp
from facenet_pytorch import InceptionResnetV1
import torch
from sklearn.preprocessing import normalize

mp_face_mesh = mp.solutions.face_mesh

# Initialize facenet model (pretrained on VGGFace2)
# We will init in functions to avoid global GPU allocation until needed.
def load_facenet(device='cpu'):
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return model

def mediapipe_face_landmarks(image, static_mode=True, max_faces=1, refine_landmarks=False):
    """Return landmarks (list of (x,y) normalized) using MediaPipe Face Mesh for first face or None."""
    with mp_face_mesh.FaceMesh(static_image_mode=static_mode,
                               max_num_faces=max_faces,
                               refine_landmarks=refine_landmarks,
                               min_detection_confidence=0.5) as face_mesh:
        # MediaPipe expects RGB
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        # return normalized landmarks for first face
        lm = results.multi_face_landmarks[0].landmark
        h, w = image.shape[:2]
        # Convert to pixel coordinates as (x,y)
        pts = np.array([(int(p.x * w), int(p.y * h)) for p in lm], dtype=np.int32)
        return pts

def face_bbox_from_landmarks(landmarks, expand=1.3, img_shape=None):
    """Compute bounding box around landmarks and expand it by factor `expand`.
       Returns x1,y1,x2,y2 in image pixel coords."""
    xs = landmarks[:,0]
    ys = landmarks[:,1]
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w//2
    cy = y1 + h//2

    new_w = int(w * expand)
    new_h = int(h * expand)
    x1n = max(cx - new_w//2, 0)
    y1n = max(cy - new_h//2, 0)
    x2n = x1n + new_w
    y2n = y1n + new_h

    if img_shape is not None:
        H, W = img_shape[:2]
        x1n = max(0, min(x1n, W-1))
        y1n = max(0, min(y1n, H-1))
        x2n = max(1, min(x2n, W))
        y2n = max(1, min(y2n, H))
    return int(x1n), int(y1n), int(x2n), int(y2n)

def upper_face_bbox(landmarks, expand=1.2, img_shape=None):
    """Return bounding box for the upper face (forehead + eyes + nose bridge).
       We'll use landmark indices for eyes/forehead region from MediaPipe face mesh.
       Note: MediaPipe face mesh has 468 landmarks. We'll select indices roughly around eyes/upper face."""
    # Common eye/upper-face indices (approximate)
    # left eye cluster ~ [33, 7, 163, 144, 145, 153, 154, 155]
    # right eye cluster ~ [263, 249, 390, 373, 374, 380, 381, 382]
    # nose bridge top ~ 6, 197 maybe. We'll combine a subset.
    indices = [33, 7, 163, 144, 145, 153, 154, 155,
               263, 249, 390, 373, 374, 380, 381, 382,
               6, 197, 195]
    pts = landmarks[np.isin(range(landmarks.shape[0]), indices)]
    if pts.size == 0:
        return face_bbox_from_landmarks(landmarks, expand, img_shape)
    return face_bbox_from_landmarks(pts, expand, img_shape)

def crop_region(img, bbox):
    x1,y1,x2,y2 = bbox
    # Clip to image
    H,W = img.shape[:2]
    x1,x2 = max(0,x1), min(W,x2)
    y1,y2 = max(0,y1), min(H,y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]

def preprocess_face_crop(crop, size=160):
    """Resize, convert BGR->RGB, normalize for facenet-pytorch."""
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop_resized = cv2.resize(crop_rgb, (size,size))
    # convert to float tensor later; here return numpy
    return crop_resized

def embedding_from_image(img_bgr, facenet_model, device='cpu', use_tensor=False):
    """Take an RGB or BGR image (as numpy), assume size 160x160 when passed.
       Output L2-normalized embedding as 1D numpy array."""
    # facenet expects float tensor in shape (1,3,160,160) with values in [-1,1]
    img = img_bgr.astype(np.float32)
    # if BGR, convert
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = (img - 0.5) / 0.5  # scale to [-1,1]
    tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        emb = facenet_model(tensor)  # size (1,512)
    emb = emb.cpu().numpy().reshape(-1)
    emb = emb / (np.linalg.norm(emb) + 1e-10)
    return emb

def cosine_similarity(a, b):
    # expects 1D arrays; returns scalar in [-1,1]
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
