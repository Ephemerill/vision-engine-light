import cv2
import numpy as np
import os
import threading
import time
import queue
import uvicorn
import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# --- CONFIGURATION ---
VIDEO_SOURCE = "WEBCAM" # Options: "WEBCAM" or "RTSP"
RTSP_URL = "rtsp://admin:mysecretpassword@100.114.210.58:8554/cam"

# If on a headless server, ALWAYS use "WEB", other is "LOCAL"
OUTPUT_MODE = "WEB" 
HTTP_PORT = 5006
JPEG_QUALITY = 70 

yolo_model = "yolo11l-pose.pt"
FACES_FOLDER = 'faces'
CONFIDENCE_THRESHOLD = 0.6
RECOGNITION_INTERVAL = 2.0  

# --- GLOBAL SHARED RESOURCES ---
latest_frame = None
frame_lock = threading.Lock()
latest_results = [] 
results_lock = threading.Lock()
recognition_queue = queue.Queue()
system_active = True
output_frame = None
output_lock = threading.Lock()
track_history = {} 

# --- VISUALIZATION SETTINGS ---

# New Color specifically for the Face Box (BGR: Cyan/Teal)
FACE_BOX_COLOR = (255, 255, 0)

# Skeleton connections (COCO Format)
SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), # Legs
    (11, 12), (5, 11), (6, 12),             # Torso
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), # Arms & Shoulders
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),  # Face
    (3, 5), (4, 6)                           # Ear to Shoulder
]

# Palette (BGR Format) - Limbs
LIMB_COLORS = [
    (255, 51, 51), (255, 102, 102), (255, 153, 153), (153, 255, 153), # Legs
    (102, 255, 102), (51, 255, 51), (0, 255, 0),                      # Torso
    (255, 128, 0), (255, 153, 51), (255, 178, 102), (230, 230, 0), (255, 255, 0), # Arms
    (153, 204, 255), (153, 204, 255), (153, 204, 255), (153, 204, 255), (153, 204, 255), # Face
    (255, 51, 255), (255, 51, 255) # Ear to shoulder
]

# Palette (BGR Format) - Joints
KPT_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), 
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255), 
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255), 
    (255, 0, 255), (255, 0, 170)
]

app = FastAPI()

# --- 1. SETUP INSIGHTFACE ---
print(f"Initializing InsightFace...")
import onnxruntime
face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)

known_embeddings = []
known_names = []

# --- 2. LOAD FACES ---
if os.path.exists(FACES_FOLDER):
    print(f"Scanning '{FACES_FOLDER}'...")
    for root, dirs, files in os.walk(FACES_FOLDER):
        for filename in files:
            if filename.startswith('.'): continue
            parent_folder = os.path.basename(root)
            name = os.path.splitext(filename)[0] if parent_folder == FACES_FOLDER else parent_folder
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')): continue
            
            path = os.path.join(root, filename)
            img = cv2.imread(path)
            if img is None: continue
            
            faces = face_app.get(img)
            if len(faces) > 0:
                faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
                known_embeddings.append(faces[0].embedding)
                known_names.append(name)
                print(f"  [OK] Learned: {name}")
else:
    print(f"Warning: Folder '{FACES_FOLDER}' not found.")

# --- 3. HELPER: ENHANCEMENT ---
def enhance_face(img_crop):
    if img_crop.size == 0: return img_crop
    h, w, _ = img_crop.shape
    try:
        lab = cv2.cvtColor(img_crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        img_crop = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    except Exception: pass 
    return img_crop

# --- 4. WORKER: RECOGNITION (Thread) ---
def recognition_worker():
    while system_active:
        try:
            task = recognition_queue.get(timeout=0.5) 
        except queue.Empty:
            continue

        track_id, face_crop = task
        enhanced_crop = enhance_face(face_crop)
        faces = face_app.get(enhanced_crop)
        
        found_name = "Unknown"
        if len(faces) > 0:
            target_emb = faces[0].embedding
            best_score = 0
            for i, known_emb in enumerate(known_embeddings):
                score = np.dot(target_emb, known_emb) / (np.linalg.norm(target_emb) * np.linalg.norm(known_emb))
                if score > CONFIDENCE_THRESHOLD and score > best_score:
                    best_score = score
                    found_name = known_names[i]
            
        if track_id in track_history:
            if found_name != "Unknown":
                track_history[track_id]['name'] = found_name
            else:
                if track_history[track_id]['name'] == "...":
                    track_history[track_id]['name'] = "Unknown"
            track_history[track_id]['is_processing'] = False
            
        recognition_queue.task_done()

# --- 5. WORKER: CAPTURE (Thread) ---
def capture_worker():
    global latest_frame
    print(f"Starting Capture Worker: {VIDEO_SOURCE}")
    
    if VIDEO_SOURCE == "RTSP":
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(0)

    while system_active:
        ret, frame = cap.read()
        if not ret:
            time.sleep(2)
            if VIDEO_SOURCE == "RTSP":
                cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            else:
                cap = cv2.VideoCapture(0)
            continue
        
        if VIDEO_SOURCE == "WEBCAM":
            frame = cv2.flip(frame, 1)

        with frame_lock:
            latest_frame = frame.copy()
        time.sleep(0.005) 
    cap.release()

# --- 6. WORKER: YOLO INFERENCE (Thread) ---
def yolo_worker():
    global latest_results
    
    print("Initializing YOLO on Worker Thread...")
    model = YOLO(yolo_model)
    if torch.cuda.is_available():
        model.to('cuda')
    
    while system_active:
        img_input = None
        with frame_lock:
            if latest_frame is not None:
                img_input = latest_frame.copy()
        
        if img_input is None:
            time.sleep(0.1)
            continue

        results = model.track(img_input, persist=True, verbose=False, classes=[0])
        
        current_results_batch = []
        for r in results:
            if r.boxes.id is not None:
                track_ids = r.boxes.id.int().cpu().tolist()
                boxes = r.boxes.xyxy.cpu().numpy()
                if r.keypoints is not None:
                    all_keypoints = r.keypoints.data.cpu().numpy() 
                else:
                    all_keypoints = [None] * len(boxes)

                for box, track_id, kpts in zip(boxes, track_ids, all_keypoints):
                    current_results_batch.append({
                        'box': box,
                        'id': track_id,
                        'kpts': kpts
                    })

                    if track_id not in track_history:
                        track_history[track_id] = {'name': "...", 'last_attempt': 0, 'is_processing': False}
                    
                    data = track_history[track_id]
                    now = time.time()
                    if (now - data['last_attempt']) > RECOGNITION_INTERVAL and not data['is_processing']:
                        x1, y1, x2, y2 = map(int, box)
                        h, w, _ = img_input.shape
                        face_crop = img_input[max(0, y1-40):min(h, y2+40), max(0, x1-20):min(w, x2+20)]
                        if face_crop.size > 0:
                            track_history[track_id]['is_processing'] = True
                            track_history[track_id]['last_attempt'] = now
                            recognition_queue.put( (track_id, face_crop) )

        with results_lock:
            latest_results = current_results_batch

# --- 7. UI HELPER ---
def draw_skeleton(img, kpts):
    if kpts is None: return
    
    # Draw Lines (Limbs)
    for i, sk in enumerate(SKELETON):
        if sk[0] < len(kpts) and sk[1] < len(kpts):
            pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
            pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))
            conf1 = kpts[sk[0], 2]
            conf2 = kpts[sk[1], 2]
            
            if conf1 > 0.5 and conf2 > 0.5:
                color = LIMB_COLORS[i % len(LIMB_COLORS)]
                cv2.line(img, pos1, pos2, color, 2)

    # Draw Points (Joints)
    for i, (x, y, conf) in enumerate(kpts):
        if conf > 0.5:
            color = KPT_COLORS[i % len(KPT_COLORS)]
            cv2.circle(img, (int(x), int(y)), 5, color, -1)

def draw_ui(img, box, track_id, name, is_processing, keypoints=None):
    x1, y1, x2, y2 = map(int, box)
    
    # Determine Main Body Box Color
    if name == "...":
        color = (0, 255, 255)
        label_text = "Scanning..."
    elif is_processing:
        color = (0, 255, 0)
        label_text = f"{name} (?)" 
    elif name == "Unknown":
        color = (150, 150, 150)
        label_text = name
    else:
        color = (0, 255, 0)
        label_text = name

    # Draw main body box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    # --- Draw Face Box (Corrected Dimensions) ---
    if keypoints is not None:
        # Keypoints 0-4 are the face landmarks
        face_kpts = keypoints[:5]
        # Filter valid points (confidence > 0.5)
        valid_face_points = face_kpts[face_kpts[:, 2] > 0.5]

        # Need at least 2 points to make a box
        if len(valid_face_points) >= 2:
            coords = valid_face_points[:, :2]
            fx_min, fy_min = np.min(coords, axis=0)
            fx_max, fy_max = np.max(coords, axis=0)
            
            # 1. Calculate Width based on Ear-to-Ear (or visible points)
            raw_w = fx_max - fx_min
            # Add 10% padding (not 40% like before)
            face_width = raw_w * 1.1 
            
            # 2. Calculate Height based on Width (Aspect Ratio)
            # Use 1.25 ratio. If face_width is 100px, height is 125px.
            face_height = face_width * 1.25

            center_x = fx_min + raw_w / 2
            center_y = fy_min + (fy_max - fy_min) / 2 # Center of the "points" (usually eye level)

            fbx1 = max(0, int(center_x - (face_width / 2)))
            fby1 = max(0, int(center_y - (face_height / 2)))
            fbx2 = min(img.shape[1], int(center_x + (face_width / 2)))
            fby2 = min(img.shape[0], int(center_y + (face_height / 2)))

            # Draw the face box in the specific FACE_BOX_COLOR
            cv2.rectangle(img, (fbx1, fby1), (fbx2, fby2), FACE_BOX_COLOR, 2)
            
        # Draw Skeleton
        draw_skeleton(img, keypoints)
    # --------------------------------------------------

    # Draw Label Background & Text
    font_scale = 0.6
    thickness = 2
    (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_y = y1 - 10 if y1 - 25 > 0 else y1 + 25
    bg_y1 = y1 - h - 14 if y1 - 25 > 0 else y1
    bg_y2 = y1 if y1 - 25 > 0 else y1 + h + 10
    cv2.rectangle(img, (x1, bg_y1), (x1 + w + 10, bg_y2), color, -1)
    cv2.putText(img, label_text, (x1 + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness)

# --- 8. FASTAPI ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
    <head><title>Multithreaded AI Stream</title></head>
    <body style="background:black; margin:0; display:flex; align-items:center; justify-content:center; height:100vh;">
        <img src="/video_feed" style="max-width:100%; max-height:100%; border: 2px solid #333;">
    </body>
    </html>
    """

def generate_frames():
    while True:
        with output_lock:
            if output_frame is None:
                time.sleep(0.01)
                continue
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame, encode_param)
            if not flag: continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
        time.sleep(0.03)

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

# --- 9. DISPLAY LOOP ---
def display_loop():
    global output_frame, system_active
    print("Starting Display Loop...")
    
    while system_active:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.01)
                continue
            display_img = latest_frame.copy()

        with results_lock:
            results_to_draw = list(latest_results)

        for res in results_to_draw:
            track_id = res['id']
            box = res['box']
            kpts = res['kpts']
            
            name = "..."
            is_proc = False
            
            if track_id in track_history:
                name = track_history[track_id]['name']
                is_proc = track_history[track_id]['is_processing']
            
            draw_ui(display_img, box, track_id, name, is_proc, kpts)

        if OUTPUT_MODE == "WEB":
            with output_lock:
                output_frame = display_img
            time.sleep(0.01)
        else:
            cv2.imshow('GPU Face Rec', display_img)
            if cv2.waitKey(1) & 0xFF == 27:
                system_active = False
                break
    
    cv2.destroyAllWindows()

# --- 10. ENTRY POINT ---
if __name__ == '__main__':
    t_rec = threading.Thread(target=recognition_worker, daemon=True)
    t_rec.start()

    t_cap = threading.Thread(target=capture_worker, daemon=True)
    t_cap.start()

    t_yolo = threading.Thread(target=yolo_worker, daemon=True)
    t_yolo.start()
    
    time.sleep(1.0)

    if OUTPUT_MODE == "WEB":
        print(f"STARTING WEB MODE. Go to http://0.0.0.0:{HTTP_PORT}")
        t_disp = threading.Thread(target=display_loop, daemon=True)
        t_disp.start()
        uvicorn.run(app, host='0.0.0.0', port=HTTP_PORT, log_level="warning")
    else: 
        print("STARTING LOCAL MODE.")
        display_loop()