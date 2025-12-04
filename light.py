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
VIDEO_SOURCE = "WEBCAM"
# RTSP_URL = "rtsp://admin:mysecretpassword@100.114.210.58:8554/cam"
RTSP_URL = "rtsp://admin:mysecretpassword@100.114.210.58:8554/cam"

# If on a headless server, ALWAYS use "WEB"
OUTPUT_MODE = "LOCAL" 
HTTP_PORT = 5006
JPEG_QUALITY = 70 

yolo_model = "yolo11n-pose.pt"
FACES_FOLDER = 'faces'
CONFIDENCE_THRESHOLD = 0.6
RECOGNITION_INTERVAL = 30  

# --- GLOBAL VARIABLES ---
rtsp_frame = None
rtsp_lock = threading.Lock()
rtsp_active = True
recognition_queue = queue.Queue()
track_history = {} 
output_frame = None
output_lock = threading.Lock()
app = FastAPI()

# --- 1. SETUP INSIGHTFACE (GPU ENABLED) ---
print(f"Initializing InsightFace...")

# CHECK: Print available providers to debug if CUDA is missing
import onnxruntime
print(f"Available ONNX Providers: {onnxruntime.get_available_providers()}")

# Prioritize CUDA. If CUDA is not found, it will warn and fall back to CPU.
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

    if w < 80:
        scale = 112 / w
        img_crop = cv2.resize(img_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    img_crop = cv2.filter2D(src=img_crop, ddepth=-1, kernel=kernel)
    return img_crop

# --- 4. WORKER: RECOGNITION ---
def recognition_worker():
    while True:
        try:
            task = recognition_queue.get(timeout=1) 
        except queue.Empty:
            continue

        track_id, face_crop = task
        enhanced_crop = enhance_face(face_crop)
        
        # InsightFace runs on GPU here if configured correctly
        faces = face_app.get(enhanced_crop)
        
        if len(faces) > 0:
            target_emb = faces[0].embedding
            best_score = 0
            found_name = "Unknown"
            
            for i, known_emb in enumerate(known_embeddings):
                score = np.dot(target_emb, known_emb) / (np.linalg.norm(target_emb) * np.linalg.norm(known_emb))
                if score > CONFIDENCE_THRESHOLD and score > best_score:
                    best_score = score
                    found_name = known_names[i]
            
            if track_id in track_history:
                current_name = track_history[track_id]['name']
                if found_name != "Unknown":
                    track_history[track_id]['name'] = found_name
                else:
                    if current_name == "...":
                        track_history[track_id]['name'] = "Unknown"
        
        if track_id in track_history:
            track_history[track_id]['is_processing'] = False
        recognition_queue.task_done()

# --- 5. WORKER: RTSP ---
def capture_rtsp():
    global rtsp_frame
    # For servers, using CUDA for decoding (cv2.CAP_CUDA) is possible but complex. 
    # Sticking to CPU decoding (FFmpeg) is safer and usually fast enough.
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    while rtsp_active:
        ret, frame = cap.read()
        if not ret:
            time.sleep(2)
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            continue
        with rtsp_lock:
            rtsp_frame = frame.copy()
    cap.release()

# --- 6. UI HELPER ---
def draw_ui(img, box, track_id, name, is_processing, keypoints=None):
    x1, y1, x2, y2 = map(int, box)
    
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

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    if keypoints is not None:
        face_points = keypoints[:5]
        valid_points = face_points[face_points[:, 0] > 1]
        if len(valid_points) >= 2:
            fx_min, fy_min = np.min(valid_points, axis=0)
            fx_max, fy_max = np.max(valid_points, axis=0)
            center_x = (fx_min + fx_max) / 2
            center_y = (fy_min + fy_max) / 2
            box_width = (fx_max - fx_min) * 1.5 
            box_height = box_width * 1.3    
            
            fx1 = max(0, int(center_x - (box_width / 2)))
            fx2 = min(img.shape[1], int(center_x + (box_width / 2)))
            fy1 = max(0, int(center_y - (box_height / 2)))
            fy2 = min(img.shape[0], int(center_y + (box_height / 2)))
            cv2.rectangle(img, (fx1, fy1), (fx2, fy2), (0, 255, 255), 2)

    font_scale = 0.6
    thickness = 2
    (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_y = y1 - 10 if y1 - 25 > 0 else y1 + 25
    bg_y1 = y1 - h - 14 if y1 - 25 > 0 else y1
    bg_y2 = y1 if y1 - 25 > 0 else y1 + h + 10
    cv2.rectangle(img, (x1, bg_y1), (x1 + w + 10, bg_y2), color, -1)
    cv2.putText(img, label_text, (x1 + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness)

# --- 7. FASTAPI ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
    <head><title>GPU Stream</title></head>
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

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

# --- 8. PROCESSING LOOP ---
def processing_loop():
    global output_frame, rtsp_active
    
    # Init YOLO on GPU
    print("Initializing YOLO on GPU...")
    model = YOLO(yolo_model)
    if torch.cuda.is_available():
        model.to('cuda')
        print("YOLO is using CUDA (GPU).")
    else:
        print("WARNING: CUDA not found. YOLO running on CPU.")

    frame_count = 0
    cap = None

    if VIDEO_SOURCE == "RTSP":
        while rtsp_frame is None: time.sleep(0.1)
    else:
        cap = cv2.VideoCapture(0)

    while True:
        if VIDEO_SOURCE == "RTSP":
            with rtsp_lock:
                if rtsp_frame is None: continue
                frame = rtsp_frame.copy()
        else:
            success, frame = cap.read()
            if not success: break
            frame = cv2.flip(frame, 1)

        frame_count += 1

        # YOLO Tracking (GPU Accelerated)
        results = model.track(frame, persist=True, verbose=False, stream=True)
        annotated_frame = frame.copy()

        for r in results:
            annotated_frame = r.plot(boxes=False) 
            if r.boxes.id is not None:
                track_ids = r.boxes.id.int().cpu().tolist()
                boxes = r.boxes.xyxy.cpu().numpy()
                if r.keypoints is not None:
                    all_keypoints = r.keypoints.xy.cpu().numpy()
                else:
                    all_keypoints = [None] * len(boxes)

                for box, track_id, kpts in zip(boxes, track_ids, all_keypoints):
                    if track_id not in track_history:
                        track_history[track_id] = {'name': "...", 'last_attempt': 0, 'is_processing': False}
                    data = track_history[track_id]
                    is_time_to_check = (frame_count - data['last_attempt']) > RECOGNITION_INTERVAL
                    
                    if is_time_to_check and not data['is_processing']:
                        x1, y1, x2, y2 = map(int, box)
                        h, w, _ = frame.shape
                        face_crop = frame[max(0, y1-40):min(h, y2+40), max(0, x1-20):min(w, x2+20)]
                        if face_crop.size > 0:
                            track_history[track_id]['is_processing'] = True
                            track_history[track_id]['last_attempt'] = frame_count
                            recognition_queue.put( (track_id, face_crop) )
                    draw_ui(annotated_frame, box, track_id, data['name'], data['is_processing'], kpts)

        if OUTPUT_MODE == "WEB":
            with output_lock:
                output_frame = annotated_frame.copy()
            time.sleep(0.001)
        else:
            cv2.imshow('GPU Face Rec', annotated_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                rtsp_active = False
                break

    if cap: cap.release()
    cv2.destroyAllWindows()

# --- 9. STARTUP ---
if __name__ == '__main__':
    if VIDEO_SOURCE == "RTSP":
        t_rtsp = threading.Thread(target=capture_rtsp, daemon=True)
        t_rtsp.start()
    
    t_rec = threading.Thread(target=recognition_worker, daemon=True)
    t_rec.start()
    
    if OUTPUT_MODE == "WEB":
        print(f"STARTING WEB MODE. Go to http://0.0.0.0:{HTTP_PORT}")
        t_main = threading.Thread(target=processing_loop, daemon=True)
        t_main.start()
        uvicorn.run(app, host='0.0.0.0', port=HTTP_PORT, log_level="warning")
    else: 
        print("STARTING LOCAL MODE.")
        processing_loop()