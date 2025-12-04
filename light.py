import cv2
import numpy as np
import os
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# --- CONFIGURATION ---
yolo_model = "yolo11n-pose.pt"
FACES_FOLDER = 'faces'
CONFIDENCE_THRESHOLD = 0.5
RECOGNITION_INTERVAL = 30  # Re-check unknown faces every 30 frames

# --- 1. SETUP INSIGHTFACE ---
print(f"Initializing InsightFace (buffalo_l)...")
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

known_embeddings = []
known_names = []

# --- 2. ROBUST FACE LOADER ---
print(f"Scanning '{FACES_FOLDER}' for known faces...")
if not os.path.exists(FACES_FOLDER):
    print(f"CRITICAL ERROR: Folder '{FACES_FOLDER}' not found.")
    exit()

for root, dirs, files in os.walk(FACES_FOLDER):
    for filename in files:
        if filename.startswith('.'): continue
        path = os.path.join(root, filename)
        parent_folder = os.path.basename(root)
        name = os.path.splitext(filename)[0] if parent_folder == FACES_FOLDER else parent_folder

        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')): continue
        img = cv2.imread(path)
        if img is None: continue

        faces = app.get(img)
        if len(faces) > 0:
            faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
            known_embeddings.append(faces[0].embedding)
            known_names.append(name)
            print(f"  [OK] Learned: {name} ({filename})")

print(f"Done. Database contains {len(known_names)} faces.")

# --- 3. HELPER FUNCTION FOR UI (FIXED BOX MATH) ---
def draw_ui(img, box, track_id, name, keypoints=None, is_scanning=False):
    x1, y1, x2, y2 = map(int, box)
    
    # Color Logic
    if is_scanning:
        color = (0, 255, 255) # Yellow
        label_name = "Scanning..."
    elif name in ["Unknown", "..."]:
        color = (150, 150, 150) # Grey
        label_name = name
    else:
        color = (0, 255, 0) # Green
        label_name = name
    
    # 1. Draw Body Box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # 2. Draw Face Box (IMPROVED MATH)
    if keypoints is not None:
        face_points = keypoints[:5]
        # Filter valid points (confidence > 0 or x,y != 0)
        valid_points = face_points[face_points[:, 0] > 1]
        
        if len(valid_points) >= 2:
            # Calculate the spread of the features
            fx_min, fy_min = np.min(valid_points, axis=0)
            fx_max, fy_max = np.max(valid_points, axis=0)
            
            # --- NEW ASPECT RATIO LOGIC ---
            # 1. Calculate true width of features (Ear to Ear)
            feature_width = fx_max - fx_min
            
            # 2. Set Box Width (User said it was too wide, so we keep padding tight: 20%)
            box_width = feature_width * 1.2
            
            # 3. Set Box Height based on Width (User said not tall enough)
            # Faces are roughly 1:1.25 width:height
            box_height = box_width * 1.25
            
            # 4. Find the Center of the features (Bridge of nose)
            center_x = (fx_min + fx_max) / 2
            center_y = (fy_min + fy_max) / 2
            
            # 5. Calculate new coordinates centered on the face
            fx1 = int(center_x - (box_width / 2))
            fx2 = int(center_x + (box_width / 2))
            fy1 = int(center_y - (box_height / 2)) # Top
            fy2 = int(center_y + (box_height / 2)) # Bottom
            
            # Clamp to screen edges
            fx1 = max(0, fx1)
            fy1 = max(0, fy1)
            fx2 = min(img.shape[1], fx2)
            fy2 = min(img.shape[0], fy2)
            
            # Draw Face Box
            cv2.rectangle(img, (fx1, fy1), (fx2, fy2), (0, 255, 255), 2)

    # 3. UI Text
    label = f"{track_id} | {label_name}"
    font_scale = 0.6
    thickness = 2
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    
    # Ensure text is readable at top edge
    if y1 - 25 < 0:
        text_y = y1 + 25
        bg_y1 = y1
        bg_y2 = y1 + h + 10
    else:
        text_y = y1 - 10
        bg_y1 = y1 - h - 14
        bg_y2 = y1

    overlay = img.copy()
    cv2.rectangle(overlay, (x1, bg_y1), (x1 + w + 10, bg_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    cv2.putText(img, label, (x1 + 5, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

# --- 4. MAIN LOOP ---
model = YOLO(yolo_model)
track_history = {} 
cap = cv2.VideoCapture(0)
frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    frame_count += 1

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
                
                # --- RECOGNITION LOGIC ---
                should_run_recognition = False
                
                if track_id not in track_history:
                    should_run_recognition = True
                    track_history[track_id] = {'name': "...", 'last_attempt': frame_count}
                else:
                    data = track_history[track_id]
                    if data['name'] in ["Unknown", "..."]:
                        if (frame_count - data['last_attempt']) > RECOGNITION_INTERVAL:
                            should_run_recognition = True
                            track_history[track_id]['last_attempt'] = frame_count

                if should_run_recognition:
                    # Crop logic for InsightFace
                    x1, y1, x2, y2 = map(int, box)
                    h, w, _ = frame.shape
                    face_crop = frame[max(0, y1-30):min(h, y2+30), max(0, x1-20):min(w, x2+20)]

                    if face_crop.size > 0:
                        faces = app.get(face_crop)
                        if len(faces) > 0:
                            target_emb = faces[0].embedding
                            best_score = 0
                            best_name = "Unknown"
                            for i, known_emb in enumerate(known_embeddings):
                                score = np.dot(target_emb, known_emb) / (np.linalg.norm(target_emb) * np.linalg.norm(known_emb))
                                if score > CONFIDENCE_THRESHOLD and score > best_score:
                                    best_score = score
                                    best_name = known_names[i]
                            track_history[track_id]['name'] = best_name
                        else:
                            track_history[track_id]['name'] = "Unknown"

                # Draw UI
                final_name = track_history[track_id]['name']
                is_scanning = (final_name == "...")
                draw_ui(annotated_frame, box, track_id, final_name, kpts, is_scanning)

    cv2.imshow('Smart Face Recognition', annotated_frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()