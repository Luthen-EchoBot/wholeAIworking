import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import socket
import pickle
import time
import math
import torch
from ai_data_class import AI_Data, Detection, Gesture_Data, SOCKET_PORT, MAX_MSG_LEN

# ----------------------------
# Configuration
# ----------------------------
HOST = '192.168.1.1' 
YOLO_MODEL_NAME = "yolo11n.engine" # Utilise .engine si dispo sur la Jetson, sinon .pt
CONF_THRESHOLD = 0.5   
DEVICE = 0 if torch.cuda.is_available() else 'cpu' # Auto-detection GPU

# Optimisation Jetson
FRAME_SKIP = 3  # Analyse YOLO 1 image sur 3

# Seuils de précision Gestes
RATIO_THRESHOLD_UP = 1.3
RATIO_THRESHOLD_DOWN = 1.1

# ----------------------------
# Init Models
# ----------------------------
print(f"Loading YOLO on device='{DEVICE}'...")
try:
    # Tente de charger le modèle optimisé TensorRT
    model = YOLO(YOLO_MODEL_NAME, task='detect')
except Exception as e:
    print(f"Engine non trouvé ou erreur: {e}. Chargement du .pt standard.")
    model = YOLO("yolo11n.pt")

# Warmup GPU
try:
    model.predict(source=np.zeros((640,640,3), dtype=np.uint8), device=DEVICE, verbose=False)
except:
    pass

print("Loading MediaPipe...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0, # Crucial pour Jetson
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ----------------------------
# Helper Functions (Math & Gestures)
# ----------------------------
def calc_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def analyze_finger_states(lm):
    """Analyse géométrique : retourne l'état et la netteté de chaque doigt."""
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    wrist = lm[0]
    states = []
    clarity_scores = []

    # --- 1. POUCE (Sécurité Anti-Poing) ---
    pinky_mcp = lm[17]
    index_mcp = lm[5]
    
    dist_thumb_tip_pinky = calc_distance(lm[4], pinky_mcp)
    dist_thumb_ip_pinky = calc_distance(lm[3], pinky_mcp)
    dist_thumb_tip_index = calc_distance(lm[4], index_mcp)
    palm_width = calc_distance(lm[5], lm[17])

    # Le pouce est levé SI : Il s'éloigne vers l'extérieur ET n'est pas collé à l'index
    condition_out = dist_thumb_tip_pinky > dist_thumb_ip_pinky * 1.1
    condition_not_fist = dist_thumb_tip_index > (palm_width * 0.35)

    thumb_is_up = condition_out and condition_not_fist
    states.append(thumb_is_up)
    clarity_scores.append(1.0 if thumb_is_up else 0.5)

    # --- 2. AUTRES DOIGTS ---
    for tip_idx, pip_idx in zip(finger_tips, finger_pips):
        d_tip = calc_distance(lm[tip_idx], wrist)
        d_pip = calc_distance(lm[pip_idx], wrist)
        
        ratio = d_tip / (d_pip + 1e-6)
        is_up = d_tip > d_pip
        states.append(is_up)
        
        if is_up:
             score = min(max((ratio - 1.0) / (RATIO_THRESHOLD_UP - 1.0), 0.5), 1.0)
        else:
             score = min(max((RATIO_THRESHOLD_DOWN - ratio) / (RATIO_THRESHOLD_DOWN - 0.8), 0.5), 1.0)
        clarity_scores.append(score)
            
    return states, clarity_scores

def classify_gesture(landmarks):
    """Classification précise avec Score"""
    states, clarity = analyze_finger_states(landmarks)
    
    patterns = {
        "Victory":      [False, True, True, False, False],
        "Pointing_Up":  [False, True, False, False, False],
        "Thumb_Up":     [True, False, False, False, False],
        "Rock_n_Roll":  [False, True, False, False, True],
    }

    best_label = "none" # Par defaut "none" pour le socket
    best_conf = 0.0

    # 1. Verification stricte
    for label, pattern in patterns.items():
        if sum(1 for i in range(5) if states[i] == pattern[i]) == 5:
            conf = (sum(clarity) / 5.0) * 100
            if conf > best_conf:
                best_conf = conf
                best_label = label

    # 2. Tolérance Pouce (Sauf Thumb_Up)
    if best_label == "none":
        # Victory pouce sorti
        if states[1] and states[2] and not states[3] and not states[4]: 
             best_label = "Victory"; best_conf = (sum(clarity)/5.0)*100
        # Rock pouce sorti
        elif states[1] and states[4] and not states[2] and not states[3]: 
             best_label = "Rock_n_Roll"; best_conf = (sum(clarity)/5.0)*100

    if best_conf < 50.0:
        return "none", 0.0

    return best_label, best_conf

def connect_socket():
    """Blocking connection loop"""
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            print(f"Attempting to connect to {HOST}:{SOCKET_PORT}...")
            s.connect((HOST, SOCKET_PORT))
            print("Connected!")
            return s
        except ConnectionRefusedError:
            print("Connection refused. Server likely down. Retrying in 2s...")
            time.sleep(2)
        except Exception as e:
            print(f"Unexpected socket error: {e}. Retrying in 2s...")
            time.sleep(2)

# ----------------------------
# Main
# ----------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Variables pour le Frame Skipping
    frame_count = 0
    cached_detection_list = []
    cached_best_person_idx = -1

    while True: # Main application loop
        
        # 1. BLOCKING: Wait for server before doing anything else
        sock = connect_socket()

        # 2. Inference Loop
        try:
            while True:
                ret, frame = cap.read()
                if not ret: 
                    print("Camera read failed")
                    break

                height, width, _ = frame.shape
                center_x = width // 2
                frame_count += 1

                # -----------------------
                # A. YOLO (1 frame sur 3)
                # -----------------------
                if frame_count % FRAME_SKIP == 0:
                    results = model.track(frame, classes=[0], persist=True, verbose=False, device=DEVICE, conf=CONF_THRESHOLD)
                    
                    cached_detection_list = []
                    cached_best_person_idx = -1
                    min_dist_to_center = float('inf')

                    if results[0].boxes:
                        for i, box in enumerate(results[0].boxes):
                            coords = box.xyxy[0].cpu().numpy().astype(int)
                            x1, y1, x2, y2 = coords
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(width, x2), min(height, y2)

                            track_id = int(box.id[0]) if box.id is not None else -1
                            confidence = float(box.conf[0])
                            
                            obj_center_x = (x1 + x2) / 2
                            dist = abs(obj_center_x - center_x)

                            det_obj = Detection(x=x1, y=y1, w=(x2-x1), h=(y2-y1), id=track_id, 
                                                probability=confidence, estimated_distance=dist, class_name="person")
                            cached_detection_list.append(det_obj)

                            if dist < min_dist_to_center:
                                min_dist_to_center = dist
                                cached_best_person_idx = i
                
                # -----------------------
                # B. GESTURE (A chaque frame)
                # -----------------------
                gesture_obj = Gesture_Data("none", -1, 0.0)
                
                # On utilise la cached_detection_list pour savoir où regarder
                if cached_best_person_idx != -1 and cached_best_person_idx < len(cached_detection_list):
                    target = cached_detection_list[cached_best_person_idx]
                    
                    # On crop sur la frame ACTUELLE avec les coordonnées (potentiellement vieilles de 2 frames max)
                    # C'est suffisant car les gens ne se téléportent pas en 0.1s
                    crop = frame[target.y : target.y+target.h, target.x : target.x+target.w]
                    
                    # Sécurité taille crop
                    h_crop, w_crop = crop.shape[:2]
                    if h_crop > 20 and w_crop > 20:
                        # Downscale optimization for MediaPipe on Jetson
                        if h_crop > 300 or w_crop > 300:
                            scale = 300 / max(h_crop, w_crop)
                            crop_small = cv2.resize(crop, (0,0), fx=scale, fy=scale)
                            crop_rgb = cv2.cvtColor(crop_small, cv2.COLOR_BGR2RGB)
                        else:
                            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

                        mp_results = hands.process(crop_rgb)
                        if mp_results.multi_hand_landmarks:
                            # Utilisation de la NOUVELLE fonction précise
                            g_name, g_conf = classify_gesture(mp_results.multi_hand_landmarks[0].landmark)
                            gesture_obj = Gesture_Data(class_name=g_name, ID=target.id, probability=g_conf)

                # -----------------------
                # C. SEND DATA
                # -----------------------
                # Debug local (optionnel)
                # print(f"Detections: {len(cached_detection_list)} | Best Gesture: {gesture_obj.class_name} ({gesture_obj.probability:.1f}%)")
                
                ai_packet = AI_Data(cached_detection_list, gesture_obj)
                try:
                    data = pickle.dumps(ai_packet)
                    if len(data) <= MAX_MSG_LEN:
                        sock.send(data)
                    else:
                        print("Packet too large, skipping")
                except BrokenPipeError:
                    print("Server disconnected.")
                    break # Break inner loop to go back to connect_socket()

        except KeyboardInterrupt:
            print("Stopping...")
            break
        except Exception as e:
            print(f"Error in inference loop: {e}")
            break # Break inner loop to go back to connect_socket()
        finally:
            sock.close()

    cap.release()

if __name__ == "__main__":
    main()
