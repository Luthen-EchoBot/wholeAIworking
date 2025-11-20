import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import socket
import pickle
import time
from ai_data_class import AI_Data, Detection, Gesture_Data, SOCKET_PORT, MAX_MSG_LEN

# ----------------------------
# Configuration
# ----------------------------
HOST = '192.168.1.1' 
YOLO_MODEL_NAME = "yolo11n.pt"
CONF_THRESHOLD = 0.5   
DEVICE = 0  # 0 = GPU (CUDA), 'cpu' = CPU

# ----------------------------
# Init Models
# ----------------------------
print(f"Loading YOLO on device='{DEVICE}'...")
try:
    model = YOLO(YOLO_MODEL_NAME)
    # Force a dummy inference to load model into GPU memory immediately
    model.predict(source=np.zeros((640,640,3), dtype=np.uint8), device=DEVICE, verbose=False)
except Exception as e:
    print(f"Error loading YOLO: {e}")
    exit(1)

print("Loading MediaPipe...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ----------------------------
# Helper Functions
# ----------------------------
def get_finger_states(landmarks):
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    states = []
    for tip_id, pip_id in zip(finger_tips, finger_pips):
        states.append(1 if landmarks[tip_id].y < landmarks[pip_id].y else 0)
    return states

def classify_gesture(landmarks):
    states = get_finger_states(landmarks)
    patterns = {
        "Open_Palm":   [1, 1, 1, 1],
        "Closed_Fist": [0, 0, 0, 0],
        "Victory":     [1, 1, 0, 0],
        "Pointing_Up": [1, 0, 0, 0],
    }
    best_label = "Unknown"
    best_conf = 0.0
    for label, pattern in patterns.items():
        matches = sum(1 for p, s in zip(pattern, states) if p == s)
        conf = (matches / 4.0) * 100.0
        if conf > best_conf:
            best_conf = conf
            best_label = label
    
    return (best_label, best_conf) if best_conf >= 75.0 else ("Unknown", 0.0)

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
    
    # Optional: Lower resolution for speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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

                # YOLO Inference on GPU
                results = model.track(frame, classes=[0], persist=True, verbose=False, device=DEVICE, conf=CONF_THRESHOLD)
                
                detection_list = []
                best_person_idx = -1
                min_dist_to_center = float('inf')

                if results[0].boxes:
                    for i, box in enumerate(results[0].boxes):
                        # Move to CPU for numpy processing
                        coords = box.xyxy[0].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = coords
                        
                        # Clamp
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(width, x2), min(height, y2)

                        track_id = int(box.id[0]) if box.id is not None else -1
                        confidence = float(box.conf[0])
                        
                        obj_center_x = (x1 + x2) / 2
                        dist = abs(obj_center_x - center_x)

                        det_obj = Detection(x=x1, y=y1, w=(x2-x1), h=(y2-y1), id=track_id, 
                                            probability=confidence, estimated_distance=dist, class_name="person")
                        detection_list.append(det_obj)

                        if dist < min_dist_to_center:
                            min_dist_to_center = dist
                            best_person_idx = i

                # Gesture Logic
                gesture_obj = Gesture_Data("none", -1, 0.0)
                if best_person_idx != -1:
                    target = detection_list[best_person_idx]
                    crop = frame[target.y : target.y+target.h, target.x : target.x+target.w]
                    
                    if crop.size > 0:
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        mp_results = hands.process(crop_rgb)
                        if mp_results.multi_hand_landmarks:
                            g_name, g_conf = classify_gesture(mp_results.multi_hand_landmarks[0].landmark)
                            gesture_obj = Gesture_Data(class_name=g_name, ID=target.id, probability=g_conf)

                # Send
                ai_packet = AI_Data(detection_list, gesture_obj)
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
