import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import socket
import pickle
import time
import math
import torch
import threading
from ai_data_class import AI_Data, Detection, Gesture_Data, SOCKET_PORT, MAX_MSG_LEN

# ----------------------------
# Configuration
# ----------------------------
HOST = '192.168.1.1'
YOLO_MODEL_NAME = "yolo11n.engine"
CONF_THRESHOLD = 0.5
DEVICE = 0 if torch.cuda.is_available() else 'cpu'
FRAME_SKIP = 3

RATIO_THRESHOLD_UP = 1.3
RATIO_THRESHOLD_DOWN = 1.1
GESTURE_CONF_THRESHOLD = 60.0
GESTURE_ACTION_DELAY = 1.0  # seconds

# ----------------------------
# FSM States
# ----------------------------
STATE_IDLE = 0
STATE_CONTROL = 1

current_state = STATE_IDLE
active_person_id = None

Following_state = 0
last_sent_following_state = None
last_action_time = 0.0

# ----------------------------
# Init Models
# ----------------------------
print(f"Loading YOLO on device='{DEVICE}'...")
try:
    model = YOLO(YOLO_MODEL_NAME, task='detect')
except Exception:
    model = YOLO("yolo11n.pt")

try:
    model.predict(source=np.zeros((640, 640, 3), dtype=np.uint8),
                  device=DEVICE, verbose=False)
except:
    pass

print("Loading MediaPipe...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ----------------------------
# Math / Gesture Functions
# ----------------------------
def calc_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def analyze_finger_states(lm):
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    wrist = lm[0]
    states = []
    clarity = []

    pinky_mcp = lm[17]
    index_mcp = lm[5]

    dist_tip_pinky = calc_distance(lm[4], pinky_mcp)
    dist_ip_pinky = calc_distance(lm[3], pinky_mcp)
    dist_tip_index = calc_distance(lm[4], index_mcp)
    palm_width = calc_distance(lm[5], lm[17])

    thumb_up = (dist_tip_pinky > dist_ip_pinky * 1.1) and \
               (dist_tip_index > palm_width * 0.35)

    states.append(thumb_up)
    clarity.append(1.0 if thumb_up else 0.5)

    for tip, pip in zip(finger_tips, finger_pips):
        d_tip = calc_distance(lm[tip], wrist)
        d_pip = calc_distance(lm[pip], wrist)
        ratio = d_tip / (d_pip + 1e-6)
        is_up = d_tip > d_pip
        states.append(is_up)

        if is_up:
            score = min(max((ratio - 1.0) /
                            (RATIO_THRESHOLD_UP - 1.0), 0.5), 1.0)
        else:
            score = min(max((RATIO_THRESHOLD_DOWN - ratio) /
                            (RATIO_THRESHOLD_DOWN - 0.8), 0.5), 1.0)
        clarity.append(score)

    return states, clarity

def classify_gesture(landmarks):
    states, clarity = analyze_finger_states(landmarks)

    patterns = {
        "Victory":     [False, True, True, False, False],
        "Pointing_Up": [False, True, False, False, False],
        "Thumb_Up":    [True, False, False, False, False],
        "Rock_n_Roll": [False, True, False, False, True],
    }

    for label, pattern in patterns.items():
        if states == pattern:
            conf = sum(clarity) / 5.0 * 100
            return label, conf

    return "none", 0.0

# ----------------------------
# TCP
# ----------------------------
def connect_socket(port):
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((HOST, port))
            return s
        except:
            time.sleep(1)

def following_state_sender():
    global last_sent_following_state
    while True:
        try:
            sock = connect_socket(SOCKET_PORT + 1)
            while True:
                if Following_state != last_sent_following_state:
                    sock.send(pickle.dumps(Following_state))
                    last_sent_following_state = Following_state
                time.sleep(0.2)
        except:
            time.sleep(1)

# ----------------------------
# Main
# ----------------------------
def main():
    global current_state, active_person_id
    global Following_state, last_action_time, last_sent_following_state

    threading.Thread(target=following_state_sender, daemon=True).start()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    sock = connect_socket(SOCKET_PORT)

    frame_count = 0
    cached_detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # ---------------- YOLO ----------------
        if frame_count % FRAME_SKIP == 0:
            results = model.track(frame, classes=[0], persist=True,
                                  device=DEVICE, conf=CONF_THRESHOLD,
                                  verbose=False)
            cached_detections = []

            if results[0].boxes:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    track_id = int(box.id[0]) if box.id is not None else -1
                    conf = float(box.conf[0])

                    cached_detections.append(
                        Detection(x1, y1, x2 - x1, y2 - y1,
                                  track_id, conf, 0.0, "person")
                    )

        # -------- Tracking loss reset --------
        if current_state == STATE_CONTROL:
            ids = [d.id for d in cached_detections]
            if active_person_id not in ids:
                current_state = STATE_IDLE
                active_person_id = None
                Following_state = 0
                last_sent_following_state = None
                last_action_time = 0.0

        # --------------- Gestures -------------
        for det in cached_detections:

            if current_state == STATE_CONTROL and det.id != active_person_id:
                continue

            crop = frame[det.y:det.y+det.h, det.x:det.x+det.w]
            if crop.size == 0:
                continue

            mp_res = hands.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            if not mp_res.multi_hand_landmarks:
                continue

            g_name, g_conf = classify_gesture(
                mp_res.multi_hand_landmarks[0].landmark)

            if g_conf < GESTURE_CONF_THRESHOLD:
                continue

            now = time.time()

            # -------- FSM --------
            if current_state == STATE_IDLE:
                if g_name == "Thumb_Up":
                    current_state = STATE_CONTROL
                    active_person_id = det.id
                    Following_state = 0
                    last_sent_following_state = None
                    last_action_time = 0.0
                    break

            elif current_state == STATE_CONTROL:

                if now - last_action_time < GESTURE_ACTION_DELAY:
                    continue

                if g_name == "Pointing_Up":
                    if Following_state != 1:
                        Following_state = 1
                        last_action_time = now

                elif g_name == "Victory":
                    if Following_state != 0:
                        Following_state = 0
                        last_action_time = now

                elif g_name == "Rock_n_Roll":
                    current_state = STATE_IDLE
                    active_person_id = None
                    Following_state = 0
                    last_sent_following_state = None
                    last_action_time = 0.0
                    break

                ai_packet = AI_Data([det],
                                    Gesture_Data(g_name, det.id, g_conf))
                try:
                    sock.send(pickle.dumps(ai_packet))
                except:
                    break

    cap.release()
    sock.close()

if __name__ == "__main__":
    main()
