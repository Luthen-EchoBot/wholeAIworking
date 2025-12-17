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

# ============================================================
# CONFIGURATION
# ============================================================

HOST = '192.168.1.1'

YOLO_MODEL_NAME = "yolo11n.engine"      # TensorRT engine on Jetson
CONF_THRESHOLD = 0.5                   # YOLO confidence threshold
DEVICE = 0 if torch.cuda.is_available() else 'cpu'

FRAME_SKIP = 3                         # YOLO runs every N frames

GESTURE_CONF_THRESHOLD = 60.0          # Minimum gesture confidence (%)
GESTURE_ACTION_DELAY = 1.0             # Debounce time (seconds)

RATIO_THRESHOLD_UP = 1.3
RATIO_THRESHOLD_DOWN = 1.1

# ============================================================
# FINITE STATE MACHINE DEFINITIONS
# ============================================================

STATE_IDLE = 0     # Waiting for Thumb_Up
STATE_CONTROL = 1  # Actively controlling one person

current_state = STATE_IDLE
active_person_id = None

# Global binary command (sent asynchronously)
Following_state = 0

# Used to avoid re-sending identical values
last_sent_following_state = None

# Timestamp used for gesture debounce
last_action_time = 0.0

# ============================================================
# MODEL INITIALIZATION
# ============================================================

print("[INFO] Loading YOLO model...")
try:
    model = YOLO(YOLO_MODEL_NAME, task='detect')
except Exception:
    print("[WARN] TensorRT engine not found, loading .pt model")
    model = YOLO("yolo11n.pt")

# GPU warmup
model.predict(
    source=np.zeros((640, 640, 3), dtype=np.uint8),
    device=DEVICE,
    verbose=False
)

print("[INFO] Loading MediaPipe Hands...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,                # Low complexity for Jetson
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ============================================================
# GEOMETRY & GESTURE CLASSIFICATION
# ============================================================

def calc_distance(p1, p2):
    """Euclidean distance between two MediaPipe landmarks"""
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def analyze_finger_states(lm):
    """
    Determines which fingers are raised using geometric ratios.
    Returns:
        states  -> [thumb, index, middle, ring, pinky] (bool)
        clarity -> per-finger confidence scores
    """
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    wrist = lm[0]

    states = []
    clarity = []

    # --- Thumb logic (anti-fist protection) ---
    pinky_mcp = lm[17]
    index_mcp = lm[5]

    thumb_up = (
        calc_distance(lm[4], pinky_mcp) >
        calc_distance(lm[3], pinky_mcp) * 1.1 and
        calc_distance(lm[4], index_mcp) >
        calc_distance(lm[5], lm[17]) * 0.35
    )

    states.append(thumb_up)
    clarity.append(1.0 if thumb_up else 0.5)

    # --- Other fingers ---
    for tip, pip in zip(finger_tips, finger_pips):
        d_tip = calc_distance(lm[tip], wrist)
        d_pip = calc_distance(lm[pip], wrist)
        is_up = d_tip > d_pip
        states.append(is_up)
        clarity.append(1.0 if is_up else 0.5)

    return states, clarity

def classify_gesture(landmarks):
    """
    Classifies the hand gesture based on finger states.
    Returns:
        gesture_name, confidence (%)
    """
    states, clarity = analyze_finger_states(landmarks)

    patterns = {
        "Victory":     [False, True, True, False, False],
        "Pointing_Up": [False, True, False, False, False],
        "Thumb_Up":    [True, False, False, False, False],
        "Rock_n_Roll": [False, True, False, False, True],
    }

    for name, pattern in patterns.items():
        if states == pattern:
            return name, sum(clarity) / 5.0 * 100.0

    return "none", 0.0

# ============================================================
# TCP COMMUNICATION
# ============================================================

def connect_socket(port):
    """Blocking TCP connection with retry"""
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((HOST, port))
            print(f"[TCP] Connected on port {port}")
            return s
        except:
            time.sleep(1)

def following_state_sender():
    """
    Asynchronous thread.
    Sends Following_state ONLY when it changes (edge-triggered).
    """
    global last_sent_following_state

    while True:
        try:
            sock = connect_socket(SOCKET_PORT + 1)
            while True:
                if Following_state != last_sent_following_state:
                    sock.send(pickle.dumps(Following_state))
                    print(f"[TCP] Following_state sent: {Following_state}")
                    last_sent_following_state = Following_state
                time.sleep(0.05)
        except:
            time.sleep(1)

# ============================================================
# MAIN LOOP
# ============================================================

def main():
    global current_state, active_person_id
    global Following_state, last_action_time, last_sent_following_state

    # Start async Following_state sender
    threading.Thread(
        target=following_state_sender,
        daemon=True
    ).start()

    cap = cv2.VideoCapture(0)
    sock = connect_socket(SOCKET_PORT)

    frame_count = 0
    cached_detections = []

    print("[INFO] System started. Waiting for gestures...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # ----------------------------------------------------
        # YOLO DETECTION (FRAME SKIPPING)
        # ----------------------------------------------------
        if frame_count % FRAME_SKIP == 0:
            results = model.track(
                frame,
                classes=[0],
                persist=True,
                device=DEVICE,
                conf=CONF_THRESHOLD,
                verbose=False
            )

            cached_detections = []
            if results[0].boxes:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cached_detections.append(
                        Detection(
                            x1, y1, x2 - x1, y2 - y1,
                            int(box.id[0]),
                            float(box.conf[0]),
                            0.0,
                            "person"
                        )
                    )

        # ----------------------------------------------------
        # TRACKING LOSS HANDLING
        # ----------------------------------------------------
        if current_state == STATE_CONTROL:
            if active_person_id not in [d.id for d in cached_detections]:
                print("[FSM] Active person lost → IDLE")
                current_state = STATE_IDLE
                active_person_id = None
                Following_state = 0
                last_sent_following_state = None
                last_action_time = 0.0

        # ----------------------------------------------------
        # GESTURE ANALYSIS & FSM
        # ----------------------------------------------------
        for det in cached_detections:

            # In CONTROL, ignore all other people
            if current_state == STATE_CONTROL and det.id != active_person_id:
                continue

            crop = frame[det.y:det.y + det.h, det.x:det.x + det.w]
            if crop.size == 0:
                continue

            mp_res = hands.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            if not mp_res.multi_hand_landmarks:
                continue

            gesture, conf = classify_gesture(
                mp_res.multi_hand_landmarks[0].landmark
            )

            if conf < GESTURE_CONF_THRESHOLD:
                gesture = "none"

            now = time.time()

            # ---------------- FSM LOGIC ----------------
            if current_state == STATE_IDLE:
                if gesture == "Thumb_Up":
                    current_state = STATE_CONTROL
                    active_person_id = det.id
                    Following_state = 0
                    last_sent_following_state = None
                    last_action_time = 0.0
                    print(f"[FSM] CONTROL activated by ID {det.id}")

            elif current_state == STATE_CONTROL:

                # ---- Debounced control gestures ----
                if now - last_action_time >= GESTURE_ACTION_DELAY:
                    if gesture == "Pointing_Up" and Following_state != 1:
                        Following_state = 1
                        last_action_time = now
                        print("[FSM] Following_state → 1")

                    elif gesture == "Victory" and Following_state != 0:
                        Following_state = 0
                        last_action_time = now
                        print("[FSM] Following_state → 0")

                    elif gesture == "Rock_n_Roll":
                        print("[FSM] Reset to IDLE")
                        current_state = STATE_IDLE
                        active_person_id = None
                        Following_state = 0
                        last_sent_following_state = None
                        last_action_time = 0.0
                        break

                # ---- ALWAYS STREAM TRACKING DATA ----
                ai_packet = AI_Data(
                    [det],
                    Gesture_Data(gesture, det.id, conf)
                )

                try:
                    sock.send(pickle.dumps(ai_packet))
                except:
                    print("[TCP] Connection lost")
                    break

    cap.release()
    sock.close()

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
