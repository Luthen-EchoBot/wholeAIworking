import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import math

# ----------------------------
# Configuration
# ----------------------------
YOLO_MODEL_NAME = "yolo11n.pt"
CONF_THRESHOLD = 0.5
DEVICE = "cpu"

# Seuils de précision
RATIO_THRESHOLD_UP = 1.3
RATIO_THRESHOLD_DOWN = 1.1

# ----------------------------
# Initialize Models
# ----------------------------
print(f"Loading YOLO on device '{DEVICE}'...")
model = YOLO(YOLO_MODEL_NAME)
model.predict(source=np.zeros((640, 640, 3), dtype=np.uint8), device=DEVICE, verbose=False)

print("Loading MediaPipe Hands...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ----------------------------
# Helper Functions (Geometry & Scoring)
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

    # --- 1. POUCE (Correction Anti-Faux Positif "Poing") ---
    # Logique précédente : on comparait au petit doigt (lm[17])
    pinky_mcp = lm[17]
    dist_thumb_tip_pinky = calc_distance(lm[4], pinky_mcp)
    dist_thumb_ip_pinky = calc_distance(lm[3], pinky_mcp)
    
    # Nouvelle sécurité : Distance Pouce-Index
    # Dans un poing, le bout du pouce (4) est collé à la base de l'index (5) ou du majeur.
    # Dans un pouce levé, il est loin.
    index_mcp = lm[5] # Base de l'index
    dist_thumb_tip_index = calc_distance(lm[4], index_mcp)
    
    # Echelle de référence (largeur de la paume : Index MCP <-> Pinky MCP)
    palm_width = calc_distance(lm[5], lm[17])

    # Le pouce est levé SI :
    # A. Il s'éloigne du corps de la main (vers l'extérieur)
    condition_out = dist_thumb_tip_pinky > dist_thumb_ip_pinky * 1.1
    # B. ET il n'est pas collé à l'index (Sécurité anti-poing)
    # Si la distance bout_pouce <-> base_index est inférieure à 35% de la largeur de la paume, c'est un poing.
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

def classify_gesture_with_score(landmarks):
    states, clarity = analyze_finger_states(landmarks)
    
    # [Pouce, Index, Majeur, Annulaire, Auriculaire]
    patterns = {
        "Victory":      [False, True, True, False, False],
        "Pointing_Up":  [False, True, False, False, False],
        "Thumb_Up":     [True, False, False, False, False],
        "Rock_n_Roll":  [False, True, False, False, True],
    }

    best_label = "Unknown"
    best_conf = 0.0

    # 1. Verification stricte
    for label, pattern in patterns.items():
        match_count = sum(1 for i in range(5) if states[i] == pattern[i])
        if match_count == 5:
            avg_clarity = sum(clarity) / 5.0
            conf = avg_clarity * 100
            if conf > best_conf:
                best_conf = conf
                best_label = label

    # 2. Tolérance Pouce (Sauf pour Thumb_Up évidemment)
    if best_label == "Unknown":
        # Victory avec pouce sorti (Accepté)
        if states[1] and states[2] and not states[3] and not states[4]: 
             best_label = "Victory"
             best_conf = (sum(clarity) / 5.0) * 100
        # Rock avec pouce sorti (Accepté)
        elif states[1] and states[4] and not states[2] and not states[3]: 
             best_label = "Rock_n_Roll"
             best_conf = (sum(clarity) / 5.0) * 100

    if best_conf < 50.0:
        return "Unknown", 0.0

    return best_label, best_conf

# ----------------------------
# Main
# ----------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): raise RuntimeError("Could not open camera")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Starting camera stream...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        height, width, _ = frame.shape
        center_x = width // 2

        # 1. Detection YOLO
        results = model.track(frame, classes=[0], persist=True, verbose=False,
                              device=DEVICE, conf=CONF_THRESHOLD)

        best_person_idx = -1
        min_dist = float("inf")

        if results[0].boxes:
            for i, box in enumerate(results[0].boxes):
                coords = box.xyxy[0].cpu().numpy().astype(int)
                obj_center_x = (coords[0] + coords[2]) / 2
                dist = abs(obj_center_x - center_x)
                if dist < min_dist:
                    min_dist = dist
                    best_person_idx = i

        if results[0].boxes:
            for i, box in enumerate(results[0].boxes):
                coords = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = coords
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, width), min(y2, height)

                track_id = int(box.id[0]) if box.id is not None else -1
                conf = float(box.conf[0])
                obj_center_x = (x1 + x2) / 2
                dist = abs(obj_center_x - center_x)

                color = (0, 255, 0) if i == best_person_idx else (100, 100, 100)
                
                # Affichage YOLO permanent
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text_label = f"ID:{track_id} Dist:{dist:.1f} Conf:{conf*100:.0f}%"
                cv2.putText(frame, text_label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Detection Geste (Seulement meilleure personne)
                if i == best_person_idx:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        mp_results = hands.process(crop_rgb)

                        if mp_results.multi_hand_landmarks:
                            lm = mp_results.multi_hand_landmarks[0].landmark
                            g_name, g_conf = classify_gesture_with_score(lm)
                            
                            if g_name != "Unknown":
                                gesture_text = f"{g_name} ({int(g_conf)}%)"
                                cv2.putText(frame, gesture_text, (x1, y1 - 35),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                                
                                mp.solutions.drawing_utils.draw_landmarks(
                                    frame[y1:y2, x1:x2], 
                                    mp_results.multi_hand_landmarks[0], 
                                    mp_hands.HAND_CONNECTIONS
                                )

        cv2.imshow("Detection Gestes Finale", frame)
        if cv2.waitKey(1) == 27: break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed.")

if __name__ == "__main__":
    main()
