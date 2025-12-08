import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import time

# ----------------------------
# Configuration
# ----------------------------
YOLO_MODEL_NAME = "yolo11n.pt"
CONF_THRESHOLD = 0.5
DEVICE = "cpu"

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
# Helper Functions
# ----------------------------
def get_finger_states(landmarks):
    """Return finger states (1=extended, 0=folded) for index, middle, ring, pinky."""
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    states = []
    for tip_id, pip_id in zip(finger_tips, finger_pips):
        states.append(1 if landmarks[tip_id].y < landmarks[pip_id].y else 0)
    return states


def classify_gesture(landmarks):
    """Classify gesture based on MediaPipe hand landmarks."""
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
        conf = (matches / 4.0) * 100
        if conf > best_conf:
            best_conf = conf
            best_label = label

    if best_conf >= 75:
        return best_label, best_conf
    return "Unknown", 0.0


# ----------------------------
# Main
# ----------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Starting camera stream...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed")
            break

        height, width, _ = frame.shape
        center_x = width // 2

        # Perform detection + tracking
        results = model.track(frame, classes=[0], persist=True, verbose=False,
                              device=DEVICE, conf=CONF_THRESHOLD)

        best_person_idx = -1
        min_dist = float("inf")
        gesture_info = ("None", 0.0)

        if results[0].boxes:
            for i, box in enumerate(results[0].boxes):
                coords = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = coords

                # Clamp coordinates
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, width), min(y2, height)

                track_id = int(box.id[0]) if box.id is not None else -1
                conf = float(box.conf[0])

                obj_center_x = (x1 + x2) / 2
                dist = abs(obj_center_x - center_x)

                # Select the person closest to the horizontal center
                if dist < min_dist:
                    min_dist = dist
                    best_person_idx = i

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0, 255, 0), 2)

                text_label = f"ID:{track_id} Dist:{dist:.1f} Conf:{conf*100:.1f}%"
                cv2.putText(frame, text_label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            # Gesture detection only for the closest person
            if best_person_idx != -1:
                box = results[0].boxes[best_person_idx]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                crop = frame[y1:y2, x1:x2]

                if crop.size > 0:
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    mp_results = hands.process(crop_rgb)

                    if mp_results.multi_hand_landmarks:
                        g_name, g_conf = classify_gesture(mp_results.multi_hand_landmarks[0].landmark)
                        gesture_info = (g_name, g_conf)

                        cv2.putText(frame,
                                    f"Gesture: {g_name} {g_conf:.1f}%",
                                    (x1, y1 - 35),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55, (0, 255, 255), 2)

        cv2.imshow("Detections", frame)

        key = cv2.waitKey(1)
        if key == 27:
            print("ESC pressed. Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed.")


if __name__ == "__main__":
    main()
