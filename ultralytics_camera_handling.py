import time

def oyez(*args,**kwargs):
    """ En cas de réclamation, demandez à Adam """
    return print(*args, **kwargs)

class Timer:
    """
simple timer for debug, usage::

  with Timer("what you are doing"):
      sleep(2)
  #prints "what you are doing: 2000ms"

or::

  t = Timer()
  sleep(1)
  t.pause()
  sleep(10) # unrelated work
  t.play()
  sleep(2)
  print(t) # 3000ms

    """
    def __init__(self,text = ""):
        self.text = text
        self.start = time.time()
        self.accu = 0.0
        self.running = True
    def elapsed(self):
        """out:ms"""
        return self.running*1000*(time.time()-self.start) + self.accu
    def __str__(self):
        return f"{self.elapsed():.1f}ms"
    def __enter__(self):
        pass
    def __exit__(self, *args):
        oyez(f"{self.text}: {self}")
        return False # do not silence exception/errors https://docs.python.org/3/reference/datamodel.html#object.__enter__
    def pause(self):
        self.accu = self.elapsed()
        self.running = False
    def play(self):
        self.running = True
        self.start = time.time()

with Timer("import numpy"):
    import numpy as np
with Timer("import cv2"):
    import cv2
with Timer("import ultralytics"):
    from ultralytics import YOLO
    from ultralytics.engine.results import Results
with Timer("import mediapipe"):
    import mediapipe as mp
import socket
import pickle
import math
with Timer("import torch"):
    import torch
from ai_data_class import AI_Data, Detection, Gesture_Data, SOCKET_PORT, MAX_MSG_LEN
from os import environ

# Config
HOST = '192.168.1.1' # IP of pi5 running ros
YOLO_MODEL_NAME = "yolo11n.pt"
OBJ_CONF_THRESHOLD = 0.5
DEVICE = 0 if torch.cuda.is_available() else 'cpu'

OBJ_DETECTION_FRAME_SKIP = 3  # Object detection only every N frames

# Experimental!
SHOW_GRAPH = False

# Seuils de précision Gestes
RATIO_THRESHOLD_UP = 1.3
RATIO_THRESHOLD_DOWN = 1.1

# for testing without launching the pi4
no_network = environ.get('NO_NETWORK') is not None

with Timer("Loading YOLO model"):
    model = YOLO(YOLO_MODEL_NAME)

with Timer("Loading mediapipe"):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

# Gesture metrics related code
def calc_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def analyze_finger_states(lm):
    """Analyse géométrique des doigts : retourne l'état (levé/replié) et la netteté de chaque doigt.
    
    Paramètres:
        lm: Liste des 21 landmarks de la main (MediaPipe)
    
    Retourne:
        states: Liste boolean pour chaque doigt [pouce, index, majeur, annulaire, auriculaire]
        clarity_scores: Scores de confiance pour chaque doigt (0.0 à 1.0)
    """
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    wrist = lm[0]
    states = []
    clarity_scores = []

    # POUCE
    pinky_mcp = lm[17]
    index_mcp = lm[5]
    
    dist_thumb_tip_pinky = calc_distance(lm[4], pinky_mcp)
    dist_thumb_ip_pinky = calc_distance(lm[3], pinky_mcp)
    dist_thumb_tip_index = calc_distance(lm[4], index_mcp)
    palm_width = calc_distance(lm[5], lm[17])

    # Le pouce est levé si il s'éloigne vers l'extérieur ET qu'il n'est pas collé à l'index
    condition_out = dist_thumb_tip_pinky > dist_thumb_ip_pinky * 1.1
    condition_not_fist = dist_thumb_tip_index > (palm_width * 0.35)

    thumb_is_up = condition_out and condition_not_fist
    states.append(thumb_is_up)
    clarity_scores.append(1.0 if thumb_is_up else 0.5)

    # OTHERS
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
    """Classifie le geste détecté en comparant avec les patterns connus.
    Reconnaît: Victory, Pointing_Up, Thumb_Up, Rock_n_Roll
    
    Paramètres:
        landmarks: Landmarks des 21 points de la main
    
    Retourne:
        (nom_geste, score_confiance): Tuple avec le nom du geste et sa confiance (%)
    """
    states, clarity = analyze_finger_states(landmarks)
    
    patterns = {
        "Victory":      [False, True, True, False, False],
        "Pointing_Up":  [False, True, False, False, False],
        "Thumb_Up":     [True, False, False, False, False],
        "Rock_n_Roll":  [False, True, False, False, True],
    }

    best_label = "none"
    best_conf = 0.0

    for label, pattern in patterns.items():
        if sum(states[i] == pattern[i] for i in range(5)) == 5:
            conf = (sum(clarity) / 5.0) * 100
            if conf > best_conf:
                best_conf = conf
                best_label = label

    # exception pour le pouce
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
            oyez(f"Attempting to connect to {HOST}:{SOCKET_PORT}...")
            s.connect((HOST, SOCKET_PORT))
            oyez("Connected!")
            return s
        except ConnectionRefusedError:
            oyez("Connection refused. Server likely down. Retrying in 2s... Set NO_NETWORK to run anyway")
            time.sleep(2)
        except Exception as e:
            oyez(f"Unexpected socket error: {e}. Retrying in 2s...")
            time.sleep(2)

def main():
    # Rolling FPS counter
    rolling_fps = [0]*15
    rolling_index = 0

    # Filtrage des gestes (3 frames consécutifs requis)
    gesture_confirmation_count = 0 
    last_detected_gesture = "none"

    frame_count = 0

    with Timer("results=..."):
        result_generator = model.track(0,True,True,classes=[0],device=DEVICE,conf=OBJ_CONF_THRESHOLD,verbose=False)

    while True:
        # 1. BLOCKING: Wait for server before doing anything else
        if not no_network:
            sock = connect_socket()

        # 2. Inference Loop
        try:
            while True:
                total = Timer()

                timer_yolotrack=0.0
                timer_yoloprocess=0.0
                t = Timer()
                results:list[Results] = [next(result_generator)]
                frame = results[0].orig_img
                timer_yolotrack = str(t)

                height, width, _ = frame.shape
                center_x = width // 2
                frame_count += 1
                
                t = Timer()
                detection_list = []
                best_person_idx = -1
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
                        detection_list.append(det_obj)

                        if dist < min_dist_to_center:
                            min_dist_to_center = dist
                            best_person_idx = i
                    timer_yoloprocess = str(t)

                    if SHOW_GRAPH:
                        cv2.imshow("obj_detection",results[0].plot()) ## => augmente les FPS !??????
                
                # MEDIAPIPE (Gestures)
                detected_gesture = "none"
                detected_gesture_prob = 0.0
                detected_gesture_id = -1
                
                # On utilise la cached_detection_list pour savoir où regarder
                timer_hands_detect = f"{Timer()}"
                timer_mediapipe_process = 0.0
                t = Timer()
                if best_person_idx != -1 and best_person_idx < len(detection_list):
                    target = detection_list[best_person_idx]
                    
                    # On crop sur la frame ACTUELLE avec les coordonnées (potentiellement vieilles de 2 frames max)
                    # C'est suffisant car les gens ne bougent pas tant que ça en 0.1s
                    crop = frame[target.y : target.y+target.h, target.x : target.x+target.w]
                    
                    h_crop, w_crop = crop.shape[:2]
                    if h_crop > 20 and w_crop > 20:
                        # Downscale if the image is too large
                        if h_crop > 300 or w_crop > 300:
                            scale = 300 / max(h_crop, w_crop)
                            crop = cv2.resize(crop, (0,0), fx=scale, fy=scale)
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

                        t.pause()

                        t2 = Timer()
                        mp_results = hands.process(crop_rgb)
                        timer_hands_detect = f"{t2}"

                        t.play()

                        if mp_results.multi_hand_landmarks:
                            detected_gesture, detected_gesture_prob = classify_gesture(mp_results.multi_hand_landmarks[0].landmark)
                            detected_gesture_id = target.id
                
                timer_mediapipe_process = str(t)
                
                # Réduction de bruit : 3 frames consécutives pour valider un geste
                if detected_gesture != "none" and detected_gesture == last_detected_gesture:
                    gesture_confirmation_count += 1
                else:
                    gesture_confirmation_count = 0
                    last_detected_gesture = detected_gesture if detected_gesture != "none" else last_detected_gesture
                
                if gesture_confirmation_count >= 3:
                    gesture_obj = Gesture_Data(class_name=detected_gesture, ID=detected_gesture_id, probability=detected_gesture_prob)
                else:
                    gesture_obj = Gesture_Data("none", -1, 0.0)

                oyez(f"Human detected: {len(detection_list)} | Best Gesture: {gesture_obj.class_name} ({gesture_obj.probability:.1f}%)")

                # Sending data to ROS2 using YAM (Yet Another Middleware)
                t = Timer()
                ai_packet = AI_Data(detection_list, gesture_obj)
                try:
                    data = pickle.dumps(ai_packet)
                    if len(data) <= MAX_MSG_LEN:
                        if not no_network:
                            sock.send(data)
                    else:
                        oyez("Packet too large, skipping")
                except BrokenPipeError:
                    oyez("Server disconnected.")
                    break # goto connect_socket
                timer_networking = str(t)

                # (Rolling) FPS counter
                rolling_fps[rolling_index] = total.elapsed()
                rolling_index = (rolling_index+1) % len(rolling_fps)
                if rolling_index == 0:
                    fps = 1000*len(rolling_fps)/sum(rolling_fps)
                    oyez(f"==> FPS: {fps:.1f}")
                oyez(f"yolotrack: {timer_yolotrack}, yoloprocess: {timer_yoloprocess}, hand_detect: {timer_hands_detect}, mediapipe_process: {timer_mediapipe_process}, network: {timer_networking}, total: {total}")

                if SHOW_GRAPH:
                    if cv2.waitKey(25) & 0xff == ord('q'): # ms, break on 'q' pressed
                        break

        except KeyboardInterrupt:
            oyez("Stopping...")
            break
        except Exception as e:
            oyez(f"Error in inference loop: {e}")
            break # goto connect_socket()
        finally:
            if not no_network:
                sock.close()

    if SHOW_GRAPH:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
