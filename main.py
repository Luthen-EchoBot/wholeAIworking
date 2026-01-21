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

OBJ_DETECTION_FRAME_SKIP = 1  # Object detection only every N frames
HANDS_MAX_SIZE = 300

# Experimental!
SHOW_GRAPH = False

# Seuils de précision Gestes
RATIO_THRESHOLD_UP = 1.3
RATIO_THRESHOLD_DOWN = 1.1

# for testing without launching the pi4
no_network = environ.get('NO_NETWORK') is not None

## https://stackoverflow.com/questions/43665208/how-to-get-the-latest-frame-from-capture-device-camera-in-opencv
from threading import Thread
import queue
class VideoCapture:

    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.q = queue.Queue()
        self.t = Thread(target=self._reader,name="stream_reader")
        self.t.daemon = True
        self.t.start()
  
    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)
  
    def read(self):
        return True,self.q.get()
    
    def release(self):
        self.cap.release()


with Timer("Loading YOLO model"):
    model = YOLO(YOLO_MODEL_NAME)

# Warmup GPU
with Timer("Warmup"):
    try:
        model.predict(source=np.zeros((640,480,3), dtype=np.uint8), device=DEVICE, verbose=False)
    except:
        pass

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

def is_up(tip,pip):
    finger_length = calc_distance(tip,pip)
    cos_pi_forth = 0.7071
    return pip.y-tip.y > finger_length * cos_pi_forth

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
    are_fingers_up = []
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
    are_fingers_up.append(is_up(lm[4],lm[2]))
    clarity_scores.append(1.0 if thumb_is_up else 0.5)

    # OTHERS
    for tip_idx, pip_idx in zip(finger_tips, finger_pips):
        d_tip = calc_distance(lm[tip_idx], wrist)
        d_pip = calc_distance(lm[pip_idx], wrist)
        
        ratio = d_tip / (d_pip + 1e-6)
        finger_is_open = d_tip > d_pip
        states.append(finger_is_open)
        are_fingers_up.append(is_up(lm[tip_idx],lm[pip_idx]))
        
        if finger_is_open:
             score = min(max((ratio - 1.0) / (RATIO_THRESHOLD_UP - 1.0), 0.5), 1.0)
        else:
             score = min(max((RATIO_THRESHOLD_DOWN - ratio) / (RATIO_THRESHOLD_DOWN - 0.8), 0.5), 1.0)
        clarity_scores.append(score)
            
    return states, are_fingers_up, clarity_scores

def classify_gesture(landmarks):
    """Classifie le geste détecté en comparant avec les patterns connus.
    Reconnaît: Victory, Pointing_Up, Thumb_Up, Rock_n_Roll
    
    Paramètres:
        landmarks: Landmarks des 21 points de la main
    
    Retourne:
        (nom_geste, score_confiance): Tuple avec le nom du geste et sa confiance (%)
    """
    states, fingers_up, clarity = analyze_finger_states(landmarks)
    
    # type: [(label, [which fingers should be "open"], should open fingers be upward)]
    patterns = [
        ("Victory",      [False, True, True, False, False],True),
        # ("Victory",      [True , True, True, False, False],True), # thumb should not be up for Victory
        ("Pointing_Up",  [False, True, False, False, False],False),
        ("Thumb_Up",     [True, False, False, False, False],True),
        ("Rock_n_Roll",  [False, True, False, False, True],False),
        ("Rock_n_Roll",  [True , True, False, False, True],False)
    ]

    best_label = "none"
    best_conf = 0.0

    for label, pattern, should_be_up in patterns:
        match = True
        for i in range(5):
            match = match and states[i] == pattern[i]
            if should_be_up and pattern[i]:
                match = match and fingers_up[i]
        if match:
            conf = (sum(clarity) / 5.0) * 100
            if conf > best_conf:
                best_conf = conf
                best_label = label

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

    t = Timer()
    cap = VideoCapture(0)
    oyez(f"opening cam: {t}")

    # Pour le Frame Skipping
    frame_count = 0
    cached_detection_list = []
    cached_best_person_idx = -1

    # Filtrage des gestes (3 frames consécutifs requis)
    gesture_confirmation_count = 0 
    last_detected_gesture = "none"

    while True:
        # 1. BLOCKING: Wait for server before doing anything else
        if not no_network:
            sock = connect_socket()

        # 2. Inference Loop
        try:
            while True:
                total = Timer()
                t = Timer()
                ret, frame = cap.read()
                timer_camread = str(t)
                if not ret: 
                    oyez("Camera read failed")
                    break

                height, width, _ = frame.shape
                center_x = width // 2
                frame_count += 1

                # YOLO (Obj detection): 1 frame out of 3
                timer_yolotrack=0.0
                timer_yoloprocess=0.0
                if frame_count % OBJ_DETECTION_FRAME_SKIP == 0:
                    t = Timer()
                    results = model.track(frame, classes=[0], persist=True, verbose=False, device=DEVICE, conf=OBJ_CONF_THRESHOLD)
                    timer_yolotrack = str(t)
                    
                    t = Timer()
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
                    timer_yoloprocess = str(t)

                    if SHOW_GRAPH:
                        cv2.imshow("obj_detection",results[0].plot()) ## => augmente les FPS !??????
                
                # MEDIAPIPE (Gestures)
                detected_gesture = "none"
                detected_gesture_prob = 0.0
                detected_gesture_id = -1
                
                # On utilise la cached_detection_list pour savoir où regarder
                timer_hands_detect = str(Timer())
                timer_mediapipe_process = 0.0
                t = Timer()
                if cached_best_person_idx != -1 and cached_best_person_idx < len(cached_detection_list):
                    target = cached_detection_list[cached_best_person_idx]
                    
                    # On crop sur la frame ACTUELLE avec les coordonnées (potentiellement vieilles de 2 frames max)
                    # C'est suffisant car les gens ne bougent pas tant que ça en 0.1s
                    crop = frame[target.y : target.y+target.h, target.x : target.x+target.w]
                    
                    h_crop, w_crop = crop.shape[:2]
                    if h_crop > 20 and w_crop > 20:
                        # Downscale if the image is too large
                        if h_crop > HANDS_MAX_SIZE or w_crop > HANDS_MAX_SIZE:
                            scale = HANDS_MAX_SIZE / max(h_crop, w_crop)
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
                
                if gesture_obj.class_name != "none":
                    oyez(f"Human detected: {len(cached_detection_list)} | Best Gesture: {gesture_obj.class_name} ({gesture_obj.probability:.1f}%)")

                # Sending data to ROS2 using YAM (Yet Another Middleware)
                t = Timer()
                ai_packet = AI_Data(cached_detection_list, gesture_obj)
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
                # oyez(f"camread: {timer_camread}, yolotrack: {timer_yolotrack}, yoloprocess: {timer_yoloprocess}, hand_detect: {timer_hands_detect}, mediapipe_process: {timer_mediapipe_process}, network: {timer_networking}, total: {total}")

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

    cap.release()
    if SHOW_GRAPH:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
