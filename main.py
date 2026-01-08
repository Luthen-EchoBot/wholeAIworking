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
# Paramètres de connexion et détection
HOST = '192.168.1.1'  # Adresse IP du serveur cible
YOLO_MODEL_NAME = "yolo11n.pt"  # Modèle YOLO pour la détection de personnes
CONF_THRESHOLD = 0.5  # Seuil de confiance minimum pour les détections YOLO
DEVICE = 0 if torch.cuda.is_available() else 'cpu'  # GPU si disponible, sinon CPU

# Optimisation pour Jetson Nano (traitement réduit)
FRAME_SKIP = 3  # Traite la détection YOLO 1 image sur 3 pour économiser les ressources

# Seuils de précision pour la classification des gestes
RATIO_THRESHOLD_UP = 1.3  # Ratio minimum pour considérer un doigt levé
RATIO_THRESHOLD_DOWN = 1.1  # Ratio pour les doigts repliés

# ----------------------------
# Initialisation des Modèles
# ----------------------------
print(f"Loading YOLO on device='{DEVICE}'...")
try:
    # Tente de charger le modèle optimisé TensorRT (plus rapide sur Jetson)
    model = YOLO(YOLO_MODEL_NAME, task='detect')
except Exception as e:
    print(f"Engine TensorRT non trouvé ou erreur: {e}. Chargement du modèle .pt standard.")
    model = YOLO("yolo11n.pt")

# Préchauffage du GPU (première inférence peut être lente)
try:
    model.predict(source=np.zeros((640,640,3), dtype=np.uint8), device=DEVICE, verbose=False)
except:
    pass

print("Chargement de MediaPipe pour la détection des mains...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,  # Une seule main à la fois
    model_complexity=0,  # Modèle léger (crucial pour Jetson)
    min_detection_confidence=0.5,  # Confiance minimale pour détecter une main
    min_tracking_confidence=0.5,  # Confiance minimale pour suivre la main
)

# ----------------------------
# Fonctions Utilitaires (Mathématiques & Reconnaissance des Gestes)
# ----------------------------
def calc_distance(p1, p2):
    """Calcule la distance euclidienne entre deux points 2D."""
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
    """Classifie le geste détecté en comparant avec les patterns connus.
    Reconnaît: Victory, Pointing_Up, Thumb_Up, Rock_n_Roll
    
    Paramètres:
        landmarks: Landmarks des 21 points de la main
    
    Retourne:
        (nom_geste, score_confiance): Tuple avec le nom du geste et sa confiance (0-100%)
    """
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
    """Établit une connexion TCP/IP avec le serveur. Bloque jusqu'à la connexion réussie.
    Réessaye automatiquement toutes les 2 secondes en cas d'erreur.
    
    Retourne:
        socket: Socket connecté au serveur
    """
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
# Fonction Principale
# ----------------------------
def main():
    """Boucle principale: capture vidéo, détecte les personnes, reconnaît les gestes et envoie les données au serveur."""
    # Initialiser la capture vidéo (webcam)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    
    # Configurer la résolution de la caméra
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Variables pour l'optimisation YOLO (frame skipping)
    frame_count = 0  # Compteur global de frames
    cached_detection_list = []  # Liste des détections en cache
    cached_best_person_idx = -1  # Index de la personne la plus proche du centre
    
    # Variables pour le filtrage des gestes (3 frames consécutifs requis)
    gesture_confirmation_count = 0  # Compteur de frames avec le même geste
    last_confirmed_gesture = "none"  # Dernier geste confirmé
    last_detected_gesture = "none"  # Dernier geste détecté
    
    while True:  # Boucle infinie : reconnexion automatique au serveur en cas de déconnexion
        
        # 1. BLOQUANT: Attendre la connexion serveur avant de commencer
        sock = connect_socket()

        # 2. Boucle d'inférence : capture et traitement des frames
        try:
            while True:
                # Lire une frame de la caméra
                ret, frame = cap.read()
                if not ret: 
                    print("Erreur de lecture caméra")
                    break

                # Récupérer les dimensions de la frame
                height, width, _ = frame.shape
                center_x = width // 2  # Centre horizontal pour la sélection de la personne
                frame_count += 1

                # -----------------------
                # A. DÉTECTION YOLO (1 frame sur 3 pour optimisation)
                # -----------------------
                # Exécute la détection YOLO seulement sur 1 image sur FRAME_SKIP pour réduire la charge CPU/GPU
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
                # B. RECONNAISSANCE DES GESTES (chaque frame)
                # -----------------------
                # Analyse la main de la personne détectée pour reconnaître un geste
                detected_gesture = "none"  # Geste détecté dans cette frame
                detected_gesture_prob = 0.0  # Confiance du geste (0-100%)
                detected_gesture_id = -1  # ID de suivi de la personne
                
                # On utilise la cached_detection_list pour savoir où regarder (région d'intérêt)
                if cached_best_person_idx != -1 and cached_best_person_idx < len(cached_detection_list):
                    target = cached_detection_list[cached_best_person_idx]
                    
                    # On crop sur la frame ACTUELLE avec les coordonnées (potentiellement vieilles de 2 frames max)
                    # C'est suffisant car les gens ne se téléportent pas en 0.1s
                    crop = frame[target.y : target.y+target.h, target.x : target.x+target.w]
                    
                    # Sécurité taille crop
                    h_crop, w_crop = crop.shape[:2]
                    if h_crop > 20 and w_crop > 20:
                        # Optimisation downscale pour MediaPipe sur Jetson
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
                            detected_gesture = g_name
                            detected_gesture_prob = g_conf
                            detected_gesture_id = target.id
                
                # ------- FILTRE: VALIDATION SUR 3 FRAMES CONSÉCUTIFS -------
                # Pour éviter les faux positifs, un geste n'est confirmé que s'il est détecté
                # pendant 3 frames consécutifs. Cela réduit le bruit et les détections erratiques.
                
                # Si c'est le même geste détecté à nouveau, incrémenter le compteur
                if detected_gesture != "none" and detected_gesture == last_detected_gesture:
                    gesture_confirmation_count += 1
                else:
                    # Si le geste a changé ou la détection a été perdue, réinitialiser
                    gesture_confirmation_count = 0
                    last_detected_gesture = detected_gesture if detected_gesture != "none" else last_detected_gesture
                
                # Confirmer le geste seulement après 3 frames consécutifs
                if gesture_confirmation_count >= 3:
                    # Geste validé : créer un objet Gesture_Data avec les données réelles
                    last_confirmed_gesture = detected_gesture
                    gesture_obj = Gesture_Data(class_name=detected_gesture, ID=detected_gesture_id, probability=detected_gesture_prob)
                else:
                    # Geste pas encore validé : envoyer "none" pour cette frame
                    gesture_obj = Gesture_Data("none", -1, 0.0)

                # -----------------------
                # C. TRANSMISSION DES DONNÉES
                # -----------------------
                # Affichage de debug (à décommenter pour diagnostiquer)
                # print(f"Détections: {len(cached_detection_list)} | Meilleur Geste: {gesture_obj.class_name} ({gesture_obj.probability:.1f}%)")
                
                # Créer un paquet avec les détections et le geste à envoyer au serveur
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
