# real_time_detect.py

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from fuzzy_system import calcular_nivel_fadiga

# Carregar modelo treinado
model = load_model('model_cnn.h5')

# Inicializar Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                   max_num_faces=1,
                                   refine_landmarks=True,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5)

# Índices dos marcos dos olhos
LEFT_EYE_IDX = [33, 133, 160, 159, 158, 144, 153, 145, 163, 7]
RIGHT_EYE_IDX = [362, 263, 387, 386, 385, 373, 380, 374, 390, 249]
EAR_IDX_LEFT = [33, 160, 158, 133, 153, 144]     # para EAR
EAR_IDX_RIGHT = [362, 385, 387, 263, 373, 380]

def extract_eye(frame, landmarks, eye_indices):
    h, w, _ = frame.shape
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]
    x_min, x_max = max(min(x_vals) - 5, 0), min(max(x_vals) + 5, w)
    y_min, y_max = max(min(y_vals) - 5, 0), min(max(y_vals) + 5, h)
    eye_img = frame[y_min:y_max, x_min:x_max]
    return cv2.resize(eye_img, (64, 64)), points

def calc_EAR(eye_pts):
    def dist(a, b): return np.linalg.norm(np.array(a) - np.array(b))
    A = dist(eye_pts[1], eye_pts[5])
    B = dist(eye_pts[2], eye_pts[4])
    C = dist(eye_pts[0], eye_pts[3])
    return (A + B) / (2.0 * C)

# Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]
        landmarks = face.landmark

        # Recortes dos olhos
        left_eye_img, _ = extract_eye(frame, landmarks, LEFT_EYE_IDX)
        right_eye_img, _ = extract_eye(frame, landmarks, RIGHT_EYE_IDX)

        # EAR
        h, w, _ = frame.shape
        left_pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in EAR_IDX_LEFT]
        right_pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in EAR_IDX_RIGHT]
        ear_esq = calc_EAR(left_pts)
        ear_dir = calc_EAR(right_pts)
        ear_medio = (ear_esq + ear_dir) / 2.0

        # CNN: processamento de imagem
        def preprocess(img):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype('float32') / 255.0
            return np.expand_dims(img, axis=0)

        pred_esq = model.predict(preprocess(left_eye_img), verbose=0)
        pred_dir = model.predict(preprocess(right_eye_img), verbose=0)

        prob_esq_aberto = float(pred_esq[0][0])  # suposição: índice 0 = aberto
        prob_dir_aberto = float(pred_dir[0][0])

        # Inferência Fuzzy
        nivel_fadiga = calcular_nivel_fadiga(prob_esq_aberto, prob_dir_aberto, ear_medio)

        # ALERTA (exemplo simples)
        if nivel_fadiga > 80:
            cv2.putText(frame, "FADIGA CRITICA!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
           #cv2.beep()  # Windows only — substitua se estiver em outro OS
        elif nivel_fadiga > 60:
            cv2.putText(frame, "Alerta: Fadiga Alta", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)

        # Mostrar valores na tela
        cv2.putText(frame, f"Nivel Fadiga: {nivel_fadiga:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.imshow("Fadiga Driver Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
