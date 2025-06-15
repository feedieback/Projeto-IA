# eye_extractor.py

import cv2
import mediapipe as mp
import numpy as np

# Inicialização do Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                   max_num_faces=1,
                                   refine_landmarks=True,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5)

# Índices dos marcos dos olhos no FaceMesh
LEFT_EYE_IDX = [33, 133, 160, 159, 158, 144, 153, 145, 163, 7]     # região do olho esquerdo
RIGHT_EYE_IDX = [362, 263, 387, 386, 385, 373, 380, 374, 390, 249] # região do olho direito

def extract_eye_region(frame, landmarks, eye_indices):
    h, w, _ = frame.shape
    eye_points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    x_vals = [p[0] for p in eye_points]
    y_vals = [p[1] for p in eye_points]
    
    x_min, x_max = max(min(x_vals) - 5, 0), min(max(x_vals) + 5, w)
    y_min, y_max = max(min(y_vals) - 5, 0), min(max(y_vals) + 5, h)
    
    eye_roi = frame[y_min:y_max, x_min:x_max]
    return cv2.resize(eye_roi, (64, 64))

# Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            
            left_eye_img = extract_eye_region(frame, landmarks, LEFT_EYE_IDX)
            right_eye_img = extract_eye_region(frame, landmarks, RIGHT_EYE_IDX)
            
            # Exibir os olhos
            cv2.imshow("Left Eye", left_eye_img)
            cv2.imshow("Right Eye", right_eye_img)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

