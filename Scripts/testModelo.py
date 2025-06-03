import cv2
import numpy as np
import tensorflow as tf

# ===============================
# 🔥 Carregar o modelo TFLite
# ===============================
interpreter = tf.lite.Interpreter(model_path="modelo_olho_aberto_fechado.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Verificar entrada do modelo
input_shape = input_details[0]['shape']
print(f"Esperado pelo modelo: {input_shape}")

# ===============================
# 😎 Carregar Haarcascade para detecção de rosto e olhos
# ===============================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# ===============================
# 🎥 Captura da webcam
# ===============================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Espelhar para parecer um espelho
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # === Detectar rostos ===
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # === Detectar olhos dentro do rosto ===
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            eye_img = roi_color[ey:ey + eh, ex:ex + ew]
            eye_img_resized = cv2.resize(eye_img, (64, 64))
            eye_img_normalized = eye_img_resized.astype(np.float32) / 255.0
            eye_img_input = np.expand_dims(eye_img_normalized, axis=0)  # Adiciona batch

            # === Rodar modelo TFLite ===
            interpreter.set_tensor(input_details[0]['index'], eye_img_input)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])

            prob = output[0][0]
            label = "Aberto" if prob > 0.5 else "Fechado"

            color = (0, 255, 0) if label == "Aberto" else (0, 0, 255)
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), color, 2)
            cv2.putText(roi_color, label, (ex, ey - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Detecção de Olhos", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
