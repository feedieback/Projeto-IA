# ... existing code ...
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Parâmetros otimizados para detecção de rosto
FACE_MIN_SIZE = (100, 100)  # Tamanho mínimo do rosto
FACE_SCALE_FACTOR = 1.1     # Fator de escala para a pirâmide de imagem
FACE_MIN_NEIGHBORS = 5      # Número mínimo de vizinhos para considerar uma detecção

# Parâmetros otimizados para detecção de olhos
EYE_MIN_SIZE = (30, 30)     # Tamanho mínimo dos olhos
EYE_SCALE_FACTOR = 1.1      # Fator de escala para a pirâmide de imagem
EYE_MIN_NEIGHBORS = 3       # Número mínimo de vizinhos para considerar uma detecção

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecção de rostos com parâmetros otimizados
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=FACE_SCALE_FACTOR,
        minNeighbors=FACE_MIN_NEIGHBORS,
        minSize=FACE_MIN_SIZE
    )

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        
        # Desenha retângulo do rosto com cor mais suave
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Adiciona texto com informações do rosto
        cv2.putText(frame, f"Face {w}x{h}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Detecção de olhos com parâmetros otimizados
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=EYE_SCALE_FACTOR,
            minNeighbors=EYE_MIN_NEIGHBORS,
            minSize=EYE_MIN_SIZE
        )

        for (ex, ey, ew, eh) in eyes:
            # Verifica se o olho está dentro da região do rosto
            if ex + ew <= w and ey + eh <= h:
                eye_img = roi_color[ey:ey + eh, ex:ex + ew]
                eye_resized = cv2.resize(eye_img, (64, 64))
                eye_normalized = eye_resized.astype(np.float32) / 255.0
                eye_input = np.expand_dims(eye_normalized, axis=0)

                # 🔥 Inferência no modelo TFLite
                interpreter.set_tensor(input_details[0]['index'], eye_input)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])

                prob = output[0][0]

                risco_inf, risco_sup = inferencia_fuzzy_tipo2(prob)
                media = (risco_inf + risco_sup) / 2

                if media < 33:
                    status = "Fechado"
                    color = (0, 0, 255)
                elif media < 66:
                    status = "Intermediário"
                    color = (0, 255, 255)
                else:
                    status = "Aberto"
                    color = (0, 255, 0)

                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), color, 2)
                cv2.putText(roi_color, f"{status} ({media:.1f})", (ex, ey - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Adiciona FPS e contagem de rostos
    cv2.putText(frame, f"Rostos detectados: {len(faces)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Fuzzy Eye Detector + IA", frame)
# ... existing code ...