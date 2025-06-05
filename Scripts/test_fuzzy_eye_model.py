import cv2
import numpy as np
import tensorflow as tf

# =========================
# Funções Fuzzy Tipo 2
# =========================
def trapezoidal_mf_interval(x, lower_params, upper_params):
    def trapezoidal(x, params):
        a, b, c, d = params
        if x <= a or x >= d:
            return 0.0
        elif a < x < b:
            return (x - a) / (b - a)
        elif b <= x <= c:
            return 1.0
        elif c < x < d:
            return (d - x) / (d - c)
        else:
            return 0.0

    inf = trapezoidal(x, lower_params)
    sup = trapezoidal(x, upper_params)
    if sup < inf:
        sup, inf = inf, sup
    return (inf, sup)

def centroid_trapezoidal(params):
    a, b, c, d = params
    return (a + 2 * b + 2 * c + d) / 6

def defuzz_interval(grau_ativacao, risco_params):
    c_lower = centroid_trapezoidal(risco_params[0])
    c_upper = centroid_trapezoidal(risco_params[1])
    inferencia_inf = grau_ativacao[0] * c_lower
    inferencia_sup = grau_ativacao[1] * c_upper
    return (inferencia_inf, inferencia_sup)

# =========================
# Parâmetros Fuzzy
# =========================
baixo_lower = [0.0, 0.0, 0.2, 0.4]
baixo_upper = [0.0, 0.0, 0.3, 0.5]
alto_lower = [0.6, 0.7, 1.0, 1.0]
alto_upper = [0.7, 0.8, 1.0, 1.0]
intermediario_lower = [0.3, 0.45, 0.55, 0.7]
intermediario_upper = [0.35, 0.5, 0.6, 0.75]
fechado_lower = [0, 0, 20, 40]
fechado_upper = [0, 0, 30, 50]
aberto_lower = [60, 80, 100, 100]
aberto_upper = [70, 85, 100, 100]
intermedio_lower = [30, 40, 60, 70]
intermedio_upper = [35, 45, 65, 75]

def regra_aberto(prob):
    m = trapezoidal_mf_interval(prob, alto_lower, alto_upper)
    return m, (aberto_lower, aberto_upper)

def regra_fechado(prob):
    m = trapezoidal_mf_interval(prob, baixo_lower, baixo_upper)
    return m, (fechado_lower, fechado_upper)

def regra_intermediario(prob):
    m = trapezoidal_mf_interval(prob, intermediario_lower, intermediario_upper)
    return m, (intermedio_lower, intermedio_upper)

def inferencia_fuzzy_tipo2(prob):
    r1_ativ, r1_saida = regra_aberto(prob)
    r2_ativ, r2_saida = regra_fechado(prob)
    r3_ativ, r3_saida = regra_intermediario(prob)

    d1 = defuzz_interval(r1_ativ, r1_saida)
    d2 = defuzz_interval(r2_ativ, r2_saida)
    d3 = defuzz_interval(r3_ativ, r3_saida)

    inf = max(d1[0], d2[0], d3[0])
    sup = max(d1[1], d2[1], d3[1])

    return (inf, sup)

# =========================
# Carregar modelo TFLite
# =========================
interpreter = tf.lite.Interpreter(model_path="modelo_olho_improved.tflite")  # ou modelo_olho_dataset.tflite
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# =========================
# OpenCV - Haarcascade
# =========================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4)
        for (ex, ey, ew, eh) in eyes:
            if ey > h * 0.6:
                continue
            eye_img = roi_color[ey:ey + eh, ex:ex + ew]
            eye_resized = cv2.resize(eye_img, (64, 64))
            eye_normalized = eye_resized.astype(np.float32) / 255.0
            eye_input = np.expand_dims(eye_normalized, axis=0)

            # Inferência no modelo TFLite
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

    cv2.imshow("Fuzzy Eye Detector + IA", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()