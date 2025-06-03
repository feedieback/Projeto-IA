import cv2
import numpy as np


# ================================
# 🔧 Funções Fuzzy Tipo 2
# ================================

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


# ================================
# 📏 Parâmetros das funções fuzzy
# ================================

# Abertura dos olhos medida em pixels (ajustável)
abertura_pequena_lower = [0, 0, 3, 5]
abertura_pequena_upper = [0, 0, 5, 7]

abertura_grande_lower = [10, 12, 15, 15]
abertura_grande_upper = [11, 13, 15, 15]

# Saída (grau de olho aberto)
status_fechado_lower = [0, 0, 20, 40]
status_fechado_upper = [0, 0, 30, 50]

status_aberto_lower = [60, 80, 100, 100]
status_aberto_upper = [70, 85, 100, 100]


# ================================
# 📐 Regras Fuzzy
# ================================

def regra_olho_aberto(abertura):
    m1 = trapezoidal_mf_interval(abertura, abertura_grande_lower, abertura_grande_upper)
    grau_ativacao = (m1[0], m1[1])
    return grau_ativacao, (status_aberto_lower, status_aberto_upper)


def regra_olho_fechado(abertura):
    m1 = trapezoidal_mf_interval(abertura, abertura_pequena_lower, abertura_pequena_upper)
    grau_ativacao = (m1[0], m1[1])
    return grau_ativacao, (status_fechado_lower, status_fechado_upper)


def inferencia_fuzzy_tipo2(abertura):
    r1_ativ, r1_risco = regra_olho_aberto(abertura)
    r2_ativ, r2_risco = regra_olho_fechado(abertura)

    d1 = defuzz_interval(r1_ativ, r1_risco)
    d2 = defuzz_interval(r2_ativ, r2_risco)

    inferencia_inf = max(d1[0], d2[0])
    inferencia_sup = max(d1[1], d2[1])
    return (inferencia_inf, inferencia_sup)


# ================================
# 🎥 OpenCV Webcam + Haarcascade
# ================================

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

        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            # Calcula a "abertura" do olho como altura da bounding box
            abertura = eh

            risco_inf, risco_sup = inferencia_fuzzy_tipo2(abertura)
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

    cv2.imshow("Fuzzy Eye Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
