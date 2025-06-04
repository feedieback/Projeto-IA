import cv2
import numpy as np
import os
import time

def create_directories():
    # Criar diretórios para armazenar as imagens
    os.makedirs('dataset/eyes/open', exist_ok=True)
    os.makedirs('dataset/eyes/closed', exist_ok=True)

def collect_eye_data():
    # Inicializar câmera
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    # Contadores para as imagens
    open_count = len(os.listdir('dataset/eyes/open'))
    closed_count = len(os.listdir('dataset/eyes/closed'))
    
    print("Pressione 'o' para capturar olhos abertos")
    print("Pressione 'c' para capturar olhos fechados")
    print("Pressione 'q' para sair")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar rostos
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            
            # Detectar olhos
            eyes = eye_cascade.detectMultiScale(roi_gray)
            
            for (ex, ey, ew, eh) in eyes:
                eye_img = roi_color[ey:ey + eh, ex:ex + ew]
                eye_resized = cv2.resize(eye_img, (64, 64))
                
                # Mostrar região do olho
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                
                # Mostrar contadores
                cv2.putText(frame, f"Olhos abertos: {open_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Olhos fechados: {closed_count}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("Coleta de Dados - Olhos", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('o'):
            # Capturar olhos abertos
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                
                for (ex, ey, ew, eh) in eyes:
                    eye_img = roi_color[ey:ey + eh, ex:ex + ew]
                    eye_resized = cv2.resize(eye_img, (64, 64))
                    cv2.imwrite(f'dataset/eyes/open/eye_{open_count}.jpg', eye_resized)
                    open_count += 1
                    print(f"Capturado olho aberto #{open_count}")
                    time.sleep(0.5)  # Pequeno delay para evitar duplicatas
        
        elif key == ord('c'):
            # Capturar olhos fechados
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                
                for (ex, ey, ew, eh) in eyes:
                    eye_img = roi_color[ey:ey + eh, ex:ex + ew]
                    eye_resized = cv2.resize(eye_img, (64, 64))
                    cv2.imwrite(f'dataset/eyes/closed/eye_{closed_count}.jpg', eye_resized)
                    closed_count += 1
                    print(f"Capturado olho fechado #{closed_count}")
                    time.sleep(0.5)  # Pequeno delay para evitar duplicatas
        
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nColeta finalizada!")
    print(f"Total de olhos abertos: {open_count}")
    print(f"Total de olhos fechados: {closed_count}")

if __name__ == "__main__":
    create_directories()
    collect_eye_data() 