
# Sistema de Detecção de Fadiga de Motoristas

Este projeto implementa um sistema em Python que detecta automaticamente sinais de fadiga em motoristas, utilizando visão computacional, redes neurais convolucionais (CNN) e lógica fuzzy tipo 2.

## 🔧 Estrutura de Pastas

```
projeto_fadiga_driver/
├── scripts/
│   ├── train_model.py        # Treinamento da CNN
│   ├── eye_extractor.py      # Extração de olhos com Mediapipe
│   └── real_time_detect.py   # Pipeline completo com alerta
├── fuzzy/
│   └── fuzzy_system.py       # Sistema Fuzzy Tipo 2
├── requirements.txt          # Bibliotecas necessárias
```

## ▶️ Como usar

1. Instale as dependências:

```
pip install -r requirements.txt
```

2. Treine o modelo (ou use um `.h5` pré-existente):

```
python scripts/train_model.py
```

3. Teste a extração dos olhos com webcam:

```
python scripts/eye_extractor.py
```

4. Rode o sistema completo de detecção:

```
python scripts/real_time_detect.py
```

## 🔍 Observações
- Pressione `q` para sair dos scripts com webcam.
- O modelo será salvo como `model_cnn.h5`.
- Para alertas sonoros, utilize `winsound` (Windows) ou uma alternativa multiplataforma.
