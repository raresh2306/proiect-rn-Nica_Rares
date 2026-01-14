import cv2
import mediapipe as mp
import csv
import os

# === CONFIGURARE ===
# Salvăm în data/generated pentru a marca clar că sunt datele originale (Etapa 4)
OUTPUT_FILE = 'data/generated/dataset_original.csv'

# Etichetele gesturilor
# 0: STOP (Palma deschisă)
# 1: INAINTE (Arătător sus)
# 2: STANGA (Arătător stânga)
# 3: DREAPTA (Arătător dreapta)
LABELS_MAP = {0: 'STOP', 1: 'INAINTE', 2: 'STANGA', 3: 'DREAPTA'}

# === INIȚIALIZARE MEDIAPIPE ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,         # Detectăm o singură mână pentru control robot
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# === PREGĂTIRE FIȘIER CSV ===
# Creăm folderul dacă nu există (deși tu îl ai deja)
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# Dacă fișierul nu există, scriem capul de tabel (Header)
if not os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Header: label, urmat de x,y,z pentru cele 21 de puncte
        header = ['label']
        for i in range(21):
            header.extend([f'lm_{i}_x', f'lm_{i}_y', f'lm_{i}_z'])
        writer.writerow(header)
        print(f"Creat fișier nou: {OUTPUT_FILE}")

# === PORNIRE CAMERĂ ===
cap = cv2.VideoCapture(0)
print("=== INSTRUCȚIUNI ===")
print("Apasa tastele 0, 1, 2, 3 pentru a salva un frame cu eticheta respectiva.")
print("0: STOP | 1: INAINTE | 2: STANGA | 3: DREAPTA")
print("Q: Ieșire")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip pentru a lucra ca în oglindă (mai natural)
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Procesare MediaPipe
    results = hands.process(rgb_frame)
    
    # Desenăm scheletul mâinii dacă este detectat
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Afișăm fereastra
    cv2.imshow('Data Acquisition', frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    # === SALVARE DATE ===
    if key in [ord('0'), ord('1'), ord('2'), ord('3')]:
        if results.multi_hand_landmarks:
            # Luăm prima mână detectată
            landmarks = results.multi_hand_landmarks[0].landmark
            
            # Convertim tasta apăsată în număr (label)
            label = int(chr(key))
            
            # Construim rândul de date: [label, x0, y0, z0, x1, y1, z1, ...]
            row = [label]
            for lm in landmarks:
                row.extend([lm.x, lm.y, lm.z])
            
            # Scriem în CSV (append mode)
            with open(OUTPUT_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            
            print(f"✅ Salvat: {LABELS_MAP[label]}")
        else:
            print("⚠️ Mâna nu este detectată! Nu pot salva.")
            
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()