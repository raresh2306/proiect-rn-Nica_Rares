import streamlit as st
import cv2
import mediapipe as mp
import torch
import numpy as np
import pickle
import sys
import os
import time

# Adăugăm calea către src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.neural_network.model import GestureClassifier

# === CONFIGURARE PAGINĂ ===
st.set_page_config(
    page_title="SIA Robot Control",
    page_icon="🤖",
    layout="wide"
)

# === CONSTANTE ===
MODEL_PATH = 'models/untrained_model.pth' # Sau 'models/trained_model.h5' mai tarziu
SCALER_PATH = 'config/preprocessing_params.pkl'
LABELS_MAP = {0: 'STOP ✋', 1: 'INAINTE ☝️', 2: 'STANGA 👈', 3: 'DREAPTA 👉'}

# === FUNCȚII DE ÎNCĂRCARE (CACHED) ===
@st.cache_resource
def load_resources():
    """Încarcă modelul și scaler-ul o singură dată pentru performanță."""
    # 1. Încărcare Scaler
    scaler = None
    try:
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        st.error(f"Lipsă Scaler! Rulează src/preprocessing/process_data.py")
        return None, None

    # 2. Încărcare Model
    model = GestureClassifier()
    if os.path.exists(MODEL_PATH):
        try:
            # Încercăm să încărcăm weights (pentru PyTorch .pth)
            model.load_state_dict(torch.load(MODEL_PATH))
        except:
            # Fallback dacă formatul e diferit sau corupt
            st.warning("Modelul nu a putut fi încărcat perfect, folosim weights random.")
    
    model.eval()
    return model, scaler

# === INTERFAȚA GRAFICĂ ===
def main():
    # Sidebar - Setări
    st.sidebar.title("🎛️ Panou Control")
    st.sidebar.info("Proiect Rețele Neuronale - Etapa 4")
    
    use_webcam = st.sidebar.toggle("Activează Camera", value=False)
    show_landmarks = st.sidebar.checkbox("Arată Schelet Mână", value=True)
    confidence_threshold = st.sidebar.slider("Prag Siguranță (Threshold)", 0.0, 1.0, 0.5, 0.05)

    st.sidebar.markdown("---")
    st.sidebar.write("### Stare Sistem")
    status_indicator = st.sidebar.empty()

    # Titlu Principal
    st.title("🤖 Control Robot prin Gesturi")
    st.markdown("Sistem Inteligent Artificial pentru detecția comenzilor vizuale.")

    # Layout pe 2 coloane
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 Flux Video Live")
        video_placeholder = st.empty()

    with col2:
        st.subheader("📊 Analiză în Timp Real")
        prediction_text = st.empty()
        confidence_bar = st.empty()
        st.markdown("---")
        probs_container = st.container()

    # === LOGICA DE PROCESARE ===
    if use_webcam:
        model, scaler = load_resources()
        
        if model is None or scaler is None:
            st.stop()

        # Init MediaPipe
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        cap = cv2.VideoCapture(0)
        status_indicator.success("Sistem Online")

        while cap.isOpened() and use_webcam:
            ret, frame = cap.read()
            if not ret:
                st.error("Nu pot accesa camera!")
                break

            # Procesare Imagine
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            predicted_label = "Așteptare..."
            max_prob = 0.0
            all_probs = [0.0] * 4

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    if show_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Extragere Features
                    landmarks = hand_landmarks.landmark
                    row = []
                    for lm in landmarks:
                        row.extend([lm.x, lm.y, lm.z])
                    
                    # Inferență
                    X = np.array([row])
                    X_scaled = scaler.transform(X)
                    
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(X_scaled)
                        outputs = model(X_tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        max_prob, predicted_idx = torch.max(probs, 1)
                        
                        max_prob = max_prob.item()
                        predicted_idx = predicted_idx.item()
                        all_probs = probs[0].tolist()

                    # Logică Threshold
                    if max_prob > confidence_threshold:
                        predicted_label = LABELS_MAP[predicted_idx]
                        # Desenăm și pe imagine
                        cv2.putText(frame, predicted_label, (10, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        predicted_label = "Nesigur (?)"

            # === ACTUALIZARE UI ===
            # 1. Video
            video_placeholder.image(frame, channels="BGR")
            
            # 2. Metrici
            if predicted_label != "Așteptare..." and predicted_label != "Nesigur (?)":
                prediction_text.markdown(f"## Comanda: **{predicted_label}**")
                confidence_bar.progress(max_prob, text=f"Încredere: {max_prob:.1%}")
            else:
                prediction_text.markdown(f"## {predicted_label}")
                confidence_bar.progress(0, text="Așteptare gest...")

            # 3. Detalii Probabilități (Grafic mic)
            with probs_container:
                st.write("Probabilități per clasă:")
                st.write(f"🛑 STOP: {all_probs[0]:.2f}")
                st.write(f"☝️ INAINTE: {all_probs[1]:.2f}")
                st.write(f"👈 STANGA: {all_probs[2]:.2f}")
                st.write(f"👉 DREAPTA: {all_probs[3]:.2f}")

            # Mic delay pentru a nu surmena CPU-ul
            time.sleep(0.03)

        cap.release()
    else:
        status_indicator.warning("Camera Oprită")
        video_placeholder.info("Apasă comutatorul din stânga pentru a porni camera.")

if __name__ == "__main__":
    main()