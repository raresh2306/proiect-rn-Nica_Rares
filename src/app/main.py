import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
import joblib
import sys
import os

# === CONFIGURARE CĂI ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.neural_network.model import GestureClassifier

MODEL_PATH = 'models/trained_model.pth'
SCALER_PATH = 'config/preprocessing_params.pkl'

LABELS_MAP = {0: 'STOP', 1: 'INAINTE', 2: 'STANGA', 3: 'DREAPTA'}

# === CONFIGURARE PRAGURI ASIMETRICE (SAFETY) ===
# STOP are prioritate (prag mai mic), Mișcarea cere siguranță (prag mai mare)
THRESHOLD_STOP = 0.60
THRESHOLD_MOVE = 0.85

@st.cache_resource
def load_resources():
    try:
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError:
        st.error(f"Eroare: Nu am găsit scalerul la {SCALER_PATH}.")
        return None, None
    try:
        model = GestureClassifier(input_size=63, num_classes=4)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
    except FileNotFoundError:
        st.error(f"Eroare: Nu am găsit modelul la {MODEL_PATH}.")
        return None, None
    return model, scaler

def extract_landmarks(frame, hands_detector):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb_frame)
    landmarks_data = []
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        for lm in hand_landmarks.landmark:
            landmarks_data.extend([lm.x, lm.y, lm.z])
        return landmarks_data, results.multi_hand_landmarks
    return None, None

# === INTERFAȚA STREAMLIT ===
st.set_page_config(page_title="Control Robot SIA", layout="wide")

st.sidebar.title("🎛️ Panou Control")
st.sidebar.info("Proiect RN - Etapa 5 (Safety Implemented)")

run_camera = st.sidebar.toggle("Activează Camera", value=False)
show_skeleton = st.sidebar.checkbox("Arată Schelet Mână", value=True)

# Afișăm valorile pragurilor în Sidebar (doar informativ)
st.sidebar.markdown("### 🛡️ Setări Siguranță")
st.sidebar.code(f"STOP Threshold: {THRESHOLD_STOP}")
st.sidebar.code(f"MOVE Threshold: {THRESHOLD_MOVE}")
st.sidebar.caption("Praguri asimetrice pentru prevenirea accidentelor.")

st.title("🤖 Control Robot prin Gesturi (Live)")

col1, col2 = st.columns([2, 1])
model, scaler = load_resources()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

with col1:
    video_placeholder = st.empty()

with col2:
    st.markdown("### 📊 Analiză în Timp Real")
    current_action_text = st.empty()
    confidence_bar = st.empty()
    st.divider()
    probs_container = st.empty()

if run_camera and model is not None:
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened() and run_camera:
        ret, frame = cap.read()
        if not ret:
            st.error("Nu pot accesa camera web.")
            break
            
        frame = cv2.flip(frame, 1)
        landmarks, multi_hand_landmarks = extract_landmarks(frame, hands)
        
        predicted_label = "Așteptare..."
        max_prob = 0.0
        probabilities = [0.0, 0.0, 0.0, 0.0]
        required_threshold = 0.0 # Pentru vizualizare
        
        if landmarks:
            input_data = np.array([landmarks], dtype=np.float32)
            input_scaled = scaler.transform(input_data)
            input_tensor = torch.tensor(input_scaled)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = F.softmax(outputs, dim=1)
                
            probabilities = probs.numpy()[0]
            predicted_idx = np.argmax(probabilities)
            max_prob = probabilities[predicted_idx]
            
            # === LOGICA ASIMETRICĂ IMPLEMENTATĂ AICI ===
            if predicted_idx == 0: # Dacă e STOP
                required_threshold = THRESHOLD_STOP
            else: # Dacă e MIȘCARE (Inainte, Stanga, Dreapta)
                required_threshold = THRESHOLD_MOVE

            if max_prob > required_threshold:
                predicted_label = LABELS_MAP[predicted_idx]
                color = (0, 255, 0) # Verde
            else:
                predicted_label = "NESIGUR"
                color = (0, 165, 255) # Portocaliu

            cv2.putText(frame, f"{predicted_label} ({max_prob:.2f})", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            if show_skeleton and multi_hand_landmarks:
                for hand_lms in multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        if predicted_label == "NESIGUR":
             current_action_text.markdown(f"## Comanda: ⚠️ **{predicted_label}**")
             st.caption(f"Necesar: > {required_threshold}")
        elif predicted_label == "Așteptare...":
             current_action_text.markdown(f"## Stare: 💤 **{predicted_label}**")
        else:
             current_action_text.markdown(f"## Comanda: 🟢 **{predicted_label}**")
        
        confidence_bar.progress(float(max_prob), text=f"Nivel Încredere: {max_prob:.1%}")
        
        with probs_container.container():
            st.markdown("#### Detalii Probabilități:")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.write(f"**STOP:** `{probabilities[0]:.2f}`")
                st.write(f"**INAINTE:** `{probabilities[1]:.2f}`")
            with col_p2:
                st.write(f"**STANGA:** `{probabilities[2]:.2f}`")
                st.write(f"**DREAPTA:** `{probabilities[3]:.2f}`")

    cap.release()
else:
    st.info("Apasă butonul 'Activează Camera' din stânga pentru a începe.")
    image_placeholder = st.empty()
    image_placeholder.markdown("""
    <div style='text-align: center; color: gray; padding: 50px; border: 2px dashed gray;'>
        Camera Oprită
    </div>
    """, unsafe_allow_html=True)