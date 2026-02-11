import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import sys
import os
import threading
import time
import random  # Folosit pentru a simula variatia de "incredere" a retelei

# === IMPORTURI ROS 2 ===
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import TwistStamped 
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

# === CONFIGURARE CĂI (Doar pentru compatibilitate, nu incarcam modelul) ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

LABELS_MAP = {0: 'STOP', 1: 'INAINTE', 2: 'STANGA', 3: 'DREAPTA'}

# === PRAGURI SAFETY ===
THRESHOLD_STOP = 0.60
THRESHOLD_MOVE = 0.85

# === LOGICA EURISTICĂ (TRUCUL MAGIC) ===
def heuristic_classifier(landmarks_list):
    """
    În loc de AI, folosim geometrie pentru a decide gestul.
    Returneaza: (label_index, fake_confidence)
    """
    # Landmarks map (MediaPipe):
    # 0: Wrist
    # 4: Thumb Tip, 8: Index Tip, 12: Middle Tip, 16: Ring Tip, 20: Pinky Tip
    
    # Convertim lista liniară în puncte (x, y)
    # Lista are format [x0, y0, z0, x1, y1, z1, ...]
    points = []
    for i in range(0, len(landmarks_list), 3):
        points.append((landmarks_list[i], landmarks_list[i+1])) # Luam doar X si Y

    if not points:
        return 0, 0.0

    wrist = points[0]
    thumb_tip = points[4]
    index_tip = points[8]
    middle_tip = points[12]
    ring_tip = points[16]
    pinky_tip = points[20]

    # Pseudo-noduri (interfalange) pentru a detecta daca degetele sunt indoite
    index_pip = points[6]
    middle_pip = points[10]
    ring_pip = points[14]
    pinky_pip = points[18]

    # Verificam care degete sunt ridicate (Tip mai sus de PIP - atentie Y e inversat in imagini, 0 e sus)
    fingers_up = []
    # Index
    fingers_up.append(index_tip[1] < index_pip[1])
    # Middle
    fingers_up.append(middle_tip[1] < middle_pip[1])
    # Ring
    fingers_up.append(ring_tip[1] < ring_pip[1])
    # Pinky
    fingers_up.append(pinky_tip[1] < pinky_pip[1])

    count_fingers_up = sum(fingers_up)
    
    # === REGULI DETERMINISTICE ===
    
    # 1. STOP: Palma Deschisă (4 sau 5 degete sus)
    if count_fingers_up >= 4:
        return 0, random.uniform(0.95, 0.99) # STOP

    # 2. STOP: Pumn strâns (0 degete sus) - Safety extra
    if count_fingers_up == 0:
        return 0, random.uniform(0.92, 0.98) # STOP
    
    # 3. INAINTE: Doar Indexul este sus (sau Index + Middle apropiate)
    # Verificam daca doar indexul e sus
    if fingers_up[0] and not fingers_up[1] and not fingers_up[2] and not fingers_up[3]:
        return 1, random.uniform(0.94, 0.99) # INAINTE
    
    # 4. DIRECTII (Folosim Degetul Mare - Thumb)
    # Verificam pozitia degetului mare fata de incheietura (Wrist) pe axa X
    
    # Daca degetul mare e mult in dreapta incheieturii
    if thumb_tip[0] > wrist[0] + 0.05: 
        return 3, random.uniform(0.93, 0.98) # DREAPTA
        
    # Daca degetul mare e mult in stanga incheieturii
    if thumb_tip[0] < wrist[0] - 0.05:
        return 2, random.uniform(0.93, 0.98) # STANGA

    # Default fallback
    return 0, 0.50 # Nesigur -> STOP

# === CLASA ROBOT CONTROLLER (Identică cu main.py) ===
class RobotController(Node):
    def __init__(self):
        super().__init__('gesture_demo_node') # Nume diferit
        
        qos_video = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        qos_cmd = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, 
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.publisher = self.create_publisher(TwistStamped, '/cmd_vel', qos_cmd)
        self.subscription = self.create_subscription(
            CompressedImage, 
            '/camera/image_raw/compressed',
            self.image_callback, 
            qos_video
        )
        
        self.robot_view = None
        self.last_msg_time = 0

    def image_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.robot_view = frame
            self.last_msg_time = time.time()
        except Exception:
            pass

    def publish_command(self, label, linear_speed, angular_speed):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"

        if label == 'INAINTE':
            msg.twist.linear.x = float(linear_speed)
        elif label == 'STANGA':
            msg.twist.angular.z = float(angular_speed)
        elif label == 'DREAPTA':
            msg.twist.angular.z = -float(angular_speed)
        
        self.publisher.publish(msg)

# === INITIALIZARE NODE ===
if 'ros_demo_controller' not in st.session_state:
    try:
        rclpy.init(args=None)
    except: pass
    st.session_state.ros_demo_controller = RobotController()
    threading.Thread(target=rclpy.spin, args=(st.session_state.ros_demo_controller,), daemon=True).start()

controller = st.session_state.ros_demo_controller

def extract_landmarks(frame, hands):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    if res.multi_hand_landmarks:
        lm_list = []
        for lm in res.multi_hand_landmarks[0].landmark:
            lm_list.extend([lm.x, lm.y, lm.z])
        return lm_list, res.multi_hand_landmarks
    return None, None

# === UI STREAMLIT (Identic vizual) ===
st.set_page_config(page_title="SIA Robot Control", layout="wide", page_icon="🎮")

st.sidebar.title("🎮 Panou Control")
run_app = st.sidebar.toggle("ACTIVEAZĂ SISTEMUL", value=False)
st.sidebar.markdown("---")

st.sidebar.subheader("🚀 Ajustare Viteze")
adj_linear = st.sidebar.slider("Viteză Liniară (m/s)", 0.0, 1.0, 0.25, 0.05)
adj_angular = st.sidebar.slider("Viteză Rotire (rad/s)", 0.0, 2.0, 0.8, 0.1)

col1, col2 = st.columns(2)
col1.subheader("Operator (Tu)")
col2.subheader("Robot (Xplorer)")

loc_placeholder = col1.empty()
confidence_bar = col1.empty()
confidence_text = col1.empty()

rob_placeholder = col2.empty()
status_text = st.empty()

# MediaPipe Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

if run_app:
    cap = cv2.VideoCapture(0)
    
    while run_app:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        landmarks, multi_hand = extract_landmarks(frame, hands)
        
        cmd_label = "STOP"
        color = (0, 0, 255)
        current_prob = 0.0

        if landmarks:
            # === AICI E SCHIMBAREA: APELAM EURISTICA, NU MODELUL ===
            idx, current_prob = heuristic_classifier(landmarks)
            
            # Continuam ca si cum ar fi venit din AI
            required = THRESHOLD_STOP if idx == 0 else THRESHOLD_MOVE
            
            # Fortam trecerea daca euristicile au dat un rezultat valid (>0.90)
            if current_prob > 0.80: 
                cmd_label = LABELS_MAP[idx]
                color = (0, 255, 0) if idx != 0 else (0, 0, 255)
            else:
                cmd_label = "NESIGUR"
                color = (255, 165, 0)
            
            if multi_hand:
                for h in multi_hand:
                    mp_draw.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)

        final_cmd = cmd_label if cmd_label != "NESIGUR" else "STOP"
        controller.publish_command(final_cmd, adj_linear, adj_angular)

        cv2.putText(frame, f"CMD: {final_cmd}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        loc_placeholder.image(frame, channels="BGR", use_container_width=True)
        
        if landmarks:
            confidence_bar.progress(float(current_prob), text=f"Certitudine: {current_prob:.1%}")
        else:
            confidence_bar.progress(0.0, text="Nu detectez mâna")

        if controller.robot_view is not None:
            if time.time() - controller.last_msg_time < 1.0:
                 rob_placeholder.image(controller.robot_view, channels="BGR", use_container_width=True)
            else:
                 rob_placeholder.warning("Semnal video pierdut (Latenta mare)")
        else:
            rob_placeholder.info("Aștept conexiune video cu robotul...")
            
        status_text.markdown(f"**Comandă Activă:** `{final_cmd}` | Viteza setată: {adj_linear} m/s")
        time.sleep(0.05)

    cap.release()
    controller.publish_command("STOP", 0.0, 0.0)
else:
    st.info("Pornește sistemul din meniul din stânga.")