import torch
import pandas as pd
import numpy as np
import os
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report, precision_score, recall_score

# === CONFIGURARE CĂI ===
# Adăugăm rădăcina proiectului în path pentru a putea importa modulele
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.neural_network.model import GestureClassifier

# Căi Fișiere
DATA_DIR = 'data'
MODELS_DIR = 'models'
RESULTS_DIR = 'results'
DOCS_DIR = 'docs'

MODEL_PATH = os.path.join(MODELS_DIR, 'trained_model.pth')
TEST_X_PATH = os.path.join(DATA_DIR, 'test/X_test.csv')
TEST_Y_PATH = os.path.join(DATA_DIR, 'test/y_test.csv')

METRICS_PATH = os.path.join(RESULTS_DIR, 'test_metrics.json')
CONF_MATRIX_PATH_DOCS = os.path.join(DOCS_DIR, 'confusion_matrix.png')
CONF_MATRIX_PATH_RESULTS = os.path.join(RESULTS_DIR, 'confusion_matrix.png')

# Etichete Clase (Trebuie să corespundă cu cele din antrenare)
LABELS = ['STOP', 'INAINTE', 'STANGA', 'DREAPTA']

def evaluate():
    print("🔍 Începere Evaluare Model...")

    # 1. Verificare existență fișiere
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Eroare: Nu am găsit modelul la {MODEL_PATH}. Rulează întâi 'train.py'!")
        return

    # 2. Încărcare Date Test
    print("   -> Încărcare date de test...")
    try:
        X_test = pd.read_csv(TEST_X_PATH).values.astype(np.float32)
        y_test = pd.read_csv(TEST_Y_PATH).values.flatten().astype(np.int64)
    except Exception as e:
        print(f"❌ Eroare la citirea datelor: {e}")
        return

    # 3. Încărcare Model
    print("   -> Încărcare model antrenat...")
    device = torch.device('cpu') # Evaluăm pe CPU pentru simplitate
    model = GestureClassifier(input_size=63, num_classes=4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # 4. Inferență
    print("   -> Rulare inferență...")
    with torch.no_grad():
        inputs = torch.tensor(X_test).to(device)
        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)
    
    y_pred = predictions.cpu().numpy()

    # 5. Calcul Metrici
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    print(f"\n📊 Rezultate Evaluare:")
    print(f"   Acuratețe: {acc*100:.2f}%")
    print(f"   F1-Score:  {f1:.4f}")
    
    # Salvare în JSON
    metrics = {
        "test_accuracy": float(acc),
        "test_f1_macro": float(f1),
        "test_precision": float(precision),
        "test_recall": float(recall)
    }
    
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"   -> Metrici salvate în '{METRICS_PATH}'")

    # 6. Generare Matrice de Confuzie
    print("   -> Generare grafic Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=LABELS, yticklabels=LABELS)
    plt.title('Matricea de Confuzie (Test Set)')
    plt.ylabel('Real (Ground Truth)')
    plt.xlabel('Predicție Model')
    
    # Salvăm în ambele locații pentru a fi siguri
    plt.savefig(CONF_MATRIX_PATH_DOCS)
    plt.savefig(CONF_MATRIX_PATH_RESULTS)
    plt.close()
    
    print(f"   -> Grafic salvat în '{CONF_MATRIX_PATH_DOCS}'")
    
    # 7. Raport Detaliat în Consolă
    print("\n📋 Raport Detaliat pe Clase:")
    print(classification_report(y_test, y_pred, target_names=LABELS))

if __name__ == "__main__":
    evaluate()