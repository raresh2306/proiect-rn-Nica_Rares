import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import json

# Adăugăm calea către rădăcina proiectului
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.neural_network.model import GestureClassifier

# === CONFIGURARE NIVEL 2 ===
HPARAMS = {
    "epochs": 100,             # Setăm mare, dar Early Stopping îl va opri devreme
    "batch_size": 32,
    "learning_rate": 0.01,     # Începem mai agresiv, Scheduler-ul îl va scădea
    "patience": 5,             # Early Stopping: Stop după 5 epoci fără îmbunătățire
    "noise_level": 0.005       # Augmentare: Zgomot Gaussian aplicat coordonatelor
}

DATA_DIR = 'data'
MODELS_DIR = 'models'
RESULTS_DIR = 'results'
DOCS_DIR = 'docs'              # Pentru loss_curve.png cerut la Nivel 2

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

def load_data():
    """Încarcă datele."""
    print("🔄 Se încarcă datele...")
    X_train = pd.read_csv(os.path.join(DATA_DIR, 'train/X_train.csv')).values.astype(np.float32)
    y_train = pd.read_csv(os.path.join(DATA_DIR, 'train/y_train.csv')).values.flatten().astype(np.int64)
    X_val = pd.read_csv(os.path.join(DATA_DIR, 'validation/X_val.csv')).values.astype(np.float32)
    y_val = pd.read_csv(os.path.join(DATA_DIR, 'validation/y_val.csv')).values.flatten().astype(np.int64)
    X_test = pd.read_csv(os.path.join(DATA_DIR, 'test/X_test.csv')).values.astype(np.float32)
    y_test = pd.read_csv(os.path.join(DATA_DIR, 'test/y_test.csv')).values.flatten().astype(np.int64)
    return X_train, y_train, X_val, y_val, X_test, y_test

def add_gaussian_noise(data, noise_level=0.005):
    """
    Augmentare Nivel 2: Adaugă zgomot aleator peste coordonate.
    Simulează tremuratul mâinii sau erori de senzor.
    """
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

def train_model():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    # Conversie la tensori (datele statice)
    y_train_t = torch.tensor(y_train)
    X_val_t = torch.tensor(X_val)
    y_val_t = torch.tensor(y_val)
    
    model = GestureClassifier(input_size=63, num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=HPARAMS['learning_rate'])
    
    # === Nivel 2: Learning Rate Scheduler (FIXED) ===
    # Am scos 'verbose=True' care cauza eroarea pe versiuni noi de PyTorch
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
    
    # Variabile pentru Early Stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\n🚀 Începe antrenarea avansată (Early Stopping={HPARAMS['patience']}, Augmentare=On)...")
    
    for epoch in range(HPARAMS['epochs']):
        model.train()
        
        # Shuffle la fiecare epocă
        permutation = torch.randperm(X_train.shape[0])
        epoch_loss = 0
        correct = 0
        total = 0
        
        for i in range(0, X_train.shape[0], HPARAMS['batch_size']):
            indices = permutation[i:i+HPARAMS['batch_size']]
            batch_x_numpy = X_train[indices]
            batch_y = y_train_t[indices]
            
            # === Nivel 2: Augmentare "On-the-fly" ===
            batch_x_augmented = add_gaussian_noise(batch_x_numpy, HPARAMS['noise_level'])
            batch_x = torch.tensor(batch_x_augmented, dtype=torch.float32)
            
            # Forward & Backward
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
        avg_loss = epoch_loss / (len(X_train) / HPARAMS['batch_size'])
        acc = correct / total
        
        # Validare
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_acc = (val_predicted == y_val_t).sum().item() / len(y_val_t)
        
        # Update Scheduler (Monitorizăm LR manual)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        history['loss'].append(avg_loss)
        history['val_loss'].append(val_loss)
        history['accuracy'].append(acc)
        history['val_accuracy'].append(val_acc)
        
        print(f"Epoch [{epoch+1}/{HPARAMS['epochs']}] Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}")
        
        # === Nivel 2: Early Stopping Logic ===
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Salvăm doar cel mai bun model
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'trained_model.pth'))
        else:
            patience_counter += 1
            print(f"   ⏳ Patience {patience_counter}/{HPARAMS['patience']}")
            if patience_counter >= HPARAMS['patience']:
                print("🛑 Early Stopping activat! Antrenarea s-a oprit pentru a preveni overfitting.")
                break

    print("\n💾 Cel mai bun model a fost salvat în 'models/trained_model.pth'")
    
    # Salvare Istoric
    pd.DataFrame(history).to_csv(os.path.join(RESULTS_DIR, 'training_history.csv'), index=False)
    
    # Evaluare Finală
    evaluate_model(model, X_test, y_test)
    plot_history_level2(history)

def evaluate_model(model, X_test, y_test):
    # Reîncărcăm cea mai bună versiune a modelului
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'trained_model.pth')))
    model.eval()
    
    with torch.no_grad():
        outputs = model(torch.tensor(X_test))
        _, predicted = torch.max(outputs, 1)
        
    acc = accuracy_score(y_test, predicted.numpy())
    f1 = f1_score(y_test, predicted.numpy(), average='macro')
    
    metrics = {"test_accuracy": acc, "test_f1_macro": f1}
    with open(os.path.join(RESULTS_DIR, 'test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Matrice confuzie
    cm = confusion_matrix(y_test, predicted.numpy())
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.savefig(os.path.join(DOCS_DIR, 'confusion_matrix.png'))
    plt.close()

def plot_history_level2(history):
    """
    Generează graficul loss_curve.png cerut la Nivel 2
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Evolution (Early Stopping & Augmentation)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Salvarea OBLIGATORIE în docs/loss_curve.png
    save_path = os.path.join(DOCS_DIR, 'loss_curve.png')
    plt.savefig(save_path)
    print(f"📈 Grafic Nivel 2 salvat în: {save_path}")
    plt.close()

if __name__ == "__main__":
    train_model()