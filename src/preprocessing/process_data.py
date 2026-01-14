import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# === CONFIGURARE ===
INPUT_FILE = 'data/generated/dataset_original.csv'
CONFIG_PATH = 'config/preprocessing_params.pkl'

# Căi pentru salvarea seturilor de date
PATHS = {
    'train': 'data/train',
    'val': 'data/validation',
    'test': 'data/test'
}

def process_data():
    # 1. Verificăm dacă există datele brute
    if not os.path.exists(INPUT_FILE):
        print(f"EROARE: Fișierul {INPUT_FILE} nu există!")
        print("Rulează întâi 'src/data_acquisition/capture_data.py' și colectează date.")
        return

    print("=== ÎNCEPERE PREPROCESARE ===")
    
    # 2. Încărcare date
    df = pd.read_csv(INPUT_FILE)
    print(f"Date încărcate: {len(df)} rânduri.")

    # Separăm Features (X) de Labels (y)
    # Coloana 'label' este ținta, restul (lm_0_x...) sunt features
    X = df.drop('label', axis=1)
    y = df['label']

    # 3. Împărțire seturi (70% Train, 15% Val, 15% Test)
    # Pas 1: Separăm Test (15%) de restul (85%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )

    # Pas 2: Separăm Validation (15% din total -> aprox 17.6% din ce a rămas)
    # 0.15 / 0.85 = 0.1764
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1764, stratify=y_temp, random_state=42
    )

    print(f"Split realizat: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # 4. Normalizare (StandardScaler)
    # Scaler-ul se antrenează DOAR pe datele de antrenare!
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Aplicăm aceeași transformare pe val și test
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 5. Salvare Scaler (CRITIC PENTRU ETAPA 5)
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler salvat în: {CONFIG_PATH}")

    # 6. Salvare fișiere CSV procesate
    # Salvăm X și y separat pentru fiecare set
    save_set(X_train_scaled, y_train, 'train')
    save_set(X_val_scaled, y_val, 'val')
    save_set(X_test_scaled, y_test, 'test')

    print("=== PREPROCESARE FINALIZATĂ CU SUCCES ===")

def save_set(X_data, y_data, set_name):
    # Creăm folderul dacă nu există
    output_dir = PATHS[set_name]
    os.makedirs(output_dir, exist_ok=True)
    
    # Convertim X înapoi în DataFrame pentru salvare ușoară (opțional, dar util pentru debug)
    df_X = pd.DataFrame(X_data)
    df_y = pd.DataFrame(y_data)
    
    df_X.to_csv(os.path.join(output_dir, f'X_{set_name}.csv'), index=False)
    df_y.to_csv(os.path.join(output_dir, f'y_{set_name}.csv'), index=False)
    print(f"Salvat {set_name} în {output_dir}")

if __name__ == "__main__":
    process_data()