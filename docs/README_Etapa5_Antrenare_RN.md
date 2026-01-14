# 📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Nica Daniel-Rares
**Link Repository GitHub:** https://github.com/raresh2306/proiect-rn-Nica_Rares
**Data predării:** 14.01.2026

---

## Scopul Etapei 5

Această etapă corespunde punctului **6. Configurarea și antrenarea modelului RN** din lista de 9 etape - slide 2 **RN Specificatii proiect.pdf**.

**Obiectiv principal:** Antrenarea efectivă a modelului RN definit în Etapa 4, evaluarea performanței și integrarea în aplicația completă.

---


##  Cerințe Structurate pe 3 Niveluri

## 1. Nivel 1 – Livrabile Obligatorii (70% din punctaj)

Am îndeplinit toate cerințele obligatorii pentru antrenarea și integrarea modelului:

**Checklist realizat:**
- [x] **Antrenare model:** Modelul definit în Etapa 4 a fost antrenat pe setul de date complet (9.000 samples, 100% originale).
- [x] **Configurație:** S-au rulat **50 epoci** (minim 10 cerute) cu un **Batch size de 32** (în intervalul 8-32).
- [x] **Împărțire date:** Setul a fost împărțit standard: Train (70%) / Validation (15%) / Test (15%).
- [x] **Metrici:** Acuratețea finală este **100%** (peste pragul de 65%), iar F1-score este **1.0** (peste 0.60).
- [x] **Salvare:** Modelul este salvat în format PyTorch: `models/trained_model.pth`.
- [x] **Integrare UI:** Aplicația `src/app/main.py` încarcă acum `trained_model.pth` și realizează inferență reală.

---

### Tabel Justificare Hiperparametri (OBLIGATORIU)

| Hiperparametru | Valoare Aleasă | Justificare |
| :--- | :--- | :--- |
| **Epochs** | 50 | Deși modelul convergea după 15 epoci, am rulat 50 pentru a garanta stabilitatea ponderilor și a minimiza Loss-ul. |
| **Batch Size** | 32 | Valoare optimă pentru o rețea MLP mică; oferă un echilibru între viteza de calcul și generalizare (evită blocarea în minime locale). |
| **Learning Rate** | 0.001 | Rata standard pentru optimizatorul Adam. Asigură o învățare rapidă la început și fină spre final. |
| **Optimizer** | Adam | Ales pentru eficiență pe date rare/sparse; converge mult mai repede decât SGD clasic. |
| **Loss Function** | CrossEntropy | Standardul matematic pentru probleme de clasificare Multi-Class (4 clase). |

---

### Integrare în UI

**Dovada inferenței reale:**
Interfața a fost actualizată pentru a folosi modelul antrenat.
Screenshot-ul demonstrativ se află în: `docs/printscreens/inference_real.png`

#### Tabel Hiperparametri și Justificări (OBLIGATORIU - Nivel 1)

Completați tabelul cu hiperparametrii folosiți și **justificați fiecare alegere**:

| **Hiperparametru** | **Valoare Aleasă** | **Justificare** |
|--------------------|-------------------|-----------------|
| Learning rate | Ex: 0.001 | Valoare standard pentru Adam optimizer, asigură convergență stabilă |
| Batch size | Ex: 32 | Compromis memorie/stabilitate pentru N=[numărul vostru] samples |
| Number of epochs | Ex: 50 | Cu early stopping după 10 epoci fără îmbunătățire |
| Optimizer | Ex: Adam | Adaptive learning rate, potrivit pentru RN cu [numărul vostru] straturi |
| Loss function | Ex: Categorical Crossentropy | Clasificare multi-class cu K=[numărul vostru] clase |
| Activation functions | Ex: ReLU (hidden), Softmax (output) | ReLU pentru non-linearitate, Softmax pentru probabilități clase |

**Justificare detaliată batch size:**
```
Am ales batch_size=32 pentru setul de antrenare de N_train=6,300 samples (70% din 9,000) → 6,300/32 ≈ 197 iterații/epocă.
Aceasta oferă un echilibru optim între:
- Generalizare (Gradient Noise): Un batch de 32 introduce suficient "zgomot" în gradient pentru a ajuta modelul să evadeze din minime locale și să prevină overfitting-ul pe datele geometrice foarte curate.
- Optimizare CPU: Deși memoria nu este o problemă (input vector 1D), dimensiunea 32 este optimizată pentru cache-ul procesorului și instrucțiunile vectoriale.
- Timp antrenare: Asigură actualizarea ponderilor de ~200 de ori pe epocă, ducând la o convergență rapidă (Loss < 0.01) în mai puțin de 20 de epoci.
```

---

### Nivel 2 – Recomandat (85-90% din punctaj):

Am implementat toate cerințele avansate pentru a asigura robustețea modelului în condiții industriale.

1. **Early Stopping** 
   - **Configurare:** Monitorizare `val_loss`, cu `patience=5`.
   - **Rezultat:** Antrenamentul s-a oprit automat la epoca **17/100** ![vezi aici](docs\printscreens\antrenare_avansata.png), prevenind overfitting-ul pe datele de antrenare.

2. **Learning Rate Scheduler** 
   - **Configurare:** Am folosit `ReduceLROnPlateau` cu `mode='min'`, `factor=0.5` și `patience=2`.
   - **Rezultat:** Rata de învățare a scăzut dinamic de la `0.01` la `0.0025` pe măsură ce modelul a atins platoul, permițând o convergență fină (vezi coloana `LR` în log-uri).

3. **Augmentări relevante domeniului (Landmarks)** 
   - **Metodă:** `Gaussian Noise Injection` (`noise_level=0.005`).
   - **Justificare:** Deoarece inputul este constituit din coordonate (x, y, z), rotațiile clasice de imagini nu se aplică direct. Am simulat **vibrațiile senzorului și tremuratul mâinii operatorului** prin adăugarea de zgomot aleator peste coordonate la fiecare batch.

4. **Grafic Loss și Val_Loss** 
   - Graficul de mai jos demonstrează convergența corectă (curbele Train/Validation merg în paralel, fără divergență majoră).
   ![Loss Curve](docs/loss_curve.png)

   **Dovada execuției (Terminal - Early Stopping & Scheduler):**
   ![Terminal Log Level 2](docs/printscreens/antrenare_avansata.png)

**Indicatori Finali Nivel 2:**
- **Acuratețe:** 100% (≥ 75% țintă)
- **F1-score:** 1.00 (≥ 0.70 țintă)


## Analiză Erori în Context Industrial:

**Nu e suficient să raportați doar acuratețea globală.** Analizați performanța în contextul aplicației voastre industriale:

### 1. Pe ce clase greșește cel mai mult modelul?

**Observație:** Deși Matricea de Confuzie pe setul de test este ideală (0% erori), în testele live modelul oscilează între clasa **'ÎNAINTE' (Aratator in sus)** și **'STOP' (Palma deschisa)** în timpul tranzițiilor rapide.
**Cauză posibilă:** Dataset-ul de antrenare conține doar posturi statice ferme. Geometriile intermediare (mâna pe jumătate deschisă) nu sunt cunoscute de model, generând predicții instabile timp de 100-200ms.


### 2. Ce caracteristici ale datelor cauzează erori?

**Observație:** Modelul devine instabil când apar **ocluzii parțiale** (degete ascunse) sau **rotații extreme** ale mâinii.
**Context Industrial:** În exploatare, operatorul nu ține mâna perfect frontal. Când mâna este rotită la 90°, camera 2D pierde vizibilitatea asupra degetului mare, iar MediaPipe generează coordonate `Z` (adâncime) eronate, ducând la confuzia între gestul "ÎNAINTE" și "DREAPTA".


### 3. Ce implicații are pentru aplicația industrială?

**FALSE NEGATIVES (STOP nedetectat):** CRITIC → Risc major de coliziune sau accidentare dacă robotul ignoră comanda de oprire.
**FALSE POSITIVES (Oprire falsă):** ACCEPTABIL → Robotul se oprește preventiv din greșeală; scade eficiența (timp pierdut), dar siguranța este menținută.

**Prioritate:** Minimizarea False Negatives pentru clasa 'STOP' (Recall Maxim).
**Soluție:** Implementare praguri asimetrice: Threshold relaxat (0.6) pentru 'STOP' vs. Threshold strict (0.85) pentru 'ÎNAINTE'.


### 4. Ce măsuri corective propuneți?

**Măsuri corective:**
**Măsuri corective implementate:**

1.  **Praguri de Decizie Asimetrice (Safety Logic):**
    * **Implementare:** Am modificat codul de inferență (`main.py`) pentru a folosi un prag relaxat (**0.60**) pentru detectarea clasei critice 'STOP' și un prag strict (**0.85**) pentru comenzile de mișcare.
    * **Scop:** Minimizarea riscului de *False Negatives* (neoprire) în situații de urgență.

2.  **Antrenare Robustă cu Zgomot (Noise Injection):**
    * **Implementare:** În scriptul `train.py`, am introdus augmentarea cu zgomot Gaussian (`noise_level=0.005`) peste coordonatele de intrare.
    * **Scop:** Modelul a învățat să ignore variațiile fine (tremuratul mâinii), devenind stabil la imperfecțiunile senzorului.

3.  **Filtrare prin Prag de Încredere (Confidence Threshold):**
    * **Implementare:** Sistemul ignoră complet orice predicție cu o probabilitate sub **70%** (afișând starea "NESIGUR"), pentru a nu trimite comenzi eronate robotului atunci când mâna se află în tranziție sau repaus.

---


## Verificare Consistență cu State Machine (Etapa 4)

Antrenarea și inferența trebuie să respecte fluxul din State Machine-ul vostru definit în Etapa 4.

| Stare (State Machine - Etapa 4) | Implementare Cod (Etapa 5) | Fișier/Resursă |
| :--- | :--- | :--- |
| **INITIALIZATION** | Încărcare model antrenat și scaler salvat | `load_resources()` încarcă `trained_model.pth` |
| **ACQUIRE_IMAGE** | Captură frame video de la camera web | `cap.read()` (OpenCV) |
| **FEATURE_EXTRACTION** | Extragere 21 puncte scheletice (x,y,z) | `mp_hands.process()` (MediaPipe) |
| **PREPROCESSING** | Normalizare date folosind scaler-ul antrenat | `scaler.transform(input_data)` |
| **RN_INFERENCE** | Forward pass prin rețeaua antrenată | `outputs = model(input_tensor)` |
| **DECISION_LOGIC** | Aplicare praguri asimetrice (Safety Logic) | `if max_prob > THRESHOLD_STOP/MOVE` |
| **ACTUATE/DISPLAY** | Afișare comandă și probabilități în UI | `st.markdown(...)` / `cv2.putText` |

**În `src/app/main.py` (UI actualizat):**


**TOATE stările** din State Machine sunt implementate cu modelul antrenat:

1. Încărcarea Modelului (load_resources)

```python
# ÎNAINTE (Etapa 4 - model dummy):
# Se încărca un model neinițializat, doar arhitectura goală
MODEL_PATH = 'models/untrained_model.pth' 
# ... în funcție:
model = GestureClassifier(input_size=63, num_classes=4)
# Nu se încărca nicio stare (state_dict), ponderile erau random


# ACUM (Etapa 5 - model antrenat):
# Se încarcă modelul salvat după antrenare
MODEL_PATH = 'models/trained_model.pth' 
# ... în funcție:
model = GestureClassifier(input_size=63, num_classes=4)
# Încărcăm "creierul" antrenat
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval() # Punem modelul în mod evaluare (fixează ponderile)
```

2. Procesarea și Predicția (în bucla while)

```python
# ÎNAINTE (Etapa 4 - model dummy):
# Se făcea o trecere prin rețea, dar rezultatul era aleatoriu
outputs = model(input_tensor)

# Uneori se folosea chiar o logică hardcodată pentru demo:
# if landmarks[0].y < 0.5: 
#     predicted_label = "INAINTE"


# ACUM (Etapa 5 - model antrenat):
# Inferență matematică reală
with torch.no_grad():
    outputs = model(input_tensor)
    probs = F.softmax(outputs, dim=1) # Calculăm probabilitățile reale (0-100%)

probabilities = probs.numpy()[0]
predicted_idx = np.argmax(probabilities)
max_prob = probabilities[predicted_idx]
```

3. Logica de Decizie (Safety)

```python
# ÎNAINTE (Etapa 4 - model dummy):
# Un simplu slider pentru prag, aplicat la fel pentru orice
if max_prob > confidence_threshold:
    predicted_label = LABELS_MAP[predicted_idx]


# ACUM (Etapa 5 - model antrenat):
# Logică diferențiată pentru siguranță
if predicted_idx == 0: # STOP
    required_threshold = THRESHOLD_STOP # 0.60 (Mai sensibil)
else: # MIȘCARE
    required_threshold = THRESHOLD_MOVE # 0.85 (Mai strict)

if max_prob > required_threshold:
    predicted_label = LABELS_MAP[predicted_idx]
```

---



## Structura Repository-ului la Finalul Etapei 5

**Clarificare organizare:** Vom folosi **README-uri separate** pentru fiecare etapă în folderul `docs/`:

```
proiect-rn-Nica_Rares/
├── README.md                           # Overview general proiect (actualizat)
├── etapa3_analiza_date.md         # Din Etapa 3
├── etapa4_arhitectura_sia.md      # Din Etapa 4
├── etapa5_antrenare_model.md      # ← ACEST FIȘIER (completat)
│
├── docs/
│   ├── state_machine.png              # Din Etapa 4
│   ├── loss_curve.png                 # NOU - Grafic antrenare
│   ├── confusion_matrix.png           # (opțional - Nivel 3)
│   └── printscreens/
│       ├── inference_real.png         # NOU - OBLIGATORIU
│       └── ui_demo.png                # Din Etapa 4
│
├── data/                             
│   ├── generated/                
│   ├── train/
│   ├── validation/
│   └── test/
│
├── src/
│   ├── data_acquisition/              
│   ├── preprocessing/                
│   ├── neural_network/
│   │   ├── model.py                   # Din Etapa 4
│   │   ├── train.py                   # NOU - Script antrenare
│   │   └── evaluate.py                # NOU - Script evaluare
│   └── app/
│       └── main.py                    # ACTUALIZAT - încarcă model antrenat
│
├── models/
│   ├── untrained_model.pth             # Din Etapa 4
│   └──trained_model.pth               # NOU - OBLIGATORIU
│
├── results/                           # NOU - Folder rezultate antrenare
│   ├── training_history.csv           # OBLIGATORIU - toate epoch-urile
│   ├── test_metrics.json              # Metrici finale pe test set
│   └── hyperparameters.yaml           # Hiperparametri folosiți
│
├── config/
│   └── preprocessing_params.pkl       # Din Etapa 3 (NESCHIMBAT)
│
├── requirements.txt                    # Actualizat
└── .gitignore
```

**Diferențe față de Etapa 4:**
- Adăugat `docs/etapa5_antrenare_model.md`
- Adăugat `docs/loss_curve.png`
- Adăugat `models/trained_model.pth` - OBLIGATORIU
- Adăugat `results/` cu history și metrici
- Adăugat `src/neural_network/train.py` și `evaluate.py`
- Actualizat `src/app/main.py` să încarce model antrenat
---

## Instrucțiuni de Rulare (Actualizate față de Etapa 4)

Aceste instrucțiuni sunt specifice implementării curente bazate pe **PyTorch** și **Streamlit**.

---

## 1. Setup Mediu

Asigurați-vă că sunteți în rădăcina proiectului și aveți toate dependențele instalate:

```bash
pip install -r requirements.txt
```

---

## 2. Pregătire Date (Opțional)

Dacă doriți să regenerați dataset-ul sau ați capturat date noi folosind:

```
src/data_acquisition/capture_data.py
```

rulați pasul de procesare, normalizare și împărțire a datelor (**Train / Val / Test**):

```bash
python src/preprocessing/process_data.py
```

**Efecte:**
- Actualizarea fișierelor din folderul `data/`
- Actualizarea scaler-ului în `config/preprocessing_params.pkl`

---

## 3. Antrenare Model

Scriptul de antrenare rulează automat cu parametrii definiți intern:
- 100 epoci
- Early Stopping
- Augmentare date

Rulare:

```bash
python src/neural_network/train.py
```

### Output așteptat
- Log-uri cu progresul epocilor și scăderea **Loss**
- Mesaj de **Early Stopping** dacă modelul converge rapid
- Salvarea automată a modelului în:
  ```
  models/trained_model.pth
  ```
- Generarea graficului:
  ```
  docs/loss_curve.png
  ```

---

## 4. Evaluare Model

Pentru a calcula metricile finale și a genera matricea de confuzie pe setul de test (separat de antrenare):

```bash
python src/neural_network/evaluate.py
```

### Output așteptat
- Raport detaliat în consolă:
  - Acuratețe
  - F1 Score
  - Precizie
  - Recall
- Salvarea metricilor în:
  ```
  results/test_metrics.json
  ```
- Actualizarea matricei de confuzie în:
  ```
  docs/confusion_matrix.png
  ```

---

## 5. Lansare Aplicație Live (Inference)

Porniți interfața grafică pentru a testa modelul în timp real folosind camera web:

```bash
streamlit run src/app/main.py
```

### Note
- Aplicația va încărca automat modelul:
  ```
  models/trained_model.pth
  ```
- Pentru a vizualiza logica de siguranță (praguri asimetrice), verificați **sidebar-ul aplicației**.

---

## Checklist Final – Bifați Totul Înainte de Predare

### Prerequisite Etapa 4 (verificare)
- [x] State Machine există și e documentat în `docs/state_machine.*`
- [x] Contribuție ≥40% date originale verificabilă în `data/generated/dataset_original.csv`
- [x] Cele 3 module din Etapa 4 funcționale

### Preprocesare și Date
- [x] Split train/val/test: 70/15/15% 
- [x] Scaler din Etapa 3 folosit consistent (`config/preprocessing_params.pkl`)

### Antrenare Model - Nivel 1 (OBLIGATORIU)
- [x] Model antrenat de la ZERO (nu fine-tuning pe model pre-antrenat)
- [x] Minimum 10 epoci rulate (verificabil în `results/training_history.csv`)
- [x] Tabel hiperparametri + justificări completat în acest README
- [x] Metrici calculate pe test set: **Accuracy ≥65%**, **F1 ≥0.60**
- [x] Model salvat în `models/trained_model.h5` (sau .pt, .lvmodel)
- [x] `results/training_history.csv` există cu toate epoch-urile

### Integrare UI și Demonstrație - Nivel 1 (OBLIGATORIU)
- [x] Model ANTRENAT încărcat în UI din Etapa 4
- [x] UI face inferență REALĂ cu predicții corecte
- [x] Screenshot inferență reală în `docs/printscreens/inference_real.png`
- [x] Verificat: predicțiile sunt diferite față de Etapa 4 (când erau random)

### Documentație Nivel 2 (dacă aplicabil)
- [x] Early stopping implementat și documentat în cod
- [x] Learning rate scheduler folosit (ReduceLROnPlateau / StepLR)
- [x] Augmentări relevante domeniu aplicate
- [x] Grafic loss/val_loss salvat în `docs/loss_curve.png`
- [x] Analiză erori în context industrial completată 
- [x] Metrici Nivel 2: **Accuracy ≥75%**, **F1 ≥0.70**

### Documentație Nivel 3 Bonus (dacă aplicabil)
- [ ] Comparație 2+ arhitecturi (tabel comparativ + justificare)
- [ ] Export ONNX/TFLite + benchmark latență (<50ms demonstrat)
- [ ] Confusion matrix + analiză 5 exemple greșite cu implicații

### Verificări Tehnice
- [x] `requirements.txt` actualizat cu toate bibliotecile noi
- [x] Toate path-urile RELATIVE
- [x] Cod nou comentat în limba română sau engleză (minimum 15%)
- [ ] `git log` arată commit-uri incrementale (NU 1 commit gigantic)
- [x] Verificare anti-plagiat: toate punctele 1-5 respectate

### Verificare State Machine (Etapa 4)
- [x] Fluxul de inferență respectă stările din State Machine
- [x] Toate stările critice (PREPROCESS, INFERENCE, ALERT) folosesc model antrenat
- [x] UI reflectă State Machine-ul pentru utilizatorul final

### Pre-Predare
- [x] `docs/etapa5_antrenare_model.md` completat cu TOATE secțiunile
- [x] Structură repository conformă: `docs/`, `results/`, `models/` actualizate
- [x] Commit: `"Etapa 5 completă – Accuracy=X.XX, F1=X.XX"`
- [ ] Tag: `git tag -a v0.5-model-trained -m "Etapa 5 - Model antrenat"`
- [ ] Push: `git push origin main --tags`
- [x] Repository accesibil (public sau privat cu acces profesori)

---

## Livrabile Obligatorii (Nivel 1)

Asigurați-vă că următoarele fișiere există și sunt completate:

1. **`docs/etapa5_antrenare_model.md`** (acest fișier) cu:
   - Tabel hiperparametri + justificări (complet)
   - Metrici test set raportate (accuracy, F1)
   - (Nivel 2) Analiză erori context industrial (4 paragrafe)

2. **`models/trained_model.h5`** (sau `.pt`, `.lvmodel`) - model antrenat funcțional

3. **`results/training_history.csv`** - toate epoch-urile salvate

4. **`results/test_metrics.json`** - metrici finale:

Exemplu:
```json
{
  "test_accuracy": 0.7823,
  "test_f1_macro": 0.7456,
  "test_precision_macro": 0.7612,
  "test_recall_macro": 0.7321
}
```

5. **`docs/screenshots/inference_real.png`** - demonstrație UI cu model antrenat

6. **(Nivel 2)** `docs/loss_curve.png` - grafic loss vs val_loss

7. **(Nivel 3)** `docs/confusion_matrix.png` + analiză în README

---
