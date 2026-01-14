# 📘 README – Etapa 3: Analiza și Pregătirea Setului de Date pentru Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Nica Daniel-Rares

**Data:** 02.12.2025  

---

## Introducere

Acest document descrie activitățile realizate în **Etapa 3**, în care se analizează și se preprocesează setul de date necesar proiectului „Rețele Neuronale". Scopul etapei este pregătirea corectă a datelor pentru instruirea modelului RN, respectând bunele practici privind calitatea, consistența și reproductibilitatea datelor.

---

##  1. Structura Repository-ului Github (versiunea Etapei 3)

```
proiect-rn-Nica_Rares/
├── README.md
├── docs/
│   └── datasets/          # descriere seturi de date, surse, diagrame
├── data/
│   ├── generated/         # date originale brut (dataset_original.csv) - 9000 samples
│   ├── train/             # set de instruire (X_train.csv, y_train.csv)
│   ├── validation/        # set de validare (X_val.csv, y_val.csv)
│   └── test/              # set de testare (X_test.csv, y_test.csv)
├── src/
│   ├── preprocessing/     # funcții pentru preprocesare (process_data.py)
│   ├── data_acquisition/  # generare / achiziție date (capture_data.py)
│   └── neural_network/    # implementarea RN (în etapa următoare)
├── config/                # fișiere de configurare (preprocessing_params.pkl)
└── requirements.txt       # dependențe Python (mediapipe, opencv, sklearn)
```

---

##  2. Descrierea Setului de Date

### 2.1 Sursa datelor

* **Origine:** Date originale, achiziționate personal (Contribuție 100% la dataset).
* **Modul de achiziție:** [x] Senzori reali (Webcam + MediaPipe) / [ ] Simulare / [ ] Fișier extern / [ ] Generare programatică.
* **Perioada / condițiile colectării:** Noiembrie 2025 - Decembrie 2025, condiții de iluminare ambientală normală, fundal variabil.

### 2.2 Caracteristicile dataset-ului

* **Număr total de observații:** 9001
* **Număr de caracteristici (features):** 63 (21 puncte cheie x 3 coordonate x,y,z)
* **Tipuri de date:** [x] Numerice (float64) / [ ] Categoriale / [ ] Temporale / [ ] Imagini
* **Format fișiere:** [x] CSV / [ ] TXT / [ ] JSON / [ ] PNG / [ ] Altele: [...]

### 2.3 Descrierea fiecărei caracteristici

| **Caracteristică** | **Tip** | **Unitate** | **Descriere** | **Domeniu valori** |
| :---               | :---           | :---        | :---                           | :---                   |
| `label`            | categorial     | -           | Clasa gestului (Target)        | {0, 1, 2, 3}           |
| `lm_0_x`           | numeric        | -           | Coord. X încheietură (Wrist)   | -1.0 ... 1.0 (aprox)   |
| `lm_0_y`           | numeric        | -           | Coord. Y încheietură (Wrist)   | -1.0 ... 1.0 (aprox)   |
| ...                | ...            | ...         | ...                            | ...                    |
| `lm_20_z`          | numeric        | -           | Coord. Z vârf deget mic        | -1.0 ... 1.0 (aprox)   |
**Fișier recomandat:**  `data/README.md`

---

##  3. Analiza Exploratorie a Datelor (EDA) – Sintetic

### 3.1 Statistici descriptive aplicate

S-a analizat distribuția claselor pentru a asigura un dataset echilibrat. În urma achiziției inițiale și a completării ulterioare, situația este următoarea:

- **Distribuția pe clase (Total 9,000):**
  - **0 (STOP)**: 2,334 observații **(~26%)**
  - **1 (INAINTE)**: 1,878 observații **(~21%)**
  - **2 (STANGA)**: 1,956 observații **(~22%)**
  - **3 (DREAPTA)**: 2,832 observații **(~31%)**

### 3.2 Analiza calității datelor

* **Detectarea valorilor lipsă:** 0% valori lipsă (MediaPipe returnează landmark-urile complete doar dacă detectează mâna; frame-urile invalide sunt ignorate la captură)
* **Detectarea valorilor inconsistente sau eronate:** Nu s-au identificat valori în afara domeniului normalizat al camerei.
* **Identificarea caracteristicilor redundante:** Deși există o corelație spațială între puncte, toate cele 63 de features sunt păstrate pentru a capta geometria completă a mâinii.

### 3.3 Probleme identificate

* Inițial a existat un dezechilibru între clasa "STOP" (672) și "DREAPTA" (2832).
* **Soluție:** S-au colectat date suplimentare (~1700 frame-uri) specifice pentru clasele STOP și INAINTE, ajungând la un echilibru satisfăcător pentru antrenare.

---

##  4. Preprocesarea Datelor

### 4.1 Curățarea datelor

* **Eliminare duplicatelor**: Nu a fost necesară; variațiile mici între frame-uri succesive ajută la generalizarea modelului.
* **Tratarea valorilor lipsă:** Nu există.
* **Tratarea outlierilor:** Nu a fost necesară intervenția manuală.

### 4.2 Transformarea caracteristicilor

- **Normalizare:** S-a aplicat StandardScaler (Scikit-Learn).
  - Scaler-ul a fost antrenat (`fit`) exclusiv pe setul de train.
  - Același scaler a fost folosit pentru transformarea seturilor `validation` și `test`.
 **Encoding pentru variabile categoriale**: Etichetele (0-3) sunt deja numerice, potrivite pentru `Loss function (CrossEntropy)`.

### 4.3 Structurarea seturilor de date

**Împărțire aplicata:**
* 70% – train (~6300 samples)
* 15% – validation (~1350 samples)
* 15% – test (~1350 samples)

**Principii respectate:**
* Stratificare pentru clasificare: S-a folosit `stratify=y` pentru a menține proporția claselor în toate subseturile.
* Fără scurgere de informație: Scalarea s-a calculat doar pe train.

### 4.4 Salvarea rezultatelor preprocesării

* Date preprocesate salvate în: `data/train/`, `data/validation/`, `data/test/`.
* Seturi train/val/test în foldere dedicate
* Parametrii de scalare salvați în: `config/preprocessing_params.pkl` (pentru a fi folosiți în aplicația live).

---

##  5. Fișiere Generate în Această Etapă

* `data/generated/` – date brute complete (9000 linii).
* `data/train/X_train.csv`, `y_train.csv` – date curățate & normalizate pentru antrenare.
* `data/validation/X_train.csv`, `y_train.csv` – seturi validare.
* `data/test/X_train.csv`, `y_train.csv` – seturi testare.
* `src/preprocessing/proces_data.py` – codul de preprocesare
* `src/data_acquisition/capture_data.py` – codul de achiziție.
* `data/README.md` – descrierea dataset-ului

---

##  6. Stare Etapă 

- [x] Structură repository configurată
- [x] Dataset analizat (EDA realizată)
- [x] Date preprocesate
- [x] Seturi train/val/test generate
- [x] Documentație actualizată în README + `data/README.md`

---
