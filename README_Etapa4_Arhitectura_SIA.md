# 📘 README – Etapa 4: Arhitectura Completă a Aplicației SIA bazată pe Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Nica Daniel-Rares
**Link Repository GitHub** https://github.com/raresh2306/proiect-rn-Nica_Rares
**Data:** 16.12.2025
---

## Scopul Etapei 4

Această etapă corespunde punctului **5. Dezvoltarea arhitecturii aplicației software bazată pe RN** din lista de 9 etape - slide 2 **RN Specificatii proiect.pdf**.


### IMPORTANT - Ce înseamnă "schelet funcțional":

 **CE TREBUIE SĂ FUNCȚIONEZE:**
- Toate modulele pornesc fără erori
- Pipeline-ul complet rulează end-to-end (de la date → până la output UI)
- Modelul RN este definit și compilat (arhitectura există)
- Web Service/UI primește input și returnează output

 **CE NU E NECESAR ÎN ETAPA 4:**
- Model RN antrenat cu performanță bună
- Hiperparametri optimizați
- Acuratețe mare pe test set
- Web Service/UI cu funcționalități avansate

---

##  Livrabile Obligatorii

## 1. Tabelul Nevoie Reală → Soluție SIA → Modul Software

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul vostru** | **Modul software responsabil** |
|---------------------------|--------------------------------|--------------------------------|
| **Teleoperare aseptică:** Controlul roboților logistici în zone sterile (spitale, laboratoare) unde contactul fizic cu panourile de comandă este interzis. | Implementarea unui sistem de control gestual "touchless" cu recunoașterea a 4 clase de mișcare, având o **acuratețe țintită ≥ 90%** pe setul de test. | **Modul 1 (Achiziție)** + **Modul 2 (RN)** |
| **Siguranță operațională (Low Latency):** Evitarea coliziunilor prin transmiterea comenzilor de direcție instantaneu către robot. | Procesarea exclusivă a coordonatelor scheletice (nu a imaginii brute) pentru a obține o **latență de inferență < 50 ms/frame** pe un procesor standard. | **Modul 1 (MediaPipe)** + **Modul 3 (UI Loop)** |
| **Oprire de urgență (Kill Switch):** Mecanism de siguranță critică pentru oprirea instantanee a robotului în caz de eroare umană. | Detectarea prioritară a gestului "Palma Deschisă" (STOP) cu un prag de **încredere (confidence threshold) setat strict la > 0.80** pentru a elimina alarmele false. | **Modul 2 (RN - Clasa 0)** + **Modul 3 (Logic)** |
| **Feedback vizual pentru operator:** Operatorul are nevoie de confirmarea vizuală că robotul a "înțeles" comanda înainte de execuție. | Interfață grafică (GUI) cu afișare în timp real (>15 FPS) a fluxului video augmentat cu scheletul mâinii și bare de probabilitate pentru fiecare clasă detectată. | **Modul 3 (Streamlit UI)** |


---

### 2. Contribuția Originală la Setul de Date – MINIM 40% din Totalul Observațiilor Finale (Realizat: 100%)


```markdown
### Contribuția originală la setul de date:

**Total observații finale:** **9,000** (Dataset complet Etapa 3 + 4)
**Observații originale:** **9,000** (**100%**)

**Tipul contribuției:**
[ ] Date generate prin simulare fizică  
[X] Date achiziționate cu senzori proprii (webcam + MediaPipe)  
[X] Etichetare/adnotare manuală  
[ ] Date sintetice prin metode avansate  

**Descriere detaliată:**
Întregul set de date a fost construit de la zero ("from scratch") printr-un proces riguros de achiziție, utilizând un modul software dedicat (`src/data_acquisition/capture_data.py`). S-a utilizat senzorul optic (webcam) pentru a captura secvențe video reale, din care s-au extras și stocat coordonatele spațiale (x, y, z) ale celor 21 de puncte biomecanice ale mâinii, folosind pipeline-ul MediaPipe.

**Justificare tehnică:**
Această abordare ("Senzori proprii") este superioară utilizării dataset-urilor generice de imagini (ex: Kaggle) din următoarele motive:
1.  **Robustete la mediu:** Datele conțin micro-mișcări și unghiuri specifice camerei robotului, antrenând rețeaua să fie robustă la "jitter"-ul natural al mâinii umane.
2.  **Eficiență Computațională:** Prin stocarea doar a coordonatelor scheletice (CSV), am redus dimensiunea dataset-ului de la câțiva GB (imagini) la câțiva MB, permițând antrenarea rapidă și re-antrenarea on-the-edge.
3.  **Distribuție controlată:** S-a asigurat balansarea perfectă a claselor critice (STOP, DREAPTA), eliminând bias-ul statistic des întâlnit în seturile publice.

**Locația codului:** `src/data_acquisition/capture_data.py`
**Locația datelor:** `data/generated/dataset_original.csv`

**Dovezi:**
- Fișier CSV brut cu 9.000 de intrări: `data/generated/dataset_original.csv`
- Statistici detaliate (distribuție 2000+ samples/clasă) în `README_Etapa3.md`
- PrintScreen-uri din timpul rularii scriptului `capture_data.py` ce se gasesc in `docs/printscreens`
```

---

## 3. Diagrama State Machine a Întregului Sistem

**Diagrama grafică:** `docs/state_machine.png` , `docs/state_machine.svg` 

### Justificarea State Machine-ului ales:

Sistemul a fost proiectat pe baza arhitecturii **Event-Driven Continuous Loop** (Buclă Continuă bazată pe Evenimente), specifică sistemelor de control în timp real (Real-Time Control Systems).

**De ce acest model?**
Spre deosebire de o aplicație web clasică (Request-Response), un robot teleoperat nu poate "aștepta". El trebuie să parcurgă ciclul **Sense → Think → Act** de zeci de ori pe secundă. Orice blocaj în acest ciclu poate duce la accidente.

**Descrierea Stărilor:**

1.  **INITIALIZATION (Start-up):**
    * Se alocă resursele hardware (Camera).
    * Se încarcă în RAM modelul RN compilat și Scaler-ul (pentru a evita latența la prima inferență).
    * *Tranziție automată spre IDLE.*

2.  **IDLE (Standby):**
    * Stare de așteptare cu consum minim de energie.
    * Așteaptă semnalul de activare ("Toggle Camera ON") de la operator.

3.  **ACQUIRE_FRAME (Sensing):**
    * Capturează cel mai recent buffer video.
    * Verifică integritatea imaginii.

4.  **DETECT & EXTRACT (Pre-processing):**
    * **Decision Node:** Algoritmul MediaPipe scanează cadrul.
    * **Ramura A (No Hand):** Dacă nu se detectează mâna, sistemul sare direct la afișare (ignoră inferența pentru a economisi CPU) și revine la ACQUIRE.
    * **Ramura B (Hand Found):** Se extrag cele 63 de coordonate și se aplică normalizarea (`StandardScaler`) pentru a aduce valorile în intervalul optim rețelei.

5.  **INFERENCE (Thinking):**
    * Rețeaua Neuronală (MLP) procesează vectorul de intrare.
    * Se calculează distribuția de probabilitate (Softmax) pentru cele 4 clase.

6.  **DISPLAY/ACT (Acting):**
    * **Threshold Gate:** Dacă probabilitatea maximă < Pragul de Siguranță (ex: 0.6), comanda este anulată ("Nesigur").
    * Se actualizează interfața grafică (UI) și se transmite comanda logică.
    * *Tranziție automată înapoi la ACQUIRE_FRAME.*

---

---

### 4. Scheletul Complet al celor 3 Module Cerute la Curs (slide 7)


| **Modul** | **Implementare Concretă** | **Status Funcțional (la predare)** |
|-----------|----------------------------|------------------------------------|
| **1. Data Logging / Acquisition** | **Script:** `src/data_acquisition/capture_data.py`<br>**Tehnologii:** OpenCV, MediaPipe Hands, CSV | **[x] FUNCȚIONAL**<br>Produce fișierul `data/generated/dataset_original.csv`.<br>Au fost generate cu succes 9.000 de eșantioane originale (100% din dataset), depășind cerința minimă de 100 samples. |
| **2. Neural Network Module** | **Script:** `src/neural_network/model.py`<br>**Tehnologii:** PyTorch (nn.Module), MLP Architecture | **[x] DEFINIT & COMPILAT**<br>Clasa `GestureClassifier` este definită și compilată fără erori.<br>Modelul poate fi inițializat și salvat în `models/untrained_model.pth`.<br>Arhitectura este justificată în docstring-ul codului. |
| **3. Web Service / UI** | **Script:** `src/app/main.py`<br>**Tehnologii:** Streamlit, OpenCV Integration | **[x] FUNCȚIONAL**<br>Interfața web pornește fără erori (`streamlit run`).<br>Primește input video de la utilizator (Webcam), procesează datele prin pipeline și afișează output-ul (predicția și încrederea) în timp real. |

#### Detalii per modul (Checklist Funcționalități):

#### **Modul 1: Data Logging / Acquisition**

**Funcționalități obligatorii:**
- [x] Cod rulează fără erori: `python src/data_acquisition/capture_data.py`
- [x] Generează CSV în format compatibil cu preprocesarea din Etapa 3 (`data/generated/dataset_original.csv`)
- [x] Include minimum 40% date originale în dataset-ul final (Realizat: 100% - 9.000 samples proprii)
- [x] Documentație în cod: Scriptul conține docstring-uri care explică parametrii de achiziție (MediaPipe Landmarks) și formatul output-ului.

#### **Modul 2: Neural Network Module**

**Funcționalități obligatorii:**
- [x] Arhitectură RN definită și compilată fără erori (`src/neural_network/model.py`)
- [x] Model poate fi salvat și reîncărcat (Implementat logică de save/load `models/untrained_model.pth` în aplicație)
- [x] Include justificare pentru arhitectura aleasă (Docstring detaliat în clasa `GestureClassifier` despre alegerea MLP vs CNN)
- [x] **NU trebuie antrenat** cu performanță bună (Modelul rulează cu weights inițializate random în această etapă)

#### **Modul 3: Web Service / UI**

**Funcționalități MINIME obligatorii:**
- [x] Propunere Interfață ce primește input de la user: Aplicație Streamlit (`src/app/main.py`) cu buton "Activează Camera" și feedback vizual live.
- [x] Includeți un screenshot demonstrativ în `docs/printscreens/ui_demo.png` (Captură interfață Streamlit funcțională).

**Ce NU e necesar în Etapa 4:**
- UI frumos/profesionist cu grafică avansată
- Funcționalități multiple (istorice, comparații, statistici)
- Predicții corecte (modelul e neantrenat, e normal să fie incorect)
- Deployment în cloud sau server de producție

**Scop:** Prima demonstrație că pipeline-ul end-to-end funcționează: input user → preprocess → model → output.




## 4. Structura Repository-ului la Finalul Etapei 4 (OBLIGATORIE)

**Verificare consistență cu Etapa 3:**

```text
proiect-rn-control-robot/
├── data/
│   ├── generated/             # Date originale (dataset_original.csv) - 9000 samples
│   ├── train/                 # X_train.csv, y_train.csv
│   ├── validation/            # X_val.csv, y_val.csv
│   └── test/                  # X_test.csv, y_test.csv
├── src/
│   ├── data_acquisition/      # capture_data.py (Modul 1)
│   ├── preprocessing/         # process_data.py (Din Etapa 3)
│   ├── neural_network/        # model.py (Modul 2)
│   └── app/                   # main.py (Modul 3 - UI Schelet)
├── docs/
│   ├── state_machine.png      # Diagrama logică a sistemului
│   ├── state_machine.svg      # Diagrama logică a sistemului (format scalabil)
│   └── printscreens/
│       └── ui_demo.png        # Screenshot interfață funcțională
├── models/
│   └── untrained_model.pth    # Modelul compilat (neantrenat)
├── config/
│   └── preprocessing_params.pkl # Scaler-ul salvat
├── README.md                  # General
├── README_Etapa3.md           # Documentație Etapa 3
├── README_Etapa4_Arhitectura_SIA.md # ← Acest fișier completat
└── requirements.txt           # Dependențe (torch, streamlit, mediapipe)
```

Diferențe față de Etapa 3 (Implementate):

[x] Adăugat data/generated/ pentru contribuția originală (Conține cele 9.000 de date proprii).
[x] Adăugat src/data_acquisition/ - MODUL 1 (Scriptul de captură funcțional).
[x] Adăugat src/neural_network/ - MODUL 2 (Clasa modelului definită).
[x] Adăugat src/app/ - MODUL 3 (Interfața Streamlit funcțională).
[x] Adăugat models/ pentru model neantrenat (Generat automat la prima rulare).
[x] Adăugat docs/state_machine.png - Diagrama OBLIGATORIE.
[x] Adăugat docs/printscreens/ pentru demonstrație UI (ui_demo.png).

---

## 5. Checklist Final – Bifați Totul Înainte de Predare

### Documentație și Structură
- [x] Tabelul Nevoie → Soluție → Modul complet (Completat în Secțiunea 1 din `README_Etapa4_Arhitectura_SIA.md`)
- [x] Declarație contribuție **100% date originale** completată în `README_Etapa4_Arhitectura_SIA.md` (Secțiunea 2)
- [x] Cod generare/achiziție date funcțional și documentat (`src/data_acquisition/capture_data.py`)
- [x] Dovezi contribuție originală: CSV brut în `data/generated/` + statistici în `README_Etapa3.md`
- [x] Diagrama State Machine creată și salvată în `docs/state_machine.png`
- [x] Legendă State Machine scrisă în `README_Etapa4_Arhitectura_SIA.md` (Secțiunea 3 - Justificare și Descriere Stări)
- [x] Repository structurat conform modelului (Verificat structura folderelor `data`, `src`, `docs`)

### Modul 1: Data Logging / Acquisition
- [x] Cod rulează fără erori: `python src/data_acquisition/capture_data.py`
- [x] Produce **100%** date originale din dataset-ul final (9.000 samples)
- [x] CSV generat în format compatibil cu preprocesarea din Etapa 3 (`dataset_original.csv` cu header corect)
- [x] Documentație tehnică (inclusă în `README_Etapa4...` și Docstrings în cod):
  - [x] Metodă de generare: Extragere Landmarks MediaPipe din flux Webcam
  - [x] Parametri folosiți: 21 puncte x 3 axe, captură Real-Time, 4 clase
  - [x] Justificare: Eliminarea zgomotului de fond și robustețe la geometrie

### Modul 2: Neural Network
- [x] Arhitectură RN definită și documentată în cod (`src/neural_network/model.py`)
- [x] Detalii arhitectură curentă: MLP (Input 63 -> 128 -> 64 -> Output 4) documentat în Docstring-ul clasei `GestureClassifier`

### Modul 3: Web Service / UI
- [x] Propunere Interfață ce pornește fără erori: `python -m streamlit run src/app/main.py`
- [x] Screenshot demonstrativ salvat în `docs/printscreens/ui_demo.png`
- [x] Instrucțiuni de lansare incluse în `README_Etapa4_Arhitectura_SIA.md` (Secțiunea Module)

---


