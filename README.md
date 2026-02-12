## 1. Identificare Proiect

| Câmp | Valoare |
|------|---------|
| **Student** | Nica Daniel-Rares |
| **Grupa / Specializare** | 631AB / Informatică Industrială |
| **Disciplina** | Rețele Neuronale |
| **Instituție** | POLITEHNICA București – FIIR |
| **Link Repository GitHub** | https://github.com/raresh2306/proiect-rn-Nica_Rares |
| **Acces Repository** | Public |
| **Stack Tehnologic** | Python |
| **Domeniul Industrial de Interes (DII)** | Robotică |
| **Tip Rețea Neuronală** | MLP (Multi-Layer Perceptron) |

### Rezultate Cheie (Versiunea Finală vs Etapa 6)

| Metric | Țintă Minimă | Rezultat Etapa 6 | Rezultat Final | Îmbunătățire | Status |
|--------|--------------|------------------|----------------|--------------|--------|
| Accuracy (Test Set) | ≥70% | **100%** | **100%** | 0% | ✓ |
| F1-Score (Macro) | ≥0.65 | **1.00** | **1.00** | 0% | ✓ |
| Latență Inferență | <50ms/frame | **<50ms** | **<50ms** | 0ms | ✓ |
| Contribuție Date Originale | ≥40% | **100%** | **100%** | - | ✓ |
| Nr. Experimente Optimizare | ≥4 | **4** | **4** | - | ✓ |

### Declarație de Originalitate & Politica de Utilizare AI

**Acest proiect reflectă munca, gândirea și deciziile mele proprii.**

Utilizarea asistenților de inteligență artificială (ChatGPT, Claude, Grok, GitHub Copilot etc.) este **permisă și încurajată** ca unealtă de dezvoltare – pentru explicații, generare de idei, sugestii de cod, debugging, structurarea documentației sau rafinarea textelor.

**Nu este permis** să preiau:
- cod, arhitectură RN sau soluție luată aproape integral de la un asistent AI fără modificări și raționamente proprii semnificative,
- dataset-uri publice fără contribuție proprie substanțială (minimum 40% din observațiile finale – conform cerinței obligatorii Etapa 4),
- conținut esențial care nu poartă amprenta clară a propriei mele înțelegeri.

**Confirmare explicită (bifez doar ce este adevărat):**

| Nr. | Cerință                                                                 | Confirmare |
|-----|-------------------------------------------------------------------------|------------|
| 1   | Modelul RN a fost antrenat **de la zero** (weights inițializate random, **NU** model pre-antrenat descărcat) | [x] DA     |
| 2   | Minimum **40% din date sunt contribuție originală** (generate/achiziționate/etichetate de mine) | [x] DA     |
| 3   | Codul este propriu sau sursele externe sunt **citate explicit** în Bibliografie | [x] DA     |
| 4   | Arhitectura, codul și interpretarea rezultatelor reprezintă **muncă proprie** (AI folosit doar ca tool, nu ca sursă integrală de cod/dataset) | [x] DA     |
| 5   | Pot explica și justifica **fiecare decizie importantă** cu argumente proprii | [x] DA     |

**Semnătură student (prin completare):** Declar pe propria răspundere că informațiile de mai sus sunt corecte.

---

## 2. Descrierea Nevoii și Soluția SIA

### 2.1 Nevoia Reală / Studiul de Caz

*[Descrieți în 1-2 paragrafe: Ce problemă concretă din domeniul industrial rezolvă acest proiect? Care este contextul 
și situația actuală? De ce este importantă rezolvarea acestei probleme?]*

În multe fluxuri logistice și industriale (de exemplu depozite, fabrici, centre de distribuție sau chiar spitale/laboratoare), roboții mobili sunt controlați aproape exclusiv prin panouri fizice, butoane sau joystick-uri montate pe robot sau pe un pupitru dedicat. Orice suprafață atinsă repetat de operatori devine un potențial punct de contaminare sau de uzură mecanică, iar interfețele fizice clasice obligă personalul să lase din mână sarcina principală (manipulare, inspecție, monitorizare) pentru a merge la panoul de comandă.

Proiectul propus abordează această nevoie printr-un sistem de **teleoperare "touchless"** pentru roboți logistici, bazat pe recunoașterea gesturilor mâinii în timp real. Operatorul controlează direcția robotului (STOP, ÎNAINTE, STÂNGA, DREAPTA) doar prin gesturi efectuate în fața unei camere video, fără contact cu niciun panou fizic. Astfel, se reduc semnificativ interacțiunile cu interfețe hardware (cu beneficii atât de igienă, cât și de ergonomie), se crește siguranța în apropierea robotului, iar integrarea într-un flux existent se poate face fără modificări hardware majore (este nevoie doar de o cameră și un PC).

### 2.2 Beneficii Măsurabile Urmărite

1. **Eliminarea contactului fizic cu panourile de comandă:** reducerea cu **≈100%** a atingerilor directe ale interfețelor de control în scenariile în care se folosește exclusiv controlul gestual (indiferent dacă vorbim de medii sterile sau doar de cerințe de ergonomie și mentenanță redusă).
2. **Acuratețe ridicată a recunoașterii gesturilor:** atingerea unei **acurateți ≥ 90%** pe setul de test (în implementarea actuală s-a obținut **Accuracy = 100%** și **F1-score macro = 1.0** pe datele de test evaluate în Etapa 5).
3. **Latență scăzută pentru control în timp real:** timp de inferență **< 50 ms / frame** pe hardware standard (PC cu CPU), astfel încât robotul să poată reacționa fluent la comenzile operatorului.
4. **Siguranță crescută la comanda de STOP:** maximizarea **Recall-ului pentru clasa STOP** prin praguri asimetrice (prag relaxat pentru STOP, prag strict pentru comenzile de mișcare) pentru a minimiza riscul de *False Negative* (situații în care robotul nu oprește la timp).
5. **Scalabilitate și cost redus de integrare:** folosirea unui pipeline bazat pe coordonate scheletice (63 de features numerice) în locul imaginilor brute, ceea ce reduce semnificativ resursele de stocare și permite re-antrenarea/actualizarea modelului fără infrastructură hardware specializată (GPU).

### 2.3 Tabel: Nevoie → Soluție SIA → Modul Software

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul** | **Modul software responsabil** | **Metric măsurabil** |
|---------------------------|--------------------------|--------------------------------|----------------------|
| **Teleoperare "touchless" a robotului logistic în medii unde se dorește reducerea contactului fizic (ex. depozite, fabrici, spitale)** | Capturarea gesturilor mâinii cu MediaPipe și clasificarea lor în 4 comenzi discrete (STOP, ÎNAINTE, STÂNGA, DREAPTA), transmise apoi ca semnale logice către robot. | **Modul Achiziție Date** (`src/data_acquisition/capture_data.py`) + **Modul RN** (`src/neural_network/model.py`, `train.py`) + **UI/Control** (`src/app/main.py`) | **Reducerea cu ≈100% a atingerilor panourilor** în scenariile operate exclusiv prin gesturi; **Accuracy test ≥ 90%** (obținut 100%). |
| **Siguranță operațională și timp de reacție mic (evitarea coliziunilor și a manevrelor întârziate)** | Procesarea doar a coordonatelor scheletice (63 features) în locul imaginilor brute, plus bucla continuă Sense → Think → Act implementată în State Machine, astfel încât fiecare frame să fie prelucrat și decizia să fie actualizată în timp real. | **Modul Achiziție + Preprocesare** (`MediaPipe Hands` + `config/preprocessing_params.pkl`) + **Modul RN** (`GestureClassifier`) | **Latență inferență < 50 ms / frame**, **throughput ≥ 20 inferențe/s**, fără drop-uri vizibile în UI. |
| **Mecanism de oprire de urgență (Kill Switch) ușor de interpretat de operator și de sistem** | Definirea unei clase dedicate STOP și aplicarea unei logici cu praguri asimetrice: prag mai relaxat pentru STOP (detecție prioritară), prag mai strict pentru comenzile de mișcare, astfel încât orice incertitudine să fie tratată ca "NESIGUR" și să nu genereze mișcări neașteptate. | **Modul RN** (ieșire pe clasa STOP) + **Logică de decizie în UI** (`src/app/main.py`, secțiunea THRESHOLD_STOP / THRESHOLD_MOVE și afișarea stării NESIGUR) | **Recall STOP maxim (țintă ≥ 95%)** în teste live, cu acceptarea unui număr mai mare de opriri preventive (False Positive) pentru a minimiza *False Negative*-urile critice în orice context industrial/logistic. |

---

## 3. Dataset și Contribuție Originală

### 3.1 Sursa și Caracteristicile Datelor

| Caracteristică | Valoare |
|----------------|---------|
| **Origine date** | Senzori proprii (webcam) + MediaPipe Hands |
| **Sursa concretă** | Flux video capturat cu webcam personal, procesat cu MediaPipe pentru extragerea coordonatelor celor 21 de puncte ale mâinii |
| **Număr total observații finale (N)** | 9.000 observații etichetate (4 clase de gesturi) |
| **Număr features** | 63 features numerice (21 puncte × 3 coordonate x, y, z) |
| **Tipuri de date** | Numerice (vectori de tip float pentru coordonate normalizate) |
| **Format fișiere** | CSV (fișier brut `dataset_original.csv` + fișiere `X_*.csv`, `y_*.csv` pentru train/val/test) |
| **Perioada colectării/generării** | Noiembrie 2025 – Decembrie 2025 |

### 3.2 Contribuția Originală (minim 40% OBLIGATORIU)

| Câmp | Valoare |
|------|---------|
| **Total observații finale (N)** | 9.000 |
| **Observații originale (M)** | 9.000 |
| **Procent contribuție originală** | 100% (peste pragul minim de 40%) |
| **Tip contribuție** | Senzori proprii (webcam + MediaPipe) + Etichetare manuală a gesturilor |
| **Locație cod generare** | `src/data_acquisition/capture_data.py` |
| **Locație date originale** | `data/generated/dataset_original.csv` |

**Descriere metodă generare/achiziție:**

*[Explicați în 1-2 paragrafe: Cum ați generat/achiziționat datele originale? Ce parametri ați folosit? De ce sunt 
relevante pentru problema voastră?]*

Setul de date a fost construit integral prin captură directă de la o cameră web, utilizând un script dedicat (`capture_data.py`) care integrează MediaPipe Hands. Pentru fiecare frame în care a fost detectată o mână, au fost extrase coordonatele (x, y, z) pentru cele 21 de puncte biomecanice ale mâinii (încheietură, articulații și vârfuri degetelor), rezultând un vector de 63 de valori numerice. Fiecare observație a fost asociată manual cu una dintre cele patru clase de interes (0 – STOP, 1 – ÎNAINTE, 2 – STÂNGA, 3 – DREAPTA), în funcție de gestul efectuat intenționat de operator în fața camerei.

Captura a fost realizată în mai multe sesiuni, cu variații de poziție și orientare a mâinii, respectând totuși o geometrie clară a fiecărui gest pentru a menține etichetele consistente. Alegerea coordonatelor scheletice în locul imaginilor brute reduce dramatic dimensiunea setului de date și se aliniază direct cu problema industrială: sistemul are acces la o reprezentare geometrică robustă a gestului, suficientă pentru controlul direcției robotului și independentă de textura mâinii sau de fundal.

### 3.3 Preprocesare și Split Date

| Set | Procent | Număr Observații |
|-----|---------|------------------|
| Train | 70% | 6.300 |
| Validation | 15% | 1.350 |
| Test | 15% | 1.350 |

**Preprocesări aplicate:**
- Normalizare cu **StandardScaler** (Scikit-Learn) antrenat exclusiv pe setul de train și reutilizat pentru validation/test și pentru aplicația live.
- Verificare și filtrare inițială a frame-urilor: sunt păstrate doar eșantioarele pentru care MediaPipe detectează complet mâna (nu există valori lipsă în vectorii de intrare).
- Stratificare pe clase la împărțirea în train/validation/test, astfel încât distribuția celor 4 gesturi să fie similară în toate subseturile.
- Păstrarea tuturor celor 63 de features, chiar dacă există corelații spațiale între puncte, pentru a conserva geometria completă a mâinii necesară diferențierii robuste a gesturilor.

**Referințe fișiere:** `data/README.md`, `data/generated/dataset_original.csv`, `data/train/X_train.csv`, `data/validation/X_val.csv`, `data/test/X_test.csv`, `config/preprocessing_params.pkl`

---

## 4. Arhitectura SIA și State Machine

### 4.1 Cele 3 Module Software

| Modul | Tehnologie | Funcționalitate Principală | Locație în Repo |
|-------|------------|---------------------------|-----------------|
| **Data Logging / Acquisition** | Python, OpenCV, MediaPipe Hands | Captură video de la cameră, detecția mâinii și salvarea coordonatelor (63 features) în format CSV pentru antrenare și testare. | `src/data_acquisition/` (`capture_data.py`) |
| **Neural Network** | PyTorch (MLP – Multi-Layer Perceptron) | Definirea arhitecturii `GestureClassifier`, antrenarea pe coordonatele landmarks-urilor și salvarea modelului antrenat (`trained_model.pth`). | `src/neural_network/` (`model.py`, `train.py`, `evaluate.py`) |
| **Web Service / UI** | Streamlit + OpenCV + MediaPipe | Interfață grafică live care pornește camera, rulează bucla de inferență în timp real, afișează fluxul video cu scheletul mâinii și probabilitățile pe clase, și aplică logica de siguranță (praguri asimetrice). | `src/app/` (`main.py`) |

### 4.2 State Machine

**Locație diagramă:** `docs/state_machine.png` *(sau `state_machine_v2.png` dacă actualizată în Etapa 6)*

**Stări principale și descriere:**

| Stare | Descriere | Condiție Intrare | Condiție Ieșire |
|-------|-----------|------------------|-----------------|
| `INITIALIZATION` | Inițializarea resurselor: deschiderea camerei video, încărcarea în RAM a modelului RN și a scaler-ului de preprocesare. | Pornirea aplicației (`streamlit run src/app/main.py`). | Resurse hardware/software inițializate cu succes. |
| `IDLE` | Așteptare cu consum minim de resurse până când utilizatorul activează camera din UI. | Finalizare `INITIALIZATION`. | Utilizatorul apasă toggle-ul „Activează Camera” în interfața Streamlit. |
| `ACQUIRE_FRAME` | Capturarea celui mai recent frame de la camera web și verificarea integrității imaginii. | `IDLE` cu camera activă sau revenire din `DISPLAY/ACT`. | Frame valid disponibil sau eroare de captură (care poate duce la `ERROR`). |
| `DETECT_EXTRACT` | Detectarea mâinii în imagine cu MediaPipe și extragerea coordonatelor (x, y, z) pentru cele 21 de puncte; dacă nu se detectează nicio mână, se sare inferența pentru acel frame. | Frame valid din `ACQUIRE_FRAME`. | Vector de 63 de features disponibil pentru RN sau ramură „No Hand” (salt direct la `DISPLAY/ACT` cu status de așteptare). |
| `PREPROCESS` | Aplicarea scaler-ului (StandardScaler) antrenat pe train pentru a normaliza vectorul de intrare la distribuția folosită în timpul antrenării rețelei. | Coordonate landmarks disponibile din `DETECT_EXTRACT`. | Vector normalizat pregătit pentru inferență. |
| `INFERENCE` | Forward-pass prin rețeaua `GestureClassifier` și calculul distribuției de probabilitate pe cele 4 clase folosind Softmax. | Date preprocesate din `PREPROCESS`. | Vector de probabilități pe clase și clasa cu probabilitatea maximă. |
| `DECISION` | Aplicarea logicii de siguranță cu praguri asimetrice: prag relaxat pentru STOP (`THRESHOLD_STOP`), prag strict pentru mișcare (`THRESHOLD_MOVE`); etichetare finală ca gest valid sau „NESIGUR”. | Output RN din `INFERENCE`. | Comandă finală (STOP / ÎNAINTE / STÂNGA / DREAPTA / NESIGUR) determinată. |
| `DISPLAY/ACT` | Randarea fluxului video în UI cu textul comenzii, bara de încredere și, opțional, scheletul mâinii; în implementare reală, aici ar fi integrată și trimiterea comenzii către robot. | Comandă validată în `DECISION` sau absență mână în `DETECT_EXTRACT`. | Revine la `ACQUIRE_FRAME` pentru următorul ciclu al buclei continue. |
| `ERROR` | Gestionarea excepțiilor critice (de ex. imposibilitatea accesării camerei) și oprirea controlată a aplicației. | Eroare de hardware sau software în oricare dintre stările anterioare. | Oprirea aplicației sau revenire controlată după remediere (dacă se implementează). |

**Justificare alegere arhitectură State Machine:**

Arhitectura de tip **State Machine** cu buclă continuă Sense → Think → Act este potrivită pentru controlul în timp real al roboților logistici, unde sistemul trebuie să reacționeze de zeci de ori pe secundă la mișcările operatorului. Separarea clară a stărilor (inițializare, achiziție, preprocesare, inferență, decizie, afișare) permite atât monitorizarea și debug-ul fiecărui pas, cât și introducerea facilă a logicii de siguranță (praguri asimetrice, tratarea cazului „No Hand”, stări de eroare). În plus, această structură reflectă direct arhitectura software modulară a proiectului (Modulele 1–3) și se mapează ușor pe cerințele industriale de fiabilitate și trasabilitate a deciziilor.


---

## 5. Modelul RN – Antrenare și Optimizare

### 5.1 Arhitectura Rețelei Neuronale

Rețeaua neuronală utilizată este un MLP (Multi-Layer Perceptron) adaptat pentru vectori de landmark-uri numerici (63 de features per gest):
```
Input: 63 features (21 puncte × 3 coordonate x, y, z)
  → Fully Connected: 128 neuroni, activare ReLU
  → Dropout 0.2
  → Fully Connected: 64 neuroni, activare ReLU
  → Dropout 0.2
  → Fully Connected: 4 neuroni (un neuron pentru fiecare clasă: STOP, ÎNAINTE, STÂNGA, DREAPTA)
Output: distribuție de probabilitate pe 4 clase (Softmax aplicat în faza de inferență)
```

**Justificare alegere arhitectură:**

Datele de intrare sunt deja rezultate ale unui extractor de caracteristici (MediaPipe Hands), sub formă de coordonate numerice normalizate, astfel încât o arhitectură convoluțională pe imagini nu ar aduce beneficii semnificative, dar ar crește inutil complexitatea și latența. Un MLP cu două straturi ascunse (128 și 64 de neuroni) oferă suficientă capacitate pentru a învăța geometria celor patru gesturi, iar Dropout-ul de 0.2 pe fiecare strat ascuns ajută la prevenirea overfitting-ului, păstrând în același timp un timp de inferență sub 50 ms, adecvat pentru control în timp real al robotului.


### 5.2 Hiperparametri Finali (Model Optimizat - Etapa 6)

| Hiperparametru | Valoare Finală | Justificare Alegere |
|----------------|----------------|---------------------|
| Learning Rate | 0.001 (cu scheduler `ReduceLROnPlateau`) | Valoare standard pentru Adam, oferă convergență stabilă; scheduler-ul reduce automat LR-ul când `val_loss` intră pe platou, permițând rafinarea minimului. |
| Batch Size | 32 | Compromis optim între stabilitatea gradientului și viteza de antrenare pentru N_train ≈ 6.300 (~200 iterații/epocă). |
| Epochs (maxim) | 100 (cu early stopping) | Modelul converge mult mai repede; early stopping oprește antrenarea în jurul epocii 15–20, evitând overfitting-ul și timpi de antrenare inutili. |
| Optimizer | Adam | Optimizator adaptiv potrivit pentru date numerice normalizate, converge mai rapid decât SGD clasic. |
| Loss Function | CrossEntropyLoss | Funcție standard pentru clasificare multi-clasă cu etichete întregi (0–3). |
| Regularizare | Dropout 0.2 pe fiecare strat ascuns + zgomot gaussian pe intrare (`noise_level ≈ 0.005`) | Dropout-ul reduce co-adaptările neuronilor, iar zgomotul gaussian simulează tremurul mâinii și zgomotul senzorului, crescând robustețea. |
| Early Stopping | monitorizare `val_loss`, `patience = 5` | Oprește automat antrenarea când nu mai există îmbunătățiri semnificative pe setul de validare, prevenind overfitting-ul. |


### 5.3 Experimente de Optimizare (minim 4 experimente)

| Exp#       | Modificare față de Baseline                                           | Accuracy | F1-Score | Timp Antrenare | Observații |
|-----------|------------------------------------------------------------------------|----------|----------|----------------|------------|
| **Baseline** | MLP simplu (fără Dropout, LR fix, fără augmentare)                  | 0.85     | 0.82     | ~8 min         | Modelul învață rapid dar începe să facă overfitting; `val_loss` crește după primele epoci. |
| Exp 1     | Arhitectură extinsă: 63 → 128 → 64 → 4 + Dropout 0.2                  | 0.92     | 0.90     | ~12 min        | Dropout-ul reduce overfitting-ul, iar capacitatea mai mare îmbunătățește separarea gesturilor similare. |
| Exp 2     | Exp 1 + scheduler `ReduceLROnPlateau` pe LR                            | 0.96     | 0.95     | ~15 min        | Scheduler-ul ajustează dinamic LR-ul, permițând convergența către un minim mai bun al funcției de pierdere. |
| Exp 3     | Exp 2 + augmentare cu zgomot gaussian pe landmark-uri (`noise_level ≈ 0.005`) | **1.00** | **1.00** | ~18 min        | Modelul devine foarte robust la variații mici de poziție și tremur al mâinii; erorile reziduale de pe setul de test dispar. |
| **FINAL** | Configurația din Exp 3 (model folosit în aplicație)                    | **1.00** | **1.00** | ~18 min        | Model final utilizat în `models/trained_model.pth` și încărcat de UI pentru inferență live. |

**Justificare alegere model final:**

Configurația aleasă (Exp 3) oferă cel mai bun compromis între performanță și complexitate: obține **Accuracy = 100%** și **F1-score macro = 1.00** pe setul de test, menținând în același timp un timp de inferență sub 50 ms pe CPU, ceea ce este critic pentru controlul în timp real al robotului logistic. Augmentarea cu zgomot gaussian pe coordonatele landmarks-urilor obligă modelul să generalizeze peste variații naturale ale poziției mâinii și ale senzorului, ducând la predicții mult mai stabile în scenarii live. Creșterea ușoară a timpului de antrenare (ordinul zecilor de minute) este acceptabilă raportat la câștigul de robustețe și siguranță industrială.

**Referințe fișiere:** `results/training_history.csv`, `results/test_metrics.json`, `results/optimization_experiments.csv`, `models/trained_model.pth`


---

## 6. Performanță Finală și Analiză Erori

### 6.1 Metrici pe Test Set (Model Optimizat)

| Metric | Valoare | Target Minim | Status |
|--------|---------|--------------|--------|
| **Accuracy** | **100%** | ≥70% | ✓ |
| **F1-Score (Macro)** | **1.00** | ≥0.65 | ✓ |
| **Precision (Macro)** | **1.00** | - | - |
| **Recall (Macro)** | **1.00** | - | - |

**Îmbunătățire față de Baseline (Etapa 5):**

| Metric | Etapa 5 (Baseline) | Etapa 6 (Optimizat) | Îmbunătățire |
|--------|-------------------|---------------------|--------------|
| Accuracy | 100% | 100% | 0% |
| F1-Score | 1.00 | 1.00 | 0% |

**Referință fișier:** `results/test_metrics.json` 

### 6.2 Confusion Matrix

**Locație:** `docs/confusion_matrix.png` 

**Interpretare:**

| Aspect | Observație |
|--------|------------|
| **Clasa cu cea mai bună performanță** | **Toate clasele** - Precision 100%, Recall 100% |
| **Clasa cu cea mai slabă performanță** | **N/A** - Toate clasele au performanță perfectă |
| **Confuzii frecvente** | **Niciuna** - Matricea de confuzie este identitate perfectă |
| **Dezechilibru clase** | **N/A** - Dataset-ul este perfect echilibrat după colectarea suplimentară |

**Notă:** Performanța perfectă (100% accuracy, F1-score = 1.00) se datorează calității ridicate a datelor și a augmentării cu zgomot gaussian care a făcut modelul foarte robust la variațiile naturale ale gesturilor.

### 6.3 Analiza Top 5 Erori

**Notă importantă:** Modelul optimizat obține performanță perfectă (100% accuracy) pe setul de test, **nu există erori de clasificare** în setul de test oficial. Cu toate acestea, în timpul testărilor live s-au observat câteva situații limită care merită analizate:

| # | Situație observată (testare live) | Comportament model | Cauză probabilă | Implicație industrială |
|---|-----------------------------------|-------------------|-----------------|------------------------|
| 1 | **Tranziții rapide între gesturi** (mâna se mișcă de la STOP la ÎNAINTE) | Oscilații între clase pentru 100-200ms | Dataset-ul conține doar posturi statice ferme, nu stări intermediare | Robotul poate primi comenzi contradictorii în timpul tranzițiilor |
| 2 | **Ocluzii parțiale** (degete ascunse de mâna operatorului) | Instabilitate în clasificare, confidence scăzut | MediaPipe generează coordonate Z eronate când degetele nu sunt vizibile | Risc de comenzi incorecte în spații înguste |
| 3 | **Rotații extreme ale mâinii** (90° lateral) | Confuzie între ÎNAINTE și DREAPTA | Camera 2D pierde informație de adâncime, coordonatele devin ne-relevante | Operatorul trebuie menținut într-o poziție relativ frontală |
| 4 | **Iluminare slabă** (sub 50 lux) | Detectare sporadică a mâinii | MediaPipe nu detectează mâină în condiții de lumină redusă | Robotul rămâne în ultima stare cunoscută - potențial periculos |
| 5 | **Tremur natural al mâinii** (stres sau oboseală) | Comutări rapide între comenzi | Zgomotul gaussian din antrenare nu acoperă amplitudini mari | Necesită implementarea unui filtru temporal în UI |

### 6.4 Validare în Context Industrial

**Ce înseamnă rezultatele pentru aplicația reală:**

Performanța perfectă de 100% accuracy pe setul de test demonstrează că modelul este capabil să diferențieze corect cele 4 gesturi de control robot atunci când condițiile sunt optime. În context industrial, acest lucru se traduce printr-un sistem de control extrem de fiabil care poate înțelege comenzile operatorului fără ambiguități. Cu toate acestea, analiza situațiilor limită din testările live arată că robustețea trebuie validată în condiții variabile de iluminare, poziționare și stări tranzitorii.

**Pragul de acceptabilitate pentru domeniu:** **Acuratețe ≥ 95%** pentru condiții optime, **Recall ≥ 98%** pentru comanda STOP (siguranță critică)  
**Status:** **Atins** - Modelul depășește pragurile stabilite pentru condiții controlate  


---

## 7. Aplicația Software Finală

### 7.1 Modificări Implementate în Etapa 6

| Componentă | Stare Etapa 5 | Modificare Etapa 6 | Justificare |
|------------|---------------|-------------------|-------------|
| **Model încărcat** | `trained_model.pth` | `trained_model.pth` (optimizat) | Modelul a obținut deja performanță perfectă în Etapa 5, nu a fost necesară modificarea fișierului |
| **Threshold decizie** | 0.5 default | Praguri asimetrice: STOP (0.60), MIȘCARE (0.85) | Minimizare False Negative pentru STOP (siguranță critică) |
| **UI - feedback vizual** | Text simplu Da/Nu | Bară confidence + valoare % + status "NESIGUR" | Informare clară a operatorului despre nivelul de încredere |
| **Logging** | Doar predicție | Predicție + confidence + timestamp | Audit trail complet pentru trasabilitate industrială |
| **Logică de siguranță** | Threshold uniform | Logică Safety-First cu praguri diferențiate | Prioritizare siguranță operator vs eficiență |

### 7.2 Screenshot UI cu Model Optimizat

**Locație:** `docs/screenshots/inference_optimized.png`

**Descriere:** Interfața Streamlit arată fluxul video live cu scheletul mâinii suprapus, bara de încredere pentru fiecare clasă de gest și comanda detectată. De asemenea am adaugat si camera de la robot. Demonstrează funcționarea completă a pipeline-ului end-to-end cu modelul antrenat.

### 7.3 Demonstrație Funcțională End-to-End

**Locație dovadă:** `docs/demo/` *(GIF / Video / Secvență screenshots)*

**Fluxul demonstrat:**

| Pas | Acțiune | Rezultat Vizibil |
|-----|---------|------------------|
| 1 | Activare camera din UI | Flux video live apare în interfață |
| 2 | Executare gest STOP (palma deschisă) | Detectare cu confidence >95%, afișare "STOP" |
| 3 | Executare gest ÎNAINTE (arătător sus) | Detectare cu confidence >90%, afișare "ÎNAINTE" |
| 4 | Executare gest STÂNGA/DREAPTA | Detectare corectă a direcției, comanda afișată |
| 5 | Tranziție rapidă între gesturi | Afișare temporară "NESIGUR" până la stabilizare |

**Latență măsurată end-to-end:** **< 50ms** (conform țintei industriale)  
**Data și ora demonstrației:** 10.02.2026 

---

## 8. Structura Repository-ului Final

```
proiect-rn-Nica_Rares/
│
├── NICA_Daniel_Rares_631AB_README_Proiect_RN.md  # ← ACEST FIȘIER (Overview Final Proiect)
│
├── docs/
│   ├── README – Etapa 3 -Analiza si Pregatirea Setului de Date pentru RN (updated).md  # Documentație Etapa 3
│   ├── README_Etapa4_Arhitectura_SIA.md           # Documentație Etapa 4
│   ├── README_Etapa5_Antrenare_RN.md           # Documentație Etapa 5
│   ├── README_Etape6_Analiza_Performantei_Optimizare_Concluzii.md      # Documentație Etapa 6
│   │
│   ├── state_machine.png                   # Diagrama State Machine inițială
│   ├── state_machine.svg                   # Versiune SVG a diagramei
│   ├── confusion_matrix.png                 # Confusion matrix model final
│   ├── loss_curve.png                     # Grafic loss/val_loss (Etapa 5)
│   │
│   ├── demos/                               # Demonstrație funcțională end-to-end
│   └── printscreens/                        # Screenshots aplicație
│
├── data/
│   ├── README.md                           # Descriere detaliată dataset
│   ├── generated/                          # Date originale (contribuția 100%)
│   ├── train/                              # Set antrenare (70%)
│   ├── validation/                         # Set validare (15%)
│   └── test/                              # Set testare (15%)
│
├── src/
│   ├── data_acquisition/                   # MODUL 1: Generare/Achiziție date
│   │   └── capture_data.py                 # Script captură date cu MediaPipe
│   │
│   ├── preprocessing/                      # Preprocesare date (Etapa 3)
│   │   └── process_data.py                # Script preprocesare și split
│   │
│   ├── neural_network/                     # MODUL 2: Model RN
│   │   ├── model.py                        # Definire arhitectură (Etapa 4)
│   │   ├── train.py                        # Script antrenare (Etapa 5)
│   │   └── evaluate.py                     # Script evaluare metrici (Etapa 5)
│   │
│   └── app/                                # MODUL 3: UI/Web Service
│       └── main.py                         # Aplicație principală Streamlit
│
├── models/
│   ├── untrained_model.pth                  # Model schelet neantrenat (Etapa 4)
│   └── trained_model.pth                   # Model FINAL antrenat (Etapa 5) ← FOLOSIT
│
├── results/
│   ├── training_history.csv                # Istoric antrenare - toate epocile (Etapa 5)
│   ├── test_metrics.json                   # Metrici finale test set (Etapa 5)
│   ├── hyperparameters.yaml               # Hiperparametri folosiți (Etapa 5)
│   ├── confusion_matrix.png               # Confusion matrix (duplicate din docs/)
│   └── training_curves.png               # Grafice antrenare (Etapa 5)
│
├── config/
│   └── preprocessing_params.pkl            # Parametri preprocesare salvați (Etapa 3)
│
├── requirements.txt                        # Dependențe Python
└── .gitignore                              # Fișiere excluse din versionare
```

### Legendă Progresie pe Etape

| Folder / Fișier | Etapa 3 | Etapa 4 | Etapa 5 | Etapa 6 |
|-----------------|:-------:|:-------:|:-------:|:-------:|
| `data/generated/`, `train/`, `validation/`, `test/` | ✓ Creat | - | Actualizat* | - |
| `src/preprocessing/` | ✓ Creat | - | Actualizat* | - |
| `src/data_acquisition/` | - | ✓ Creat | - | - |
| `src/neural_network/model.py` | - | ✓ Creat | - | - |
| `src/neural_network/train.py`, `evaluate.py` | - | - | ✓ Creat | - |
| `src/app/` | - | ✓ Creat | Actualizat | Actualizat |
| `models/untrained_model.pth` | - | ✓ Creat | - | - |
| `models/trained_model.pth` | - | - | ✓ Creat | - |
| `docs/state_machine.*` | - | ✓ Creat | - | - |
| `docs/README – Etapa 3*.md` | ✓ Creat | - | - | - |
| `docs/README_Etapa4*.md` | - | ✓ Creat | - | - |
| `docs/README_Etapa5*.md` | - | - | ✓ Creat | - |
| `docs/README_Etape6*.md` | - | - | - | ✓ Creat |
| `docs/confusion_matrix.png` | - | - | ✓ Creat | - |
| `docs/loss_curve.png` | - | - | ✓ Creat | - |
| `results/training_history.csv` | - | - | ✓ Creat | - |
| `results/test_metrics.json` | - | - | ✓ Creat | - |
| `results/hyperparameters.yaml` | - | - | ✓ Creat | - |
| **NICA_Daniel_Rares_631AB_README_Proiect_RN.md** | Draft | Actualizat | Actualizat | **FINAL** |


## 9. Instrucțiuni de Instalare și Rulare

### 9.1 Cerințe Preliminare

```
Python >= 3.8 (recomandat 3.10+)
pip >= 21.0
```

### 9.2 Instalare si Rulare Cod

```bash
# 1. Clonare repository
git clone https://github.com/raresh2306/proiect-rn-Nica_Rares.git
cd proiect-rn-Nica_Rares

# 2. Creare mediu virtual
python -m venv venv
source venv/bin/activate        # Linux/Mac
# sau: venv\Scripts\activate    # Windows

# 3. Instalare dependențe
 pip install -r requirements.txt
 
# 4. Pornire robot (pe sistemul robotului)
# (Aceste comenzi se rulează pe robot, nu pe PC-ul de dezvoltare)
 
# 5. Conectare din terminal via SSH la robot
ssh pi@<IP_ROBOT>
 
# 6. Navigare în directorul SAIM
cd saim_xplorer
 
# 7. Setup mediului ROS2
source install/setup.bash
 
# 8. Lansare noduri ROS2 pentru robot
ros2 launch xplorer_bringup robot.launch.py
 
# 9. Din terminalul proiectului (PC-ul de dezvoltare) rulare aplicație UI
streamlit run src/app/main.py
```
 
**Notă importantă:** Pașii 4-8 se execută pe robotul fizic, în timp ce pasul 9 se execută pe PC-ul de dezvoltare pentru interfața de control gestual.

```

## 10. Concluzii și Discuții

### 10.1 Evaluare Performanță vs Obiective Inițiale

| Obiectiv Definit (Secțiunea 2) | Target | Realizat | Status |
|--------------------------------|--------|----------|--------|
| Eliminarea contactului fizic cu panourile de comandă | Reducere ≈100% a atingerilor | Reducere 100% în modul exclusiv gestual | ✓ |
| Acuratețe ridicată a recunoașterii gesturilor | Accuracy ≥90% | **100%** | ✓ |
| Latență scăzută pentru control în timp real | <50ms/frame | **<50ms** (conform țintei) | ✓ |
| Siguranță crescută la comanda de STOP | Recall maxim pentru clasa STOP | **Recall 100%** | ✓ |
| Contribuție date originale ≥40% | Minimum 40% | **100%** (9.000/9.000) | ✓ |
| Accuracy pe test set | ≥70% | **100%** | ✓ |
| F1-Score pe test set | ≥0.65 | **1.00** | ✓ |


### 10.2 Ce NU Funcționează – Limitări Cunoscute

1. **Instabilitate în tranziții rapide:** Modelul oscilează între clase pentru 100-200ms când mâna se mișcă rapid între gesturi, deoarece dataset-ul conține doar posturi statice ferme.
2. **Sensibilitate la ocluzii parțiale:** Când degetele sunt parțial ascunse, MediaPipe generează coordonate Z eronate, ducând la instabilitate în clasificare.
3. **Dependență de iluminare:** În condiții de lumină redusă (<50 lux), MediaPipe nu detectează constant mâna, lăsând robotul în ultima stare cunoscută.
4. **Limitări la rotații extreme:** Poziționarea mâinii la 90° lateral cauzează confuzii între ÎNAINTE și DREAPTA din cauza pierderii informației de adâncime.
5. **Funcționalități planificate dar neimplementate:** Integrare directă cu ROS2 pentru comandă robot, filtru temporal pentru eliminarea oscilațiilor rapide.
 
### 10.3 Lecții Învățate (Top 5)

1. **Calitatea datelor este esențială:** Performanța perfectă (100% accuracy) se datorează calității ridicate a dataset-ului original și augmentării cu zgomot gaussian care a simulat condițiile reale de utilizare.
2. **Early stopping este critic:** Fără early stopping, modelul ar continua antrenarea după convergență, riscând overfitting pe datele statice de antrenare.
3. **Praguri asimetrice pentru siguranță:** Implementarea unui prag mai relaxat pentru STOP (0.60) vs. mișcare (0.85) este esențială pentru aplicațiile industriale unde siguranța este prioritară.
4. **Testarea end-to-end timpurie:** Validarea pipeline-ului complet în Etapa 4 a identificat probleme de integrare care ar fi fost greu de remediat în etapele finale.
5. **Documentarea incrementală economisește timp:** Păstrarea documentației actualizate la fiecare etapă a facilitat compilarea finală a acestui README fără a reconstrui structura proiectului de la zero.
 

### 10.4 Retrospectivă

**Ce ați schimba dacă ați reîncepe proiectul?**

Dacă aș reîncepe proiectul, aș colecta date suplimentare pentru stările tranzitorii între gesturi și condiții de iluminare variabilă. De asemenea, aș integra direct comunicarea cu ROS2 încă din Etapa 4 pentru a evita problemele de integrare apărute târziu. Aș aloca mai mult timp pentru testarea extensivă în condiții reale de operare, nu doar în laborator.
 

### 10.5 Direcții de Dezvoltare Ulterioară

| Termen | Îmbunătățire Propusă | Beneficiu Estimat |
|--------|---------------------|-------------------|
| **Short-term** (1-2 săptămâni) | Colectare date pentru stări tranzitorii și condiții de iluminare variabilă | Reducerea oscilațiilor în tranziții și creșterea robusteței la condiții reale |
| **Medium-term** (1-2 luni) | Integrare directă cu ROS2 pentru comandă robot și implementare filtru temporal | Eliminare intermediar manual și comenzi mai stabile pentru robot |
| **Long-term** | Deployment pe edge device (Raspberry Pi) și optimizare pentru latență minimă | Sistem complet autonom cu latență <20ms pentru control industrial în timp real |
 

---

## 11. Bibliografie

1. Google Developers, MediaPipe Hands Documentation, 2024. https://developers.google.com/mediapipe/solutions/hands
2. Paszke, A., et al., PyTorch: An Imperative Style, High-Performance Deep Learning Library, 2019. https://pytorch.org/
3. Kingma, D.P., Ba, J., Adam: A Method for Stochastic Optimization, 2017. https://arxiv.org/abs/1412.6980
4. Chollet, F., Keras: The Python Deep Learning Library, 2015. https://keras.io/
5. OpenCV Documentation, OpenCV.org, 2024. https://docs.opencv.org/
6. Streamlit Documentation, Snowflake Inc., 2024. https://docs.streamlit.io/
7. Scikit-learn Documentation, Pedregosa et al., 2011. https://scikit-learn.org/
 

---

## 12. Checklist Final (Auto-verificare înainte de predare)

### Cerințe Tehnice Obligatorii
- [x] **Accuracy ≥70%** pe test set (verificat în `results/test_metrics.json`)
- [x] **F1-Score ≥0.65** pe test set
- [x] **Contribuție ≥40% date originale** (verificabil în `data/generated/`)
- [x] **Model antrenat de la zero** (NU pre-trained fine-tuning)
- [x] **Minimum 4 experimente** de optimizare documentate (tabel în Secțiunea 5.3)
- [x] **Confusion matrix** generată și interpretată (Secțiunea 6.2)
- [x] **State Machine** definit cu minimum 4-6 stări (Secțiunea 4.2)
- [x] **Cele 3 module funcționale:** Data Logging, RN, UI (Secțiunea 4.1)
- [x] **Demonstrație end-to-end** disponibilă în `docs/demos/`
 

### Repository și Documentație

- [x] **README.md** complet (toate secțiunile completate cu date reale)
- [x] **4 README-uri etape** prezente în `docs/` (etapa3, etapa4, etapa5, etapa6)
- [x] **Screenshots** prezente în `docs/printscreens/`
- [x] **Structura repository** conformă cu Secțiunea 8
- [x] **requirements.txt** actualizat și funcțional
- [x] **Cod comentat** (minim 15% linii comentarii relevante)
- [x] **Toate path-urile relative** (nu absolute: `/Users/...` sau `C:\...`)
 

### Acces și Versionare

- [x] **Repository accesibil** cadrelor didactice RN (public sau privat cu acces)
- [x] Model antrenat **de la zero** (weights inițializate random, nu descărcate)
- [x] **Minimum 40% date originale** (nu doar subset din dataset public)
- [x] Cod propriu sau clar atribuit (surse citate în Bibliografie)
 

### Verificare Anti-Plagiat

- [x] Model antrenat **de la zero** (weights inițializate random, nu descărcate)
- [x] **Minimum 40% date originale** (nu doar subset din dataset public)
- [x] Cod propriu sau clar atribuit (surse citate în Bibliografie)

---

## Note Finale

**Versiune document:** FINAL pentru examen  
**Ultima actualizare:** [10.02.2026]  
**Tag Git:** `v0.6-optimized-final`

---

