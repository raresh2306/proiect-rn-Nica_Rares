# 🤖 Sistem de Control Robot prin Gesturi (SIA)

**Instituție:** POLITEHNICA București – FIIR  
**Disciplina:** Rețele Neuronale  
**Student:** Nica Daniel-Rares  

---

## 📖 Descrierea Proiectului

Acest proiect implementează un sistem de **teleoperare aseptică ("touchless")** pentru roboți logistici, utilizând recunoașterea gesturilor mâinii bazată pe Rețele Neuronale Artificiale.

Sistemul este proiectat pentru medii în care contactul fizic cu panourile de control este interzis sau periculos (ex: spitale, laboratoare sterile), permițând operatorului să controleze direcția robotului strict prin mișcarea mâinii în fața unei camere video.

### Obiective Principale
1. **Teleoperare Aseptică:** Eliminarea contactului fizic.
2. **Latență Scăzută:** Procesare rapidă (< 50ms) a coordonatelor scheletice.
3. **Siguranță (Kill Switch):** Detectarea prioritară a gestului de STOP.

---

## 🛠️ Arhitectura Sistemului

Proiectul este structurat pe o arhitectură de tip **Event-Driven Continuous Loop** (Sense → Think → Act), compusă din trei module principale:

### 1. Modulul de Achiziție (Sense)
- **Tehnologie:** MediaPipe Hands  
- **Funcție:** Extrage în timp real coordonatele spațiale (x, y, z) pentru 21 de puncte biomecanice ale mâinii  
- **Output:** Vector de 63 de caracteristici numerice

### 2. Modulul de Rețea Neuronală (Think)
- **Model:** Multi-Layer Perceptron (MLP)
- **Arhitectură:**
  - Input Layer: 63 neuroni
  - Hidden Layer 1: 128 neuroni (ReLU + Dropout 0.2)
  - Hidden Layer 2: 64 neuroni (ReLU + Dropout 0.2)
  - Output Layer: 4 neuroni (clasele de gesturi)
- **Framework:** PyTorch

### 3. Interfața Utilizator (Act)
- **Tehnologie:** Streamlit
- **Funcție:** Feedback vizual în timp real (video + schelet mână + probabilități)

---

## 📊 Setul de Date

Dataset-ul a fost construit integral și conține **9.000 de observații** balansate manual.

| Label | Gest     | Descriere                          | Nr. Exemple |
|------:|----------|------------------------------------|-------------|
| 0     | STOP     | Palma deschisă (urgență)            | 2,334       |
| 1     | INAINTE  | Arătătorul sus                     | 1,878       |
| 2     | STANGA   | Arătător stânga                    | 1,956       |
| 3     | DREAPTA  | Arătător dreapta                   | 2,832       |

**Sursă:** Date capturate cu webcam propriu (100% contribuție personală).

---

## 📂 Structura Proiectului

```text
proiect-rn-control-robot/
├── data/
│   ├── generated/
│   └── train/test/val/
├── src/
│   ├── app/
│   ├── data_acquisition/
│   ├── neural_network/
│   └── preprocessing/
├── models/
├── docs/
└── requirements.txt
```

---

## 🚀 Ghid de Utilizare

### 1. Instalare Dependențe
```bash
pip install -r requirements.txt
```

### 2. Rulare Aplicație
```bash
streamlit run src/app/main.py
```

### 3. Colectare Date Noi (Opțional)
```bash
python src/data_acquisition/capture_data.py
```

---

## 📈 Performanță

Datele sunt normalizate cu **StandardScaler** (media 0, deviația 1) antrenat pe setul de training, asigurând convergență rapidă și stabilitate.

---

**Proiect realizat în cadrul laboratorului de Rețele Neuronale, 2025–2026**
