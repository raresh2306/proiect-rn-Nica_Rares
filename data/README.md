# 📂 Descrierea Setului de Date - Gesturi Mână

Acest dataset conține coordonatele scheletului mâinii (Landmarks) extrase folosind MediaPipe Hands, pentru clasificarea a 4 gesturi de control robot.

## 1. Statistici Generale
* **Număr total observații:** 9,000
* **Număr caracteristici (features):** 63 (21 puncte x 3 axe: x, y, z)
* **Tip date:** Numerice (float64)
* **Sursă:** Date originale, achiziționate prin webcam propriu (Contribuție 100%)

## 2. Distribuția Claselor
Dataset-ul este echilibrat manual pentru a evita bias-ul:

| Label | Gest | Nr. Exemple | Descriere |
|-------|------|-------------|-----------|
| **0** | STOP | 2,334 | Palma deschisă |
| **1** | INAINTE | 1,878 | Arătătorul sus |
| **2** | STANGA | 1,956 | Rotire/Arătător stânga |
| **3** | DREAPTA | 2,832 | Rotire/Arătător dreapta |

## 3. Structura Datelor
Fiecare rând din CSV conține:
1. **label**: 0, 1, 2 sau 3 (Target)
2. **lm_0_x, lm_0_y, lm_0_z**: Coordonatele încheieturii
3. ...
4. **lm_20_x, lm_20_y, lm_20_z**: Coordonatele vârfului degetului mic

## 4. Preprocesare Aplicată
* **Split:** 70% Train / 15% Validation / 15% Test (Stratificat)
* **Normalizare:** StandardScaler (media 0, deviația 1) antrenat pe Train set.