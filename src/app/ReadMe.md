# 🖥️ Modul 3: Interfață Utilizator (SIA Robot Control)

Acest modul implementează interfața grafică (Web Service / UI) pentru controlul robotului, folosind biblioteca **Streamlit**.

## 📂 Fișiere
* **`main.py`**: Scriptul principal care gestionează:
  * Fluxul video de la cameră.
  * Procesarea MediaPipe.
  * Inferența cu modelul RN.
  * Afișarea rezultatelor în browser.

## 🚀 Instrucțiuni de Lansare

Asigurați-vă că sunteți în **rădăcina proiectului** (nu în acest folder) și rulați comanda:

```bash
python -m streamlit run src/app/main.py
sau
py -m streamlit run src/app/main.py
```

`⚠️ Note importante`:
La prima rulare, veți fi întrebat de email – puteți lăsa gol și apăsa Enter.
Dacă browserul nu se deschide automat, accesați adresa afișată în terminal (de obicei `http://localhost:8501`).
Pentru a opri aplicația, apăsați `Ctrl + C` în terminal.

`🛠️ Dependențe necesare`
Acestea sunt deja incluse în `requirements.txt` din rădăcină:
**streamlit**, **opencv-python**, **mediapipe**, **torch**