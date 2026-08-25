# AI-Based Fake Identity & Document Screening System

Stage 1 prototype: document upload → image quality check → OCR field extraction.

## What's implemented so far

- **`src/preprocessing.py`** — image loading, blur/brightness/glare quality checks, CLAHE preview.
- **`src/ocr.py`** — OCR text extraction (Tesseract) + structured field parsing:
  - If the document has an **MRZ** (passports, most visas), fields are parsed from the
    standardized ICAO 9303 TD3 machine-readable zone — this is fast and reliable.
  - Otherwise, a **regex/keyword fallback** looks for labeled fields ("Name:", "DOB:", etc.)
    in the OCR text — used for national IDs, licenses, permits.
- **`app.py`** — Streamlit UI wiring both stages together.

## Local setup

1. Install the Tesseract OCR engine (system binary, not just the Python wrapper):
   - **Windows:** download installer from https://github.com/UB-Mannheim/tesseract/wiki
   - **Mac:** `brew install tesseract`
   - **Linux/Debian/Ubuntu:** `sudo apt-get install tesseract-ocr`

2. Create a virtual environment and install Python deps:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

4. On Windows, if Tesseract isn't on your PATH, add this near the top of `src/ocr.py`:
   ```python
   pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
   ```

## Deploying (Streamlit Community Cloud)

`packages.txt` is already included — Streamlit Cloud reads it and installs `tesseract-ocr`
automatically as a system package, in addition to `requirements.txt` for Python deps.

## Project structure

```
.
├── app.py
├── requirements.txt
├── packages.txt
└── src/
    ├── preprocessing.py
    └── ocr.py
```

## Next stages (roadmap)

- Module 2: Document validation (field-format/rule checks against document standards)
- Module 3: Tampering detection (photo replacement, text manipulation, stamp forgery, metadata analysis)
- Module 4: Face verification (match document photo to live capture)
- Risk scoring layer combining all module outputs
