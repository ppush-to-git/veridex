# AI Fake Identity & Document Screening

Current stage: document ingestion and image-quality assessment.

## Run

```bash
pip install streamlit opencv-python numpy
streamlit run app.py
```

Upload a JPG, JPEG, PNG, or WebP document image.

## Current pipeline

Upload image -> quality checks -> preprocessing preview

Checks currently include:
- resolution
- blur/sharpness
- brightness
- possible glare

## Project structure

```text
fake-id-screening/
├── app.py
├── src/
│   └── preprocessing.py
├── data/
│   ├── genuine/
│   └── tampered/
└── output/
```

## Next

1. Document boundary detection
2. Perspective correction
3. Document type classification
4. OCR + MRZ extraction
5. Rule validation
6. Tamper detection
7. Face verification
8. Explainable risk scoring
