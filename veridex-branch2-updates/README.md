# AI-Based Fake Identity & Document Screening System

An AI-based document screening platform designed to assist border-security personnel in detecting forged or tampered identity and travel documents.

## Overview

Veridex analyzes identity documents such as passports and national IDs and assists with:

- OCR-based information extraction
- Document validation
- Tampering and forgery detection
- Face verification
- Risk assessment

The prototype currently focuses on a defined set of document templates, with the architecture intended to support additional document types later.

## Current Features

### OCR Extraction

Extracts text from document images using OCR and provides:

- Detected text
- Confidence scores
- Text locations / bounding boxes
- Structured document fields
- MRZ extraction for supported passports

### Image Quality Analysis

Performs basic image-quality checks such as:

- Blur / sharpness
- Brightness
- Glare

These checks are treated as image-quality indicators rather than the main forgery detector.

### Document Validation

Extracted fields can be checked against document-specific formats and rules, such as:

- Required fields
- Date formats
- Identifier formats
- Expiry dates

### Tampering Detection

The main AI component of the system.

The current prototype investigates localized manipulation in regions such as:

- Name
- Date of birth
- Personal/document number
- Photograph

A transfer-learning vision model is being used to classify suspicious document regions.

### Face Verification

Planned module for comparing the photograph on the document with the presented person's face.

### Risk Scoring

Planned layer that combines OCR, validation, tampering, face verification, and image-quality results into an overall risk assessment.

## Project Pipeline

```text
Document Image
      ↓
Image Quality Check
      ↓
Document / Template Selection
      ↓
OCR Extraction
      ↓
Field Validation
      ↓
Tampering Detection
      ↓
Face Verification
      ↓
Risk Assessment
      ↓
Screening Decision
```

## Prototype Scope

The current prototype focuses on a limited number of document templates so that each supported format can be processed and validated reliably.

The main focus of the project is detecting document tampering and providing an explainable screening result.

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── packages.txt
├── dataset/
├── models/
├── src/
└── ...
```

## Local Setup

### 1. Install Tesseract

Tesseract OCR must be installed separately from the Python package.

**Windows:** Download the Tesseract installer from the UB Mannheim distribution.

**macOS:**

```bash
brew install tesseract
```

**Linux/Debian/Ubuntu:**

```bash
sudo apt-get install tesseract-ocr
```

### 2. Install Python Dependencies

```bash
python -m venv venv
```

**Windows:**

```bash
venv\Scripts\activate
```

**macOS/Linux:**

```bash
source venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run app.py
```

## Development Roadmap

- Improve tampering detection and localization
- Complete document validation
- Add face verification
- Integrate all module outputs into a unified risk score
- Add audit/logging support
- Expand support for additional document templates
