# Veridex — AI-Based Fake Identity & Document Screening System

Veridex is a prototype document-screening system designed to assist border-security personnel in screening passports, national IDs, visas, and other identity/travel documents for possible forgery, tampering, and inconsistencies.

The system is deliberately structured as a **pipeline rather than a single AI model**. It combines OCR, document/country routing, rule-based validation, image-quality analysis, AI-based field detection/forgery analysis, optional face verification, and a weighted risk engine.

> **Important:** Veridex is a prototype screening/decision-support system. Its risk score is a heuristic screening score, not a calibrated probability that a document is forged.

---

## 1. Problem Statement

Manual inspection of identity and travel documents can be difficult under high passenger volume. Common threats include:

- Altered names, dates, document numbers, or personal details
- Photograph replacement
- Modified or forged visa stamps/regions
- Identity impersonation
- Expired or invalid documents
- Multiple identities using related or altered documents
- Documents whose visual appearance has been digitally manipulated

A practical automated screening system should therefore do more than OCR text. It should combine **what the document says**, **whether the values are structurally valid**, **whether the image shows suspicious manipulation**, and, where available, **whether the document photograph matches the presented person**.

---

## 2. What Veridex Currently Does

The current prototype can perform the following stages:

1. **Image-quality analysis** — checks blur/sharpness, brightness, and glare.
2. **OCR extraction** — extracts visible text and confidence/location information.
3. **Document/type and country routing** — determines the likely document/country and selects a specialized model where available.
4. **Document validation** — checks required fields, formats, dates, and MRZ structure/check digits for supported passports.
5. **Tampering/forgery analysis** — uses AI model outputs and localized document-region information.
6. **Risk aggregation** — combines model and preprocessing signals.
7. **Decision** — produces a screening status such as CLEAR, REFER, or REJECT.
8. **Face verification** — present in the architecture as an optional/planned module.

The application is implemented as a Streamlit prototype with Python modules for preprocessing, OCR, classification/routing, validation, tampering, and risk aggregation.

---

## 3. High-Level Architecture

```text
                     ┌──────────────────────┐
                     │    Document Image    │
                     └──────────┬───────────┘
                                ↓
                     ┌──────────────────────┐
                     │ Image Quality Checks │
                     │ blur / brightness /  │
                     │ glare / preprocessing│
                     └──────────┬───────────┘
                                ↓
                     ┌──────────────────────┐
                     │ OCR + MRZ Extraction  │
                     └──────────┬───────────┘
                                ↓
                     ┌──────────────────────┐
                     │ Document / Country    │
                     │ Detection & Routing   │
                     └──────────┬───────────┘
                                ↓
          ┌─────────────────────┼──────────────────────┐
          ↓                     ↓                      ↓
   SIDTD grouped YOLO     MobileNetV3 forgery     Visa YOLO
   field/region model        classifier         Japan/USA/Korea
          └─────────────────────┼──────────────────────┘
                                ↓
                     ┌──────────────────────┐
                     │ Field Validation /   │
                     │ MRZ Checks / Rules   │
                     └──────────┬───────────┘
                                ↓
                     ┌──────────────────────┐
                     │ Optional Face Match │
                     └──────────┬───────────┘
                                ↓
                     ┌──────────────────────┐
                     │   Risk Aggregator    │
                     │ model-dominant +    │
                     │ preprocessing signal│
                     └──────────┬───────────┘
                                ↓
                 ┌──────────────┴──────────────┐
                 ↓                             ↓
             CLEAR / REFER                 REJECT
```

---

## 4. OCR Module

OCR is used to convert document pixels into machine-readable information.

The OCR stage provides:

- Extracted text
- OCR confidence information
- Approximate text locations/bounding boxes
- Structured fields where they can be recognized
- MRZ text for supported passports

OCR is important because later validation and country/document routing can use the extracted text. OCR itself is **not treated as proof that a document is genuine**: a forged document can contain perfectly readable text.

### Why OCR alone is insufficient

A document can contain valid-looking text while the pixels containing that text have been edited. Therefore Veridex combines OCR with visual analysis and validation.

---

## 5. Image-Quality / Preprocessing Module

The preprocessing stage checks whether the image itself is suitable for reliable downstream analysis.

Current checks include:

- **Blur/sharpness** — heavily blurred images can reduce OCR and visual-model reliability.
- **Brightness** — extremely dark or bright images can reduce useful information.
- **Glare** — reflections can hide or distort document regions.

These are supporting signals rather than primary forgery evidence.

### Why these checks matter

Poor image quality can look suspicious without actually being a forgery. For example, glare can cause OCR failures even when the physical document is genuine. Therefore quality information is better treated as a supporting risk factor instead of an automatic forged decision.

---

## 6. Document Validation

Validation uses extracted fields and document-specific rules.

Examples include:

- Required-field checks
- Date-format checks
- Identifier/document-number format checks
- Date plausibility
- MRZ structure validation
- MRZ check-digit validation
- Consistency between OCR-extracted data and MRZ data where supported

### MRZ validation

For supported passport MRZs, Veridex uses ICAO-style structure/check-digit logic. This can detect certain inconsistencies such as invalid check digits or malformed MRZ data.

A valid MRZ does **not** prove a document is genuine. It only shows that the encoded information is structurally consistent with the expected format.

---

# 7. AI Components

Veridex does **not** have one single "main AI model." The AI layer consists of multiple models serving different purposes.

## 7.1 Grouped YOLO Field Detector

A grouped YOLO model is trained on SIDTD-based document data.

Its seven field classes are:

```text
name
birth_date
expiry_date
number
photo
signature
other
```

The detector produces, for each accepted detection:

- class
- bounding box
- confidence

For example:

```text
class = name
confidence = 0.7138
box = [x1, y1, x2, y2]
```

The confidence is the model's **detection confidence for that class/box**, not a calibrated probability of forgery.

This distinction is important when interpreting the risk engine.

### Why YOLO is useful

A whole-document classifier can say:

```text
document → forged
```

but does not necessarily explain *where* the suspicious evidence is.

A detector can instead provide a localized region such as:

```text
name → [bounding box]
photo → [bounding box]
number → [bounding box]
```

which helps produce an explainable screening result and can provide regions for later tampering analysis.

---

## 7.2 MobileNetV3-Small Forgery Classifier

A MobileNetV3-Small binary classifier is used for a selected national-ID/passport subset.

Its two classes are:

```text
bona_fide
forged
```

The checkpoint currently used for this experiment is:

```text
models/national_id_azj_slv_grc/sidtd_3country_mobilenetv3_small.pth
```

The checkpoint metadata identifies:

- Architecture: MobileNetV3-Small
- Image size: 320 × 320
- Classes: bona_fide / forged
- Country subset: Azerbaijan, Greece, Slovakia

Unlike the grouped YOLO detector, this model directly performs **whole-image binary forgery classification**.

### Limitation

A binary classifier can identify a document/region as more consistent with forged or bona-fide examples, but it does not inherently provide the precise manipulated location unless a separate localization method is used.

---

## 7.3 Region-Level Forgery Classifiers

The project also experimented with ResNet18 transfer-learning models for suspicious document regions.

Regions investigated include:

- name
- date of birth
- personal/document number
- photo

Two input approaches were explored:

- RGB image regions
- ELA (Error Level Analysis) representations

The purpose of these experiments was to investigate whether localized visual/compression information makes forgery classification easier than whole-document classification.

These experiments are research/prototype components and should not automatically be treated as the production decision model.

---

## 7.4 Specialized Visa YOLO Model

A separate specialized YOLO model is used for selected visas.

Current routing targets include visas associated with:

- Japan
- USA
- South Korea

The reason for specialization is that document layouts vary significantly between countries and document types. A detector trained for one family of layouts may not generalize equally well to another.

---

# 8. Model Routing

The application uses country/document information to select the most appropriate model.

The current routing design is approximately:

```text
Azerbaijan / Greece / Slovakia
        → MobileNetV3 forgery classifier

Japan / USA / South Korea + VISA
        → specialized visa YOLO

Albania / Estonia / Finland / Latvia /
Russia / Serbia / Spain
        → grouped SIDTD YOLO
```

Country detection can use OCR text and MRZ country codes where available, with filename/template information used as a fallback where appropriate.

### Why route models?

A single model is convenient but may lose accuracy when document layouts change significantly. Routing allows each model to specialize in a smaller, more consistent document distribution.

The trade-off is increased system complexity and the need to maintain multiple models.

---

# 9. Risk Scoring

The system combines multiple evidence sources into a single screening score.

The current design is **model-dominant**, with model output contributing approximately 80% and preprocessing/image-quality signals contributing approximately 20% to the final risk signal.

Conceptually:

```text
Final Risk
    = 0.80 × Model Risk
    + 0.20 × Preprocessing Risk
```

The result is then compared against decision thresholds.

Current prototype decision bands are:

```text
risk < 25       → CLEAR
25 ≤ risk < 65  → REFER
risk ≥ 65       → REJECT
```

Hard validation failures can also contribute direct reject conditions depending on the module.

### Important interpretation

The final score is a **screening/risk heuristic**, not a statistical probability.

For example, a YOLO output such as:

```text
name confidence = 0.7138
```

means the detector is confident about its `name` detection. It should not be described in a presentation as:

> "71.38% probability that the document is forged."

The risk engine may use that model output as one evidence signal, but the meaning of the underlying confidence remains a detector confidence.

### Why use weighted scoring?

No single signal should automatically decide the result.

For example:

```text
Model signal → high
Image quality → good
Validation → passes
```

may produce a REFER/CLEAR result instead of an immediate REJECT, depending on the exact score.

This reduces the chance that one noisy signal creates an unnecessarily severe decision.

---

# 10. Current Decision Philosophy

The intended behavior is:

### CLEAR

The available evidence does not currently indicate enough risk to require escalation.

### REFER

There is enough suspicious evidence or uncertainty that a human operator should manually inspect the document.

### REJECT

The combined evidence or a hard validation condition crosses the rejection criteria.

The system is designed as **decision support**, so borderline or uncertain documents should be routed for human review rather than treated as automatically fraudulent.

---

# 11. SIDTD Dataset

The project uses the **SIDTD** dataset for document-field annotations and tampering experiments.

The SIDTD data available in this project contains country/document subsets including:

```text
ALB  Albania
AZE  Azerbaijan
ESP  Spain
EST  Estonia
FIN  Finland
GRC  Greece
LVA  Latvia
RUS  Russia
SRB  Serbia
SVK  Slovakia
```

### Dataset structure

The project contains genuine/source documents and multiple fake/tampered variants derived from some of those sources.

A simplified example is:

```text
alb_id_01.jpg
alb_id_01_fake_6_56.jpg
alb_id_01_fake_6_61.jpg
```

The source document may therefore have several manipulated variants.

### Why source-level splitting matters

A naive random image split can leak the same underlying document into both training and validation:

```text
TRAIN:
alb_id_01_fake_6_56

VAL:
alb_id_01_fake_6_61
```

The model would then be evaluated on a document it effectively saw during training.

For a more honest evaluation, all genuine/fake variants derived from the same source document should stay in the same split.

---

# 12. YOLO Dataset and Annotation Format

The grouped YOLO training data uses field-level bounding boxes with the seven classes:

```text
name
birth_date
expiry_date
number
photo
signature
other
```

A YOLO label line has the normalized format:

```text
<class_id> <x_center> <y_center> <width> <height>
```

For example:

```text
0 0.41 0.32 0.18 0.06
```

means that class `0` occupies a normalized bounding box centered at `(0.41, 0.32)` with normalized width `0.18` and height `0.06`.

The existing SIDTD annotation information provides field regions for conversion into YOLO-style boxes.

Fake metadata can identify the manipulated field, while the corresponding document annotation provides the field geometry needed to construct the bounding box.

---

# 13. Current 3-Country Training Plan

The grouped-country training plan is:

```text
Group 1 → ALB + AZE + ESP
Group 2 → EST + FIN + GRC
Group 3 → LVA + RUS + SRB
Group 4 → SVK
```

The purpose is to reduce the diversity each specialist model must handle and test whether country-group specialization improves field localization/forgery-related performance.

For Group 1, the current dataset inventory is:

```text
ALB → 222 images / 122 YOLO-labelled variants
AZE → 226 images / 126 YOLO-labelled variants
ESP → 222 images / 122 YOLO-labelled variants
-----------------------------------------------
Total → 670 images / 370 labelled variants
```

The remaining 300 images in this group are currently unlabeled in the YOLO directory and are believed to correspond to genuine/base documents. Before training, this should be verified rather than assumed blindly.

---

# 14. Existing Grouped YOLO Training Result

An earlier grouped SIDTD YOLO11n model was trained with the seven field classes.

Known validation metrics from that run were:

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| All classes | 0.753 | 0.554 | 0.674 | 0.479 |
| name | 0.913 | 0.930 | 0.953 | 0.644 |
| birth_date | 0.616 | 0.583 | 0.590 | 0.496 |
| expiry_date | 0.989 | 0.842 | 0.932 | 0.698 |
| number | 0.851 | 0.694 | 0.840 | 0.607 |
| photo | 1.000 | 0.000 | 0.497 | 0.322 |
| signature | 0.000 | 0.000 | 0.000 | 0.000 |
| other | 0.903 | 0.827 | 0.904 | 0.582 |

Training details recorded for that run include:

- Model: YOLO11n
- Image size: 640 × 640
- Epochs: 15
- Device: CPU
- Approximate training time: 1.337 hours
- Training images: approximately 977
- Validation images: approximately 245

### Interpretation

The aggregate metrics indicate that the model can learn useful field localization, but performance is uneven between classes.

For example:

- `name`, `expiry_date`, `number`, and `other` show useful detection performance.
- `birth_date` is moderate.
- `photo` has high precision but zero recall in the recorded validation result, which means it did not reliably recover all annotated photo instances.
- `signature` performed poorly in this run.

Therefore the aggregate mAP should not be interpreted as equally strong performance for every document field.

---

# 15. Forgery-Classification Experiments

Several transfer-learning experiments were performed to understand the difficulty of forgery classification.

## Whole-document ResNet18 — Imbalanced Dataset

Training distribution was approximately:

```text
genuine → 24
tampered → 96
```

Recorded result:

```text
accuracy  = 80.0%
precision = 80.0%
recall    = 100.0%
f1         = 88.9%
```

However, the confusion matrix was:

```text
[[0, 7],
 [0, 28]]
```

This means the model predicted every evaluated sample as tampered. Therefore the 80% accuracy was misleading because of class imbalance.

### Lesson

Accuracy alone is dangerous on imbalanced forgery datasets. Precision, recall, F1, confusion matrices, and class distributions must be examined together.

---

## Whole-document ResNet18 — Balanced Dataset

Recorded result:

```text
accuracy  = 57.1%
precision = 60.0%
recall    = 42.9%
f1         = 50.0%
```

Confusion matrix:

```text
[[5, 2],
 [4, 3]]
```

Balancing removed the extreme class shortcut but also showed that whole-document forgery classification is difficult when manipulations are small.

---

## Region ResNet18

An early region-based experiment evaluated localized regions such as name, DOB, personal number, and photo.

A later small test reported approximately:

```text
accuracy  = 71.4%
precision = 63.6%
recall    = 100.0%
f1         = 77.8%
```

Because this evaluation set was small, it should be treated as an experiment rather than a reliable generalization estimate.

---

## RGB vs ELA Region Experiment

A later comparison produced:

| Input | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| RGB | 0.887 | 0.444 | 0.800 | 0.571 |
| ELA | 0.919 | 0.562 | 0.600 | 0.581 |

ELA achieved slightly better accuracy/F1 in that experiment, but the dataset was still limited. These results are not sufficient to claim broad real-world superiority of ELA.

---

# 16. Important Evaluation Lesson: Validation vs Real Documents

A model can perform well on its SIDTD validation set and still behave differently on independent genuine documents.

Reasons include:

- Different document templates or printing layouts
- Different image capture conditions
- Different resolution/compression
- Distribution shift between benchmark data and real images
- Limited genuine hard negatives during training
- Model learning dataset-specific visual artifacts

For this reason, the project should evaluate at least three categories separately:

```text
1. Genuine documents
2. SIDTD tampered documents
3. Independently generated/scripted tampered documents
```

The scripted tampered set should ideally remain completely outside training and be used as an independent test set.

---

# 17. Current False-Positive Observation

Some genuine Albanian IDs were tested with the grouped YOLO model.

Ten example images had identical dimensions:

```text
2167 × 1360
```

and were converted to RGB before the standalone YOLO test.

Observed detections were:

```text
00 → name 0.7138
01 → name 0.3513
02 → name 0.2558
03 → name 0.4397
04 → name 0.4039
05 → no detections
06 → no detections
07 → no detections
08 → no detections
09 → no detections
```

Repeated inference on `00.jpg` produced the same `0.7138` result five times with the same bounding box, showing that this particular output is reproducible rather than random inference noise.

This observation does **not** establish that the model is detecting 71% forgery. It shows that the field detector is confidently producing a `name` detection on that genuine document.

The final weighted risk decision can still remain below the REFER threshold because the risk engine combines multiple signals.

### Engineering takeaway

The system should distinguish between:

```text
YOLO detection confidence
```

and:

```text
calibrated forgery probability
```

before presenting the number as a probability to an operator.

---

# 18. Synthetic / Scripted Tampering Evaluation

The project also considers independently scripted document manipulations, such as changing text, dates, numbers, or replacing document regions.

These are useful because they test whether the model generalizes beyond the exact tampering patterns represented in SIDTD.

A strong experimental design is:

```text
TRAIN:
SIDTD genuine + SIDTD tampered

TEST:
A. unseen SIDTD documents
B. genuine documents from outside training
C. independently scripted tampered documents
```

This helps distinguish:

- memorization of source documents
- learning SIDTD-specific artifacts
- genuine generalization to new forgeries

---


# 19. Reproducibility and Model Management

The repository should keep the following information alongside trained models and training scripts:

- Dataset/source used for training
- Country/document group
- Class mapping and `data.yaml`
- Train/validation/test split strategy
- Image size, batch size, epochs, optimizer, learning rate, and augmentation settings
- Final checkpoint used by the application
- Validation metrics and confusion matrices where applicable
- Known limitations or failure cases

For SIDTD experiments, splits should be made at the **source-document level** so genuine documents and all derived fake variants remain in the same split. This avoids leakage between training and validation/test sets.

Model files should be versioned or clearly named so the checkpoint used by `app.py` can be reproduced from the documented training configuration.

# 20. Security and Operational Considerations

Veridex is intended as a screening aid, not an autonomous authority for immigration, identity, or law-enforcement decisions. A production implementation would require extensive validation, threshold calibration, monitoring, auditability, access control, and human oversight.

Potential operational concerns include:

- Sensitive identity documents must be handled securely and retained only as required.
- Uploaded images and OCR output can contain personally identifiable information.
- Logs should avoid unnecessarily storing raw document data or extracted sensitive fields.
- Model failures should be visible to operators rather than silently treated as genuine/forged certainty.
- Country/model routing failures must have a safe fallback instead of silently applying an unrelated specialist model.
- Thresholds should be validated against the cost of false positives and false negatives in the intended deployment environment.

# 21. Explainability and Operator Output

The screening result should expose enough information for an operator to understand why a document was escalated. Useful evidence includes:

- Document/country classification
- OCR and validation findings
- Detected document-field regions and bounding boxes
- Tampering/forgery model outputs
- Image-quality warnings
- Final risk score and decision band

A detector confidence should be labelled as detector confidence. It should not be presented as a probability of forgery unless the score has been calibrated and validated for that interpretation.

# 22. Testing Strategy

Testing should be separated into distinct datasets so that a strong benchmark result is not mistaken for broad generalization:

```text
Training
  └── SIDTD training documents

Validation
  └── unseen SIDTD source documents

Independent test
  ├── genuine documents outside training
  └── independently scripted tampered documents
```

Recommended measurements include precision, recall, F1, confusion matrix, mAP for detection models, false-positive rate on genuine documents, false-negative rate on forged documents, and inference latency.

For the screening decision itself, report the number of CLEAR, REFER, and REJECT outcomes on known genuine and forged test sets. Thresholds should be chosen using validation data and then held fixed for the final test.

# 23. Deployment Notes

The current prototype is designed for local development and demonstration through Streamlit. The model router makes it possible to add specialist checkpoints without replacing the whole application.

A production-oriented deployment would typically separate the UI from inference services, centralize model/version management, add structured logging and monitoring, and use hardware-appropriate inference optimization.

The system should also preserve a clear distinction between:

```text
Detection
→ where/what a model sees

Classification
→ genuine/forged prediction

Validation
→ whether extracted data follows expected rules

Risk aggregation
→ combining evidence for screening
```

This separation makes the pipeline easier to test, debug, explain, and replace module-by-module.

# 24. Limitations

The current prototype has several known limitations:

### Limited country/document coverage

The system supports a defined set of document groups rather than every passport, visa, or national ID worldwide.

### Dataset size and diversity

Some experiments use relatively small datasets. Results should therefore be treated as prototype evidence, not production-grade generalization guarantees.

### Class imbalance

Forgery datasets can be heavily imbalanced. Raw accuracy can therefore be misleading.

### Detection vs forgery calibration

Raw detector confidence is not a calibrated forgery probability. The risk engine currently treats model outputs as weighted evidence rather than statistically calibrated probabilities.

### False positives on genuine documents

Some genuine documents can trigger high-confidence field detections. This motivates stronger genuine hard-negative data and threshold calibration.

### Real-world validation

Benchmark validation does not fully represent photographed/scanned documents encountered in the field. Independent real-world-like testing is required.

### CPU inference speed

The current prototype has been developed and tested on CPU in several experiments. GPU or optimized inference formats can reduce latency for deployment.

### Face verification

Face verification remains optional/planned rather than the strongest validated part of the current prototype.

---

# 25. Future Improvements

The next improvements are prioritized around reliability rather than simply increasing model complexity:

1. Train specialized country groups using source-document-aware train/validation splits.
2. Add more genuine hard negatives so field detectors see diverse legitimate documents.
3. Evaluate independent scripted tampering that was not used during training.
4. Calibrate decision thresholds against genuine and forged validation sets.
5. Separate localization confidence from actual forgery probability in the UI/risk engine.
6. Improve difficult field classes such as signature and photo localization.
7. Use higher-resolution or targeted regional models for small edits.
8. Explore stronger region-level forgery classifiers after reliable field localization.
9. Optimize inference with GPU/ONNX/TensorRT where deployment hardware permits.
10. Add robust logging/audit trails for operator review.
11. Complete and validate face verification.
12. Expand the model registry as more document families are added.

---

# 26. Project Structure

A simplified repository structure is:

```text
.
├── app.py
├── requirements.txt
├── packages.txt
├── dataset/
│   └── SIDTD/
├── models/
│   ├── sidtd_grouped/
│   ├── visa_jp_usa_kr/
│   └── national_id_azj_slv_grc/
├── src/
│   ├── preprocessing.py
│   ├── doc_classifier.py
│   ├── validator.py
│   ├── mrz_validation.py
│   ├── tampering.py
│   └── risk_aggregator.py
└── ...
```

Exact filenames can vary as the prototype evolves; the important separation is between the Streamlit application, source modules, datasets, and trained model checkpoints.

---

# 27. Local Setup

## Install Tesseract

Tesseract OCR must be installed separately from the Python package.

**Windows:** install a Tesseract Windows distribution and ensure the executable is available to the application/configuration.

**macOS:**

```bash
brew install tesseract
```

**Linux/Debian/Ubuntu:**

```bash
sudo apt-get install tesseract-ocr
```

## Create the Python environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### macOS/Linux

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

# 28. Internals Defense Summary

The simplest way to explain Veridex during an internal review is:

> **Veridex is a multi-stage document-screening pipeline. OCR extracts information, validation checks whether the information is structurally consistent, AI models analyze document regions/forgery patterns, and a weighted risk engine combines the evidence into a screening decision.**

The key technical contribution is not merely OCR. It is the attempt to combine **document understanding, localized field detection, forgery analysis, validation, and risk aggregation** while routing different document families to models trained for their specific distributions.

The strongest claims should remain tied to measured validation results. The prototype should not claim universal forgery detection, calibrated probabilities, or production-level accuracy without larger and more independent evaluations.

---

## Development Roadmap

- Improve field localization and region-level forgery detection
- Complete 3-country specialized training experiments
- Improve hard-negative coverage and reduce false positives
- Calibrate the final risk thresholds
- Expand country/document support
- Complete face verification
- Improve independent real-world/scripted tampering evaluation
- Add audit/logging support
- Improve CPU/GPU inference performance
