# MindPulse FER
## Project Report: Facial Expression Recognition for Therapy Session Analysis

## 1. Executive Summary

MindPulse FER is a Streamlit-based computer-vision application for observing facial expressions during therapy-session recordings or webcam captures. It combines pretrained ONNX facial-expression models with face detection, lighting correction, temporal smoothing, session analytics, SQLite persistence, and PDF report generation.

The project is intended as a **clinical decision-support and visualization prototype**. It does not diagnose mental-health conditions, replace a licensed clinician, or provide clinically validated measurements by itself.

## 2. Project Objectives

The application is designed to:

- Detect a visible face in a webcam frame or recorded therapy video.
- Estimate the dominant facial-expression category and confidence.
- Estimate continuous valence and arousal values when available from the model.
- Reduce prediction instability caused by lighting and individual video frames.
- Present emotion timelines and session-level statistics.
- Identify possible distress spikes and estimate recovery latency.
- Store patient and session information locally in SQLite.
- Generate a structured SOAP-style summary and multi-page PDF report.

## 3. Technology Stack

| Area | Technology |
|---|---|
| User interface | Streamlit |
| Image and video processing | OpenCV |
| Neural-network inference | ONNX Runtime |
| Expression models | HSEmotion EfficientNet-B0 ONNX models |
| Face detection | OpenCV Haar Cascade |
| Numerical processing | NumPy and Pandas |
| Interactive visualization | Plotly |
| Static charts and PDF charts | Matplotlib and Seaborn |
| PDF generation | ReportLab |
| Persistence | SQLite |
| Language | Python |

## 4. System Architecture

The main execution path is:

```text
Webcam frame / uploaded video
        |
        v
Lighting and facial-quality assessment
        |
        v
CLAHE and gamma enhancement when required
        |
        v
OpenCV face detection
        |
        v
Largest-face selection and padded face crop
        |
        v
224 x 224 RGB ImageNet-normalized tensor
        |
        v
HSEmotion ONNX model inference
        |
        v
Auxiliary-model blending and custom post-processing
        |
        v
Temporal smoothing
        |
        v
Emotion overlay, timeline, analytics, database, and PDF report
```

The main modules are:

- `app.py`: Streamlit interface, navigation, webcam/video workflows, dashboards, and report actions.
- `fer_engine.py`: Image preprocessing, face detection, ONNX inference, smoothing, and visual overlays.
- `analytics.py`: Timeline construction, emotion statistics, ESI, entropy, transitions, distress episodes, and recovery metrics.
- `database.py`: SQLite schema and patient/session persistence.
- `report_generator.py`: SOAP-style narrative generation, charts, and PDF compilation.
- `demo_data.py`: Synthetic benchmark scenarios and sample-video generation.

## 5. Facial-Expression Models

### 5.1 Primary model

The primary model is:

```text
models/hsemotion/enet_b0_8_va_mtl.onnx
```

It is loaded directly with ONNX Runtime. The model output is interpreted as:

- Eight emotion logits.
- A continuous valence output.
- A continuous arousal output, when the output contains at least ten values.

### 5.2 Auxiliary models

Two additional ONNX models may be loaded:

```text
models/hsemotion/enet_b0_8_best_vgaf.onnx
models/hsemotion/enet_b0_8_best_afew.onnx
```

The primary model contributes 65% of the final probability distribution. The available auxiliary models jointly contribute 35%. This is an inference-time ensemble; the model weights are not retrained or modified by this repository.

### 5.3 Emotion classes

The implementation maps model output indices to:

```text
angry, contempt, disgust, fear,
happy, neutral, sad, surprise
```

The application also associates each class with display metadata, including a label, color, valence estimate, arousal estimate, and affect category.

## 6. Processing Pipeline

### 6.1 Frame-quality assessment

Each frame is assessed for:

- Mean luminance.
- Contrast, measured using grayscale standard deviation.
- Sharpness, measured using the variance of the Laplacian.

These values produce a facial-quality score used in the UI and stored with the frame record.

### 6.2 Lighting correction

When a frame is underexposed, overexposed, or has low contrast, the project can apply:

- Gamma correction.
- CLAHE on the luminance channel in LAB color space.

This preprocessing is applied before face detection and expression inference.

### 6.3 Face localization

The project uses the local cascade model:

```text
models/haarcascade_frontalface_default.xml
```

If the local file is unavailable, it falls back to OpenCV's installed Haar-cascade directory. If multiple faces are detected, the largest detected face is treated as the therapy participant.

The selected face region receives proportional padding before it is sent to the expression model.

### 6.4 Tensor preparation

The face crop is:

1. Converted from BGR to RGB.
2. Resized to `224 x 224`.
3. Divided by `255`.
4. Normalized using ImageNet channel statistics.
5. Transposed to channel-first format.
6. Converted to a `float32` ONNX input tensor.

### 6.5 Model inference

The primary ONNX model produces raw outputs. The first eight values are converted to probabilities with softmax. Auxiliary model probabilities are calculated in the same way and blended with the primary probabilities.

### 6.6 Custom post-processing

The repository applies several inference-time adjustments:

- **Positive-expression consistency rule:** when happiness or learned positive valence is strong, residual anger and fear probabilities are reduced to suppress likely visual artifacts.
- **Temporal smoothing:** recent probability vectors are retained in a configurable history buffer. Newer frames receive larger weights, reducing frame-to-frame flicker.
- **Valence/arousal clipping:** learned values are constrained to the expected ranges.
- **Fallback affect calculation:** if continuous values are unavailable, valence and arousal are estimated from emotion probabilities and display metadata.

These are application-level changes around the pretrained models. They do not modify the ONNX model weights.

## 7. User Workflows

### 7.1 Snapshot workflow

The user can capture a webcam frame, run the recognition pipeline, and append the result to the active patient's timeline.

The displayed result includes:

- Face bounding box.
- Dominant emotion.
- Confidence.
- Valence.
- Arousal.
- Facial-quality information.

### 7.2 Video workflow

The user can:

- Load a preconfigured clinical scenario.
- Generate or process the included sample therapy video.
- Upload an `.mp4`, `.avi`, or `.mov` file.

The video is processed frame by frame. Each frame contributes a record to the active analytics timeline.

### 7.3 Demo scenarios

The project contains simulated scenarios representing:

- Anxiety and panic vulnerability.
- Depressive affect activation.
- Trauma-trigger exposure and grounding regulation.

These scenarios generate synthetic time-series records for demonstrating dashboards without requiring a live camera or video.

## 8. Analytics Engine

The analytics engine converts frame-level predictions into a session summary.

### 8.1 Emotion distribution

For detected frames, it calculates the count and percentage of each emotion and identifies the dominant and secondary emotions.

### 8.2 Valence and arousal

It calculates:

- Mean valence.
- Valence standard deviation.
- Mean arousal.
- Arousal standard deviation.

These values are used to position the session in a Russell-style valence/arousal interpretation.

### 8.3 Positive-to-negative affect ratio

The project calculates PNAR as:

```text
positive happy frames / negative sad, fear, angry, and disgust frames
```

Neutral and surprise frames are not included in the negative count.

### 8.4 Transition analysis

The engine counts changes between consecutive detected emotion labels and reports the transition rate per minute. It also builds an emotion transition probability matrix based on observed sequential states.

### 8.5 Affective entropy

Shannon entropy is calculated from the empirical emotion distribution. A normalized entropy value is used to describe whether the session appears emotionally constricted or broadly varied.

### 8.6 Emotional Stability Index

The ESI is an application-defined score from 0 to 100. It penalizes:

- Valence volatility.
- Frequent emotion transitions.
- Negative mean valence.
- Entropy that is far from the configured reference level.

The resulting score is classified into tiers such as high emotional grounding, moderate adaptive range, elevated affective turbulence, or acute emotional dysregulation.

### 8.7 Distress episodes and recovery

The engine looks for abrupt high-confidence negative-expression intervals and estimates the time taken to return toward a neutral or positive affective state.

These are heuristic analytics, not clinical diagnoses or validated psychiatric measurements.

## 9. User Interface and Visualization

The Streamlit interface provides:

- A patient selector and patient-registration form.
- A webcam snapshot workspace.
- A therapy-video analysis workspace.
- Preloaded benchmark sessions.
- A clinical analytics dashboard.
- A therapy-report and PDF-export workspace.

The interface presents:

- Emotion distribution charts.
- Valence and arousal timelines.
- Emotion transition visualizations.
- Distress episode annotations.
- Facial-quality indicators.
- ESI and stability-tier cards.
- Patient and longitudinal-session information.

The visual theme uses soft sage, slate, and neutral colors intended to reduce visual intensity in a therapy-oriented interface.

## 10. Database and Persistence

The application uses SQLite at:

```text
data/therapy_sessions.db
```

The database contains two main entities:

### Patients

Patient records include:

- Patient ID.
- Name.
- Age.
- Gender.
- Diagnosis or diagnostic impression.
- Attending therapist.
- Clinical notes.

### Therapy sessions

Session records include:

- Session ID and patient ID.
- Session title and date.
- Duration.
- Dominant emotion.
- ESI score and stability tier.
- Distress-episode count.
- Average recovery time.
- PNAR.
- Therapist notes.
- Clinical summary.
- Serialized analytics metrics.
- Serialized timeline records.

Three demonstration patients are seeded automatically when the database has no patient records.

## 11. Clinical Narrative and PDF Reporting

The report generator creates a SOAP-style narrative:

- **Subjective:** patient context, diagnostic consideration, dominant affect, and therapist notes.
- **Objective:** frame count, ESI, valence, arousal, distress episodes, recovery time, and entropy.
- **Assessment:** rule-based interpretation of stability and positive-to-negative affect.
- **Plan:** suggested grounding, breathing, cognitive-reframing, distress-tolerance, and longitudinal-tracking actions.

The PDF export includes:

- Patient/session details.
- Metric scorecards.
- Emotion distribution charts.
- Affect timelines.
- Transition visualizations.
- Circumplex/radar-style charts.
- SOAP narrative.
- Clinician sign-off area.
- Confidentiality footer and page numbering.

Reports are generated in the local `reports/` directory at runtime.

## 12. Installation and Execution

### Requirements

- Python 3.9 or newer.
- A working webcam for live capture, if desired.
- Sufficient disk space for Python dependencies and ONNX model files.

### Setup

```bash
cd /Users/pratyush/Projexx/facial-emotion-therapy

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Start the application

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## 13. Repository Structure

```text
facial-emotion-therapy/
├── app.py
├── fer_engine.py
├── analytics.py
├── database.py
├── report_generator.py
├── demo_data.py
├── requirements.txt
├── README.md
├── PROJECT_REPORT.md
├── models/
│   ├── facial_expression_model_weights.onnx
│   ├── haarcascade_frontalface_default.xml
│   ├── haarcascade_smile.xml
│   └── hsemotion/
│       ├── enet_b0_8_va_mtl.onnx
│       ├── enet_b0_8_best_vgaf.onnx
│       ├── enet_b0_8_best_afew.onnx
│       └── enet_b2_8.onnx
├── data/
│   └── therapy_sessions.db
├── samples/
└── reports/
```

## 14. Limitations and Responsible Use

- Facial expression is not equivalent to a person's internal emotional state.
- Lighting, camera position, occlusion, pose, skin tone, facial anatomy, and video quality can affect predictions.
- The largest-face assumption may be incorrect in multi-person scenes.
- The model and application-level heuristics require independent validation before clinical deployment.
- ESI, PNAR, distress episodes, recovery latency, and generated recommendations are project-specific analytics.
- The generated SOAP report is an aid for review and must not be treated as an autonomous medical record.
- Patient data should be handled according to applicable privacy, consent, security, and healthcare regulations.

## 15. Conclusion

MindPulse FER is not simply a direct call to a facial-expression model. It uses pretrained HSEmotion EfficientNet-B0 ONNX weights as the neural inference layer and builds a broader session-monitoring system around them. The additional layers perform image-quality improvement, face localization, input preparation, ensemble blending, heuristic probability correction, temporal smoothing, clinical-style analytics, database persistence, interactive visualization, and PDF report generation.

The result is a demonstrable end-to-end prototype for exploring facial-expression trends during therapy-session media, while keeping the clinician responsible for interpretation and decision-making.
