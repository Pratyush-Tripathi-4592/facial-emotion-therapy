"""
Clinical Demo Data & Sample Therapy Scenarios
Provides realistic, medically sound simulated therapy session scenarios for instant evaluation:
1. Scenario A: Generalized Anxiety Disorder & Panic Vulnerability (Elena Rostova)
2. Scenario B: Major Depressive Disorder Breakthrough & Affect Activation (David Chen)
3. Scenario C: Trauma Exposure & Grounding Regulation (Sophia Martinez)
Also provides a video generator to create realistic synthetic video files for video upload testing.
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Any


def get_clinical_demo_scenarios() -> Dict[str, Dict[str, Any]]:
    """
    Returns pre-configured clinical therapy scenarios with authentic time-series emotion data.
    """
    scenarios = {
        'scenario_anxiety': {
            'title': 'Anxiety & Panic Vulnerability (Cognitive Restructuring)',
            'patient_id': 'PT-101',
            'patient_name': 'Elena Rostova',
            'age': 32,
            'gender': 'Female',
            'diagnosis': 'Generalized Anxiety Disorder & Panic Vulnerability',
            'therapist_name': 'Dr. Marcus Vance, Psy.D.',
            'therapist_notes': 'Patient discussed panic sensations during public presentations. Somatic tension elevated at minute 00:22; guided 4-7-8 breathing successfully restored equilibrium at 00:45.',
            'duration_sec': 60.0,
            'generator_func': _generate_anxiety_session_timeline
        },
        'scenario_depression': {
            'title': 'Depressive Episode (Affect Activation & Hope Discovery)',
            'patient_id': 'PT-102',
            'patient_name': 'David Chen',
            'age': 41,
            'gender': 'Male',
            'diagnosis': 'Major Depressive Episode (Moderate, Recurrent)',
            'therapist_name': 'Dr. Sarah Lin, Ph.D.',
            'therapist_notes': 'Persistent psychomotor flat affect during initial half. When reflecting on family support, genuine positive affect and emotional release emerged with visible relief.',
            'duration_sec': 60.0,
            'generator_func': _generate_depression_session_timeline
        },
        'scenario_ptsd': {
            'title': 'PTSD Trigger Exposure & DBT Grounding Regulation',
            'patient_id': 'PT-103',
            'patient_name': 'Sophia Martinez',
            'age': 27,
            'gender': 'Female',
            'diagnosis': 'Post-Traumatic Stress & Emotional Dysregulation',
            'therapist_name': 'Dr. Marcus Vance, Psy.D.',
            'therapist_notes': 'Trauma memory narrative triggered abrupt fear and agitation spike at 00:18. Initiated DBT TIPP temperature shift and sensory grounding; patient regained affective baseline.',
            'duration_sec': 60.0,
            'generator_func': _generate_ptsd_session_timeline
        }
    }
    return scenarios


def _generate_anxiety_session_timeline(duration_sec: float = 60.0, fps: float = 2.0) -> List[Dict[str, Any]]:
    """
    Generates a realistic anxiety therapy trajectory:
    - 0-15s: Neutral baseline with mild apprehension (Neutral 70%, Fear 20%)
    - 15-35s: Anxiety & Panic spike upon reviewing catastrophic cognition (Fear 85%, Angry 10%)
    - 35-50s: Cognitive restructuring & paced respiration (Fear subsiding, Neutral rising)
    - 50-60s: Grounded relief and cognitive consolidation (Neutral 60%, Happy 35%)
    """
    records = []
    total_frames = int(duration_sec * fps)

    for i in range(total_frames):
        sec = round(i / fps, 2)
        if sec < 15.0:
            probs = {'angry': 0.02, 'disgust': 0.01, 'fear': 0.18, 'happy': 0.04, 'sad': 0.05, 'surprise': 0.08, 'neutral': 0.62}
            dom = 'neutral'
            conf = 62.0
            val = 0.05
            aro = 0.25
        elif sec < 35.0:
            # Acute anxiety spike
            peak_factor = min(1.0, (sec - 15.0) / 8.0)
            fear_p = 0.65 + 0.25 * peak_factor
            probs = {'angry': 0.08, 'disgust': 0.02, 'fear': fear_p, 'happy': 0.01, 'sad': 0.06, 'surprise': 0.08, 'neutral': max(0.05, 1.0 - fear_p - 0.25)}
            dom = 'fear'
            conf = round(fear_p * 100.0, 1)
            val = -0.82
            aro = 0.88
        elif sec < 48.0:
            # Calming phase
            decay = (sec - 35.0) / 13.0
            fear_p = max(0.12, 0.75 - decay * 0.60)
            neut_p = 0.20 + decay * 0.55
            probs = {'angry': 0.02, 'disgust': 0.01, 'fear': fear_p, 'happy': 0.08, 'sad': 0.04, 'surprise': 0.03, 'neutral': neut_p}
            dom = 'neutral' if neut_p > fear_p else 'fear'
            conf = round(max(neut_p, fear_p) * 100.0, 1)
            val = -0.25 + decay * 0.45
            aro = 0.70 - decay * 0.40
        else:
            # Grounded relief
            probs = {'angry': 0.01, 'disgust': 0.01, 'fear': 0.04, 'happy': 0.42, 'sad': 0.02, 'surprise': 0.05, 'neutral': 0.45}
            dom = 'neutral' if probs['neutral'] > probs['happy'] else 'happy'
            conf = 45.0
            val = 0.55
            aro = 0.35

        records.append({
            'elapsed_sec': sec,
            'detected': True,
            'emotion': dom,
            'confidence': conf,
            'probabilities': {k: round(v, 4) for k, v in probs.items()},
            'valence': round(val + np.random.normal(0, 0.02), 3),
            'arousal': round(aro + np.random.normal(0, 0.02), 3),
            'fqa_score': round(90.0 + np.random.normal(0, 3.0), 1)
        })

    return records


def _generate_depression_session_timeline(duration_sec: float = 60.0, fps: float = 2.0) -> List[Dict[str, Any]]:
    """
    Generates a realistic depressive breakthrough trajectory:
    - 0-25s: Constricted flat affect / dysphoric sadness (Sad 65%, Neutral 25%)
    - 25-42s: Emotional processing & cognitive exploration (Sad 40%, Neutral 45%)
    - 42-60s: Affective activation, emerging warmth and genuine smile (Happy 70%, Neutral 20%)
    """
    records = []
    total_frames = int(duration_sec * fps)

    for i in range(total_frames):
        sec = round(i / fps, 2)
        if sec < 25.0:
            probs = {'angry': 0.03, 'disgust': 0.04, 'fear': 0.05, 'happy': 0.02, 'sad': 0.68, 'surprise': 0.02, 'neutral': 0.16}
            dom = 'sad'
            conf = 68.0
            val = -0.75
            aro = 0.22
        elif sec < 42.0:
            probs = {'angry': 0.02, 'disgust': 0.02, 'fear': 0.04, 'happy': 0.18, 'sad': 0.35, 'surprise': 0.06, 'neutral': 0.33}
            dom = 'sad' if sec < 33.0 else 'neutral'
            conf = 42.0
            val = -0.20
            aro = 0.38
        else:
            growth = min(1.0, (sec - 42.0) / 15.0)
            happy_p = 0.45 + 0.35 * growth
            probs = {'angry': 0.01, 'disgust': 0.01, 'fear': 0.02, 'happy': happy_p, 'sad': 0.04, 'surprise': 0.08, 'neutral': max(0.05, 1.0 - happy_p - 0.16)}
            dom = 'happy'
            conf = round(happy_p * 100.0, 1)
            val = 0.72
            aro = 0.62

        records.append({
            'elapsed_sec': sec,
            'detected': True,
            'emotion': dom,
            'confidence': conf,
            'probabilities': {k: round(v, 4) for k, v in probs.items()},
            'valence': round(val + np.random.normal(0, 0.02), 3),
            'arousal': round(aro + np.random.normal(0, 0.02), 3),
            'fqa_score': round(92.0 + np.random.normal(0, 2.5), 1)
        })

    return records


def _generate_ptsd_session_timeline(duration_sec: float = 60.0, fps: float = 2.0) -> List[Dict[str, Any]]:
    """
    Generates a realistic PTSD trauma exposure and grounding trajectory:
    - 0-18s: Guarded composure (Neutral 80%)
    - 18-36s: Abrupt flashback trigger with fear and angry frustration (Fear 60%, Angry 30%)
    - 36-60s: Grounding down-regulation back to calm composed baseline (Neutral 75%)
    """
    records = []
    total_frames = int(duration_sec * fps)

    for i in range(total_frames):
        sec = round(i / fps, 2)
        if sec < 18.0:
            probs = {'angry': 0.04, 'disgust': 0.02, 'fear': 0.06, 'happy': 0.03, 'sad': 0.05, 'surprise': 0.05, 'neutral': 0.75}
            dom = 'neutral'
            conf = 75.0
            val = 0.08
            aro = 0.18
        elif sec < 36.0:
            probs = {'angry': 0.32, 'disgust': 0.04, 'fear': 0.54, 'happy': 0.01, 'sad': 0.04, 'surprise': 0.02, 'neutral': 0.03}
            dom = 'fear'
            conf = 54.0
            val = -0.88
            aro = 0.92
        else:
            decay = min(1.0, (sec - 36.0) / 18.0)
            neut_p = 0.40 + decay * 0.42
            probs = {'angry': max(0.02, 0.20 - decay * 0.18), 'disgust': 0.01, 'fear': max(0.04, 0.35 - decay * 0.30), 'happy': 0.05, 'sad': 0.04, 'surprise': 0.04, 'neutral': neut_p}
            dom = 'neutral'
            conf = round(neut_p * 100.0, 1)
            val = -0.40 + decay * 0.45
            aro = 0.85 - decay * 0.60

        records.append({
            'elapsed_sec': sec,
            'detected': True,
            'emotion': dom,
            'confidence': conf,
            'probabilities': {k: round(v, 4) for k, v in probs.items()},
            'valence': round(val + np.random.normal(0, 0.02), 3),
            'arousal': round(aro + np.random.normal(0, 0.02), 3),
            'fqa_score': round(88.0 + np.random.normal(0, 3.0), 1)
        })

    return records


def generate_sample_therapy_video(output_path: str, duration_sec: int = 15, fps: int = 15):
    """
    Synthesizes a realistic sample therapy session video clip (.mp4) using OpenCV.
    Uses the real public face photo with subtle emotional expression and illumination variations
    so the user can immediately test the Video File Upload feature.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sample_face_path = os.path.join(base_dir, 'samples', 'patient_test_face.jpg')

    if not os.path.exists(sample_face_path):
        # Fallback: create synthetic canvas
        face_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(face_img, "Simulated Therapy Patient", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
    else:
        face_img = cv2.imread(sample_face_path)
        face_img = cv2.resize(face_img, (640, 480))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))

    total_frames = int(duration_sec * fps)
    for f in range(total_frames):
        t = f / fps
        frame = face_img.copy()

        # Add subtle lighting fluctuations and micro-shifts
        brightness_mod = 1.0 + 0.12 * np.sin(t * 1.5)
        frame = cv2.convertScaleAbs(frame, alpha=brightness_mod, beta=int(5 * np.cos(t * 2.0)))

        # Add timestamp watermark
        time_text = f"Therapy Session Feed [REC] {int(t // 60):02d}:{int(t % 60):02d}.{int((t % 1)*10):01d}"
        cv2.putText(frame, time_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 220, 50), 2, cv2.LINE_AA)

        out.write(frame)

    out.release()
    return output_path
