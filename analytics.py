"""
Clinical Emotion Analytics Engine
Calculates:
- Comprehensive Emotion Timeline (Valence, Arousal, Dominant Emotion trajectories)
- Emotion Stability Index (ESI™) & Emotional Volatility
- Markov Transition Probabilities (7x7 State Transition Matrix)
- Therapeutic Distress Spikes & Recovery Latency
- Russell's Circumplex Affect Coordinates (Valence vs Arousal)
- Shannon Affective Entropy & Positive-to-Negative Affect Ratio (PNAR)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from fer_engine import EMOTION_LABELS, EMOTION_META


class EmotionAnalyticsEngine:
    """
    Analyzes frame-by-frame emotion records gathered across a therapy session.
    Transforms raw neural predictions into clinically actionable psychiatric metrics.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Clears current session records."""
        self.records: List[Dict[str, Any]] = []

    def add_record(self, elapsed_sec: float, prediction: Optional[Dict[str, Any]], fqa_score: float = 85.0):
        """
        Record a single temporal data point during therapy session.
        If prediction is None (e.g. face temporarily turned away), records missing indicator.
        """
        if prediction is not None:
            self.records.append({
                'elapsed_sec': round(elapsed_sec, 2),
                'detected': True,
                'emotion': prediction['dominant_emotion'],
                'confidence': prediction['confidence'],
                'probabilities': prediction['probabilities'],
                'valence': prediction['valence'],
                'arousal': prediction['arousal'],
                'fqa_score': fqa_score
            })
        else:
            self.records.append({
                'elapsed_sec': round(elapsed_sec, 2),
                'detected': False,
                'emotion': 'undetected',
                'confidence': 0.0,
                'probabilities': {e: 0.0 for e in EMOTION_LABELS},
                'valence': 0.0,
                'arousal': 0.0,
                'fqa_score': fqa_score
            })

    def get_timeline_dataframe(self) -> pd.DataFrame:
        """
        Converts the session records into a structured Pandas DataFrame.
        """
        if not self.records:
            return pd.DataFrame()

        rows = []
        for r in self.records:
            row = {
                'elapsed_sec': r['elapsed_sec'],
                'time_str': self._format_seconds(r['elapsed_sec']),
                'detected': r['detected'],
                'emotion': r['emotion'],
                'confidence': r['confidence'],
                'valence': r['valence'],
                'arousal': r['arousal'],
                'fqa_score': r['fqa_score']
            }
            # Flatten probability keys
            for e in EMOTION_LABELS:
                row[f'prob_{e}'] = r['probabilities'].get(e, 0.0)
            rows.append(row)

        df = pd.DataFrame(rows)
        return df

    def compute_session_analytics(self) -> Dict[str, Any]:
        """
        Calculates all clinical emotion analytics and emotional stability index.
        """
        if not self.records:
            return self._empty_analytics()

        df = self.get_timeline_dataframe()
        valid_df = df[df['detected'] == True]

        total_frames = len(df)
        detected_frames = len(valid_df)
        face_visibility_pct = round((detected_frames / total_frames * 100.0) if total_frames > 0 else 0.0, 1)

        if detected_frames == 0:
            return self._empty_analytics(face_visibility_pct=face_visibility_pct)

        # 1. Emotion Frequency & Percentages
        emotion_counts = {e: 0 for e in EMOTION_LABELS}
        for emo in valid_df['emotion']:
            if emo in emotion_counts:
                emotion_counts[emo] += 1

        emotion_percentages = {
            e: round((cnt / detected_frames) * 100.0, 1)
            for e, cnt in emotion_counts.items()
        }

        # Sort emotions by prevalence
        sorted_emotions = sorted(emotion_percentages.items(), key=lambda x: x[1], reverse=True)
        dominant_emotion, dominant_pct = sorted_emotions[0]
        secondary_emotion, secondary_pct = sorted_emotions[1] if len(sorted_emotions) > 1 else ('none', 0.0)

        # 2. Valence & Arousal Statistics
        valences = valid_df['valence'].values
        arousals = valid_df['arousal'].values

        mean_valence = float(np.mean(valences))
        valence_std = float(np.std(valences))
        mean_arousal = float(np.mean(arousals))
        arousal_std = float(np.std(arousals))

        # 3. Positive vs Negative Affect Ratio
        pos_frames = sum(emotion_counts[e] for e in ['happy'])
        neg_frames = sum(emotion_counts[e] for e in ['sad', 'fear', 'angry', 'disgust'])
        neutral_frames = sum(emotion_counts[e] for e in ['neutral', 'surprise'])

        if neg_frames > 0:
            pnar_ratio = round(pos_frames / neg_frames, 2)
        else:
            pnar_ratio = round(pos_frames if pos_frames > 0 else 1.0, 2)

        # 4. Emotional Stability Index (ESI™) & Volatility Calculation
        # Volatility = average magnitude of valence jumps between consecutive timepoints
        if len(valences) > 1:
            valence_deltas = np.abs(np.diff(valences))
            volatility = float(np.mean(valence_deltas))
        else:
            volatility = 0.0

        # State transition rate (emotion switches per minute)
        emotions_seq = valid_df['emotion'].tolist()
        transitions_count = sum(1 for i in range(1, len(emotions_seq)) if emotions_seq[i] != emotions_seq[i - 1])
        total_duration_sec = valid_df['elapsed_sec'].iloc[-1] - valid_df['elapsed_sec'].iloc[0] if len(valid_df) > 1 else 1.0
        total_minutes = max(total_duration_sec / 60.0, 0.05)
        transition_rate_per_min = round(transitions_count / total_minutes, 1)

        # Shannon Affective Entropy: measures uniformity of emotional expression
        probs_empirical = np.array([emotion_counts[e] / detected_frames for e in EMOTION_LABELS])
        nonzero_probs = probs_empirical[probs_empirical > 0]
        entropy = -float(np.sum(nonzero_probs * np.log2(nonzero_probs)))
        max_entropy = np.log2(len(EMOTION_LABELS))
        normalized_entropy = round(entropy / max_entropy, 2)

        # Mathematical ESI Formula:
        # High volatility, erratic transitions, and extreme negative valence lower stability
        # Scaled from 0 to 100
        volatility_penalty = min(volatility / 0.5, 1.0) * 35.0
        transition_penalty = min(transition_rate_per_min / 30.0, 1.0) * 25.0
        negative_bias_penalty = max(0.0, -mean_valence) * 20.0
        entropy_extremes_penalty = abs(normalized_entropy - 0.55) * 20.0  # Either rigid 0 or pure chaos 1 is penalized

        raw_esi = 100.0 - (volatility_penalty + transition_penalty + negative_bias_penalty + entropy_extremes_penalty)
        esi_score = round(max(5.0, min(98.5, raw_esi)), 1)

        # Stability Classification
        if esi_score >= 78.0:
            stability_tier = "High Emotional Grounding & Equilibrium"
            stability_color = "#4CAF50"
            stability_interpretation = "Patient maintains strong affective regulation with stable baseline recovery."
        elif esi_score >= 58.0:
            stability_tier = "Moderate Adaptive Affective Range"
            stability_color = "#00ACC1"
            stability_interpretation = "Patient expresses appropriate emotional reactivity with normal therapeutic modulation."
        elif esi_score >= 40.0:
            stability_tier = "Elevated Affective Turbulence / Vulnerability"
            stability_color = "#FF9800"
            stability_interpretation = "Elevated shifts between states; susceptible to acute anxiety or negative mood reactivation."
        else:
            stability_tier = "Acute Emotional Dysregulation"
            stability_color = "#E53935"
            stability_interpretation = "Severe affective volatility, erratic transitions, or intense unmodulated distress episodes."

        # 5. Distress Episodes & Recovery Latency
        distress_episodes, avg_recovery_sec = self._detect_distress_spikes_and_recovery(valid_df)

        # 6. Markov Emotion Transition Matrix (7x7)
        transition_matrix = self._compute_transition_matrix(emotions_seq)

        # 7. Circumplex Quadrant Assessment
        if mean_valence >= 0.0 and mean_arousal >= 0.45:
            quadrant = "Active Engagement / Energized Positive"
        elif mean_valence >= 0.0 and mean_arousal < 0.45:
            quadrant = "Calm Equilibrium / Receptive Serenity"
        elif mean_valence < 0.0 and mean_arousal >= 0.45:
            quadrant = "Acute Distress / Agitation & Anxiety"
        else:
            quadrant = "Dysphoric Withdrawal / Depressive Affect"

        return {
            'total_session_seconds': round(total_duration_sec, 1),
            'total_frames': total_frames,
            'detected_frames': detected_frames,
            'face_visibility_pct': face_visibility_pct,
            'dominant_emotion': dominant_emotion,
            'dominant_pct': dominant_pct,
            'secondary_emotion': secondary_emotion,
            'secondary_pct': secondary_pct,
            'emotion_counts': emotion_counts,
            'emotion_percentages': emotion_percentages,
            'mean_valence': round(mean_valence, 2),
            'valence_std': round(valence_std, 2),
            'mean_arousal': round(mean_arousal, 2),
            'arousal_std': round(arousal_std, 2),
            'pnar_ratio': pnar_ratio,
            'volatility': round(volatility, 3),
            'transition_rate_per_min': transition_rate_per_min,
            'normalized_entropy': normalized_entropy,
            'esi_score': esi_score,
            'stability_tier': stability_tier,
            'stability_color': stability_color,
            'stability_interpretation': stability_interpretation,
            'distress_episodes_count': len(distress_episodes),
            'distress_episodes': distress_episodes,
            'avg_recovery_sec': avg_recovery_sec,
            'transition_matrix': transition_matrix,
            'quadrant': quadrant
        }

    def _detect_distress_spikes_and_recovery(self, df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], float]:
        """
        Identifies moments where negative distress emotions (sad, fear, angry) spike > 55%,
        and calculates how many seconds it took to recover to baseline (neutral or happy).
        """
        distress_emotions = {'sad', 'fear', 'angry'}
        episodes = []
        in_episode = False
        spike_start_sec = 0.0
        spike_emotion = ''
        peak_conf = 0.0
        recovery_times = []

        records = df.to_dict('records')
        for idx, row in enumerate(records):
            emo = row['emotion']
            sec = row['elapsed_sec']
            conf = row['confidence']

            if emo in distress_emotions and conf >= 50.0:
                if not in_episode:
                    in_episode = True
                    spike_start_sec = sec
                    spike_emotion = emo
                    peak_conf = conf
                else:
                    if conf > peak_conf:
                        peak_conf = conf
                        spike_emotion = emo
            else:
                if in_episode:
                    # Episode ended, check recovery
                    recovery_latency = round(sec - spike_start_sec, 1)
                    recovery_times.append(recovery_latency)
                    episodes.append({
                        'start_sec': round(spike_start_sec, 1),
                        'start_str': self._format_seconds(spike_start_sec),
                        'duration_sec': recovery_latency,
                        'peak_emotion': spike_emotion,
                        'peak_confidence': round(peak_conf, 1)
                    })
                    in_episode = False

        # If session ended while in distress episode
        if in_episode and records:
            recovery_latency = round(records[-1]['elapsed_sec'] - spike_start_sec, 1)
            recovery_times.append(recovery_latency)
            episodes.append({
                'start_sec': round(spike_start_sec, 1),
                'start_str': self._format_seconds(spike_start_sec),
                'duration_sec': recovery_latency,
                'peak_emotion': spike_emotion,
                'peak_confidence': round(peak_conf, 1)
            })

        avg_recovery = round(float(np.mean(recovery_times)), 1) if recovery_times else 0.0
        return episodes, avg_recovery

    def _compute_transition_matrix(self, sequence: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Builds a 7x7 Markov transition matrix:
        P(emotion_j at t | emotion_i at t-1).
        """
        n = len(EMOTION_LABELS)
        matrix = {e1: {e2: 0.0 for e2 in EMOTION_LABELS} for e1 in EMOTION_LABELS}
        totals = {e: 0 for e in EMOTION_LABELS}

        for i in range(len(sequence) - 1):
            s_from = sequence[i]
            s_to = sequence[i + 1]
            if s_from in matrix and s_to in matrix[s_from]:
                matrix[s_from][s_to] += 1
                totals[s_from] += 1

        # Normalize to conditional probabilities
        prob_matrix = {}
        for e_from in EMOTION_LABELS:
            prob_matrix[e_from] = {}
            tot = totals[e_from]
            for e_to in EMOTION_LABELS:
                if tot > 0:
                    prob_matrix[e_from][e_to] = round(matrix[e_from][e_to] / tot, 3)
                else:
                    prob_matrix[e_from][e_to] = 0.0

        return prob_matrix

    def _format_seconds(self, sec: float) -> str:
        """Formats seconds into MM:SS string."""
        m = int(sec // 60)
        s = int(sec % 60)
        return f"{m:02d}:{s:02d}"

    def _empty_analytics(self, face_visibility_pct: float = 0.0) -> Dict[str, Any]:
        """Fallback empty analytics structure."""
        return {
            'total_session_seconds': 0.0,
            'total_frames': 0,
            'detected_frames': 0,
            'face_visibility_pct': face_visibility_pct,
            'dominant_emotion': 'neutral',
            'dominant_pct': 0.0,
            'secondary_emotion': 'none',
            'secondary_pct': 0.0,
            'emotion_counts': {e: 0 for e in EMOTION_LABELS},
            'emotion_percentages': {e: 0.0 for e in EMOTION_LABELS},
            'mean_valence': 0.0,
            'valence_std': 0.0,
            'mean_arousal': 0.0,
            'arousal_std': 0.0,
            'pnar_ratio': 1.0,
            'volatility': 0.0,
            'transition_rate_per_min': 0.0,
            'normalized_entropy': 0.0,
            'esi_score': 75.0,
            'stability_tier': 'Baseline State',
            'stability_color': '#607D8B',
            'stability_interpretation': 'Awaiting sufficient session observation data.',
            'distress_episodes_count': 0,
            'distress_episodes': [],
            'avg_recovery_sec': 0.0,
            'transition_matrix': {e1: {e2: 0.0 for e2 in EMOTION_LABELS} for e1 in EMOTION_LABELS},
            'quadrant': 'Neutral Baseline'
        }
