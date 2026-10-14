"""
Therapy Session & Patient History Database Management (SQLite)
Provides persistent storage for:
- Patient clinical demographic profiles
- Longitudinal therapy sessions
- Frame-level emotion history and timeline records
- Session analytics, ESI scores, and therapist clinical notes
"""

import os
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any


class TherapyDatabase:
    """
    Manages SQLite storage for clinical therapy sessions and patient longitudinal history.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(base_dir, 'data')
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, 'therapy_sessions.db')

        self.db_path = db_path
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_database(self):
        """Initializes tables and seeds initial clinical demo patients."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Patients Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                age INTEGER,
                gender TEXT,
                diagnosis TEXT,
                attending_therapist TEXT,
                clinical_notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # Therapy Sessions Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS therapy_sessions (
                session_id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                session_title TEXT,
                session_date TEXT,
                duration_sec REAL,
                dominant_emotion TEXT,
                esi_score REAL,
                stability_tier TEXT,
                distress_episodes INTEGER,
                avg_recovery_sec REAL,
                pnar_ratio REAL,
                therapist_notes TEXT,
                clinical_summary TEXT,
                metrics_json TEXT,
                timeline_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
            )
            """)

            conn.commit()

        # Seed sample clinical demo patients if empty
        self._seed_default_patients()

    def _seed_default_patients(self):
        """Seeds initial clinical patient profiles for demo and practice."""
        patients = self.get_all_patients()
        if not patients:
            default_patients = [
                {
                    'patient_id': 'PT-101',
                    'name': 'Elena Rostova',
                    'age': 32,
                    'gender': 'Female',
                    'diagnosis': 'Generalized Anxiety Disorder & Panic Vulnerability',
                    'attending_therapist': 'Dr. Marcus Vance, Psy.D.',
                    'clinical_notes': 'Experiencing somatic tension and acute anxiety spikes during career transition discussions.'
                },
                {
                    'patient_id': 'PT-102',
                    'name': 'David Chen',
                    'age': 41,
                    'gender': 'Male',
                    'diagnosis': 'Major Depressive Episode (Moderate, Recurrent)',
                    'attending_therapist': 'Dr. Sarah Lin, Ph.D.',
                    'clinical_notes': 'Affective blunting, psychomotor slowing, difficulty accessing positive emotional states.'
                },
                {
                    'patient_id': 'PT-103',
                    'name': 'Sophia Martinez',
                    'age': 27,
                    'gender': 'Female',
                    'diagnosis': 'Post-Traumatic Stress & Emotional Dysregulation',
                    'attending_therapist': 'Dr. Marcus Vance, Psy.D.',
                    'clinical_notes': 'Triggers evoke sudden hyper-arousal and fear/anger micro-expressions followed by prolonged dysregulation.'
                }
            ]
            for p in default_patients:
                self.save_patient(p)

    def save_patient(self, patient_dict: Dict[str, Any]):
        """Creates or updates a patient profile."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO patients (patient_id, name, age, gender, diagnosis, attending_therapist, clinical_notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(patient_id) DO UPDATE SET
                name=excluded.name,
                age=excluded.age,
                gender=excluded.gender,
                diagnosis=excluded.diagnosis,
                attending_therapist=excluded.attending_therapist,
                clinical_notes=excluded.clinical_notes
            """, (
                patient_dict['patient_id'],
                patient_dict['name'],
                patient_dict.get('age', 30),
                patient_dict.get('gender', 'Other'),
                patient_dict.get('diagnosis', 'Adjustment Disorder'),
                patient_dict.get('attending_therapist', 'Dr. Clinician'),
                patient_dict.get('clinical_notes', '')
            ))
            conn.commit()

    def get_all_patients(self) -> List[Dict[str, Any]]:
        """Retrieves list of all registered patients."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM patients ORDER BY name ASC")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves single patient profile by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def save_session(self, session_dict: Dict[str, Any]):
        """Persists a complete therapy session record."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO therapy_sessions (
                session_id, patient_id, session_title, session_date, duration_sec,
                dominant_emotion, esi_score, stability_tier, distress_episodes,
                avg_recovery_sec, pnar_ratio, therapist_notes, clinical_summary,
                metrics_json, timeline_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                therapist_notes=excluded.therapist_notes,
                clinical_summary=excluded.clinical_summary,
                esi_score=excluded.esi_score,
                metrics_json=excluded.metrics_json,
                timeline_json=excluded.timeline_json
            """, (
                session_dict['session_id'],
                session_dict['patient_id'],
                session_dict.get('session_title', 'Cognitive Affective Session'),
                session_dict.get('session_date', datetime.now().strftime("%Y-%m-%d %H:%M")),
                session_dict.get('duration_sec', 0.0),
                session_dict.get('dominant_emotion', 'neutral'),
                session_dict.get('esi_score', 75.0),
                session_dict.get('stability_tier', 'Moderate Adaptive Affect'),
                session_dict.get('distress_episodes', 0),
                session_dict.get('avg_recovery_sec', 0.0),
                session_dict.get('pnar_ratio', 1.0),
                session_dict.get('therapist_notes', ''),
                session_dict.get('clinical_summary', ''),
                json.dumps(session_dict.get('metrics', {})),
                json.dumps(session_dict.get('timeline', []))
            ))
            conn.commit()

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Retrieves all sessions across all patients."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT s.*, p.name as patient_name, p.diagnosis
            FROM therapy_sessions s
            JOIN patients p ON s.patient_id = p.patient_id
            ORDER BY s.created_at DESC
            """)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves full session record including decoded JSON metrics and timeline."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT s.*, p.name as patient_name, p.age, p.gender, p.diagnosis, p.attending_therapist
            FROM therapy_sessions s
            JOIN patients p ON s.patient_id = p.patient_id
            WHERE s.session_id = ?
            """, (session_id,))
            row = cursor.fetchone()
            if not row:
                return None
            data = dict(row)
            data['metrics'] = json.loads(data['metrics_json']) if data.get('metrics_json') else {}
            data['timeline'] = json.loads(data['timeline_json']) if data.get('timeline_json') else []
            return data

    def get_patient_sessions(self, patient_id: str) -> List[Dict[str, Any]]:
        """Retrieves longitudinal sessions for a single patient."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT * FROM therapy_sessions
            WHERE patient_id = ?
            ORDER BY session_date ASC
            """, (patient_id,))
            rows = cursor.fetchall()
            result = []
            for r in rows:
                item = dict(r)
                item['metrics'] = json.loads(item['metrics_json']) if item.get('metrics_json') else {}
                result.append(item)
            return result
