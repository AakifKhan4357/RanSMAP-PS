"""
classifier_pipeline.py
Core RanSMAP preprocessing and inference pipeline.
Replicates the exact preprocessing steps from RanSMAP_Ransomware_Binary_Classification.ipynb:
1. Reads memory CSV files (detecting headers or raw RanSMAP format).
2. Cleans missing values (dropna).
3. Validates features: ['GPA', 'size', 'entropy', 'type'].
4. Supports multi-model selection:
   - Baseline Model: BaggingClassifier with Decision Trees (Accuracy: 63.9%, F1: 51.1%)
   - Temporal-Aware Transformer: Single Optimized Transformer with harmonic temporal encodings (Accuracy: 96.5%, F1: 96.5%)
"""

import os
import io
import warnings
from typing import List, Dict, Any, Tuple, Union, Optional

try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
except ImportError:
    pass
warnings.filterwarnings('ignore', category=UserWarning)

import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn

# Standard memory page type mapping
TYPE_MAPPING = {
    4: '4 KiB page',
    2: '2 MiB page',
    1: '1 GiB page',
    0: 'MMIO access'
}

# Canonical columns in raw RanSMAP memory CSVs
RAW_MEM_COLUMNS = ['UNIX_time_sec', 'UNIX_time_ns', 'GPA', 'size', 'entropy', 'type']
FEATURE_COLUMNS = ['GPA', 'size', 'entropy', 'type']

AVAILABLE_MODELS = {
    'temporal_transformer': {
        'id': 'temporal_transformer',
        'name': 'Temporal-Aware Transformer Model',
        'f1_score': '96.5%',
        'accuracy': '96.5%',
        'description': 'Single Optimized Transformer with harmonic temporal positional encodings (Recommended).',
        'is_default': True,
        'filename': 'temporal_transformer_model.pkl'
    },
    'baseline': {
        'id': 'baseline',
        'name': 'Baseline Model (Bagging Classifier)',
        'f1_score': '51.1%',
        'accuracy': '63.9%',
        'description': 'Baseline BaggingClassifier with 10 Decision Trees from notebook Cell 165.',
        'is_default': False,
        'filename': 'bagging_ransomware_model.pkl'
    }
}


class TemporalAwareTransformerClassifier(nn.Module):
    """
    Single Optimized Transformer with temporal positional awareness.
    Attends across harmonic temporal coordinates and low-level memory operations.
    """
    def __init__(self, in_dim: int = 37, d_model: int = 64, nhead: int = 4, dim_feedforward: int = 128, dropout: float = 0.05):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        proj = self.proj(x)
        trans_out = self.transformer(proj)
        pooled = trans_out.squeeze(1)
        return self.fc(pooled)


class TemporalTransformerWrapper:
    """
    Self-contained picklable wrapper for the Temporal-Aware Transformer model.
    Provides scikit-learn compatible predict() and predict_proba() interfaces.
    """
    def __init__(self, model_state_dict: dict, scaler: Any, B_freq: np.ndarray, t_min: float, t_max: float, in_dim: int = 37):
        self.model_state_dict = model_state_dict
        self.scaler = scaler
        self.B_freq = B_freq
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.in_dim = in_dim
        self._init_model()

    def _init_model(self):
        self.model = TemporalAwareTransformerClassifier(in_dim=self.in_dim)
        self.model.load_state_dict(self.model_state_dict)
        self.model.eval()

    def __getstate__(self):
        return {
            'model_state_dict': {k: v.cpu() for k, v in self.model.state_dict().items()},
            'scaler': self.scaler,
            'B_freq': self.B_freq,
            't_min': self.t_min,
            't_max': self.t_max,
            'in_dim': self.in_dim
        }

    def __setstate__(self, state):
        self.model_state_dict = state['model_state_dict']
        self.scaler = state['scaler']
        self.B_freq = state['B_freq']
        self.t_min = state['t_min']
        self.t_max = state['t_max']
        self.in_dim = state['in_dim']
        self._init_model()

    def _prepare_features(self, X_input: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(X_input, pd.DataFrame):
            df_in = X_input
        else:
            df_in = pd.DataFrame(X_input, columns=['GPA', 'size', 'entropy', 'type'])

        mem_scaled = self.scaler.transform(df_in[['GPA', 'size', 'entropy', 'type']]).astype(np.float32)

        if 'UNIX_time_sec' in df_in.columns and 'UNIX_time_ns' in df_in.columns:
            t_s = pd.to_numeric(df_in['UNIX_time_sec'], errors='coerce').fillna((self.t_min + self.t_max) / 2.0).values
            t_n = pd.to_numeric(df_in['UNIX_time_ns'], errors='coerce').fillna(0).values
        else:
            t_s = np.full(len(df_in), (self.t_min + self.t_max) / 2.0)
            t_n = np.zeros(len(df_in))

        t_norm = (t_s - self.t_min) / (self.t_max - self.t_min + 1e-9)
        ns_norm = (t_n / 1e9).astype(np.float32)

        t_proj = 2.0 * np.pi * t_norm[:, None] @ self.B_freq
        sin_t = np.sin(t_proj).astype(np.float32)
        cos_t = np.cos(t_proj).astype(np.float32)

        return np.hstack([mem_scaled, sin_t, cos_t, ns_norm[:, None].astype(np.float32)])

    def predict_proba(self, X_input: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        X_feats = self._prepare_features(X_input)
        self.model.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(X_feats, dtype=torch.float32)
            logits = self.model(tensor_x)
            probs = torch.softmax(logits, dim=1).numpy()
        return probs

    def predict(self, X_input: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        probs = self.predict_proba(X_input)
        return np.argmax(probs, axis=1)


class RanSMAPPipeline:
    def __init__(
        self,
        baseline_model_path: str = 'bagging_ransomware_model.pkl',
        baseline_scaler_path: str = 'feature_scaler.pkl',
        temporal_model_path: str = 'temporal_transformer_model.pkl',
        default_model: str = 'temporal_transformer',
        # Backward compatibility with existing tests
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None
    ):
        """Load trained models (Baseline BaggingClassifier and Temporal-Aware Transformer)."""
        if model_path is not None:
            baseline_model_path = model_path
        if scaler_path is not None:
            baseline_scaler_path = scaler_path

        self.baseline_model_path = baseline_model_path
        self.baseline_scaler_path = baseline_scaler_path
        self.temporal_model_path = temporal_model_path
        self.default_model = default_model

        self.baseline_model = None
        self.baseline_scaler = None
        self.temporal_model = None

        self._load_artifacts()

        # For backwards compatibility with test_pipeline_initialization:
        self.model = self.baseline_model
        self.scaler = self.baseline_scaler

    def _load_artifacts(self):
        # Load baseline model and scaler
        if os.path.exists(self.baseline_model_path) and os.path.exists(self.baseline_scaler_path):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.baseline_model = joblib.load(self.baseline_model_path)
                self.baseline_scaler = joblib.load(self.baseline_scaler_path)

        # Load temporal transformer model
        if os.path.exists(self.temporal_model_path):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.temporal_model = joblib.load(self.temporal_model_path)

    @staticmethod
    def _parse_gpa(val: Any) -> float:
        """Parse Guest Physical Address (handles int, float, and hex strings like 0x80000000)."""
        if pd.isna(val):
            return np.nan
        if isinstance(val, (int, float, np.integer, np.floating)):
            return float(val)
        val_str = str(val).strip()
        if val_str.lower().startswith('0x'):
            try:
                return float(int(val_str, 16))
            except ValueError:
                return np.nan
        try:
            return float(val_str)
        except ValueError:
            return np.nan

    def parse_single_csv(
        self,
        file_source: Union[str, io.BytesIO, io.StringIO],
        filename: str = '',
        keep_timestamps: bool = False
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Parse and preprocess a single CSV file according to notebook conventions:
        - Detect header vs headless
        - Distinguish memory operations vs ATA disk operations
        - Extract GPA, size, entropy, type (and timestamps if keep_timestamps=True)
        - Clean null values
        """
        meta = {
            'filename': filename,
            'is_memory': False,
            'is_storage': False,
            'raw_rows': 0,
            'cleaned_rows': 0,
            'dropped_nulls': 0,
            'notes': ''
        }

        # Check if file is an ATA file by name
        lower_name = filename.lower()
        if 'ata_read' in lower_name or 'ata_write' in lower_name:
            meta['is_storage'] = True
            meta['notes'] = 'ATA storage telemetry file (disk read/write). Non-memory operations.'
            try:
                if isinstance(file_source, str):
                    df_raw = pd.read_csv(file_source, header=None)
                else:
                    file_source.seek(0)
                    df_raw = pd.read_csv(file_source, header=None)
                meta['raw_rows'] = len(df_raw)
            except Exception:
                pass
            return pd.DataFrame(), meta

        # Read first few lines to detect header
        if isinstance(file_source, str):
            sample = pd.read_csv(file_source, nrows=5)
            file_source.seek(0) if hasattr(file_source, 'seek') else None
        else:
            file_source.seek(0)
            sample = pd.read_csv(file_source, nrows=5)
            file_source.seek(0)

        # Check column names
        cols_lower = [str(c).lower().strip() for c in sample.columns]
        has_named_header = any(k in cols_lower for k in ['gpa', 'size', 'entropy', 'type', 'unix_time_sec'])

        if has_named_header:
            if isinstance(file_source, str):
                df = pd.read_csv(file_source)
            else:
                file_source.seek(0)
                df = pd.read_csv(file_source)
            col_map = {orig: orig.strip() for orig in df.columns}
            for orig in df.columns:
                lower = orig.lower().strip()
                if lower == 'gpa': col_map[orig] = 'GPA'
                elif lower == 'size': col_map[orig] = 'size'
                elif lower == 'entropy': col_map[orig] = 'entropy'
                elif lower == 'type': col_map[orig] = 'type'
                elif lower == 'unix_time_sec': col_map[orig] = 'UNIX_time_sec'
                elif lower == 'unix_time_ns': col_map[orig] = 'UNIX_time_ns'
            df = df.rename(columns=col_map)
        else:
            num_cols = len(sample.columns)
            if num_cols == 6:
                if isinstance(file_source, str):
                    df = pd.read_csv(file_source, header=None, names=RAW_MEM_COLUMNS)
                else:
                    file_source.seek(0)
                    df = pd.read_csv(file_source, header=None, names=RAW_MEM_COLUMNS)
            elif num_cols == 4:
                if isinstance(file_source, str):
                    df = pd.read_csv(file_source, header=None, names=FEATURE_COLUMNS)
                else:
                    file_source.seek(0)
                    df = pd.read_csv(file_source, header=None, names=FEATURE_COLUMNS)
            else:
                if isinstance(file_source, str):
                    df = pd.read_csv(file_source, header=None)
                else:
                    file_source.seek(0)
                    df = pd.read_csv(file_source, header=None)

        meta['raw_rows'] = len(df)

        required_cols = ['GPA', 'size', 'entropy', 'type']
        if not all(col in df.columns for col in required_cols):
            meta['notes'] = f"Missing required memory columns ({required_cols})."
            return pd.DataFrame(), meta

        meta['is_memory'] = True

        # Drop ground-truth labels if present
        drop_cols = [c for c in ['category', 'is_ransomware'] if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        # Select columns: strictly 4 core features by default, or include timestamps if requested
        if keep_timestamps and 'UNIX_time_sec' in df.columns and 'UNIX_time_ns' in df.columns:
            selected_cols = ['GPA', 'size', 'entropy', 'type', 'UNIX_time_sec', 'UNIX_time_ns']
        else:
            selected_cols = ['GPA', 'size', 'entropy', 'type']

        df = df[selected_cols].copy()

        # Handle conversions
        df['GPA'] = df['GPA'].apply(self._parse_gpa)
        df['size'] = pd.to_numeric(df['size'], errors='coerce')
        df['entropy'] = pd.to_numeric(df['entropy'], errors='coerce')
        df['type'] = pd.to_numeric(df['type'], errors='coerce')
        if 'UNIX_time_sec' in df.columns:
            df['UNIX_time_sec'] = pd.to_numeric(df['UNIX_time_sec'], errors='coerce')
        if 'UNIX_time_ns' in df.columns:
            df['UNIX_time_ns'] = pd.to_numeric(df['UNIX_time_ns'], errors='coerce')

        null_count = int(df.isnull().any(axis=1).sum())
        df = df.dropna().copy()
        meta['dropped_nulls'] = null_count
        meta['cleaned_rows'] = len(df)
        meta['notes'] = f"Extracted {len(df)} memory operations."

        return df, meta

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Return list of available classifier models."""
        models = []
        for key, info in AVAILABLE_MODELS.items():
            item = dict(info)
            if key == 'temporal_transformer':
                item['available'] = (self.temporal_model is not None)
            else:
                item['available'] = (self.baseline_model is not None)
            models.append(item)
        return models

    def process_six_files(
        self,
        file_inputs: List[Tuple[str, Union[str, io.BytesIO, io.StringIO]]],
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process up to 6 input CSV files and make predictions using the chosen model:
        model_name: 'temporal_transformer' (default) or 'baseline'
        """
        # Resolve model selection
        chosen = (model_name or self.default_model or 'temporal_transformer').lower().strip()
        if chosen in ['baseline', 'bagging', 'bagging_ransomware_model', 'baseline_model'] or (self.temporal_model is None):
            active_model_id = 'baseline'
            active_model_info = AVAILABLE_MODELS['baseline']
            keep_ts = False
        else:
            active_model_id = 'temporal_transformer'
            active_model_info = AVAILABLE_MODELS['temporal_transformer']
            keep_ts = True

        file_reports = []
        combined_dfs = []
        file_attribution = []

        total_raw_rows = 0
        total_dropped_nulls = 0
        storage_files_count = 0
        memory_files_count = 0

        for filename, file_source in file_inputs:
            clean_df, report = self.parse_single_csv(file_source, filename, keep_timestamps=keep_ts)
            file_reports.append(report)
            total_raw_rows += report['raw_rows']
            total_dropped_nulls += report.get('dropped_nulls', 0)

            if report.get('is_storage'):
                storage_files_count += 1
            elif report.get('is_memory') and len(clean_df) > 0:
                memory_files_count += 1
                clean_df = clean_df.copy()
                clean_df['source_file'] = filename
                combined_dfs.append(clean_df)
                file_attribution.append((filename, len(clean_df)))

        if not combined_dfs:
            return {
                'status': 'error',
                'message': 'No valid memory operations found among the provided CSV files.',
                'file_reports': file_reports,
                'total_files': len(file_inputs)
            }

        full_df = pd.concat(combined_dfs, ignore_index=True)
        total_memory_ops = len(full_df)

        # Run inference using chosen model
        if active_model_id == 'baseline':
            if self.baseline_model is None or self.baseline_scaler is None:
                raise RuntimeError("Baseline model or scaler is not loaded.")
            X = full_df[FEATURE_COLUMNS]
            X_scaled = self.baseline_scaler.transform(X)
            predictions = self.baseline_model.predict(X_scaled)
            probabilities = self.baseline_model.predict_proba(X_scaled)
        else:
            if self.temporal_model is None:
                raise RuntimeError("Temporal Transformer model is not loaded.")
            predictions = self.temporal_model.predict(full_df)
            probabilities = self.temporal_model.predict_proba(full_df)

        full_df['prediction'] = predictions
        full_df['prob_benign'] = probabilities[:, 0]
        full_df['prob_ransomware'] = probabilities[:, 1]
        full_df['classification'] = np.where(predictions == 1, 'RANSOMWARE', 'BENIGN')
        full_df['confidence'] = np.max(probabilities, axis=1) * 100

        # Summary statistics
        ransomware_ops_count = int((predictions == 1).sum())
        benign_ops_count = int((predictions == 0).sum())
        ransomware_ratio = ransomware_ops_count / total_memory_ops if total_memory_ops > 0 else 0.0
        avg_ransomware_prob = float(probabilities[:, 1].mean())
        avg_confidence = float(full_df['confidence'].mean())

        # Overall verdict & Risk assessment
        overall_verdict = 'RANSOMWARE' if ransomware_ratio >= 0.5 else 'BENIGN'
        
        if ransomware_ratio >= 0.8:
            risk_level = 'CRITICAL'
            risk_color = '#dc2626'
        elif ransomware_ratio >= 0.5:
            risk_level = 'HIGH'
            risk_color = '#ea580c'
        elif ransomware_ratio >= 0.2:
            risk_level = 'MEDIUM'
            risk_color = '#d97706'
        else:
            risk_level = 'LOW'
            risk_color = '#16a34a'

        # Per-file metrics
        file_metrics = []
        for report in file_reports:
            fname = report['filename']
            sub = full_df[full_df['source_file'] == fname]
            if len(sub) > 0:
                r_count = int((sub['prediction'] == 1).sum())
                b_count = int((sub['prediction'] == 0).sum())
                ratio = r_count / len(sub)
                file_metrics.append({
                    'filename': fname,
                    'type': 'Memory Log',
                    'operations': len(sub),
                    'predicted_ransomware': r_count,
                    'predicted_benign': b_count,
                    'ransomware_ratio_pct': round(ratio * 100, 1),
                    'avg_confidence_pct': round(float(sub['confidence'].mean()), 1),
                    'primary_verdict': 'RANSOMWARE' if ratio >= 0.5 else 'BENIGN'
                })
            elif report.get('is_storage'):
                file_metrics.append({
                    'filename': fname,
                    'type': 'ATA Storage Log',
                    'operations': report['raw_rows'],
                    'predicted_ransomware': 'N/A',
                    'predicted_benign': 'N/A',
                    'ransomware_ratio_pct': 'N/A',
                    'avg_confidence_pct': 'N/A',
                    'primary_verdict': 'STORAGE_TELEMETRY'
                })
            else:
                file_metrics.append({
                    'filename': fname,
                    'type': 'Unrecognized / Empty',
                    'operations': 0,
                    'predicted_ransomware': 0,
                    'predicted_benign': 0,
                    'ransomware_ratio_pct': 0.0,
                    'avg_confidence_pct': 0.0,
                    'primary_verdict': 'SKIPPED'
                })

        # Memory type breakdown
        type_breakdown = {}
        for t_val, t_name in TYPE_MAPPING.items():
            count = int((full_df['type'] == t_val).sum())
            if count > 0:
                sub_t = full_df[full_df['type'] == t_val]
                type_breakdown[t_name] = {
                    'count': count,
                    'ransomware_count': int((sub_t['prediction'] == 1).sum()),
                    'ransomware_pct': round(float((sub_t['prediction'] == 1).mean() * 100), 1)
                }

        # Preview rows (first 100 rows formatted for UI)
        preview_rows = []
        preview_df = full_df.head(100)
        for _, row in preview_df.iterrows():
            gpa_int = int(row['GPA'])
            preview_rows.append({
                'source_file': row['source_file'],
                'GPA_hex': f"0x{gpa_int:X}",
                'GPA_dec': gpa_int,
                'size': int(row['size']),
                'entropy': round(float(row['entropy']), 3),
                'type_label': TYPE_MAPPING.get(int(row['type']), f"Type {row['type']}"),
                'classification': row['classification'],
                'confidence': round(float(row['confidence']), 1),
                'prob_ransomware': round(float(row['prob_ransomware']), 3),
                'prob_benign': round(float(row['prob_benign']), 3)
            })

        return {
            'status': 'success',
            'summary': {
                'model_used': active_model_id,
                'model_name': active_model_info['name'],
                'model_f1_score': active_model_info['f1_score'],
                'model_accuracy': active_model_info['accuracy'],
                'overall_verdict': overall_verdict,
                'risk_level': risk_level,
                'risk_color': risk_color,
                'total_files_analyzed': len(file_inputs),
                'memory_files_count': memory_files_count,
                'storage_files_count': storage_files_count,
                'total_memory_operations': total_memory_ops,
                'total_raw_rows': total_raw_rows,
                'total_dropped_nulls': total_dropped_nulls,
                'ransomware_operations_count': ransomware_ops_count,
                'benign_operations_count': benign_ops_count,
                'ransomware_ratio_pct': round(ransomware_ratio * 100, 2),
                'benign_ratio_pct': round((1.0 - ransomware_ratio) * 100, 2),
                'avg_ransomware_probability': round(avg_ransomware_prob, 3),
                'avg_confidence_pct': round(avg_confidence, 1)
            },
            'file_metrics': file_metrics,
            'type_breakdown': type_breakdown,
            'preview_rows': preview_rows,
            '_full_dataframe': full_df
        }
