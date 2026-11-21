"""
test_app.py
Comprehensive test suite for the RanSMAP 6-CSV preprocessor and classifier application.
Tests:
- Classifier pipeline loading & preprocessing
- Hex and string GPA parsing
- Headless and headed CSV handling
- 6-CSV input processing
- Flask API routes (/ , /api/demo, /api/predict, /download/predictions)
"""

import unittest
import os
import io
import json
import warnings

try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
except ImportError:
    pass
warnings.filterwarnings('ignore', category=UserWarning)

import pandas as pd
from classifier_pipeline import RanSMAPPipeline
from app import app

class TestRanSMAPClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = RanSMAPPipeline(
            model_path='bagging_ransomware_model.pkl',
            scaler_path='feature_scaler.pkl'
        )
        cls.client = app.test_client()

    def test_pipeline_initialization(self):
        """Test that model and scaler are loaded properly."""
        self.assertIsNotNone(self.pipeline.model)
        self.assertIsNotNone(self.pipeline.scaler)
        self.assertEqual(list(self.pipeline.scaler.feature_names_in_), ['GPA', 'size', 'entropy', 'type'])

    def test_hex_gpa_parsing(self):
        """Test that hex address strings like 0x80000000 are correctly parsed to float/int."""
        gpa_hex = "0x80000000"
        parsed = self.pipeline._parse_gpa(gpa_hex)
        self.assertEqual(parsed, float(0x80000000))
        self.assertEqual(self.pipeline._parse_gpa(123456), 123456.0)

    def test_parse_single_csv_headless(self):
        """Test parsing a headless 6-column RanSMAP CSV."""
        csv_content = """1674094033,8577689,18608038176,0,-1.0,2
1674093980,244086569,48161696,4096,0.92,4
"""
        sio = io.StringIO(csv_content)
        df, meta = self.pipeline.parse_single_csv(sio, 'mem_read.csv')
        self.assertTrue(meta['is_memory'])
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df.columns), ['GPA', 'size', 'entropy', 'type'])
        self.assertEqual(df['GPA'].iloc[0], 18608038176.0)
        self.assertEqual(df['type'].iloc[0], 2.0)

    def test_parse_single_csv_with_header(self):
        """Test parsing a CSV that already has headers."""
        csv_content = """GPA,size,entropy,type
0x80000000,4096,0.92,4
0x20000000,0,0.12,4
"""
        sio = io.StringIO(csv_content)
        df, meta = self.pipeline.parse_single_csv(sio, 'custom_mem.csv')
        self.assertTrue(meta['is_memory'])
        self.assertEqual(len(df), 2)
        self.assertEqual(df['GPA'].iloc[0], float(0x80000000))

    def test_parse_ata_file(self):
        """Test that ATA disk files are recognized as storage telemetry."""
        csv_content = "1674094033,8577689,2048000,4096,-1,0\n"
        sio = io.StringIO(csv_content)
        df, meta = self.pipeline.parse_single_csv(sio, 'ata_read.csv')
        self.assertTrue(meta['is_storage'])
        self.assertEqual(len(df), 0)

    def test_process_six_files(self):
        """Test end-to-end processing of 6 files."""
        # 4 memory files + 2 ATA files
        mem_row = "1674094000,1000,2147483648,4096,0.85,4\n"
        ata_row = "1674094000,1000,100000,4096,-1,0\n"

        files = [
            ('mem_read.csv', io.StringIO(mem_row * 10)),
            ('mem_write.csv', io.StringIO(mem_row * 10)),
            ('mem_exec.csv', io.StringIO(mem_row * 10)),
            ('mem_readwrite.csv', io.StringIO(mem_row * 10)),
            ('ata_read.csv', io.StringIO(ata_row * 5)),
            ('ata_write.csv', io.StringIO(ata_row * 5)),
        ]

        result = self.pipeline.process_six_files(files)
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['summary']['total_files_analyzed'], 6)
        self.assertEqual(result['summary']['memory_files_count'], 4)
        self.assertEqual(result['summary']['storage_files_count'], 2)
        self.assertEqual(result['summary']['total_memory_operations'], 40)
        self.assertIn(result['summary']['overall_verdict'], ['RANSOMWARE', 'BENIGN'])
        self.assertEqual(len(result['file_metrics']), 6)

    def test_flask_index_route(self):
        """Test GET / renders HTML interface."""
        res = self.client.get('/')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'RanSMAP Ransomware Classifier', res.data)
        self.assertIn(b'Upload 6 CSV Files', res.data)

    def test_flask_demo_benign(self):
        """Test GET /api/demo?type=benign."""
        res = self.client.get('/api/demo?type=benign')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['summary']['total_files_analyzed'], 6)
        self.assertGreater(data['summary']['total_memory_operations'], 0)

    def test_flask_demo_ransomware(self):
        """Test GET /api/demo?type=ransomware."""
        res = self.client.get('/api/demo?type=ransomware')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['summary']['overall_verdict'], 'RANSOMWARE')

    def test_flask_download_predictions(self):
        """Test downloading CSV predictions after running demo."""
        # Run demo first to populate LATEST_PREDICTIONS_DF
        self.client.get('/api/demo?type=benign')
        res = self.client.get('/download/predictions')
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get('Content-Type'), 'text/csv; charset=utf-8')
        self.assertIn(b'GPA', res.data)
        self.assertIn(b'prediction', res.data)

    def test_flask_predict_upload(self):
        """Test POST /api/predict with 6 multipart files."""
        mem_data = b"1674094000,1000,2147483648,4096,0.85,4\n"
        ata_data = b"1674094000,1000,100000,4096,-1,0\n"

        data = {
            'file_0': (io.BytesIO(mem_data * 5), 'mem_read.csv'),
            'file_1': (io.BytesIO(mem_data * 5), 'mem_write.csv'),
            'file_2': (io.BytesIO(mem_data * 5), 'mem_exec.csv'),
            'file_3': (io.BytesIO(mem_data * 5), 'mem_readwrite.csv'),
            'file_4': (io.BytesIO(ata_data * 2), 'ata_read.csv'),
            'file_5': (io.BytesIO(ata_data * 2), 'ata_write.csv'),
        }

        res = self.client.post('/api/predict', data=data, content_type='multipart/form-data')
        self.assertEqual(res.status_code, 200)
        json_data = json.loads(res.data)
        self.assertEqual(json_data['status'], 'success')
        self.assertEqual(json_data['summary']['total_memory_operations'], 20)

    def test_temporal_transformer_model_loaded(self):
        """Test that temporal transformer model is loaded and ready."""
        self.assertIsNotNone(self.pipeline.temporal_model)

    def test_api_models_endpoint(self):
        """Test GET /api/models returns available models."""
        res = self.client.get('/api/models')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data['status'], 'success')
        self.assertGreaterEqual(len(data['models']), 2)
        model_ids = [m['id'] for m in data['models']]
        self.assertIn('temporal_transformer', model_ids)
        self.assertIn('baseline', model_ids)

    def test_model_selection_process_files(self):
        """Test choosing between baseline and temporal_transformer models."""
        mem_row = "1674094000,1000,2147483648,4096,0.85,4\n"
        files = [('mem_read.csv', io.StringIO(mem_row * 5))]

        # Run with baseline
        res_baseline = self.pipeline.process_six_files(files, model_name='baseline')
        self.assertEqual(res_baseline['summary']['model_used'], 'baseline')

        # Run with temporal_transformer
        res_temporal = self.pipeline.process_six_files(files, model_name='temporal_transformer')
        self.assertEqual(res_temporal['summary']['model_used'], 'temporal_transformer')

    def test_flask_predict_model_switching(self):
        """Test POST /api/predict with explicit model choice."""
        mem_data = b"1674094000,1000,2147483648,4096,0.85,4\n"
        data1 = {
            'file_0': (io.BytesIO(mem_data * 5), 'mem_read.csv'),
            'model': 'baseline'
        }
        res1 = self.client.post('/api/predict', data=data1, content_type='multipart/form-data')
        self.assertEqual(res1.status_code, 200)
        json_data1 = json.loads(res1.data)
        self.assertEqual(json_data1['summary']['model_used'], 'baseline')

        data2 = {
            'file_0': (io.BytesIO(mem_data * 5), 'mem_read.csv'),
            'model': 'temporal_transformer'
        }
        res2 = self.client.post('/api/predict', data=data2, content_type='multipart/form-data')
        self.assertEqual(res2.status_code, 200)
        json_data2 = json.loads(res2.data)
        self.assertEqual(json_data2['summary']['model_used'], 'temporal_transformer')

if __name__ == '__main__':
    unittest.main()
