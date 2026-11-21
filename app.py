"""
app.py
RanSMAP Ransomware Classification App
Takes 6 CSV files (RanSMAP memory & storage telemetry), applies exact notebook preprocessing,
and makes predictions using either:
1. Temporal-Aware Transformer Model (Recommended, 96.5% F1)
2. Baseline BaggingClassifier Model (63.9% Accuracy)

Usage:
  1. Web App mode:
     python app.py [--port 5000] [--host 0.0.0.0]
     Then open http://localhost:5000

  2. CLI mode:
     python app.py file1.csv file2.csv file3.csv file4.csv file5.csv file6.csv [--model temporal_transformer]
     python app.py --dir samples/benign/ [--model baseline]
     python app.py --sample ransomware [--model temporal_transformer]
"""

import os
import sys
import argparse
import io
import warnings

try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
except ImportError:
    pass
warnings.filterwarnings('ignore', category=UserWarning)

import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file, Response
from classifier_pipeline import RanSMAPPipeline, FEATURE_COLUMNS, TYPE_MAPPING, AVAILABLE_MODELS

app = Flask(__name__)
pipeline = RanSMAPPipeline()

# In-memory storage for the latest prediction results for CSV download
LATEST_PREDICTIONS_DF = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/models', methods=['GET'])
def get_models_endpoint():
    return jsonify({
        'status': 'success',
        'models': pipeline.get_available_models()
    })

@app.route('/api/predict', methods=['POST'])
def predict_endpoint():
    global LATEST_PREDICTIONS_DF
    try:
        model_name = request.form.get('model') or request.args.get('model') or 'temporal_transformer'
        uploaded_files = []
        # Check files from slot inputs (file_0 through file_5)
        for i in range(6):
            key = f'file_{i}'
            if key in request.files:
                f = request.files[key]
                if f and f.filename:
                    content = io.BytesIO(f.read())
                    uploaded_files.append((f.filename, content))

        # Also support arbitrary multiple files under 'files'
        if not uploaded_files and 'files' in request.files:
            for f in request.files.getlist('files'):
                if f and f.filename:
                    content = io.BytesIO(f.read())
                    uploaded_files.append((f.filename, content))

        if not uploaded_files:
            return jsonify({'status': 'error', 'message': 'No CSV files uploaded. Please provide 6 CSV files.'}), 400

        result = pipeline.process_six_files(uploaded_files, model_name=model_name)
        if result['status'] == 'success':
            LATEST_PREDICTIONS_DF = result['_full_dataframe']
            del result['_full_dataframe']
        return jsonify(result)

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/demo', methods=['GET'])
def demo_endpoint():
    global LATEST_PREDICTIONS_DF
    try:
        sample_type = request.args.get('type', 'benign').lower()
        model_name = request.args.get('model') or 'temporal_transformer'
        folder = os.path.join('samples', 'benign' if sample_type == 'benign' else 'ransomware')
        if not os.path.exists(folder):
            return jsonify({'status': 'error', 'message': f'Sample folder not found: {folder}. Run generate_test_samples.py first.'}), 404

        files = []
        for fname in ['mem_read.csv', 'mem_write.csv', 'mem_exec.csv', 'mem_readwrite.csv', 'ata_read.csv', 'ata_write.csv']:
            fpath = os.path.join(folder, fname)
            if os.path.exists(fpath):
                files.append((fname, fpath))

        result = pipeline.process_six_files(files, model_name=model_name)
        if result['status'] == 'success':
            LATEST_PREDICTIONS_DF = result['_full_dataframe']
            del result['_full_dataframe']
        return jsonify(result)

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/download/predictions')
def download_predictions():
    global LATEST_PREDICTIONS_DF
    if LATEST_PREDICTIONS_DF is None or len(LATEST_PREDICTIONS_DF) == 0:
        return "No prediction data available to download.", 404

    csv_data = LATEST_PREDICTIONS_DF.to_csv(index=False)
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=ransmap_predictions.csv"}
    )

def run_cli(file_paths, model_name: str = 'temporal_transformer'):
    """CLI runner for processing files and printing results."""
    print("=" * 70)
    print(" RanSMAP Ransomware Classifier - CLI Mode")
    print("=" * 70)
    print(f"Loading {len(file_paths)} input files with model '{model_name}'...")

    file_inputs = []
    for p in file_paths:
        if not os.path.exists(p):
            print(f"Error: File not found -> {p}")
            sys.exit(1)
        fname = os.path.basename(p)
        file_inputs.append((fname, p))

    result = pipeline.process_six_files(file_inputs, model_name=model_name)
    if result['status'] == 'error':
        print(f"\n[ERROR] {result['message']}")
        sys.exit(1)

    s = result['summary']
    print("\n" + "=" * 70)
    print(f" OVERALL VERDICT:  >>> {s['overall_verdict']} <<<  ({s['risk_level']} RISK)")
    print("=" * 70)
    print(f"  Model Used:                  {s['model_name']} (Accuracy: {s['model_accuracy']}, F1: {s['model_f1_score']})")
    print(f"  Total Files Analyzed:        {s['total_files_analyzed']} ({s['memory_files_count']} memory, {s['storage_files_count']} storage)")
    print(f"  Total Memory Operations:     {s['total_memory_operations']:,}")
    print(f"  Predicted Ransomware Ops:    {s['ransomware_operations_count']:,} ({s['ransomware_ratio_pct']}%)")
    print(f"  Predicted Benign Ops:        {s['benign_operations_count']:,} ({s['benign_ratio_pct']}%)")
    print(f"  Average Model Confidence:    {s['avg_confidence_pct']}%")
    print(f"  Average Ransomware Prob:     {s['avg_ransomware_probability']:.3f}")

    print("\n" + "-" * 70)
    print(f"{'Filename':<22} {'Role':<15} {'Total Ops':<10} {'Malicious':<10} {'Ratio':<8} {'Verdict'}")
    print("-" * 70)
    for f in result['file_metrics']:
        ops_str = f"{f['operations']:,}" if isinstance(f['operations'], int) else str(f['operations'])
        mal_str = f"{f['predicted_ransomware']:,}" if isinstance(f['predicted_ransomware'], int) else str(f['predicted_ransomware'])
        rat_str = f"{f['ransomware_ratio_pct']}%" if isinstance(f['ransomware_ratio_pct'], (int, float)) else str(f['ransomware_ratio_pct'])
        print(f"{f['filename']:<22} {f['type']:<15} {ops_str:<10} {mal_str:<10} {rat_str:<8} {f['primary_verdict']}")

    print("\n" + "-" * 70)
    print(f"{'Page Type':<20} {'Operations':<12} {'Malicious Ops':<15} {'Ransomware Rate'}")
    print("-" * 70)
    for t_name, t_info in result['type_breakdown'].items():
        print(f"{t_name:<20} {t_info['count']:<12,} {t_info['ransomware_count']:<15,} {t_info['ransomware_pct']}%")

    out_file = 'predictions_output.csv'
    result['_full_dataframe'].to_csv(out_file, index=False)
    print(f"\nSaved full predictions to: {out_file} ({len(result['_full_dataframe']):,} rows)")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="RanSMAP 6-CSV Ransomware Classification App")
    parser.add_argument('files', nargs='*', help="Paths to 6 CSV files")
    parser.add_argument('--dir', type=str, help="Directory containing 6 CSV files")
    parser.add_argument('--sample', type=str, choices=['benign', 'ransomware'], help="Run on pre-generated sample (benign or ransomware)")
    parser.add_argument('--model', type=str, choices=['temporal_transformer', 'baseline'], default='temporal_transformer', help="Model to use: temporal_transformer (default) or baseline")
    parser.add_argument('--port', type=int, default=5000, help="Web server port (default: 5000)")
    parser.add_argument('--host', type=str, default='127.0.0.1', help="Web server host (default: 127.0.0.1)")

    args = parser.parse_args()

    # CLI Sample Mode
    if args.sample:
        folder = os.path.join('samples', args.sample)
        files = [
            os.path.join(folder, f)
            for f in ['mem_read.csv', 'mem_write.csv', 'mem_exec.csv', 'mem_readwrite.csv', 'ata_read.csv', 'ata_write.csv']
        ]
        run_cli(files, model_name=args.model)
        return

    # CLI Directory Mode
    if args.dir:
        if not os.path.exists(args.dir):
            print(f"Directory not found: {args.dir}")
            sys.exit(1)
        found_files = [os.path.join(args.dir, f) for f in os.listdir(args.dir) if f.endswith('.csv')]
        if len(found_files) == 0:
            print(f"No CSV files found in {args.dir}")
            sys.exit(1)
        run_cli(sorted(found_files), model_name=args.model)
        return

    # CLI Direct Files Mode
    if args.files:
        run_cli(args.files, model_name=args.model)
        return

    # Default: Run Flask Web Server
    print("=" * 70)
    print(" Starting RanSMAP Ransomware Detector Web App")
    print(f" Open your browser at: http://{args.host}:{args.port}")
    print(" Press Ctrl+C to stop.")
    print("=" * 70)
    app.run(host=args.host, port=args.port, debug=False)

if __name__ == '__main__':
    main()
