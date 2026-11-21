"""
generate_test_samples.py
Generate realistic 6-CSV sample datasets from mem_all_operations_sample.csv:
1. samples/benign/ (e.g., Firefox / Office / Idle)
2. samples/ransomware/ (e.g., WannaCry / REvil / LockBit)

Each directory contains 6 CSV files matching RanSMAP dataset format:
- mem_read.csv
- mem_write.csv
- mem_exec.csv
- mem_readwrite.csv
- ata_read.csv
- ata_write.csv
"""

import os
import pandas as pd
import numpy as np

def generate_samples():
    print("Loading sample data...")
    # Load sample rows for benign and ransomware categories
    df = pd.read_csv('mem_all_operations_sample.csv', nrows=50000)
    
    benign_cats = ['Firefox', 'Office', 'Idle', 'AESCrypt', 'Zip', 'SDelete']
    ransomware_cats = ['WannaCry', 'REvil', 'LockBit', 'Conti', 'Darkside', 'Ryuk']
    
    benign_df = df[df['category'].isin(benign_cats)].copy()
    ransomware_df = df[df['category'].isin(ransomware_cats)].copy()
    
    # Columns in raw RanSMAP memory CSVs:
    # UNIX_time_sec, UNIX_time_ns, GPA, size, entropy, type (no header in raw files)
    mem_cols = ['UNIX_time_sec', 'UNIX_time_ns', 'GPA', 'size', 'entropy', 'type']
    
    for label, data, out_dir in [
        ('Benign', benign_df, 'samples/benign'),
        ('Ransomware', ransomware_df, 'samples/ransomware')
    ]:
        os.makedirs(out_dir, exist_ok=True)
        print(f"Creating {label} sample set in {out_dir}...")
        
        # Partition into 4 memory operation files
        n = len(data)
        chunk = n // 4
        
        m_read = data.iloc[0:chunk][mem_cols]
        m_write = data.iloc[chunk:2*chunk][mem_cols]
        m_exec = data.iloc[2*chunk:3*chunk][mem_cols]
        m_readwrite = data.iloc[3*chunk:4*chunk][mem_cols]
        
        # Write 4 memory CSVs (without header to simulate raw RanSMAP files)
        m_read.to_csv(os.path.join(out_dir, 'mem_read.csv'), index=False, header=False)
        m_write.to_csv(os.path.join(out_dir, 'mem_write.csv'), index=False, header=False)
        m_exec.to_csv(os.path.join(out_dir, 'mem_exec.csv'), index=False, header=False)
        m_readwrite.to_csv(os.path.join(out_dir, 'mem_readwrite.csv'), index=False, header=False)
        
        # Generate 2 ATA CSV files: UNIX_time_sec, UNIX_time_ns, LBA, size, -1, 0
        ata_read_rows = []
        ata_write_rows = []
        base_time = int(data['UNIX_time_sec'].iloc[0]) if len(data) > 0 else 1674094000
        for i in range(100):
            ata_read_rows.append([base_time + i, 10000000 * i, 2048000 + i * 8, 4096, -1, 0])
            ata_write_rows.append([base_time + i, 10000000 * i + 500, 4096000 + i * 8, 4096, -1, 0])
            
        pd.DataFrame(ata_read_rows).to_csv(os.path.join(out_dir, 'ata_read.csv'), index=False, header=False)
        pd.DataFrame(ata_write_rows).to_csv(os.path.join(out_dir, 'ata_write.csv'), index=False, header=False)
        
        print(f"  Created 6 files in {out_dir}: mem_read, mem_write, mem_exec, mem_readwrite, ata_read, ata_write")

if __name__ == '__main__':
    generate_samples()
