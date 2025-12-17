import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# === Configuration ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TXT_FOLDER = os.path.join(BASE_DIR, 'AUMC_maldi_preprocessed')  #  spectra folder
OUTPUT_FILE = os.path.join(BASE_DIR, 'aumc_peaks200.npy')
UIDS_FILE = os.path.join(BASE_DIR, 'aumc_peaks200_uids.csv')
TOP_N = 200  # number of peaks to retain

# === Create output directory if needed ===
Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

# === Helper: load one spectrum and extract top N peaks ===
def extract_top_peaks(txt_file, top_n=TOP_N):
    try:
        with open(txt_file, 'r') as f:
            lines = f.readlines()

        # Remove comments and empty lines
        data_lines = [line for line in lines if not line.startswith('#') and line.strip() != ""]

        # Skip the header line (column names)
        data_lines = data_lines[1:]

        # Convert to float array
        parsed = []
        for line in data_lines:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            try:
                mz, intensity = float(parts[0]), float(parts[1])
                parsed.append([mz, intensity])
            except ValueError:
                continue

        data = np.array(parsed)
        
        
        

        if data.shape[0] == 0:
            return None

        intensities = data[:, 1]
        if len(intensities) < top_n:
            padded = np.zeros((top_n, 2))
            padded[:len(data)] = data
            return padded

        top_idx = np.argsort(intensities)[-top_n:]
        top_peaks = data[top_idx]
        return top_peaks[np.argsort(top_peaks[:, 0])]

    except Exception as e:
        print(f"Failed to parse {txt_file.name}: {e}")
        return None


# === Main: extract peaks from all .txt files ===
peaks_list = []
uids = []

print(f"🔍 Reading spectra from: {TXT_FOLDER}")
txt_files = sorted(Path(TXT_FOLDER).glob("*.txt"))
print(f"📄 Found {len(txt_files)} .txt files")

for txt_file in tqdm(txt_files):
    spectrum = extract_top_peaks(txt_file)
    if spectrum is not None:
        peaks_list.append(spectrum)
        uids.append(txt_file.stem)

# === Save results ===
np.save(OUTPUT_FILE, np.stack(peaks_list))
pd.Series(uids).to_csv(UIDS_FILE, index=False, header=False)

print(f"✅ Saved {len(peaks_list)} spectra to: {OUTPUT_FILE}")
print(f"🆔 Saved UIDs to: {UIDS_FILE}")
