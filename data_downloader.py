"""
data_downloader.py
──────────────────
Downloads raw EEG waveforms and BIS values from the VitalDB open dataset.

Tracks downloaded:
  • BIS/BIS       – Bispectral Index (target)
  • BIS/EEG1_WAV  – Frontal EEG waveform (input)

All tracks are resampled to 128 Hz using vitaldb's built-in
interpolation so that EEG and BIS are perfectly time-aligned.
"""

import vitaldb
import numpy as np
import os


def download_data(max_cases: int = 6400, data_dir: str = "data") -> None:
    track_names = ["BIS/BIS", "BIS/EEG1_WAV"]

    os.makedirs(data_dir, exist_ok=True)
    print(f"[downloader] target dir : {data_dir}")
    print(f"[downloader] scanning up to {max_cases} cases …")

    n_saved = 0

    for icase in range(1, max_cases + 1):
        save_path = os.path.join(data_dir, f"case_{icase}.npy")

        # Skip already-downloaded files
        if os.path.exists(save_path):
            n_saved += 1
            continue

        try:
            vf   = vitaldb.VitalFile(icase, track_names)
            data = vf.to_numpy(track_names, 1 / 128)   # 128 Hz

            if data is None or np.isnan(data).all():
                continue

            np.save(save_path, data)
            n_saved += 1
            print(f"  case {icase:>5d}  →  {save_path}  (shape {data.shape})")

        except Exception:
            # VitalDB raises for missing case IDs – expected behaviour
            pass

    print(f"[downloader] done – {n_saved} valid case(s) in '{data_dir}/'")


if __name__ == "__main__":
    download_data()
