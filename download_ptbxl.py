"""
Download PTB-XL (PhysioNet) for training: the label tables plus every record at 100 Hz or 500 Hz.

    python download_ptbxl.py              # 100 Hz, about 0.5 GB  -> ptbxl/records100/...
    python download_ptbxl.py --hz 500     # 500 Hz, about 2.6 GB  -> ptbxl/records500/...

Safe to stop and re-run: files already downloaded are skipped.
Data: PTB-XL 1.0.3 (Wagner et al., 2020), PhysioNet, CC BY 4.0.
"""

import argparse
import csv
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "https://physionet.org/files/ptb-xl/1.0.3/"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ptbxl")
TABLES = ["ptbxl_database.csv", "scp_statements.csv"]


def fetch(rel_path, retries=4):
    """Download BASE/rel_path to OUT/rel_path unless it's already there. Returns bytes written (0 if skipped)."""
    dest = os.path.join(OUT, rel_path)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return 0
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(BASE + rel_path, timeout=60) as r:
                data = r.read()
            tmp = dest + ".part"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, dest)  # only a complete file ever gets the real name
            return len(data)
        except Exception as err:
            if attempt == retries - 1:
                raise RuntimeError(f"{rel_path}: {err}") from err
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hz", type=int, choices=[100, 500], default=100)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    for t in TABLES:
        fetch(t)
    column = "filename_lr" if args.hz == 100 else "filename_hr"
    with open(os.path.join(OUT, "ptbxl_database.csv"), encoding="utf-8") as f:
        records = [row[column] for row in csv.DictReader(f)]
    files = [rec + ext for rec in records for ext in (".hea", ".dat")]
    print(f"{len(records)} records at {args.hz} Hz ({len(files)} files) -> {OUT}", flush=True)

    done, written, failed, start = 0, 0, [], time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        for fut in as_completed([pool.submit(fetch, p) for p in files]):
            try:
                written += fut.result()
            except RuntimeError as err:
                failed.append(str(err))
            done += 1
            if done % 2000 == 0 or done == len(files):
                rate = written / max(1e-9, time.time() - start) / 1e6
                print(f"  {done}/{len(files)} files, {written / 1e6:.0f} MB new, {rate:.1f} MB/s", flush=True)

    if failed:
        print(f"{len(failed)} files failed (re-run to retry):", *failed[:10], sep="\n  ")
        sys.exit(1)
    print("Done.")


if __name__ == "__main__":
    main()
