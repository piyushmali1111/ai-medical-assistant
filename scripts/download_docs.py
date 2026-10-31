import os
import sys
import urllib.request
import ssl
from pathlib import Path

DATA_RAW = Path("data/raw")
DATA_RAW.mkdir(parents=True, exist_ok=True)

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

SOURCES = {
    "who_hypertension_2021.pdf": "https://iris.who.int/server/api/core/bitstreams/f062769d-f075-4a00-87af-0a2106e0bd04/content",
    "who_anaemia_hb_cutoffs_2024.pdf": "https://iris.who.int/server/api/core/bitstreams/f9f74397-1440-478d-a63c-26f29a01552f/content",
    "icmr_stw_diabetes_t2.pdf": "https://icmr.gov.in/icmrobject/uploads/STWs/1726567245_diabetes_mellitus_type_2.pdf",
    "icmr_stw_endocrinology.pdf": "https://icmr.gov.in/icmrobject/uploads/STWs/1733214641_endocrinology.pdf",
    "icmr_stw_cardiology.pdf": "https://www.icmr.gov.in/icmrobject/uploads/STWs/1793170714_cardiologyall.pdf",
    "icmr_stw_obg.pdf": "https://www.icmr.gov.in/icmrobject/uploads/STWs/1771568637_obg_all-2.pdf",
    "icmr_stw_ds_tb_ntep.pdf": "https://icmr.gov.in/icmrobject/uploads/STWs/1725964686_2_ntep_18032022.pdf",
    "ntep_dr_tb_guidelines_2024.pdf": "https://tbcindia.mohfw.gov.in/wp-content/uploads/2025/01/National-Guidelines-for-Management-of-DR-TB_Final.pdf",
}

def download_file(target_name: str, url: str) -> bool:
    dest = DATA_RAW / target_name
    # If the file already exists and is valid PDF (> 300KB)
    if dest.exists() and dest.stat().st_size > 300000:
        # Check if it has %PDF header
        with open(dest, "rb") as f:
            if f.read(4) == b"%PDF":
                print(f"[SKIP] {target_name} already valid ({dest.stat().st_size} bytes)")
                return True

    print(f"[FETCH] Downloading {target_name} from {url}...")
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
            data = resp.read()
            if data[:4] == b"%PDF" or b"%PDF" in data[:1024]:
                dest.write_bytes(data)
                print(f"[SUCCESS] Saved {target_name} ({len(data)} bytes)")
                return True
            else:
                print(f"[ERROR] {target_name} download did not yield PDF header (got {len(data)} bytes)")
                return False
    except Exception as e:
        print(f"[ERROR] Failed to download {target_name}: {e}")
        return False

def main():
    print("Starting document download into data/raw/ ...")
    successes = 0
    for name, url in SOURCES.items():
        if download_file(name, url):
            successes += 1
    print(f"\nCompleted: {successes}/{len(SOURCES)} documents ready in data/raw/")

if __name__ == "__main__":
    main()
