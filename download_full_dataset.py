"""
Download the full TPEHGDB v1.0.1 dataset ZIP from PhysioNet and extract
missing record files.

Safety rules:
  - Does NOT overwrite valid existing .hea/.dat files.
  - Does NOT modify RECORDS, metadata, or feature definitions.
  - Logs every action.
  - Validates extracted files with WFDB.
"""

import os
import sys
import time
import zipfile
import shutil
from pathlib import Path
from urllib.request import urlretrieve
from datetime import datetime

# Ensure workspace is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
ZIP_URL = "https://physionet.org/static/published-projects/tpehgdb/term-preterm-ehg-database-1.0.1.zip"
DATA_DIR = BASE_DIR / "data" / "tpehgdb"
TPEHGDB_SUBDIR = DATA_DIR / "tpehgdb"       # Where .hea/.dat files live
DOWNLOAD_DIR = BASE_DIR / "data" / "download_temp"
ZIP_FILENAME = "term-preterm-ehg-database-1.0.1.zip"
REPORT_PATH = BASE_DIR / "outputs" / "features" / "full_zip_download_report.txt"

# ──────────────────────────────────────────────────────────────────────────────


def progress_hook(block_num, block_size, total_size):
    """Report download progress."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded / total_size * 100.0)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  Downloading: {mb_down:.1f} / {mb_total:.1f} MB ({pct:.1f}%)")
        sys.stdout.flush()
    else:
        mb_down = downloaded / (1024 * 1024)
        sys.stdout.write(f"\r  Downloading: {mb_down:.1f} MB (total size unknown)")
        sys.stdout.flush()


def main():
    print("=" * 80)
    print("  TPEHGDB v1.0.1 — FULL DATASET ZIP DOWNLOAD & EXTRACTION")
    print("=" * 80)
    start_time = time.time()

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("  TPEHGDB v1.0.1 — FULL DATASET ZIP DOWNLOAD & EXTRACTION REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Timestamp:     {datetime.now().isoformat()}")
    report_lines.append(f"Source URL:    {ZIP_URL}")
    report_lines.append(f"Target Dir:   {TPEHGDB_SUBDIR}")
    report_lines.append("")

    # ── Step 1: Inventory existing local files ────────────────────────────────
    print("\n[STEP 1] Inventorying existing local .hea/.dat files ...")
    existing_hea = set()
    existing_dat = set()
    if TPEHGDB_SUBDIR.exists():
        for f in TPEHGDB_SUBDIR.iterdir():
            if f.suffix == ".hea" and f.stat().st_size > 0:
                existing_hea.add(f.stem)
            elif f.suffix == ".dat" and f.stat().st_size > 0:
                existing_dat.add(f.stem)
    existing_complete = existing_hea & existing_dat
    print(f"  Existing valid records (both .hea + .dat): {len(existing_complete)}")
    report_lines.append(f"Existing valid records before download: {len(existing_complete)}")
    report_lines.append(f"  Records: {sorted(existing_complete)}")
    report_lines.append("")

    # ── Step 2: Read RECORDS list ─────────────────────────────────────────────
    records_path = DATA_DIR / "RECORDS"
    all_records = []
    with open(records_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                # Extract record stem, e.g. "tpehgdb/tpehg1007" -> "tpehg1007"
                stem = Path(line).name
                all_records.append(stem)
    print(f"  Total records in RECORDS file: {len(all_records)}")
    missing_records = [r for r in all_records if r not in existing_complete]
    print(f"  Missing records to recover: {len(missing_records)}")
    report_lines.append(f"Total records in RECORDS: {len(all_records)}")
    report_lines.append(f"Missing records: {len(missing_records)}")
    report_lines.append("")

    # ── Step 3: Download the ZIP ──────────────────────────────────────────────
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DOWNLOAD_DIR / ZIP_FILENAME

    if zip_path.exists() and zip_path.stat().st_size > 200_000_000:
        print(f"\n[STEP 2] ZIP already downloaded ({zip_path.stat().st_size / 1e6:.1f} MB). Reusing.")
        report_lines.append(f"ZIP file: Reused existing download ({zip_path.stat().st_size} bytes)")
    else:
        print(f"\n[STEP 2] Downloading ZIP from PhysioNet (~245 MB) ...")
        print(f"  URL: {ZIP_URL}")
        download_start = time.time()
        try:
            urlretrieve(ZIP_URL, str(zip_path), reporthook=progress_hook)
            print()  # newline after progress bar
            download_time = time.time() - download_start
            zip_size = zip_path.stat().st_size
            print(f"  Download complete: {zip_size / 1e6:.1f} MB in {download_time:.1f}s")
            report_lines.append(f"ZIP downloaded: {zip_size} bytes in {download_time:.1f}s")
        except Exception as e:
            msg = f"DOWNLOAD FAILED: {e}"
            print(f"\n  ERROR: {msg}")
            report_lines.append(msg)
            _write_report(report_lines)
            sys.exit(1)

    # ── Step 4: Extract missing records from ZIP ──────────────────────────────
    print(f"\n[STEP 3] Extracting missing records from ZIP ...")
    TPEHGDB_SUBDIR.mkdir(parents=True, exist_ok=True)

    extracted_count = 0
    skipped_count = 0
    errors = []

    try:
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            # List all members to find the directory structure
            all_members = zf.namelist()
            print(f"  Total files in ZIP: {len(all_members)}")
            report_lines.append(f"Total files in ZIP archive: {len(all_members)}")

            # Find .hea and .dat files in the zip
            signal_files = [m for m in all_members if m.endswith(".hea") or m.endswith(".dat")]
            print(f"  Signal files (.hea + .dat) in ZIP: {len(signal_files)}")
            report_lines.append(f"Signal files in ZIP: {len(signal_files)}")
            report_lines.append("")

            for member_path in signal_files:
                member_name = Path(member_path).name  # e.g., "tpehg1007.hea"
                member_stem = Path(member_path).stem   # e.g., "tpehg1007"
                member_ext = Path(member_path).suffix   # e.g., ".hea"

                dest_path = TPEHGDB_SUBDIR / member_name

                # Safety: do not overwrite existing valid files
                if dest_path.exists() and dest_path.stat().st_size > 0:
                    skipped_count += 1
                    continue

                # Extract to temp then move
                try:
                    data = zf.read(member_path)
                    if len(data) == 0:
                        errors.append(f"Empty file in ZIP: {member_path}")
                        continue
                    dest_path.write_bytes(data)
                    extracted_count += 1
                except Exception as e:
                    errors.append(f"Error extracting {member_path}: {e}")

    except zipfile.BadZipFile as e:
        msg = f"BAD ZIP FILE: {e}"
        print(f"  ERROR: {msg}")
        report_lines.append(msg)
        _write_report(report_lines)
        sys.exit(1)

    print(f"  Files extracted:  {extracted_count}")
    print(f"  Files skipped (existing): {skipped_count}")
    print(f"  Extraction errors: {len(errors)}")
    report_lines.append(f"Files extracted: {extracted_count}")
    report_lines.append(f"Files skipped (already existed): {skipped_count}")
    report_lines.append(f"Extraction errors: {len(errors)}")
    if errors:
        report_lines.append("Errors:")
        for err in errors[:50]:
            report_lines.append(f"  - {err}")
    report_lines.append("")

    # ── Step 5: Post-extraction inventory ─────────────────────────────────────
    print(f"\n[STEP 4] Post-extraction inventory ...")
    post_hea = set()
    post_dat = set()
    for f in TPEHGDB_SUBDIR.iterdir():
        if f.suffix == ".hea" and f.stat().st_size > 0:
            post_hea.add(f.stem)
        elif f.suffix == ".dat" and f.stat().st_size > 0:
            post_dat.add(f.stem)
    post_complete = post_hea & post_dat
    newly_recovered = post_complete - existing_complete
    still_missing = [r for r in all_records if r not in post_complete]

    print(f"  Total valid records (both .hea + .dat): {len(post_complete)}")
    print(f"  Newly recovered records:  {len(newly_recovered)}")
    print(f"  Still missing records:    {len(still_missing)}")
    report_lines.append(f"Post-extraction valid records: {len(post_complete)}")
    report_lines.append(f"Newly recovered: {len(newly_recovered)}")
    report_lines.append(f"Still missing: {len(still_missing)}")
    if still_missing:
        report_lines.append(f"  Missing record IDs: {still_missing[:20]} ...")
    report_lines.append("")

    # ── Step 6: WFDB validation of newly recovered records ────────────────────
    print(f"\n[STEP 5] Validating newly recovered records with WFDB ...")
    try:
        import wfdb
    except ImportError:
        print("  WARNING: wfdb not installed. Skipping WFDB validation.")
        report_lines.append("WFDB validation skipped (not installed).")
    else:
        wfdb_pass = 0
        wfdb_fail = 0
        wfdb_errors_list = []
        for rec_stem in sorted(newly_recovered):
            rec_path = str(TPEHGDB_SUBDIR / rec_stem)
            try:
                record = wfdb.rdrecord(rec_path)
                if record.p_signal is not None and record.p_signal.shape[0] > 0:
                    wfdb_pass += 1
                else:
                    wfdb_fail += 1
                    wfdb_errors_list.append(f"{rec_stem}: Empty signal array")
            except Exception as e:
                wfdb_fail += 1
                wfdb_errors_list.append(f"{rec_stem}: {e}")

        print(f"  WFDB validation passed: {wfdb_pass}")
        print(f"  WFDB validation failed: {wfdb_fail}")
        report_lines.append(f"WFDB validation passed: {wfdb_pass}")
        report_lines.append(f"WFDB validation failed: {wfdb_fail}")
        if wfdb_errors_list:
            report_lines.append("WFDB failures:")
            for err in wfdb_errors_list[:50]:
                report_lines.append(f"  - {err}")
        report_lines.append("")

    # ── Step 7: Also validate metadata coverage ───────────────────────────────
    print(f"\n[STEP 6] Checking metadata label coverage ...")
    smr_path = DATA_DIR / "tpehgdb.smr"
    labeled_records = set()
    if smr_path.exists():
        try:
            with open(smr_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        rec_name = parts[0].strip()
                        if rec_name.startswith("tpehg"):
                            labeled_records.add(rec_name)
        except Exception:
            pass
    records_with_labels = post_complete & labeled_records
    records_without_labels = post_complete - labeled_records
    print(f"  Records with metadata labels: {len(records_with_labels)}")
    print(f"  Records WITHOUT labels:       {len(records_without_labels)}")
    report_lines.append(f"Records with metadata labels: {len(records_with_labels)}")
    report_lines.append(f"Records without labels: {len(records_without_labels)}")
    if records_without_labels:
        report_lines.append(f"  Unlabeled: {sorted(records_without_labels)[:20]}")
    report_lines.append("")

    # ── Timing ────────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\n  Total elapsed time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    report_lines.append(f"Total elapsed time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("END OF DOWNLOAD & EXTRACTION REPORT")
    report_lines.append("=" * 80)

    # ── Write report ──────────────────────────────────────────────────────────
    _write_report(report_lines)
    print(f"\n  Report saved to: {REPORT_PATH}")
    print("=" * 80)
    print("  DOWNLOAD & EXTRACTION COMPLETE")
    print("=" * 80)


def _write_report(lines):
    """Write report lines to the output file."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
