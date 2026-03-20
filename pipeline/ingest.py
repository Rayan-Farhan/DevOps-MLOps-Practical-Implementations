"""
Stage 1: Data Ingestion

Copies or downloads the diabetes dataset into data/raw/diabetes.csv.

Supported modes (--source):
  local  (default) -- copies from app/model/diabetes.csv or a custom --local-path
  url              -- downloads from --url <URL>

Outputs:
  data/raw/diabetes.csv
  data/raw/ingest_metadata.json
"""

import argparse
import hashlib
import json
import logging
import shutil
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ingest")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_params() -> dict:
    params_path = ROOT / "params.yaml"
    with open(params_path) as f:
        return yaml.safe_load(f)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_metadata(dest: Path, source_type: str, source_location: str) -> None:
    import pandas as pd

    df = pd.read_csv(dest)
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_type": source_type,
        "source_location": source_location,
        "rows": len(df),
        "columns": list(df.columns),
        "sha256": sha256_of(dest),
    }
    meta_path = RAW_DIR / "ingest_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(
        "Metadata written → %s  (rows=%d, sha256=%s…)",
        meta_path,
        metadata["rows"],
        metadata["sha256"][:12],
    )


# ---------------------------------------------------------------------------
# Ingestion modes
# ---------------------------------------------------------------------------

def ingest_local(src: Path, dest: Path) -> None:
    if not src.exists():
        logger.error("Source file not found: %s", src)
        sys.exit(1)
    shutil.copy2(src, dest)
    logger.info("Copied  %s → %s", src, dest)


def ingest_url(url: str, dest: Path) -> None:
    logger.info("Downloading %s …", url)
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        sys.exit(1)
    logger.info("Saved → %s", dest)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    params = load_params()
    default_local = ROOT / params["data"]["source_local_path"]

    parser = argparse.ArgumentParser(description="Ingest diabetes dataset")
    parser.add_argument(
        "--source",
        choices=["local", "url"],
        default="local",
        help="Where to pull data from (default: local)",
    )
    parser.add_argument(
        "--local-path",
        type=Path,
        default=default_local,
        help=f"Path to local CSV (default: {default_local})",
    )
    parser.add_argument(
        "--url",
        default="",
        help="URL to download CSV from (required when --source url)",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / "diabetes.csv"

    if args.source == "local":
        ingest_local(args.local_path, dest)
        write_metadata(dest, "local", str(args.local_path))
    else:
        if not args.url:
            logger.error("--url is required when --source url")
            sys.exit(1)
        ingest_url(args.url, dest)
        write_metadata(dest, "url", args.url)

    logger.info("Ingestion complete.")


if __name__ == "__main__":
    main()
