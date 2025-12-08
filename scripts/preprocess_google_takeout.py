#!/usr/bin/env python
"""
Preprocess Google Takeout MyActivity data into text chunks.

Reads zip archives from data/google-takeout/ and outputs sorted text files
to data/preprocessed/.

Usage:
    uv run python scripts/preprocess_google_takeout.py
    uv run python scripts/preprocess_google_takeout.py --input data/google-takeout/takeout.zip
"""
import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile, is_zipfile

import polars as pl

DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_DIR = DATA_DIR / "google-takeout"
OUTPUT_DIR = DATA_DIR / "preprocessed"

MYACTIVITY_PATH = "Takeout/My Activity/Search/MyActivity.json"


def parse_takeout_zip(archive_path: Path) -> pl.DataFrame:
    """Parse Google Takeout MyActivity from zip."""
    with archive_path.open("rb") as f:
        if not is_zipfile(f):
            raise ValueError(f"{archive_path} is not a valid zip archive")

        with ZipFile(f, "r") as zip_ref:
            if MYACTIVITY_PATH not in zip_ref.namelist():
                raise ValueError(f"{MYACTIVITY_PATH} not found in archive")
            with zip_ref.open(MYACTIVITY_PATH) as zip_f:
                raw_data = json.load(zip_f)

    if not raw_data:
        raise ValueError("Empty JSON data in archive")

    return _process_activity(raw_data)


def _extract_location(entry: dict) -> str | None:
    """Extract location coordinates from entry."""
    location_infos = entry.get("locationInfos", [])
    if not location_infos:
        return None

    # Parse coordinates from URL
    url = location_infos[0].get("url", "")
    match = re.search(r"center=([0-9.-]+),([0-9.-]+)", url)
    if match:
        return f"{match.group(1)},{match.group(2)}"
    return None


def _process_activity(entries: list[dict]) -> pl.DataFrame:
    """Extract all activity with timestamps."""
    processed = []

    for entry in entries:
        title = entry.get("title", "")
        time_str = entry.get("time", "")
        if not time_str or not title:
            continue

        # Parse ISO timestamp
        dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))

        # Determine activity type and extract content
        if title.startswith("Searched for "):
            activity_type = "search"
            content = title.replace("Searched for ", "")
        elif title.startswith("Visited "):
            activity_type = "visit"
            content = title.replace("Visited ", "")
        elif title.startswith("Viewed "):
            activity_type = "view"
            content = title.replace("Viewed ", "")
        else:
            activity_type = "other"
            content = title

        location = _extract_location(entry)
        url = entry.get("titleUrl", "")

        processed.append({
            "datetime": dt,
            "activity_type": activity_type,
            "content": content,
            "url": url,
            "location": location,
            "hour": dt.hour,
            "day_of_week": dt.strftime("%A"),
        })

    df = pl.DataFrame(processed)
    return df.sort("datetime")


def export_as_text_chunks(df: pl.DataFrame, output_path: Path) -> None:
    """Export dataframe as newline-delimited text chunks."""
    chunks = []
    for row in df.iter_rows(named=True):
        loc_str = f" @ {row['location']}" if row['location'] else ""
        chunk = f"[{row['datetime']}] ({row['day_of_week']} {row['hour']:02d}:00){loc_str} [{row['activity_type']}] {row['content']}"
        chunks.append(chunk)

    output_path.write_text("\n\n---\n\n".join(chunks))
    print(f"Wrote {len(chunks)} chunks to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess Google Takeout data")
    parser.add_argument(
        "--input", "-i",
        type=Path,
        help="Specific zip file to process (default: all zips in data/google-takeout/)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for preprocessed files",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        input_files = [args.input]
    else:
        input_files = list(INPUT_DIR.glob("*.zip"))

    if not input_files:
        print(f"No zip files found in {INPUT_DIR}")
        return

    for zip_path in input_files:
        print(f"Processing {zip_path.name}...")
        try:
            df = parse_takeout_zip(zip_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = args.output_dir / f"google_activity_{timestamp}.txt"
            export_as_text_chunks(df, output_path)
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    main()
