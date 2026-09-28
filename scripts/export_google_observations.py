"""Export the private Google observation log as three aligned wide CSV files.

    python scripts/export_google_observations.py [--secrets .streamlit/secrets.toml] [--out exports]

Needs the service-account credentials and the [google_observations] section
(see README), so only someone holding them can export. Writes
google_distance_m.csv, google_duration_s.csv and google_speed_kmh.csv to
--out (git-ignored by default). Each file: timestamp_utc (ISO 8601, UTC),
then one column per Brussels segment id in ascending order, one row per
stored Google batch; cells without a returned value are empty.

The files contain Google data that may not be redistributed: do not commit
or share them.
"""
from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.google_routes import observations  # noqa: E402


def brussels_segment_ids() -> list[str]:
    import geopandas as gpd

    ids = gpd.read_file(REPO_ROOT / "data" / "Brussels_map_6km.gpkg")["id"]
    return [str(value) for value in sorted(ids.astype(int).unique())]


def open_worksheet(secrets_path: Path):
    import gspread

    secrets = tomllib.loads(secrets_path.read_text(encoding="utf-8"))
    config = secrets["google_observations"]
    return observations._open_worksheet(
        dict(secrets["gcp_service_account"]),
        config["spreadsheet_id"],
        config.get("worksheet_name", "observations"),
        gspread,
    )


def export(worksheet, out_dir: Path) -> list[Path]:
    exports = observations.build_wide_exports(observations.read_observations(worksheet), brussels_segment_ids())
    return observations.write_exports(exports, out_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--secrets", type=Path, default=REPO_ROOT / ".streamlit" / "secrets.toml")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "exports")
    args = parser.parse_args(argv)

    for path in export(open_worksheet(args.secrets), args.out):
        rows = sum(1 for _ in path.open(encoding="utf-8")) - 1
        print(f"{path}: {rows} batch row(s)")


if __name__ == "__main__":
    main()
