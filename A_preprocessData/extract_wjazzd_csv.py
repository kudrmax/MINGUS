"""Extract per-solo csv files from wjazzd.db in the format expected by
authorial wjazzDB_csv_to_xml.py.

Authorial script reads from two parallel directories:
    csv_beats/<song_name>.csv   columns include: signature, bar, beat, onset, chord
    csv_melody/<song_name>.csv  columns include: bar, beat, onset, duration, beatdur, pitch

Public jazzomat exports (df_notes.csv / df_solos.csv) do NOT match this layout
(one file across all solos, missing per-bar chord/beat structure), so we
generate the per-solo csv directly from the SQLite DB.

We name files by CMT's song_id format: <melid:03d>_<perf_underscored>_<title_underscored>_Solo
so the same wjazzd_split.json keys route MINGUS xml output too.

4/4 filter and song_id formatting are reused from CMT-fork's dataset_converter
to guarantee identity with the 430-solo set CMT trains on.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

# Reuse CMT-fork's filter + song_id formatter
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CMT_FORK_ROOT = _REPO_ROOT / "models" / "CMT-pytorch"
sys.path.insert(0, str(_CMT_FORK_ROOT))

from jazz.wjazzd.dataset_converter.filter import should_skip  # noqa: E402
from jazz.wjazzd.dataset_converter.wjazzd_to_cmt_midi import (  # noqa: E402
    _solo_id,
    load_solo,
)


def iter_eligible_melids(db_path: Path) -> Iterator[int]:
    """Yield melids of all 4/4-eligible solos in melid order.

    Wraps CMT's filter pattern (see wjazzd_to_cmt_midi.py main()): for each
    melid in solo_info, load metadata via load_solo, apply should_skip, yield
    if not skipped. Returns ~430 melids out of 456 (drops 21 non-4/4 + 5 with
    internal signature change).
    """
    conn = sqlite3.connect(str(db_path))
    try:
        melids = [
            r[0] for r in conn.execute(
                "SELECT melid FROM solo_info ORDER BY melid"
            ).fetchall()
        ]
    finally:
        conn.close()

    for melid in melids:
        solo = load_solo(db_path, melid)
        skip, _reason = should_skip(solo.meta)
        if skip:
            continue
        yield melid


def _read_beats_for_melid(conn: sqlite3.Connection, melid: int) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM beats WHERE melid = ? ORDER BY bar, beat",
        conn,
        params=(melid,),
    )


def _read_melody_for_melid(conn: sqlite3.Connection, melid: int) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM melody WHERE melid = ? ORDER BY onset",
        conn,
        params=(melid,),
    )


def extract_all_solos(
    db_path: Path,
    csv_beats_dir: Path,
    csv_melody_dir: Path,
    limit: Optional[int] = None,
) -> int:
    """Write csv_beats/<song_id>.csv + csv_melody/<song_id>.csv for each
    eligible 4/4 solo in wjazzd.db.

    Returns the number of solos written.
    """
    csv_beats_dir = Path(csv_beats_dir)
    csv_melody_dir = Path(csv_melody_dir)
    csv_beats_dir.mkdir(parents=True, exist_ok=True)
    csv_melody_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    conn = sqlite3.connect(str(db_path))
    try:
        for i, melid in enumerate(iter_eligible_melids(db_path)):
            if limit is not None and i >= limit:
                break

            solo = load_solo(db_path, melid)
            song_id = _solo_id(solo)

            beats_df = _read_beats_for_melid(conn, melid)
            melody_df = _read_melody_for_melid(conn, melid)

            beats_df.to_csv(csv_beats_dir / f"{song_id}.csv", index=False)
            melody_df.to_csv(csv_melody_dir / f"{song_id}.csv", index=False)
            n_written += 1

            if (i + 1) % 25 == 0:
                print(f"[extract] {i + 1} solos written", flush=True)
    finally:
        conn.close()

    print(f"[extract] done: {n_written} solos")
    return n_written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True, help="path to wjazzd.db")
    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="output root; csv_beats/ and csv_melody/ are created inside",
    )
    parser.add_argument("--limit", type=int, default=None, help="for smoke testing")
    args = parser.parse_args(argv)

    extract_all_solos(
        args.db,
        args.out_root / "csv_beats",
        args.out_root / "csv_melody",
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
