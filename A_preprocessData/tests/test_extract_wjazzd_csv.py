"""Tests for wjazzd.db → MINGUS-csv extractor.

Pin down: (a) per-solo csv pairs are written, (b) column sets match what
authorial wjazzDB_csv_to_xml.py reads, (c) filenames are CMT-compatible
(<melid:03d>_<perf>_<title>_Solo), (d) only 4/4 solos are extracted (430).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from A_preprocessData.extract_wjazzd_csv import (
    extract_all_solos,
    iter_eligible_melids,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
WJAZZD_DB = REPO_ROOT / "datasets" / "wjazzd" / "wjazzd.db"


def test_iter_eligible_melids_count():
    n = sum(1 for _ in iter_eligible_melids(WJAZZD_DB))
    assert n == 430


def test_extract_writes_per_solo_pair(tmp_path):
    csv_beats = tmp_path / "csv_beats"
    csv_melody = tmp_path / "csv_melody"
    extract_all_solos(WJAZZD_DB, csv_beats, csv_melody, limit=3)
    assert len(list(csv_beats.glob("*.csv"))) == 3
    assert len(list(csv_melody.glob("*.csv"))) == 3
    assert {p.name for p in csv_beats.iterdir()} == {p.name for p in csv_melody.iterdir()}


def test_csv_beats_has_required_columns(tmp_path):
    csv_beats = tmp_path / "csv_beats"
    csv_melody = tmp_path / "csv_melody"
    extract_all_solos(WJAZZD_DB, csv_beats, csv_melody, limit=1)
    df = pd.read_csv(next(csv_beats.iterdir()))
    required = {"signature", "bar", "beat", "onset", "chord"}
    assert required.issubset(df.columns)


def test_csv_beats_signature_is_4_4_in_first_row(tmp_path):
    csv_beats = tmp_path / "csv_beats"
    csv_melody = tmp_path / "csv_melody"
    extract_all_solos(WJAZZD_DB, csv_beats, csv_melody, limit=1)
    df = pd.read_csv(next(csv_beats.iterdir()))
    first_signature = df[df["signature"].notnull()]["signature"].values[0]
    assert first_signature == "4/4"


def test_csv_melody_has_required_columns(tmp_path):
    csv_beats = tmp_path / "csv_beats"
    csv_melody = tmp_path / "csv_melody"
    extract_all_solos(WJAZZD_DB, csv_beats, csv_melody, limit=1)
    df = pd.read_csv(next(csv_melody.iterdir()))
    required = {"bar", "beat", "onset", "duration", "beatdur", "pitch"}
    assert required.issubset(df.columns)


def test_csv_filename_matches_cmt_song_id_format(tmp_path):
    csv_beats = tmp_path / "csv_beats"
    csv_melody = tmp_path / "csv_melody"
    extract_all_solos(WJAZZD_DB, csv_beats, csv_melody, limit=1)
    name = next(csv_beats.iterdir()).stem
    parts = name.split("_")
    assert parts[-1] == "Solo"
    assert parts[0].isdigit() and len(parts[0]) == 3
