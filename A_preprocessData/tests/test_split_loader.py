"""Verify data_preprocessing.py routes songs by song_id from split.json."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from A_preprocessData.data_preprocessing import _route_song_to_bucket, _song_id_from_xml_path


def test_route_song_to_bucket_uses_song_id():
    split = {
        "train": ["001_Foo_Bar_Solo", "002_Baz_Qux_Solo"],
        "eval": ["003_Eval_Tune_Solo"],
        "test": ["004_Test_Tune_Solo"],
    }
    assert _route_song_to_bucket("001_Foo_Bar_Solo", split) == "train"
    assert _route_song_to_bucket("003_Eval_Tune_Solo", split) == "validation"
    assert _route_song_to_bucket("004_Test_Tune_Solo", split) == "test"


def test_route_song_to_bucket_returns_none_for_unknown():
    split = {"train": [], "eval": [], "test": []}
    assert _route_song_to_bucket("999_Unknown_Solo", split) is None


def test_song_id_extracted_from_xml_path():
    p = "A_preprocessData/data/xml/001_Art_Pepper_Anthropology_Solo.xml"
    assert _song_id_from_xml_path(p) == "001_Art_Pepper_Anthropology_Solo"


def test_song_id_extracted_handles_absolute_path():
    p = "/tmp/foo/bar/baz/042_Test_Solo.xml"
    assert _song_id_from_xml_path(p) == "042_Test_Solo"
