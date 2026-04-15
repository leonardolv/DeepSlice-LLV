import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from DeepSlice.read_and_write.QuickNII_functions import (
    read_QUINT_JSON,
    read_QuickNII_XML,
    write_QUINT_JSON,
    write_QuickNII_XML,
)


ALIGNMENT_COLUMNS = ["ox", "oy", "oz", "ux", "uy", "uz", "vx", "vy", "vz"]


def _sample_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Filenames": ["slice_s001.png", "slice_s002.png", "slice_s003.png"],
            "nr": [1, 2, 3],
            "height": [640, 640, 640],
            "width": [1024, 1024, 1024],
            "ox": [481.1, 482.2, 483.3],
            "oy": [310.0, 300.0, 290.0],
            "oz": [333.0, 334.0, 335.0],
            "ux": [-502.0, -501.5, -503.1],
            "uy": [0.70, 0.72, 0.71],
            "uz": [9.8, 10.1, 9.6],
            "vx": [-8.1, -8.0, -8.3],
            "vy": [1.30, 1.31, 1.29],
            "vz": [-381.0, -380.5, -381.4],
            "markers": [[{"label": "A", "x": 1}], [], [{"label": "B", "x": 4}]],
        }
    )


def test_json_roundtrip_preserves_alignment_and_markers(tmp_path):
    predictions = _sample_predictions()
    base = tmp_path / "roundtrip"

    write_QUINT_JSON(
        df=predictions.copy(),
        filename=str(base),
        aligner="pytest",
        target="ABA_Mouse_CCFv3_2017_25um.cutlas",
    )

    loaded, target = read_QUINT_JSON(str(base) + ".json")

    assert target == "ABA_Mouse_CCFv3_2017_25um.cutlas"
    assert loaded["Filenames"].tolist() == predictions["Filenames"].tolist()
    assert loaded["nr"].astype(int).tolist() == predictions["nr"].tolist()
    np.testing.assert_allclose(
        loaded[ALIGNMENT_COLUMNS].to_numpy(dtype=float),
        predictions[ALIGNMENT_COLUMNS].to_numpy(dtype=float),
        rtol=1e-6,
        atol=1e-6,
    )
    assert loaded["markers"].iloc[0] == predictions["markers"].iloc[0]
    assert loaded["markers"].iloc[2] == predictions["markers"].iloc[2]


def test_xml_roundtrip_preserves_alignment(tmp_path):
    predictions = _sample_predictions()
    base = tmp_path / "roundtrip_xml"

    write_QuickNII_XML(df=predictions.copy(), filename=str(base), aligner="pytest")
    loaded = read_QuickNII_XML(str(base) + ".xml")

    assert loaded["Filenames"].tolist() == predictions["Filenames"].tolist()
    np.testing.assert_allclose(
        loaded[ALIGNMENT_COLUMNS].to_numpy(dtype=float),
        predictions[ALIGNMENT_COLUMNS].to_numpy(dtype=float),
        rtol=1e-6,
        atol=1e-6,
    )
    assert loaded["nr"].astype(int).tolist() == predictions["nr"].tolist()


def test_xml_reader_handles_legacy_unescaped_ampersands(tmp_path):
    xml_path = tmp_path / "legacy.xml"
    xml_path.write_text(
        """<?xml version='1.0' encoding='utf-8'?>
<series first='1' last='1' name='legacy' aligner='pytest'>
  <slice anchoring='ox=1&oy=2&oz=3&ux=4&uy=5&uz=6&vx=7&vy=8&vz=9' filename='legacy_s001.png' height='100' width='200' nr='1'/>
</series>
""",
        encoding="utf-8",
    )

    loaded = read_QuickNII_XML(str(xml_path))
    assert loaded["Filenames"].tolist() == ["legacy_s001.png"]
    np.testing.assert_allclose(
        loaded[ALIGNMENT_COLUMNS].to_numpy(dtype=float),
        np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=float),
        rtol=1e-6,
        atol=1e-6,
    )


def test_xml_writer_assigns_nr_when_missing(tmp_path):
    predictions = _sample_predictions().drop(columns=["nr"])
    base = tmp_path / "xml_without_nr"

    write_QuickNII_XML(df=predictions.copy(), filename=str(base), aligner="pytest")
    loaded = read_QuickNII_XML(str(base) + ".xml")

    assert loaded["nr"].astype(int).tolist() == [1, 2, 3]
    assert loaded["Filenames"].tolist() == predictions["Filenames"].tolist()
