import pathlib
import sys

import numpy as np
import pandas as pd
import pandas.testing as pdt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from DeepSlice.gui.state import DeepSliceAppState


def _sample_predictions() -> pd.DataFrame:
    rows = []
    for idx in range(6):
        rows.append(
            {
                "Filenames": f"brain_s{idx + 1:03d}.png",
                "nr": (idx + 1) * 5,
                "height": 640,
                "width": 1024,
                "ox": 480.0 + idx,
                "oy": 320.0 - (idx * 8.0),
                "oz": 332.0 + (idx * 0.5),
                "ux": -505.0 + (idx * 0.2),
                "uy": 0.72 + (idx * 0.01),
                "uz": 8.5 + (idx * 0.1),
                "vx": -8.0 - (idx * 0.1),
                "vy": 1.30 + (idx * 0.01),
                "vz": -380.0 - (idx * 0.3),
            }
        )
    return pd.DataFrame(rows)


def test_linearity_payload_confidence_outputs_are_valid():
    state = DeepSliceAppState(species="mouse")
    state.predictions = _sample_predictions()

    payload = state.linearity_payload()

    confidence = payload["confidence"]
    assert len(confidence) == len(state.predictions)
    assert np.all(confidence >= 0.0)
    assert np.all(confidence <= 1.0)

    levels = set(payload["confidence_level"].tolist())
    assert levels.issubset({"high", "medium", "low"})

    components = payload["confidence_components"]
    assert set(components.keys()) == {"residual", "angle", "spacing", "center_weight"}


def test_manual_reorder_and_undo_redo_roundtrip():
    state = DeepSliceAppState(species="mouse")
    original = _sample_predictions()
    state.predictions = original.copy()

    ordered_indices = [5, 3, 1, 0, 2, 4]
    state.apply_manual_order(ordered_indices)
    reordered = state.predictions.copy()

    assert reordered.iloc[0]["Filenames"] == original.iloc[5]["Filenames"]
    assert reordered["nr"].astype(int).tolist() == sorted(original["nr"].astype(int).tolist())

    state.undo()
    pdt.assert_frame_equal(
        state.predictions.reset_index(drop=True),
        original.reset_index(drop=True),
        check_dtype=False,
    )

    state.redo()
    pdt.assert_frame_equal(
        state.predictions.reset_index(drop=True),
        reordered.reset_index(drop=True),
        check_dtype=False,
    )
