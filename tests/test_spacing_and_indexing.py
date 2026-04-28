"""Regression tests for spacing_and_indexing bug fixes."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from DeepSlice.coord_post_processing import spacing_and_indexing


def test_number_sections_legacy_returns_int():
    result = spacing_and_indexing.number_sections(
        ["brain001.png", "brain042.png", "scan_999.tif"], legacy=True
    )
    assert result == [1, 42, 999]
    assert all(isinstance(value, int) for value in result)


def test_number_sections_legacy_rejects_filename_without_digits():
    with pytest.raises(ValueError, match="at least one digit"):
        spacing_and_indexing.number_sections(["brain.png"], legacy=True)


def test_number_sections_modern_returns_int():
    result = spacing_and_indexing.number_sections(
        ["brain_s001.png", "brain_s042.png"], legacy=False
    )
    assert result == [1, 42]
    assert all(isinstance(value, int) for value in result)


def test_calculate_average_section_thickness_rejects_duplicate_section_numbers():
    section_numbers = pd.Series([1, 1, 3])
    section_depth = pd.Series([100.0, 100.0, 300.0])

    with pytest.raises(ValueError, match="Duplicate section numbers"):
        spacing_and_indexing.calculate_average_section_thickness(
            section_numbers,
            section_depth,
            bad_sections=None,
        )


def test_calculate_average_section_thickness_rejects_too_few_sections():
    section_numbers = pd.Series([1])
    section_depth = pd.Series([100.0])

    with pytest.raises(ValueError, match="At least two valid sections"):
        spacing_and_indexing.calculate_average_section_thickness(
            section_numbers,
            section_depth,
            bad_sections=None,
        )


def test_calculate_average_section_thickness_happy_path_no_inf():
    section_numbers = pd.Series([1, 2, 3, 4])
    section_depth = pd.Series([100.0, 110.0, 120.0, 130.0])

    thickness = spacing_and_indexing.calculate_average_section_thickness(
        section_numbers,
        section_depth,
        bad_sections=None,
    )
    assert np.isfinite(thickness)
    assert thickness == pytest.approx(-10.0)
