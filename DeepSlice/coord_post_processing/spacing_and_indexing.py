import logging
from typing import Union, List, Optional
import numpy as np
import pandas as pd
import re
from pathlib import Path
from .depth_estimation import calculate_brain_center_depths
from .plane_alignment_functions import plane_alignment
from ..metadata import metadata_loader

_logger = logging.getLogger("DeepSlice.spacing_and_indexing")


def trim_mean(arr: np.array, percent: int) -> float:
    """ "
    Calculates the trimmed mean of an array, sourced from:
    https://gist.github.com/StuffbyYuki/6f25f9f2f302cb5c1e82e4481016ccde

    :param arr: the array to calculate the trimmed mean of
    :type arr: np.array
    :param percent: the percentage of values to trim
    :type percent: int
    :return: the trimmed mean
    :rtype: float
    """
    if not 0 <= percent < 100:
        raise ValueError("percent must be in range [0, 100)")

    arr = np.sort(np.asarray(arr, dtype=float))
    n = len(arr)
    if n == 0:
        raise ValueError("arr must contain at least one value")

    k = int(np.floor(n * (float(percent) / 100) / 2))
    if (2 * k) >= n:
        raise ValueError("percent trims all values; provide a smaller percent")
    return float(np.mean(arr[k : n - k]))


def calculate_average_section_thickness(
    section_numbers: List[Union[int, float]],
    section_depth: List[Union[int, float]],
    bad_sections,
    method="weighted",
    species="mouse",
) -> float:
    """
    Calculates the average section thickness for a series of predictions

    :param section_numbers: List of section numbers
    :param section_depth: List of section depths
    :type section_numbers: List[int, float]
    :type section_depth: List[int, float]
    :return: the average section thickness
    :rtype: float
    """
    # inter section number differences
    if bad_sections is not None:
        good_mask = np.logical_not(np.asarray(bad_sections, dtype=bool))
        section_numbers = section_numbers[good_mask].reset_index(drop=True)
        section_depth = section_depth[good_mask]

    if len(section_numbers) < 2:
        raise ValueError(
            "At least two valid sections are required to calculate section thickness"
        )

    number_spacing = section_numbers[:-1].values - section_numbers[1:].values
    if np.any(number_spacing == 0):
        raise ValueError(
            "Duplicate section numbers detected (after dropping bad sections). "
            "Each section number must be unique to compute thickness."
        )
    # inter section depth differences
    depth_spacing = section_depth[:-1] - section_depth[1:]
    # dividing depth spacing by number spacing allows us to control for missing sections
    weighted_accuracy = calculate_weighted_accuracy(
        section_numbers, section_depth, species, None, method
    )
    section_thicknesses = depth_spacing / number_spacing
    if not np.isfinite(section_thicknesses).all():
        raise ValueError(
            "Computed section thicknesses contain non-finite values (NaN/Inf)"
        )
    thickness_weights = np.asarray(weighted_accuracy[1:], dtype=float)
    if len(thickness_weights) != len(section_thicknesses):
        raise ValueError(
            "Internal weighting error: section thicknesses and weights are mismatched"
        )
    if np.all(thickness_weights == 0):
        thickness_weights = np.ones_like(section_thicknesses, dtype=float)

    average_thickness = np.average(section_thicknesses, weights=thickness_weights)
    return average_thickness


def ideal_spacing(
    section_numbers: List[Union[int, float]],
    section_depth: List[Union[int, float]],
    average_thickness: Union[int, float],
    bad_sections: List[bool] = None,
    method="weighted",
    species="mouse",
) -> float:
    """
    Calculates the ideal spacing for a series of predictions

    :param section_numbers: List of section numbers
    :param section_depth: List of section depths
    :param average_thickness: The average section thickness
    :type section_numbers: List[int, float]
    :type section_depth: List[int, float]
    :type average_thickness: int, float
    :return: the ideal spacing
    :rtype: float
    """
    # unaligned voxel position of section numbers (evenly spaced depths)
    index_spaced_depth = section_numbers * average_thickness
    # average distance between the depths and the evenly spaced depths

    weighted_accuracy = calculate_weighted_accuracy(
        section_numbers, section_depth, species, bad_sections, method
    )
    distance_to_ideal = np.average(
        section_depth - index_spaced_depth, weights=weighted_accuracy
    )
    # adjust the evenly spaced depths to minimise their distance to the predicted depths
    ideal_index_spaced_depth = index_spaced_depth + distance_to_ideal
    return ideal_index_spaced_depth


def determine_direction_of_indexing(depth: List[Union[int, float]]) -> str:
    """
    Determines the direction of indexing for a series of predictions

    :param depth: List of depths sorted by section index
    :type depth: List[int, float]
    :return: the direction of indexing
    :rtype: str
    """

    if trim_mean(depth[1:] - depth[:-1], 10) > 0:
        direction = "rostro-caudal"
    else:
        direction = "caudal-rostro"
    return direction


def enforce_section_ordering(predictions, species="mouse"):
    """
    Ensures that the predictions are ordered by section number

    :param predictions: dataframe of predictions
    :type predictions: pandas.DataFrame
    :return: the input dataframe ordered by section number
    :rtype: pandas.DataFrame
    """
    if "nr" not in predictions:
        raise ValueError(
            "No section indexes found, cannot enforce index order. You likely did not run predict() with section_numbers=True"
        )
    predictions = predictions.sort_values(by=["nr"], ascending=True).reset_index(
        drop=True
    )
    if len(predictions) == 1:
        raise ValueError("Only one section found, cannot space according to index")
    else:
        predictions = predictions.reset_index(drop=True)
        depths = calculate_brain_center_depths(predictions, species=species)
        depths = np.array(depths)
        direction = determine_direction_of_indexing(depths)
        predictions["depths"] = depths

        temp = predictions.copy()
        if direction == "caudal-rostro":
            ascending = False
        if direction == "rostro-caudal":
            ascending = True
        if "bad_section" in temp.columns:
            temp_good = temp[temp["bad_section"] == False].copy().reset_index(drop=True)
            temp_good_copy = temp_good.copy()
            temp_good_copy = temp_good_copy.sort_values(
                by=["depths"], ascending=ascending
            ).reset_index(drop=True)
            temp_good["oy"] = temp_good_copy["oy"]

            predictions.loc[predictions["bad_section"] == False, "oy"] = temp_good[
                "oy"
            ].values
        else:
            temp = temp.sort_values(by=["depths"], ascending=ascending).reset_index(
                drop=True
            )

            predictions["oy"] = temp["oy"].values
    return predictions


def space_according_to_index(
    predictions,
    section_thickness=None,
    voxel_size=None,
    suppress=False,
    species="mouse",
):
    """
    Space evenly according to the section indexes, if these indexes do not represent the precise order in which the sections were
    cut, this will lead to less accurate predictions. Section indexes must account for missing sections (ie, if section 3 is missing
    indexes must be 1, 2, 4).

    :param predictions: dataframe of predictions
    :type predictions: pandas.DataFrame
    :return: the input dataframe with evenly spaced sections
    :rtype: pandas.DataFrame
    """
    if voxel_size is None:
        raise ValueError("voxel_size must be specified")
    if section_thickness is not None:
        section_thickness /= voxel_size
    predictions["oy"] = predictions["oy"].astype(float)
    if len(predictions) == 1:
        raise ValueError("Only one section found, cannot space according to index")
    if "nr" not in predictions:
        raise ValueError(
            "No section indexes found, cannot space according to a missing index. You likely did not run predict() with section_numbers=True"
        )
    else:
        if "bad_section" in predictions.columns:
            bad_sections = predictions["bad_section"].values
        else:
            bad_sections = None
        predictions = enforce_section_ordering(predictions, species=species)
        depths = calculate_brain_center_depths(predictions, species=species)
        depths = np.array(depths)
        if section_thickness is None:
            section_thickness = calculate_average_section_thickness(
                predictions["nr"],
                section_depth=depths,
                bad_sections=bad_sections,
                species=species,
            )
            if not suppress:
                _logger.info(
                    "predicted thickness is %sµm", section_thickness * voxel_size
                )
        else:
            if not suppress:
                _logger.info(
                    "specified thickness is %sµm", section_thickness * voxel_size
                )

        calculated_spacing = ideal_spacing(
            predictions["nr"], depths, section_thickness, bad_sections, species=species
        )
        distance_to_ideal = calculated_spacing - depths
        predictions["oy"] = predictions["oy"] + distance_to_ideal
    return predictions


def number_sections(filenames: List[str], legacy=False) -> List[int]:
    """
    returns the section numbers of filenames

    :param filenames: list of filenames
    :type filenames: list[str]
    :return: list of section numbers
    :rtype: list[int]
    """
    filenames = [Path(filename).name for filename in filenames]
    section_numbers = []
    for filename in filenames:
        if not legacy:
            match = re.findall(r"\_s\d+", filename)
            if len(match) == 0:
                raise ValueError(f"No section number found in filename: {filename}")
            if len(match) > 1:
                raise ValueError(
                    "Multiple section numbers found in filename, ensure only one instance of _s### is present, where ### is the section number"
                )
            section_numbers.append(int(match[-1][2:]))
        else:
            digits = re.sub("[^0-9]", "", filename)
            ###this gets the three numbers closest to the end
            tail = digits[-3:]
            if not tail:
                raise ValueError(
                    f"Legacy section numbering requires at least one digit in filename: {filename}"
                )
            section_numbers.append(int(tail))
    return section_numbers


def set_bad_sections_util(
    df: pd.DataFrame, bad_sections: List[str], auto=False, species="mouse"
) -> pd.DataFrame:
    """
    Sets the damaged sections and sections which deepslice may not perform well on for a series of predictions

    :param bad_sections: List of bad sections
    :param df: dataframe of predictions
    :param auto: automatically set bad sections based on if theyre badly positioned relative to their section index
    :type bad_sections: List[int]
    :type df: pandas.DataFrame
    :type auto: bool
    :return: the input dataframe with bad sections labeled as such
    :rtype: pandas.DataFrame
    """

    if bad_sections is None:
        bad_sections = []

    df["bad_section"] = False

    bad_section_indexes = [
        df.Filenames.str.contains(bad_section) for bad_section in bad_sections
    ]
    if np.any([np.sum(x) > 1 for x in bad_section_indexes]):
        raise ValueError(
            "Multiple sections match the same bad section string, make sure each bad section string is unique"
        )
    if len(bad_section_indexes) > 0:
        bad_section_indexes = [np.where(x)[0] for x in bad_section_indexes]
        bad_section_indexes = np.concatenate(bad_section_indexes)
    else:
        bad_section_indexes = np.array([], dtype=int)

    manual_bad_section_mask = np.zeros(len(df), dtype=bool)
    if len(bad_section_indexes) > 0:
        manual_bad_section_mask[bad_section_indexes] = True

    if auto:
        df["depths"] = calculate_brain_center_depths(df, species=species)
        x = df["nr"].values
        y = df["depths"].values
        m, b = np.polyfit(x, y, 1)
        residuals = y - (m * x + b)
        outliers = np.abs(residuals) > 1.5 * np.std(residuals)
        df["bad_section"] = df["bad_section"].astype(bool) | outliers

    df["bad_section"] = df["bad_section"].astype(bool) | manual_bad_section_mask

    flagged_indexes = np.where(df["bad_section"].to_numpy(dtype=bool))[0]
    bad_sections_found = len(flagged_indexes)
    # Tell the user which sections were identified as bad
    if bad_sections_found > 0:
        flagged_names = df.Filenames.iloc[flagged_indexes].tolist()
        _logger.info(
            "%s sections out of %s were marked as bad. They are: %s",
            bad_sections_found,
            len(df),
            flagged_names,
        )
    return df


def calculate_weighted_accuracy(
    section_numbers: List[int],
    depths: List[float],
    species: str,
    bad_sections: List[Optional[bool]] = None,
    method: str = "weighted",
) -> List[float]:
    """
    Calculates the weighted accuracy of a list of section numbers for a given species

    :param section_numbers: List of section numbers
    :param species: Species to calculate accuracy for
    :param bad_sections: List of bad sections
    :param method: Method to use for weighting, defaults to "weighted"
    :type section_numbers: List[int]
    :type species: str
    :type bad_sections: List[Optional[bool]]
    :type method: str
    :return: List of weighted accuracies
    :rtype: List[float]
    """
    min_depth, max_depth = metadata_loader.get_species_depth_range(species)

    if method == "weighted":
        weighted_accuracy = plane_alignment.make_gaussian_weights(max_depth + 1)
        depths = np.array(depths)
        depths[depths < min_depth] = min_depth
        depths[depths > max_depth] = max_depth
        weighted_accuracy = [weighted_accuracy[int(y)] for y in depths]
    elif method is None:
        weighted_accuracy = [1.0 for _ in section_numbers]
    else:
        raise ValueError("method must be one of 'weighted' or None")

    if len(section_numbers) <= 2:
        weighted_accuracy = [1.0 for _ in section_numbers]

    if bad_sections is not None:
        if len(bad_sections) != len(weighted_accuracy):
            raise ValueError("bad_sections and section_numbers must have the same length")
        weighted_accuracy = [
            x if y is False else 0.0 for x, y in zip(weighted_accuracy, bad_sections)
        ]

    return weighted_accuracy
