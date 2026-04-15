import numpy as np
from .plane_alignment_functions import plane_alignment


ATLAS_DIMS = {
    "mouse": (528, 320, 456),
    "rat": (512, 1024, 512),
}


def calculate_brain_center_depth(section, species="mouse"):
    """
    Calculates the depth of the brain center for a given section

    :param section: the section coordinates as an array consisting of Oxyz,Uxyz,Vxyz
    :type section: np.array
    :return: the depth of the brain center
    :rtype: float
    """
    if species not in ATLAS_DIMS:
        raise ValueError("species must be one of 'mouse' or 'rat'")

    cross, k = plane_alignment.find_plane_equation(section)
    if np.isclose(cross[1], 0.0):
        raise ValueError(
            "Cannot estimate brain center depth for a plane parallel to the Y axis"
        )

    atlas_x, _, atlas_z = ATLAS_DIMS[species]
    translated_volume = np.array((atlas_x, 0, atlas_z))
    linear_point = (
        ((translated_volume[0] / 2) * cross[0])
        + ((translated_volume[2] / 2) * cross[2])
    ) + k
    depth = -(linear_point / cross[1])
    return depth


def calculate_brain_center_depths(predictions, species="mouse"):
    """
    Calculates the depths of the brain center for a series of predictions

    :param predictions: dataframe of predictions
    :type predictions: pandas.DataFrame
    :return: a list of depths
    :rtype: list[float]
    """
    depths = []
    for prediction in predictions[
        ["ox", "oy", "oz", "ux", "uy", "uz", "vx", "vy", "vz"]
    ].values:
        depths.append(calculate_brain_center_depth(prediction, species=species))
    return depths
