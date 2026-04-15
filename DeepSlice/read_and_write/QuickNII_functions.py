import io
import json
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


def write_QuickNII_XML(df: pd.DataFrame, filename: str, aligner: str) -> None:
    """
    Converts a pandas DataFrame to a quickNII compatible XML
    """
    df_temp = df.copy()
    if "nr" not in df_temp.columns:
        df_temp["nr"] = np.arange(len(df_temp)) + 1
    df_temp[["ox", "oy", "oz", "ux", "uy", "uz", "vx", "vy", "vz", "nr"]] = df_temp[
        ["ox", "oy", "oz", "ux", "uy", "uz", "vx", "vy", "vz", "nr"]
    ].astype(str)
    out_df = pd.DataFrame(
        {
            "anchoring": "ox="
            + (df_temp.ox)
            + "&oy="
            + (df_temp.oy)
            + "&oz="
            + (df_temp.oz)
            + "&ux="
            + (df_temp.ux)
            + "&uy="
            + (df_temp.uy)
            + "&uz="
            + (df_temp.uz)
            + "&vx="
            + (df_temp.vx)
            + "&vy="
            + (df_temp.vy)
            + "&vz="
            + (df_temp.vz),
            "filename": df_temp.Filenames,
            "height": df_temp.height,
            "width": df_temp.width,
            "nr": df_temp.nr,
        }
    )
    print(f"saving to {filename}.xml")

    first_nr = str(out_df["nr"].iloc[0]) if len(out_df) else ""
    last_nr = str(out_df["nr"].iloc[-1]) if len(out_df) else ""

    root = ET.Element(
        "series",
        attrib={
            "first": first_nr,
            "last": last_nr,
            "name": str(filename),
            "aligner": str(aligner),
        },
    )

    for row in out_df.itertuples(index=False):
        ET.SubElement(
            root,
            "slice",
            attrib={
                "anchoring": str(row.anchoring),
                "filename": str(row.filename),
                "height": str(row.height),
                "width": str(row.width),
                "nr": str(row.nr),
            },
        )

    tree = ET.ElementTree(root)
    tree.write(filename + ".xml", encoding="utf-8", xml_declaration=True)



def read_QuickNII_XML(filename: str) -> pd.DataFrame:
    """
    Converts a QuickNII XML to a pandas dataframe
    """
    # read raw XML and escape bare & characters found in anchoring attrs
    with open(filename, "r", encoding="utf-8") as f:
        raw = f.read()
    # only replace & that are not pre-escaped entities
    raw = re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;)", "&amp;", raw)
    # parse from cleaned string
    df = pd.read_xml(io.StringIO(raw), parser="etree")

    if "anchoring" not in df.columns:
        raise ValueError("Invalid QuickNII XML: missing 'anchoring' field")

    vector_columns = ["ox", "oy", "oz", "ux", "uy", "uz", "vx", "vy", "vz"]
    parsed_anchoring = []
    for anchoring in df["anchoring"].astype(str).values:
        key_values = {}
        for item in anchoring.split("&"):
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            try:
                key_values[key] = float(value)
            except ValueError:
                key_values[key] = np.nan
        parsed_anchoring.append([key_values.get(column, np.nan) for column in vector_columns])

    out_df = pd.DataFrame({"Filenames": df["filename"]})
    out_df[vector_columns] = parsed_anchoring
    for optional_column in ["nr", "height", "width"]:
        if optional_column in df.columns:
            out_df[optional_column] = df[optional_column]
    return out_df



def write_QUINT_JSON(
    df: pd.DataFrame, filename: str, aligner: str, target: str
) -> None:
    """
    Converts a pandas DataFrame to a QUINT (QuickNII, Visualign, & Nutil) compatible JSON
    """
    if "nr" not in df.columns:
        df["nr"] = np.arange(len(df)) + 1
    alignments = df[["ox", "oy", "oz", "ux", "uy", "uz", "vx", "vy", "vz"]].values
    if "markers" in df.columns:
        markers = df.markers.values
    else:
        markers = [[] for _ in range(len(df))]

    def serialize_markers(marker_value):
        if marker_value is None:
            return []
        if isinstance(marker_value, (list, tuple)):
            return list(marker_value)
        if isinstance(marker_value, float) and np.isnan(marker_value):
            return []
        return marker_value

    alignment_metadata = [
        {
            "filename": fn,
            "anchoring": list(alignment),
            "height": h,
            "width": w,
            "nr": nr,
            "markers": serialize_markers(marker),
        }
        for fn, alignment, nr, marker, h, w in zip(
            df.Filenames, alignments, df.nr, markers, df.height, df.width
        )
    ]
    QUINT_json = {
        "name": "",
        "target": target,
        "aligner": aligner,
        "slices": alignment_metadata,
    }
    print(f"saving to {filename}.json")
    with open(filename + ".json", "w") as f:
        json.dump(QUINT_json, f)


def read_QUINT_JSON(filename: str) -> pd.DataFrame:
    """
    Converts a QUINT JSON to a pandas dataframe

    :param json: The path to the QUINT JSON
    :type json: str
    :return: A pandas dataframe
    :rtype: pd.DataFrame
    """
    with open(filename, "r") as f:
        data = json.load(f)
    sections = data["slices"]
    target_volume = data["target"]
    alignments = [
        row["anchoring"] if "anchoring" in row else 9 * [np.nan] for row in sections
    ]
    height = [row["height"] if "height" in row else [] for row in sections]
    width = [row["width"] if "width" in row else [] for row in sections]
    filenames = [row["filename"] if "filename" in row else [] for row in sections]
    section_numbers = [row["nr"] if "nr" in row else [] for row in sections]
    markers = [row["markers"] if "markers" in row else [] for row in sections]
    df = pd.DataFrame({"Filenames": filenames, "nr": section_numbers})
    df[["ox", "oy", "oz", "ux", "uy", "uz", "vx", "vy", "vz"]] = alignments
    df["markers"] = markers
    df["height"] = height
    df["width"] = width
    return df, target_volume
