import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import requests


def load_config() -> dict:
    """
    Loads the config file

    :return: the config file
    :rtype: dict
    """
    path = str(Path(__file__).parent) + os.sep
    with open(path + "config.json", "r") as f:
        config = json.loads(f.read())
    return config, path


@lru_cache(maxsize=1)
def get_cached_config() -> dict:
    config, _ = load_config()
    return config


def get_species_depth_range(species: str):
    config = get_cached_config()
    target_volumes = config.get("target_volumes", {})
    if species not in target_volumes:
        raise ValueError(
            f"Invalid species '{species}'. Expected one of {sorted(target_volumes.keys())}"
        )

    depth_range = target_volumes[species].get("depth_range")
    if (
        not isinstance(depth_range, (list, tuple))
        or len(depth_range) != 2
        or depth_range[0] > depth_range[1]
    ):
        raise ValueError(
            f"Invalid depth_range for species '{species}' in metadata/config.json"
        )

    return int(depth_range[0]), int(depth_range[1])


def _file_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    url: str,
    path: str,
    retries: int = 3,
    timeout: int = 60,
    chunk_size: int = 1024 * 1024,
    progress_callback=None,
    expected_sha256: str = None,
):
    """
    Downloads a file from a url to a path

    :param url: the url of the file to download
    :type url: str
    :param path: the path to save the file to
    :type path: str
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    for attempt in range(1, retries + 1):
        try:
            print("Downloading file from " + url + " to " + path)
            with requests.get(
                url, allow_redirects=True, stream=True, timeout=timeout
            ) as response:
                response.raise_for_status()
                total_bytes = int(response.headers.get("content-length", 0))
                downloaded_bytes = 0

                with open(path, "wb") as output_file:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        output_file.write(chunk)
                        downloaded_bytes += len(chunk)
                        if progress_callback is not None and total_bytes > 0:
                            progress_callback(downloaded_bytes, total_bytes)

            if expected_sha256:
                observed_sha256 = _file_sha256(path)
                if observed_sha256.lower() != expected_sha256.lower():
                    raise ValueError(
                        f"Checksum mismatch for {path}. expected={expected_sha256}, observed={observed_sha256}"
                    )
            return
        except Exception as exc:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

            if attempt < retries:
                print(
                    f"Download attempt {attempt}/{retries} failed for {path}: {exc}. Retrying..."
                )
            else:
                raise RuntimeError(
                    f"Failed to download {url} after {retries} attempts"
                ) from exc


def get_data_path(
    url_path_dict,
    path,
    download_callback=None,
    retries: int = 3,
    timeout: int = 60,
):
    """
    If the data is not present, download it from the DeepSlice github. Else return the path to the data.

    :param url_path_dict: a dictionary of a url and path to the data
    :type url_path_dict: dict
    :param path: the path to the DeepSlice metadata directory
    :type path: str
    :return: the path to the data
    :rtype: str
    """
    local_path = path + url_path_dict["path"]
    expected_sha256 = url_path_dict.get("sha256")

    needs_download = not os.path.exists(local_path)

    if not needs_download and expected_sha256:
        existing_sha256 = _file_sha256(local_path)
        if existing_sha256.lower() != expected_sha256.lower():
            print(
                f"Checksum mismatch for existing file {local_path}. Re-downloading."
            )
            needs_download = True

    if needs_download:
        download_file(
            url_path_dict["url"],
            local_path,
            retries=retries,
            timeout=timeout,
            progress_callback=download_callback,
            expected_sha256=expected_sha256,
        )

    return local_path
