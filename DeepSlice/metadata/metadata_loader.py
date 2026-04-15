import hashlib
import json
import os
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
                        if progress_callback is not None:
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
