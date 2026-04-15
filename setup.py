from setuptools import find_packages, setup
from pathlib import Path
import subprocess, os

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# if GH action hasn’t replaced it, grab the latest tag
version = os.getenv("DEEPSLICE_VERSION", "{{VERSION_PLACEHOLDER}}")
if version.startswith("{{"):
    try:
        version = (
            subprocess
            .check_output(["git", "describe", "--tags", "--abbrev=0"], cwd=this_directory)
            .decode()
            .strip()
        )
    except Exception:
        version = "0.0.0"

setup(
    name="DeepSlice",
    python_requires=">=3.9,<3.13",
    packages=find_packages(),
    version=version,
    license="GPL-3.0",
    description="A package to align histology to 3D brain atlases",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="DeepSlice Team",
    package_data={
        "DeepSlice": [
            "metadata/volumes/placeholder.txt",
            "metadata/config.json",
            "metadata/weights/*.txt",
        ]
    },
    include_package_data=True,
    author_email="harry.carey@medisin.uio.no",
    url="https://github.com/PolarBean/DeepSlice",
    download_url="https://github.com/PolarBean/DeepSlice/archive/refs/tags/{{VERSION_PLACEHOLDER}}.tar.gz",
    keywords=["histology", "brain", "atlas", "alignment"],
    install_requires=[
        "numpy>=1.24",
        "pandas>=1.5",
        "scikit-image>=0.22",
        "scipy>=1.10",
        "tensorflow>=2.13",
        "h5py>=3.9",
        "requests>=2.31",
        "protobuf>=4.21",
        "lxml>=4.9",
        "Pillow>=10.0",
        "nibabel>=5.2",
        "matplotlib>=3.8",
        "PySide6>=6.6",
        "reportlab>=4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "deepslice-gui=DeepSlice.gui.app:main",
        ]
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python",
    ],
)
