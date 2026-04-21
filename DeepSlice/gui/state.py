from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import pandas as pd

from ..coord_post_processing import angle_methods, spacing_and_indexing
from ..coord_post_processing.depth_estimation import calculate_brain_center_depths
from ..metadata import metadata_loader

if TYPE_CHECKING:
    from ..main import DSModel

SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
ALIGNMENT_COLUMNS = ["ox", "oy", "oz", "ux", "uy", "uz", "vx", "vy", "vz"]


@dataclass
class DeepSliceAppState:
    is_dirty: bool = False
    species: str = "mouse"
    image_paths: List[str] = field(default_factory=list)
    predictions: Optional[pd.DataFrame] = None
    model: Optional["DSModel"] = None
    section_numbers: bool = True
    legacy_section_numbers: bool = False
    ensemble: Optional[bool] = None
    use_secondary_model: bool = False
    outlier_sigma_threshold: float = 1.5
    confidence_high_threshold: float = 0.75
    confidence_medium_threshold: float = 0.50
    inference_batch_size: int = 8
    detected_indexing_direction: Optional[str] = None
    selected_indexing_direction: Optional[str] = None
    undo_stack: List[pd.DataFrame] = field(default_factory=list)
    redo_stack: List[pd.DataFrame] = field(default_factory=list)
    _config: Optional[dict] = None
    _metadata_path: Optional[str] = None
    _atlas_cache: Dict[str, np.ndarray] = field(default_factory=dict)
    _partial_prediction_candidate: Optional[pd.DataFrame] = None
    _partial_prediction_reason: Optional[str] = None

    def __post_init__(self):
        self._config, self._metadata_path = metadata_loader.load_config()

    def set_species(self, species: str):
        species = species.lower().strip()
        if species not in {"mouse", "rat"}:
            raise ValueError("Species must be one of 'mouse' or 'rat'")
        if self.species != species:
            self.species = species
            self.model = None
            self._atlas_cache = {}

    def set_quality_controls(
        self,
        outlier_sigma: float,
        confidence_medium: float,
        confidence_high: float,
    ):
        outlier_sigma = float(outlier_sigma)
        confidence_medium = float(confidence_medium)
        confidence_high = float(confidence_high)

        if not np.isfinite(outlier_sigma):
            raise ValueError("Outlier sensitivity must be a finite number")
        if not 1.0 <= outlier_sigma <= 3.0:
            raise ValueError("Outlier sensitivity must be between 1.0 and 3.0 sigma")

        if not np.isfinite(confidence_medium) or not np.isfinite(confidence_high):
            raise ValueError("Confidence thresholds must be finite numbers")
        if not 0.05 <= confidence_medium <= 0.95:
            raise ValueError("Medium confidence threshold must be between 0.05 and 0.95")
        if not confidence_medium < confidence_high < 1.0:
            raise ValueError("High confidence threshold must be greater than medium and below 1.0")

        self.outlier_sigma_threshold = outlier_sigma
        self.confidence_medium_threshold = confidence_medium
        self.confidence_high_threshold = confidence_high

    def ensure_model(self, log_callback=None) -> "DSModel":
        from ..main import DSModel

        if self.model is None or self.model.species != self.species:
            download_callback = None
            if log_callback is not None:
                def download_callback(downloaded, total):
                    if total and total > 0:
                        percent = (downloaded / total) * 100.0
                        log_callback(f"Download progress: {percent:.1f}%")
                    else:
                        log_callback(f"Download progress: {downloaded} bytes")

            if log_callback is not None:
                log_callback(f"Initializing {self.species} model")
            self.model = DSModel(
                self.species,
                download_callback=download_callback,
                log_callback=log_callback,
            )
        if self.predictions is not None:
            self.model.predictions = self.predictions.copy()
        return self.model

    def set_images(self, image_paths: List[str]):
        self.is_dirty = True
        self.clear_partial_prediction_candidate()
        deduplicated = []
        seen = set()
        for path in image_paths:
            absolute_path = os.path.abspath(path)
            if absolute_path in seen:
                continue
            if not os.path.isfile(absolute_path):
                continue
            seen.add(absolute_path)
            deduplicated.append(absolute_path)
        self.image_paths = deduplicated

    def add_images(self, image_paths: List[str]):
        self.is_dirty = True
        self.set_images(self.image_paths + image_paths)

    def clear_images(self):
        self.is_dirty = True
        self.image_paths = []
        self.clear_partial_prediction_candidate()

    def remove_image(self, path: str):
        if path in self.image_paths:
            self.image_paths.remove(path)
            self.is_dirty = True

    def has_partial_prediction_candidate(self) -> bool:
        return self._partial_prediction_candidate is not None and len(self._partial_prediction_candidate) > 0

    def partial_prediction_reason(self) -> str:
        return str(self._partial_prediction_reason or "Secondary ensemble pass failed")

    def clear_partial_prediction_candidate(self):
        self._partial_prediction_candidate = None
        self._partial_prediction_reason = None

    def use_partial_prediction_candidate(self, log_callback=None) -> Dict[str, object]:
        if not self.has_partial_prediction_candidate():
            raise ValueError("No partial prediction candidate is available")

        self.is_dirty = True
        self.predictions = self._partial_prediction_candidate.copy()
        self.undo_stack = []
        self.redo_stack = []
        self._sync_model_predictions()

        reason = self.partial_prediction_reason()
        self.clear_partial_prediction_candidate()

        direction = self.detect_indexing_direction()
        self.detected_indexing_direction = direction
        self.selected_indexing_direction = direction

        predicted_thickness_um = None
        if self.section_numbers and len(self.predictions) >= 2:
            try:
                predicted_thickness_um = self.estimate_section_thickness_um()
            except Exception:
                predicted_thickness_um = None

        diagnostics = self._annotate_prediction_diagnostics()

        if log_callback is not None:
            log_callback(
                "Recovered partial prediction result from primary ensemble pass after secondary failure"
            )

        return {
            "slice_count": len(self.predictions),
            "direction": direction,
            "predicted_thickness_um": predicted_thickness_um,
            "partial_recovery": True,
            "partial_reason": reason,
            **diagnostics,
        }

    def image_format_report(self) -> Dict[str, List[str]]:
        supported, unsupported = [], []
        for image_path in self.image_paths:
            extension = os.path.splitext(image_path)[1].lower()
            if extension in SUPPORTED_IMAGE_FORMATS:
                supported.append(image_path)
            else:
                unsupported.append(image_path)
        return {"supported": supported, "unsupported": unsupported}

    def atlas_volume_options(self) -> List[str]:
        volume_config = self._config["volume_paths"][self.species]
        if isinstance(volume_config, dict) and "path" in volume_config and "url" in volume_config:
            return ["MRI"]
        return sorted(list(volume_config.keys()))

    def default_atlas_volume(self) -> str:
        default_volume = self._config["default_volumes"].get(self.species, "")
        if self.species == "rat":
            return "MRI"
        return str(default_volume)

    def _resolve_volume_entry(self, volume_key: Optional[str]):
        volume_config = self._config["volume_paths"][self.species]
        if isinstance(volume_config, dict) and "path" in volume_config and "url" in volume_config:
            return "MRI", volume_config

        available = self.atlas_volume_options()
        selected = (volume_key or self.default_atlas_volume()).lower()
        if selected not in available:
            selected = self.default_atlas_volume().lower()
        return selected, volume_config[selected]

    @staticmethod
    def _normalize_atlas_slice(raw_slice: np.ndarray) -> np.ndarray:
        atlas_slice = np.nan_to_num(raw_slice.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if atlas_slice.size == 0:
            return np.zeros((1, 1), dtype=np.uint8)

        low, high = np.percentile(atlas_slice, [1, 99])
        if high <= low:
            low = float(np.min(atlas_slice))
            high = float(np.max(atlas_slice))
        if high <= low:
            return np.zeros_like(atlas_slice, dtype=np.uint8)

        atlas_slice = np.clip(atlas_slice, low, high)
        atlas_slice = (atlas_slice - low) / (high - low)
        atlas_slice = (atlas_slice * 255.0).astype(np.uint8)
        # Keep native X/Z orientation to preserve aspect ratio relative to histology overlays.
        atlas_slice = np.flipud(atlas_slice)
        return atlas_slice

    def get_atlas_slice(
        self,
        depth_value: Optional[float],
        volume_key: Optional[str] = None,
        progress_callback=None,
        log_callback=None,
    ) -> Dict[str, object]:
        volume_label, volume_entry = self._resolve_volume_entry(volume_key)
        cache_key = f"{self.species}:{volume_label}"

        if cache_key not in self._atlas_cache:
            if log_callback is not None:
                log_callback(f"Loading atlas volume: {self.species}:{volume_label}")

            def download_callback(downloaded, total):
                if progress_callback is not None:
                    progress_callback(int(downloaded), int(total), "atlas-download")

            local_path = metadata_loader.get_data_path(
                volume_entry,
                self._metadata_path,
                download_callback=download_callback,
            )

            try:
                import nibabel as nib
            except ImportError as exc:
                raise RuntimeError(
                    "nibabel is required for atlas previews. Install with 'pip install nibabel'."
                ) from exc

            nifti_img = nib.load(local_path)
            volume = nifti_img.get_fdata(dtype=np.float32)
            if volume.ndim == 4:
                volume = volume[..., 0]
            if volume.ndim != 3:
                raise ValueError(f"Atlas volume must be 3D, got shape {volume.shape}")
            self._atlas_cache[cache_key] = volume
            if log_callback is not None:
                log_callback(f"Atlas loaded with shape {volume.shape}")

        volume = self._atlas_cache[cache_key]
        if depth_value is None:
            if self.predictions is not None and len(self.predictions) > 0:
                depth_value = float(
                    np.median(
                        calculate_brain_center_depths(
                            self.predictions, species=self.species
                        )
                    )
                )
            else:
                depth_value = float(volume.shape[1] / 2.0)

        slice_index = int(np.clip(int(round(depth_value)), 0, volume.shape[1] - 1))
        raw_slice = volume[:, slice_index, :]
        image = self._normalize_atlas_slice(raw_slice)

        if progress_callback is not None:
            progress_callback(1, 1, "atlas-ready")

        return {
            "image": image,
            "slice_index": slice_index,
            "depth": float(depth_value),
            "shape": tuple(int(x) for x in volume.shape),
            "volume_label": volume_label,
        }

    def build_index_report(self, legacy_section_numbers: bool = False) -> Dict[str, object]:
        rows = []
        missing_indices: List[int] = []
        duplicate_indices: List[int] = []
        parse_error = None

        filenames = [os.path.basename(path) for path in self.image_paths]
        if len(filenames) == 0:
            return {
                "rows": rows,
                "missing_indices": missing_indices,
                "duplicate_indices": duplicate_indices,
                "parse_error": parse_error,
            }

        try:
            detected_indices = spacing_and_indexing.number_sections(
                filenames, legacy=legacy_section_numbers
            )
            detected_indices = [int(index) for index in detected_indices]
            counts = pd.Series(detected_indices).value_counts()
            duplicate_indices = sorted(counts[counts > 1].index.tolist())
            unique_values = sorted(set(detected_indices))
            if len(unique_values) > 1:
                full_range = set(range(unique_values[0], unique_values[-1] + 1))
                missing_indices = sorted(full_range - set(unique_values))

            for filename, index in zip(filenames, detected_indices):
                status = "OK"
                if index in duplicate_indices:
                    status = "Duplicate"
                rows.append(
                    {
                        "filename": filename,
                        "detected_index": index,
                        "status": status,
                    }
                )
        except Exception as exc:
            parse_error = str(exc)
            for filename in filenames:
                rows.append(
                    {
                        "filename": filename,
                        "detected_index": "",
                        "status": "Parse Error",
                    }
                )

        return {
            "rows": rows,
            "missing_indices": missing_indices,
            "duplicate_indices": duplicate_indices,
            "parse_error": parse_error,
        }

    def snapshot_predictions(self):
        if self.predictions is None:
            return
        self.undo_stack.append(self.predictions.copy())
        if len(self.undo_stack) > 50:
            self.undo_stack = self.undo_stack[-50:]
        self.redo_stack = []

    def _sync_model_predictions(self):
        if self.model is not None and self.predictions is not None:
            self.model.predictions = self.predictions.copy()

    def _recommended_inference_batch_size(
        self,
        progress_callback=None,
        log_callback=None,
        requested_batch_size: Optional[int] = None,
    ) -> int:
        if requested_batch_size is not None:
            requested = int(requested_batch_size)
            if requested <= 0:
                raise ValueError("inference_batch_size must be a positive integer")
            if log_callback is not None:
                log_callback(f"Using user-configured inference batch size {requested}")
            return requested

        if progress_callback is None:
            return int(max(1, self.inference_batch_size))

        batch_size = 2
        try:


            try:
                import tensorflow as tf
                gpu_count = len(tf.config.list_physical_devices("GPU"))
            except Exception:
                gpu_count = 0
            if gpu_count > 0:
                batch_size = 8
        except Exception:
            batch_size = 2

        if log_callback is not None:
            log_callback(f"Using inference batch size {batch_size} for current runtime")
        return batch_size

    def _annotate_prediction_diagnostics(self) -> Dict[str, object]:
        diagnostics = {
            "out_of_bounds_count": 0,
            "angle_outlier_count": 0,
            "orthogonality_count": 0,
        }
        if self.predictions is None or len(self.predictions) == 0:
            return diagnostics

        try:
            depths = np.asarray(
                calculate_brain_center_depths(self.predictions, species=self.species),
                dtype=float,
            )
            min_depth, max_depth = metadata_loader.get_species_depth_range(self.species)
            out_of_bounds = (depths < float(min_depth)) | (depths > float(max_depth))
            self.predictions["ap_depth"] = depths
            self.predictions["ap_out_of_bounds"] = out_of_bounds.astype(bool)
            diagnostics["out_of_bounds_count"] = int(np.sum(out_of_bounds))
        except Exception:
            self.predictions["ap_out_of_bounds"] = False

        try:
            u = self.predictions[["ux", "uy", "uz"]].to_numpy(dtype=float)
            v = self.predictions[["vx", "vy", "vz"]].to_numpy(dtype=float)
            uv_dot = np.sum(u * v, axis=1)
            uv_norm = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                uv_cosine = np.where(uv_norm > 1e-9, uv_dot / uv_norm, np.nan)
            orthogonality_flags = (~np.isfinite(uv_cosine)) | (np.abs(uv_cosine) > 0.10)

            self.predictions["uv_dot"] = uv_dot
            self.predictions["uv_cosine"] = uv_cosine
            self.predictions["orthogonality_flag"] = orthogonality_flags.astype(bool)
            diagnostics["orthogonality_count"] = int(np.sum(orthogonality_flags))
        except Exception:
            self.predictions["orthogonality_flag"] = False

        try:
            dv_angles, ml_angles = angle_methods.calculate_angles(self.predictions)
            dv_angles = np.asarray(dv_angles, dtype=float)
            ml_angles = np.asarray(ml_angles, dtype=float)
            n = len(dv_angles)
            neighbor_deviation = np.zeros(n, dtype=float)
            if n >= 3:
                for idx in range(n):
                    dv_neighbors = []
                    ml_neighbors = []
                    if idx > 0:
                        dv_neighbors.append(float(dv_angles[idx - 1]))
                        ml_neighbors.append(float(ml_angles[idx - 1]))
                    if idx + 1 < n:
                        dv_neighbors.append(float(dv_angles[idx + 1]))
                        ml_neighbors.append(float(ml_angles[idx + 1]))
                    if len(dv_neighbors) == 0:
                        continue
                    mean_dv = float(np.mean(dv_neighbors))
                    mean_ml = float(np.mean(ml_neighbors))
                    neighbor_deviation[idx] = float(
                        np.sqrt((dv_angles[idx] - mean_dv) ** 2 + (ml_angles[idx] - mean_ml) ** 2)
                    )

            deviation_std = float(np.std(neighbor_deviation))
            if deviation_std > 1e-9:
                angle_outliers = neighbor_deviation > (3.0 * deviation_std)
            else:
                angle_outliers = np.array([False] * n)

            self.predictions["angle_neighbor_deviation"] = neighbor_deviation
            self.predictions["angle_outlier"] = angle_outliers.astype(bool)
            diagnostics["angle_outlier_count"] = int(np.sum(angle_outliers))
        except Exception:
            self.predictions["angle_outlier"] = False

        return diagnostics

    def undo(self):
        self.is_dirty = True
        if len(self.undo_stack) == 0:
            raise ValueError("Nothing to undo")
        if self.predictions is not None:
            self.redo_stack.append(self.predictions.copy())
        self.predictions = self.undo_stack.pop()
        self._sync_model_predictions()

    def redo(self):
        self.is_dirty = True
        if len(self.redo_stack) == 0:
            raise ValueError("Nothing to redo")
        if self.predictions is not None:
            self.undo_stack.append(self.predictions.copy())
        self.predictions = self.redo_stack.pop()
        self._sync_model_predictions()

    def run_prediction(
        self,
        section_numbers: bool,
        legacy_section_numbers: bool,
        ensemble: Optional[bool],
        use_secondary_model: bool,
        inference_batch_size: Optional[int] = None,
        progress_callback=None,
        log_callback=None,
        cancel_check=None,
    ) -> Dict[str, object]:
        self.is_dirty = True
        self.clear_partial_prediction_candidate()
        if len(self.image_paths) == 0:
            raise ValueError("No images selected")

        self.section_numbers = section_numbers
        self.legacy_section_numbers = legacy_section_numbers
        self.ensemble = ensemble
        self.use_secondary_model = use_secondary_model
        if inference_batch_size is not None:
            self.inference_batch_size = int(max(1, inference_batch_size))

        model = self.ensure_model(log_callback=log_callback)
        inference_batch_size = self._recommended_inference_batch_size(
            progress_callback=progress_callback,
            log_callback=log_callback,
            requested_batch_size=self.inference_batch_size,
        )
        if progress_callback is not None:
            progress_callback(0, max(len(self.image_paths), 1), "prepare")
        try:
            model.predict(
                image_list=self.image_paths,
                ensemble=ensemble,
                section_numbers=section_numbers,
                legacy_section_numbers=legacy_section_numbers,
                use_secondary_model=use_secondary_model,
                batch_size=inference_batch_size,
                progress_callback=progress_callback,
                log_callback=log_callback,
                cancel_check=cancel_check,
            )
        except Exception as exc:
            partial_predictions = getattr(exc, "partial_predictions", None)
            if isinstance(partial_predictions, pd.DataFrame) and len(partial_predictions) > 0:
                self._partial_prediction_candidate = partial_predictions.copy()
                self._partial_prediction_reason = str(exc)
                raise RuntimeError(
                    "PARTIAL_PREDICTIONS_AVAILABLE: "
                    + str(exc)
                ) from exc
            raise

        self.predictions = model.predictions.copy()
        self.clear_partial_prediction_candidate()
        self.undo_stack = []
        self.redo_stack = []
        diagnostics = self._annotate_prediction_diagnostics()

        progress_total = max(len(self.predictions), 1)
        if progress_callback is not None:
            progress_callback(max(1, int(round(progress_total * 0.25))), progress_total, "finalize")

        direction = self.detect_indexing_direction()
        self.detected_indexing_direction = direction
        self.selected_indexing_direction = direction

        if progress_callback is not None:
            progress_callback(max(1, int(round(progress_total * 0.60))), progress_total, "finalize")

        predicted_thickness_um = None
        if section_numbers and len(self.predictions) >= 2:
            try:
                predicted_thickness_um = self.estimate_section_thickness_um()
            except Exception:
                predicted_thickness_um = None

        if progress_callback is not None:
            progress_callback(progress_total, progress_total, "finalize")

        return {
            "slice_count": len(self.predictions),
            "direction": direction,
            "predicted_thickness_um": predicted_thickness_um,
            **diagnostics,
        }

    def load_quint(self, filename: str, log_callback=None) -> Dict[str, object]:
        self.is_dirty = False
        self.clear_partial_prediction_candidate()
        model = self.ensure_model(log_callback=log_callback)
        model.load_QUINT(filename)
        self.species = model.species
        self.predictions = model.predictions.copy()
        self.undo_stack = []
        self.redo_stack = []
        self._annotate_prediction_diagnostics()

        self.detected_indexing_direction = self.detect_indexing_direction()
        self.selected_indexing_direction = self.detected_indexing_direction

        marker_count = 0
        if "markers" in self.predictions.columns:
            marker_count = int(
                np.sum(
                    [
                        isinstance(marker, (list, tuple)) and len(marker) > 0
                        for marker in self.predictions["markers"]
                    ]
                )
            )

        return {
            "slice_count": len(self.predictions),
            "species": self.species,
            "marker_count": marker_count,
        }

    def save_predictions(self, filename_without_extension: str, output_format: str):
        model = self.ensure_model()
        if self.predictions is None:
            raise ValueError("No predictions available to save")
        model.predictions = self.predictions.copy()
        model.save_predictions(filename_without_extension, output_format=output_format)

    def set_bad_sections(self, bad_sections: List[str], auto: bool = False):
        self.is_dirty = True
        if self.predictions is None:
            raise ValueError("No predictions available")
        self.snapshot_predictions()
        model = self.ensure_model()
        model.predictions = self.predictions.copy()
        model.set_bad_sections(bad_sections, auto=auto)
        self.predictions = model.predictions.copy()

    def apply_manual_order(self, ordered_row_indices: List[int]):
        self.is_dirty = True
        if self.predictions is None:
            raise ValueError("No predictions available")
        if len(ordered_row_indices) != len(self.predictions):
            raise ValueError("Ordered index list does not match prediction length")

        self.snapshot_predictions()
        reordered = self.predictions.iloc[ordered_row_indices].reset_index(drop=True)

        # Preserve the set of index numbers while mapping them to the user-defined order.
        if "nr" in reordered.columns:
            sorted_indices = sorted(reordered["nr"].astype(int).tolist())
            reordered["nr"] = sorted_indices

        self.predictions = reordered
        self._sync_model_predictions()

    def propagate_angles(self):
        self.is_dirty = True
        if self.predictions is None:
            raise ValueError("No predictions available")
        self.snapshot_predictions()
        model = self.ensure_model()
        model.predictions = self.predictions.copy()
        model.propagate_angles()
        self.predictions = model.predictions.copy()

    def adjust_angles(self, ml_angle: float, dv_angle: float):
        self.is_dirty = True
        if self.predictions is None:
            raise ValueError("No predictions available")
        self.snapshot_predictions()
        model = self.ensure_model()
        model.predictions = self.predictions.copy()
        model.adjust_angles(ml_angle, dv_angle)
        self.predictions = model.predictions.copy()

    def enforce_index_order(self):
        self.is_dirty = True
        if self.predictions is None:
            raise ValueError("No predictions available")
        self.snapshot_predictions()
        model = self.ensure_model()
        model.predictions = self.predictions.copy()
        model.enforce_index_order()
        self.predictions = model.predictions.copy()

    def enforce_index_spacing(self, section_thickness_um: Optional[float] = None):
        self.is_dirty = True
        if self.predictions is None:
            raise ValueError("No predictions available")
        self.snapshot_predictions()
        model = self.ensure_model()
        model.predictions = self.predictions.copy()

        requested_thickness = section_thickness_um
        if requested_thickness is not None and self.selected_indexing_direction is not None:
            if self.selected_indexing_direction == "rostro-caudal":
                requested_thickness = -abs(requested_thickness)
            elif self.selected_indexing_direction == "caudal-rostro":
                requested_thickness = abs(requested_thickness)

        model.enforce_index_spacing(section_thickness=requested_thickness)
        self.predictions = model.predictions.copy()

    def detect_indexing_direction(self) -> Optional[str]:
        if self.predictions is None or "nr" not in self.predictions.columns:
            return None
        if len(self.predictions) < 2:
            return None
        depths = np.array(
            calculate_brain_center_depths(self.predictions, species=self.species)
        )
        return spacing_and_indexing.determine_direction_of_indexing(depths)

    def estimate_section_thickness_um(self) -> float:
        if self.predictions is None:
            raise ValueError("No predictions available")
        if "nr" not in self.predictions.columns:
            raise ValueError("No section numbers available")
        if len(self.predictions) < 2:
            raise ValueError("At least two sections are required")

        bad_sections = None
        if "bad_section" in self.predictions.columns:
            bad_sections = self.predictions["bad_section"].astype(bool).values

        depths = np.array(
            calculate_brain_center_depths(self.predictions, species=self.species)
        )
        thickness_voxels = spacing_and_indexing.calculate_average_section_thickness(
            section_numbers=self.predictions["nr"],
            section_depth=depths,
            bad_sections=bad_sections,
            species=self.species,
        )
        voxel_size = self._config["target_volumes"][self.species]["voxel_size_microns"]
        return float(thickness_voxels * voxel_size)

    @staticmethod
    def _scaled_deviation(values: np.ndarray) -> np.ndarray:
        values = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        if values.size == 0:
            return values

        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        if mad > 1e-9:
            scaled = np.abs(values - median) / (1.4826 * mad)
        else:
            max_value = float(np.max(np.abs(values)))
            if max_value <= 1e-9:
                return np.zeros_like(values, dtype=float)
            scaled = np.abs(values) / max_value
        return np.clip(scaled / 3.0, 0.0, 1.0)

    def linearity_payload(self) -> Dict[str, object]:
        if self.predictions is None:
            raise ValueError("No predictions available")
        if len(self.predictions) == 0:
            raise ValueError("Predictions table is empty")

        predictions = self.predictions.copy()
        if "nr" in predictions.columns:
            x_values = predictions["nr"].astype(float).values
        else:
            x_values = np.arange(1, len(predictions) + 1, dtype=float)

        y_values = np.array(
            calculate_brain_center_depths(predictions, species=self.species),
            dtype=float,
        )
        if len(y_values) > 1:
            slope, intercept = np.polyfit(x_values, y_values, 1)
            trend = slope * x_values + intercept
            residuals = y_values - trend
        else:
            slope, intercept = 0.0, float(y_values[0])
            trend = np.array(y_values)
            residuals = np.zeros_like(y_values)

        section_numbers_for_weighting = (
            predictions["nr"].astype(int).tolist()
            if "nr" in predictions.columns
            else list(range(1, len(predictions) + 1))
        )
        try:
            weights = spacing_and_indexing.calculate_weighted_accuracy(
                section_numbers=section_numbers_for_weighting,
                depths=y_values.tolist(),
                species=self.species,
                bad_sections=None,
                method="weighted",
            )
            weights = np.array(weights, dtype=float)
        except Exception:
            weights = np.ones(len(predictions), dtype=float)

        max_weight = float(np.max(weights)) if len(weights) else 1.0
        if max_weight <= 0:
            normalized_weights = np.ones(len(weights), dtype=float)
        else:
            normalized_weights = weights / max_weight

        dv_angles, ml_angles = angle_methods.calculate_angles(predictions)
        dv_angles = np.array(dv_angles, dtype=float)
        ml_angles = np.array(ml_angles, dtype=float)
        if len(dv_angles) > 0:
            mean_dv, mean_ml = angle_methods.get_mean_angle(
                dv_angles,
                ml_angles,
                method="weighted_mean",
                depths=y_values,
                species=self.species,
            )
            angle_deviation = np.sqrt((dv_angles - mean_dv) ** 2 + (ml_angles - mean_ml) ** 2)
        else:
            angle_deviation = np.zeros(len(predictions), dtype=float)

        spacing_deviation = np.zeros(len(predictions), dtype=float)
        if len(predictions) > 1:
            depth_steps = np.diff(y_values)
            index_steps = np.diff(x_values)
            with np.errstate(divide="ignore", invalid="ignore"):
                normalized_steps = np.where(
                    np.abs(index_steps) > 1e-9,
                    depth_steps / index_steps,
                    np.nan,
                )
            finite_steps = normalized_steps[np.isfinite(normalized_steps)]
            if finite_steps.size > 0:
                median_step = float(np.median(finite_steps))
                step_deviation = np.abs(normalized_steps - median_step)
                for index in range(len(predictions)):
                    adjacent = []
                    if index > 0 and np.isfinite(step_deviation[index - 1]):
                        adjacent.append(float(step_deviation[index - 1]))
                    if index < len(step_deviation) and np.isfinite(step_deviation[index]):
                        adjacent.append(float(step_deviation[index]))
                    spacing_deviation[index] = float(np.mean(adjacent)) if adjacent else 0.0

        residual_component = 1.0 - self._scaled_deviation(np.abs(residuals))
        angle_component = 1.0 - self._scaled_deviation(angle_deviation)
        spacing_component = 1.0 - self._scaled_deviation(spacing_deviation)
        center_component = np.clip(normalized_weights, 0.0, 1.0)

        confidence = (
            (0.40 * residual_component)
            + (0.30 * angle_component)
            + (0.15 * spacing_component)
            + (0.15 * center_component)
        )
        confidence = np.clip(confidence, 0.0, 1.0)

        outlier_sigma = float(np.clip(self.outlier_sigma_threshold, 1.0, 3.0))
        medium_threshold = float(np.clip(self.confidence_medium_threshold, 0.05, 0.95))
        high_threshold = float(
            np.clip(self.confidence_high_threshold, max(medium_threshold + 0.01, 0.06), 0.99)
        )

        bad_sections = np.array([False] * len(predictions))
        if "bad_section" in predictions.columns:
            bad_sections = predictions["bad_section"].astype(bool).values

        confidence[bad_sections] = 0.0
        confidence_level = np.where(
            confidence >= high_threshold,
            "high",
            np.where(confidence >= medium_threshold, "medium", "low"),
        )

        residual_std = float(np.std(residuals))
        if residual_std == 0:
            residual_outliers = np.array([False] * len(residuals))
        else:
            residual_outliers = np.abs(residuals) > (outlier_sigma * residual_std)
        outliers = (confidence < medium_threshold) | residual_outliers | bad_sections

        return {
            "x": x_values,
            "y": y_values,
            "trend": trend,
            "residuals": residuals,
            "outliers": outliers,
            "weights": normalized_weights,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "angle_deviation": angle_deviation,
            "spacing_deviation": spacing_deviation,
            "confidence_components": {
                "residual": residual_component,
                "angle": angle_component,
                "spacing": spacing_component,
                "center_weight": center_component,
            },
            "filenames": predictions["Filenames"].astype(str).values,
            "bad_section": bad_sections,
            "slope": slope,
            "intercept": intercept,
        }

    def summary_metrics(self) -> Dict[str, float]:
        if self.predictions is None:
            return {
                "processed": 0,
                "excluded": 0,
                "mean_angular_deviation": 0.0,
                "slice_count": 0,
            }

        excluded = 0
        if "bad_section" in self.predictions.columns:
            excluded = int(self.predictions["bad_section"].astype(bool).sum())

        processed = int(len(self.predictions) - excluded)

        dv_angles, ml_angles = angle_methods.calculate_angles(self.predictions)
        dv_angles = np.array(dv_angles, dtype=float)
        ml_angles = np.array(ml_angles, dtype=float)

        if len(dv_angles) == 0:
            mean_angular_deviation = 0.0
        else:
            mean_dv = float(np.mean(dv_angles))
            mean_ml = float(np.mean(ml_angles))
            mean_angular_deviation = float(
                np.mean(np.sqrt((dv_angles - mean_dv) ** 2 + (ml_angles - mean_ml) ** 2))
            )

        return {
            "processed": processed,
            "excluded": excluded,
            "mean_angular_deviation": mean_angular_deviation,
            "slice_count": int(len(self.predictions)),
        }

    def to_session_dict(self) -> Dict[str, object]:
        prediction_records = None
        if self.predictions is not None:
            prediction_records = json.loads(self.predictions.to_json(orient="records"))

        return {
            "session_format": "deepslice_gui_v1",
            "species": self.species,
            "image_paths": self.image_paths,
            "section_numbers": self.section_numbers,
            "legacy_section_numbers": self.legacy_section_numbers,
            "ensemble": self.ensemble,
            "use_secondary_model": self.use_secondary_model,
            "inference_batch_size": int(self.inference_batch_size),
            "outlier_sigma_threshold": float(self.outlier_sigma_threshold),
            "confidence_high_threshold": float(self.confidence_high_threshold),
            "confidence_medium_threshold": float(self.confidence_medium_threshold),
            "detected_indexing_direction": self.detected_indexing_direction,
            "selected_indexing_direction": self.selected_indexing_direction,
            "predictions": prediction_records,
        }

    def load_session_dict(self, payload: Dict[str, object]):
        self.is_dirty = False
        self.clear_partial_prediction_candidate()
        self.species = payload.get("species", "mouse")
        self.image_paths = payload.get("image_paths", [])
        self.section_numbers = bool(payload.get("section_numbers", True))
        self.legacy_section_numbers = bool(payload.get("legacy_section_numbers", False))
        self.ensemble = payload.get("ensemble", None)
        self.use_secondary_model = bool(payload.get("use_secondary_model", False))
        try:
            self.inference_batch_size = int(max(1, int(payload.get("inference_batch_size", self.inference_batch_size))))
        except Exception:
            self.inference_batch_size = int(max(1, self.inference_batch_size))

        outlier_sigma = payload.get("outlier_sigma_threshold", self.outlier_sigma_threshold)
        confidence_high = payload.get("confidence_high_threshold", self.confidence_high_threshold)
        confidence_medium = payload.get("confidence_medium_threshold", self.confidence_medium_threshold)
        try:
            self.set_quality_controls(
                outlier_sigma=float(outlier_sigma),
                confidence_medium=float(confidence_medium),
                confidence_high=float(confidence_high),
            )
        except Exception:
            pass

        self.detected_indexing_direction = payload.get("detected_indexing_direction", None)
        self.selected_indexing_direction = payload.get("selected_indexing_direction", None)

        prediction_rows = payload.get("predictions", None)
        if prediction_rows is None:
            self.predictions = None
        else:
            self.predictions = pd.DataFrame(prediction_rows)
            if len(self.predictions) > 0:
                self._annotate_prediction_diagnostics()

        self.undo_stack = []
        self.redo_stack = []
        if self.model is not None and self.model.species != self.species:
            self.model = None
        self._sync_model_predictions()
