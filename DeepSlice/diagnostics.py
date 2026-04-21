"""
Structured diagnostic logging for DeepSlice.

Each event written by log_issue() is JSON serializable and follows a schema
that an AI coding agent can parse to apply safe fixes semi-automatically.
"""

from __future__ import annotations

import json
import logging
import os
import traceback
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# In-memory accumulator, queryable by tooling/agents.
ISSUES: list[dict] = []

_logger = logging.getLogger("DeepSlice.diagnostics")


RULE_CATALOGUE: dict[str, dict] = {
    "DS-001": {
        "title": "gray_scale() hard-coded output shape",
        "file": "DeepSlice/neural_network/neural_network.py",
        "suggested_fix": {
            "patch": (
                "- img = rgb2gray(img).reshape(299, 299, 1)\n"
                "+ h, w = img.shape[:2]\n"
                "+ img = rgb2gray(img).reshape(h, w, 1)"
            ),
            "effort": "trivial",
        },
    },
    "DS-002": {
        "title": "Image size list misalignment after format filtering",
        "file": "DeepSlice/neural_network/neural_network.py",
        "suggested_fix": {
            "patch": (
                "Ensure sizes is built after VALID_IMAGE_FORMATS filtering and remains parallel "
                "to the generator filename list."
            ),
            "effort": "low",
        },
    },
    "DS-003": {
        "title": "number_sections() uses hard-coded Windows backslash separator",
        "file": "DeepSlice/coord_post_processing/spacing_and_indexing.py",
        "suggested_fix": {
            "patch": (
                "- filenames = [filename.split('\\\\')[-1] for filename in filenames]\n"
                "+ from pathlib import Path\n"
                "+ filenames = [Path(filename).name for filename in filenames]"
            ),
            "effort": "trivial",
        },
    },
    "DS-004": {
        "title": "space_according_to_index() falsy check can ignore thickness=0 intent",
        "file": "DeepSlice/coord_post_processing/spacing_and_indexing.py",
        "suggested_fix": {
            "patch": "- if not section_thickness:\n+ if section_thickness is None:",
            "effort": "trivial",
        },
    },
    "DS-005": {
        "title": "load_QUINT() species switch bypasses model log callback",
        "file": "DeepSlice/main.py",
        "suggested_fix": {
            "patch": (
                "Replace print() calls with self._log() and normalize message casing."
            ),
            "effort": "trivial",
        },
    },
    "DS-006": {
        "title": "initialise_network() uses training=True for rat inference",
        "file": "DeepSlice/neural_network/neural_network.py",
        "suggested_fix": {
            "patch": (
                "- base_model_layer = base_model(inputs, training=True)\n"
                "+ base_model_layer = base_model(inputs, training=False)"
            ),
            "effort": "trivial",
        },
    },
    "DS-007": {
        "title": "load_xception_weights() uses fragile layer index assumptions",
        "file": "DeepSlice/neural_network/neural_network.py",
        "status": "resolved",
        "resolved_in": "rat-prediction-production refactor",
        "suggested_fix": {
            "patch": "Resolved: layers are now resolved by name via model.get_layer().",
            "effort": "medium",
        },
    },
    "DS-008": {
        "title": "propagate_angles() convergence loop has no non-convergence warning",
        "file": "DeepSlice/main.py",
        "suggested_fix": {
            "patch": "Add a for/else warning path with structured diagnostics.",
            "effort": "trivial",
        },
    },
    "DS-009": {
        "title": "download_file() progress callback receives total_bytes=0",
        "file": "DeepSlice/metadata/metadata_loader.py",
        "suggested_fix": {
            "patch": (
                "- if progress_callback is not None:\n"
                "+ if progress_callback is not None and total_bytes > 0:"
            ),
            "effort": "trivial",
        },
    },
    "DS-010": {
        "title": "set_bad_sections_util() may leave NaN bad_section values",
        "file": "DeepSlice/coord_post_processing/spacing_and_indexing.py",
        "suggested_fix": {
            "patch": "Initialize df['bad_section'] = False before conditional writes.",
            "effort": "trivial",
        },
    },
    "DS-011": {
        "title": "get_mean_angle() shadows built-in names min/max",
        "file": "DeepSlice/coord_post_processing/angle_methods.py",
        "suggested_fix": {
            "patch": "Rename min/max to depth_min/depth_max.",
            "effort": "trivial",
        },
    },
    "DS-012": {
        "title": "Species depth ranges duplicated across modules",
        "file": "DeepSlice/coord_post_processing/ (multiple files)",
        "suggested_fix": {
            "patch": "Centralize depth bounds in config and load from metadata config.",
            "effort": "low",
        },
    },
}


def log_issue(
    rule_id: str,
    severity: str,
    description: str,
    context: Optional[Dict[str, Any]] = None,
    exc: Optional[BaseException] = None,
    location: Optional[Dict[str, Any]] = None,
) -> dict:
    """Record a structured diagnostic event and return the event object."""
    severity = severity.upper()
    if severity not in ("ERROR", "WARNING", "INFO"):
        severity = "INFO"

    catalogue_entry = RULE_CATALOGUE.get(rule_id, {})
    if location is None:
        location = {
            "file": catalogue_entry.get("file", "unknown"),
            "function": None,
            "line": None,
        }

    event: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "severity": severity,
        "rule_id": rule_id,
        "title": catalogue_entry.get("title", description),
        "location": location,
        "description": description,
        "context": context or {},
        "suggested_fix": catalogue_entry.get("suggested_fix", {}),
        "traceback": (
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            if exc is not None
            else None
        ),
    }

    ISSUES.append(event)
    level = getattr(logging, severity, logging.INFO)
    _logger.log(level, "[%s] %s | %s", rule_id, severity, description)
    return event


def flush_log(output_path: Optional[str] = None) -> str:
    """Write all accumulated events to JSON and return the output path."""
    if output_path is None:
        output_path = os.path.join(os.getcwd(), "deepslice_diagnostics.json")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_handle:
        json.dump(ISSUES, file_handle, indent=2, default=str)

    _logger.info("Diagnostics written to %s (%d events)", output_path, len(ISSUES))
    return output_path


def clear_log() -> None:
    """Clear all accumulated in-memory issues."""
    ISSUES.clear()


def get_issues_by_severity(severity: str) -> list[dict]:
    """Return all events matching a severity."""
    return [event for event in ISSUES if event["severity"] == severity.upper()]


def get_trivial_fixes() -> list[dict]:
    """Return events whose suggested fix effort is marked as trivial."""
    return [
        event
        for event in ISSUES
        if event.get("suggested_fix", {}).get("effort") == "trivial"
    ]


def monitored(rule_id: str, severity: str = "ERROR") -> Callable:
    """
    Decorator that logs unhandled exceptions in the wrapped function under
    the provided diagnostic rule_id and re-raises the exception.
    """

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                log_issue(
                    rule_id=rule_id,
                    severity=severity,
                    description=f"Unhandled exception in {fn.__qualname__}: {exc}",
                    context={
                        "args_repr": repr(args)[:200],
                        "kwargs_repr": repr(kwargs)[:200],
                    },
                    exc=exc,
                    location={
                        "file": fn.__code__.co_filename,
                        "function": fn.__qualname__,
                        "line": fn.__code__.co_firstlineno,
                    },
                )
                raise

        return wrapper

    return decorator


def run_static_audit() -> list[dict]:
    """Emit one INFO event per catalogue entry and return emitted events."""
    emitted = []
    for rule_id, entry in RULE_CATALOGUE.items():
        event = log_issue(
            rule_id=rule_id,
            severity="INFO",
            description=f"[STATIC AUDIT] {entry['title']}",
        )
        emitted.append(event)
    return emitted