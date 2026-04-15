from __future__ import annotations

import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

from .error_logging import get_logger


MODULE_PACKAGE_MAP: Dict[str, Tuple[str, str]] = {
    "matplotlib": ("matplotlib", "matplotlib"),
    "pyside6": ("PySide6", "PySide6"),
    "pyqt6": ("PyQt6", "PyQt6"),
    "numpy": ("numpy", "numpy"),
    "pandas": ("pandas", "pandas"),
    "requests": ("requests", "requests"),
    "pil": ("Pillow", "PIL"),
    "skimage": ("scikit-image", "skimage"),
    "scipy": ("scipy", "scipy"),
    "h5py": ("h5py", "h5py"),
    "nibabel": ("nibabel", "nibabel"),
    "reportlab": ("reportlab", "reportlab"),
    "lxml": ("lxml", "lxml"),
    "tensorflow": ("tensorflow<3.0", "tensorflow"),
}


class ErrorAutoFixer:
    """Rule-based analyzer and safe auto-fixer for common DeepSlice runtime errors."""

    def __init__(self):
        self.logger = get_logger("auto_fix")

    def analyze_error(self, context: str, error_text: str) -> dict:
        clean_text = str(error_text or "").strip()
        lower_text = clean_text.lower()

        missing_module = self._extract_missing_module(clean_text)
        if missing_module is not None:
            package_spec = self._resolve_install_target(missing_module)
            analysis = {
                "category": "missing_dependency",
                "summary": f"Missing Python dependency detected: {missing_module}",
                "recommendations": [
                    "Install the missing dependency into the active DeepSlice environment.",
                    "Restart DeepSlice after installation so imports are refreshed.",
                ],
                "auto_fix_available": package_spec is not None,
                "auto_fix_plan": (
                    f"Run: {sys.executable} -m pip install {package_spec[0]}"
                    if package_spec is not None
                    else "No safe auto-fix mapping found for this module."
                ),
            }
            return analysis

        if "no predictions available" in lower_text:
            return {
                "category": "workflow_precondition",
                "summary": "Workflow precondition error: predictions are missing.",
                "recommendations": [
                    "Run alignment/predict before calling curation or export actions.",
                    "If this happened during normal flow, report the copied error report for diagnosis.",
                ],
                "auto_fix_available": False,
                "auto_fix_plan": "No automatic fix is safe for workflow-state issues.",
            }

        if "section number" in lower_text and "no section number found" in lower_text:
            return {
                "category": "filename_validation",
                "summary": "Filename index parsing failed.",
                "recommendations": [
                    "Rename files to include section IDs like _s001, _s002, ...",
                    "Or disable section-number parsing for this run.",
                ],
                "auto_fix_available": False,
                "auto_fix_plan": "No automatic fix applied because filenames require user confirmation.",
            }

        if "permission" in lower_text and "denied" in lower_text:
            return {
                "category": "permission_error",
                "summary": "Permission-related file system error detected.",
                "recommendations": [
                    "Use a writable folder for outputs/logs.",
                    "Retry with normal local path permissions.",
                ],
                "auto_fix_available": False,
                "auto_fix_plan": "No automatic fix available for permission constraints.",
            }

        return {
            "category": "unknown",
            "summary": "No known safe auto-fix pattern matched this error.",
            "recommendations": [
                "Copy the generated error report and share it for diagnosis.",
                "Open the persistent error log and inspect the full traceback.",
            ],
            "auto_fix_available": False,
            "auto_fix_plan": "No automatic fix attempted.",
        }

    def format_analysis(self, analysis: dict) -> str:
        lines: List[str] = []
        summary = str(analysis.get("summary", "")).strip()
        if summary:
            lines.append(f"Summary: {summary}")

        category = str(analysis.get("category", "")).strip()
        if category:
            lines.append(f"Category: {category}")

        recommendations = analysis.get("recommendations", []) or []
        if recommendations:
            lines.append("Recommendations:")
            for recommendation in recommendations:
                lines.append(f"- {recommendation}")

        lines.append(
            "Auto-fix available: "
            + ("yes" if bool(analysis.get("auto_fix_available", False)) else "no")
        )

        auto_fix_plan = str(analysis.get("auto_fix_plan", "")).strip()
        if auto_fix_plan:
            lines.append(f"Auto-fix plan: {auto_fix_plan}")

        return "\n".join(lines)

    def try_auto_fix(self, context: str, error_text: str) -> dict:
        analysis = self.analyze_error(context, error_text)

        if not analysis.get("auto_fix_available", False):
            return {
                "attempted": False,
                "succeeded": False,
                "summary": "No safe automatic fix available for this error.",
                "details": self.format_analysis(analysis),
                "analysis": analysis,
            }

        if analysis.get("category") != "missing_dependency":
            return {
                "attempted": False,
                "succeeded": False,
                "summary": "Auto-fix currently supports missing dependency errors only.",
                "details": self.format_analysis(analysis),
                "analysis": analysis,
            }

        missing_module = self._extract_missing_module(error_text)
        if missing_module is None:
            return {
                "attempted": False,
                "succeeded": False,
                "summary": "Could not extract missing module name from traceback.",
                "details": self.format_analysis(analysis),
                "analysis": analysis,
            }

        resolved = self._resolve_install_target(missing_module)
        if resolved is None:
            return {
                "attempted": False,
                "succeeded": False,
                "summary": f"No safe package mapping for missing module '{missing_module}'.",
                "details": self.format_analysis(analysis),
                "analysis": analysis,
            }

        package_spec, import_name = resolved
        return self._install_and_verify(package_spec, import_name, analysis)

    def _extract_missing_module(self, error_text: str) -> Optional[str]:
        match = re.search(r"No module named ['\"]([^'\"]+)['\"]", str(error_text))
        if not match:
            return None

        module_name = match.group(1).strip()
        if not module_name:
            return None

        # Avoid command injection in generated shell snippets.
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", module_name):
            return None

        return module_name

    def _resolve_install_target(self, module_name: str) -> Optional[Tuple[str, str]]:
        root = module_name.split(".")[0].lower()
        return MODULE_PACKAGE_MAP.get(root)

    def _install_and_verify(self, package_spec: str, import_name: str, analysis: dict) -> dict:
        install_cmd = [sys.executable, "-m", "pip", "install", package_spec]
        self.logger.info("Running auto-fix install command: %s", " ".join(install_cmd))

        try:
            install_result = subprocess.run(
                install_cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except Exception as exc:
            self.logger.exception("Auto-fix install command failed")
            return {
                "attempted": True,
                "succeeded": False,
                "summary": f"Auto-fix failed while running pip install for '{package_spec}'.",
                "details": str(exc),
                "analysis": analysis,
            }

        install_output = "\n".join(
            [
                install_result.stdout.strip(),
                install_result.stderr.strip(),
            ]
        ).strip()

        if install_result.returncode != 0:
            self.logger.error(
                "Auto-fix install failed for %s with return code %s",
                package_spec,
                install_result.returncode,
            )
            return {
                "attempted": True,
                "succeeded": False,
                "summary": f"Auto-fix could not install '{package_spec}'.",
                "details": install_output or "pip install failed with no output.",
                "analysis": analysis,
            }

        verify_cmd = [
            sys.executable,
            "-c",
            (
                "import importlib; "
                f"importlib.import_module('{import_name}')"
            ),
        ]

        verify_result = subprocess.run(
            verify_cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if verify_result.returncode != 0:
            verify_output = "\n".join(
                [verify_result.stdout.strip(), verify_result.stderr.strip()]
            ).strip()
            self.logger.error("Auto-fix verify import failed for %s", import_name)
            return {
                "attempted": True,
                "succeeded": False,
                "summary": f"Package '{package_spec}' installed but import verification failed.",
                "details": verify_output or install_output,
                "analysis": analysis,
            }

        self.logger.info(
            "Auto-fix succeeded. Installed %s and verified import %s.",
            package_spec,
            import_name,
        )
        return {
            "attempted": True,
            "succeeded": True,
            "summary": f"Auto-fix succeeded. Installed '{package_spec}'.",
            "details": install_output or "Dependency installed and import check passed.",
            "analysis": analysis,
        }
