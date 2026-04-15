from __future__ import annotations

import logging
import sys
import threading
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

_LOGGER_NAME = "DeepSlice"
_DEFAULT_LOG_FILENAME = "errors.log"
_MAX_LOG_BYTES = 5 * 1024 * 1024
_BACKUP_LOG_COUNT = 5

_configured_log_path: Optional[Path] = None
_hooks_installed = False
_previous_sys_excepthook = sys.excepthook
_previous_threading_excepthook = getattr(threading, "excepthook", None)


def get_error_log_path() -> str:
    """Return the absolute path to the persistent DeepSlice error log file."""
    global _configured_log_path

    if _configured_log_path is None:
        log_dir = Path.home() / ".deepslice" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        _configured_log_path = log_dir / _DEFAULT_LOG_FILENAME

    return str(_configured_log_path)


def configure_error_logging(log_path: Optional[str] = None) -> str:
    """Configure rotating file logging for DeepSlice errors and diagnostics."""
    global _configured_log_path

    if log_path:
        resolved = Path(log_path).expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        _configured_log_path = resolved

    resolved_log_path = Path(get_error_log_path())

    base_logger = logging.getLogger(_LOGGER_NAME)
    base_logger.setLevel(logging.INFO)
    base_logger.propagate = False

    existing_file_handlers = [
        handler
        for handler in base_logger.handlers
        if isinstance(handler, RotatingFileHandler)
    ]

    has_target_handler = any(
        Path(handler.baseFilename) == resolved_log_path
        for handler in existing_file_handlers
    )

    for handler in existing_file_handlers:
        if Path(handler.baseFilename) != resolved_log_path:
            base_logger.removeHandler(handler)
            handler.close()

    if not has_target_handler:
        file_handler = RotatingFileHandler(
            filename=resolved_log_path,
            maxBytes=_MAX_LOG_BYTES,
            backupCount=_BACKUP_LOG_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            )
        )
        base_logger.addHandler(file_handler)

    base_logger.info("Error logging initialized at %s", resolved_log_path)
    return str(resolved_log_path)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a DeepSlice logger, configuring file logging if needed."""
    configure_error_logging()
    if not name:
        return logging.getLogger(_LOGGER_NAME)
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")


def log_error_text(context: str, error_text: str, level: int = logging.ERROR) -> None:
    """Write a text error payload to the DeepSlice error log."""
    logger = get_logger("errors")
    logger.log(level, "%s\n%s", str(context), str(error_text).rstrip())


def log_exception(
    context: str,
    exc: Optional[BaseException] = None,
    level: int = logging.ERROR,
) -> None:
    """Write an exception (with traceback) to the DeepSlice error log."""
    logger = get_logger("errors")

    if exc is None:
        logger.exception(str(context))
        return

    logger.log(
        level,
        "%s: %s",
        str(context),
        str(exc),
        exc_info=(type(exc), exc, exc.__traceback__),
    )


def read_error_log_tail(max_chars: int = 16000) -> str:
    """Return the tail of the error log for easy copy/paste in bug reports."""
    log_path = Path(get_error_log_path())
    if not log_path.exists():
        return ""

    content = log_path.read_text(encoding="utf-8", errors="replace")
    if len(content) <= max_chars:
        return content

    return content[-max_chars:]


def build_error_report(context: str, error_text: str, log_path: Optional[str] = None) -> str:
    """Build a copy/paste-friendly error report payload."""
    resolved_log_path = log_path or get_error_log_path()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return (
        "DeepSlice Error Report\n"
        f"Timestamp: {timestamp}\n"
        f"Context: {context}\n"
        f"Log file: {resolved_log_path}\n\n"
        "Details:\n"
        f"{str(error_text).strip()}\n"
    )


def _global_sys_exception_hook(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        if _previous_sys_excepthook is not None:
            _previous_sys_excepthook(exc_type, exc_value, exc_traceback)
        return

    logger = get_logger("uncaught")
    logger.critical(
        "Uncaught exception",
        exc_info=(exc_type, exc_value, exc_traceback),
    )

    if _previous_sys_excepthook is not None:
        _previous_sys_excepthook(exc_type, exc_value, exc_traceback)


def _global_thread_exception_hook(args):
    logger = get_logger("uncaught")
    thread_name = getattr(args.thread, "name", "unknown-thread")
    logger.critical(
        "Uncaught thread exception in %s",
        thread_name,
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )

    if _previous_threading_excepthook is not None:
        _previous_threading_excepthook(args)


def install_global_exception_hooks() -> None:
    """Install process-wide uncaught exception hooks for persistent logging."""
    global _hooks_installed

    if _hooks_installed:
        return

    sys.excepthook = _global_sys_exception_hook
    if hasattr(threading, "excepthook"):
        threading.excepthook = _global_thread_exception_hook

    _hooks_installed = True
