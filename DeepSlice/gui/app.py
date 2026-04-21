from __future__ import annotations

import os

from ..error_logging import (
    configure_error_logging,
    get_logger,
    install_global_exception_hooks,
    log_exception,
)


def configure_tensorflow_runtime():
    from ..error_logging import get_logger
    logger = get_logger('gui.app')
    """Prefer dedicated GPU, enable safe memory growth, and turn on XLA JIT when available."""
    try:
        import tensorflow as tf
    except Exception as exc:
        logger.info("TensorFlow not available during startup runtime tuning: %s", exc)
        return

    try:
        physical_gpus = tf.config.list_physical_devices("GPU")
    except Exception as exc:
        logger.info("Unable to list TensorFlow GPUs: %s", exc)
        return

    if len(physical_gpus) == 0:
        logger.info("No TensorFlow GPU devices detected; CPU mode will be used")
        return

    def _device_name(device) -> str:
        try:
            details = tf.config.experimental.get_device_details(device)
            return str(details.get("device_name") or device.name)
        except Exception:
            return str(device.name)

    dedicated_keywords = ("nvidia", "rtx", "gtx", "quadro", "radeon", "rx", "arc")
    integrated_keywords = ("intel", "uhd", "iris")

    selected_gpu = physical_gpus[0]
    selected_name = _device_name(selected_gpu).lower()
    best_score = -1

    for candidate in physical_gpus:
        name = _device_name(candidate).lower()
        score = 1
        if any(keyword in name for keyword in dedicated_keywords):
            score = 3
        if any(keyword in name for keyword in integrated_keywords):
            score = 0
        if score > best_score:
            best_score = score
            selected_gpu = candidate
            selected_name = name

    try:
        tf.config.set_visible_devices([selected_gpu], "GPU")
    except RuntimeError:
        # Visible devices cannot be changed once runtime is initialized.
        pass
    except Exception as exc:
        logger.warning("Failed to force dedicated GPU selection: %s", exc)

    try:
        visible_gpus = tf.config.list_physical_devices("GPU")
        for gpu in visible_gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as exc:
        logger.warning("Failed to configure TensorFlow GPU memory growth: %s", exc)

    try:
        tf.config.optimizer.set_jit(True)
    except Exception:
        pass

    logger.info("TensorFlow GPU runtime configured. Selected GPU: %s", selected_name)


def main():
    os.environ.setdefault("QT_OPENGL", "desktop")

    log_path = configure_error_logging()
    install_global_exception_hooks()

    logger = get_logger("gui.app")
    logger.info("Starting DeepSlice GUI. Error log: %s", log_path)

    import threading
    tf_thread = threading.Thread(target=configure_tensorflow_runtime)
    tf_thread.daemon = True
    tf_thread.start()
    _configure_tensorflow_runtime(logger)

    try:
        from .main_window import launch_gui

        return launch_gui()
    except Exception as exc:
        log_exception("DeepSlice GUI failed to start", exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
