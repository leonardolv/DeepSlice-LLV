from ..error_logging import (
    configure_error_logging,
    get_logger,
    install_global_exception_hooks,
    log_exception,
)


def main():
    log_path = configure_error_logging()
    install_global_exception_hooks()

    logger = get_logger("gui.app")
    logger.info("Starting DeepSlice GUI. Error log: %s", log_path)

    try:
        from .main_window import launch_gui

        return launch_gui()
    except Exception as exc:
        log_exception("DeepSlice GUI failed to start", exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
