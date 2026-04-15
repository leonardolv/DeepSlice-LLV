import traceback

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from ..error_logging import get_logger


LOGGER = get_logger("gui.worker")


class WorkerSignals(QObject):
    finished = Signal(object)
    error = Signal(str)
    progress = Signal(int, int, str)
    log = Signal(str)


class FunctionWorker(QRunnable):
    def __init__(self, fn, *args, inject_callbacks: bool = False, **kwargs):
        super().__init__()
        self.fn = fn
        self.task_name = getattr(fn, "__name__", str(fn))
        self.args = args
        self.kwargs = kwargs
        self.inject_callbacks = inject_callbacks
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            kwargs = dict(self.kwargs)
            if self.inject_callbacks:
                kwargs["progress_callback"] = self._emit_progress
                kwargs["log_callback"] = self.signals.log.emit
            result = self.fn(*self.args, **kwargs)
        except Exception as exc:
            error_text = traceback.format_exc()
            LOGGER.error(
                "Background task '%s' failed: %s",
                self.task_name,
                exc,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            self.signals.error.emit(error_text)
        else:
            self.signals.finished.emit(result)

    def _emit_progress(self, completed: int, total: int, phase: str):
        self.signals.progress.emit(int(completed), int(total), str(phase))
