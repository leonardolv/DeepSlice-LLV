import traceback

from PySide6.QtCore import QObject, QRunnable, Signal, Slot


class WorkerSignals(QObject):
    finished = Signal(object)
    error = Signal(str)
    progress = Signal(int, int, str)
    log = Signal(str)


class FunctionWorker(QRunnable):
    def __init__(self, fn, *args, inject_callbacks: bool = False, **kwargs):
        super().__init__()
        self.fn = fn
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
        except Exception:
            self.signals.error.emit(traceback.format_exc())
        else:
            self.signals.finished.emit(result)

    def _emit_progress(self, completed: int, total: int, phase: str):
        self.signals.progress.emit(int(completed), int(total), str(phase))
