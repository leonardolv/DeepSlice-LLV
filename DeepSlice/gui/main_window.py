from __future__ import annotations

import json
import os
import subprocess
import traceback
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import (
    QEasingCurve,
    QElapsedTimer,
    QPropertyAnimation,
    QSettings,
    QSize,
    Qt,
    QThreadPool,
    QTimer,
    QUrl,
    Signal,
)
from PySide6.QtGui import (
    QCloseEvent,
    QColor,
    QDesktopServices,
    QFont,
    QIcon,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QMenu,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsOpacityEffect,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QTextEdit,
    QPushButton,
    QProgressBar,
    QPlainTextEdit,
    QRadioButton,
    QSplashScreen,
    QSlider,
    QSplitter,
    QStyle,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QDoubleSpinBox,
    QSpinBox,
)
from PIL import Image

from ..error_auto_fix import ErrorAutoFixer
from ..error_logging import (
    build_error_report,
    configure_error_logging,
    get_logger,
    log_error_text,
    read_error_log_tail,
)
from ..metadata import metadata_loader
from . import reporting
from .state import DeepSliceAppState, SUPPORTED_IMAGE_FORMATS
from .workers import FunctionWorker


class DropArea(QFrame):
    pathsDropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setObjectName("DropArea")
        self.setMinimumHeight(80)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(4)

        icon = QLabel("[ ]")
        icon.setObjectName("DropIcon")
        icon.setAlignment(Qt.AlignCenter)

        title = QLabel("Drag and Drop Images or Folders")
        title.setObjectName("DropTitle")
        subtitle = QLabel("Supports JPG, PNG, TIFF. Folder drops recurse into subfolders.")
        subtitle.setWordWrap(True)
        subtitle.setObjectName("DropSubtitle")

        layout.addWidget(icon)
        layout.addWidget(title)
        layout.addWidget(subtitle)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if not event.mimeData().hasUrls():
            event.ignore()
            return
        dropped_paths = []
        for url in event.mimeData().urls():
            local_path = url.toLocalFile()
            if local_path:
                dropped_paths.append(local_path)
        self.pathsDropped.emit(dropped_paths)
        event.acceptProposedAction()


class ThumbnailListWidget(QListWidget):
    pathsDropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            dropped_paths = []
            for url in event.mimeData().urls():
                local_path = url.toLocalFile()
                if local_path:
                    dropped_paths.append(local_path)
            self.pathsDropped.emit(dropped_paths)
            event.acceptProposedAction()
            return
        super().dropEvent(event)


class FlagListWidget(QListWidget):
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            item = self.currentItem()
            if item is not None and (item.flags() & Qt.ItemIsUserCheckable):
                new_state = (
                    Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked
                )
                item.setCheckState(new_state)
                event.accept()
                return
        super().keyPressEvent(event)


class ToastOverlay(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ToastOverlay")
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setVisible(False)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        self.message_label = QLabel("")
        self.message_label.setWordWrap(True)
        layout.addWidget(self.message_label)

        self._dismiss_timer = QTimer(self)
        self._dismiss_timer.setSingleShot(True)
        self._dismiss_timer.timeout.connect(self._start_fade_out)

        self._opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._opacity_effect)
        self._opacity_effect.setOpacity(0.0)

        self._fade_in = QPropertyAnimation(self._opacity_effect, b"opacity", self)
        self._fade_in.setDuration(120)
        self._fade_in.setStartValue(0.0)
        self._fade_in.setEndValue(1.0)
        self._fade_in.setEasingCurve(QEasingCurve.OutCubic)

        self._fade_out = QPropertyAnimation(self._opacity_effect, b"opacity", self)
        self._fade_out.setDuration(180)
        self._fade_out.setStartValue(1.0)
        self._fade_out.setEndValue(0.0)
        self._fade_out.setEasingCurve(QEasingCurve.InCubic)
        self._fade_out.finished.connect(self.hide)

    def show_message(self, message: str, timeout_ms: int = 3000, level: str = "info"):
        level_key = str(level).strip().lower()
        palette = {
            "info": ("#0D2538", "#69C0FF", "#D8EEFF"),
            "success": ("#103321", "#2CC784", "#D9FBEA"),
            "warning": ("#3A2A0A", "#E3A33B", "#FFF2D6"),
            "error": ("#3A151B", "#D33E56", "#FFE4EA"),
        }
        bg, border, text = palette.get(level_key, palette["info"])
        self.setStyleSheet(
            "QFrame#ToastOverlay {"
            f"background: {bg};"
            f"border: 1px solid {border};"
            "border-radius: 10px;"
            "}"
            "QFrame#ToastOverlay QLabel {"
            f"color: {text};"
            "font-weight: 600;"
            "}"
        )
        self.message_label.setText(str(message))
        self.adjustSize()
        self._reposition()
        self.raise_()
        self.show()

        self._dismiss_timer.stop()
        self._fade_out.stop()
        self._fade_in.stop()
        self._opacity_effect.setOpacity(0.0)
        self._fade_in.start()
        self._dismiss_timer.start(max(int(timeout_ms), 500))

    def _start_fade_out(self):
        if not self.isVisible():
            return
        self._fade_in.stop()
        self._fade_out.stop()
        self._fade_out.start()

    def _reposition(self):
        parent = self.parentWidget()
        if parent is None:
            return
        margin = 18
        max_width = int(parent.width() * 0.45)
        max_width = max(max_width, 280)
        self.setMaximumWidth(max_width)
        self.adjustSize()
        x_pos = max(margin, parent.width() - self.width() - margin)
        y_pos = max(margin, parent.height() - self.height() - margin)
        self.move(x_pos, y_pos)


class SliceGraphicsView(QGraphicsView):
    viewTransformChanged = Signal(float, float, int, int)
    mouseSceneMoved = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setRenderHints(self.renderHints())
        self.setBackgroundBrush(QColor("#11161D"))
        self._zoom = 0
        self._scale_bar_item = None
        self._pixel_spacing_um = None
        self._sync_partner: Optional["SliceGraphicsView"] = None
        self._sync_guard = False
        self._loupe_enabled = False
        self.setMouseTracking(True)
        self.horizontalScrollBar().valueChanged.connect(self._on_scrollbar_changed)
        self.verticalScrollBar().valueChanged.connect(self._on_scrollbar_changed)

    def set_sync_partner(self, partner: Optional["SliceGraphicsView"]):
        self._sync_partner = partner

    def set_loupe_enabled(self, enabled: bool):
        self._loupe_enabled = bool(enabled)

    def _emit_transform_changed(self):
        transform = self.transform()
        self.viewTransformChanged.emit(
            float(transform.m11()),
            float(transform.m22()),
            int(self.horizontalScrollBar().value()),
            int(self.verticalScrollBar().value()),
        )

    def apply_external_transform(self, sx: float, sy: float, h_scroll: int, v_scroll: int):
        if self._sync_guard:
            return
        self._sync_guard = True
        try:
            self.resetTransform()
            self.scale(sx, sy)
            self.horizontalScrollBar().setValue(int(h_scroll))
            self.verticalScrollBar().setValue(int(v_scroll))
        finally:
            self._sync_guard = False

    def _on_scrollbar_changed(self, _value: int):
        if self._sync_partner is None or self._sync_guard:
            return
        self._sync_guard = True
        try:
            self._sync_partner.apply_external_transform(
                float(self.transform().m11()),
                float(self.transform().m22()),
                int(self.horizontalScrollBar().value()),
                int(self.verticalScrollBar().value()),
            )
        finally:
            self._sync_guard = False

    def clear_with_text(self, message: str):
        self._scene.clear()
        self._scale_bar_item = None
        text_item = QGraphicsTextItem(message)
        text_item.setDefaultTextColor(QColor("#C4CBD3"))
        text_item.setFont(QFont("Segoe UI", 10))
        self._scene.addItem(text_item)
        self._scene.setSceneRect(self._scene.itemsBoundingRect())

    def _draw_scale_bar(self):
        if self._scale_bar_item is not None:
            self._scene.removeItem(self._scale_bar_item)
            self._scale_bar_item = None
            
        if self._pixel_spacing_um is None:
            return
            
        rect = self.sceneRect()
        width = rect.width()
        
        target_um = max(100.0, 10 ** np.floor(np.log10(width * self._pixel_spacing_um * 0.2)))
        bar_width_px = target_um / self._pixel_spacing_um
        
        if bar_width_px > width * 0.5:
            return
            
        from PySide6.QtWidgets import QGraphicsRectItem, QGraphicsItemGroup
        
        group = QGraphicsItemGroup()
        
        bar = QGraphicsRectItem(0, 0, bar_width_px, 4)
        bar.setBrush(QColor("#F7FBFF"))
        bar.setPen(Qt.NoPen)
        group.addToGroup(bar)
        
        text = QGraphicsTextItem(f"{int(target_um)} µm")
        text.setDefaultTextColor(QColor("#F7FBFF"))
        text.setFont(QFont("Segoe UI", 10, QFont.Bold))
        text_rect = text.boundingRect()
        text.setPos(bar_width_px / 2 - text_rect.width() / 2, -text_rect.height())
        group.addToGroup(text)
        
        margin = 20
        group.setPos(rect.right() - bar_width_px - margin, rect.bottom() - margin)
        
        self._scale_bar_item = group
        self._scene.addItem(self._scale_bar_item)

    def set_image(
        self,
        image_path: Optional[str],
        overlay_lines: Optional[List[str]] = None,
        border_color: Optional[QColor] = None,
        pixel_spacing_um: Optional[float] = None,
    ):
        self._scene.clear()
        self._scale_bar_item = None
        self._zoom = 0
        self._pixel_spacing_um = pixel_spacing_um

        if image_path is None or not os.path.exists(image_path):
            self.clear_with_text("Image preview unavailable")
            return

        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.clear_with_text("Unable to load image preview")
            return

        pix_item = self._scene.addPixmap(pixmap)

        if border_color is not None:
            pen_width = 4
            self._scene.addRect(
                pix_item.boundingRect(),
                border_color,
            )

        if overlay_lines:
            overlay_text = "\n".join(overlay_lines)
            text_item = QGraphicsTextItem(overlay_text)
            text_item.setDefaultTextColor(QColor("#52E5FF"))
            text_item.setFont(QFont("Segoe UI", 10))
            text_item.setPos(12, 12)
            self._scene.addItem(text_item)

        self.setSceneRect(self._scene.itemsBoundingRect())
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        self._draw_scale_bar()

    def set_array_image(
        self,
        image_array: np.ndarray,
        overlay_lines: Optional[List[str]] = None,
        border_color: Optional[QColor] = None,
    ):
        self._scene.clear()
        self._zoom = 0

        if image_array is None or image_array.size == 0:
            self.clear_with_text("Atlas preview unavailable")
            return

        data = np.asarray(image_array)
        if data.ndim not in {2, 3}:
            self.clear_with_text("Array preview expects a 2D or RGB image")
            return

        if data.ndim == 2:
            normalized = np.nan_to_num(
                data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
            )
            if normalized.dtype != np.uint8:
                min_value = float(np.min(normalized))
                max_value = float(np.max(normalized))
                if max_value <= min_value:
                    normalized = np.zeros_like(normalized, dtype=np.uint8)
                else:
                    normalized = (
                        (normalized - min_value) / (max_value - min_value) * 255.0
                    ).astype(np.uint8)

            normalized = np.ascontiguousarray(normalized)
            height, width = normalized.shape
            image = QImage(
                normalized.data,
                width,
                height,
                normalized.strides[0],
                QImage.Format_Grayscale8,
            ).copy()
        else:
            normalized = np.nan_to_num(
                data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
            )
            if normalized.shape[2] != 3:
                self.clear_with_text("RGB array preview expects exactly 3 channels")
                return
            if normalized.dtype != np.uint8:
                min_value = float(np.min(normalized))
                max_value = float(np.max(normalized))
                if max_value <= min_value:
                    normalized = np.zeros_like(normalized, dtype=np.uint8)
                else:
                    normalized = (
                        (normalized - min_value) / (max_value - min_value) * 255.0
                    ).astype(np.uint8)
            normalized = np.ascontiguousarray(normalized)
            height, width, _ = normalized.shape
            image = QImage(
                normalized.data,
                width,
                height,
                normalized.strides[0],
                QImage.Format_RGB888,
            ).copy()

        pixmap = QPixmap.fromImage(image)
        pix_item = self._scene.addPixmap(pixmap)

        if border_color is not None:
            self._scene.addRect(pix_item.boundingRect(), border_color)

        if overlay_lines:
            overlay_text = "\n".join(overlay_lines)
            text_item = QGraphicsTextItem(overlay_text)
            text_item.setDefaultTextColor(QColor("#52E5FF"))
            text_item.setFont(QFont("Segoe UI", 10))
            text_item.setPos(12, 12)
            self._scene.addItem(text_item)

        self.setSceneRect(self._scene.itemsBoundingRect())
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            zoom_factor = 1.25 if self._loupe_enabled else 1.15
            self._zoom += 1
        else:
            zoom_factor = 1.0 / (1.25 if self._loupe_enabled else 1.15)
            self._zoom -= 1

        if self._zoom < -5:
            self._zoom = -5
            return
        self.scale(zoom_factor, zoom_factor)
        self._emit_transform_changed()

        if self._sync_partner is not None and not self._sync_guard:
            self._sync_guard = True
            try:
                self._sync_partner.apply_external_transform(
                    float(self.transform().m11()),
                    float(self.transform().m22()),
                    int(self.horizontalScrollBar().value()),
                    int(self.verticalScrollBar().value()),
                )
            finally:
                self._sync_guard = False

    def mouseMoveEvent(self, event):
        scene_pos = self.mapToScene(event.position().toPoint())
        self.mouseSceneMoved.emit(float(scene_pos.x()), float(scene_pos.y()))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self._emit_transform_changed()
        if self._sync_partner is not None and not self._sync_guard:
            self._sync_guard = True
            try:
                self._sync_partner.apply_external_transform(
                    float(self.transform().m11()),
                    float(self.transform().m22()),
                    int(self.horizontalScrollBar().value()),
                    int(self.verticalScrollBar().value()),
                )
            finally:
                self._sync_guard = False


def _build_deepslice_icon(size: int = 128) -> QIcon:
    size = max(int(size), 64)
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)

    painter = QPainter(pix)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setPen(Qt.NoPen)

    painter.setBrush(QColor("#0F223D"))
    painter.drawRoundedRect(0, 0, size, size, size * 0.18, size * 0.18)

    painter.setBrush(QColor("#1E6FFF"))
    painter.drawEllipse(int(size * 0.18), int(size * 0.24), int(size * 0.30), int(size * 0.52))
    painter.drawEllipse(int(size * 0.52), int(size * 0.24), int(size * 0.30), int(size * 0.52))

    painter.setPen(QPen(QColor("#8EC5FF"), max(2, int(size * 0.03))))
    painter.drawArc(int(size * 0.30), int(size * 0.30), int(size * 0.40), int(size * 0.38), 16 * 200, 16 * 140)
    painter.drawArc(int(size * 0.30), int(size * 0.42), int(size * 0.40), int(size * 0.26), 16 * 20, 16 * 160)

    painter.setPen(QPen(QColor("#D9EEFF"), max(2, int(size * 0.03))))
    painter.drawLine(int(size * 0.50), int(size * 0.22), int(size * 0.50), int(size * 0.78))
    painter.end()
    return QIcon(pix)


def _build_startup_splash(app_icon: QIcon) -> Tuple[QSplashScreen, QLabel, QProgressBar]:
    width, height = 620, 300
    pix = QPixmap(width, height)
    pix.fill(QColor("#0C1420"))

    painter = QPainter(pix)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setPen(Qt.NoPen)

    painter.setBrush(QColor("#122238"))
    painter.drawRoundedRect(14, 14, width - 28, height - 28, 20, 20)

    painter.setBrush(QColor("#17385F"))
    painter.drawRoundedRect(28, 28, width - 56, height - 56, 16, 16)

    painter.drawPixmap(42, 54, app_icon.pixmap(QSize(88, 88)))

    painter.setPen(QColor("#E8F2FF"))
    painter.setFont(QFont("Segoe UI", 19, QFont.Bold))
    painter.drawText(150, 96, "DeepSlice Desktop")

    painter.setPen(QColor("#9DB7D8"))
    painter.setFont(QFont("Segoe UI", 10))
    painter.drawText(150, 126, "Initializing atlas workspace and prediction runtime")
    painter.end()

    splash = QSplashScreen(pix)
    splash.setWindowFlag(Qt.WindowStaysOnTopHint)

    status_label = QLabel("Starting...", splash)
    status_label.setGeometry(42, 232, width - 84, 22)
    status_label.setStyleSheet("color: #C8DDF6; font: 10pt 'Segoe UI';")

    progress = QProgressBar(splash)
    progress.setGeometry(42, 260, width - 84, 16)
    progress.setRange(0, 100)
    progress.setValue(0)
    progress.setTextVisible(False)
    progress.setStyleSheet(
        "QProgressBar {"
        " border: 1px solid #2A3B53;"
        " border-radius: 7px;"
        " background: #0F1D2F;"
        "}"
        "QProgressBar::chunk {"
        " background-color: #2E8BFF;"
        " border-radius: 6px;"
        "}"
    )
    return splash, status_label, progress


class DeepSliceMainWindow(QMainWindow):
    STEP_LABELS = [
        "Ingestion",
        "Configuration",
        "Prediction",
        "Curation",
        "Export",
    ]

    def __init__(
        self,
        startup_progress: Optional[Callable[[str, int], None]] = None,
        app_icon: Optional[QIcon] = None,
    ):
        super().__init__()
        self._startup_progress_callback = startup_progress
        self.error_log_path = configure_error_logging()
        self._logger = get_logger("gui.main_window")
        self._error_autofixer = ErrorAutoFixer()
        self._last_error_report = ""
        self._last_error_context = ""
        self._last_error_text = ""
        self._last_error_analysis = None

        self.state = DeepSliceAppState()
        self._apply_startup_preferences_to_state()
        self._session_base_text = "Session: New"
        self.thread_pool = QThreadPool.globalInstance()
        self.active_workers = []
        self.last_export_basepath: Optional[str] = None
        self._linearity_payload = None
        self._atlas_request_token = 0
        self._latest_atlas_slice: Optional[np.ndarray] = None
        self._latest_atlas_meta: Optional[dict] = None
        self._baseline_predictions = None
        self._curation_modified = False
        self._last_completed_steps = set()
        self._anchor_depth_targets: Dict[int, float] = {}
        self._window_title_base = "DeepSlice Desktop"

        self._prediction_elapsed_timer = QElapsedTimer()
        self._prediction_clock_timer = QTimer(self)
        self._prediction_clock_timer.setInterval(1000)
        self._prediction_clock_timer.timeout.connect(self._update_prediction_timing)
        self._prediction_cancel_requested = False
        self._prediction_total = 0
        self._prediction_completed = 0
        self._prediction_phase = "idle"
        self._phase_total = 1

        self._run_animation_timer = QTimer(self)
        self._run_animation_timer.setInterval(350)
        self._run_animation_timer.timeout.connect(self._animate_run_button)
        self._run_button_dots = 0

        self.app_version = "Unknown"
        self._theme_name = self._load_theme_preference()

        self.setWindowTitle(self._window_title_base)
        self.resize(1600, 980)
        self.setWindowIcon(app_icon or _build_deepslice_icon())

        self._notify_startup_progress("Preparing user interface", 12)

        self._build_ui()
        self._restore_window_preferences()
        self.setAcceptDrops(True)
        self._notify_startup_progress("Wiring shortcuts", 38)
        self._setup_shortcuts()
        self._notify_startup_progress("Applying theme", 55)
        self._apply_theme()
        self._notify_startup_progress("Checking hardware", 70)
        self._update_hardware_mode_label()
        self._notify_startup_progress("Refreshing views", 86)
        self._refresh_all_views()
        self._notify_startup_progress("Finalizing startup", 96)
        self._setup_tab_order()
        QTimer.singleShot(150, self._show_startup_dialogs)
        self._notify_startup_progress("Ready", 100)
        self._logger.info("DeepSlice GUI initialized. Error log: %s", self.error_log_path)

    def _notify_startup_progress(self, message: str, percent: int):
        callback = self._startup_progress_callback
        if callback is None:
            return
        try:
            callback(message, int(np.clip(percent, 0, 100)))
        except Exception:
            pass

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            dropped_paths = []
            for url in event.mimeData().urls():
                local_path = url.toLocalFile()
                if local_path:
                    dropped_paths.append(local_path)
            if dropped_paths:
                self._handle_dropped_paths(dropped_paths)
                self._show_toast(f"Added {len(dropped_paths)} dropped path(s)", timeout_ms=2500)
            event.acceptProposedAction()
            return
        super().dropEvent(event)

    def _track_worker(self, worker: FunctionWorker):
        self.active_workers.append(worker)
        self._set_global_busy(True)
        worker.signals.finished.connect(
            lambda _result, tracked_worker=worker: self._release_worker(tracked_worker)
        )
        worker.signals.error.connect(
            lambda _error, tracked_worker=worker: self._release_worker(tracked_worker)
        )

    def _release_worker(self, worker: FunctionWorker):
        try:
            self.active_workers.remove(worker)
        except ValueError:
            pass
        if len(self.active_workers) == 0:
            self._set_global_busy(False)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        self.top_task_progress = QProgressBar()
        self.top_task_progress.setRange(0, 0)
        self.top_task_progress.setTextVisible(False)
        self.top_task_progress.setFixedHeight(3)
        self.top_task_progress.setVisible(False)
        root.addWidget(self.top_task_progress)

        root.addWidget(self._build_top_bar())

        body_split = QSplitter(Qt.Horizontal)
        body_split.setHandleWidth(6)
        self.body_split = body_split

        self.sidebar_container = QWidget()
        sidebar_layout = QVBoxLayout(self.sidebar_container)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        
        self.sidebar_header = QHBoxLayout()
        self.completion_label = QLabel("0% Complete")
        self.completion_label.setObjectName("CompletionLabel")
        self.collapse_sidebar_button = QPushButton("<<")
        self.collapse_sidebar_button.setFixedWidth(30)
        self.collapse_sidebar_button.clicked.connect(self._toggle_sidebar)
        self.sidebar_header.addWidget(self.completion_label)
        self.sidebar_header.addStretch()
        self.sidebar_header.addWidget(self.collapse_sidebar_button)
        
        sidebar_layout.addLayout(self.sidebar_header)

        self.step_list = QListWidget()
        self.step_list.setObjectName("StepNavigator")
        self.step_list.setFixedWidth(230)
        for step in self.STEP_LABELS:
            self.step_list.addItem(QListWidgetItem(step))
            
        sidebar_layout.addWidget(self.step_list, stretch=1)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_ingestion_page())
        self.stack.addWidget(self._build_configuration_page())
        self.stack.addWidget(self._build_prediction_page())
        self.stack.addWidget(self._build_curation_page())
        self.stack.addWidget(self._build_export_page())

        # Connect navigation after stack exists because setCurrentRow emits currentRowChanged.
        self.step_list.currentRowChanged.connect(self._on_step_changed)
        self.step_list.setCurrentRow(0)

        body_split.addWidget(self.sidebar_container)
        body_split.addWidget(self.stack)
        body_split.setStretchFactor(0, 1)
        body_split.setStretchFactor(1, 3)

        root.addWidget(body_split, stretch=1)

        self._assign_button_icons()
        self._apply_detailed_tooltips()
        self._apply_accessibility_metadata()

        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
        self.global_progress = QProgressBar()
        self.global_progress.setMaximumWidth(200)
        self.global_progress.setMaximumHeight(14)
        self.global_progress.setTextVisible(False)
        self.global_progress.setVisible(False)
        self.status_bar.addPermanentWidget(self.global_progress)
        self.toast_overlay = ToastOverlay(self)

    def _set_global_busy(self, busy: bool):
        self.top_task_progress.setVisible(bool(busy))
        self.global_progress.setVisible(bool(busy))

    def _set_session_io_busy(self, busy: bool):
        if hasattr(self, "session_io_spinner"):
            self.session_io_spinner.setVisible(bool(busy))

    def _assign_button_icons(self):
        style = self.style()
        icon_map = {
            "hardware_button": "SP_ComputerIcon",
            "new_session_button": "SP_FileIcon",
            "save_session_button": "SP_DialogSaveButton",
            "load_session_button": "SP_DialogOpenButton",
            "shortcut_help_button": "SP_MessageBoxInformation",
            "preferences_button": "SP_FileDialogDetailedView",
            "about_button": "SP_MessageBoxInformation",
            "error_menu_button": "SP_MessageBoxWarning",

            "add_folder_button": "SP_DirOpenIcon",
            "add_files_button": "SP_FileIcon",
            "clear_images_button": "SP_DialogResetButton",
            "naming_helper_button": "SP_MessageBoxQuestion",
            "orientation_guide_button": "SP_MessageBoxInformation",

            "suggest_thickness_button": "SP_BrowserReload",
            "direction_guide_button": "SP_MessageBoxInformation",
            "ensemble_help_button": "SP_MessageBoxInformation",
            "validate_configuration_button": "SP_DialogApplyButton",

            "run_alignment_button": "SP_MediaPlay",
            "cancel_alignment_button": "SP_BrowserStop",
            "accept_predicted_thickness_button": "SP_DialogApplyButton",
            "console_toggle": "SP_TitleBarShadeButton",
            "clear_console_button": "SP_TrashIcon",
            "copy_console_button": "SP_FileDialogInfoView",

            "curation_prev_button": "SP_ArrowBack",
            "curation_next_button": "SP_ArrowForward",
            "curation_select_all_btn": "SP_DialogYesButton",
            "curation_deselect_all_btn": "SP_DialogNoButton",
            "move_slice_up_button": "SP_ArrowUp",
            "move_slice_down_button": "SP_ArrowDown",
            "apply_manual_order_button": "SP_BrowserReload",
            "apply_bad_sections_button": "SP_DialogApplyButton",
            "detect_outliers_button": "SP_MessageBoxWarning",
            "reset_flags_button": "SP_DialogResetButton",
            "normalize_angles_button": "SP_BrowserReload",
            "enforce_order_button": "SP_ArrowUp",
            "enforce_spacing_button": "SP_ArrowRight",
            "apply_manual_angles_button": "SP_DialogApplyButton",
            "save_slice_note_button": "SP_DialogApplyButton",
            "set_anchor_button": "SP_DialogApplyButton",
            "remove_anchor_button": "SP_DialogResetButton",
            "apply_anchor_interpolation_button": "SP_ArrowForward",
            "clear_anchor_button": "SP_TrashIcon",
            "undo_button": "SP_ArrowBack",
            "redo_button": "SP_ArrowForward",
            "zoom_fit_button": "SP_DesktopIcon",
            "zoom_in_button": "SP_ArrowUp",
            "zoom_out_button": "SP_ArrowDown",
            "confidence_panel_toggle": "SP_FileDialogDetailedView",

            "browse_output_dir_button": "SP_DirOpenIcon",
            "output_format_help_button": "SP_MessageBoxInformation",
            "export_button": "SP_DialogSaveButton",
            "report_button": "SP_FileDialogDetailedView",
            "preview_report_button": "SP_FileDialogContentsView",
            "open_export_dir_button": "SP_DirOpenIcon",
            "copy_export_path_button": "SP_FileDialogInfoView",
            "quicknii_browse_button": "SP_DirOpenIcon",
            "open_quicknii_button": "SP_ArrowForward",
        }

        for attr_name, icon_name in icon_map.items():
            button = getattr(self, attr_name, None)
            if button is None:
                continue
            icon_id = getattr(QStyle, icon_name, None)
            if icon_id is None:
                continue
            button.setIcon(style.standardIcon(icon_id))
            button.setIconSize(QSize(16, 16))

    def _apply_detailed_tooltips(self):
        tooltips = {
            "new_session_button": "Create a new empty workspace state. If unsaved edits exist, you will be prompted before reset.",
            "save_session_button": "Save current GUI state (images, options, predictions, curation) to a .deepslice-session.json file.",
            "load_session_button": "Load a DeepSlice session file or QuickNII JSON/XML into the current workspace.",
            "hardware_button": "Show TensorFlow runtime details, GPU detection, CUDA/cuDNN versions, and available VRAM diagnostics.",
            "theme_toggle_button": "Switch between dark and light interface themes. Preference is saved for future launches.",
            "shortcut_help_button": "Open the keyboard shortcut cheat sheet for navigation, session actions, and curation editing.",
            "preferences_button": "Open persistent user preferences (defaults, paths, theme, and console visibility).",
            "about_button": "Show application version, citation information, and documentation pointers.",
            "error_menu_button": "Open error tools: view log file, copy latest error report, and run auto-fix suggestions.",

            "add_folder_button": "Recursively import supported images from a folder and its subfolders.",
            "add_files_button": "Import one or more supported image files into the current ingestion list.",
            "clear_images_button": "Remove all currently loaded input images from the session.",
            "enable_section_numbers_checkbox": "Enable filename parsing for section indices using _sXXX naming. Required for index-based spacing tools.",
            "naming_helper_button": "Open examples for supported filename patterns and legacy fallback behavior.",
            "legacy_parsing_checkbox": "Fallback parser: read trailing 3 digits from filenames when _sXXX pattern is not available.",
            "orientation_combo": "Select anatomical slicing orientation. Coronal is currently the fully supported pipeline.",
            "orientation_guide_button": "Show visual explanation of coronal, sagittal, and horizontal slicing orientations.",
            "thumbnail_filter_edit": "Filter thumbnails by filename substring.",
            "thumbnail_sort_combo": "Choose thumbnail ordering: filename, detected index, or modification date.",

            "mouse_radio": "Use mouse atlas configuration and model weights (Allen CCFv3 workflow).",
            "rat_radio": "Use rat atlas configuration and model weights (Waxholm rat workflow).",
            "auto_thickness_checkbox": "Automatically estimate section thickness from predicted spacing and detected indexing.",
            "thickness_spin": "Manual section thickness override in microns; used by spacing enforcement tools.",
            "suggest_thickness_button": "Estimate a recommended section thickness from current predictions.",
            "direction_override_combo": "Override inferred indexing direction when enforcing section spacing.",
            "direction_guide_button": "Explain rostro-caudal vs caudal-rostro indexing directions.",
            "ensemble_checkbox": "Run primary and secondary models and average predictions (higher robustness, slower runtime).",
            "ensemble_help_button": "Detailed explanation of ensemble accuracy vs speed tradeoff.",
            "secondary_model_checkbox": "Run only secondary model weights for comparison/testing workflows.",
            "legacy_from_config_checkbox": "Use legacy filename parser from configuration page for compatibility with older datasets.",
            "outlier_sigma_spin": "Sensitivity multiplier for residual outlier detection (1.0 = strict, 3.0 = lenient).",
            "confidence_high_spin": "Confidence score threshold for the High confidence bucket.",
            "confidence_medium_spin": "Confidence score threshold for the Medium confidence bucket.",
            "inference_batch_spin": "Inference batch size per model pass. Higher values are faster but use more memory.",
            "validate_configuration_button": "Run preflight checks and list blocking errors/warnings before prediction starts.",
            "tech_toggle": "Expand/collapse technical notes describing coordinate vectors, angle propagation, and thickness estimation.",

            "run_alignment_button": "Start prediction on the loaded dataset with current configuration options.",
            "cancel_alignment_button": "Request graceful cancellation of current prediction at a safe batch boundary.",
            "accept_predicted_thickness_button": "Apply model-estimated section thickness to manual thickness field.",
            "console_toggle": "Show or hide the runtime console panel for progress and diagnostic logs.",
            "console_autoscroll_toggle": "When enabled, console view follows new log lines automatically.",
            "clear_console_button": "Clear all visible runtime log lines from the console panel.",
            "copy_console_button": "Copy the full runtime console text to clipboard for debugging/sharing.",
            "prediction_compare_checkbox": "Atlas comparison panel is always enabled during prediction to provide continuous visual alignment context.",

            "curation_select_all_btn": "Mark all visible slices as bad sections in the checklist.",
            "curation_deselect_all_btn": "Clear bad-section checkmarks for all visible slices.",
            "curation_prev_button": "Jump to previous visible slice in curation list.",
            "curation_next_button": "Jump to next visible slice in curation list.",
            "confidence_filter_combo": "Filter curation list to high/medium/low-confidence slices only.",
            "move_slice_up_button": "Move selected slice one position up in manual curation order.",
            "move_slice_down_button": "Move selected slice one position down in manual curation order.",
            "apply_manual_order_button": "Apply current drag-and-drop list order to section ordering in predictions.",
            "slice_note_edit": "Optional atlas alignment note for selected slice (stored in predictions as atlas_note).",
            "save_slice_note_button": "Save current note for selected slice.",
            "apply_bad_sections_button": "Commit checked slices as bad_section flags in predictions.",
            "detect_outliers_button": "Auto-detect likely outlier sections from linearity residuals and confidence metrics.",
            "reset_flags_button": "Clear all bad-section checkboxes in the current filtered view.",
            "normalize_angles_button": "Propagate and normalize DV/ML angles using Gaussian-weighted center-depth averaging.",
            "enforce_order_button": "Reassign Oy depths to follow detected section index order while preserving spacing trends.",
            "enforce_spacing_button": "Space sections evenly according to index and selected section thickness.",
            "ml_spin": "Manual mediolateral angle override in degrees.",
            "dv_spin": "Manual dorsoventral angle override in degrees.",
            "apply_manual_angles_button": "Apply manual ML/DV angles to all predicted sections.",
            "undo_button": "Undo the most recent curation operation.",
            "redo_button": "Redo the latest undone curation operation.",
            "zoom_fit_button": "Reset plot zoom to include the full predicted-position range.",
            "zoom_in_button": "Zoom in on the predicted-position scatter plot.",
            "zoom_out_button": "Zoom out on the predicted-position scatter plot.",
            "enable_atlas_preview_checkbox": "Load and display atlas volume slices for visual comparison in curation.",
            "atlas_volume_combo": "Select atlas volume variant for preview (e.g., MRI/STPT/Nissl depending on species).",
            "enable_blend_overlay_checkbox": "Blend atlas intensity map directly over histology image for visual fit assessment.",
            "blend_slider": "Adjust atlas overlay opacity on histology preview.",
            "atlas_flip_x_checkbox": "Mirror the atlas horizontally before fitting to histology.",
            "atlas_flip_y_checkbox": "Mirror the atlas vertically before fitting to histology.",
            "atlas_rotate_combo": "Rotate atlas orientation before fitting and blending. Use this when tissue orientation differs from atlas slice orientation.",
            "atlas_scale_slider": "Scale atlas size relative to automatic fit (100% = default fit). Increase to zoom in, decrease to zoom out.",
            "atlas_offset_x_slider": "Shift atlas overlay horizontally after fitting to align landmarks.",
            "atlas_offset_y_slider": "Shift atlas overlay vertically after fitting to align landmarks.",
            "confidence_panel_toggle": "Show or hide the confidence overlay panel (red histology + green confidence support).",
            "before_after_toggle": "Show baseline vs edited depth information while curating.",
            "loupe_toggle": "Increase zoom response for detailed manual inspection of alignment edges.",

            "browse_output_dir_button": "Choose export directory for predictions and reports.",
            "output_dir_edit": "Target output folder. Must be writable before export.",
            "output_basename_edit": "Base filename used for JSON/XML/CSV and report outputs.",
            "output_format_combo": "Select primary output format: JSON (recommended) or legacy XML.",
            "output_format_help_button": "Show compatibility notes for JSON vs XML export formats.",
            "export_button": "Export predictions to selected format plus CSV sidecar.",
            "open_export_dir_button": "Open export directory in system file browser.",
            "copy_export_path_button": "Copy most recently exported primary file path to clipboard.",
            "report_button": "Generate PDF report using current summary and selected content options.",
            "preview_report_button": "Generate temporary report PDF and open it immediately for review.",
            "pdf_include_stats": "Include summary statistics section in generated PDF report.",
            "pdf_include_plot": "Include linearity plot section in generated PDF report.",
            "pdf_include_images": "Include sample image section in generated PDF report.",
            "pdf_include_angles": "Include angle-metrics notes section in generated PDF report.",
            "quicknii_path_edit": "Optional QuickNII executable path for one-click launch with exported JSON.",
            "quicknii_browse_button": "Locate QuickNII executable on disk.",
            "open_quicknii_button": "Launch QuickNII with latest JSON export file.",
        }

        for attr_name, text in tooltips.items():
            widget = getattr(self, attr_name, None)
            if widget is not None:
                widget.setToolTip(text)

        for widget in self.findChildren(QPushButton) + self.findChildren(QToolButton):
            if not widget.toolTip().strip():
                label = widget.text().strip() or "button"
                widget.setToolTip(f"Action control: {label}")

    @staticmethod
    def _infer_accessible_label(widget: QWidget) -> str:
        object_name = str(widget.objectName() or "").strip()
        if object_name:
            return object_name

        if hasattr(widget, "text"):
            try:
                text_value = str(widget.text()).strip()
                if text_value:
                    return text_value
            except Exception:
                pass

        if isinstance(widget, QLineEdit):
            placeholder = widget.placeholderText().strip()
            if placeholder:
                return placeholder

        if isinstance(widget, QComboBox):
            current = widget.currentText().strip()
            if current:
                return current

        return widget.__class__.__name__

    def _apply_accessibility_metadata(self):
        interactive_types = (
            QPushButton,
            QToolButton,
            QCheckBox,
            QRadioButton,
            QLineEdit,
            QComboBox,
            QDoubleSpinBox,
            QSpinBox,
            QSlider,
            QListWidget,
            QTableWidget,
            QTextEdit,
            QPlainTextEdit,
        )

        for widget in self.findChildren(interactive_types):
            if widget.focusPolicy() == Qt.NoFocus:
                continue

            if not str(widget.accessibleName() or "").strip():
                widget.setAccessibleName(self._infer_accessible_label(widget))

            if not str(widget.accessibleDescription() or "").strip():
                tooltip_text = str(widget.toolTip() or "").strip()
                if tooltip_text:
                    widget.setAccessibleDescription(tooltip_text)
                else:
                    widget.setAccessibleDescription(
                        f"Interactive {widget.__class__.__name__}"
                    )

    def _toggle_sidebar(self):
        is_visible = self.step_list.isVisible()
        self.step_list.setVisible(not is_visible)
        self.completion_label.setVisible(not is_visible)
        if not is_visible:
            self.collapse_sidebar_button.setText("<<")
            self.sidebar_container.setFixedWidth(230)
        else:
            self.collapse_sidebar_button.setText(">>")
            self.sidebar_container.setFixedWidth(30)

    def _build_top_bar(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("TopBar")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(12)

        try:
            from importlib.metadata import version
            self.app_version = version("DeepSlice")
        except Exception:
            self.app_version = "Unknown"

        self.project_label = QLabel(f"DeepSlice Desktop v{self.app_version}")
        self.project_label.setObjectName("ProjectLabel")

        self.session_status_label = QLabel("Session: New")
        self.session_status_label.setObjectName("SessionLabel")

        self.hardware_mode_label = QLabel("Mode: Detecting")
        self.hardware_mode_label.setObjectName("HardwareLabel")

        self.hardware_button = QPushButton("Hardware Health")
        self.hardware_button.clicked.connect(self._show_hardware_health)

        self.theme_toggle_button = QToolButton()
        self.theme_toggle_button.setText("Theme")
        self.theme_menu = QMenu(self.theme_toggle_button)
        self.theme_dark_action = self.theme_menu.addAction("Dark")
        self.theme_light_action = self.theme_menu.addAction("Light")
        self.theme_dark_action.triggered.connect(lambda: self._set_theme("dark"))
        self.theme_light_action.triggered.connect(lambda: self._set_theme("light"))
        self.theme_toggle_button.setMenu(self.theme_menu)
        self.theme_toggle_button.setPopupMode(QToolButton.InstantPopup)

        self.new_session_button = QPushButton("New Session")
        self.new_session_button.clicked.connect(self._reset_session)

        self.save_session_button = QPushButton("Save Session")
        self.save_session_button.clicked.connect(self._save_session)

        self.load_session_button = QPushButton("Load Session / QuickNII")
        self.load_session_menu = QMenu(self.load_session_button)
        self.load_session_action = self.load_session_menu.addAction("Browse...")
        self.load_session_action.triggered.connect(self._load_session_or_quint)
        self.load_session_menu.addSeparator()
        self.recent_sessions_actions = []
        for i in range(5):
            action = self.load_session_menu.addAction(f"Recent {i+1}")
            action.setVisible(False)
            self.recent_sessions_actions.append(action)
        self.load_session_button.setMenu(self.load_session_menu)
        self._update_recent_sessions_menu()

        self.session_io_spinner = QProgressBar()
        self.session_io_spinner.setRange(0, 0)
        self.session_io_spinner.setTextVisible(False)
        self.session_io_spinner.setMaximumWidth(70)
        self.session_io_spinner.setMaximumHeight(12)
        self.session_io_spinner.setVisible(False)

        self.shortcut_help_button = QToolButton()
        self.shortcut_help_button.setText("Shortcuts")
        self.shortcut_help_button.clicked.connect(self._show_shortcuts_help)

        self.preferences_button = QToolButton()
        self.preferences_button.setText("Preferences")
        self.preferences_button.clicked.connect(self._open_preferences_dialog)

        self.about_button = QToolButton()
        self.about_button.setText("About")
        self.about_button.clicked.connect(self._show_about_dialog)

        self.error_menu_button = QPushButton("Errors")
        self.error_menu = QMenu(self.error_menu_button)
        
        self._unread_error_count = 0
        
        self.open_log_action = self.error_menu.addAction("Open Error Log")
        self.open_log_action.triggered.connect(self._open_error_log)
        
        self.copy_error_action = self.error_menu.addAction("Copy Last Error")
        self.copy_error_action.triggered.connect(self._copy_last_error_report)
        self.copy_error_action.setEnabled(False)
        
        self.auto_fix_action = self.error_menu.addAction("Try Auto-Fix Last Error")
        self.auto_fix_action.triggered.connect(self._try_auto_fix_last_error)
        self.auto_fix_action.setEnabled(False)

        self.error_menu_button.setMenu(self.error_menu)
        self.error_menu_button.setObjectName("ErrorMenuButton")

        self.copy_error_button = self.copy_error_action
        self.auto_fix_button = self.auto_fix_action

        layout.addWidget(self.project_label)
        layout.addWidget(self.session_status_label)
        layout.addStretch(1)
        layout.addWidget(self.hardware_mode_label)
        layout.addWidget(self.theme_toggle_button)
        layout.addWidget(self.hardware_button)
        layout.addWidget(self.new_session_button)
        layout.addWidget(self.save_session_button)
        layout.addWidget(self.load_session_button)
        layout.addWidget(self.session_io_spinner)
        layout.addWidget(self.shortcut_help_button)
        layout.addWidget(self.preferences_button)
        layout.addWidget(self.about_button)
        layout.addWidget(self.error_menu_button)
        return frame

    def _update_recent_sessions_menu(self):
        settings = QSettings("DeepSlice", "GUI")
        recent = settings.value("recent_sessions", [])
        if not isinstance(recent, list):
            recent = []
        for i, action in enumerate(self.recent_sessions_actions):
            if i < len(recent):
                path = str(recent[i])
                stamp = ""
                try:
                    if os.path.exists(path):
                        stamp = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
                except Exception:
                    stamp = ""
                action.setText(f"{path} ({stamp})" if stamp else path)
                action.setToolTip(path)
                action.setVisible(True)
                try: action.triggered.disconnect()
                except Exception: pass
                # Lambda with default arg to capture the current path
                action.triggered.connect(lambda checked=False, path=path: self._load_session_file(path))
            else:
                action.setVisible(False)

    def _add_recent_session(self, path: str):
        settings = QSettings("DeepSlice", "GUI")
        recent = settings.value("recent_sessions", [])
        if not isinstance(recent, list):
            recent = []
        if path in recent:
            recent.remove(path)
        recent.insert(0, path)
        recent = recent[:5]
        settings.setValue("recent_sessions", recent)
        self._update_recent_sessions_menu()

    @staticmethod
    def _setting_to_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _apply_startup_preferences_to_state(self):
        settings = QSettings("DeepSlice", "GUI")
        preferred_species = str(settings.value("default_species", self.state.species)).strip().lower()
        if preferred_species in {"mouse", "rat"}:
            try:
                self.state.set_species(preferred_species)
            except Exception:
                pass

        try:
            batch_size = int(settings.value("inference_batch_size", self.state.inference_batch_size))
            if batch_size > 0:
                self.state.inference_batch_size = batch_size
        except Exception:
            pass

        outlier_sigma = settings.value("outlier_sigma", self.state.outlier_sigma_threshold)
        confidence_medium = settings.value("confidence_medium_threshold", self.state.confidence_medium_threshold)
        confidence_high = settings.value("confidence_high_threshold", self.state.confidence_high_threshold)
        try:
            self.state.set_quality_controls(
                outlier_sigma=float(outlier_sigma),
                confidence_medium=float(confidence_medium),
                confidence_high=float(confidence_high),
            )
        except Exception:
            pass

    @staticmethod
    def _coerce_int_list(value) -> Optional[List[int]]:
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            converted: List[int] = []
            for entry in value:
                try:
                    converted.append(int(entry))
                except Exception:
                    continue
            return converted if converted else None
        return None

    def _restore_window_preferences(self):
        settings = QSettings("DeepSlice", "GUI")
        geometry = settings.value("window_geometry", None)
        if geometry is not None:
            try:
                self.restoreGeometry(geometry)
            except Exception:
                pass

        body_sizes = self._coerce_int_list(settings.value("body_split_sizes", None))
        if body_sizes and hasattr(self, "body_split"):
            self.body_split.setSizes(body_sizes)

        curation_sizes = self._coerce_int_list(settings.value("curation_split_sizes", None))
        if curation_sizes and hasattr(self, "curation_vertical_split"):
            self.curation_vertical_split.setSizes(curation_sizes)

        always_console = self._setting_to_bool(settings.value("console_always_visible", False))
        if always_console and hasattr(self, "console_toggle"):
            self.console_toggle.setChecked(True)

    def _persist_window_preferences(self):
        settings = QSettings("DeepSlice", "GUI")
        settings.setValue("window_geometry", self.saveGeometry())
        if hasattr(self, "body_split"):
            settings.setValue("body_split_sizes", self.body_split.sizes())
        if hasattr(self, "curation_vertical_split"):
            settings.setValue("curation_split_sizes", self.curation_vertical_split.sizes())

    def _open_preferences_dialog(self):
        settings = QSettings("DeepSlice", "GUI")

        dialog = QDialog(self)
        dialog.setWindowTitle("Preferences")
        layout = QVBoxLayout(dialog)
        form = QFormLayout()

        species_combo = QComboBox()
        species_combo.addItems(["mouse", "rat"])
        current_default_species = str(settings.value("default_species", self.state.species)).strip().lower()
        if current_default_species in {"mouse", "rat"}:
            species_combo.setCurrentText(current_default_species)

        theme_combo = QComboBox()
        theme_combo.addItems(["dark", "light"])
        theme_combo.setCurrentText(self._theme_name)

        output_dir_row = QHBoxLayout()
        output_dir_edit = QLineEdit(str(settings.value("default_output_directory", "")).strip())
        output_dir_browse = QPushButton("Browse")

        def _browse_default_output_dir():
            selected = QFileDialog.getExistingDirectory(dialog, "Default Output Directory")
            if selected:
                output_dir_edit.setText(selected)

        output_dir_browse.clicked.connect(_browse_default_output_dir)
        output_dir_row.addWidget(output_dir_edit)
        output_dir_row.addWidget(output_dir_browse)

        quicknii_row = QHBoxLayout()
        quicknii_edit = QLineEdit(str(settings.value("quicknii_path", "")).strip())
        quicknii_browse = QPushButton("Browse")

        def _browse_default_quicknii_path():
            selected, _ = QFileDialog.getOpenFileName(
                dialog,
                "Default QuickNII Executable",
                "",
                "Executables (*.exe);;All Files (*)",
            )
            if selected:
                quicknii_edit.setText(selected)

        quicknii_browse.clicked.connect(_browse_default_quicknii_path)
        quicknii_row.addWidget(quicknii_edit)
        quicknii_row.addWidget(quicknii_browse)

        console_always_visible = QCheckBox("Show runtime console by default")
        console_always_visible.setChecked(
            self._setting_to_bool(settings.value("console_always_visible", False))
        )

        form.addRow("Default species for new sessions", species_combo)
        form.addRow("Theme", theme_combo)
        form.addRow("Default output directory", output_dir_row)
        form.addRow("Default QuickNII path", quicknii_row)
        form.addRow("Console", console_always_visible)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() != QDialog.Accepted:
            return

        selected_species = species_combo.currentText().strip().lower()
        selected_theme = theme_combo.currentText().strip().lower()
        default_output_dir = output_dir_edit.text().strip()
        default_quicknii_path = quicknii_edit.text().strip()
        always_console = bool(console_always_visible.isChecked())

        settings.setValue("default_species", selected_species)
        settings.setValue("theme", selected_theme)
        settings.setValue("default_output_directory", default_output_dir)
        settings.setValue("quicknii_path", default_quicknii_path)
        settings.setValue("console_always_visible", always_console)

        self._set_theme(selected_theme)
        if default_output_dir and hasattr(self, "output_dir_edit"):
            self.output_dir_edit.setText(default_output_dir)
        if hasattr(self, "quicknii_path_edit"):
            self.quicknii_path_edit.setText(default_quicknii_path)
        if hasattr(self, "console_toggle"):
            self.console_toggle.setChecked(always_console)

        # Keep current session stable: only apply species change immediately when no predictions exist.
        if self.state.predictions is None and selected_species in {"mouse", "rat"}:
            self.state.set_species(selected_species)
            self.mouse_radio.setChecked(selected_species == "mouse")
            self.rat_radio.setChecked(selected_species == "rat")
            self._refresh_atlas_volume_options()
            self._update_processing_estimate()

        self._show_toast("Preferences updated", timeout_ms=3000)

    def _show_about_dialog(self):
        text = (
            f"DeepSlice Desktop v{self.app_version}\n\n"
            "Citation:\n"
            "- Carey et al., Nature Communications (2023)\n"
            "  https://www.nature.com/articles/s41467-023-41645-4\n"
            "- Wang et al., Cell (2020) for Allen CCFv3 mouse atlas\n"
            "- Kleven et al., Nature Methods (2023) for Waxholm rat atlas\n\n"
            "License: See LICENSE file in this repository.\n"
            "Documentation: docs/gui_help.html or built docs/index.html"
        )
        QMessageBox.information(self, "About DeepSlice", text)


    def _setup_shortcuts(self):
        self._shortcuts = []
        for i in range(5):
            shortcut = QShortcut(QKeySequence(f"Ctrl+{i+1}"), self)
            shortcut.activated.connect(lambda index=i: self.step_list.setCurrentRow(index))
            self._shortcuts.append(shortcut)

        help_shortcut = QShortcut(QKeySequence("Ctrl+/"), self)
        help_shortcut.activated.connect(self._show_shortcuts_help)
        self._shortcuts.append(help_shortcut)

        help_shortcut_alt = QShortcut(QKeySequence("Ctrl+?"), self)
        help_shortcut_alt.activated.connect(self._show_shortcuts_help)
        self._shortcuts.append(help_shortcut_alt)

        new_shortcut = QShortcut(QKeySequence.New, self)
        new_shortcut.activated.connect(self._reset_session)
        self._shortcuts.append(new_shortcut)

        save_shortcut = QShortcut(QKeySequence.Save, self)
        save_shortcut.activated.connect(self._save_session)
        self._shortcuts.append(save_shortcut)

        load_shortcut = QShortcut(QKeySequence.Open, self)
        load_shortcut.activated.connect(self._load_session_or_quint)
        self._shortcuts.append(load_shortcut)

        export_shortcut = QShortcut(QKeySequence("Ctrl+E"), self)
        export_shortcut.activated.connect(lambda: self.step_list.setCurrentRow(4))
        self._shortcuts.append(export_shortcut)

        f1_shortcut = QShortcut(QKeySequence("F1"), self)
        f1_shortcut.activated.connect(self._open_context_help)
        self._shortcuts.append(f1_shortcut)

        undo_shortcut = QShortcut(QKeySequence.Undo, self)
        undo_shortcut.activated.connect(self._undo)
        self._shortcuts.append(undo_shortcut)

        redo_shortcut = QShortcut(QKeySequence.Redo, self)
        redo_shortcut.activated.connect(self._redo)
        self._shortcuts.append(redo_shortcut)

    def _show_shortcuts_help(self):
        text = (
            "Keyboard Shortcuts:\n\n"
            "Ctrl+1 to Ctrl+5 : Navigate between pages\n"
            "Ctrl+N : New session\n"
            "Ctrl+S : Save session\n"
            "Ctrl+O : Load session\n"
            "Ctrl+E : Jump to Export page\n"
            "Ctrl+Z : Undo curation changes\n"
            "Ctrl+Y : Redo curation changes\n"
            "Ctrl+/ or Ctrl+? : Show shortcuts\n"
            "F1 : Open page-specific help"
        )
        QMessageBox.information(self, "Shortcuts", text)

    def _open_context_help(self):
        anchors = {
            0: "#ingestion",
            1: "#configuration",
            2: "#prediction",
            3: "#curation",
            4: "#export",
        }
        current = int(self.stack.currentIndex()) if hasattr(self, "stack") else 0
        anchor = anchors.get(current, "#global")

        help_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "docs", "gui_help.html")
        )
        if os.path.exists(help_path):
            url = QUrl.fromLocalFile(help_path)
            url.setFragment(anchor.lstrip("#"))
            QDesktopServices.openUrl(url)
            return

        docs_index = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "docs", "_build", "html", "index.html")
        )
        if os.path.exists(docs_index):
            QDesktopServices.openUrl(QUrl.fromLocalFile(docs_index))
            return

        self._show_toast("Documentation file not found in this workspace", timeout_ms=3200, level="warning")

    def _load_theme_preference(self) -> str:
        settings = QSettings("DeepSlice", "GUI")
        theme = str(settings.value("theme", "dark")).strip().lower()
        return theme if theme in {"dark", "light"} else "dark"

    def _save_theme_preference(self, theme: str):
        settings = QSettings("DeepSlice", "GUI")
        settings.setValue("theme", theme)

    def _set_theme(self, theme: str):
        theme = str(theme).strip().lower()
        if theme not in {"dark", "light"}:
            return
        self._theme_name = theme
        self._save_theme_preference(theme)
        self._apply_theme()

    def _show_startup_dialogs(self):
        settings = QSettings("DeepSlice", "GUI")
        onboarding_complete = bool(settings.value("onboarding_complete", False))
        if not onboarding_complete:
            self._show_onboarding_dialog()
            settings.setValue("onboarding_complete", True)

        last_seen_version = str(settings.value("last_seen_version", "")).strip()
        if last_seen_version != self.app_version:
            self._show_whats_new_dialog(last_seen_version)
            settings.setValue("last_seen_version", self.app_version)

    def _show_onboarding_dialog(self):
        steps = "\n".join(
            [
                "1. Ingestion: add images and check section indices.",
                "2. Configuration: choose species and prediction options.",
                "3. Prediction: run alignment and monitor progress/logs.",
                "4. Curation: review confidence, adjust flags and angles.",
                "5. Export: write JSON/XML/CSV and reports.",
            ]
        )
        QMessageBox.information(
            self,
            "Welcome to DeepSlice",
            "First-run walkthrough:\n\n"
            + steps
            + "\n\nTips: Use Ctrl+1..Ctrl+5 to jump between steps and F1 for contextual help.",
        )

    def _show_whats_new_dialog(self, previous_version: str):
        previous = previous_version if previous_version else "first run"
        notes = (
            f"Updated from: {previous}\n"
            f"Current version: {self.app_version}\n\n"
            "Highlights:\n"
            "- Better keyboard navigation and shortcut help\n"
            "- Session safety with unsaved-change indicators\n"
            "- Expanded curation and export controls\n"
            "- Improved runtime feedback and progress reporting"
        )
        QMessageBox.information(self, "What's New", notes)

    def _show_toast(self, text: str, timeout_ms: int = 3500, level: str = "info"):
        self.status_bar.showMessage(text, int(timeout_ms))
        if hasattr(self, "toast_overlay"):
            self.toast_overlay.show_message(text, timeout_ms=int(timeout_ms), level=level)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "toast_overlay"):
            self.toast_overlay._reposition()
        
    def _reset_session(self):
        if getattr(self.state, "is_dirty", False):
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Start a new session anyway?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
        self.state = DeepSliceAppState()
        self._apply_startup_preferences_to_state()
        self._baseline_predictions = None
        self._curation_modified = False
        self._anchor_depth_targets = {}
        self._session_base_text = "Session: New"
        self._apply_state_to_widgets()
        self._update_session_status()
        self._refresh_all_views()
        
    def _update_session_status(self):
        text = self._session_base_text
        is_dirty = getattr(self.state, "is_dirty", False)
        if is_dirty:
            text = f"● {text}"
        self.session_status_label.setText(text)
        self.setWindowTitle(f"{self._window_title_base}{' *' if is_dirty else ''}")

    def closeEvent(self, event):
        if getattr(self.state, "is_dirty", False):
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Are you sure you want to exit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self._persist_window_preferences()
                event.accept()
            else:
                event.ignore()
        else:
            self._persist_window_preferences()
            event.accept()
            
    def _record_error(self, context: str, error_text: str):
        clean_context = str(context).strip() or "DeepSlice error"
        clean_text = str(error_text).strip() or "No additional error details were provided."
        analysis = self._error_autofixer.analyze_error(clean_context, clean_text)
        analysis_text = self._error_autofixer.format_analysis(analysis)

        report_body = clean_text
        if analysis_text:
            report_body = f"{clean_text}\n\nAuto Analysis:\n{analysis_text}"

        if hasattr(self, "console_output"):
            self._append_console_log(f"[ERROR] {clean_context}")
            self._append_console_log(clean_text)
            if analysis.get("summary"):
                self._append_console_log(f"[ANALYSIS] {analysis['summary']}")

        self._last_error_report = build_error_report(
            context=clean_context,
            error_text=report_body,
            log_path=self.error_log_path,
        )
        self._last_error_context = clean_context
        self._last_error_text = clean_text
        self._last_error_analysis = analysis

        if hasattr(self, "copy_error_button"):
            self.copy_error_button.setEnabled(True)
        if hasattr(self, "auto_fix_button"):
            self.auto_fix_button.setEnabled(bool(analysis.get("auto_fix_available", False)))
            
        if hasattr(self, "_unread_error_count") and hasattr(self, "error_menu_button"):
            self._unread_error_count += 1
            self.error_menu_button.setText(f"Errors ({self._unread_error_count})")
            self.error_menu_button.setStyleSheet("QPushButton { background-color: #A43344; }")

        log_error_text(clean_context, report_body)

    def _copy_last_error_report(self, show_message: bool = True):
        report = self._last_error_report.strip()
        if not report:
            tail_text = read_error_log_tail()
            if not tail_text.strip():
                tail_text = "No logged errors found yet."
            report = build_error_report(
                context="DeepSlice log tail",
                error_text=tail_text,
                log_path=self.error_log_path,
            )

        QApplication.clipboard().setText(report)

        if show_message:
            self._show_toast(
                "Error report copied to clipboard",
                timeout_ms=2500,
                level="success",
            )

    def _try_auto_fix_last_error(self):
        if not self._last_error_text:
            self._show_toast("No previous error available for auto-fix", timeout_ms=2600, level="warning")
            return

        self._start_auto_fix(
            context=self._last_error_context or "Last recorded error",
            error_text=self._last_error_text,
        )

    def _start_auto_fix(self, context: str, error_text: str):
        analysis = self._error_autofixer.analyze_error(context, error_text)
        if not analysis.get("auto_fix_available", False):
            details = self._error_autofixer.format_analysis(analysis)
            if details:
                self._append_console_log(f"[AUTO-FIX] {details}")
            self._show_toast("No automatic fix is available", timeout_ms=3200, level="info")
            return

        self._append_console_log("Starting automatic error-fix attempt...")
        self.auto_fix_button.setEnabled(False)

        worker = FunctionWorker(self._auto_fix_task, context, error_text)
        worker.signals.finished.connect(self._on_auto_fix_finished)
        worker.signals.error.connect(self._on_auto_fix_error)
        self._track_worker(worker)
        self.thread_pool.start(worker)

    def _auto_fix_task(self, context: str, error_text: str):
        return self._error_autofixer.try_auto_fix(context, error_text)

    def _on_auto_fix_finished(self, result: dict):
        analysis = result.get("analysis", {}) or {}
        self.auto_fix_button.setEnabled(bool(analysis.get("auto_fix_available", False)))

        summary = str(result.get("summary", "Automatic fix finished.")).strip()
        details = str(result.get("details", "")).strip()
        combined = summary if not details else f"{summary}\n\n{details}"

        if bool(result.get("succeeded", False)):
            self._append_console_log(f"[AUTO-FIX] {summary}")
            self._show_toast("Auto-fix succeeded", timeout_ms=3200, level="success")
            return

        if bool(result.get("attempted", False)):
            self._show_logged_error(
                title="Auto-Fix Failed",
                context="Automatic fix attempt failed",
                error_text=combined,
                icon=QMessageBox.Warning,
            )
            return

        self._show_toast("Auto-fix completed with no changes", timeout_ms=3000, level="info")

    def _on_auto_fix_error(self, error_text: str):
        self.auto_fix_button.setEnabled(bool(self._last_error_analysis and self._last_error_analysis.get("auto_fix_available", False)))
        self._show_logged_error(
            title="Auto-Fix Error",
            context="Automatic fix process crashed",
            error_text=error_text,
            icon=QMessageBox.Warning,
        )

    def _open_error_log(self):
        if hasattr(self, "_unread_error_count"):
            self._unread_error_count = 0
            self.error_menu_button.setText("Errors")
            self.error_menu_button.setStyleSheet("")

        if not os.path.exists(self.error_log_path):
            self._show_toast("No error log file exists yet", timeout_ms=2800, level="warning")
            return

        try:
            if os.name == "nt":
                os.startfile(self.error_log_path)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", self.error_log_path])
        except Exception as exc:
            self._show_logged_exception(
                title="Open Error Log",
                context="Unable to open the error log file",
                exc=exc,
                icon=QMessageBox.Warning,
            )

    def _show_logged_error(
        self,
        title: str,
        context: str,
        error_text: str,
        icon=QMessageBox.Critical,
    ):
        self._record_error(context, error_text)
        analysis = self._last_error_analysis or {}
        analysis_text = self._error_autofixer.format_analysis(analysis)
        can_auto_fix = bool(analysis.get("auto_fix_available", False))

        message_box = QMessageBox(self)
        message_box.setWindowTitle(title)
        message_box.setIcon(icon)
        message_box.setText(
            (
                f"{context}\n\n"
                f"Details have been written to:\n{self.error_log_path}\n\n"
                f"Analysis: {analysis.get('summary', 'No automatic pattern match found.')}"
            )
        )
        detail_blocks = [str(error_text)]
        if analysis_text:
            detail_blocks.append("Auto Analysis\n" + analysis_text)
        message_box.setDetailedText("\n\n".join(detail_blocks))
        message_box.setStandardButtons(QMessageBox.Ok)
        copy_button = message_box.addButton("Copy Error Report", QMessageBox.ActionRole)
        auto_fix_button = None
        if can_auto_fix:
            auto_fix_button = message_box.addButton("Try Auto-Fix", QMessageBox.ActionRole)
        message_box.exec()

        if message_box.clickedButton() == copy_button:
            self._copy_last_error_report(show_message=False)
        elif auto_fix_button is not None and message_box.clickedButton() == auto_fix_button:
            self._start_auto_fix(context=context, error_text=error_text)

    def _show_logged_exception(
        self,
        title: str,
        context: str,
        exc: BaseException,
        icon=QMessageBox.Critical,
    ):
        error_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self._show_logged_error(title, context, error_text, icon=icon)

    def _build_ingestion_page(self) -> QWidget:
        page = QWidget()
        split = QSplitter(Qt.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(8)

        self.drop_area = DropArea()
        self.drop_area.pathsDropped.connect(self._handle_dropped_paths)
        left_layout.addWidget(self.drop_area)

        button_row = QHBoxLayout()
        self.add_folder_button = QPushButton("Add Folder")
        self.add_folder_button.clicked.connect(self._add_folder)
        self.add_files_button = QPushButton("Add Files")
        self.add_files_button.clicked.connect(self._add_files)
        self.clear_images_button = QPushButton("Clear All")
        self.clear_images_button.clicked.connect(self._clear_images)
        button_row.addWidget(self.add_folder_button)
        button_row.addWidget(self.add_files_button)
        button_row.addWidget(self.clear_images_button)
        left_layout.addLayout(button_row)

        options_group = QGroupBox("Pre-flight Options")
        options_layout = QVBoxLayout(options_group)
        
        section_number_layout = QHBoxLayout()
        self.enable_section_numbers_checkbox = QCheckBox(
            "Detect section numbers from filename (_sXXX)"
        )
        self.enable_section_numbers_checkbox.setChecked(True)
        self.enable_section_numbers_checkbox.toggled.connect(self._update_run_button_state)
        self.naming_helper_button = QToolButton()
        self.naming_helper_button.setText("?")
        self.naming_helper_button.setToolTip("Help with naming conventions")
        self.naming_helper_button.clicked.connect(self._show_naming_helper)
        section_number_layout.addWidget(self.enable_section_numbers_checkbox)
        section_number_layout.addWidget(self.naming_helper_button)
        section_number_layout.addStretch(1)

        self.legacy_parsing_checkbox = QCheckBox(
            "Legacy parser fallback (last 3 digits)"
        )
        self.legacy_parsing_checkbox.toggled.connect(self._refresh_ingestion_views)
        self.orientation_combo = QComboBox()
        self.orientation_combo.addItems(
            [
                "Coronal (supported)",
                "Sagittal (coming soon)",
                "Horizontal (coming soon)",
            ]
        )
        self.orientation_combo.currentIndexChanged.connect(self._update_run_button_state)
        self.orientation_guide_button = QToolButton()
        self.orientation_guide_button.setText("Guide")
        self.orientation_guide_button.clicked.connect(self._show_orientation_guide)
        options_layout.addLayout(section_number_layout)
        options_layout.addWidget(self.legacy_parsing_checkbox)
        orientation_row = QHBoxLayout()
        orientation_row.addWidget(self.orientation_combo)
        orientation_row.addWidget(self.orientation_guide_button)
        options_layout.addLayout(orientation_row)
        left_layout.addWidget(options_group)

        self.slice_count_label = QLabel("Slices: 0")
        self.file_size_summary_label = QLabel("0 files loaded - 0 B")
        self.ingestion_summary_banner = QLabel("No files loaded")
        self.ingestion_summary_banner.setObjectName("SummaryBanner")
        self.ingestion_warning_label = QLabel("")
        self.ingestion_warning_label.setWordWrap(True)
        self.ingestion_warning_label.setObjectName("WarningText")
        left_layout.addWidget(self.slice_count_label)
        left_layout.addWidget(self.file_size_summary_label)
        left_layout.addWidget(self.ingestion_summary_banner)
        left_layout.addWidget(self.ingestion_warning_label)

        self.index_table = QTableWidget(0, 3)
        self.index_table.setHorizontalHeaderLabels(["Filename", "Detected Index", "Status"])
        self.index_table.horizontalHeader().setStretchLastSection(True)
        self.index_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.index_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.index_table.setSortingEnabled(True)
        left_layout.addWidget(self.index_table, stretch=1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)
        
        thumbnail_header = QHBoxLayout()
        thumbnail_header.addWidget(QLabel("Thumbnail Grid (cached previews)"))

        self.thumbnail_sort_combo = QComboBox()
        self.thumbnail_sort_combo.addItems(
            [
                "Sort: Filename",
                "Sort: Detected Index",
                "Sort: Modified Date",
            ]
        )
        self.thumbnail_sort_combo.currentIndexChanged.connect(self._refresh_ingestion_views)
        thumbnail_header.addWidget(self.thumbnail_sort_combo)

        self.thumbnail_filter_edit = QLineEdit()
        self.thumbnail_filter_edit.setPlaceholderText("Filter by filename...")
        self.thumbnail_filter_edit.setFixedWidth(200)
        self.thumbnail_filter_edit.textChanged.connect(self._filter_thumbnails)
        thumbnail_header.addWidget(self.thumbnail_filter_edit)
        
        right_layout.addLayout(thumbnail_header)

        self.thumbnail_list = ThumbnailListWidget()
        self.thumbnail_list.setViewMode(QListWidget.IconMode)
        self.thumbnail_list.setIconSize(QSize(256, 256))
        self.thumbnail_list.setResizeMode(QListWidget.Adjust)
        self.thumbnail_list.setSpacing(8)
        self.thumbnail_list.itemSelectionChanged.connect(self._on_thumbnail_selection_changed)
        self.thumbnail_list.itemDoubleClicked.connect(self._open_fullscreen_thumbnail_preview)
        self.thumbnail_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.thumbnail_list.customContextMenuRequested.connect(self._show_thumbnail_context_menu)
        self.thumbnail_list.pathsDropped.connect(self._handle_dropped_paths)

        self.thumbnail_progress = QProgressBar()
        self.thumbnail_progress.setRange(0, 100)
        self.thumbnail_progress.setValue(0)
        self.thumbnail_progress.setVisible(False)

        right_layout.addWidget(self.thumbnail_progress)
        right_layout.addWidget(self.thumbnail_list, stretch=1)

        self.ingestion_preview = SliceGraphicsView()
        right_layout.addWidget(self.ingestion_preview, stretch=2)

        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 3)

        root = QVBoxLayout(page)
        root.addWidget(split)
        return page

    def _show_naming_helper(self):
        text = (
            "Naming Convention Help:\n\n"
            "DeepSlice can automatically detect the section index from filenames if they follow the pattern `_sXXX`.\n"
            "For example:\n"
            "  - `brain1_s001.png` -> Index 1\n"
            "  - `mouse_A_s142_fluoro.tif` -> Index 142\n\n"
            "If this fails, you can try the 'Legacy parser fallback' which looks at the last 3 digits in the filename."
        )
        QMessageBox.information(self, "Naming Convention", text)

    def _show_orientation_guide(self):
        guide = (
            "Orientation Guide:\n\n"
            "Coronal: front-to-back slice stack (supported now).\n"
            "Sagittal: left-to-right slice stack.\n"
            "Horizontal: top-to-bottom slice stack.\n\n"
            "Tip: current release is optimized for coronal workflows."
        )
        QMessageBox.information(self, "Orientation Guide", guide)

    def _show_direction_guide(self):
        text = (
            "Direction Override Guide:\n\n"
            "rostro-caudal: index numbers increase from rostral toward caudal.\n"
            "caudal-rostro: index numbers increase from caudal toward rostral.\n"
            "Auto: inferred from predicted AP depths."
        )
        QMessageBox.information(self, "Direction Guide", text)

    def _show_ensemble_explanation(self):
        QMessageBox.information(
            self,
            "Ensemble Prediction",
            "Ensemble runs both primary and secondary models and averages outputs.\n\n"
            "Pros: often higher stability/accuracy.\n"
            "Cons: slower and higher memory use.",
        )

    def _show_configuration_validation(self):
        errors, warnings = self._validate_before_prediction()
        if len(errors) == 0 and len(warnings) == 0:
            self._show_toast("Configuration looks good", timeout_ms=2500, level="success")
            return
        lines = []
        if errors:
            lines.append("Errors:")
            lines.extend([f"- {item}" for item in errors])
        if warnings:
            if lines:
                lines.append("")
            lines.append("Warnings:")
            lines.extend([f"- {item}" for item in warnings])
        QMessageBox.warning(self, "Validation", "\n".join(lines))

    def _show_export_format_help(self):
        QMessageBox.information(
            self,
            "Export Format Help",
            "JSON: recommended for QuickNII and VisuAlign pipelines.\n"
            "Legacy XML: compatibility mode for older tooling.\n\n"
            "CSV sidecar is always generated.",
        )

    @staticmethod
    def _format_bytes(total_bytes: int) -> str:
        units = ["B", "KB", "MB", "GB", "TB"]
        value = float(max(total_bytes, 0))
        for unit in units:
            if value < 1024.0 or unit == units[-1]:
                if unit == "B":
                    return f"{int(value)} {unit}"
                return f"{value:.1f} {unit}"
            value /= 1024.0
        return "0 B"

    @staticmethod
    def _infer_bit_depth(image_mode: str) -> str:
        mapping = {
            "1": "1-bit",
            "L": "8-bit",
            "P": "8-bit indexed",
            "I;16": "16-bit",
            "I": "32-bit integer",
            "F": "32-bit float",
            "RGB": "24-bit (8x3)",
            "RGBA": "32-bit (8x4)",
        }
        return mapping.get(str(image_mode), str(image_mode))

    def _get_pixel_spacing_um(self, image_path: Optional[str]) -> Optional[float]:
        if image_path is None:
            return None
        try:
            with Image.open(image_path) as image_handle:
                x_res = image_handle.info.get("dpi", None)
                if isinstance(x_res, tuple) and len(x_res) >= 1 and x_res[0] > 0:
                    # 1 inch = 25,400 um
                    return 25400.0 / float(x_res[0])

                tags = getattr(image_handle, "tag_v2", None)
                if tags is not None:
                    x_resolution = tags.get(282, None)  # XResolution
                    resolution_unit = tags.get(296, 2)  # ResolutionUnit (2=inches, 3=cm)
                    if x_resolution:
                        if isinstance(x_resolution, tuple) and len(x_resolution) == 2:
                            numerator, denominator = x_resolution
                            x_resolution = float(numerator) / float(denominator)
                        x_resolution = float(x_resolution)
                        if x_resolution > 0:
                            if int(resolution_unit) == 3:
                                return 10000.0 / x_resolution
                            return 25400.0 / x_resolution
        except Exception:
            return None
        return None

    def _make_species_preview_pixmap(self, species: str) -> QPixmap:
        pix = QPixmap(84, 42)
        pix.fill(QColor("#11161D"))
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.Antialiasing)
        border = QColor("#5C738C") if species == "mouse" else QColor("#7D6C4D")
        fill = QColor("#2B3E57") if species == "mouse" else QColor("#4A3A24")
        painter.setPen(QPen(border, 1.5))
        painter.setBrush(fill)
        painter.drawRoundedRect(2, 2, 80, 38, 6, 6)
        painter.setPen(QColor("#E7EEF8"))
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        label = "Mouse" if species == "mouse" else "Rat"
        painter.drawText(pix.rect(), Qt.AlignCenter, label + " atlas")
        painter.end()
        return pix

    def _estimate_runtime_seconds(self) -> int:
        count = len(self.state.image_paths)
        if count == 0:
            return 0

        hardware_mode = "cpu"
        if "GPU" in self.hardware_mode_label.text().upper():
            hardware_mode = "gpu"

        per_slice = 0.40 if hardware_mode == "gpu" else 1.20
        if self.ensemble_checkbox.isChecked():
            per_slice *= 1.8
        elif self.secondary_model_checkbox.isChecked():
            per_slice *= 1.1

        batch_size = int(getattr(self, "inference_batch_spin", None).value() if hasattr(self, "inference_batch_spin") else 8)
        batch_scale = 8.0 / float(max(batch_size, 1))
        per_slice *= np.clip(batch_scale, 0.35, 2.0)

        return int(max(1, round(count * per_slice)))

    def _update_processing_estimate(self):
        if not hasattr(self, "processing_estimate_label"):
            return
        total_seconds = self._estimate_runtime_seconds()
        if total_seconds <= 0:
            self.processing_estimate_label.setText("Estimated processing time: -")
            self.slice_count_reminder_label.setText("Will process 0 slices")
            return
        mode = "GPU" if "GPU" in self.hardware_mode_label.text().upper() else "CPU"
        self.processing_estimate_label.setText(
            f"Estimated processing time: ~{self._format_duration(total_seconds)} on {mode}"
        )
        self.slice_count_reminder_label.setText(f"Will process {len(self.state.image_paths)} slices")

    def _open_fullscreen_thumbnail_preview(self, item: QListWidgetItem):
        image_path = str(item.data(Qt.UserRole))
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Preview - {os.path.basename(image_path)}")
        dialog.setWindowState(Qt.WindowFullScreen)
        root = QVBoxLayout(dialog)
        viewer = SliceGraphicsView()
        viewer.set_image(image_path)
        info = QLabel(image_path)
        info.setWordWrap(True)
        controls = QDialogButtonBox(QDialogButtonBox.Close)
        controls.rejected.connect(dialog.reject)
        controls.accepted.connect(dialog.accept)
        controls.button(QDialogButtonBox.Close).setText("Close")
        root.addWidget(viewer, stretch=1)
        root.addWidget(info)
        root.addWidget(controls)
        dialog.exec()

    def _clear_console(self):
        self.console_output.clear()

    def _copy_console(self):
        QApplication.clipboard().setText(self.console_output.toPlainText())

    @staticmethod
    def _format_duration(total_seconds: int) -> str:
        total_seconds = int(max(total_seconds, 0))
        minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def _filter_thumbnails(self, text: str):
        query = text.lower()
        for i in range(self.thumbnail_list.count()):
            item = self.thumbnail_list.item(i)
            filename = item.text().lower()
            item.setHidden(query not in filename)

    def _show_thumbnail_context_menu(self, pos):
        item = self.thumbnail_list.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)
        remove_action = menu.addAction("Remove this image")
        action = menu.exec(self.thumbnail_list.viewport().mapToGlobal(pos))
        if action == remove_action:
            image_path = item.data(Qt.UserRole)
            self.state.remove_image(image_path)
            self._refresh_all_views()

    def _build_configuration_page(self) -> QWidget:
        page = QWidget()
        split = QSplitter(Qt.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)

        species_group = QGroupBox("Species / Atlas")
        species_layout = QVBoxLayout(species_group)
        self.mouse_radio = QRadioButton("Mouse (Allen CCFv3)")
        self.rat_radio = QRadioButton("Rat (Waxholm Rat Atlas)")
        self.mouse_radio.setChecked(True)
        self.mouse_radio.toggled.connect(self._on_species_changed)
        self.rat_radio.toggled.connect(self._on_species_changed)

        mouse_row = QHBoxLayout()
        self.mouse_preview_label = QLabel()
        self.mouse_preview_label.setPixmap(self._make_species_preview_pixmap("mouse"))
        mouse_row.addWidget(self.mouse_radio)
        mouse_row.addStretch(1)
        mouse_row.addWidget(self.mouse_preview_label)

        rat_row = QHBoxLayout()
        self.rat_preview_label = QLabel()
        self.rat_preview_label.setPixmap(self._make_species_preview_pixmap("rat"))
        self.rat_beta_badge = QLabel("Beta")
        self.rat_beta_badge.setStyleSheet(
            "QLabel {"
            " background-color: #B5651D;"
            " color: #FFFFFF;"
            " padding: 1px 6px;"
            " border-radius: 6px;"
            " font-size: 10px;"
            " font-weight: bold;"
            "}"
        )
        self.rat_beta_badge.setToolTip(
            "Rat support is in beta. Model weights are still being refined "
            "and ensemble inference is not yet available for rat. See the "
            "README 'Rat Support' section for the current status."
        )
        rat_row.addWidget(self.rat_radio)
        rat_row.addWidget(self.rat_beta_badge)
        rat_row.addStretch(1)
        rat_row.addWidget(self.rat_preview_label)

        species_layout.addLayout(mouse_row)
        species_layout.addLayout(rat_row)
        left_layout.addWidget(species_group)

        geometry_group = QGroupBox("Physical Geometry")
        geometry_layout = QFormLayout(geometry_group)
        self.auto_thickness_checkbox = QCheckBox("Auto-estimate thickness")
        self.auto_thickness_checkbox.setChecked(True)
        self.auto_thickness_checkbox.toggled.connect(self._on_auto_thickness_toggled)
        self.thickness_spin = QDoubleSpinBox()
        self.thickness_spin.setRange(0.0, 1000.0)
        self.thickness_spin.setDecimals(2)
        self.thickness_spin.setSuffix(" um")
        self.thickness_spin.setEnabled(False)
        self.thickness_hint_label = QLabel("Typical range: 20-100 um for standard histology")
        self.thickness_hint_label.setObjectName("HintText")
        self.suggest_thickness_button = QPushButton("Suggest Thickness")
        self.suggest_thickness_button.clicked.connect(self._suggest_thickness)
        self.detected_direction_label = QLabel("Detected direction: unknown")
        self.direction_override_combo = QComboBox()
        self.direction_override_combo.addItems(
            ["Auto", "rostro-caudal", "caudal-rostro"]
        )
        self.direction_override_combo.currentTextChanged.connect(
            self._on_direction_override_changed
        )
        self.direction_guide_button = QToolButton()
        self.direction_guide_button.setText("Guide")
        self.direction_guide_button.clicked.connect(self._show_direction_guide)
        geometry_layout.addRow(self.auto_thickness_checkbox)
        geometry_layout.addRow("Section Thickness", self.thickness_spin)
        geometry_layout.addRow("", self.thickness_hint_label)
        geometry_layout.addRow(self.suggest_thickness_button)
        geometry_layout.addRow(self.detected_direction_label)
        direction_row = QHBoxLayout()
        direction_row.addWidget(self.direction_override_combo)
        direction_row.addWidget(self.direction_guide_button)
        geometry_layout.addRow("Direction Override", direction_row)
        left_layout.addWidget(geometry_group)

        prediction_group = QGroupBox("Prediction Modes")
        prediction_layout = QVBoxLayout(prediction_group)
        ensemble_row = QHBoxLayout()
        self.ensemble_checkbox = QCheckBox("Ensemble prediction (if available)")
        self.ensemble_checkbox.setChecked(self.state.supports_ensemble())
        self.ensemble_checkbox.toggled.connect(self._update_processing_estimate)
        self.ensemble_help_button = QToolButton()
        self.ensemble_help_button.setText("?")
        self.ensemble_help_button.clicked.connect(self._show_ensemble_explanation)
        ensemble_row.addWidget(self.ensemble_checkbox)
        ensemble_row.addWidget(self.ensemble_help_button)
        ensemble_row.addStretch(1)

        self.secondary_model_checkbox = QCheckBox(
            "Use secondary model only (for comparison)"
        )
        self.secondary_model_checkbox.setChecked(False)
        self.secondary_model_checkbox.toggled.connect(self._update_processing_estimate)
        self.secondary_model_checkbox.setToolTip(
            "Runs only the secondary model weights. Do not use together with ensemble."
        )
        self.legacy_from_config_checkbox = QCheckBox(
            "Legacy section-number parser"
        )
        self.legacy_from_config_checkbox.setToolTip(
            "Uses the legacy parser that reads the trailing 3 digits in filenames (for older datasets not using _sXXX naming)."
        )
        self.legacy_from_config_checkbox.toggled.connect(self._sync_legacy_checkbox)
        prediction_layout.addLayout(ensemble_row)
        prediction_layout.addWidget(self.secondary_model_checkbox)
        prediction_layout.addWidget(self.legacy_from_config_checkbox)
        left_layout.addWidget(prediction_group)
        self._update_ensemble_availability()

        quality_group = QGroupBox("Quality and Runtime")
        quality_layout = QFormLayout(quality_group)

        self.outlier_sigma_spin = QDoubleSpinBox()
        self.outlier_sigma_spin.setRange(1.0, 3.0)
        self.outlier_sigma_spin.setDecimals(2)
        self.outlier_sigma_spin.setSingleStep(0.1)
        self.outlier_sigma_spin.setValue(float(self.state.outlier_sigma_threshold))
        self.outlier_sigma_spin.setSuffix(" sigma")
        self.outlier_sigma_spin.valueChanged.connect(self._on_quality_controls_changed)

        self.confidence_high_spin = QDoubleSpinBox()
        self.confidence_high_spin.setRange(0.55, 0.99)
        self.confidence_high_spin.setDecimals(2)
        self.confidence_high_spin.setSingleStep(0.01)
        self.confidence_high_spin.setValue(float(self.state.confidence_high_threshold))
        self.confidence_high_spin.valueChanged.connect(self._on_quality_controls_changed)

        self.confidence_medium_spin = QDoubleSpinBox()
        self.confidence_medium_spin.setRange(0.20, 0.90)
        self.confidence_medium_spin.setDecimals(2)
        self.confidence_medium_spin.setSingleStep(0.01)
        self.confidence_medium_spin.setValue(float(self.state.confidence_medium_threshold))
        self.confidence_medium_spin.valueChanged.connect(self._on_quality_controls_changed)

        self.inference_batch_spin = QSpinBox()
        self.inference_batch_spin.setRange(1, 64)
        self.inference_batch_spin.setValue(int(self.state.inference_batch_size))
        self.inference_batch_spin.valueChanged.connect(self._on_inference_batch_changed)

        quality_layout.addRow("Outlier sensitivity", self.outlier_sigma_spin)
        quality_layout.addRow("High confidence >=", self.confidence_high_spin)
        quality_layout.addRow("Medium confidence >=", self.confidence_medium_spin)
        quality_layout.addRow("Inference batch size", self.inference_batch_spin)
        left_layout.addWidget(quality_group)

        self.slice_count_reminder_label = QLabel("Will process 0 slices")
        self.processing_estimate_label = QLabel("Estimated processing time: -")
        self.validate_configuration_button = QPushButton("Validate Configuration")
        self.validate_configuration_button.clicked.connect(self._show_configuration_validation)
        left_layout.addWidget(self.slice_count_reminder_label)
        left_layout.addWidget(self.processing_estimate_label)
        left_layout.addWidget(self.validate_configuration_button)

        self.config_validation_label = QLabel("")
        self.config_validation_label.setObjectName("WarningText")
        self.config_validation_label.setWordWrap(True)
        left_layout.addWidget(self.config_validation_label)
        left_layout.addStretch(1)

        right = QWidget()
        right_layout = QVBoxLayout(right)

        self.tech_toggle = QToolButton()
        self.tech_toggle.setText("v Technical Insights")
        self.tech_toggle.setCheckable(True)
        self.tech_toggle.setChecked(True)
        self.tech_toggle.toggled.connect(self._toggle_tech_insights)

        self.tech_insights = QPlainTextEdit()
        self.tech_insights.setReadOnly(True)
        self.tech_insights.setPlainText(
            "O/U/V vectors define each section plane in atlas space.\n\n"
            "Angle normalization is not a simple vector average. DeepSlice calculates\n"
            "DV and ML angle per section, computes a Gaussian-weighted mean around\n"
            "atlas center depth, then rotates each section plane toward those means.\n"
            "The process runs twice because adjusting one plane perturbs the other.\n\n"
            "Thickness suggestion is estimated from section-number spacing relative to\n"
            "predicted depth spacing, then weighted by center-proximal Gaussian scores."
        )

        right_layout.addWidget(self.tech_toggle)
        right_layout.addWidget(self.tech_insights, stretch=1)

        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 3)

        root = QVBoxLayout(page)
        root.addWidget(split)
        return page

    def _build_prediction_page(self) -> QWidget:
        page = QWidget()
        split = QSplitter(Qt.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)

        run_layout = QHBoxLayout()
        self.run_alignment_button = QPushButton("Run Alignment")
        self.run_alignment_button.clicked.connect(self._run_alignment)
        self.run_alignment_button.setMinimumHeight(44)
        
        self.cancel_alignment_button = QPushButton("Cancel Alignment")
        self.cancel_alignment_button.clicked.connect(self._cancel_alignment)
        self.cancel_alignment_button.setMinimumHeight(44)
        self.cancel_alignment_button.setEnabled(False)
        
        run_layout.addWidget(self.run_alignment_button, stretch=3)
        run_layout.addWidget(self.cancel_alignment_button, stretch=1)

        self.prediction_phase_label = QLabel("Phase 0/1: idle")
        self.prediction_progress_label = QLabel("Progress: 0 / 0")
        self.prediction_elapsed_label = QLabel("Elapsed: 00:00")
        self.prediction_eta_label = QLabel("Remaining: --:--")
        self.prediction_progress_bar = QProgressBar()
        self.prediction_progress_bar.setRange(0, 100)
        self.prediction_progress_bar.setValue(0)

        self.predicted_thickness_label = QLabel("Estimated thickness: -")
        self.accept_predicted_thickness_button = QPushButton("Use Predicted Thickness")
        self.accept_predicted_thickness_button.clicked.connect(
            self._accept_predicted_thickness
        )
        self.accept_predicted_thickness_button.setEnabled(False)

        self.prediction_direction_label = QLabel("Detected indexing direction: -")

        console_tools = QHBoxLayout()
        self.console_toggle = QToolButton()
        self.console_toggle.setCheckable(True)
        self.console_toggle.setText("Show Runtime Console")
        self.console_toggle.toggled.connect(self._toggle_console)

        self.console_autoscroll_toggle = QToolButton()
        self.console_autoscroll_toggle.setCheckable(True)
        self.console_autoscroll_toggle.setChecked(True)
        self.console_autoscroll_toggle.setText("Auto-scroll")
        self.console_autoscroll_toggle.setVisible(False)
        
        self.clear_console_button = QToolButton()
        self.clear_console_button.setText("Clear")
        self.clear_console_button.clicked.connect(self._clear_console)
        self.clear_console_button.setVisible(False)
        
        self.copy_console_button = QToolButton()
        self.copy_console_button.setText("Copy")
        self.copy_console_button.clicked.connect(self._copy_console)
        self.copy_console_button.setVisible(False)
        
        self.console_toggle.toggled.connect(lambda v: self.clear_console_button.setVisible(v))
        self.console_toggle.toggled.connect(lambda v: self.copy_console_button.setVisible(v))
        self.console_toggle.toggled.connect(lambda v: self.console_autoscroll_toggle.setVisible(v))
        
        console_tools.addWidget(self.console_toggle)
        console_tools.addStretch(1)
        console_tools.addWidget(self.console_autoscroll_toggle)
        console_tools.addWidget(self.clear_console_button)
        console_tools.addWidget(self.copy_console_button)

        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setVisible(False)

        left_layout.addLayout(run_layout)
        left_layout.addWidget(self.prediction_phase_label)
        left_layout.addWidget(self.prediction_progress_label)
        left_layout.addWidget(self.prediction_elapsed_label)
        left_layout.addWidget(self.prediction_eta_label)
        left_layout.addWidget(self.prediction_progress_bar)
        left_layout.addWidget(self.predicted_thickness_label)
        left_layout.addWidget(self.accept_predicted_thickness_button)
        left_layout.addWidget(self.prediction_direction_label)
        left_layout.addLayout(console_tools)
        left_layout.addWidget(self.console_output, stretch=1)

        right = QWidget()
        right_layout = QVBoxLayout(right)

        self.prediction_slice_selector = QComboBox()
        self.prediction_slice_selector.currentIndexChanged.connect(
            self._refresh_prediction_preview
        )
        right_layout.addWidget(self.prediction_slice_selector)

        self.prediction_compare_checkbox = QCheckBox("Show Atlas Comparison")
        self.prediction_compare_checkbox.setChecked(True)
        self.prediction_compare_checkbox.setEnabled(False)
        self.prediction_compare_checkbox.setText("Atlas Comparison (always on)")
        self.prediction_compare_checkbox.toggled.connect(self._refresh_prediction_preview)
        self.prediction_atlas_info_label = QLabel("Atlas comparison: waiting for prediction")
        right_layout.addWidget(self.prediction_compare_checkbox)
        right_layout.addWidget(self.prediction_atlas_info_label)

        self.prediction_viewer = SliceGraphicsView()
        self.prediction_atlas_viewer = SliceGraphicsView()
        self.prediction_atlas_viewer.clear_with_text("Atlas view will update during prediction")

        prediction_view_split = QSplitter(Qt.Horizontal)
        prediction_view_split.addWidget(self.prediction_viewer)
        prediction_view_split.addWidget(self.prediction_atlas_viewer)
        prediction_view_split.setStretchFactor(0, 1)
        prediction_view_split.setStretchFactor(1, 3)
        right_layout.addWidget(prediction_view_split, stretch=1)

        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 3)

        root = QVBoxLayout(page)
        root.addWidget(split)
        return page

    def _build_curation_page(self) -> QWidget:
        page = QWidget()
        split = QSplitter(Qt.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)

        list_header_layout = QHBoxLayout()
        self.curation_prev_button = QToolButton()
        self.curation_prev_button.setText("Prev")
        self.curation_prev_button.clicked.connect(lambda: self._step_curation_slice(-1))
        self.curation_next_button = QToolButton()
        self.curation_next_button.setText("Next")
        self.curation_next_button.clicked.connect(lambda: self._step_curation_slice(1))

        self.curation_select_all_btn = QToolButton()
        self.curation_select_all_btn.setText("Select All")
        self.curation_select_all_btn.clicked.connect(lambda: self._set_all_flags(Qt.Checked))
        
        self.curation_deselect_all_btn = QToolButton()
        self.curation_deselect_all_btn.setText("Deselect All")
        self.curation_deselect_all_btn.clicked.connect(lambda: self._set_all_flags(Qt.Unchecked))
        
        self.confidence_filter_combo = QComboBox()
        self.confidence_filter_combo.addItems(["All Confidences", "High Only", "Medium Only", "Low Only"])
        self.confidence_filter_combo.currentIndexChanged.connect(self._filter_curation_list)
        
        list_header_layout.addWidget(self.curation_prev_button)
        list_header_layout.addWidget(self.curation_next_button)
        list_header_layout.addWidget(self.curation_select_all_btn)
        list_header_layout.addWidget(self.curation_deselect_all_btn)
        list_header_layout.addStretch()
        list_header_layout.addWidget(QLabel("Filter:"))
        list_header_layout.addWidget(self.confidence_filter_combo)
        
        left_layout.addLayout(list_header_layout)

        self.slice_flag_list = FlagListWidget()
        self.slice_flag_list.currentRowChanged.connect(self._on_curation_slice_selected)
        self.slice_flag_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.slice_flag_list.setDefaultDropAction(Qt.MoveAction)
        self.slice_flag_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.slice_flag_list.customContextMenuRequested.connect(self._show_curation_context_menu)

        self.apply_bad_sections_button = QPushButton("Apply Bad Section Flags")
        self.apply_bad_sections_button.clicked.connect(self._apply_bad_section_flags)
        self.apply_manual_order_button = QPushButton("Apply Manual Reordering")
        self.apply_manual_order_button.clicked.connect(self._apply_manual_order)
        self.move_slice_up_button = QToolButton()
        self.move_slice_up_button.setText("Move Up")
        self.move_slice_up_button.clicked.connect(lambda: self._move_selected_slice(-1))
        self.move_slice_down_button = QToolButton()
        self.move_slice_down_button.setText("Move Down")
        self.move_slice_down_button.clicked.connect(lambda: self._move_selected_slice(1))
        self.detect_outliers_button = QPushButton("Detect Outliers")
        self.detect_outliers_button.clicked.connect(self._detect_outliers)
        self.reset_flags_button = QPushButton("Reset All Flags")
        self.reset_flags_button.clicked.connect(lambda: self._set_all_flags(Qt.Unchecked))
        self.slice_note_edit = QLineEdit()
        self.slice_note_edit.setPlaceholderText("Optional per-slice atlas note")
        self.save_slice_note_button = QPushButton("Save Slice Note")
        self.save_slice_note_button.clicked.connect(self._save_slice_note)

        controls_group = QGroupBox("Propagation / Curation")
        controls_layout = QGridLayout(controls_group)

        self.normalize_angles_button = QPushButton("Normalize Angles")
        self.normalize_angles_button.setToolTip(
            "Applies Gaussian-weighted DV/ML mean-angle propagation (two-pass adjustment)."
        )
        self.normalize_angles_button.clicked.connect(self._normalize_angles)

        self.enforce_order_button = QPushButton("Enforce Index Order")
        self.enforce_order_button.setToolTip(
            "Reorders Oy values to match index ordering while preserving measured spacing."
        )
        self.enforce_order_button.clicked.connect(self._enforce_index_order)

        self.enforce_spacing_button = QPushButton("Enforce Index Spacing")
        self.enforce_spacing_button.setToolTip(
            "Recalculates Oy values to be evenly spaced based on section thickness."
        )
        self.enforce_spacing_button.clicked.connect(self._enforce_index_spacing)

        self.ml_spin = QDoubleSpinBox()
        self.ml_spin.setRange(-90.0, 90.0)
        self.ml_spin.setDecimals(2)
        self.ml_spin.setSuffix(" deg")

        self.dv_spin = QDoubleSpinBox()
        self.dv_spin.setRange(-90.0, 90.0)
        self.dv_spin.setDecimals(2)
        self.dv_spin.setSuffix(" deg")

        self.apply_manual_angles_button = QPushButton("Apply Manual Angles")
        self.apply_manual_angles_button.setToolTip(
            "Directly overrides dataset ML and DV angles in degrees."
        )
        self.apply_manual_angles_button.clicked.connect(self._apply_manual_angles)

        self.undo_button = QPushButton("Undo (0)")
        self.undo_button.clicked.connect(self._undo)
        self.redo_button = QPushButton("Redo (0)")
        self.redo_button.clicked.connect(self._redo)

        controls_layout.addWidget(self.normalize_angles_button, 0, 0, 1, 2)
        controls_layout.addWidget(self.enforce_order_button, 1, 0, 1, 2)
        controls_layout.addWidget(self.enforce_spacing_button, 2, 0, 1, 2)
        controls_layout.addWidget(QLabel("ML Angle"), 3, 0)
        controls_layout.addWidget(self.ml_spin, 3, 1)
        controls_layout.addWidget(QLabel("DV Angle"), 4, 0)
        controls_layout.addWidget(self.dv_spin, 4, 1)
        controls_layout.addWidget(self.apply_manual_angles_button, 5, 0, 1, 2)
        controls_layout.addWidget(self.undo_button, 6, 0)
        controls_layout.addWidget(self.redo_button, 6, 1)

        anchor_group = QGroupBox("Anchor Alignment")
        anchor_layout = QVBoxLayout(anchor_group)
        self.anchor_summary_label = QLabel(
            "Set 3 or more known slices as anchors, then interpolate all other AP depths between anchors."
        )
        self.anchor_summary_label.setWordWrap(True)

        self.anchor_depth_spin = QDoubleSpinBox()
        self.anchor_depth_spin.setRange(-5000.0, 5000.0)
        self.anchor_depth_spin.setDecimals(2)
        self.anchor_depth_spin.setSingleStep(1.0)
        self.anchor_depth_spin.setSuffix(" AP")

        self.set_anchor_button = QToolButton()
        self.set_anchor_button.setText("Set/Update Anchor")
        self.set_anchor_button.clicked.connect(self._set_anchor_for_current_slice)

        self.remove_anchor_button = QToolButton()
        self.remove_anchor_button.setText("Remove Anchor")
        self.remove_anchor_button.clicked.connect(self._remove_anchor_for_current_slice)

        self.apply_anchor_interpolation_button = QPushButton("Distribute Between Anchors")
        self.apply_anchor_interpolation_button.clicked.connect(self._apply_anchor_interpolation)

        self.clear_anchor_button = QToolButton()
        self.clear_anchor_button.setText("Clear")
        self.clear_anchor_button.clicked.connect(self._clear_anchor_points)

        anchor_action_row = QHBoxLayout()
        anchor_action_row.addWidget(self.set_anchor_button)
        anchor_action_row.addWidget(self.remove_anchor_button)
        anchor_action_row.addWidget(self.clear_anchor_button)

        self.anchor_list = QListWidget()
        self.anchor_list.setMaximumHeight(110)

        anchor_layout.addWidget(self.anchor_summary_label)
        anchor_layout.addWidget(QLabel("Target AP depth for selected slice"))
        anchor_layout.addWidget(self.anchor_depth_spin)
        anchor_layout.addLayout(anchor_action_row)
        anchor_layout.addWidget(self.anchor_list)
        anchor_layout.addWidget(self.apply_anchor_interpolation_button)

        left_layout.addWidget(self.slice_flag_list, stretch=2)
        reorder_row = QHBoxLayout()
        reorder_row.addWidget(self.move_slice_up_button)
        reorder_row.addWidget(self.move_slice_down_button)
        left_layout.addLayout(reorder_row)
        left_layout.addWidget(self.apply_manual_order_button)
        left_layout.addWidget(self.slice_note_edit)
        left_layout.addWidget(self.save_slice_note_button)
        left_layout.addWidget(self.apply_bad_sections_button)
        left_layout.addWidget(self.detect_outliers_button)
        left_layout.addWidget(self.reset_flags_button)
        left_layout.addWidget(controls_group, stretch=1)
        left_layout.addWidget(anchor_group)

        right = QWidget()
        right_layout = QVBoxLayout(right)

        self.linearity_figure = Figure(figsize=(5.2, 2.2), facecolor="#1A1F26")
        self.linearity_canvas = FigureCanvas(self.linearity_figure)
        self.linearity_axis = self.linearity_figure.add_subplot(111)
        self.linearity_canvas.mpl_connect("button_press_event", self._on_linearity_click)
        self.linearity_canvas.setMinimumHeight(130)

        plot_toolbar = QHBoxLayout()
        self.zoom_fit_button = QToolButton()
        self.zoom_fit_button.setText("Fit")
        self.zoom_fit_button.clicked.connect(self._linearity_zoom_fit)
        self.zoom_in_button = QToolButton()
        self.zoom_in_button.setText("+")
        self.zoom_in_button.clicked.connect(lambda: self._linearity_zoom(0.8))
        self.zoom_out_button = QToolButton()
        self.zoom_out_button.setText("-")
        self.zoom_out_button.clicked.connect(lambda: self._linearity_zoom(1.25))
        plot_toolbar.addWidget(QLabel("Plot"))
        plot_toolbar.addWidget(self.zoom_fit_button)
        plot_toolbar.addWidget(self.zoom_in_button)
        plot_toolbar.addWidget(self.zoom_out_button)
        plot_toolbar.addStretch(1)

        self.hist_figure = Figure(figsize=(5.2, 1.0), facecolor="#1A1F26")
        self.hist_canvas = FigureCanvas(self.hist_figure)
        self.hist_axis = self.hist_figure.add_subplot(111)
        self.hist_canvas.setMinimumHeight(80)

        atlas_controls = QHBoxLayout()
        self.enable_atlas_preview_checkbox = QCheckBox("Atlas volume preview")
        self.enable_atlas_preview_checkbox.toggled.connect(self._on_atlas_preview_toggled)
        self.atlas_volume_combo = QComboBox()
        self.atlas_volume_combo.currentTextChanged.connect(self._on_atlas_volume_changed)
        self.enable_blend_overlay_checkbox = QCheckBox("Blend atlas on histology")
        self.enable_blend_overlay_checkbox.toggled.connect(self._on_blend_overlay_toggled)
        self.blend_slider = QSlider(Qt.Horizontal)
        self.blend_slider.setRange(5, 95)
        self.blend_slider.setValue(35)
        self.blend_slider.valueChanged.connect(self._on_blend_slider_changed)
        self.blend_percent_label = QLabel("Blend: 35%")
        self.atlas_slice_info_label = QLabel("Atlas: disabled")
        self.before_after_toggle = QCheckBox("Before/After Overlay")
        self.before_after_toggle.toggled.connect(self._refresh_curation_views)
        self.loupe_toggle = QCheckBox("Loupe")
        self.loupe_toggle.toggled.connect(self._on_loupe_toggled)
        self.atlas_coords_label = QLabel("Atlas coords: -")
        atlas_controls.addWidget(self.enable_atlas_preview_checkbox)
        atlas_controls.addWidget(QLabel("Volume"))
        atlas_controls.addWidget(self.atlas_volume_combo)
        atlas_controls.addWidget(self.enable_blend_overlay_checkbox)
        atlas_controls.addWidget(self.blend_slider)
        atlas_controls.addWidget(self.blend_percent_label)
        atlas_controls.addWidget(self.before_after_toggle)
        atlas_controls.addWidget(self.loupe_toggle)
        atlas_controls.addWidget(self.atlas_coords_label)
        atlas_controls.addWidget(self.atlas_slice_info_label, stretch=1)

        atlas_transform_row = QHBoxLayout()
        self.atlas_flip_x_checkbox = QCheckBox("Flip X")
        self.atlas_flip_x_checkbox.toggled.connect(self._on_atlas_transform_changed)
        self.atlas_flip_y_checkbox = QCheckBox("Flip Y")
        self.atlas_flip_y_checkbox.toggled.connect(self._on_atlas_transform_changed)

        self.atlas_rotate_combo = QComboBox()
        self.atlas_rotate_combo.addItem("Rotate: 0 deg", 0)
        self.atlas_rotate_combo.addItem("Rotate: Auto", "auto")
        self.atlas_rotate_combo.addItem("Rotate: 90 deg CW", -1)
        self.atlas_rotate_combo.addItem("Rotate: 180 deg", 2)
        self.atlas_rotate_combo.addItem("Rotate: 270 deg CW", 1)
        self.atlas_rotate_combo.setCurrentIndex(0)
        self.atlas_rotate_combo.currentIndexChanged.connect(self._on_atlas_transform_changed)

        self.atlas_scale_slider = QSlider(Qt.Horizontal)
        self.atlas_scale_slider.setRange(60, 180)
        self.atlas_scale_slider.setValue(100)
        self.atlas_scale_slider.valueChanged.connect(self._on_atlas_transform_changed)
        self.atlas_scale_label = QLabel("Scale: 100%")

        atlas_transform_row.addWidget(self.atlas_flip_x_checkbox)
        atlas_transform_row.addWidget(self.atlas_flip_y_checkbox)
        atlas_transform_row.addWidget(self.atlas_rotate_combo)
        atlas_transform_row.addWidget(self.atlas_scale_label)
        atlas_transform_row.addWidget(self.atlas_scale_slider, stretch=1)

        atlas_offset_row = QHBoxLayout()
        self.atlas_offset_x_slider = QSlider(Qt.Horizontal)
        self.atlas_offset_x_slider.setRange(-50, 50)
        self.atlas_offset_x_slider.setValue(0)
        self.atlas_offset_x_slider.valueChanged.connect(self._on_atlas_transform_changed)
        self.atlas_offset_x_label = QLabel("Offset X: 0%")

        self.atlas_offset_y_slider = QSlider(Qt.Horizontal)
        self.atlas_offset_y_slider.setRange(-50, 50)
        self.atlas_offset_y_slider.setValue(0)
        self.atlas_offset_y_slider.valueChanged.connect(self._on_atlas_transform_changed)
        self.atlas_offset_y_label = QLabel("Offset Y: 0%")

        atlas_offset_row.addWidget(self.atlas_offset_x_label)
        atlas_offset_row.addWidget(self.atlas_offset_x_slider, stretch=1)
        atlas_offset_row.addWidget(self.atlas_offset_y_label)
        atlas_offset_row.addWidget(self.atlas_offset_y_slider, stretch=1)

        self.confidence_panel_toggle = QToolButton()
        self.confidence_panel_toggle.setText("Confidence Overlay Preview")
        self.confidence_panel_toggle.setCheckable(True)
        self.confidence_panel_toggle.setChecked(False)
        self.confidence_panel_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.confidence_panel_toggle.setArrowType(Qt.RightArrow)
        self.confidence_panel_toggle.toggled.connect(self._toggle_confidence_panel)

        self.confidence_panel = QFrame()
        confidence_panel_layout = QVBoxLayout(self.confidence_panel)
        confidence_panel_layout.setContentsMargins(0, 0, 0, 0)
        confidence_panel_layout.setSpacing(4)
        self.confidence_overlay_viewer = SliceGraphicsView()
        self.confidence_overlay_viewer.setMinimumHeight(180)
        self.confidence_overlay_viewer.clear_with_text(
            "Select a slice to inspect confidence overlay"
        )
        self.confidence_overlay_legend = QLabel(
            "Red = histology intensity, Green = confidence/atlas agreement"
        )
        self.confidence_overlay_legend.setStyleSheet("color: #AFC3DD;")
        confidence_panel_layout.addWidget(self.confidence_overlay_viewer)
        confidence_panel_layout.addWidget(self.confidence_overlay_legend)
        self.confidence_panel.setVisible(False)

        self.curation_viewer = SliceGraphicsView()
        self.atlas_viewer = SliceGraphicsView()
        self.atlas_viewer.clear_with_text("Enable atlas preview to load atlas slices")
        self.curation_viewer.set_sync_partner(self.atlas_viewer)
        self.atlas_viewer.set_sync_partner(self.curation_viewer)
        self.atlas_viewer.mouseSceneMoved.connect(self._on_atlas_mouse_moved)

        viewer_split = QSplitter(Qt.Horizontal)
        viewer_split.addWidget(self.curation_viewer)
        viewer_split.addWidget(self.atlas_viewer)
        viewer_split.setStretchFactor(0, 1)
        viewer_split.setStretchFactor(1, 3)

        plot_panel = QWidget()
        plot_panel_layout = QVBoxLayout(plot_panel)
        plot_panel_layout.setContentsMargins(0, 0, 0, 0)
        plot_panel_layout.setSpacing(6)
        plot_panel_layout.addLayout(plot_toolbar)
        plot_panel_layout.addWidget(self.linearity_canvas)
        plot_panel_layout.addWidget(self.hist_canvas)

        self.curation_vertical_split = QSplitter(Qt.Vertical)
        self.curation_vertical_split.addWidget(viewer_split)
        self.curation_vertical_split.addWidget(plot_panel)
        self.curation_vertical_split.setStretchFactor(0, 4)
        self.curation_vertical_split.setStretchFactor(1, 3)
        self.curation_vertical_split.setSizes([680, 230])

        right_layout.addLayout(atlas_controls)
        right_layout.addLayout(atlas_transform_row)
        right_layout.addLayout(atlas_offset_row)
        right_layout.addWidget(self.confidence_panel_toggle)
        right_layout.addWidget(self.confidence_panel)
        right_layout.addWidget(self.curation_vertical_split, stretch=1)

        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 3)

        root = QVBoxLayout(page)
        root.addWidget(split)
        self._refresh_atlas_volume_options()
        self._update_anchor_depth_range()
        return page

    def _build_export_page(self) -> QWidget:
        page = QWidget()
        split = QSplitter(Qt.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)

        output_group = QGroupBox("Export Configuration")
        output_layout = QFormLayout(output_group)

        output_dir_row = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setText(self._get_persisted_export_path())
        self.output_dir_edit.textChanged.connect(self._persist_export_path)
        self.browse_output_dir_button = QPushButton("Browse")
        self.browse_output_dir_button.clicked.connect(self._browse_output_directory)
        output_dir_row.addWidget(self.output_dir_edit)
        output_dir_row.addWidget(self.browse_output_dir_button)

        self.output_basename_edit = QLineEdit("DeepSliceResults")
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItems(
            [
                "JSON (QuickNII/VisuAlign)",
                "Legacy XML",
            ]
        )
        self.output_format_help_button = QToolButton()
        self.output_format_help_button.setText("?")
        self.output_format_help_button.clicked.connect(self._show_export_format_help)
        output_format_row = QHBoxLayout()
        output_format_row.addWidget(self.output_format_combo)
        output_format_row.addWidget(self.output_format_help_button)
        
        self.export_size_estimate_label = QLabel("~0 MB")
        self.output_format_combo.currentIndexChanged.connect(self._update_export_size_estimate)

        output_layout.addRow("Output Directory", output_dir_row)
        output_layout.addRow("Base Filename", self.output_basename_edit)
        output_layout.addRow("Primary Export", output_format_row)
        output_layout.addRow("Estimated Size", self.export_size_estimate_label)

        export_actions_layout = QHBoxLayout()
        self.export_button = QPushButton("Export Predictions")
        self.export_button.clicked.connect(self._export_predictions)
        
        self.open_export_dir_button = QToolButton()
        self.open_export_dir_button.setText("Open Folder")
        self.open_export_dir_button.clicked.connect(self._open_export_directory)
        
        self.copy_export_path_button = QToolButton()
        self.copy_export_path_button.setText("Copy Path")
        self.copy_export_path_button.clicked.connect(self._copy_export_path)
        
        export_actions_layout.addWidget(self.export_button, stretch=1)
        export_actions_layout.addWidget(self.open_export_dir_button)
        export_actions_layout.addWidget(self.copy_export_path_button)

        self.report_button = QPushButton("Generate Report (PDF)")
        self.report_button.clicked.connect(self._generate_report)
        
        self.preview_report_button = QToolButton()
        self.preview_report_button.setText("Preview Report")
        self.preview_report_button.clicked.connect(self._preview_report)
        
        report_layout = QHBoxLayout()
        report_layout.addWidget(self.report_button, stretch=3)
        report_layout.addWidget(self.preview_report_button, stretch=1)

        self.pdf_content_group = QGroupBox("PDF Contents")
        pdf_content_layout = QHBoxLayout(self.pdf_content_group)
        self.pdf_include_stats = QCheckBox("Summary Stats")
        self.pdf_include_stats.setChecked(True)
        self.pdf_include_plot = QCheckBox("Linearity Plot")
        self.pdf_include_plot.setChecked(True)
        self.pdf_include_images = QCheckBox("Sample Images")
        self.pdf_include_images.setChecked(True)
        self.pdf_include_angles = QCheckBox("Angle Metrics")
        self.pdf_include_angles.setChecked(True)
        pdf_content_layout.addWidget(self.pdf_include_stats)
        pdf_content_layout.addWidget(self.pdf_include_plot)
        pdf_content_layout.addWidget(self.pdf_include_images)
        pdf_content_layout.addWidget(self.pdf_include_angles)

        quicknii_row = QHBoxLayout()
        self.quicknii_path_edit = QLineEdit()
        self.quicknii_path_edit.setPlaceholderText("Optional path to QuickNII executable")
        self.quicknii_path_edit.setText(self._get_persisted_quicknii_path())
        self.quicknii_path_edit.textChanged.connect(self._persist_quicknii_path)
        self.quicknii_browse_button = QPushButton("Browse")
        self.quicknii_browse_button.clicked.connect(self._browse_quicknii_path)
        quicknii_row.addWidget(self.quicknii_path_edit)
        quicknii_row.addWidget(self.quicknii_browse_button)

        self.open_quicknii_button = QPushButton("Open in QuickNII")
        self.open_quicknii_button.clicked.connect(self._open_in_quicknii)

        self.summary_label = QLabel("Processed: 0 | Excluded: 0")
        self.deviation_label = QLabel("Mean angular deviation: 0.00 deg")
        self.markers_label = QLabel("")
        self.markers_label.setWordWrap(True)
        self.markers_label.setObjectName("WarningText")

        left_layout.addWidget(output_group)
        left_layout.addLayout(export_actions_layout)
        left_layout.addWidget(self.pdf_content_group)
        left_layout.addLayout(report_layout)
        left_layout.addLayout(quicknii_row)
        left_layout.addWidget(self.open_quicknii_button)
        left_layout.addWidget(self.summary_label)
        left_layout.addWidget(self.deviation_label)
        left_layout.addWidget(self.markers_label)
        left_layout.addStretch(1)

        right = QWidget()
        right_layout = QVBoxLayout(right)

        self.export_notes = QPlainTextEdit()
        self.export_notes.setReadOnly(True)
        self.export_notes.setPlainText(
            "Export details:\n\n"
            "1) CSV is always exported alongside JSON/XML.\n"
            "2) JSON is QuickNII/VisuAlign-compatible and preserves markers.\n"
            "3) Legacy XML is provided for older workflows.\n"
            "4) Load Session can re-open previous QuickNII JSON/XML for re-curation."
        )

        right_layout.addWidget(self.export_notes)

        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 3)

        root = QVBoxLayout(page)
        root.addWidget(split)
        return page

    def _apply_theme(self):
        dark_theme = """
            QWidget {
                background: #0B0F14;
                color: #DEE6EF;
                font-family: 'Segoe UI', 'Noto Sans', sans-serif;
                font-size: 10pt;
            }
            #TopBar {
                background: #1A1F26;
                border-radius: 10px;
            }
            #ProjectLabel {
                font-size: 13pt;
                font-weight: 600;
                color: #F5FAFF;
            }
            #SessionLabel, #HardwareLabel {
                color: #9CB0C7;
            }
            #WarningText {
                color: #F2B544;
            }
            #HintText {
                color: #8FB7FF;
                font-size: 9pt;
            }
            QGroupBox {
                border: 1px solid #27303A;
                border-radius: 10px;
                margin-top: 10px;
                padding: 8px;
                background: #1A1F26;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px 0 4px;
                color: #8FB7FF;
            }
            QPushButton {
                background: #1E6FFF;
                border: 1px solid #2A79FF;
                border-radius: 10px;
                padding: 7px 12px;
                color: #F7FBFF;
                font-weight: 500;
            }
            QPushButton:hover {
                background: #2A79FF;
            }
            QPushButton:disabled {
                background: #2A2F39;
                border-color: #2A2F39;
                color: #7A8492;
            }
            QLineEdit, QPlainTextEdit, QTextEdit, QListWidget, QTableWidget, QComboBox, QDoubleSpinBox {
                background: #11161D;
                border: 1px solid #2A313B;
                border-radius: 9px;
                padding: 5px;
            }
            QHeaderView::section {
                background: #1A1F26;
                border: 1px solid #2A313B;
                padding: 4px;
                color: #9CB0C7;
            }
            #DropArea {
                border: 1px dashed #416EAE;
                border-radius: 12px;
                background: #131A23;
            }
            #DropIcon {
                color: #8FB7FF;
                font-weight: 700;
                font-size: 16pt;
            }
            #DropTitle {
                color: #CFE0FF;
                font-weight: 600;
            }
            #DropSubtitle {
                color: #9CB0C7;
                font-size: 9pt;
            }
            #SummaryBanner {
                border-radius: 8px;
                padding: 6px;
            }
            #StepNavigator {
                background: #1A1F26;
                border: 1px solid #27303A;
                border-radius: 12px;
                padding: 6px;
            }
            #StepNavigator::item {
                border-radius: 8px;
                padding: 8px;
                margin: 3px;
            }
            #StepNavigator::item:selected {
                background: #164DB4;
                color: #F7FBFF;
            }
            QToolButton {
                background: #1A1F26;
                border: 1px solid #2A313B;
                border-radius: 9px;
                padding: 6px 10px;
            }
            QProgressBar {
                border: 1px solid #2A313B;
                border-radius: 6px;
                background: #11161D;
            }
            QProgressBar::chunk {
                background: #2A79FF;
                border-radius: 6px;
            }
            QPushButton:focus, QToolButton:focus,
            QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus,
            QListWidget:focus, QTableWidget:focus, QComboBox:focus,
            QDoubleSpinBox:focus, QSlider:focus {
                border: 2px solid #69C0FF;
                outline: none;
            }
            QCheckBox:focus, QRadioButton:focus {
                outline: 2px solid #69C0FF;
            }
        """

        light_theme = """
            QWidget {
                background: #F2F6FB;
                color: #1A2733;
                font-family: 'Segoe UI', 'Noto Sans', sans-serif;
                font-size: 10pt;
            }
            #TopBar {
                background: #E7EEF7;
                border-radius: 10px;
            }
            #ProjectLabel {
                font-size: 13pt;
                font-weight: 600;
                color: #1B2B3D;
            }
            #SessionLabel, #HardwareLabel {
                color: #34516E;
            }
            #WarningText {
                color: #9D5F00;
            }
            #HintText {
                color: #295F95;
                font-size: 9pt;
            }
            QGroupBox {
                border: 1px solid #C8D8EA;
                border-radius: 10px;
                margin-top: 10px;
                padding: 8px;
                background: #FFFFFF;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px 0 4px;
                color: #295F95;
            }
            QPushButton {
                background: #2F7BCE;
                border: 1px solid #2668AF;
                border-radius: 10px;
                padding: 7px 12px;
                color: #FFFFFF;
                font-weight: 500;
            }
            QPushButton:hover {
                background: #3C89DC;
            }
            QPushButton:disabled {
                background: #D9E3EE;
                border-color: #D9E3EE;
                color: #7B8898;
            }
            QLineEdit, QPlainTextEdit, QTextEdit, QListWidget, QTableWidget, QComboBox, QDoubleSpinBox {
                background: #FFFFFF;
                border: 1px solid #C4D2E1;
                border-radius: 9px;
                padding: 5px;
            }
            QHeaderView::section {
                background: #E7EEF7;
                border: 1px solid #C4D2E1;
                padding: 4px;
                color: #34516E;
            }
            #DropArea {
                border: 1px dashed #5F8FBF;
                border-radius: 12px;
                background: #EEF4FB;
            }
            #DropIcon {
                color: #2F7BCE;
                font-weight: 700;
                font-size: 16pt;
            }
            #DropTitle {
                color: #1E3F62;
                font-weight: 600;
            }
            #DropSubtitle {
                color: #476784;
                font-size: 9pt;
            }
            #SummaryBanner {
                border-radius: 8px;
                padding: 6px;
            }
            #StepNavigator {
                background: #FFFFFF;
                border: 1px solid #C8D8EA;
                border-radius: 12px;
                padding: 6px;
            }
            #StepNavigator::item {
                border-radius: 8px;
                padding: 8px;
                margin: 3px;
            }
            #StepNavigator::item:selected {
                background: #2F7BCE;
                color: #FFFFFF;
            }
            QToolButton {
                background: #FFFFFF;
                border: 1px solid #C4D2E1;
                border-radius: 9px;
                padding: 6px 10px;
            }
            QProgressBar {
                border: 1px solid #C4D2E1;
                border-radius: 6px;
                background: #FFFFFF;
            }
            QProgressBar::chunk {
                background: #2F7BCE;
                border-radius: 6px;
            }
            QPushButton:focus, QToolButton:focus,
            QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus,
            QListWidget:focus, QTableWidget:focus, QComboBox:focus,
            QDoubleSpinBox:focus, QSlider:focus {
                border: 2px solid #1F6CB8;
                outline: none;
            }
            QCheckBox:focus, QRadioButton:focus {
                outline: 2px solid #1F6CB8;
            }
        """

        stylesheet = dark_theme if self._theme_name == "dark" else light_theme
        self.setStyleSheet(stylesheet)

        if hasattr(self, "theme_toggle_button"):
            self.theme_toggle_button.setText(
                "Theme: Dark" if self._theme_name == "dark" else "Theme: Light"
            )

    def _on_step_changed(self, index: int):
        if index < 0:
            return
        if index > self._max_unlocked_step():
            self.step_list.setCurrentRow(self._max_unlocked_step())
            return
        self.stack.setCurrentIndex(index)
        self._animate_page_transition()
        self._refresh_step_states()

    def _animate_page_transition(self):
        current_page = self.stack.currentWidget()
        if current_page is None:
            return
        effect = QGraphicsOpacityEffect(current_page)
        current_page.setGraphicsEffect(effect)
        animation = QPropertyAnimation(effect, b"opacity", self)
        animation.setDuration(90)
        animation.setStartValue(0.2)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.OutCubic)

        def clear_effect():
            current_page.setGraphicsEffect(None)

        animation.finished.connect(clear_effect)
        animation.start()
        self._active_transition_animation = animation

    def _max_unlocked_step(self) -> int:
        if len(self.state.image_paths) == 0:
            return 0
        if self.state.predictions is None:
            return 2
        return 4

    def _refresh_step_states(self):
        max_unlocked = self._max_unlocked_step()
        completed_indexes = set(range(max_unlocked))
        newly_completed = completed_indexes - self._last_completed_steps
        self._last_completed_steps = completed_indexes

        for idx in range(self.step_list.count()):
            item = self.step_list.item(idx)
            label = self.STEP_LABELS[idx]
            if idx == 3 and self._curation_modified:
                label = label + " *"
            if idx <= max_unlocked:
                item.setFlags(item.flags() | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            else:
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)

            prefix = ""
            if idx < max_unlocked:
                prefix = "✓ "
            elif idx == self.stack.currentIndex():
                prefix = "● "
            elif idx > max_unlocked:
                prefix = "◌ "
            item.setText(prefix + label)

            if idx in newly_completed:
                item.setBackground(QColor("#1E6FFF"))
                QTimer.singleShot(
                    320,
                    lambda i=idx: (
                        self.step_list.item(i).setBackground(QColor("transparent"))
                        if self.step_list.item(i) is not None
                        else None
                    ),
                )

        completed_count = max(0, max_unlocked)
        percent = int((completed_count / len(self.STEP_LABELS)) * 100)
        if hasattr(self, "completion_label"):
            self.completion_label.setText(
                f"{completed_count} of {len(self.STEP_LABELS)} steps complete ({percent}%)"
            )

    def _collect_supported_files_from_paths(self, dropped_paths: List[str]) -> List[str]:
        image_paths = []
        for path in dropped_paths:
            if os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for filename in files:
                        full_path = os.path.join(root, filename)
                        extension = os.path.splitext(full_path)[1].lower()
                        if extension in SUPPORTED_IMAGE_FORMATS:
                            image_paths.append(full_path)
            elif os.path.isfile(path):
                extension = os.path.splitext(path)[1].lower()
                if extension in SUPPORTED_IMAGE_FORMATS:
                    image_paths.append(path)
        return image_paths

    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder:
            return
        image_paths = self._collect_supported_files_from_paths([folder])
        self.state.add_images(image_paths)
        self._refresh_all_views()

    def _add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Histology Images",
            "",
            "Images (*.jpg *.jpeg *.png *.tif *.tiff)",
        )
        if not files:
            return
        self.state.add_images(files)
        self._refresh_all_views()

    def _handle_dropped_paths(self, paths: List[str]):
        image_paths = self._collect_supported_files_from_paths(paths)
        self.state.add_images(image_paths)
        self._refresh_all_views()

    def _clear_images(self):
        self.state.clear_images()
        self._refresh_all_views()

    def _on_thumbnail_selection_changed(self):
        selected = self.thumbnail_list.selectedItems()
        if len(selected) == 0:
            self.ingestion_preview.clear_with_text("Select a thumbnail to preview")
            return
        image_path = selected[0].data(Qt.UserRole)
        self.ingestion_preview.set_image(
            image_path,
            pixel_spacing_um=self._get_pixel_spacing_um(image_path),
        )

    def _refresh_ingestion_views(self):
        total_files = len(self.state.image_paths)
        self.slice_count_label.setText(f"Slices: {total_files}")

        total_size = 0
        for image_path in self.state.image_paths:
            try:
                total_size += int(os.path.getsize(image_path))
            except OSError:
                continue
        self.file_size_summary_label.setText(
            f"{total_files} files loaded - {self._format_bytes(total_size)}"
        )

        report = self.state.image_format_report()
        warnings = []
        if len(report["unsupported"]) > 0:
            warnings.append(
                f"Unsupported files excluded: {len(report['unsupported'])}. Accepted: {', '.join(sorted(SUPPORTED_IMAGE_FORMATS))}"
            )
        if len(report["supported"]) < 2 and len(report["supported"]) > 0:
            warnings.append("At least 2 sections are required for spacing and curation tools.")
        if 0 < len(report["supported"]) < 10:
            warnings.append("Fewer than 10 slices may reduce angle-propagation reliability.")
        if self.orientation_combo.currentIndex() != 0:
            warnings.append("Non-coronal orientation is currently not supported in this release.")

        index_report = self.state.build_index_report(
            legacy_section_numbers=self.legacy_parsing_checkbox.isChecked()
        )
        duplicate_count = len(index_report["duplicate_indices"])
        missing_count = len(index_report["missing_indices"])

        if self.enable_section_numbers_checkbox.isChecked():
            if index_report["parse_error"]:
                warnings.append(index_report["parse_error"])
            if duplicate_count > 0:
                warnings.append(
                    "Duplicate indices detected: "
                    + ", ".join([str(x) for x in index_report["duplicate_indices"]])
                )
            if missing_count > 0:
                warnings.append(
                    "Missing indices: "
                    + ", ".join([str(x) for x in index_report["missing_indices"][:20]])
                )

        self.ingestion_warning_label.setText("\n".join(warnings))

        valid_index_count = 0
        self.index_table.setRowCount(len(index_report["rows"]))
        warning_icon = self.style().standardIcon(QStyle.SP_MessageBoxWarning)
        for row_idx, row in enumerate(index_report["rows"]):
            status = str(row["status"])
            if status == "OK":
                valid_index_count += 1

            filename_item = QTableWidgetItem(row["filename"])
            index_item = QTableWidgetItem(str(row["detected_index"]))
            status_item = QTableWidgetItem(status)

            if status == "Duplicate":
                red = QColor("#8B2A37")
                filename_item.setBackground(red)
                index_item.setBackground(red)
                status_item.setBackground(red)
                status_item.setIcon(warning_icon)
            elif status != "OK":
                amber = QColor("#A26B1D")
                filename_item.setBackground(amber)
                index_item.setBackground(amber)
                status_item.setBackground(amber)

            self.index_table.setItem(row_idx, 0, filename_item)
            self.index_table.setItem(row_idx, 1, index_item)
            self.index_table.setItem(row_idx, 2, status_item)

        if total_files == 0:
            self.ingestion_summary_banner.setText("No files loaded")
            self.ingestion_summary_banner.setStyleSheet("QLabel { background: #1A1F26; border: 1px solid #2A313B; border-radius: 8px; padding: 6px; }")
        elif duplicate_count > 0:
            self.ingestion_summary_banner.setText(
                f"{total_files} files - {valid_index_count} valid indices - {duplicate_count} duplicates detected"
            )
            self.ingestion_summary_banner.setStyleSheet("QLabel { background: #5A1F2A; border: 1px solid #A43344; border-radius: 8px; padding: 6px; color: #F9DDE3; }")
        else:
            self.ingestion_summary_banner.setText(
                f"{total_files} files - {valid_index_count} valid indices - {len(warnings)} warnings"
            )
            self.ingestion_summary_banner.setStyleSheet("QLabel { background: #234033; border: 1px solid #2F6E52; border-radius: 8px; padding: 6px; color: #DDF7EA; }")

        entries = []
        for idx, image_path in enumerate(self.state.image_paths):
            row = index_report["rows"][idx] if idx < len(index_report["rows"]) else {}
            try:
                mtime = float(os.path.getmtime(image_path))
            except OSError:
                mtime = 0.0
            detected = row.get("detected_index", "")
            entries.append(
                {
                    "path": image_path,
                    "detected_index": detected,
                    "mtime": mtime,
                }
            )

        sort_mode = self.thumbnail_sort_combo.currentIndex() if hasattr(self, "thumbnail_sort_combo") else 0
        if sort_mode == 1:
            entries.sort(
                key=lambda entry: (
                    0 if str(entry["detected_index"]).isdigit() else 1,
                    int(entry["detected_index"]) if str(entry["detected_index"]).isdigit() else 10**9,
                    os.path.basename(entry["path"]).lower(),
                )
            )
        elif sort_mode == 2:
            entries.sort(key=lambda entry: entry["mtime"], reverse=True)
        else:
            entries.sort(key=lambda entry: os.path.basename(entry["path"]).lower())

        self.thumbnail_list.clear()
        if len(entries) > 40:
            skeleton = QPixmap(256, 256)
            skeleton.fill(QColor("#2A313B"))
            skeleton_icon = QIcon(skeleton)
            for _ in range(min(18, len(entries))):
                self.thumbnail_list.addItem(QListWidgetItem(skeleton_icon, "Loading..."))
            QApplication.processEvents()
            self.thumbnail_list.clear()

        show_progress = len(entries) >= 100
        self.thumbnail_progress.setVisible(show_progress)
        if show_progress:
            self.thumbnail_progress.setRange(0, len(entries))
            self.thumbnail_progress.setValue(0)

        for idx, entry in enumerate(entries, start=1):
            image_path = entry["path"]
            icon = QIcon(
                QPixmap(image_path).scaled(
                    256, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )
            filename = os.path.basename(image_path)
            item = QListWidgetItem(icon, filename)
            item.setData(Qt.UserRole, image_path)

            try:
                with Image.open(image_path) as image_handle:
                    width, height = image_handle.size
                    bit_depth = self._infer_bit_depth(image_handle.mode)
            except Exception:
                width, height, bit_depth = 0, 0, "unknown"

            item.setToolTip(
                "\n".join(
                    [
                        f"Path: {image_path}",
                        f"Dimensions: {width} x {height}",
                        f"Bit depth: {bit_depth}",
                        f"Detected index: {entry['detected_index']}",
                    ]
                )
            )
            self.thumbnail_list.addItem(item)

            if show_progress and (idx % 8 == 0 or idx == len(entries)):
                self.thumbnail_progress.setValue(idx)
                QApplication.processEvents()

        if show_progress:
            self.thumbnail_progress.setVisible(False)

        self._filter_thumbnails(self.thumbnail_filter_edit.text())

        if self.thumbnail_list.count() == 0:
            self.ingestion_preview.clear_with_text("No images loaded\nDrag your brain slice images here")

    def _on_species_changed(self):
        species = "mouse" if self.mouse_radio.isChecked() else "rat"
        self.state.set_species(species)
        self._refresh_atlas_volume_options()
        self._update_anchor_depth_range()
        self._update_ensemble_availability()
        self._update_processing_estimate()
        self._update_run_button_state()

    def _update_ensemble_availability(self):
        """Enable or disable the ensemble checkbox based on species config."""
        if not hasattr(self, "ensemble_checkbox"):
            return
        supported = self.state.supports_ensemble()
        self.ensemble_checkbox.setEnabled(supported)
        if supported:
            self.ensemble_checkbox.setToolTip(
                "Run primary and secondary models and average predictions "
                "(higher robustness, slower runtime)."
            )
        else:
            if self.ensemble_checkbox.isChecked():
                self.ensemble_checkbox.setChecked(False)
            self.ensemble_checkbox.setToolTip(
                f"Ensemble inference is not yet available for species "
                f"'{self.state.species}'. Predictions will use the primary model only."
            )

    def _on_auto_thickness_toggled(self, checked: bool):
        self.thickness_spin.setEnabled(not checked)
        self._update_run_button_state()

    def _suggest_thickness(self):
        try:
            value = self.state.estimate_section_thickness_um()
        except Exception as exc:
            self._show_logged_exception(
                title="Thickness Suggestion",
                context="Unable to estimate thickness yet",
                exc=exc,
                icon=QMessageBox.Warning,
            )
            return
        self.thickness_spin.setValue(float(value))
        self.auto_thickness_checkbox.setChecked(False)

    def _sync_legacy_checkbox(self, checked: bool):
        self.legacy_parsing_checkbox.setChecked(checked)
        self._refresh_ingestion_views()

    def _toggle_tech_insights(self, visible: bool):
        self.tech_insights.setVisible(visible)
        self.tech_toggle.setText("v Technical Insights" if visible else "> Technical Insights")

    def _toggle_console(self, visible: bool):
        self.console_output.setVisible(visible)

    def _on_direction_override_changed(self, value: str):
        if value == "Auto":
            self.state.selected_indexing_direction = self.state.detected_indexing_direction
        else:
            self.state.selected_indexing_direction = value
        self._update_processing_estimate()

    def _on_quality_controls_changed(self, *_args):
        try:
            self.state.set_quality_controls(
                outlier_sigma=float(self.outlier_sigma_spin.value()),
                confidence_medium=float(self.confidence_medium_spin.value()),
                confidence_high=float(self.confidence_high_spin.value()),
            )
        except ValueError as exc:
            self.config_validation_label.setText(f"Validation: {exc}")
            return

        settings = QSettings("DeepSlice", "GUI")
        settings.setValue("outlier_sigma", float(self.state.outlier_sigma_threshold))
        settings.setValue("confidence_medium_threshold", float(self.state.confidence_medium_threshold))
        settings.setValue("confidence_high_threshold", float(self.state.confidence_high_threshold))

        self._update_run_button_state()
        if self.state.predictions is not None:
            self._refresh_curation_views()
            self._refresh_export_views()

    def _on_inference_batch_changed(self, value: int):
        batch_size = int(max(1, value))
        self.state.inference_batch_size = batch_size
        settings = QSettings("DeepSlice", "GUI")
        settings.setValue("inference_batch_size", batch_size)
        self._update_processing_estimate()

    def _validate_before_prediction(self) -> (List[str], List[str]):
        errors: List[str] = []
        warnings: List[str] = []

        if len(self.state.image_paths) == 0:
            errors.append("No images loaded")

        if len(self.state.image_paths) < 2 and len(self.state.image_paths) > 0:
            errors.append("At least 2 sections are required")

        if len(self.state.image_paths) < 10 and len(self.state.image_paths) > 0:
            warnings.append("Dataset has fewer than 10 slices")

        if self.orientation_combo.currentIndex() != 0:
            errors.append("Only coronal orientation is currently supported")

        if self.confidence_high_spin.value() <= self.confidence_medium_spin.value():
            errors.append("High confidence threshold must be greater than medium threshold")

        if self.enable_section_numbers_checkbox.isChecked():
            index_report = self.state.build_index_report(
                legacy_section_numbers=self.legacy_parsing_checkbox.isChecked()
            )
            if index_report["parse_error"]:
                errors.append(index_report["parse_error"])
            if len(index_report["duplicate_indices"]) > 0:
                errors.append("Duplicate section numbers are not allowed")

        if not self.auto_thickness_checkbox.isChecked() and self.thickness_spin.value() <= 0:
            errors.append("Section thickness must be greater than zero")

        return errors, warnings

    def _update_run_button_state(self):
        errors, _ = self._validate_before_prediction()
        self.run_alignment_button.setEnabled(len(errors) == 0)
        if len(errors) == 0:
            self.config_validation_label.setText("Validation: ready")
        else:
            self.config_validation_label.setText("Validation: " + " | ".join(errors[:3]))
        self._update_processing_estimate()

    def _run_alignment(self):
        errors, warnings = self._validate_before_prediction()
        if len(errors) > 0:
            QMessageBox.warning(self, "Cannot Run Alignment", "\n".join(errors))
            return

        if len(warnings) > 0:
            answer = QMessageBox.question(
                self,
                "Proceed With Warnings",
                "\n".join(warnings) + "\n\nContinue anyway?",
            )
            if answer != QMessageBox.Yes:
                return

        self.state.section_numbers = self.enable_section_numbers_checkbox.isChecked()
        self.state.legacy_section_numbers = self.legacy_parsing_checkbox.isChecked()
        self.state.ensemble = self.ensemble_checkbox.isChecked()
        self.state.use_secondary_model = self.secondary_model_checkbox.isChecked()
        self._phase_total = 3 if self.state.ensemble else 2
        self._prediction_cancel_requested = False
        self._prediction_total = max(len(self.state.image_paths), 1)
        self._prediction_completed = 0
        self._prediction_phase = "initializing"

        self.run_alignment_button.setEnabled(False)
        if hasattr(self, "cancel_alignment_button"):
            self.cancel_alignment_button.setEnabled(True)
        self.prediction_progress_bar.setRange(0, max(self._prediction_total * self._phase_total, 1))
        self.prediction_progress_bar.setValue(0)
        self.console_output.clear()
        self.prediction_phase_label.setText(f"Phase 0/{self._phase_total}: initializing")
        self.prediction_progress_label.setText("Progress: 0 / 0")
        self.prediction_elapsed_label.setText("Elapsed: 00:00")
        self.prediction_eta_label.setText("Remaining: --:--")
        self._prediction_elapsed_timer.start()
        self._prediction_clock_timer.start()
        self._run_button_dots = 0
        self._run_animation_timer.start()
        self.run_alignment_button.setText("Running")
        self.setWindowTitle(f"{self._window_title_base} - Running (0%)")

        options = {
            "section_numbers": self.state.section_numbers,
            "legacy_section_numbers": self.state.legacy_section_numbers,
            "ensemble": self.state.ensemble,
            "use_secondary_model": self.state.use_secondary_model,
            "inference_batch_size": int(self.inference_batch_spin.value()),
        }

        self._current_prediction_worker = FunctionWorker(
            self._run_prediction_task,
            options,
            inject_callbacks=True,
        )
        self._current_prediction_worker.signals.progress.connect(self._on_prediction_progress)
        self._current_prediction_worker.signals.log.connect(self._append_console_log)
        self._current_prediction_worker.signals.error.connect(self._on_prediction_error)
        self._current_prediction_worker.signals.finished.connect(self._on_prediction_finished)
        self._track_worker(self._current_prediction_worker)
        self.thread_pool.start(self._current_prediction_worker)

    def _cancel_alignment(self):
        self._prediction_cancel_requested = True
        self._append_console_log("[WARNING] Cancellation requested. Waiting for current batch boundary...")
        self.cancel_alignment_button.setEnabled(False)
        self.run_alignment_button.setText("Cancelling...")

    def _run_prediction_task(self, options: dict, progress_callback=None, log_callback=None):
        def is_cancelled() -> bool:
            return bool(self._prediction_cancel_requested)

        def guarded_progress(completed: int, total: int, phase: str):
            if is_cancelled():
                raise RuntimeError("Prediction cancelled by user")
            if progress_callback is not None:
                progress_callback(completed, total, phase)

        if is_cancelled():
            raise RuntimeError("Prediction cancelled by user")

        return self.state.run_prediction(
            section_numbers=options["section_numbers"],
            legacy_section_numbers=options["legacy_section_numbers"],
            ensemble=options["ensemble"],
            use_secondary_model=options["use_secondary_model"],
            inference_batch_size=options["inference_batch_size"],
            progress_callback=guarded_progress,
            log_callback=log_callback,
            cancel_check=is_cancelled,
        )

    def _on_prediction_progress(self, completed: int, total: int, phase: str):
        self._prediction_completed = max(0, int(completed))
        self._prediction_total = max(1, int(total))
        self._prediction_phase = str(phase)

        phase_key = str(phase).lower().strip()
        if phase_key == "primary":
            phase_index = 1
            phase_name = "Primary inference"
        elif phase_key == "secondary":
            phase_index = 2 if self._phase_total >= 3 else 1
            phase_name = "Secondary inference"
        elif phase_key in {"finalize", "finalizing", "postprocess"}:
            phase_index = self._phase_total
            phase_name = "Finalize alignment"
        elif phase_key == "prepare":
            phase_index = 1
            phase_name = "Preparing runtime"
        else:
            phase_index = min(max(1, self._phase_total), 2)
            phase_name = str(phase).capitalize()

        percent = int(round((self._prediction_completed / self._prediction_total) * 100.0))

        self.prediction_phase_label.setText(
            f"Phase {phase_index}/{self._phase_total} ({phase_name}): {percent}%"
        )
        self.prediction_progress_label.setText(f"Progress: {completed} / {total}")
        overall_total = self._prediction_total * self._phase_total
        overall_completed = min(
            (phase_index - 1) * self._prediction_total + self._prediction_completed,
            overall_total,
        )
        overall_percent = int(round((overall_completed / max(overall_total, 1)) * 100.0))
        self.prediction_progress_bar.setRange(0, max(overall_total, 1))
        self.prediction_progress_bar.setValue(overall_completed)
        self.setWindowTitle(f"{self._window_title_base} - Running ({overall_percent}%)")
        self._update_prediction_timing()
        self._update_prediction_realtime_views(completed, total, phase)

    def _estimate_depth_from_progress(self, completed: int, total: int) -> float:
        total = max(int(total), 1)
        completed = int(np.clip(completed, 1, total))
        min_depth, max_depth = metadata_loader.get_species_depth_range(self.state.species)
        fraction = (completed - 1) / float(max(total - 1, 1))
        return float(min_depth + fraction * (max_depth - min_depth))

    def _update_prediction_atlas_view(
        self,
        histology_path: Optional[str],
        depth_estimate: float,
        status_prefix: str,
    ):
        try:
            atlas_result = self.state.get_atlas_slice(
                depth_value=depth_estimate,
                volume_key=self.state.default_atlas_volume().lower(),
            )
            fitted = self._fit_atlas_slice_to_histology(histology_path, atlas_result["image"])
            overlay = [
                f"Volume: {atlas_result['volume_label']}",
                f"Estimated y index: {atlas_result['slice_index']}",
                f"Estimated AP depth: {depth_estimate:.1f}",
            ]
            self.prediction_atlas_viewer.set_array_image(fitted, overlay_lines=overlay)
            self.prediction_atlas_info_label.setText(
                f"{status_prefix} | Atlas y={atlas_result['slice_index']}"
            )
        except Exception as exc:
            self.prediction_atlas_viewer.clear_with_text("Atlas preview unavailable")
            self.prediction_atlas_info_label.setText(
                f"Atlas comparison unavailable: {type(exc).__name__}"
            )

    def _update_prediction_realtime_views(self, completed: int, total: int, phase: str):
        if str(phase).lower() != "primary":
            return
        if len(self.state.image_paths) == 0:
            return

        index = int(np.clip(completed - 1, 0, len(self.state.image_paths) - 1))
        preview_path = self.state.image_paths[index]
        self.prediction_viewer.set_image(
            preview_path,
            [f"Inferring slice {index + 1}/{len(self.state.image_paths)}"],
            pixel_spacing_um=self._get_pixel_spacing_um(preview_path),
        )
        depth_estimate = self._estimate_depth_from_progress(index + 1, len(self.state.image_paths))
        self._update_prediction_atlas_view(
            preview_path,
            depth_estimate,
            status_prefix=f"Realtime slice {index + 1}/{len(self.state.image_paths)}",
        )

    def _append_console_log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}"
        lower = str(message).lower()
        color = "#DEE6EF"
        if "error" in lower or "failed" in lower:
            color = "#E05E6E"
        elif "warn" in lower:
            color = "#E3A33B"
        elif "analysis" in lower or "system" in lower:
            color = "#69C0FF"
        html = (
            f"<span style='color:{color};'>"
            + full_message.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            + "</span>"
        )
        self.console_output.append(html)
        if self.console_autoscroll_toggle.isChecked():
            scroll = self.console_output.verticalScrollBar()
            scroll.setValue(scroll.maximum())

    def _update_prediction_timing(self):
        if not self._prediction_elapsed_timer.isValid():
            return
        elapsed_seconds = max(0, int(self._prediction_elapsed_timer.elapsed() / 1000))
        self.prediction_elapsed_label.setText(f"Elapsed: {self._format_duration(elapsed_seconds)}")

        if self._prediction_completed <= 0 or self._prediction_total <= 0:
            self.prediction_eta_label.setText("Remaining: --:--")
            return

        rate = self._prediction_completed / max(elapsed_seconds, 1)
        remaining = max(self._prediction_total - self._prediction_completed, 0)
        eta = int(round(remaining / max(rate, 1e-6)))
        self.prediction_eta_label.setText(f"Remaining: ~{self._format_duration(eta)}")

    def _animate_run_button(self):
        if self.run_alignment_button.isEnabled():
            self.run_alignment_button.setText("Run Alignment")
            return
        self._run_button_dots = (self._run_button_dots + 1) % 4
        dots = "." * self._run_button_dots
        self.run_alignment_button.setText("Running" + dots)

    def _stop_prediction_activity(self):
        self._prediction_clock_timer.stop()
        self._run_animation_timer.stop()
        self.run_alignment_button.setText("Run Alignment")
        self._update_session_status()

    def _finalize_prediction_result(self, result: dict, recovered_partial: bool = False):
        self._curation_modified = False
        self._baseline_predictions = self.state.predictions.copy() if self.state.predictions is not None else None
        if recovered_partial:
            self._session_base_text = f"Session: Partial recovery ({result['slice_count']} slices)"
        else:
            self._session_base_text = f"Session: Predicted {result['slice_count']} slices"
        self._update_session_status()

        direction = result.get("direction")
        if direction:
            self.detected_direction_label.setText(f"Detected direction: {direction}")
            self.prediction_direction_label.setText(
                f"Detected indexing direction: {direction}"
            )

        predicted_thickness = result.get("predicted_thickness_um")
        if predicted_thickness is not None:
            self.predicted_thickness_label.setText(
                f"Estimated thickness: {predicted_thickness:.2f} um"
            )
            self.accept_predicted_thickness_button.setEnabled(True)
        else:
            self.predicted_thickness_label.setText("Estimated thickness: unavailable")
            self.accept_predicted_thickness_button.setEnabled(False)

        elapsed_seconds = max(0, int(self._prediction_elapsed_timer.elapsed() / 1000))
        self._refresh_prediction_selector()
        self._refresh_curation_views()
        self._refresh_export_views()
        self._refresh_step_states()
        self._update_run_button_state()

        mean_conf = "-"
        outlier_count = 0
        if self._linearity_payload is not None and "confidence" in self._linearity_payload:
            mean_conf = f"{float(np.mean(self._linearity_payload['confidence'])):.2f}"
        if self._linearity_payload is not None and "outliers" in self._linearity_payload:
            outlier_count = int(np.sum(np.asarray(self._linearity_payload["outliers"], dtype=bool)))

        diagnostics: List[str] = []
        out_of_bounds_count = int(result.get("out_of_bounds_count", 0) or 0)
        angle_outlier_count = int(result.get("angle_outlier_count", 0) or 0)
        orthogonality_count = int(result.get("orthogonality_count", 0) or 0)
        if out_of_bounds_count > 0:
            diagnostics.append(f"{out_of_bounds_count} AP depth(s) outside atlas range")
        if angle_outlier_count > 0:
            diagnostics.append(f"{angle_outlier_count} section(s) with abrupt angle deviations")
        if orthogonality_count > 0:
            diagnostics.append(f"{orthogonality_count} section(s) with non-orthogonal U/V vectors")

        if recovered_partial:
            reason = str(result.get("partial_reason", "secondary ensemble pass failed")).strip()
            summary_text = (
                f"Recovered partial result in {self._format_duration(elapsed_seconds)} - "
                f"{int(result['slice_count'])} slices - Mean confidence: {mean_conf} - Outliers: {outlier_count}"
            )
            self._show_toast(summary_text, timeout_ms=7000, level="warning")
            self._append_console_log("[WARNING] " + reason)
            for detail in diagnostics:
                self._append_console_log("[WARNING] " + detail)
            self._append_console_log("[SYSTEM] " + summary_text)
            return

        summary_text = (
            f"Completed in {self._format_duration(elapsed_seconds)} - "
            f"{int(result['slice_count'])} slices processed - Mean confidence: {mean_conf} - Outliers: {outlier_count}"
        )
        toast_level = "warning" if len(diagnostics) > 0 else "success"
        self._show_toast(summary_text, timeout_ms=6000, level=toast_level)
        for detail in diagnostics:
            self._append_console_log("[WARNING] " + detail)
        self._append_console_log("[SYSTEM] " + summary_text)

    def _on_prediction_error(self, error_text: str):
        self._stop_prediction_activity()
        self._prediction_cancel_requested = False
        self.run_alignment_button.setEnabled(True)
        if hasattr(self, "cancel_alignment_button"):
            self.cancel_alignment_button.setEnabled(False)

        if "cancelled" in str(error_text).lower():
            self._show_toast("Alignment cancelled", timeout_ms=3500)
            self._append_console_log("[SYSTEM] Alignment cancelled by user")
            self.prediction_phase_label.setText("Phase: cancelled")
            self.prediction_eta_label.setText("Remaining: --:--")
            return

        if "PARTIAL_PREDICTIONS_AVAILABLE" in str(error_text):
            if self.state.has_partial_prediction_candidate():
                reason = self.state.partial_prediction_reason().splitlines()[0].strip()
                if len(reason) > 220:
                    reason = reason[:217] + "..."
                reply = QMessageBox.question(
                    self,
                    "Recover Partial Result",
                    (
                        "The secondary ensemble pass failed, but primary-model predictions are available.\n\n"
                        f"Reason: {reason}\n\n"
                        "Use the partial result now?"
                    ),
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )
                if reply == QMessageBox.Yes:
                    try:
                        result = self.state.use_partial_prediction_candidate(log_callback=self._append_console_log)
                    except Exception as exc:
                        self._show_logged_exception(
                            title="Partial Recovery Failed",
                            context="Could not adopt partial primary-model predictions",
                            exc=exc,
                            icon=QMessageBox.Critical,
                        )
                        return
                    self._finalize_prediction_result(result, recovered_partial=True)
                    return

                self.state.clear_partial_prediction_candidate()
                self._show_toast("Partial result discarded", timeout_ms=2800, level="info")
                return

            error_text = str(error_text).replace("PARTIAL_PREDICTIONS_AVAILABLE:", "").strip()

        self._show_logged_error(
            title="Prediction Failed",
            context="Alignment prediction task failed",
            error_text=error_text,
            icon=QMessageBox.Critical,
        )

    def _on_prediction_finished(self, result: dict):
        self._stop_prediction_activity()
        self._prediction_cancel_requested = False
        self.run_alignment_button.setEnabled(True)
        if hasattr(self, "cancel_alignment_button"):
            self.cancel_alignment_button.setEnabled(False)
        self._finalize_prediction_result(result, recovered_partial=False)

    def _accept_predicted_thickness(self):
        label_text = self.predicted_thickness_label.text()
        try:
            value_text = label_text.split(":", 1)[1].replace("um", "").strip()
            value = float(value_text)
        except Exception as exc:
            self._show_logged_exception(
                title="Thickness",
                context="No predicted thickness value is available to apply",
                exc=exc,
                icon=QMessageBox.Information,
            )
            return
        self.auto_thickness_checkbox.setChecked(False)
        self.thickness_spin.setValue(value)

    def _refresh_prediction_selector(self):
        self.prediction_slice_selector.blockSignals(True)
        self.prediction_slice_selector.clear()

        if self.state.predictions is not None and len(self.state.predictions) > 0:
            for _, row in self.state.predictions.iterrows():
                nr = row["nr"] if "nr" in row else "-"
                self.prediction_slice_selector.addItem(f"{nr} | {row['Filenames']}")

        self.prediction_slice_selector.blockSignals(False)
        self._refresh_prediction_preview()

    def _refresh_prediction_preview(self):
        if self.state.predictions is None or len(self.state.predictions) == 0:
            self.prediction_viewer.clear_with_text("No predictions to preview")
            self.prediction_atlas_viewer.clear_with_text("No atlas comparison available")
            self.prediction_atlas_info_label.setText("Atlas comparison: waiting for prediction")
            return

        row_index = self.prediction_slice_selector.currentIndex()
        if row_index < 0:
            row_index = 0
        row = self.state.predictions.iloc[row_index]

        image_path = self._resolve_image_path_for_filename(row["Filenames"])
        overlay = [
            f"Filename: {row['Filenames']}",
            f"O=({row['ox']:.2f}, {row['oy']:.2f}, {row['oz']:.2f})",
            f"U=({row['ux']:.2f}, {row['uy']:.2f}, {row['uz']:.2f})",
            f"V=({row['vx']:.2f}, {row['vy']:.2f}, {row['vz']:.2f})",
        ]
        self.prediction_viewer.set_image(
            image_path,
            overlay,
            pixel_spacing_um=self._get_pixel_spacing_um(image_path),
        )

        depth_estimate = self._depth_for_prediction_index(row_index)
        self._update_prediction_atlas_view(
            image_path,
            depth_estimate,
            status_prefix=f"Selected slice {row_index + 1}/{len(self.state.predictions)}",
        )

    def _depth_for_prediction_index(self, row_index: int) -> float:
        if self._linearity_payload is not None and "y" in self._linearity_payload:
            y_values = self._linearity_payload["y"]
            if 0 <= row_index < len(y_values):
                return float(y_values[row_index])

        if self.state.predictions is not None and len(self.state.predictions) > 0:
            try:
                payload = self.state.linearity_payload()
                y_values = payload["y"]
                if 0 <= row_index < len(y_values):
                    return float(y_values[row_index])
            except Exception:
                pass

            return self._estimate_depth_from_progress(row_index + 1, len(self.state.predictions))

        return self._estimate_depth_from_progress(row_index + 1, 1)

    def _resolve_image_path_for_filename(self, filename: str) -> Optional[str]:
        filename = os.path.basename(filename)
        for image_path in self.state.image_paths:
            if os.path.basename(image_path) == filename:
                return image_path
        return None

    def _refresh_atlas_volume_options(self):
        if not hasattr(self, "atlas_volume_combo"):
            return
        options = self.state.atlas_volume_options()
        current = self.atlas_volume_combo.currentText().strip().lower()
        normalized_options = [option.lower() for option in options]

        self.atlas_volume_combo.blockSignals(True)
        self.atlas_volume_combo.clear()
        for option in options:
            self.atlas_volume_combo.addItem(option)

        target = current if current in normalized_options else self.state.default_atlas_volume().lower()
        index = -1
        for option_index in range(self.atlas_volume_combo.count()):
            if self.atlas_volume_combo.itemText(option_index).strip().lower() == target:
                index = option_index
                break
        if index >= 0:
            self.atlas_volume_combo.setCurrentIndex(index)
        elif self.atlas_volume_combo.count() > 0:
            self.atlas_volume_combo.setCurrentIndex(0)
        self.atlas_volume_combo.blockSignals(False)

    def _on_atlas_preview_toggled(self, enabled: bool):
        row = self.slice_flag_list.currentRow()
        if row < 0 and self.state.predictions is not None and len(self.state.predictions) > 0:
            row = 0
            self.slice_flag_list.setCurrentRow(0)

        if not enabled:
            self.atlas_slice_info_label.setText("Atlas: disabled")
            self.atlas_coords_label.setText("Atlas coords: -")
            self.atlas_viewer.clear_with_text("Enable atlas preview to load atlas slices")
            self._latest_atlas_slice = None
            self._latest_atlas_meta = None
            self._render_histology_preview(row)
            self._render_confidence_overlay_preview(row)
            return
        self._request_atlas_preview(row)

    def _on_atlas_volume_changed(self, _value: str):
        if not self.enable_atlas_preview_checkbox.isChecked():
            return
        row = self.slice_flag_list.currentRow()
        if row < 0 and self.state.predictions is not None and len(self.state.predictions) > 0:
            row = 0
            self.slice_flag_list.setCurrentRow(0)
        self._request_atlas_preview(row)

    def _on_blend_overlay_toggled(self, enabled: bool):
        self.blend_slider.setEnabled(enabled)
        row = self.slice_flag_list.currentRow()
        if row < 0 and self.state.predictions is not None and len(self.state.predictions) > 0:
            row = 0
            self.slice_flag_list.setCurrentRow(0)
        if enabled and self.enable_atlas_preview_checkbox.isChecked() and self._latest_atlas_slice is None:
            self._request_atlas_preview(row)
        else:
            self._render_histology_preview(row)
            self._render_confidence_overlay_preview(row)

    def _on_blend_slider_changed(self, value: int):
        self.blend_percent_label.setText(f"Blend: {value}%")
        if not self.enable_blend_overlay_checkbox.isChecked():
            return
        row = self.slice_flag_list.currentRow()
        if row < 0 and self.state.predictions is not None and len(self.state.predictions) > 0:
            row = 0
            self.slice_flag_list.setCurrentRow(0)
        self._render_histology_preview(row)
        self._render_confidence_overlay_preview(row)

    def _on_loupe_toggled(self, enabled: bool):
        self.curation_viewer.set_loupe_enabled(enabled)
        self.atlas_viewer.set_loupe_enabled(enabled)

    def _on_atlas_mouse_moved(self, scene_x: float, scene_y: float):
        if self._latest_atlas_meta is None:
            self.atlas_coords_label.setText("Atlas coords: -")
            return

        depth = self._latest_atlas_meta.get("depth", None)
        if depth is None:
            depth_text = "AP: -"
        else:
            depth_text = f"AP: {float(depth):.1f}"
        self.atlas_coords_label.setText(
            f"Atlas coords: {depth_text} | DV: {scene_y:.1f} | ML: {scene_x:.1f}"
        )

    def _step_curation_slice(self, delta: int):
        if not hasattr(self, "slice_flag_list"):
            return
        count = self.slice_flag_list.count()
        if count <= 0:
            return

        step = 1 if delta >= 0 else -1
        row = self.slice_flag_list.currentRow()
        if row < 0:
            row = 0 if step > 0 else count - 1

        for _ in range(count):
            row = int(np.clip(row + step, 0, count - 1))
            item = self.slice_flag_list.item(row)
            if item is not None and not item.isHidden():
                self.slice_flag_list.setCurrentRow(row)
                return
            if row in {0, count - 1}:
                break

    def _selected_prediction_row_from_list(self) -> Optional[int]:
        if not hasattr(self, "slice_flag_list"):
            return None
        row = self.slice_flag_list.currentRow()
        if row < 0:
            return None
        item = self.slice_flag_list.item(row)
        if item is None:
            return None
        source_index = item.data(Qt.UserRole)
        if source_index is None:
            source_index = row
        try:
            return int(source_index)
        except Exception:
            return row

    def _list_row_for_prediction_index(self, source_index: int) -> int:
        if not hasattr(self, "slice_flag_list"):
            return -1
        for row in range(self.slice_flag_list.count()):
            item = self.slice_flag_list.item(row)
            if item is None:
                continue
            candidate = item.data(Qt.UserRole)
            if candidate is None:
                candidate = row
            try:
                if int(candidate) == int(source_index):
                    return row
            except Exception:
                continue
        return -1

    def _refresh_anchor_list(self):
        if not hasattr(self, "anchor_list"):
            return
        self.anchor_list.clear()

        if self.state.predictions is None:
            self._anchor_depth_targets = {}
            return

        n_rows = len(self.state.predictions)
        cleaned: Dict[int, float] = {}
        for row_idx, target_depth in self._anchor_depth_targets.items():
            try:
                index = int(row_idx)
                depth = float(target_depth)
            except Exception:
                continue
            if 0 <= index < n_rows and np.isfinite(depth):
                cleaned[index] = depth
        self._anchor_depth_targets = cleaned

        for row_idx in sorted(self._anchor_depth_targets.keys()):
            row = self.state.predictions.iloc[row_idx]
            nr = row["nr"] if "nr" in self.state.predictions.columns else row_idx + 1
            filename = os.path.basename(str(row["Filenames"]))
            target_depth = self._anchor_depth_targets[row_idx]
            item = QListWidgetItem(f"#{nr} | AP {target_depth:.2f} | {filename}")
            item.setData(Qt.UserRole, row_idx)
            self.anchor_list.addItem(item)

    def _update_anchor_depth_range(self):
        if not hasattr(self, "anchor_depth_spin"):
            return
        try:
            min_depth, max_depth = metadata_loader.get_species_depth_range(self.state.species)
            min_depth = float(min_depth)
            max_depth = float(max_depth)
            padding = max(5.0, 0.1 * abs(max_depth - min_depth))
            self.anchor_depth_spin.setRange(min_depth - padding, max_depth + padding)
        except Exception:
            self.anchor_depth_spin.setRange(-5000.0, 5000.0)

    def _sync_anchor_editor_with_selection(self, source_row_index: int):
        if not hasattr(self, "anchor_depth_spin"):
            return
        if source_row_index < 0:
            return

        if source_row_index in self._anchor_depth_targets:
            value = float(self._anchor_depth_targets[source_row_index])
        else:
            value = float(self._depth_for_prediction_index(source_row_index))

        self.anchor_depth_spin.blockSignals(True)
        self.anchor_depth_spin.setValue(value)
        self.anchor_depth_spin.blockSignals(False)

    def _set_anchor_for_current_slice(self):
        if self.state.predictions is None:
            return
        source_index = self._selected_prediction_row_from_list()
        if source_index is None or source_index < 0 or source_index >= len(self.state.predictions):
            return

        self._anchor_depth_targets[source_index] = float(self.anchor_depth_spin.value())
        self._refresh_anchor_list()
        self._show_toast(f"Anchor set for slice {source_index + 1}", timeout_ms=2200)

    def _remove_anchor_for_current_slice(self):
        source_index = self._selected_prediction_row_from_list()
        if source_index is None:
            return
        if source_index in self._anchor_depth_targets:
            self._anchor_depth_targets.pop(source_index, None)
            self._refresh_anchor_list()
            self._show_toast(f"Anchor removed for slice {source_index + 1}", timeout_ms=2200)

    def _clear_anchor_points(self):
        if len(self._anchor_depth_targets) == 0:
            return
        self._anchor_depth_targets = {}
        self._refresh_anchor_list()
        self._show_toast("Anchor set cleared", timeout_ms=2200)

    def _apply_anchor_interpolation(self):
        if self.state.predictions is None:
            return
        if len(self._anchor_depth_targets) < 2:
            self._show_toast(
                "Set at least two anchors before distributing AP depths",
                timeout_ms=3200,
                level="warning",
            )
            return

        n_rows = len(self.state.predictions)
        anchors = sorted(
            [
                (int(index), float(depth))
                for index, depth in self._anchor_depth_targets.items()
                if 0 <= int(index) < n_rows
            ],
            key=lambda pair: pair[0],
        )
        if len(anchors) < 2:
            self._show_toast(
                "Anchors are out of range for current predictions",
                timeout_ms=3200,
                level="warning",
            )
            return

        payload = self.state.linearity_payload()
        current_depths = np.asarray(payload["y"], dtype=float)
        if current_depths.shape[0] != n_rows:
            QMessageBox.warning(self, "Anchor Alignment", "Depth payload does not match prediction table length.")
            return

        target_depths = np.asarray(current_depths, dtype=float).copy()

        first_index, first_target = anchors[0]
        first_shift = first_target - current_depths[first_index]
        target_depths[first_index] = first_target
        if first_index > 0:
            target_depths[:first_index] = current_depths[:first_index] + first_shift

        for (start_index, start_depth), (end_index, end_depth) in zip(anchors[:-1], anchors[1:]):
            if end_index <= start_index:
                continue
            segment_x = np.arange(start_index, end_index + 1, dtype=float)
            alpha = (segment_x - float(start_index)) / float(end_index - start_index)
            target_depths[start_index : end_index + 1] = (
                (1.0 - alpha) * float(start_depth)
                + (alpha * float(end_depth))
            )

        last_index, last_target = anchors[-1]
        last_shift = last_target - current_depths[last_index]
        target_depths[last_index] = last_target
        if last_index + 1 < n_rows:
            target_depths[last_index + 1 :] = current_depths[last_index + 1 :] + last_shift

        delta_depths = target_depths - current_depths
        if not np.all(np.isfinite(delta_depths)):
            QMessageBox.warning(self, "Anchor Alignment", "Computed depth correction contains invalid values.")
            return

        self.state.snapshot_predictions()
        self.state.predictions["oy"] = self.state.predictions["oy"].astype(float) + delta_depths
        self.state.is_dirty = True
        self.state._sync_model_predictions()

        self._mark_curation_modified()
        self._refresh_curation_views()

        focused_index = anchors[0][0]
        target_row = self._list_row_for_prediction_index(focused_index)
        if target_row >= 0:
            self.slice_flag_list.setCurrentRow(target_row)

        self._show_toast(
            f"Anchor interpolation applied using {len(anchors)} anchors",
            timeout_ms=3500,
        )

    def _toggle_confidence_panel(self, expanded: bool):
        if hasattr(self, "confidence_panel"):
            self.confidence_panel.setVisible(bool(expanded))
        if hasattr(self, "confidence_panel_toggle"):
            self.confidence_panel_toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        if expanded:
            self._render_confidence_overlay_preview(self.slice_flag_list.currentRow())

    def _build_confidence_overlay_image(
        self,
        histology_path: Optional[str],
        confidence: float,
        atlas_slice: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if histology_path is None or not os.path.exists(histology_path):
            return None

        try:
            with Image.open(histology_path) as image:
                hist_gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        except Exception:
            return None

        agreement = np.ones_like(hist_gray, dtype=np.float32)
        if atlas_slice is not None:
            fitted = self._fit_atlas_slice_to_histology(histology_path, atlas_slice)
            if fitted is not None and np.asarray(fitted).ndim == 2:
                atlas_map = np.asarray(fitted, dtype=np.float32) / 255.0
                if atlas_map.shape != hist_gray.shape:
                    atlas_map = np.asarray(
                        Image.fromarray(np.uint8(np.clip(atlas_map * 255.0, 0, 255))).resize(
                            (hist_gray.shape[1], hist_gray.shape[0]), Image.BILINEAR
                        ),
                        dtype=np.float32,
                    ) / 255.0
                agreement = 1.0 - np.abs(hist_gray - atlas_map)
                agreement = np.clip((agreement - 0.15) / 0.85, 0.0, 1.0)

        confidence = float(np.clip(confidence, 0.0, 1.0))
        red = np.clip(80.0 + (hist_gray * 175.0), 0.0, 255.0)
        green = np.clip((40.0 + (215.0 * confidence)) * agreement, 0.0, 255.0)
        blue = np.clip(hist_gray * 55.0, 0.0, 255.0)
        return np.stack([red, green, blue], axis=2).astype(np.uint8)

    def _render_confidence_overlay_preview(self, row_index: int):
        if not hasattr(self, "confidence_overlay_viewer"):
            return
        if self.state.predictions is None or len(self.state.predictions) == 0:
            self.confidence_overlay_viewer.clear_with_text("No predictions available")
            return
        if row_index < 0 or row_index >= len(self.state.predictions):
            return

        row = self.state.predictions.iloc[row_index]
        image_path = self._resolve_image_path_for_filename(row["Filenames"])

        confidence = 0.0
        level = "low"
        if self._linearity_payload is not None:
            confidence = float(self._linearity_payload["confidence"][row_index])
            level = str(self._linearity_payload["confidence_level"][row_index])

        if level == "high":
            border_color = QColor("#2CC784")
        elif level == "medium":
            border_color = QColor("#E3A33B")
        else:
            border_color = QColor("#D33E56")

        overlay_image = self._build_confidence_overlay_image(
            image_path,
            confidence,
            self._latest_atlas_slice,
        )
        if overlay_image is None:
            self.confidence_overlay_viewer.clear_with_text("Confidence overlay unavailable")
            return

        self.confidence_overlay_viewer.set_array_image(
            overlay_image,
            overlay_lines=[
                f"Slice: {row['Filenames']}",
                f"Composite confidence: {confidence:.2f} ({level})",
                "Green intensity increases with confidence and atlas agreement",
            ],
            border_color=border_color,
        )

    def _on_atlas_transform_changed(self, _value=None):
        if hasattr(self, "atlas_scale_label"):
            self.atlas_scale_label.setText(f"Scale: {self.atlas_scale_slider.value()}%")
        if hasattr(self, "atlas_offset_x_label"):
            self.atlas_offset_x_label.setText(f"Offset X: {self.atlas_offset_x_slider.value()}%")
        if hasattr(self, "atlas_offset_y_label"):
            self.atlas_offset_y_label.setText(f"Offset Y: {self.atlas_offset_y_slider.value()}%")

        row = self.slice_flag_list.currentRow()
        if row < 0 and self.state.predictions is not None and len(self.state.predictions) > 0:
            row = 0
            self.slice_flag_list.setCurrentRow(0)

        self._refresh_atlas_viewer_display(row)
        self._render_histology_preview(row)
        self._render_confidence_overlay_preview(row)

    @staticmethod
    def _mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        ys, xs = np.nonzero(mask)
        if ys.size < 16 or xs.size < 16:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)

    @staticmethod
    def _bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x0, y0, x1, y1 = bbox
        return (0.5 * (x0 + x1), 0.5 * (y0 + y1))

    def _best_bbox_from_masks(
        self,
        masks: List[np.ndarray],
        frame_shape: Tuple[int, int],
    ) -> Optional[Tuple[int, int, int, int]]:
        frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
        frame_area = float(max(frame_h * frame_w, 1))
        best_bbox = None
        best_score = float("inf")

        for mask in masks:
            bbox = self._mask_bbox(mask)
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            area_ratio = ((x1 - x0) * (y1 - y0)) / frame_area
            if area_ratio < 0.01 or area_ratio > 0.98:
                continue
            # Favor compact tissue-like regions around one-third to one-half of frame.
            score = abs(area_ratio - 0.38)
            if score < best_score:
                best_score = score
                best_bbox = bbox

        return best_bbox

    def _fit_atlas_slice_to_histology(
        self,
        histology_path: Optional[str],
        atlas_slice: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if atlas_slice is None:
            return None

        source = np.asarray(atlas_slice, dtype=np.uint8)
        if source.ndim != 2 or source.size == 0:
            return atlas_slice

        if hasattr(self, "atlas_flip_x_checkbox") and self.atlas_flip_x_checkbox.isChecked():
            source = np.fliplr(source)
        if hasattr(self, "atlas_flip_y_checkbox") and self.atlas_flip_y_checkbox.isChecked():
            source = np.flipud(source)

        if histology_path is None or not os.path.exists(histology_path):
            return source

        try:
            with Image.open(histology_path) as image:
                target_width, target_height = image.size
                hist_gray = np.asarray(image.convert("L"), dtype=np.float32)
        except Exception:
            return source

        if hasattr(self, "atlas_rotate_combo"):
            rotation_k = self.atlas_rotate_combo.currentData()
            if rotation_k == "auto":
                target_ratio = target_width / float(max(target_height, 1))
                candidates = [0, -1, 2, 1]
                best_k = 0
                best_error = float("inf")
                for candidate in candidates:
                    rotated = np.rot90(source, int(candidate)) if int(candidate) != 0 else source
                    ratio = rotated.shape[1] / float(max(rotated.shape[0], 1))
                    error = abs(np.log((ratio + 1e-6) / (target_ratio + 1e-6)))
                    if error < best_error:
                        best_error = error
                        best_k = int(candidate)
                if best_k != 0:
                    source = np.rot90(source, best_k)
            else:
                if rotation_k is None:
                    rotation_k = 0
                if int(rotation_k) != 0:
                    source = np.rot90(source, int(rotation_k))

        src_h, src_w = source.shape
        if src_h <= 0 or src_w <= 0:
            return source

        hist_masks = [
            hist_gray <= np.percentile(hist_gray, 80.0),
            hist_gray <= np.percentile(hist_gray, 88.0),
            hist_gray >= np.percentile(hist_gray, 25.0),
        ]
        hist_bbox = self._best_bbox_from_masks(hist_masks, (target_height, target_width))

        atlas_float = source.astype(np.float32)
        atlas_masks = [
            atlas_float >= np.percentile(atlas_float, 55.0),
            atlas_float >= np.percentile(atlas_float, 65.0),
            atlas_float > 0,
        ]
        atlas_bbox = self._best_bbox_from_masks(atlas_masks, (src_h, src_w))

        if hist_bbox is not None and atlas_bbox is not None:
            hist_bw = max(hist_bbox[2] - hist_bbox[0], 1)
            hist_bh = max(hist_bbox[3] - hist_bbox[1], 1)
            atlas_bw = max(atlas_bbox[2] - atlas_bbox[0], 1)
            atlas_bh = max(atlas_bbox[3] - atlas_bbox[1], 1)
            base_scale = min(hist_bw / float(atlas_bw), hist_bh / float(atlas_bh))
        else:
            # Fallback: contain-fit atlas in full frame without forced crop.
            base_scale = min(target_width / float(src_w), target_height / float(src_h))

        base_scale = max(base_scale, 1e-3)
        manual_scale = 1.0
        if hasattr(self, "atlas_scale_slider"):
            manual_scale = max(self.atlas_scale_slider.value(), 1) / 100.0
        scale = base_scale * manual_scale
        resized_w = max(1, int(round(src_w * scale)))
        resized_h = max(1, int(round(src_h * scale)))

        resized = Image.fromarray(source, mode="L").resize(
            (resized_w, resized_h), Image.BILINEAR
        )
        resized_array = np.array(resized, dtype=np.uint8)

        resized_masks = [
            resized_array >= np.percentile(resized_array, 55.0),
            resized_array >= np.percentile(resized_array, 65.0),
            resized_array > 0,
        ]
        atlas_resized_bbox = self._best_bbox_from_masks(resized_masks, (resized_h, resized_w))

        hist_center_x, hist_center_y = target_width / 2.0, target_height / 2.0
        if hist_bbox is not None:
            hist_center_x, hist_center_y = self._bbox_center(hist_bbox)

        atlas_center_x, atlas_center_y = resized_w / 2.0, resized_h / 2.0
        if atlas_resized_bbox is not None:
            atlas_center_x, atlas_center_y = self._bbox_center(atlas_resized_bbox)

        offset_x_pct = self.atlas_offset_x_slider.value() if hasattr(self, "atlas_offset_x_slider") else 0
        offset_y_pct = self.atlas_offset_y_slider.value() if hasattr(self, "atlas_offset_y_slider") else 0
        offset_x_px = (offset_x_pct / 100.0) * target_width
        offset_y_px = (offset_y_pct / 100.0) * target_height

        place_x = int(round(hist_center_x - atlas_center_x + offset_x_px))
        place_y = int(round(hist_center_y - atlas_center_y + offset_y_px))

        canvas = np.zeros((target_height, target_width), dtype=np.uint8)

        src_x0 = max(0, -place_x)
        src_y0 = max(0, -place_y)
        dst_x0 = max(0, place_x)
        dst_y0 = max(0, place_y)

        copy_w = min(target_width - dst_x0, resized_w - src_x0)
        copy_h = min(target_height - dst_y0, resized_h - src_y0)

        if copy_w > 0 and copy_h > 0:
            canvas[dst_y0 : dst_y0 + copy_h, dst_x0 : dst_x0 + copy_w] = resized_array[
                src_y0 : src_y0 + copy_h,
                src_x0 : src_x0 + copy_w,
            ]

        return canvas

    def _build_blended_overlay_image(
        self,
        histology_path: Optional[str],
        atlas_slice: Optional[np.ndarray],
        alpha: float,
    ) -> Optional[np.ndarray]:
        if histology_path is None or atlas_slice is None:
            return None
        if not os.path.exists(histology_path):
            return None

        try:
            with Image.open(histology_path) as image:
                image = image.convert("RGB")
                hist_array = np.array(image, dtype=np.float32)
        except Exception:
            return None

        atlas_array = np.asarray(atlas_slice, dtype=np.float32)
        if atlas_array.ndim != 2:
            return None

        fitted = self._fit_atlas_slice_to_histology(
            histology_path,
            np.uint8(np.clip(atlas_array, 0, 255)),
        )
        if fitted is None:
            return None
        atlas_resized = np.asarray(fitted, dtype=np.float32) / 255.0

        # Tint atlas signal as cyan heat overlay.
        overlay = np.stack(
            [
                atlas_resized * 35.0,
                atlas_resized * 210.0,
                atlas_resized * 245.0,
            ],
            axis=2,
        )
        alpha = float(np.clip(alpha, 0.0, 1.0))
        blended = ((1.0 - alpha) * hist_array) + (alpha * overlay)
        return np.uint8(np.clip(blended, 0, 255))

    def _render_histology_preview(self, row_index: int):
        if self.state.predictions is None:
            self.curation_viewer.clear_with_text("No predictions to curate")
            return
        if row_index < 0 or row_index >= len(self.state.predictions):
            return

        row = self.state.predictions.iloc[row_index]
        filename = row["Filenames"]
        image_path = self._resolve_image_path_for_filename(filename)

        confidence = 1.0
        confidence_level = "high"
        if self._linearity_payload is not None:
            confidence = float(self._linearity_payload["confidence"][row_index])
            confidence_level = str(self._linearity_payload["confidence_level"][row_index])

        if confidence_level == "high":
            border_color = QColor("#2CC784")
        elif confidence_level == "medium":
            border_color = QColor("#E3A33B")
        else:
            border_color = QColor("#D33E56")

        overlay = [
            f"Slice: {filename}",
            f"Composite confidence: {confidence:.2f} ({confidence_level})",
        ]

        ap_depth_vox = float(row["oy"])
        if self._linearity_payload is not None and row_index < len(self._linearity_payload["y"]):
            ap_depth_vox = float(self._linearity_payload["y"][row_index])

        voxel_um = None
        try:
            voxel_um = float(self.state._config["target_volumes"][self.state.species]["voxel_size_microns"])
        except Exception:
            voxel_um = None

        if voxel_um is not None and np.isfinite(ap_depth_vox):
            ap_mm = (ap_depth_vox * voxel_um) / 1000.0
            overlay.append(f"AP: {ap_depth_vox:.2f} vox ({ap_mm:.3f} mm)")
        else:
            overlay.append(f"AP: {ap_depth_vox:.2f} vox")

        overlay.append(f"Oy: {row['oy']:.2f}")

        if (
            self.before_after_toggle.isChecked()
            and self._baseline_predictions is not None
            and row_index < len(self._baseline_predictions)
        ):
            baseline_row = self._baseline_predictions.iloc[row_index]
            baseline_oy = float(baseline_row.get("oy", row["oy"]))
            delta_oy = float(row["oy"]) - baseline_oy
            overlay.append(f"Before Oy: {baseline_oy:.2f} | Delta: {delta_oy:+.2f}")

        if self.enable_blend_overlay_checkbox.isChecked():
            alpha = self.blend_slider.value() / 100.0
            blended = self._build_blended_overlay_image(image_path, self._latest_atlas_slice, alpha)
            if blended is not None:
                overlay.append(f"Atlas blend alpha: {alpha:.2f}")
                self.curation_viewer.set_array_image(
                    blended,
                    overlay_lines=overlay,
                    border_color=border_color,
                )
                return

        self.curation_viewer.set_image(
            image_path,
            overlay,
            border_color=border_color,
            pixel_spacing_um=self._get_pixel_spacing_um(image_path),
        )

    def _request_atlas_preview(self, row_index: int):
        if not hasattr(self, "enable_atlas_preview_checkbox"):
            return
        if not self.enable_atlas_preview_checkbox.isChecked():
            return
        if self.state.predictions is None or len(self.state.predictions) == 0:
            return
        if row_index < 0 or row_index >= len(self.state.predictions):
            return

        self._atlas_request_token += 1
        request_token = self._atlas_request_token

        depth_value = None
        if self._linearity_payload is not None and "y" in self._linearity_payload:
            depth_value = float(self._linearity_payload["y"][row_index])
        volume_key = self.atlas_volume_combo.currentText().strip().lower()
        self.atlas_slice_info_label.setText("Atlas: loading...")

        worker = FunctionWorker(
            self._atlas_preview_task,
            depth_value,
            volume_key,
            request_token,
            inject_callbacks=True,
        )
        worker.signals.progress.connect(self._on_atlas_progress)
        worker.signals.log.connect(self._append_console_log)
        worker.signals.error.connect(self._on_atlas_error)
        worker.signals.finished.connect(self._on_atlas_ready)
        self._track_worker(worker)
        self.thread_pool.start(worker)

    def _atlas_preview_task(
        self,
        depth_value: Optional[float],
        volume_key: str,
        request_token: int,
        progress_callback=None,
        log_callback=None,
    ):
        result = self.state.get_atlas_slice(
            depth_value=depth_value,
            volume_key=volume_key,
            progress_callback=progress_callback,
            log_callback=log_callback,
        )
        result["request_token"] = request_token
        return result

    def _on_atlas_progress(self, completed: int, total: int, phase: str):
        if phase == "atlas-download":
            if total > 0:
                percent = (completed / total) * 100.0
                self.atlas_slice_info_label.setText(f"Atlas download: {percent:.1f}%")
            else:
                self.atlas_slice_info_label.setText(f"Atlas download: {completed} bytes")
        elif phase == "atlas-ready":
            self.atlas_slice_info_label.setText("Atlas: rendering")

    def _on_atlas_error(self, error_text: str):
        self._record_error("Atlas preview task failed", error_text)
        self.atlas_slice_info_label.setText("Atlas: failed")
        self._latest_atlas_slice = None
        self._latest_atlas_meta = None
        self.atlas_viewer.clear_with_text("Atlas preview failed. Check console for details.")

    def _on_atlas_ready(self, result: dict):
        if result.get("request_token") != self._atlas_request_token:
            return

        image = result["image"]
        slice_index = result["slice_index"]
        shape = result["shape"]
        volume_label = str(result["volume_label"])
        self.atlas_slice_info_label.setText(
            f"Atlas: {volume_label} | y={slice_index}/{shape[1] - 1}"
        )
        self._latest_atlas_slice = image
        self._latest_atlas_meta = result

        row = self.slice_flag_list.currentRow()
        self._refresh_atlas_viewer_display(row)
        self._render_confidence_overlay_preview(row)

        if self.enable_blend_overlay_checkbox.isChecked():
            self._render_histology_preview(row)

    def _refresh_atlas_viewer_display(self, row_index: int):
        if self._latest_atlas_slice is None or self._latest_atlas_meta is None:
            return

        result = self._latest_atlas_meta
        image = self._latest_atlas_slice
        slice_index = result["slice_index"]
        shape = result["shape"]
        volume_label = str(result["volume_label"])

        row = self.slice_flag_list.currentRow()
        if row_index >= 0:
            row = row_index

        histology_path = None
        if self.state.predictions is not None and 0 <= row < len(self.state.predictions):
            filename = str(self.state.predictions.iloc[row]["Filenames"])
            histology_path = self._resolve_image_path_for_filename(filename)

        display_image = self._fit_atlas_slice_to_histology(histology_path, image)
        if display_image is None:
            display_image = image

        overlay = [
            f"Volume: {volume_label}",
            f"Coronal index (y): {slice_index}",
            f"Volume shape: {shape}",
            f"Display shape: {display_image.shape}",
        ]
        self.atlas_viewer.set_array_image(display_image, overlay_lines=overlay)

    def _refresh_curation_views(self):
        self.slice_flag_list.clear()
        self.linearity_axis.clear()
        self.hist_axis.clear()
        self._linearity_payload = None

        if self.state.predictions is None or len(self.state.predictions) == 0:
            self.curation_viewer.clear_with_text("No predictions to curate")
            self.atlas_slice_info_label.setText("Atlas: disabled")
            self.atlas_coords_label.setText("Atlas coords: -")
            self.atlas_viewer.clear_with_text("Enable atlas preview to load atlas slices")
            if hasattr(self, "confidence_overlay_viewer"):
                self.confidence_overlay_viewer.clear_with_text("No predictions available")
            if hasattr(self, "slice_note_edit"):
                self.slice_note_edit.clear()
            self.linearity_canvas.draw_idle()
            self.hist_canvas.draw_idle()
            self._update_undo_redo_labels()
            self._refresh_anchor_list()
            return

        payload = self.state.linearity_payload()
        self._linearity_payload = payload

        for idx, row in self.state.predictions.iterrows():
            nr = row["nr"] if "nr" in row else idx + 1
            filename = row["Filenames"]
            ap_pos = float(payload["y"][idx]) if idx < len(payload["y"]) else float("nan")
            item = QListWidgetItem(f"{nr} | AP {ap_pos:.1f} | {filename}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)

            is_bad = False
            if "bad_section" in self.state.predictions.columns:
                is_bad = bool(self.state.predictions.iloc[idx]["bad_section"])
            item.setCheckState(Qt.Checked if is_bad else Qt.Unchecked)

            confidence = float(payload["confidence"][idx])
            confidence_level = str(payload["confidence_level"][idx])
            if confidence_level == "high":
                item.setBackground(QColor(44, 129, 79, 120))
            elif confidence_level == "medium":
                item.setBackground(QColor(150, 111, 36, 120))
            else:
                item.setBackground(QColor(164, 51, 68, 120))

            components = payload["confidence_components"]
            tooltip_lines = [
                f"Score: {confidence:.3f} ({confidence_level})",
                f"Residual: {payload['residuals'][idx]:.3f}",
                f"Angle deviation: {payload['angle_deviation'][idx]:.3f}",
                f"Spacing deviation: {payload['spacing_deviation'][idx]:.3f}",
                f"Gaussian weight: {payload['weights'][idx]:.3f}",
                f"Components: residual={components['residual'][idx]:.2f}, angle={components['angle'][idx]:.2f}, spacing={components['spacing'][idx]:.2f}, center={components['center_weight'][idx]:.2f}",
            ]
            if "atlas_note" in self.state.predictions.columns:
                note_text = str(self.state.predictions.iloc[idx].get("atlas_note", "")).strip()
                if note_text and note_text.lower() != "nan":
                    tooltip_lines.append(f"Note: {note_text}")
            item.setToolTip("\n".join(tooltip_lines))
            item.setData(Qt.UserRole, idx)
            self.slice_flag_list.addItem(item)

        x = payload["x"]
        y = payload["y"]
        trend = payload["trend"]
        weights = payload["weights"]
        outliers = payload["outliers"]

        self.linearity_axis.set_facecolor("#11161D")
        self.linearity_axis.scatter(
            x,
            y,
            c=payload["confidence"],
            cmap="RdYlGn",
            edgecolors="white",
            linewidths=0.5,
            s=36,
            label="Sections",
        )
        self.linearity_axis.plot(x, trend, color="#1E6FFF", linewidth=2.0, label="Linear fit")

        if len(y) > 0:
            y_min, y_max = float(np.min(y)), float(np.max(y))
            if y_max == y_min:
                y_max = y_min + 1.0
            scaled_weights = y_min + (weights * (y_max - y_min))
            self.linearity_axis.plot(
                x,
                scaled_weights,
                color="#52E5FF",
                linestyle="--",
                linewidth=1.6,
                label="Gaussian weighting (scaled)",
            )

        if np.any(outliers):
            self.linearity_axis.scatter(
                x[outliers],
                y[outliers],
                facecolors="none",
                edgecolors="#E03A4F",
                linewidths=1.6,
                s=80,
                label="Outliers",
            )

        self.linearity_axis.set_xlabel("Section Index")
        self.linearity_axis.set_ylabel("Predicted AP Position")
        self.linearity_axis.tick_params(colors="#B9C7D8")
        self.linearity_axis.xaxis.label.set_color("#B9C7D8")
        self.linearity_axis.yaxis.label.set_color("#B9C7D8")
        self.linearity_axis.spines["bottom"].set_color("#3B4655")
        self.linearity_axis.spines["left"].set_color("#3B4655")
        self.linearity_axis.spines["top"].set_color("#3B4655")
        self.linearity_axis.spines["right"].set_color("#3B4655")
        legend = self.linearity_axis.legend(loc="best", facecolor="#1A1F26", edgecolor="#2A313B")
        if legend is not None:
            legend.get_frame().set_alpha(0.95)
            for text in legend.get_texts():
                text.set_color("#EAF2FF")

        self.hist_axis.set_facecolor("#11161D")
        bins = np.linspace(0.0, 1.0, 11)
        self.hist_axis.hist(payload["confidence"], bins=bins, color="#3A8DFF", edgecolor="#CFE0FF", alpha=0.9)
        self.hist_axis.set_xlabel("Confidence")
        self.hist_axis.set_ylabel("Count")
        self.hist_axis.tick_params(colors="#B9C7D8")
        self.hist_axis.xaxis.label.set_color("#B9C7D8")
        self.hist_axis.yaxis.label.set_color("#B9C7D8")
        self.hist_axis.spines["bottom"].set_color("#3B4655")
        self.hist_axis.spines["left"].set_color("#3B4655")
        self.hist_axis.spines["top"].set_color("#3B4655")
        self.hist_axis.spines["right"].set_color("#3B4655")

        self.linearity_canvas.draw_idle()
        self.hist_canvas.draw_idle()
        self._update_undo_redo_labels()
        self._refresh_anchor_list()

        self.slice_flag_list.setCurrentRow(0)
        self._refresh_export_views()

    def _on_linearity_click(self, event):
        if self._linearity_payload is None or event.xdata is None:
            return
        x_values = self._linearity_payload["x"]
        nearest = int(np.argmin(np.abs(x_values - event.xdata)))
        target_row = self._list_row_for_prediction_index(nearest)
        if target_row < 0:
            target_row = nearest
        self.slice_flag_list.setCurrentRow(target_row)
        item = self.slice_flag_list.item(target_row)
        if item is not None:
            self.slice_flag_list.scrollToItem(item, QAbstractItemView.PositionAtCenter)
        self._on_curation_slice_selected(target_row)

    def _linearity_zoom(self, scale: float):
        if self._linearity_payload is None:
            return
        scale = float(scale)
        xlim = self.linearity_axis.get_xlim()
        ylim = self.linearity_axis.get_ylim()
        x_center = 0.5 * (xlim[0] + xlim[1])
        y_center = 0.5 * (ylim[0] + ylim[1])
        x_half = 0.5 * (xlim[1] - xlim[0]) * scale
        y_half = 0.5 * (ylim[1] - ylim[0]) * scale
        self.linearity_axis.set_xlim(x_center - x_half, x_center + x_half)
        self.linearity_axis.set_ylim(y_center - y_half, y_center + y_half)
        self.linearity_canvas.draw_idle()

    def _linearity_zoom_fit(self):
        if self._linearity_payload is None:
            return
        self.linearity_axis.relim()
        self.linearity_axis.autoscale_view()
        self.linearity_canvas.draw_idle()

    def _update_undo_redo_labels(self):
        if hasattr(self, "undo_button"):
            self.undo_button.setText(f"Undo ({len(self.state.undo_stack)})")
        if hasattr(self, "redo_button"):
            self.redo_button.setText(f"Redo ({len(self.state.redo_stack)})")

    def _mark_curation_modified(self):
        self._curation_modified = True
        self._refresh_step_states()

    def _set_all_flags(self, check_state: Qt.CheckState):
        if self.state.predictions is None:
            return
        for idx in range(self.slice_flag_list.count()):
            item = self.slice_flag_list.item(idx)
            if not item.isHidden():
                item.setCheckState(check_state)
                
    def _filter_curation_list(self, filter_index: int):
        if self.state.predictions is None or self._linearity_payload is None:
            return
        
        levels = self._linearity_payload["confidence_level"]
        
        for idx in range(self.slice_flag_list.count()):
            item = self.slice_flag_list.item(idx)
            source_index = item.data(Qt.UserRole)
            if source_index is None:
                source_index = idx
            source_index = int(source_index)
            level = levels[source_index]
            
            if filter_index == 0:  # All Confidences
                item.setHidden(False)
            elif filter_index == 1:  # High Only
                item.setHidden(level != "high")
            elif filter_index == 2:  # Medium Only
                item.setHidden(level != "medium")
            elif filter_index == 3:  # Low Only
                item.setHidden(level != "low")

    def _show_curation_context_menu(self, pos):
        item = self.slice_flag_list.itemAt(pos)
        if item is None:
            return

        source_index = item.data(Qt.UserRole)
        if source_index is None:
            source_index = self.slice_flag_list.row(item)
        source_index = int(source_index)

        menu = QMenu(self)
        if item.checkState() == Qt.Checked:
            toggle_flag_action = menu.addAction("Unflag as Bad")
        else:
            toggle_flag_action = menu.addAction("Flag as Bad")
        jump_viewer_action = menu.addAction("Jump to Viewer")
        set_anchor_action = menu.addAction("Set Anchor Here")
        remove_anchor_action = menu.addAction("Remove Anchor Here")
        edit_note_action = menu.addAction("Edit Atlas Note")
        copy_path_action = menu.addAction("Copy File Path")
        show_details_action = menu.addAction("Show Confidence Details")

        action = menu.exec(self.slice_flag_list.viewport().mapToGlobal(pos))

        if action == toggle_flag_action:
            new_state = Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked
            item.setCheckState(new_state)
        elif action == jump_viewer_action:
            self.slice_flag_list.setCurrentItem(item)
        elif action == set_anchor_action:
            self.slice_flag_list.setCurrentItem(item)
            self._set_anchor_for_current_slice()
        elif action == remove_anchor_action:
            self.slice_flag_list.setCurrentItem(item)
            self._remove_anchor_for_current_slice()
        elif action == edit_note_action:
            self.slice_flag_list.setCurrentItem(item)
            if hasattr(self, "slice_note_edit"):
                self.slice_note_edit.setFocus()
                self.slice_note_edit.selectAll()
        elif action == copy_path_action:
            if self.state.predictions is not None and 0 <= source_index < len(self.state.predictions):
                filename = str(self.state.predictions.iloc[source_index]["Filenames"])
                image_path = self._resolve_image_path_for_filename(filename)
                QApplication.clipboard().setText(image_path or filename)
                self._show_toast("Slice path copied", timeout_ms=1800)
        elif action == show_details_action:
            QMessageBox.information(self, "Confidence Details", item.toolTip())

    def _on_curation_slice_selected(self, row_index: int):
        if self.state.predictions is None:
            return
        item = self.slice_flag_list.item(row_index) if hasattr(self, "slice_flag_list") else None
        source_index = row_index
        if item is not None:
            source = item.data(Qt.UserRole)
            if source is not None:
                try:
                    source_index = int(source)
                except Exception:
                    source_index = row_index

        if source_index < 0 or source_index >= len(self.state.predictions):
            return

        if hasattr(self, "slice_note_edit"):
            note_text = ""
            if "atlas_note" in self.state.predictions.columns:
                note_text = str(self.state.predictions.iloc[source_index].get("atlas_note", ""))
                if note_text.lower() == "nan":
                    note_text = ""
            self.slice_note_edit.setText(note_text)

        self._sync_anchor_editor_with_selection(source_index)

        self._render_histology_preview(source_index)
        self._render_confidence_overlay_preview(source_index)
        self._request_atlas_preview(source_index)

    def _apply_bad_section_flags(self):
        if self.state.predictions is None:
            return

        bad_sections = []
        for idx in range(self.slice_flag_list.count()):
            item = self.slice_flag_list.item(idx)
            if item.checkState() == Qt.Checked:
                source_index = item.data(Qt.UserRole)
                source_index = idx if source_index is None else int(source_index)
                filename = self.state.predictions.iloc[source_index]["Filenames"]
                bad_sections.append(str(filename))

        try:
            self.state.set_bad_sections(bad_sections, auto=False)
        except Exception as exc:
            self._show_logged_exception(
                title="Bad Section Flagging",
                context="Unable to apply bad section flags",
                exc=exc,
                icon=QMessageBox.Warning,
            )
            return

        self._mark_curation_modified()
        self._refresh_curation_views()

    def _move_selected_slice(self, direction: int):
        if self.state.predictions is None:
            return
        count = self.slice_flag_list.count()
        if count <= 1:
            return

        current = self.slice_flag_list.currentRow()
        if current < 0:
            return

        target = int(np.clip(current + int(np.sign(direction)), 0, count - 1))
        if target == current:
            return

        item = self.slice_flag_list.takeItem(current)
        self.slice_flag_list.insertItem(target, item)
        self.slice_flag_list.setCurrentRow(target)

    def _save_slice_note(self):
        if self.state.predictions is None:
            return

        row_index = self.slice_flag_list.currentRow()
        if row_index < 0 or row_index >= self.slice_flag_list.count():
            return

        item = self.slice_flag_list.item(row_index)
        source_index = item.data(Qt.UserRole)
        source_index = row_index if source_index is None else int(source_index)

        if "atlas_note" not in self.state.predictions.columns:
            self.state.snapshot_predictions()
            self.state.predictions["atlas_note"] = ""

        note_text = self.slice_note_edit.text().strip()
        previous = str(self.state.predictions.iloc[source_index].get("atlas_note", ""))
        if note_text != previous:
            self.state.snapshot_predictions()
            self.state.predictions.at[source_index, "atlas_note"] = note_text
            self._mark_curation_modified()
            self._refresh_curation_views()
            self.slice_flag_list.setCurrentRow(row_index)

    def _apply_manual_order(self):
        if self.state.predictions is None:
            return

        ordered_indices = []
        for idx in range(self.slice_flag_list.count()):
            item = self.slice_flag_list.item(idx)
            source_index = item.data(Qt.UserRole)
            ordered_indices.append(int(source_index))

        try:
            self.state.apply_manual_order(ordered_indices)
        except Exception as exc:
            self._show_logged_exception(
                title="Manual Reordering",
                context="Unable to apply manual section reordering",
                exc=exc,
                icon=QMessageBox.Warning,
            )
            return

        self._mark_curation_modified()
        self._refresh_curation_views()

    def _detect_outliers(self):
        if self.state.predictions is None:
            return
        try:
            self.state.set_bad_sections([], auto=True)
        except Exception as exc:
            self._show_logged_exception(
                title="Outlier Detection",
                context="Unable to detect outlier sections",
                exc=exc,
                icon=QMessageBox.Warning,
            )
            return
        self._mark_curation_modified()
        self._refresh_curation_views()

    def _normalize_angles(self):
        if self.state.predictions is None:
            return
        try:
            self.state.propagate_angles()
        except Exception as exc:
            self._show_logged_exception(
                title="Normalize Angles",
                context="Unable to normalize section angles",
                exc=exc,
                icon=QMessageBox.Warning,
            )
            return
        self._mark_curation_modified()
        self._refresh_curation_views()

    def _apply_manual_angles(self):
        if self.state.predictions is None:
            return
        try:
            self.state.adjust_angles(
                ml_angle=float(self.ml_spin.value()),
                dv_angle=float(self.dv_spin.value()),
            )
        except Exception as exc:
            self._show_logged_exception(
                title="Manual Angle Override",
                context="Unable to apply manual angle override",
                exc=exc,
                icon=QMessageBox.Warning,
            )
            return
        self._mark_curation_modified()
        self._refresh_curation_views()

    def _enforce_index_order(self):
        if self.state.predictions is None:
            return
        try:
            self.state.enforce_index_order()
        except Exception as exc:
            self._show_logged_exception(
                title="Enforce Index Order",
                context="Unable to enforce index order",
                exc=exc,
                icon=QMessageBox.Warning,
            )
            return
        self._mark_curation_modified()
        self._refresh_curation_views()

    def _enforce_index_spacing(self):
        if self.state.predictions is None:
            return
        section_thickness = None
        if not self.auto_thickness_checkbox.isChecked():
            section_thickness = float(self.thickness_spin.value())
        try:
            self.state.enforce_index_spacing(section_thickness_um=section_thickness)
        except Exception as exc:
            self._show_logged_exception(
                title="Enforce Index Spacing",
                context="Unable to enforce index spacing",
                exc=exc,
                icon=QMessageBox.Warning,
            )
            return
        self._mark_curation_modified()
        self._refresh_curation_views()

    def _undo(self):
        try:
            self.state.undo()
        except Exception as exc:
            self._show_logged_exception(
                title="Undo",
                context="Undo operation failed",
                exc=exc,
                icon=QMessageBox.Information,
            )
            return
        self._mark_curation_modified()
        self._refresh_curation_views()

    def _redo(self):
        try:
            self.state.redo()
        except Exception as exc:
            self._show_logged_exception(
                title="Redo",
                context="Redo operation failed",
                exc=exc,
                icon=QMessageBox.Information,
            )
            return
        self._mark_curation_modified()
        self._refresh_curation_views()

    def _browse_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if directory:
            self.output_dir_edit.setText(directory)

    def _browse_quicknii_path(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Locate QuickNII Executable",
            "",
            "Executables (*.exe);;All Files (*)",
        )
        if path:
            self.quicknii_path_edit.setText(path)

    def _get_persisted_export_path(self) -> str:
        settings = QSettings("DeepSlice", "GUI")
        path = settings.value("export_directory", "")
        if not path:
            path = settings.value("default_output_directory", "")
        return path if path else os.getcwd()

    def _persist_export_path(self, path: str):
        settings = QSettings("DeepSlice", "GUI")
        settings.setValue("export_directory", path)

    def _get_persisted_quicknii_path(self) -> str:
        settings = QSettings("DeepSlice", "GUI")
        return settings.value("quicknii_path", "")

    def _persist_quicknii_path(self, path: str):
        settings = QSettings("DeepSlice", "GUI")
        settings.setValue("quicknii_path", path)

    def _is_output_dir_writable(self, output_dir: str) -> bool:
        try:
            os.makedirs(output_dir, exist_ok=True)
            probe = os.path.join(output_dir, ".deepslice_write_test")
            with open(probe, "w", encoding="utf-8") as handle:
                handle.write("ok")
            os.remove(probe)
            return True
        except Exception:
            return False

    def _update_export_size_estimate(self, index: int):
        if self.state.predictions is None:
            self.export_size_estimate_label.setText("~0 MB")
            return
            
        num_slices = len(self.state.predictions)
        # Very rough estimates: JSON ~ 50KB/slice, XML ~ 80KB/slice
        kb_per_slice = 50 if index == 0 else 80
        total_mb = (num_slices * kb_per_slice) / 1024
        
        # Add sidecar CSV estimate
        total_mb += (num_slices * 2) / 1024
        
        if total_mb < 0.1:
            self.export_size_estimate_label.setText("< 0.1 MB")
        else:
            self.export_size_estimate_label.setText(f"~{total_mb:.1f} MB")

    def _open_export_directory(self):
        output_dir = self.output_dir_edit.text().strip()
        if not os.path.exists(output_dir):
            QMessageBox.warning(self, "Export Folder", "The export folder does not exist yet.")
            return
            
        try:
            if os.name == "nt":
                os.startfile(output_dir)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", output_dir])
        except Exception as exc:
            self._show_logged_exception(
                title="Open Export Folder",
                context="Unable to open the export directory",
                exc=exc,
                icon=QMessageBox.Warning,
            )

    def _copy_export_path(self):
        if self.last_export_basepath is None:
            self._show_toast(
                "Export predictions first to generate an output path",
                timeout_ms=3200,
                level="warning",
            )
            return

        output_format = "json" if self.output_format_combo.currentIndex() == 0 else "xml"
        target_file = self.last_export_basepath + f".{output_format}"
        
        QApplication.clipboard().setText(target_file)
        self._show_toast(f"Copied export path: {target_file}", timeout_ms=3000)

    def _export_predictions(self):
        if self.state.predictions is None:
            QMessageBox.warning(self, "Export", "No predictions available")
            return

        output_dir = self.output_dir_edit.text().strip()
        base_name = self.output_basename_edit.text().strip()
        if not output_dir or not base_name:
            QMessageBox.warning(
                self,
                "Export",
                "Output directory and base filename are required",
            )
            return

        if not self._is_output_dir_writable(output_dir):
            QMessageBox.warning(
                self,
                "Export",
                "Output directory is not writable. Please choose another location.",
            )
            return

        low_confidence_count = 0
        try:
            payload = self.state.linearity_payload()
            confidence_values = np.asarray(payload.get("confidence", []), dtype=float)
            low_confidence_count = int(np.sum(confidence_values < 0.50))
        except Exception:
            low_confidence_count = 0

        if low_confidence_count > 0:
            answer = QMessageBox.question(
                self,
                "Quality Warning",
                (
                    f"{low_confidence_count} slice(s) are currently low confidence.\n"
                    "Export anyway?"
                ),
            )
            if answer != QMessageBox.Yes:
                return

        os.makedirs(output_dir, exist_ok=True)
        base_path = os.path.join(output_dir, base_name)
        output_format = "json" if self.output_format_combo.currentIndex() == 0 else "xml"

        try:
            self.state.save_predictions(base_path, output_format=output_format)
        except Exception as exc:
            self._show_logged_exception(
                title="Export Failed",
                context="Prediction export failed",
                exc=exc,
                icon=QMessageBox.Critical,
            )
            return

        self.last_export_basepath = base_path
        self._session_base_text = "Session: Export complete"
        self._update_session_status()
        self._show_toast("Export complete - 2 files saved", timeout_ms=4500)

    def _generate_report(self):
        if self.state.predictions is None:
            QMessageBox.warning(self, "Report", "No predictions available")
            return

        output_dir = self.output_dir_edit.text().strip()
        base_name = self.output_basename_edit.text().strip() or "DeepSliceResults"
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, base_name + "_report.pdf")

        summary = self.state.summary_metrics()
        options = {
            "species": self.state.species,
            "ensemble": bool(self.ensemble_checkbox.isChecked()),
            "use_secondary_model": bool(self.secondary_model_checkbox.isChecked()),
            "section_numbers": bool(self.enable_section_numbers_checkbox.isChecked()),
            "legacy_section_numbers": bool(self.legacy_parsing_checkbox.isChecked()),
            "direction": self.state.selected_indexing_direction,
            "thickness_um": None
            if self.auto_thickness_checkbox.isChecked()
            else float(self.thickness_spin.value()),
            "include_stats": self.pdf_include_stats.isChecked(),
            "include_plot": self.pdf_include_plot.isChecked(),
            "include_images": self.pdf_include_images.isChecked(),
            "include_angles": self.pdf_include_angles.isChecked(),
        }
        if self._linearity_payload is not None:
            options["linearity_payload"] = {
                "x": np.asarray(self._linearity_payload.get("x", []), dtype=float).tolist(),
                "y": np.asarray(self._linearity_payload.get("y", []), dtype=float).tolist(),
                "trend": np.asarray(self._linearity_payload.get("trend", []), dtype=float).tolist(),
                "confidence": np.asarray(self._linearity_payload.get("confidence", []), dtype=float).tolist(),
            }

        try:
            reporting.generate_pdf_report(
                output_path=report_path,
                summary=summary,
                options=options,
            )
        except Exception as exc:
            self._show_logged_exception(
                title="Report Failed",
                context="Report generation failed",
                exc=exc,
                icon=QMessageBox.Critical,
            )
            return

        self._show_toast(f"Report created: {report_path}", timeout_ms=4200, level="success")

    def _preview_report(self):
        if self.state.predictions is None:
            QMessageBox.warning(self, "Report Preview", "No predictions available")
            return
            
        import tempfile
        temp_dir = tempfile.gettempdir()
        temp_report_path = os.path.join(temp_dir, "DeepSlice_Report_Preview.pdf")
        
        summary = self.state.summary_metrics()
        options = {
            "species": self.state.species,
            "ensemble": bool(self.ensemble_checkbox.isChecked()),
            "use_secondary_model": bool(self.secondary_model_checkbox.isChecked()),
            "section_numbers": bool(self.enable_section_numbers_checkbox.isChecked()),
            "legacy_section_numbers": bool(self.legacy_parsing_checkbox.isChecked()),
            "direction": self.state.selected_indexing_direction,
            "thickness_um": None
            if self.auto_thickness_checkbox.isChecked()
            else float(self.thickness_spin.value()),
            "include_stats": self.pdf_include_stats.isChecked(),
            "include_plot": self.pdf_include_plot.isChecked(),
            "include_images": self.pdf_include_images.isChecked(),
            "include_angles": self.pdf_include_angles.isChecked(),
        }
        if self._linearity_payload is not None:
            options["linearity_payload"] = {
                "x": np.asarray(self._linearity_payload.get("x", []), dtype=float).tolist(),
                "y": np.asarray(self._linearity_payload.get("y", []), dtype=float).tolist(),
                "trend": np.asarray(self._linearity_payload.get("trend", []), dtype=float).tolist(),
                "confidence": np.asarray(self._linearity_payload.get("confidence", []), dtype=float).tolist(),
            }

        self.status_bar.showMessage("Generating report preview...", 5000)
        QApplication.processEvents()

        try:
            reporting.generate_pdf_report(
                output_path=temp_report_path,
                summary=summary,
                options=options,
            )
            
            if os.name == "nt":
                os.startfile(temp_report_path)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", temp_report_path])
                
        except Exception as exc:
            self._show_logged_exception(
                title="Preview Failed",
                context="Report preview generation failed",
                exc=exc,
                icon=QMessageBox.Critical,
            )
            return

    def _open_in_quicknii(self):
        if self.last_export_basepath is None:
            self._show_toast(
                "Export predictions first to open QuickNII",
                timeout_ms=3200,
                level="warning",
            )
            return

        target_file = self.last_export_basepath + ".json"
        if not os.path.exists(target_file):
            self._show_toast(
                "QuickNII launch expects a JSON export",
                timeout_ms=3200,
                level="warning",
            )
            return

        quicknii_path = self.quicknii_path_edit.text().strip()
        if not quicknii_path:
            candidates = [
                os.path.join(os.environ.get("ProgramFiles", ""), "QuickNII", "QuickNII.exe"),
                os.path.join(
                    os.environ.get("ProgramFiles(x86)", ""),
                    "QuickNII",
                    "QuickNII.exe",
                ),
            ]
            for candidate in candidates:
                if candidate and os.path.exists(candidate):
                    quicknii_path = candidate
                    break

        if not quicknii_path or not os.path.exists(quicknii_path):
            QMessageBox.warning(
                self,
                "QuickNII",
                "QuickNII executable not found. Set path in the export panel.",
            )
            return

        try:
            subprocess.Popen([quicknii_path, target_file])
        except Exception as exc:
            self._show_logged_exception(
                title="QuickNII",
                context="Failed to launch QuickNII",
                exc=exc,
                icon=QMessageBox.Critical,
            )
            return

    def _refresh_export_views(self):
        summary = self.state.summary_metrics()
        self.summary_label.setText(
            f"Processed: {summary['processed']} | Excluded: {summary['excluded']} | Total: {summary['slice_count']}"
        )
        self.deviation_label.setText(
            f"Mean angular deviation: {summary['mean_angular_deviation']:.2f} deg"
        )

        if self.state.predictions is not None and len(self.state.predictions) > 0:
            payload = self.state.linearity_payload()
            levels = payload["confidence_level"]
            high = int(np.sum(levels == "high"))
            medium = int(np.sum(levels == "medium"))
            low = int(np.sum(levels == "low"))
            self.deviation_label.setText(
                f"Mean angular deviation: {summary['mean_angular_deviation']:.2f} deg | Confidence H/M/L: {high}/{medium}/{low}"
            )

        if self.state.predictions is not None and "markers" in self.state.predictions.columns:
            count = int(
                np.sum(
                    [
                        isinstance(marker, (list, tuple)) and len(marker) > 0
                        for marker in self.state.predictions["markers"]
                    ]
                )
            )
            self.markers_label.setText(
                f"Loaded marker annotations found in {count} slices and preserved on JSON export."
            )
        else:
            self.markers_label.setText("")

    def _load_anchor_targets_from_payload(self, payload: dict):
        self._anchor_depth_targets = {}
        anchors_payload = payload.get("anchor_depth_targets", {})
        if not isinstance(anchors_payload, dict):
            self._refresh_anchor_list()
            return

        for key, value in anchors_payload.items():
            try:
                row_index = int(key)
                target_depth = float(value)
            except Exception:
                continue
            if np.isfinite(target_depth):
                self._anchor_depth_targets[row_index] = target_depth

        self._refresh_anchor_list()

    def _save_session(self):
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save DeepSlice GUI Session",
            "",
            "DeepSlice Session (*.deepslice-session.json)",
        )
        if not filename:
            return

        if not filename.endswith(".deepslice-session.json"):
            filename = filename + ".deepslice-session.json"

        self._set_session_io_busy(True)
        self.save_session_button.setText("Saving...")
        self.save_session_button.setEnabled(False)
        QApplication.processEvents()

        try:
            payload = self.state.to_session_dict()
            payload["anchor_depth_targets"] = {
                str(index): float(depth)
                for index, depth in self._anchor_depth_targets.items()
            }
            with open(filename, "w", encoding="utf-8") as file_handle:
                json.dump(payload, file_handle, indent=2)
        except Exception as exc:
            self._set_session_io_busy(False)
            self.save_session_button.setText("Save Session")
            self.save_session_button.setEnabled(True)
            self._show_logged_exception(
                title="Save Session",
                context="Failed to save DeepSlice session",
                exc=exc,
                icon=QMessageBox.Critical,
            )
            return

        self._set_session_io_busy(False)
        self.save_session_button.setText("Save Session")
        self.save_session_button.setEnabled(True)
        self.state.is_dirty = False
        self._session_base_text = f"Session: Saved {os.path.basename(filename)}"
        self._update_session_status()
        self._add_recent_session(filename)
        self._show_toast("Session saved", timeout_ms=3000)

    def _load_session_or_quint(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load Session or QuickNII",
            "",
            "Session/QuickNII (*.json *.xml *.deepslice-session.json)",
        )
        if not filename:
            return
        self._load_session_file(filename)

    def _load_session_file(self, filename: str):
        self._set_session_io_busy(True)
        if filename.lower().endswith(".deepslice-session.json"):
            try:
                with open(filename, "r", encoding="utf-8") as file_handle:
                    payload = json.load(file_handle)
                self.state.load_session_dict(payload)
                self._load_anchor_targets_from_payload(payload)
                self.state.is_dirty = False
                self._apply_state_to_widgets()
                self._baseline_predictions = self.state.predictions.copy() if self.state.predictions is not None else None
                self._curation_modified = False
                self._session_base_text = f"Session: Loaded {os.path.basename(filename)}"
                self._update_session_status()
                self._refresh_all_views()
                self._add_recent_session(filename)
            except Exception as exc:
                self._set_session_io_busy(False)
                self._show_logged_exception(
                    title="Load Session",
                    context="Failed to load DeepSlice session",
                    exc=exc,
                    icon=QMessageBox.Critical,
                )
                return
            self._set_session_io_busy(False)
            return

        if filename.lower().endswith(".json"):
            try:
                with open(filename, "r", encoding="utf-8") as file_handle:
                    payload = json.load(file_handle)
                if payload.get("session_format") == "deepslice_gui_v1":
                    self.state.load_session_dict(payload)
                    self._load_anchor_targets_from_payload(payload)
                    self.state.is_dirty = False
                    self._apply_state_to_widgets()
                    self._baseline_predictions = self.state.predictions.copy() if self.state.predictions is not None else None
                    self._curation_modified = False
                    self._session_base_text = f"Session: Loaded {os.path.basename(filename)}"
                    self._update_session_status()
                    self._refresh_all_views()
                    self._add_recent_session(filename)
                    self._set_session_io_busy(False)
                    return
            except Exception:
                pass

        worker = FunctionWorker(self._load_quint_task, filename, inject_callbacks=True)
        worker.signals.log.connect(self._append_console_log)
        worker.signals.error.connect(self._on_load_quint_error)
        worker.signals.finished.connect(lambda res: self._on_load_quint_finished(res, filename))
        self._track_worker(worker)
        self.thread_pool.start(worker)

    def _load_quint_task(self, filename: str, progress_callback=None, log_callback=None):
        return self.state.load_quint(filename, log_callback=log_callback)

    def _on_load_quint_finished(self, result: dict, filename: str = None):
        self._set_session_io_busy(False)
        self.state.is_dirty = False
        self._anchor_depth_targets = {}
        if self.state.species == "mouse":
            self.mouse_radio.setChecked(True)
        else:
            self.rat_radio.setChecked(True)

        self._refresh_prediction_selector()
        self._refresh_curation_views()
        self._refresh_export_views()
        self._refresh_step_states()
        self._curation_modified = False
        self._baseline_predictions = self.state.predictions.copy() if self.state.predictions is not None else None

        marker_note = ""
        if result.get("marker_count", 0) > 0:
            marker_note = f"\nMarkers preserved: {result['marker_count']}"

        self._session_base_text = f"Session: Loaded {result['slice_count']} slices ({result['species']})"
        self._update_session_status()
        if filename:
            self._add_recent_session(filename)
        self._show_toast(
            f"Loaded {result['slice_count']} slices as {result['species']}{marker_note}",
            timeout_ms=4200,
            level="success",
        )

    def _on_load_quint_error(self, error_text: str):
        self._set_session_io_busy(False)
        self._show_logged_error(
            title="Load Session",
            context="Failed to load QuickNII file",
            error_text=error_text,
            icon=QMessageBox.Critical,
        )

    def _apply_state_to_widgets(self):
        self.mouse_radio.setChecked(self.state.species == "mouse")
        self.rat_radio.setChecked(self.state.species == "rat")
        self._refresh_atlas_volume_options()
        self._update_anchor_depth_range()
        self.enable_section_numbers_checkbox.setChecked(self.state.section_numbers)
        self.legacy_parsing_checkbox.setChecked(self.state.legacy_section_numbers)
        self.legacy_from_config_checkbox.setChecked(self.state.legacy_section_numbers)
        if self.state.ensemble is not None:
            self.ensemble_checkbox.setChecked(bool(self.state.ensemble))
        self.secondary_model_checkbox.setChecked(bool(self.state.use_secondary_model))

        if self.state.selected_indexing_direction in {"rostro-caudal", "caudal-rostro"}:
            self.direction_override_combo.setCurrentText(self.state.selected_indexing_direction)
        else:
            self.direction_override_combo.setCurrentText("Auto")

        if hasattr(self, "outlier_sigma_spin"):
            self.outlier_sigma_spin.blockSignals(True)
            self.outlier_sigma_spin.setValue(float(self.state.outlier_sigma_threshold))
            self.outlier_sigma_spin.blockSignals(False)
        if hasattr(self, "confidence_medium_spin"):
            self.confidence_medium_spin.blockSignals(True)
            self.confidence_medium_spin.setValue(float(self.state.confidence_medium_threshold))
            self.confidence_medium_spin.blockSignals(False)
        if hasattr(self, "confidence_high_spin"):
            self.confidence_high_spin.blockSignals(True)
            self.confidence_high_spin.setValue(float(self.state.confidence_high_threshold))
            self.confidence_high_spin.blockSignals(False)
        if hasattr(self, "inference_batch_spin"):
            self.inference_batch_spin.blockSignals(True)
            self.inference_batch_spin.setValue(int(self.state.inference_batch_size))
            self.inference_batch_spin.blockSignals(False)
        self._refresh_anchor_list()

    def _show_hardware_health(self):
        try:


            import tensorflow as tf
            gpus = tf.config.list_physical_devices("GPU")
            build_info = tf.sysconfig.get_build_info()
            cuda_version = build_info.get("cuda_version", "unknown")
            cudnn_version = build_info.get("cudnn_version", "unknown")

            mode = "GPU" if len(gpus) > 0 else "CPU"
            self.hardware_mode_label.setText(f"Mode: {mode}")

            lines = [
                f"Mode: {mode}",
                f"TensorFlow: {tf.__version__}",
                f"CUDA: {cuda_version}",
                f"cuDNN: {cudnn_version}",
                f"Detected GPUs: {len(gpus)}",
            ]

            for gpu_idx, gpu in enumerate(gpus):
                lines.append(f"GPU {gpu_idx}: {gpu.name}")
                try:
                    memory_info = tf.config.experimental.get_memory_info(f"GPU:{gpu_idx}")
                    current_mb = memory_info.get("current", 0) / (1024 * 1024)
                    peak_mb = memory_info.get("peak", 0) / (1024 * 1024)
                    lines.append(f"  VRAM current: {current_mb:.1f} MB, peak: {peak_mb:.1f} MB")
                except Exception:
                    lines.append("  VRAM usage unavailable in this TensorFlow build")

            QMessageBox.information(self, "Hardware Health", "\n".join(lines))
        except Exception as exc:
            self._show_logged_exception(
                title="Hardware Health",
                context="Unable to query TensorFlow hardware details",
                exc=exc,
                icon=QMessageBox.Warning,
            )

    def _update_hardware_mode_label(self):
        try:

            import tensorflow as tf
            gpus = tf.config.list_physical_devices("GPU")
            mode = "GPU" if len(gpus) > 0 else "CPU"
            build_info = tf.sysconfig.get_build_info()
            cuda_version = build_info.get("cuda_version", "unknown")
            cudnn_version = build_info.get("cudnn_version", "unknown")
            self.hardware_mode_label.setToolTip(
                f"Mode: {mode}\n"
                f"TensorFlow: {tf.__version__}\n"
                f"CUDA: {cuda_version}\n"
                f"cuDNN: {cudnn_version}"
            )
        except Exception:
            mode = "CPU"
            self.hardware_mode_label.setToolTip("Hardware info not available")
        self.hardware_mode_label.setText(f"Mode: {mode}")
        self._update_processing_estimate()

    def _refresh_all_views(self):
        self._update_session_status()
        self._refresh_ingestion_views()
        self._refresh_prediction_selector()
        self._refresh_curation_views()
        self._refresh_export_views()
        self._update_processing_estimate()
        self._update_export_size_estimate(self.output_format_combo.currentIndex())
        self._update_undo_redo_labels()

        if self.state.detected_indexing_direction:
            self.detected_direction_label.setText(
                f"Detected direction: {self.state.detected_indexing_direction}"
            )
            self.prediction_direction_label.setText(
                f"Detected indexing direction: {self.state.detected_indexing_direction}"
            )

        self._refresh_step_states()
        self._update_run_button_state()

    def _setup_tab_order(self):
        try:
            QWidget.setTabOrder(self.add_folder_button, self.add_files_button)
            QWidget.setTabOrder(self.add_files_button, self.clear_images_button)
            QWidget.setTabOrder(self.clear_images_button, self.thumbnail_filter_edit)
            QWidget.setTabOrder(self.thumbnail_filter_edit, self.thumbnail_list)

            QWidget.setTabOrder(self.mouse_radio, self.rat_radio)
            QWidget.setTabOrder(self.rat_radio, self.auto_thickness_checkbox)
            QWidget.setTabOrder(self.auto_thickness_checkbox, self.thickness_spin)
            QWidget.setTabOrder(self.thickness_spin, self.direction_override_combo)
            QWidget.setTabOrder(self.direction_override_combo, self.outlier_sigma_spin)
            QWidget.setTabOrder(self.outlier_sigma_spin, self.confidence_medium_spin)
            QWidget.setTabOrder(self.confidence_medium_spin, self.confidence_high_spin)
            QWidget.setTabOrder(self.confidence_high_spin, self.inference_batch_spin)
            QWidget.setTabOrder(self.inference_batch_spin, self.validate_configuration_button)

            QWidget.setTabOrder(self.run_alignment_button, self.cancel_alignment_button)
            QWidget.setTabOrder(self.cancel_alignment_button, self.console_toggle)
            QWidget.setTabOrder(self.console_toggle, self.console_output)

            QWidget.setTabOrder(self.slice_flag_list, self.apply_bad_sections_button)
            QWidget.setTabOrder(self.apply_bad_sections_button, self.undo_button)
            QWidget.setTabOrder(self.undo_button, self.redo_button)
            QWidget.setTabOrder(self.redo_button, self.anchor_depth_spin)
            QWidget.setTabOrder(self.anchor_depth_spin, self.set_anchor_button)
            QWidget.setTabOrder(self.set_anchor_button, self.remove_anchor_button)
            QWidget.setTabOrder(self.remove_anchor_button, self.apply_anchor_interpolation_button)
            QWidget.setTabOrder(self.apply_anchor_interpolation_button, self.anchor_list)

            QWidget.setTabOrder(self.output_dir_edit, self.output_basename_edit)
            QWidget.setTabOrder(self.output_basename_edit, self.output_format_combo)
            QWidget.setTabOrder(self.output_format_combo, self.export_button)
            QWidget.setTabOrder(self.export_button, self.report_button)
        except Exception:
            pass


def launch_gui():
    try:
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    except Exception:
        pass

    try:
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    except Exception:
        pass

    try:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except Exception:
        pass

    try:
        QApplication.setAttribute(Qt.AA_UseDesktopOpenGL, True)
    except Exception:
        pass

    app = QApplication.instance() or QApplication([])
    app_icon = _build_deepslice_icon()
    app.setWindowIcon(app_icon)

    splash, splash_status, splash_progress = _build_startup_splash(app_icon)
    splash.show()
    QApplication.processEvents()

    def _startup_progress(message: str, percent: int):
        splash_status.setText(str(message))
        splash_progress.setValue(int(np.clip(percent, 0, 100)))
        QApplication.processEvents()

    _startup_progress("Bootstrapping application", 4)
    window = DeepSliceMainWindow(startup_progress=_startup_progress, app_icon=app_icon)
    _startup_progress("Opening workspace", 99)
    window.show()
    splash.finish(window)
    return app.exec()
