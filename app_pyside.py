import ast
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
from PySide6 import QtCore, QtGui, QtWidgets

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mts"]
ORIGINAL_PREFIX = "original"
PROCESSED_NAME = "processed.mp4"
FIREFLY_JSON_NAME = "firefly.json"
PROJECT_SCRIPT = os.path.join(BASE_DIR, "project.py")

TEXT_MENU_BAR = "\u529f\u80fd\u5217"
TEXT_FILE = "\u6a94\u6848"
TEXT_SETTINGS = "\u8a2d\u5b9a"
TEXT_DISPLAY = "\u986f\u793a"
TEXT_REFRESH = "\u91cd\u65b0\u6574\u7406"
TEXT_OPEN = "\u958b\u555f"
TEXT_ADD_VIDEO = "\u65b0\u589e\u5f71\u7247"
TEXT_VIDEO_LIST = "\u5f71\u7247\u5217\u8868"
TEXT_FIREFLY_LIST = "\u87a2\u706b\u87f2\u5217\u8868"
TEXT_VIDEO_PROGRESS = "\u5f71\u7247\u9032\u5ea6\u689d"
TEXT_PLAY = "\u64ad\u653e"
TEXT_PAUSE = "\u66ab\u505c"
TEXT_SPEED = "\u8abf\u6574\u5f71\u7247\u64ad\u653e\u901f\u5ea6"
TEXT_BROWSE = "\u700f\u89bd"
TEXT_ANALYZE = "\u5206\u6790\u5f71\u7247"
TEXT_ANALYZE_CANCELED = "\u7d42\u6b62\u5206\u6790"
TEXT_EMPTY_FIREFLY = "\u5c1a\u7121\u87a2\u706b\u87f2\u8cc7\u6599\uff0c\u8acb\u5148\u5206\u6790\u5f71\u7247"
TEXT_VIDEO_DISPLAY = "\u5f71\u7247\u986f\u793a"
TEXT_DETAIL_TITLE = "\u87a2\u706b\u87f2\u8cc7\u6599"
TEXT_PATH_CHART = "\u8def\u5f91\u5716"
TEXT_DETAIL_INFO = "\u8a73\u7d30\u8cc7\u6599"
TEXT_DURATION = "\u6642\u9577"
TEXT_PROGRESS_TITLE = "\u986f\u793a\u5206\u6790\u9032\u5ea6"
TEXT_CANCEL = "\u53d6\u6d88"
TEXT_THEME = "\u4e3b\u984c"
TEXT_THEME_LIGHT = "\u6dfa\u8272"
TEXT_THEME_DARK = "\u6df1\u8272"
TEXT_FONT_SIZE = "\u5b57\u9ad4\u5927\u5c0f"
TEXT_ORIGINAL_VIDEO = "\u539f\u5f71\u7247"
TEXT_PROCESSED_VIDEO = "\u8655\u7406\u5f8c\u5f71\u7247"
TEXT_ORIGINAL_MISSING = "\u627e\u4e0d\u5230\u539f\u5f71\u7247"
TEXT_PROCESSED_MISSING = "\u627e\u4e0d\u5230\u8655\u7406\u5f8c\u5f71\u7247"
TEXT_ORIGINAL_ENDED = "\u539f\u5f71\u7247\u5df2\u7d50\u675f"
TEXT_PROCESSED_ENDED = "\u8655\u7406\u5f8c\u5f71\u7247\u5df2\u7d50\u675f"
WINDOW_TITLE = "\u87a2\u706b\u87f2\u8ffd\u8e64"


def is_video_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS


def sanitize_folder_name(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]', "_", name).strip()
    return cleaned or "video"


def _find_original_video(folder: str) -> Optional[str]:
    if not os.path.isdir(folder):
        return None
    names = [n for n in os.listdir(folder) if is_video_file(n)]
    for name in names:
        if name.lower().startswith(ORIGINAL_PREFIX):
            return os.path.join(folder, name)
    for name in names:
        lower = name.lower()
        if lower.startswith("processed") or lower.startswith("result_"):
            continue
        return os.path.join(folder, name)
    return None


def _find_processed_video(folder: str) -> Optional[str]:
    if not os.path.isdir(folder):
        return None
    for name in os.listdir(folder):
        if is_video_file(name) and name.lower() == PROCESSED_NAME:
            return os.path.join(folder, name)
    for name in os.listdir(folder):
        if is_video_file(name) and name.lower().startswith("processed"):
            return os.path.join(folder, name)
    for name in os.listdir(folder):
        if is_video_file(name) and name.lower().startswith("result_"):
            return os.path.join(folder, name)
    return None


def scan_video_entries(src_dir: str) -> List["VideoEntry"]:
    entries: List[VideoEntry] = []
    if not os.path.isdir(src_dir):
        return entries
    for name in sorted(os.listdir(src_dir)):
        folder = os.path.join(src_dir, name)
        if not os.path.isdir(folder):
            continue
        original = _find_original_video(folder)
        processed = _find_processed_video(folder)
        json_path = os.path.join(folder, FIREFLY_JSON_NAME)
        display_name = name
        entries.append(
            VideoEntry(
                name=display_name,
                folder=folder,
                original_path=original,
                processed_path=processed,
                json_path=json_path,
            )
        )
    return entries


def parse_project_output(lines: List[str]) -> Dict[str, List[dict]]:
    records: Dict[str, List[dict]] = {}
    current_video: Optional[str] = None
    pending: Optional[dict] = None

    start_marker = "\u958b\u59cb\u8655\u7406\u5f71\u7247"
    dur_marker = "\u6642\u9577"
    path_marker = "\u79fb\u52d5\u8ecc\u8de1"
    bright_marker = "\u6700\u4eae BGR"

    for raw in lines:
        line = raw.strip()
        if start_marker in line:
            current_video = line.split(":", 1)[-1].strip().strip("-").strip()
            records.setdefault(current_video, [])
            pending = None
            continue

        if "[Track Summary]" in line:
            match = re.search(r"ID:\s*(\d+)", line)
            if match:
                pending = {
                    "track_id": int(match.group(1)),
                    "duration_s": None,
                    "path": [],
                    "brightest_bgr": None,
                }
            continue

        if pending and line.startswith(dur_marker):
            value = line.split(":")[-1].strip().replace("s", "")
            try:
                pending["duration_s"] = float(value)
            except ValueError:
                pending["duration_s"] = None
            continue

        if pending and line.startswith(path_marker):
            payload = line.split(":", 1)[-1].strip()
            try:
                pending["path"] = ast.literal_eval(payload)
            except (ValueError, SyntaxError):
                pending["path"] = []
            continue

        if pending and line.startswith(bright_marker):
            payload = line.split(":", 1)[-1].strip()
            try:
                pending["brightest_bgr"] = tuple(ast.literal_eval(payload))
            except (ValueError, SyntaxError):
                pending["brightest_bgr"] = None
            if current_video is None:
                current_video = "unknown"
                records.setdefault(current_video, [])
            records[current_video].append(pending)
            pending = None

    return records


def load_firefly_json(path: Optional[str]) -> List[dict]:
    if not path or not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("records"), list):
        return data["records"]
    return []


@dataclass
class VideoEntry:
    name: str
    folder: str
    original_path: Optional[str]
    processed_path: Optional[str]
    json_path: Optional[str]


@dataclass
class FireflyRecord:
    track_id: int
    duration_s: Optional[float]
    path: List[Tuple[int, int]]
    brightest_bgr: Optional[Tuple[int, int, int]]


class PathCanvas(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._path: List[Tuple[int, int]] = []
        self._theme = "dark"

    def set_path(self, points: List[Tuple[int, int]]) -> None:
        self._path = points
        self.update()

    def set_theme(self, theme: str) -> None:
        self._theme = theme
        self.update()

    def _theme_colors(self) -> Dict[str, QtGui.QColor]:
        if self._theme == "light":
            return {
                "bg": QtGui.QColor("#ffffff"),
                "text": QtGui.QColor("#6b7280"),
                "line": QtGui.QColor("#10b981"),
                "start": QtGui.QColor("#f59e0b"),
                "end": QtGui.QColor("#ef4444"),
            }
        return {
            "bg": QtGui.QColor("#0f1117"),
            "text": QtGui.QColor("#9aa4b2"),
            "line": QtGui.QColor("#3ddc97"),
            "start": QtGui.QColor("#f9c74f"),
            "end": QtGui.QColor("#f94144"),
        }

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        rect = self.rect().adjusted(12, 12, -12, -12)
        colors = self._theme_colors()
        painter.fillRect(self.rect(), colors["bg"])

        if not self._path:
            painter.setPen(colors["text"])
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, TEXT_PATH_CHART)
            return

        xs = [p[0] for p in self._path]
        ys = [p[1] for p in self._path]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        dx = max_x - min_x or 1
        dy = max_y - min_y or 1

        points: List[QtCore.QPointF] = []
        for x, y in self._path:
            px = rect.left() + (x - min_x) / dx * rect.width()
            py = rect.bottom() - (y - min_y) / dy * rect.height()
            points.append(QtCore.QPointF(px, py))

        painter.setPen(QtGui.QPen(colors["line"], 2))
        painter.drawPolyline(QtGui.QPolygonF(points))

        painter.setBrush(colors["start"])
        painter.drawEllipse(points[0], 4, 4)
        painter.setBrush(colors["end"])
        painter.drawEllipse(points[-1], 4, 4)


class FireflyDetailDialog(QtWidgets.QDialog):
    def __init__(self, record: FireflyRecord, parent: QtWidgets.QWidget, theme: str) -> None:
        super().__init__(parent)
        self.setWindowTitle(TEXT_DETAIL_TITLE)
        self.resize(720, 420)

        title = QtWidgets.QLabel(f"ID: {record.track_id}")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")

        path_canvas = PathCanvas()
        path_canvas.setMinimumSize(360, 260)
        path_canvas.set_path(record.path)
        path_canvas.set_theme(theme)

        detail_box = QtWidgets.QGroupBox(TEXT_DETAIL_INFO)
        detail_layout = QtWidgets.QVBoxLayout(detail_box)

        duration = (
            f"{record.duration_s:.2f}s" if record.duration_s is not None else "-"
        )
        bgr_text = (
            str(record.brightest_bgr) if record.brightest_bgr is not None else "-"
        )

        detail_layout.addWidget(QtWidgets.QLabel(f"ID: {record.track_id}"))
        detail_layout.addWidget(QtWidgets.QLabel(f"{TEXT_DETAIL_INFO}:"))
        detail_layout.addWidget(QtWidgets.QLabel(f"{TEXT_DURATION}: {duration}"))
        detail_layout.addWidget(QtWidgets.QLabel(f"BGR: {bgr_text}"))
        detail_layout.addStretch(1)

        content_layout = QtWidgets.QHBoxLayout()
        content_layout.addWidget(path_canvas, 2)
        content_layout.addWidget(detail_box, 1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(title)
        layout.addLayout(content_layout)


class FireflyCardWidget(QtWidgets.QFrame):
    browse_clicked = QtCore.Signal(int)

    def __init__(
        self,
        record: FireflyRecord,
        theme: str = "dark",
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._record = record
        self._theme = theme
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self._apply_theme()

        title = QtWidgets.QLabel(f"ID {record.track_id}")
        title.setStyleSheet("font-weight: 600; font-size: 14px;")

        duration = (
            f"{record.duration_s:.2f}s" if record.duration_s is not None else "-"
        )
        info = QtWidgets.QLabel(f"{TEXT_DURATION}: {duration}")

        button = QtWidgets.QPushButton(TEXT_BROWSE)
        button.setCursor(QtCore.Qt.PointingHandCursor)
        button.clicked.connect(self._emit_browse)

        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(title)
        layout.addStretch(1)
        layout.addWidget(info)
        layout.addWidget(button)

    def _emit_browse(self) -> None:
        self.browse_clicked.emit(self._record.track_id)

    def set_theme(self, theme: str) -> None:
        self._theme = theme
        self._apply_theme()

    def _apply_theme(self) -> None:
        if self._theme == "light":
            self.setStyleSheet(
                "QFrame { background: #ffffff; border-radius: 10px; padding: 8px;"
                "border: 1px solid #e5e7eb; }"
                "QLabel { color: #111827; }"
            )
        else:
            self.setStyleSheet(
                "QFrame { background: #161b22; border-radius: 10px; padding: 8px; }"
                "QLabel { color: #d0d7de; }"
            )


class DisplaySettingsDialog(QtWidgets.QDialog):
    def __init__(self, theme: str, font_size: int, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(TEXT_DISPLAY)
        self.resize(360, 200)

        theme_label = QtWidgets.QLabel(TEXT_THEME)
        self.theme_combo = QtWidgets.QComboBox()
        self.theme_combo.addItem(TEXT_THEME_DARK, "dark")
        self.theme_combo.addItem(TEXT_THEME_LIGHT, "light")
        index = 0 if theme == "dark" else 1
        self.theme_combo.setCurrentIndex(index)

        font_label = QtWidgets.QLabel(TEXT_FONT_SIZE)
        self.font_spin = QtWidgets.QSpinBox()
        self.font_spin.setRange(8, 28)
        self.font_spin.setValue(font_size)

        form_layout = QtWidgets.QFormLayout()
        form_layout.addRow(theme_label, self.theme_combo)
        form_layout.addRow(font_label, self.font_spin)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form_layout)
        layout.addStretch(1)
        layout.addWidget(buttons)

    def selected_theme(self) -> str:
        return str(self.theme_combo.currentData())

    def selected_font_size(self) -> int:
        return int(self.font_spin.value())


class AnalysisProgressDialog(QtWidgets.QDialog):
    canceled = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(TEXT_PROGRESS_TITLE)
        self.setModal(True)
        self.resize(560, 320)

        title = QtWidgets.QLabel(TEXT_PROGRESS_TITLE)
        title.setStyleSheet("font-size: 16px; font-weight: 600;")

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)

        cancel_button = QtWidgets.QPushButton(TEXT_CANCEL)
        cancel_button.setCursor(QtCore.Qt.PointingHandCursor)
        cancel_button.clicked.connect(self._cancel)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(self.log_view, 1)
        layout.addWidget(cancel_button, 0, QtCore.Qt.AlignRight)

    def append_line(self, line: str) -> None:
        if line:
            self.log_view.appendPlainText(line)

    def _cancel(self) -> None:
        self.canceled.emit()
        self.close()


class AnalysisWorker(QtCore.QThread):
    log_line = QtCore.Signal(str)
    finished_ok = QtCore.Signal()
    failed = QtCore.Signal(str)
    canceled = QtCore.Signal()

    def __init__(
        self,
        input_path: str,
        output_path: str,
        json_path: str,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.input_path = input_path
        self.output_path = output_path
        self.json_path = json_path
        self._cancel_requested = False
        self._process: Optional[subprocess.Popen] = None

    def cancel(self) -> None:
        self._cancel_requested = True
        if self._process and self._process.poll() is None:
            try:
                self._process.terminate()
            except OSError:
                pass

    def run(self) -> None:
        if not os.path.isfile(PROJECT_SCRIPT):
            self.failed.emit("\u627e\u4e0d\u5230 project.py")
            return

        if not self.input_path or not os.path.isfile(self.input_path):
            self.failed.emit("\u627e\u4e0d\u5230\u539f\u59cb\u5f71\u7247")
            return

        cmd = [
            sys.executable,
            PROJECT_SCRIPT,
            "--input",
            self.input_path,
            "--output",
            self.output_path,
            "--json",
            self.json_path,
        ]
        try:
            self._process = subprocess.Popen(
                cmd,
                cwd=BASE_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError as exc:
            self.failed.emit(str(exc))
            return

        output_lines: List[str] = []
        if self._process.stdout:
            for line in self._process.stdout:
                output_lines.append(line)
                self.log_line.emit(line.rstrip("\n"))

        if self._process:
            self._process.wait()

        if self._cancel_requested:
            self.canceled.emit()
            return

        if self._process and self._process.returncode not in (0, None):
            self.failed.emit(f"\u5206\u6790\u5931\u6557 (code {self._process.returncode})")
            return

        if self.json_path and not os.path.isfile(self.json_path):
            parsed = parse_project_output(output_lines)
            video_name = os.path.basename(self.input_path)
            records = parsed.get(video_name) or next(iter(parsed.values()), [])
            if records:
                try:
                    folder = os.path.dirname(self.json_path)
                    if folder:
                        os.makedirs(folder, exist_ok=True)
                    with open(self.json_path, "w", encoding="utf-8") as handle:
                        json.dump(records, handle, ensure_ascii=False, indent=2)
                except OSError as exc:
                    self.failed.emit(str(exc))
                    return
            else:
                self.failed.emit("\u5206\u6790\u5b8c\u6210\u4f46\u6c92\u6709\u751f\u6210\u8cc7\u6599")
                return

        self.finished_ok.emit()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1200, 720)

        self.capture_original: Optional[cv2.VideoCapture] = None
        self.capture_processed: Optional[cv2.VideoCapture] = None
        self.frame_count = 0
        self.fps = 30.0
        self.playback_speed = 1.0
        self.current_entry: Optional[VideoEntry] = None
        self.current_records: List[FireflyRecord] = []
        self.video_entries: List[VideoEntry] = []
        self.pending_entry: Optional[VideoEntry] = None
        self.worker: Optional[AnalysisWorker] = None
        self.progress_dialog: Optional[AnalysisProgressDialog] = None
        self.theme = "dark"
        app = QtWidgets.QApplication.instance()
        self.font_size = app.font().pointSize() if app else 12

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._next_frame)

        self._build_ui()
        self._apply_font_size(self.font_size)
        self._apply_theme(self.theme)
        self._refresh_video_list()

    def _build_ui(self) -> None:
        self._build_menu()

        central = QtWidgets.QWidget()
        root_layout = QtWidgets.QHBoxLayout(central)
        root_layout.setContentsMargins(12, 12, 12, 12)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)

        self.original_video_label = QtWidgets.QLabel(TEXT_ORIGINAL_VIDEO)
        self.original_video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.original_video_label.setMinimumSize(320, 200)

        self.processed_video_label = QtWidgets.QLabel(TEXT_PROCESSED_VIDEO)
        self.processed_video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.processed_video_label.setMinimumSize(320, 200)

        original_box = QtWidgets.QGroupBox(TEXT_ORIGINAL_VIDEO)
        original_layout = QtWidgets.QVBoxLayout(original_box)
        original_layout.addWidget(self.original_video_label)

        processed_box = QtWidgets.QGroupBox(TEXT_PROCESSED_VIDEO)
        processed_layout = QtWidgets.QVBoxLayout(processed_box)
        processed_layout.addWidget(self.processed_video_label)

        self.video_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.video_splitter.addWidget(original_box)
        self.video_splitter.addWidget(processed_box)
        self.video_splitter.setStretchFactor(0, 1)
        self.video_splitter.setStretchFactor(1, 1)

        controls_box = QtWidgets.QGroupBox(TEXT_VIDEO_PROGRESS)
        controls_layout = QtWidgets.QHBoxLayout(controls_box)

        self.progress_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.progress_slider.setRange(0, 0)
        self.progress_slider.sliderReleased.connect(self._seek_frame)

        self.play_button = QtWidgets.QPushButton(TEXT_PLAY)
        self.pause_button = QtWidgets.QPushButton(TEXT_PAUSE)

        self.play_button.clicked.connect(self._play)
        self.pause_button.clicked.connect(self._pause)

        self.speed_combo = QtWidgets.QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "1x", "1.5x", "2x"])
        self.speed_combo.setCurrentText("1x")
        self.speed_combo.currentTextChanged.connect(self._set_speed)

        speed_label = QtWidgets.QLabel(TEXT_SPEED)

        controls_layout.addWidget(self.progress_slider, 3)
        controls_layout.addWidget(self.play_button)
        controls_layout.addWidget(self.pause_button)
        controls_layout.addWidget(speed_label)
        controls_layout.addWidget(self.speed_combo)

        left_layout.addWidget(self.video_splitter, 4)
        left_layout.addWidget(controls_box)

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)

        video_box = QtWidgets.QGroupBox(TEXT_VIDEO_LIST)
        video_layout = QtWidgets.QVBoxLayout(video_box)

        self.video_list = QtWidgets.QListWidget()
        self.video_list.itemSelectionChanged.connect(self._on_video_selected)

        video_actions = QtWidgets.QHBoxLayout()
        self.refresh_button = QtWidgets.QPushButton(TEXT_REFRESH)
        self.refresh_button.clicked.connect(self._refresh_video_list)

        video_actions.addWidget(self.refresh_button)

        video_layout.addWidget(self.video_list)
        video_layout.addLayout(video_actions)

        firefly_box = QtWidgets.QGroupBox(TEXT_FIREFLY_LIST)
        firefly_layout = QtWidgets.QVBoxLayout(firefly_box)

        self.firefly_scroll = QtWidgets.QScrollArea()
        self.firefly_scroll.setWidgetResizable(True)
        self.firefly_container = QtWidgets.QWidget()
        self.firefly_cards_layout = QtWidgets.QVBoxLayout(self.firefly_container)
        self.firefly_cards_layout.setSpacing(8)

        self.firefly_empty_label = QtWidgets.QLabel(TEXT_EMPTY_FIREFLY)
        self.firefly_empty_label.setStyleSheet("color: #9aa4b2;")
        self.firefly_cards_layout.addWidget(self.firefly_empty_label)
        self.firefly_cards_layout.addStretch(1)

        self.firefly_scroll.setWidget(self.firefly_container)
        firefly_layout.addWidget(self.firefly_scroll)

        right_layout.addWidget(video_box, 1)
        right_layout.addWidget(firefly_box, 2)

        root_layout.addWidget(left_panel, 3)
        root_layout.addWidget(right_panel, 2)

        self.setCentralWidget(central)
        self.statusBar().showMessage(TEXT_MENU_BAR)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu(TEXT_FILE)
        self.add_video_action = QtGui.QAction(TEXT_ADD_VIDEO, self)
        self.add_video_action.triggered.connect(self._add_video_files)
        refresh_action = QtGui.QAction(TEXT_REFRESH, self)
        refresh_action.triggered.connect(self._refresh_video_list)

        exit_action = QtGui.QAction("Exit", self)
        exit_action.triggered.connect(self.close)

        file_menu.addAction(self.add_video_action)
        file_menu.addAction(refresh_action)
        file_menu.addAction(exit_action)

        settings_menu = self.menuBar().addMenu(TEXT_SETTINGS)
        display_action = QtGui.QAction(TEXT_DISPLAY, self)
        display_action.triggered.connect(self._open_display_settings)
        settings_menu.addAction(display_action)

    def _open_display_settings(self) -> None:
        dialog = DisplaySettingsDialog(self.theme, self.font_size, self)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        self._apply_font_size(dialog.selected_font_size())
        self._apply_theme(dialog.selected_theme())

    def _apply_font_size(self, font_size: int) -> None:
        self.font_size = font_size
        app = QtWidgets.QApplication.instance()
        if not app:
            return
        font = app.font()
        font.setPointSize(font_size)
        app.setFont(font)

    def _apply_theme(self, theme: str) -> None:
        self.theme = theme if theme in ("dark", "light") else "dark"
        app = QtWidgets.QApplication.instance()
        if not app:
            return
        if self.theme == "light":
            app.setStyleSheet(
                "QWidget { background: #f5f7fb; color: #111827; }"
                "QGroupBox { border: 1px solid #d1d5db; margin-top: 8px; }"
                "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
                "QListWidget, QPlainTextEdit { background: #ffffff; border: 1px solid #d1d5db; }"
                "QPushButton { background: #ffffff; border: 1px solid #d1d5db; padding: 6px 10px; }"
                "QPushButton:hover { background: #f3f4f6; }"
                "QSlider::groove:horizontal { height: 6px; background: #e5e7eb; }"
                "QSlider::handle:horizontal { width: 14px; background: #9ca3af; margin: -4px 0; }"
            )
        else:
            app.setStyleSheet(
                "QWidget { background: #0d1117; color: #d0d7de; }"
                "QGroupBox { border: 1px solid #30363d; margin-top: 8px; }"
                "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
                "QListWidget, QPlainTextEdit { background: #0b0f14; border: 1px solid #30363d; }"
                "QPushButton { background: #161b22; border: 1px solid #30363d; padding: 6px 10px; }"
                "QPushButton:hover { background: #1f2937; }"
                "QSlider::groove:horizontal { height: 6px; background: #1f2937; }"
                "QSlider::handle:horizontal { width: 14px; background: #6b7280; margin: -4px 0; }"
            )

        self._update_video_label_styles()
        self._render_firefly_cards(self.current_records)

    def _update_video_label_styles(self) -> None:
        if self.theme == "light":
            background = "#f8fafc"
            text_color = "#64748b"
            border = "#d1d5db"
            empty_color = "#64748b"
        else:
            background = "#0d1117"
            text_color = "#9aa4b2"
            border = "#30363d"
            empty_color = "#9aa4b2"

        style = (
            f"background: {background}; color: {text_color};"
            f"border: 1px solid {border}; border-radius: 12px;"
        )
        self.original_video_label.setStyleSheet(style)
        self.processed_video_label.setStyleSheet(style)

        if hasattr(self, "firefly_empty_label") and self.firefly_empty_label:
            self.firefly_empty_label.setStyleSheet(f"color: {empty_color};")

    def _refresh_video_list(self, selected_folder: Optional[str] = None) -> None:
        self.video_list.clear()
        self.video_entries = scan_video_entries(SRC_DIR)
        for entry in self.video_entries:
            item = QtWidgets.QListWidgetItem(entry.name)
            item.setData(QtCore.Qt.UserRole, entry)
            self.video_list.addItem(item)

        if selected_folder:
            for idx in range(self.video_list.count()):
                item = self.video_list.item(idx)
                if item:
                    entry = item.data(QtCore.Qt.UserRole)
                    if entry and entry.folder == selected_folder:
                        self.video_list.setCurrentRow(idx)
                        return

        if self.video_list.count() > 0:
            self.video_list.setCurrentRow(0)

    def _add_video_files(self) -> None:
        os.makedirs(SRC_DIR, exist_ok=True)
        filter_text = "Video Files (*.mp4 *.MP4 *.avi *.AVI *.mov *.MOV *.mts *.MTS);;All Files (*.*)"
        source, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            TEXT_ADD_VIDEO,
            SRC_DIR,
            filter_text,
        )
        if not source:
            return
        if not is_video_file(source):
            QtWidgets.QMessageBox.warning(self, TEXT_ADD_VIDEO, "\u4e0d\u652f\u63f4\u7684\u5f71\u7247\u683c\u5f0f")
            return

        try:
            entry = self._prepare_video_entry(source)
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, TEXT_ADD_VIDEO, str(exc))
            return

        self._start_analysis_for_entry(entry)

    def _prepare_video_entry(self, source: str) -> VideoEntry:
        base_name = os.path.splitext(os.path.basename(source))[0]
        base_name = sanitize_folder_name(base_name)
        folder = self._unique_video_folder(base_name)
        os.makedirs(folder, exist_ok=True)

        ext = os.path.splitext(source)[1].lower()
        original_path = os.path.join(folder, f"{ORIGINAL_PREFIX}{ext}")
        shutil.copy2(source, original_path)

        processed_path = os.path.join(folder, PROCESSED_NAME)
        json_path = os.path.join(folder, FIREFLY_JSON_NAME)
        return VideoEntry(
            name=os.path.basename(folder),
            folder=folder,
            original_path=original_path,
            processed_path=processed_path,
            json_path=json_path,
        )

    def _unique_video_folder(self, base_name: str) -> str:
        candidate = os.path.join(SRC_DIR, base_name)
        if not os.path.exists(candidate):
            return candidate
        index = 1
        while True:
            candidate = os.path.join(SRC_DIR, f"{base_name}_{index}")
            if not os.path.exists(candidate):
                return candidate
            index += 1

    def _convert_records(self, records: List[dict]) -> List[FireflyRecord]:
        converted: List[FireflyRecord] = []
        for record in records:
            path = [tuple(p) for p in record.get("path", [])]
            brightest = record.get("brightest_bgr")
            converted.append(
                FireflyRecord(
                    track_id=int(record.get("track_id", 0)),
                    duration_s=record.get("duration_s"),
                    path=path,
                    brightest_bgr=tuple(brightest) if brightest else None,
                )
            )
        return converted

    def _on_video_selected(self) -> None:
        items = self.video_list.selectedItems()
        if not items:
            return

        entry = items[0].data(QtCore.Qt.UserRole)
        if not entry:
            return

        self.current_entry = entry
        self._open_video(entry)

        records = self._convert_records(load_firefly_json(entry.json_path))
        self.current_records = records
        self._render_firefly_cards(records)

    def _open_video(self, entry: VideoEntry) -> None:
        self._release_captures()
        self.capture_original = self._open_capture(entry.original_path)
        self.capture_processed = self._open_capture(entry.processed_path)

        if not self.capture_original:
            self._show_video_placeholder(self.original_video_label, TEXT_ORIGINAL_MISSING)
        if not self.capture_processed:
            self._show_video_placeholder(self.processed_video_label, TEXT_PROCESSED_MISSING)

        self.fps = self._resolve_fps()
        self.frame_count = self._resolve_frame_count()
        self.progress_slider.setRange(0, max(self.frame_count - 1, 0))

        self._pause()
        self._next_frame()

    def _release_captures(self) -> None:
        if self.capture_original:
            self.capture_original.release()
        if self.capture_processed:
            self.capture_processed.release()
        self.capture_original = None
        self.capture_processed = None

    def _open_capture(self, path: Optional[str]) -> Optional[cv2.VideoCapture]:
        if not path or not os.path.isfile(path):
            return None
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            capture.release()
            return None
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return capture

    def _resolve_fps(self) -> float:
        for capture in (self.capture_original, self.capture_processed):
            if capture and capture.isOpened():
                fps = capture.get(cv2.CAP_PROP_FPS)
                if fps:
                    return fps
        return 30.0

    def _resolve_frame_count(self) -> int:
        counts: List[int] = []
        for capture in (self.capture_original, self.capture_processed):
            if capture and capture.isOpened():
                count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                if count > 0:
                    counts.append(count)
        if not counts:
            return 0
        return min(counts) if len(counts) > 1 else counts[0]

    def _primary_capture(self) -> Optional[cv2.VideoCapture]:
        return self.capture_original or self.capture_processed

    def _read_capture_frame(self, capture: Optional[cv2.VideoCapture]) -> Optional[object]:
        if not capture:
            return None
        ok, frame = capture.read()
        if not ok:
            return None
        return frame

    def _render_frame(self, label: QtWidgets.QLabel, frame: object) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        image = QtGui.QImage(
            rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
        ).copy()
        pixmap = QtGui.QPixmap.fromImage(image)
        label.setPixmap(
            pixmap.scaled(
                label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        )
        label.setText("")

    def _show_video_placeholder(self, label: QtWidgets.QLabel, text: str) -> None:
        label.setPixmap(QtGui.QPixmap())
        label.setText(text)

    def _next_frame(self) -> None:
        base_capture = self._primary_capture()
        if not base_capture:
            return

        if base_capture == self.capture_original:
            original_frame = self._read_capture_frame(self.capture_original)
            processed_frame = self._read_capture_frame(self.capture_processed)
        else:
            processed_frame = self._read_capture_frame(self.capture_processed)
            original_frame = self._read_capture_frame(self.capture_original)

        if base_capture == self.capture_original and original_frame is None:
            self._pause()
            if self.capture_original:
                self._show_video_placeholder(self.original_video_label, TEXT_ORIGINAL_ENDED)
            return
        if base_capture == self.capture_processed and processed_frame is None:
            self._pause()
            if self.capture_processed:
                self._show_video_placeholder(self.processed_video_label, TEXT_PROCESSED_ENDED)
            return

        if original_frame is not None:
            self._render_frame(self.original_video_label, original_frame)
        elif self.capture_original:
            self._show_video_placeholder(self.original_video_label, TEXT_ORIGINAL_ENDED)

        if processed_frame is not None:
            self._render_frame(self.processed_video_label, processed_frame)
        elif self.capture_processed:
            self._show_video_placeholder(self.processed_video_label, TEXT_PROCESSED_ENDED)

        frame_index = int(base_capture.get(cv2.CAP_PROP_POS_FRAMES) or 0)
        self.progress_slider.blockSignals(True)
        self.progress_slider.setValue(frame_index)
        self.progress_slider.blockSignals(False)

    def _play(self) -> None:
        interval = max(int(1000 / (self.fps * self.playback_speed)), 1)
        self.timer.start(interval)

    def _pause(self) -> None:
        self.timer.stop()

    def _seek_frame(self) -> None:
        if not self.capture_original and not self.capture_processed:
            return
        frame_index = self.progress_slider.value()
        if self.capture_original:
            self.capture_original.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        if self.capture_processed:
            self.capture_processed.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        self._next_frame()

    def _set_speed(self, value: str) -> None:
        try:
            self.playback_speed = float(value.replace("x", ""))
        except ValueError:
            self.playback_speed = 1.0
        if self.timer.isActive():
            self._play()

    def _render_firefly_cards(self, records: List[FireflyRecord]) -> None:
        while self.firefly_cards_layout.count() > 0:
            item = self.firefly_cards_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        if not records:
            self.firefly_empty_label = QtWidgets.QLabel(TEXT_EMPTY_FIREFLY)
            color = "#64748b" if self.theme == "light" else "#9aa4b2"
            self.firefly_empty_label.setStyleSheet(f"color: {color};")
            self.firefly_cards_layout.addWidget(self.firefly_empty_label)
            self.firefly_cards_layout.addStretch(1)
            return

        for record in records:
            card = FireflyCardWidget(record, theme=self.theme)
            card.browse_clicked.connect(self._open_firefly_detail)
            self.firefly_cards_layout.addWidget(card)

        self.firefly_cards_layout.addStretch(1)

    def _open_firefly_detail(self, track_id: int) -> None:
        if not self.current_records:
            return
        for record in self.current_records:
            if record.track_id == track_id:
                dialog = FireflyDetailDialog(record, self, self.theme)
                dialog.exec()
                break

    def _set_analysis_controls_enabled(self, enabled: bool) -> None:
        if hasattr(self, "add_video_action"):
            self.add_video_action.setEnabled(enabled)
        self.refresh_button.setEnabled(enabled)

    def _start_analysis_for_entry(self, entry: VideoEntry) -> None:
        if self.worker and self.worker.isRunning():
            QtWidgets.QMessageBox.information(self, TEXT_ANALYZE, TEXT_ANALYZE)
            return

        if not entry.original_path or not entry.processed_path or not entry.json_path:
            QtWidgets.QMessageBox.warning(self, TEXT_ANALYZE, "\u5206\u6790\u8def\u5f91\u7f3a\u6f0f")
            return

        self.pending_entry = entry
        self.statusBar().showMessage(TEXT_ANALYZE)
        self._set_analysis_controls_enabled(False)

        self.progress_dialog = AnalysisProgressDialog(self)
        self.progress_dialog.canceled.connect(self._cancel_analysis)
        self.progress_dialog.show()

        self.worker = AnalysisWorker(
            entry.original_path,
            entry.processed_path,
            entry.json_path,
            self,
        )
        self.worker.log_line.connect(self._log_line)
        self.worker.log_line.connect(self.progress_dialog.append_line)
        self.worker.finished_ok.connect(self._analysis_done)
        self.worker.failed.connect(self._analysis_failed)
        self.worker.canceled.connect(self._analysis_canceled)
        self.worker.start()

    def _cancel_analysis(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.cancel()

    def _log_line(self, line: str) -> None:
        if line:
            self.statusBar().showMessage(line)

    def _analysis_done(self) -> None:
        self._close_progress_dialog()
        self._set_analysis_controls_enabled(True)
        if self.pending_entry:
            self._refresh_video_list(selected_folder=self.pending_entry.folder)
        self.pending_entry = None
        self.statusBar().showMessage(TEXT_ANALYZE)

    def _analysis_failed(self, message: str) -> None:
        self._close_progress_dialog()
        self._set_analysis_controls_enabled(True)
        self._cleanup_pending_entry()
        QtWidgets.QMessageBox.warning(self, TEXT_ANALYZE, message)

    def _analysis_canceled(self) -> None:
        self._close_progress_dialog()
        self._set_analysis_controls_enabled(True)
        self._cleanup_pending_entry()
        QtWidgets.QMessageBox.information(self, TEXT_ANALYZE, TEXT_ANALYZE_CANCELED)

    def _cleanup_pending_entry(self) -> None:
        if not self.pending_entry:
            return
        try:
            shutil.rmtree(self.pending_entry.folder)
        except OSError:
            pass
        self.pending_entry = None

    def _close_progress_dialog(self) -> None:
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._pause()
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
        self._release_captures()
        super().closeEvent(event)


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
