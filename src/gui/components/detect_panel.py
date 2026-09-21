"""Detect results side panel — list + score filter + actions."""
from __future__ import annotations

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QGroupBox,
)


class DetectPanel(QWidget):
    """Right-side panel for Detect mode results."""

    score_filter_changed = Signal(float)
    run_detect_requested = Signal()
    clear_requested = Signal()
    export_json_requested = Signal()
    export_csv_requested = Signal()
    export_overlay_requested = Signal()
    convert_to_annotations_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        title = QLabel("Detect 结果")
        title.setObjectName("detectPanelTitle")
        layout.addWidget(title)

        hint = QLabel("预览框不会写入工程标注。可导出或一键转入 Annotate。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #8A9A8A;")
        layout.addWidget(hint)

        filter_box = QGroupBox("置信度过滤")
        filter_layout = QVBoxLayout(filter_box)
        self._score_label = QLabel("阈值: 0.00")
        self._score_slider = QSlider(Qt.Horizontal)
        self._score_slider.setRange(0, 100)
        self._score_slider.setValue(0)
        self._score_slider.valueChanged.connect(self._on_slider)
        filter_layout.addWidget(self._score_label)
        filter_layout.addWidget(self._score_slider)
        layout.addWidget(filter_box)

        self._list = QListWidget()
        layout.addWidget(self._list, stretch=1)

        self._count_label = QLabel("0 个可见目标")
        layout.addWidget(self._count_label)

        run_btn = QPushButton("运行检测…")
        run_btn.clicked.connect(self.run_detect_requested.emit)
        layout.addWidget(run_btn)

        convert_btn = QPushButton("转入标注")
        convert_btn.setObjectName("detectConvertBtn")
        convert_btn.clicked.connect(self.convert_to_annotations_requested.emit)
        layout.addWidget(convert_btn)

        export_row = QHBoxLayout()
        json_btn = QPushButton("导出 JSON")
        json_btn.clicked.connect(self.export_json_requested.emit)
        csv_btn = QPushButton("导出 CSV")
        csv_btn.clicked.connect(self.export_csv_requested.emit)
        overlay_btn = QPushButton("导出叠加图")
        overlay_btn.clicked.connect(self.export_overlay_requested.emit)
        export_row.addWidget(json_btn)
        export_row.addWidget(csv_btn)
        export_row.addWidget(overlay_btn)
        layout.addLayout(export_row)

        clear_btn = QPushButton("清除预览")
        clear_btn.clicked.connect(self.clear_requested.emit)
        layout.addWidget(clear_btn)

    def _on_slider(self, value: int) -> None:
        threshold = value / 100.0
        self._score_label.setText(f"阈值: {threshold:.2f}")
        self.score_filter_changed.emit(threshold)

    def set_results(self, results: list) -> None:
        self._list.clear()
        for item in results:
            name = getattr(item, "class_name", "?")
            score = float(getattr(item, "score", 0.0))
            text = f"{name}  {score:.3f}"
            QListWidgetItem(text, self._list)
        self._count_label.setText(f"{len(results)} 个可见目标")

    def set_score_filter(self, threshold: float) -> None:
        value = int(round(max(0.0, min(1.0, threshold)) * 100))
        self._score_slider.blockSignals(True)
        self._score_slider.setValue(value)
        self._score_label.setText(f"阈值: {value / 100.0:.2f}")
        self._score_slider.blockSignals(False)
