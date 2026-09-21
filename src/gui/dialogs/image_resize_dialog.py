"""Image resize dialog — standalone preprocessing tool (PySide6)."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PIL import Image, ImageOps

from ..utils.image_preprocess_common import collect_images, existing_targets
from ..utils.image_resize import plan_resize_outputs, process_resize_one


class ResizeWorker(QThread):
    progress = Signal(int, str)  # 0-100, message
    finished_ok = Signal(int, int, bool)  # success, skipped, cancelled

    def __init__(
        self,
        images: list[Path],
        input_path: Path,
        target: tuple[int, int],
        mode: str,
        output_dir: Path | None,
        suffix: str,
        pad_color: tuple[int, int, int],
        quality: int,
        parent=None,
    ):
        super().__init__(parent)
        self._images = images
        self._input_path = input_path
        self._target = target
        self._mode = mode
        self._output_dir = output_dir
        self._suffix = suffix
        self._pad_color = pad_color
        self._quality = quality
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        success = 0
        skipped = 0
        total = len(self._images)
        for i, src in enumerate(self._images):
            if self._cancel:
                self.finished_ok.emit(success, skipped, True)
                return
            try:
                process_resize_one(
                    src,
                    self._input_path,
                    self._target,
                    self._mode,
                    self._output_dir,
                    self._suffix,
                    self._pad_color,
                    self._quality,
                )
                success += 1
                msg = f"[OK] {src.name} ({i + 1}/{total})"
            except Exception as e:
                skipped += 1
                msg = f"[FAIL] {src.name}: {e}"
            pct = int((i + 1) / total * 100) if total else 100
            self.progress.emit(pct, msg)

        self.finished_ok.emit(success, skipped, False)


def _spin(min_v: int, max_v: int, value: int, width: int = 80) -> QSpinBox:
    sp = QSpinBox()
    sp.setRange(min_v, max_v)
    sp.setValue(value)
    sp.setFixedWidth(width)
    sp.setAlignment(Qt.AlignRight)
    return sp


class ImageResizeDialog(QDialog):
    """Batch/single image resize tool."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("图片缩放")
        self.setMinimumWidth(520)
        self.resize(560, 640)
        self._worker: ResizeWorker | None = None
        self._busy = False
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # ---- Source ----
        src_group = QGroupBox("源文件")
        src_col = QVBoxLayout(src_group)
        src_col.setContentsMargins(8, 8, 8, 8)
        src_col.setSpacing(8)

        src_row = QHBoxLayout()
        self._src_edit = QLineEdit()
        self._src_edit.setPlaceholderText("选择图片文件或目录...")
        btn_file = QPushButton("选择文件")
        btn_file.setFixedWidth(88)
        btn_file.clicked.connect(self._browse_file)
        btn_dir = QPushButton("选择目录")
        btn_dir.setFixedWidth(88)
        btn_dir.clicked.connect(self._browse_dir)
        src_row.addWidget(self._src_edit, 1)
        src_row.addWidget(btn_file)
        src_row.addWidget(btn_dir)
        src_col.addLayout(src_row)

        self._recursive_cb = QCheckBox("包含子文件夹")
        self._recursive_cb.setChecked(False)
        src_col.addWidget(self._recursive_cb)
        layout.addWidget(src_group)

        # ---- Size ----
        size_group = QGroupBox("目标分辨率")
        size_col = QVBoxLayout(size_group)
        size_col.setContentsMargins(8, 8, 8, 8)

        size_row = QHBoxLayout()
        size_row.setSpacing(8)
        self._width_spin = _spin(1, 65535, 800)
        self._height_spin = _spin(1, 65535, 600)
        detect_btn = QPushButton("检测原图分辨率")
        detect_btn.clicked.connect(self._detect_resolution)
        size_row.addWidget(QLabel("宽"))
        size_row.addWidget(self._width_spin)
        size_row.addWidget(QLabel("x"))
        size_row.addWidget(QLabel("高"))
        size_row.addWidget(self._height_spin)
        size_row.addStretch(1)
        size_row.addWidget(detect_btn)
        size_col.addLayout(size_row)
        layout.addWidget(size_group)

        # ---- Mode ----
        mode_group = QGroupBox("缩放模式")
        mode_col = QVBoxLayout(mode_group)
        mode_col.setContentsMargins(8, 8, 8, 8)
        mode_col.setSpacing(4)
        self._mode_group = QButtonGroup(self)
        self._mode_fit = QRadioButton("fit - 等比缩放并填充")
        self._mode_fill = QRadioButton("fill - 等比缩放并裁剪")
        self._mode_exact = QRadioButton("exact - 直接拉伸")
        self._mode_fit.setChecked(True)
        for i, rb in enumerate((self._mode_fit, self._mode_fill, self._mode_exact)):
            self._mode_group.addButton(rb, i)
            mode_col.addWidget(rb)
        layout.addWidget(mode_group)

        # ---- Pad + quality on one row of groups ----
        opts_row = QHBoxLayout()
        opts_row.setSpacing(10)

        pad_group = QGroupBox("填充色 (fit)")
        pad_row = QHBoxLayout(pad_group)
        pad_row.setContentsMargins(8, 8, 8, 8)
        pad_row.setSpacing(6)
        self._r_spin = _spin(0, 255, 0, 64)
        self._g_spin = _spin(0, 255, 0, 64)
        self._b_spin = _spin(0, 255, 0, 64)
        for spin, name in ((self._r_spin, "R"), (self._g_spin, "G"), (self._b_spin, "B")):
            pad_row.addWidget(QLabel(name))
            pad_row.addWidget(spin)
        pad_row.addStretch(1)
        opts_row.addWidget(pad_group, 1)

        qual_group = QGroupBox("JPEG / WebP 质量")
        qual_row = QHBoxLayout(qual_group)
        qual_row.setContentsMargins(8, 8, 8, 8)
        self._quality_spin = _spin(1, 100, 90, 72)
        qual_row.addWidget(self._quality_spin)
        qual_row.addStretch(1)
        opts_row.addWidget(qual_group)
        layout.addLayout(opts_row)

        # ---- Output ----
        out_group = QGroupBox("输出设置")
        out_col = QVBoxLayout(out_group)
        out_col.setContentsMargins(8, 8, 8, 8)
        out_col.setSpacing(8)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("输出目录"))
        self._out_edit = QLineEdit()
        self._out_edit.setPlaceholderText("留空则在原图旁生成(加后缀)")
        out_btn = QPushButton("浏览")
        out_btn.setFixedWidth(72)
        out_btn.clicked.connect(self._browse_output)
        out_row.addWidget(self._out_edit, 1)
        out_row.addWidget(out_btn)
        out_col.addLayout(out_row)

        suffix_row = QHBoxLayout()
        suffix_row.addWidget(QLabel("文件后缀"))
        self._suffix_edit = QLineEdit("_resized")
        self._suffix_edit.setFixedWidth(140)
        suffix_row.addWidget(self._suffix_edit)
        suffix_row.addStretch(1)
        out_col.addLayout(suffix_row)
        layout.addWidget(out_group)

        # ---- Progress + log ----
        self._progress = QProgressBar()
        self._progress.setValue(0)
        self._progress.setFixedHeight(18)
        layout.addWidget(self._progress)

        self._status = QLabel("就绪")
        layout.addWidget(self._status)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(90)
        self._log.setMaximumHeight(160)
        self._log.setPlaceholderText("处理日志...")
        layout.addWidget(self._log)

        # ---- Buttons ----
        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("开始处理")
        self._run_btn.setMinimumWidth(100)
        self._run_btn.clicked.connect(self._start)
        self._cancel_btn = QPushButton("取消")
        self._cancel_btn.setMinimumWidth(80)
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._on_cancel_clicked)
        self._close_btn = QPushButton("关闭")
        self._close_btn.setMinimumWidth(80)
        self._close_btn.clicked.connect(self.reject)
        btn_row.addWidget(self._run_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self._cancel_btn)
        btn_row.addWidget(self._close_btn)
        layout.addLayout(btn_row)

        self._interactive_widgets: list[QWidget] = [
            self._src_edit,
            btn_file,
            btn_dir,
            self._recursive_cb,
            self._width_spin,
            self._height_spin,
            detect_btn,
            self._mode_fit,
            self._mode_fill,
            self._mode_exact,
            self._r_spin,
            self._g_spin,
            self._b_spin,
            self._quality_spin,
            self._out_edit,
            out_btn,
            self._suffix_edit,
            self._run_btn,
            self._close_btn,
        ]

    def _append_log(self, msg: str):
        self._log.append(msg)

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All (*.*)",
        )
        if path:
            self._src_edit.setText(path)

    def _browse_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择目录")
        if path:
            self._src_edit.setText(path)

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self._out_edit.setText(path)

    def _detect_resolution(self):
        src = self._src_edit.text().strip()
        if not src:
            QMessageBox.warning(self, "提示", "请先选择源文件")
            return
        p = Path(src)
        if not p.is_file():
            QMessageBox.warning(self, "提示", "请选择单个文件（不支持目录）")
            return
        try:
            with Image.open(p) as img:
                img = ImageOps.exif_transpose(img) or img
                w, h = img.size
            self._width_spin.setValue(w)
            self._height_spin.setValue(h)
            self._append_log(f"检测分辨率: {p.name} = {w}x{h}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法读取图像: {e}")

    def _current_mode(self) -> str:
        idx = self._mode_group.checkedId()
        return ("fit", "fill", "exact")[idx]

    def _set_busy(self, busy: bool):
        self._busy = busy
        for w in self._interactive_widgets:
            w.setEnabled(not busy)
        self._cancel_btn.setEnabled(busy)

    def _start(self):
        if self._busy:
            return

        src = self._src_edit.text().strip()
        if not src:
            QMessageBox.warning(self, "提示", "请选择源文件或目录")
            return
        src_path = Path(src)
        if not src_path.exists():
            QMessageBox.critical(self, "错误", f"路径不存在: {src}")
            return

        tw = self._width_spin.value()
        th = self._height_spin.value()
        images = collect_images(src_path, recursive=self._recursive_cb.isChecked())
        if not images:
            QMessageBox.warning(self, "提示", "未找到支持的图像文件")
            return

        out_str = self._out_edit.text().strip()
        output_dir = Path(out_str) if out_str else None
        suffix = self._suffix_edit.text().strip() or "_resized"

        planned = plan_resize_outputs(images, src_path, output_dir, suffix)
        conflicts = existing_targets(planned)
        if conflicts:
            reply = QMessageBox.question(
                self,
                "确认覆盖",
                f"将覆盖已有的 {len(conflicts)} 个文件，是否继续？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        mode = self._current_mode()
        pad = (self._r_spin.value(), self._g_spin.value(), self._b_spin.value())
        quality = self._quality_spin.value()

        self._set_busy(True)
        self._progress.setValue(0)
        self._status.setText(f"开始处理 {len(images)} 张 -> {tw}x{th} [{mode}]")
        self._append_log(self._status.text())

        self._worker = ResizeWorker(
            images,
            src_path,
            (tw, th),
            mode,
            output_dir,
            suffix,
            pad,
            quality,
            parent=self,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.start()

    def _on_progress(self, pct: int, msg: str):
        self._progress.setValue(pct)
        self._status.setText(msg)
        self._append_log(msg)

    def _on_finished(self, success: int, skipped: int, cancelled: bool):
        self._set_busy(False)
        if cancelled:
            summary = f"已取消: 成功 {success}，跳过 {skipped}"
        else:
            summary = f"完成: {success} 成功" + (f"，{skipped} 跳过" if skipped else "")
            self._progress.setValue(100)
        self._status.setText(summary)
        self._append_log(summary)
        self._worker = None

    def _on_cancel_clicked(self):
        if self._worker and self._busy:
            self._worker.cancel()
            self._status.setText("正在取消...")
            self._cancel_btn.setEnabled(False)

    def reject(self):
        if self._busy:
            return
        super().reject()

    def closeEvent(self, event):
        if self._busy:
            event.ignore()
            return
        super().closeEvent(event)
