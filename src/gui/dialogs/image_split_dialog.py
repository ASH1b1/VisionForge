"""Image tile-split dialog — standalone preprocessing tool (PySide6)."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PIL import Image, ImageOps

from ..utils.image_preprocess_common import collect_images, existing_targets
from ..utils.image_split import (
    OUTPUT_FORMATS,
    plan_split_outputs,
    split_one,
)


class SplitWorker(QThread):
    progress = Signal(int, str)  # 0-100, message
    finished_ok = Signal(int, int, bool)  # images_ok, tiles_written, cancelled

    def __init__(
        self,
        images: list[Path],
        input_path: Path,
        output_dir: Path,
        tile_w: int,
        tile_h: int,
        overlap: float,
        fmt: str,
        total_tiles: int,
        parent=None,
    ):
        super().__init__(parent)
        self._images = images
        self._input_path = input_path
        self._output_dir = output_dir
        self._tile_w = tile_w
        self._tile_h = tile_h
        self._overlap = overlap
        self._fmt = fmt
        self._total_tiles = max(1, total_tiles)
        self._cancel = False
        self._tiles_done = 0

    def cancel(self):
        self._cancel = True

    def run(self):
        images_ok = 0
        n_img = len(self._images)

        for i, src in enumerate(self._images):
            if self._cancel:
                self.finished_ok.emit(images_ok, self._tiles_done, True)
                return

            try:
                written_before = self._tiles_done

                def on_tile_progress(done_here: int, total_here: int):
                    self._tiles_done = written_before + done_here
                    pct = int(self._tiles_done / self._total_tiles * 100)
                    pct = min(99, pct)
                    self.progress.emit(
                        pct,
                        f"图 {i + 1}/{n_img} · {src.name} · 瓦片 {done_here}/{total_here}",
                    )

                written, cancelled = split_one(
                    src,
                    self._input_path,
                    self._output_dir,
                    self._tile_w,
                    self._tile_h,
                    self._overlap,
                    self._fmt,
                    should_cancel=lambda: self._cancel,
                    on_tile=on_tile_progress,
                )
                self._tiles_done = written_before + written
                if cancelled:
                    self.finished_ok.emit(images_ok, self._tiles_done, True)
                    return
                images_ok += 1
                self.progress.emit(
                    int(self._tiles_done / self._total_tiles * 100),
                    f"✓ {src.name}: {written} 瓦片 ({i + 1}/{n_img})",
                )
            except Exception as e:
                self.progress.emit(
                    int(self._tiles_done / self._total_tiles * 100),
                    f"✗ {src.name}: {e}",
                )

        self.progress.emit(100, f"完成 {images_ok}/{n_img} 张图，共 {self._tiles_done} 瓦片")
        self.finished_ok.emit(images_ok, self._tiles_done, False)


class ImageSplitDialog(QDialog):
    """Batch/single image tile-split tool."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("图片切分")
        self.setMinimumWidth(520)
        self.resize(560, 580)
        self._worker: SplitWorker | None = None
        self._busy = False
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Source
        src_group = QGroupBox("输入影像 / 文件夹")
        src_col = QVBoxLayout(src_group)
        src_col.setContentsMargins(8, 8, 8, 8)
        src_col.setSpacing(8)

        src_row = QHBoxLayout()
        self._src_edit = QLineEdit()
        self._src_edit.setPlaceholderText("选择影像或文件夹...")
        btn_file = QPushButton("选择影像")
        btn_file.setFixedWidth(88)
        btn_file.clicked.connect(self._browse_file)
        btn_dir = QPushButton("选择文件夹")
        btn_dir.setFixedWidth(96)
        btn_dir.clicked.connect(self._browse_dir)
        src_row.addWidget(self._src_edit, 1)
        src_row.addWidget(btn_file)
        src_row.addWidget(btn_dir)
        src_col.addLayout(src_row)

        self._recursive_cb = QCheckBox("包含子文件夹")
        self._recursive_cb.setChecked(False)
        src_col.addWidget(self._recursive_cb)
        layout.addWidget(src_group)

        # Output (required)
        out_group = QGroupBox("输出目录 (必填)")
        out_row = QHBoxLayout(out_group)
        out_row.setContentsMargins(8, 8, 8, 8)
        self._out_edit = QLineEdit()
        out_btn = QPushButton("浏览")
        out_btn.setFixedWidth(72)
        out_btn.clicked.connect(self._browse_output)
        out_row.addWidget(self._out_edit, 1)
        out_row.addWidget(out_btn)
        layout.addWidget(out_group)

        # Tile params
        tile_group = QGroupBox("切分参数")
        tile_col = QVBoxLayout(tile_group)
        tile_col.setContentsMargins(8, 8, 8, 8)
        tile_col.setSpacing(8)

        wh_row = QHBoxLayout()
        wh_row.setSpacing(8)
        self._tile_w = QSpinBox()
        self._tile_w.setRange(64, 32768)
        self._tile_w.setValue(1024)
        self._tile_w.setFixedWidth(90)
        self._tile_h = QSpinBox()
        self._tile_h.setRange(64, 32768)
        self._tile_h.setValue(1024)
        self._tile_h.setFixedWidth(90)
        wh_row.addWidget(QLabel("瓦片宽"))
        wh_row.addWidget(self._tile_w)
        wh_row.addWidget(QLabel("高"))
        wh_row.addWidget(self._tile_h)
        wh_row.addStretch(1)
        tile_col.addLayout(wh_row)

        param_row = QHBoxLayout()
        param_row.setSpacing(8)
        self._overlap = QDoubleSpinBox()
        self._overlap.setRange(0.0, 0.9)
        self._overlap.setSingleStep(0.05)
        self._overlap.setValue(0.20)
        self._overlap.setDecimals(2)
        self._overlap.setFixedWidth(90)
        self._format = QComboBox()
        self._format.addItems(list(OUTPUT_FORMATS))
        self._format.setCurrentText("PNG")
        self._format.setFixedWidth(90)
        param_row.addWidget(QLabel("重叠率"))
        param_row.addWidget(self._overlap)
        param_row.addWidget(QLabel("输出格式"))
        param_row.addWidget(self._format)
        param_row.addStretch(1)
        tile_col.addLayout(param_row)
        layout.addWidget(tile_group)

        # Detect info
        info_row = QHBoxLayout()
        detect_btn = QPushButton("识别分辨率")
        detect_btn.setFixedWidth(100)
        detect_btn.clicked.connect(self._detect_info)
        self._info_label = QLabel("请选择影像或文件夹")
        info_row.addWidget(detect_btn)
        info_row.addWidget(self._info_label, 1)
        layout.addLayout(info_row)

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

        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("开始切分")
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
            self._out_edit,
            out_btn,
            self._tile_w,
            self._tile_h,
            self._overlap,
            self._format,
            detect_btn,
            self._run_btn,
            self._close_btn,
        ]

    def _append_log(self, msg: str):
        self._log.append(msg)

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择影像",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All (*.*)",
        )
        if path:
            self._src_edit.setText(path)
            self._detect_info()

    def _browse_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if path:
            self._src_edit.setText(path)
            self._detect_info()

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self._out_edit.setText(path)

    def _detect_info(self):
        src = self._src_edit.text().strip()
        if not src:
            self._info_label.setText("未选择输入")
            return
        p = Path(src)
        images = collect_images(p, recursive=self._recursive_cb.isChecked())
        if not images:
            self._info_label.setText("未发现支持的影像")
            return
        try:
            with Image.open(images[0]) as im:
                im = ImageOps.exif_transpose(im) or im
                w, h = im.size
                dpi = im.info.get("dpi")
            prefix = f"{len(images)} 张 · 首张 " if len(images) > 1 else ""
            if dpi:
                self._info_label.setText(f"{prefix}{w}×{h}px, DPI {dpi}")
            else:
                self._info_label.setText(f"{prefix}{w}×{h}px")
        except Exception as e:
            self._info_label.setText(f"识别失败: {e}")

    def _set_busy(self, busy: bool):
        self._busy = busy
        for w in self._interactive_widgets:
            w.setEnabled(not busy)
        self._cancel_btn.setEnabled(busy)

    def _start(self):
        if self._busy:
            return

        src = self._src_edit.text().strip()
        out = self._out_edit.text().strip()
        if not src or not out:
            QMessageBox.warning(self, "提示", "请设置输入与输出目录")
            return

        src_path = Path(src)
        out_dir = Path(out)
        if not src_path.exists():
            QMessageBox.critical(self, "错误", f"路径不存在: {src}")
            return

        images = collect_images(src_path, recursive=self._recursive_cb.isChecked())
        if not images:
            QMessageBox.warning(self, "提示", "未找到支持的图像文件")
            return

        tw = self._tile_w.value()
        th = self._tile_h.value()
        ov = float(self._overlap.value())
        fmt = self._format.currentText()

        try:
            planned, total_tiles = plan_split_outputs(
                images, src_path, out_dir, tw, th, ov, fmt
            )
        except Exception as e:
            QMessageBox.critical(self, "错误", f"规划输出失败: {e}")
            return

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

        out_dir.mkdir(parents=True, exist_ok=True)
        self._set_busy(True)
        self._progress.setValue(0)
        msg = f"开始切分 {len(images)} 张 · 预计 {total_tiles} 瓦片 · {tw}×{th} · 重叠 {ov:.0%}"
        self._status.setText(msg)
        self._append_log(msg)

        self._worker = SplitWorker(
            images,
            src_path,
            out_dir,
            tw,
            th,
            ov,
            fmt,
            total_tiles,
            parent=self,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.start()

    def _on_progress(self, pct: int, msg: str):
        self._progress.setValue(pct)
        self._status.setText(msg)
        if msg.startswith("✓") or msg.startswith("✗") or msg.startswith("完成"):
            self._append_log(msg)

    def _on_finished(self, images_ok: int, tiles: int, cancelled: bool):
        self._set_busy(False)
        if cancelled:
            summary = f"已取消: 完成 {images_ok} 张图，写出 {tiles} 瓦片"
        else:
            summary = f"完成: {images_ok} 张图，共 {tiles} 瓦片"
            self._progress.setValue(100)
        self._status.setText(summary)
        self._append_log(summary)
        self._worker = None

    def _on_cancel_clicked(self):
        if self._worker and self._busy:
            self._worker.cancel()
            self._status.setText("正在取消…")
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
