# src/gui/main_window.py
from PySide6.QtWidgets import (
    QMainWindow, QMenuBar, QToolBar, QStatusBar,
    QDockWidget, QWidget, QHBoxLayout, QApplication, QFileDialog,
    QMessageBox, QDialog, QTreeWidgetItem, QSlider, QLabel, QPushButton, QMenu,
    QButtonGroup,
)
from PySide6.QtCore import Qt, Signal, QThreadPool
from PySide6.QtGui import QKeySequence, QAction, QIcon, QImage
from pathlib import Path
import cv2
import numpy as np
import json
from datetime import datetime
import os
import sys as _sys
from typing import Optional

def _get_project_root() -> Path:
    """获取项目根目录，兼容 PyInstaller 打包（sys._MEIPASS）和源码运行（__file__）。
    main_window.py 位于 源码版/src/gui/main_window.py，项目根目录（源码版/）需往上 2 级。
    """
    if getattr(_sys, "frozen", False):
        return Path(_sys._MEIPASS)
    return Path(__file__).resolve().parents[2]

from src.product_config import get_product_config

from .components.explorer_panel import ExplorerPanel
from .components.model_load_progress import make_model_load_progress_bar
from .components.inspector_panel import InspectorPanel
from .components.canvas_widget import CanvasWidget, Tool
from .components.detect_panel import DetectPanel
from .managers.shortcut_manager import ShortcutManager
from .models.app_state import AppState, AppStateManager
from .models.project_document import Annotation
from .models.project_document import ProjectDocument
from .models.detection_state import AppMode
from .controllers.file_controller import FileController
from .controllers.annotation_controller import AnnotationController
from .controllers.model_controller import ModelController
from .controllers.detect_controller import DetectController
from .controllers.sam_click_controller import SamClickController
from .controllers.export_controller import ExportController
from .controllers.playback_controller import PlaybackController
from .dialogs.process_video_dialog import ProcessVideoDialog
from .dialogs.auto_annotate_dialog import AutoAnnotateDialog
from .dialogs.detect_dialog import DetectDialog
from .dialogs.replace_class_dialog import ReplaceClassDialog
from .dialogs.image_resize_dialog import ImageResizeDialog
from .dialogs.image_split_dialog import ImageSplitDialog
from .dialogs.refine_segmentation_dialog import (
    RefineSegmentationDialog,
    RefineSegmentationWorker,
    _prompt_from_class_names,
)
from .utils.class_utils import coerce_class_id, display_class_name, normalize_class_name
from .utils.replace_frame_warning import (
    POLYGON_REPLACE_NOTE,
    count_frames_with_polygons,
    frames_have_polygons,
    replace_warning_message,
)
from .utils.exporters.mask_exporter import MaskExporter
from .utils.importers import YOLOImporter, VOCImporter, COCOImporter

class MainWindow(QMainWindow):
    """主应用窗口 - IDE 风格三栏布局"""

    # Signals
    file_loaded = Signal(object)  # Path
    frame_changed = Signal(int)   # frame index
    annotations_changed = Signal(list)  # annotations

    # Class-level shared ModelController — all windows reuse loaded models
    _shared_model_ctrl: Optional[ModelController] = None

    def __init__(self):
        super().__init__()
        self._product = get_product_config()
        self._load_stylesheet()
        self.setWindowTitle(self._product.window_title)

        # 设置窗口图标
        logo_path = _get_project_root() / self._product.logo_filename
        if logo_path.exists():
            self.setWindowIcon(QIcon(str(logo_path)))
        self.resize(1400, 900)
        self._setup_ui()
        self._setup_panels()
        self._setup_toolbar()
        self._setup_shortcuts()
        self._setup_menu_bar()
        self._setup_status_bar()

        # Initialize state managers
        self.state_manager = AppStateManager()
        # NEW: Unified project model (Phase 1 side-by-side)
        self.project = ProjectDocument()
        self.file_ctrl = FileController(self.project)
        self.ann_ctrl = AnnotationController(self.project, self.state_manager)
        # Use shared ModelController — all windows share the same loaded models
        if MainWindow._shared_model_ctrl is None:
            MainWindow._shared_model_ctrl = ModelController()
        self.model_ctrl = MainWindow._shared_model_ctrl
        self.export_ctrl = ExportController(self.project)
        self.playback_ctrl = PlaybackController(self.project, self.state_manager)
        self.detect_ctrl = DetectController()
        self.sam_click_ctrl = SamClickController(
            self.project,
            self.state_manager,
            self.ann_ctrl,
            self.model_ctrl,
            self.canvas,
            parent=self,
            parent_widget=self,
            selected_class_provider=lambda: self.state_manager.state.selected_class_id,
        )
        self._device = "CPU"  # Default device display
        self._auto_annotation_logs = []  # List of auto-annotation sessions
        self._previous_frame = 0  # Track previous frame for save-on-switch
        self._current_image_path = None

        # Models (managed by ModelController)
        self._init_grounding_model()
        self._init_sam_model()

        # Connect signals
        self._connect_signals()

        # Initialize default classes
        self._init_classes()
        self._on_tool_changed(self.state_manager._state.selected_tool)
        self._apply_app_mode(AppMode.ANNOTATE)

    def _init_classes(self):
        """初始化默认类别"""
        default_classes = {
            0: "person",
            1: "car",
            2: "bicycle",
            3: "dog",
            4: "cat"
        }
        self.project.class_names = default_classes.copy()
        self.explorer.set_classes(default_classes)
        self.inspector.set_class_names(default_classes)
        self.explorer.set_selected_class(0)

    def _connect_signals(self):
        """连接信号槽"""
        # Canvas signals
        self.canvas.annotation_created.connect(self._on_annotation_created)
        self.canvas.annotation_modified.connect(self._on_annotation_modified)
        self.canvas.annotation_deleted.connect(self._on_annotation_deleted)
        self.canvas.annotation_selected.connect(self._on_annotation_selected)
        self.canvas.polygon_created.connect(self._on_polygon_created)
        self.canvas.polygon_edited.connect(self.ann_ctrl.edit_polygon)
        self.canvas.sam_click_at.connect(self.sam_click_ctrl.handle_click)
        self.canvas.sam_confirm_requested.connect(self.sam_click_ctrl.commit)
        self.canvas.annotation_selected.connect(self.sam_click_ctrl.note_annotation_selected)

        # State manager signals
        self.state_manager.frame_changed.connect(self._on_frame_changed)

        # Explorer panel signals
        self.explorer.tool_selected.connect(self._on_tool_selected)
        self.explorer.class_selected.connect(self._on_class_selected)
        self.explorer.class_added.connect(self._on_class_added)
        self.explorer.class_renamed.connect(self._on_class_renamed)
        self.explorer.class_deleted.connect(self._on_class_deleted)

        self.state_manager.tool_changed.connect(self._on_tool_changed)
        self.state_manager.tool_changed.connect(self.sam_click_ctrl.on_tool_changed)
        self.state_manager.frame_changed.connect(self.sam_click_ctrl.on_frame_changed)
        self.state_manager.file_changed.connect(
            lambda _path: self.sam_click_ctrl.discard_session()
        )
        self.detect_ctrl.state.mode_changed.connect(self.sam_click_ctrl.on_mode_changed)
        self.project.data_cleared.connect(self.sam_click_ctrl.discard_session)
        self._sam_commit_action.triggered.connect(self.sam_click_ctrl.commit)
        self.sam_click_ctrl.commit_available.connect(self._sam_commit_action.setEnabled)
        self.sam_click_ctrl.status_message.connect(
            lambda msg: self.statusBar().showMessage(msg, 0)
        )

        # Connect canvas callbacks
        self.canvas.set_callbacks(
            get_annotations=lambda: self.project.visible_annotations,
            get_class_names=lambda: self.project.class_names
        )

        # --- ProjectDocument signals (Phase 1) ---
        self.project.dirty_changed.connect(self._on_project_dirty_changed)
        self.project.current_annotations_changed.connect(self._on_annotations_changed)
        self.project.history_changed.connect(self._on_history_changed)
        self.project.selection_changed.connect(self.canvas.set_selected_ids)

        # Detect mode
        self.detect_ctrl.state.detections_changed.connect(self._refresh_detect_overlays)
        self.detect_ctrl.state.score_filter_changed.connect(self._on_detect_score_filter)
        self.detect_ctrl.state.mode_changed.connect(self._on_app_mode_changed)

    def _init_grounding_model(self):
        """初始化 GroundingDINO 模型（仅首次加载，后续窗口复用）"""
        # Only the first window triggers model loading
        model = self.model_ctrl.get_grounding_model()
        if model is not None and model.is_loaded():
            self._on_model_loaded(True)
        elif model is None:
            # First window: start loading
            self._show_model_load_progress()
            self.model_ctrl.load_grounding_dino(str(_get_project_root()))
            model = self.model_ctrl.get_grounding_model()
            if model and model.is_loaded():
                self._on_model_loaded(True)
            elif model:
                model.loading_progress.connect(self._on_model_loading)
                model.loading_finished.connect(self._on_model_loaded)
                self._show_model_load_progress()
        else:
            model.loading_progress.connect(self._on_model_loading)
            model.loading_finished.connect(self._on_model_loaded)
            self._show_model_load_progress()

    def _init_sam_model(self):
        """SAM3 延迟加载（由用户手动触发）"""
        pass

    def _on_model_loading(self, message: str):
        """模型加载进度更新"""
        self._show_model_load_progress()
        if "使用设备:" in message:
            device = message.split("使用设备:")[-1].strip()
            self._device = device.upper()
            # Update device label with color
            if device == "cuda":
                self._device_label.setText("设备: GPU")
                self._device_label.setStyleSheet("color: #4CAF50; padding: 2px 8px; border: 1px solid #444; border-radius: 3px;")
            else:
                self._device_label.setText("设备: CPU")
                self._device_label.setStyleSheet("color: #FF9800; padding: 2px 8px; border: 1px solid #444; border-radius: 3px;")

        self.statusBar().showMessage(f"正在加载模型: {message}")

    def _on_model_loaded(self, success: bool):
        """模型加载完成"""
        self._hide_model_load_progress()
        if success:
            self.statusBar().showMessage("模型加载成功", 3000)
        else:
            self.statusBar().showMessage("模型加载失败", 5000)

    def _load_stylesheet(self):
        """加载产品对应的样式表（Delivery=vscode_dark，Product=product_shell）。"""
        style_path = _get_project_root() / self._product.stylesheet_relpath
        if not style_path.exists():
            # Fallback for incomplete bundles
            style_path = _get_project_root() / "src" / "gui" / "styles" / "vscode_dark.qss"
        if style_path.exists():
            with open(style_path, "r", encoding="utf-8") as f:
                self.setStyleSheet(f.read())

    def _setup_ui(self):
        """设置主 UI 布局"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = CanvasWidget()
        layout.addWidget(self.canvas)

    def _setup_panels(self):
        """设置左右面板"""
        # 左侧 Explorer 面板
        self.explorer = ExplorerPanel()
        self.explorer_dock = QDockWidget("资源管理器", self)
        self.explorer_dock.setWidget(self.explorer)
        self.explorer_dock.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetClosable |
            QDockWidget.DockWidgetFloatable
        )
        self.addDockWidget(Qt.LeftDockWidgetArea, self.explorer_dock)

        # 右侧 Inspector 面板
        self.inspector = InspectorPanel()
        self.inspector_dock = QDockWidget("检查器", self)
        self.inspector_dock.setWidget(self.inspector)
        self.inspector_dock.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetClosable |
            QDockWidget.DockWidgetFloatable
        )
        self.addDockWidget(Qt.RightDockWidgetArea, self.inspector_dock)

        # Connect inspector signals
        self.inspector.annotation_selected.connect(self._on_inspector_selection)
        self.inspector.annotation_deleted.connect(self._on_annotation_deleted)
        self.inspector.class_changed.connect(self._on_class_changed)
        self.inspector.select_all_requested.connect(self._on_select_all_annotations)
        self.inspector.invert_selection_requested.connect(self._on_invert_selection)
        self.inspector.batch_delete_requested.connect(self._on_batch_delete)
        self.inspector.multi_selection_changed.connect(self._on_multi_selection_changed)

        # Detect results panel (Product edition)
        self.detect_panel = DetectPanel()
        self.detect_dock = QDockWidget("Detect", self)
        self.detect_dock.setWidget(self.detect_panel)
        self.detect_dock.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetClosable |
            QDockWidget.DockWidgetFloatable
        )
        self.addDockWidget(Qt.RightDockWidgetArea, self.detect_dock)
        self.tabifyDockWidget(self.inspector_dock, self.detect_dock)
        self.inspector_dock.raise_()
        self.detect_dock.setVisible(False)

        self.detect_panel.run_detect_requested.connect(self._run_detect)
        self.detect_panel.clear_requested.connect(self._clear_detect_preview)
        self.detect_panel.export_json_requested.connect(self._export_detect_json)
        self.detect_panel.export_csv_requested.connect(self._export_detect_csv)
        self.detect_panel.export_overlay_requested.connect(self._export_detect_overlay)
        self.detect_panel.convert_to_annotations_requested.connect(self._convert_detect_to_annotations)
        self.detect_panel.score_filter_changed.connect(
            lambda v: self.detect_ctrl.state.set_score_filter(v)
        )

    def _setup_toolbar(self):
        """设置视频导航工具栏"""
        toolbar = QToolBar("播放控制", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        self._tool_actions = {}

        # Frame info label
        self._frame_label = QLabel("帧: - / -")
        self._frame_label.setMinimumWidth(100)
        toolbar.addWidget(self._frame_label)

        toolbar.addSeparator()

        # Navigation buttons
        first_btn = QPushButton("|<<")
        first_btn.setMinimumWidth(50)
        first_btn.setToolTip("首帧 (Home)")
        first_btn.clicked.connect(self._first_frame)
        toolbar.addWidget(first_btn)

        prev_btn = QPushButton("<")
        prev_btn.setMinimumWidth(40)
        prev_btn.setToolTip("上一帧 (←)")
        prev_btn.clicked.connect(self._prev_frame)
        toolbar.addWidget(prev_btn)

        # Play/Pause button
        self._play_btn = QPushButton("▶ 播放")
        self._play_btn.setCheckable(True)
        self._play_btn.setMinimumWidth(60)
        self._play_btn.setToolTip("播放/暂停 (Space)")
        self._play_btn.clicked.connect(self._toggle_playback)
        toolbar.addWidget(self._play_btn)

        # Next frame button
        next_btn = QPushButton(">")
        next_btn.setMinimumWidth(40)
        next_btn.setToolTip("下一帧 (→)")
        next_btn.clicked.connect(self._next_frame)
        toolbar.addWidget(next_btn)

        # Last frame button
        last_btn = QPushButton(">>|")
        last_btn.setMinimumWidth(50)
        last_btn.setToolTip("末帧 (End)")
        last_btn.clicked.connect(self._last_frame)
        toolbar.addWidget(last_btn)

        toolbar.addSeparator()

        # Frame slider
        self._frame_slider = QSlider(Qt.Horizontal)
        self._frame_slider.setMinimum(0)
        self._frame_slider.setMaximum(0)
        self._frame_slider.valueChanged.connect(self._on_slider_changed)
        toolbar.addWidget(self._frame_slider)

        toolbar.addSeparator()

        # Product: Annotate | Detect mode switcher
        self._mode_annotate_btn = None
        self._mode_detect_btn = None
        if self._product.feature_mode_switcher:
            self._mode_annotate_btn = QPushButton("Annotate")
            self._mode_annotate_btn.setObjectName("modeAnnotateBtn")
            self._mode_annotate_btn.setCheckable(True)
            self._mode_annotate_btn.setChecked(True)
            self._mode_annotate_btn.setToolTip("标注模式：编辑工程标注")
            self._mode_annotate_btn.clicked.connect(lambda: self._set_app_mode(AppMode.ANNOTATE))
            toolbar.addWidget(self._mode_annotate_btn)

            self._mode_detect_btn = QPushButton("Detect")
            self._mode_detect_btn.setObjectName("modeDetectBtn")
            self._mode_detect_btn.setCheckable(True)
            self._mode_detect_btn.setToolTip("检测模式：非破坏性预览与导出")
            self._mode_detect_btn.clicked.connect(lambda: self._set_app_mode(AppMode.DETECT))
            toolbar.addWidget(self._mode_detect_btn)

            self._mode_group = QButtonGroup(self)
            self._mode_group.setExclusive(True)
            self._mode_group.addButton(self._mode_annotate_btn)
            self._mode_group.addButton(self._mode_detect_btn)

        toolbar.addSeparator()

        # Tool buttons
        tool_specs = [
            ("选择", Tool.SELECT, "V"),
            ("矩形", Tool.RECTANGLE, "R"),
            ("多边形", Tool.POLYGON, "P"),
            ("橡皮擦", Tool.ERASER, "E"),
            ("SAM 点选", Tool.SAM_CLICK, "A"),
        ]
        for label, tool, shortcut in tool_specs:
            action = QAction(label, self)
            action.setCheckable(True)
            action.setToolTip(f"{label}工具 ({shortcut})")
            action.triggered.connect(lambda checked=False, t=tool: self.state_manager.set_tool(t))
            toolbar.addAction(action)
            self._tool_actions[tool] = action

        self._sam_commit_action = QAction("确认分割", self)
        self._sam_commit_action.setToolTip("将当前 SAM 预览写入工程 (Enter)")
        self._sam_commit_action.setEnabled(False)
        toolbar.addAction(self._sam_commit_action)

        toolbar.addSeparator()

        # Zoom info
        self._zoom_label = QLabel("100%")
        self._zoom_label.setMinimumWidth(50)
        toolbar.addWidget(self._zoom_label)

        # Zoom buttons
        zoom_out_btn = QPushButton("- 缩小")
        zoom_out_btn.setMinimumWidth(60)
        zoom_out_btn.setToolTip("缩小")
        zoom_out_btn.clicked.connect(lambda: self._set_zoom(self.canvas._zoom * 0.9))
        toolbar.addWidget(zoom_out_btn)

        zoom_in_btn = QPushButton("+ 放大")
        zoom_in_btn.setMinimumWidth(60)
        zoom_in_btn.setToolTip("放大")
        zoom_in_btn.clicked.connect(lambda: self._set_zoom(self.canvas._zoom * 1.1))
        toolbar.addWidget(zoom_in_btn)

        # Connect canvas zoom signal
        self.canvas.zoom_changed.connect(self._on_zoom_changed)

    def _setup_shortcuts(self):
        """设置快捷键"""
        self.shortcut_manager = ShortcutManager(self)

    def _setup_menu_bar(self):
        """设置菜单栏"""
        menubar = self.menuBar()

        # File Menu
        file_menu = menubar.addMenu("文件(&F)")
        file_menu.addAction("打开视频...", self._open_video, QKeySequence.Open)
        file_menu.addAction("打开图片目录...", self._open_images, QKeySequence("Ctrl+Shift+O"))
        file_menu.addSeparator()
        file_menu.addAction("保存", self._save, QKeySequence.Save)
        file_menu.addSeparator()
        file_menu.addAction("打开项目...", self._open_project, QKeySequence("Ctrl+Shift+P"))
        file_menu.addSeparator()

        # Export submenu
        export_menu = QMenu("导出", self)
        export_menu.setToolTipsVisible(True)

        # LabelImg / VOC XML export (支持 train/val/test 划分)
        # LabelImg uses PASCAL VOC XML (JSON label format is not implemented)
        voc_menu = QMenu("导出 LabelImg/VOC XML", self)
        voc_menu.addAction("当前帧", lambda: self._export_current_frame("voc"))
        voc_menu.addAction("批量导出", lambda: self._export_batch("voc"))
        export_menu.addMenu(voc_menu)

        # COCO export (支持 train/val/test 划分)
        coco_menu = QMenu("导出 COCO 数据集", self)
        coco_menu.addAction("当前帧", lambda: self._export_current_frame("coco"))
        coco_menu.addAction("批量导出", lambda: self._export_batch("coco"))
        export_menu.addMenu(coco_menu)

        export_menu.addSeparator()

        # ── Mask 导出子菜单 ──
        mask_menu = QMenu("导出 Mask 图像", self)
        mask_menu.addAction("当前帧", self._export_mask_current)
        mask_menu.addAction("批量导出", self._export_mask_batch)
        export_menu.addMenu(mask_menu)

        # ── YOLO 导出子菜单 (detection + segmentation) ──
        yolo_menu = QMenu("导出 YOLO 数据集", self)
        yolo_menu.addAction("检测格式 (bbox)", self._export_yolo_dataset)
        yolo_menu.addAction("分割格式 (polygon)", self._export_yolo_seg_dataset)
        yolo_menu.addAction("姿态格式 (pose)", lambda: self._export_yolo_task("yolo_pose", "YOLO 姿态"))
        yolo_menu.addAction("旋转框格式 (OBB)", lambda: self._export_yolo_task("yolo_obb", "YOLO OBB"))
        export_menu.addMenu(yolo_menu)

        file_menu.addMenu(export_menu)

        # Import submenu
        import_menu = QMenu("导入已标注数据集", self)
        import_menu.addAction("导入 YOLO 数据集", self._import_yolo_dataset)
        import_menu.addAction("导入 VOC 数据集", self._import_voc_dataset)
        import_menu.addAction("导入 COCO 数据集", self._import_coco_dataset)
        file_menu.addMenu(import_menu)

        file_menu.addSeparator()
        file_menu.addAction("退出", self.close, QKeySequence.Quit)

        # Edit Menu
        edit_menu = menubar.addMenu("编辑(&E)")
        edit_menu.addAction("撤销", self._undo, QKeySequence.Undo)
        edit_menu.addAction("重做", self._redo, QKeySequence.Redo)
        edit_menu.addSeparator()
        edit_menu.addAction("删除选中", self._delete_selected, QKeySequence.Delete)

        # View Menu
        view_menu = menubar.addMenu("视图(&V)")
        view_menu.addAction("切换资源管理器", self._toggle_explorer, QKeySequence("Ctrl+B"))
        view_menu.addAction("切换检查器", self._toggle_inspector, QKeySequence("Ctrl+Shift+B"))
        view_menu.addSeparator()
        view_menu.addAction("全屏", self._toggle_fullscreen, QKeySequence.FullScreen)

        # Tools Menu
        tools_menu = menubar.addMenu("工具(&T)")
        tools_menu.addAction("处理视频...", self._process_video)
        tools_menu.addAction("自动标注...", self._auto_annotate)
        if self._product.feature_detect_mode:
            tools_menu.addAction("Detect 当前帧...", self._run_detect)
        tools_menu.addAction("对当前图补分割", self._refine_segmentation_current)
        tools_menu.addAction("批量补分割...", self._refine_segmentation_batch)
        tools_menu.addAction("替换类别标签...", self._replace_class_labels)
        tools_menu.addSeparator()
        tools_menu.addAction("图片缩放...", self._open_image_resize)
        tools_menu.addAction("图片切分...", self._open_image_split)
        tools_menu.addSeparator()
        tools_menu.addAction("导出日志...", self._export_logs)

        model_menu = menubar.addMenu("模型(&M)")
        model_menu.addAction("加载 GroundingDINO", self._load_grounding_model)
        model_menu.addAction("卸载 GroundingDINO", self._unload_grounding_model)
        model_menu.addSeparator()
        model_menu.addAction("加载 SAM3", self._load_sam_model)
        model_menu.addAction("卸载 SAM3", self._unload_sam_model)
        model_menu.addSeparator()
        model_menu.addAction("加载自定义 YOLO...", self._load_onnx_yolo_model)
        model_menu.addAction("卸载自定义 YOLO", self._unload_onnx_yolo_model)
        model_menu.addSeparator()
        model_menu.addAction("查看模型状态", self._show_model_status)

        # Help Menu
        help_menu = menubar.addMenu("帮助(&H)")
        help_menu.addAction("快捷键", self._show_shortcuts, QKeySequence("Ctrl+?"))
        help_menu.addAction("关于", self._show_about)

    # ==================== Dataset Import ====================

    def _import_yolo_dataset(self):
        """导入 YOLO 格式数据集（选含 images/ + labels/ 的根目录）。"""
        from pathlib import Path
        root_dir = QFileDialog.getExistingDirectory(
            self, "选择数据集根目录（含 images/ 和 labels/）", ""
        )
        if not root_dir:
            return

        root = Path(root_dir)
        image_path = root / "images"
        label_path = root / "labels"

        if not image_path.is_dir() or not label_path.is_dir():
            QMessageBox.warning(
                self,
                "无效目录",
                "未找到 images/ 和 labels/ 子目录。\n"
                "请选择包含这两个子目录的数据集根目录（与导出 YOLO 后的目录一致）。",
            )
            return

        task = self.file_ctrl.prompt_yolo_import_task(self)
        if not task:
            return

        importer = YOLOImporter()
        try:
            image_files, frame_annotations, class_names, split_map, result = \
                importer.import_dataset(image_path, label_path, task_choice=task)
            if result.needs_task_choice:
                QMessageBox.information(
                    self,
                    "需要选择格式",
                    "未在 data.yaml 中找到 task / kpt_shape，不能把姿态或旋转框静默当成多边形。",
                )
                task = self.file_ctrl.prompt_yolo_import_task(self, required=True)
                if not task:
                    return
                image_files, frame_annotations, class_names, split_map, result = \
                    importer.import_dataset(image_path, label_path, task_choice=task)
            self._show_import_result(result, "YOLO",
                                      image_files, frame_annotations,
                                      class_names, split_map)
        except Exception as e:
            QMessageBox.critical(self, "导入失败", f"导入时发生错误:\n{str(e)}")

    def _import_voc_dataset(self):
        """导入 VOC 格式数据集"""
        from pathlib import Path
        root_dir = QFileDialog.getExistingDirectory(
            self, "选择数据集根目录（含 Annotations/ 和 JPEGImages/）", ""
        )
        if not root_dir:
            return

        root = Path(root_dir)
        image_dir = root / "JPEGImages"
        label_dir = root / "Annotations"

        if not image_dir.is_dir() or not label_dir.is_dir():
            QMessageBox.warning(self, "无效目录",
                "未找到 Annotations/ 和 JPEGImages/ 子目录。\n"
                "请选择包含这两个子目录的根目录。")
            return

        importer = VOCImporter()
        try:
            image_files, frame_annotations, class_names, split_map, result = \
                importer.import_dataset(image_dir, label_dir)
            self._show_import_result(result, "VOC",
                                      image_files, frame_annotations,
                                      class_names, split_map)
        except Exception as e:
            QMessageBox.critical(self, "导入失败", f"导入时发生错误:\n{str(e)}")

    def _import_coco_dataset(self):
        """导入 COCO 格式数据集"""
        from pathlib import Path
        root_dir = QFileDialog.getExistingDirectory(
            self, "选择数据集根目录（含 annotations/ 和 images/）", ""
        )
        if not root_dir:
            return

        root = Path(root_dir)
        image_dir = root / "images"
        label_dir = root / "annotations"

        if not image_dir.is_dir() or not label_dir.is_dir():
            QMessageBox.warning(self, "无效目录",
                "未找到 annotations/ 和 images/ 子目录。\n"
                "请选择包含这两个子目录的根目录。")
            return

        importer = COCOImporter()
        try:
            image_files, frame_annotations, class_names, split_map, result = \
                importer.import_dataset(image_dir, label_dir)
            self._show_import_result(result, "COCO",
                                      image_files, frame_annotations,
                                      class_names, split_map)
        except Exception as e:
            QMessageBox.critical(self, "导入失败", f"导入时发生错误:\n{str(e)}")

    def _show_import_result(self, result, format_name: str,
                             image_files, frame_annotations,
                             class_names, split_map):
        """显示导入结果汇总对话框，确认后在独立窗口中打开"""
        msg = f"[{format_name}] 导入完成\n\n{result.summary()}"
        if result.parse_errors:
            msg += f"\n\n共 {len(result.parse_errors)} 个警告"

        reply = QMessageBox.question(
            self, "导入结果",
            msg + "\n\n是否在新窗口中打开？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )

        if reply == QMessageBox.Yes:
            self._open_dataset_in_new_window(
                image_files, frame_annotations, class_names, split_map,
                kpt_shape=getattr(result, "kpt_shape", None),
                kpt_names=getattr(result, "kpt_names", None),
            )
        else:
            self._load_imported_dataset(
                image_files, frame_annotations, class_names, split_map,
                kpt_shape=getattr(result, "kpt_shape", None),
                kpt_names=getattr(result, "kpt_names", None),
            )

    def _open_dataset_in_new_window(self, image_files, frame_annotations,
                                     class_names, split_map,
                                     kpt_shape=None, kpt_names=None):
        """在新 MainWindow 中加载导入的数据集"""
        new_window = MainWindow()
        new_window._load_imported_dataset(
            image_files, frame_annotations, class_names, split_map,
            kpt_shape=kpt_shape, kpt_names=kpt_names,
        )
        new_window.show()

    def _load_imported_dataset(self, image_files, frame_annotations,
                                class_names, split_map,
                                kpt_shape=None, kpt_names=None):
        """将导入的数据集加载到当前窗口（供新窗口调用）"""
        if not self._confirm_project_replacement():
            return
        self.project.replace_data(
            image_files=list(image_files),
            frame_annotations=dict(frame_annotations),
            class_names=dict(class_names),
            split_map=dict(split_map),
            current_frame_index=0,
            kpt_shape=kpt_shape,
            kpt_names=kpt_names,
        )
        self.explorer.set_classes(self.project.class_names)
        self.inspector.set_class_names(self.project.class_names)

        if self.project.image_files:
            self.state_manager._state.current_file = self.project.image_files[0].parent
            self.state_manager._state.total_frames = len(self.project.image_files)
            self.state_manager.set_frame(0)
            self._frame_slider.setMaximum(len(self.project.image_files) - 1)
            self._update_frame_label()
            self._load_frame(0)

            directory = self.project.image_files[0].parent
            self._populate_file_tree(directory, self.project.image_files)

            self.statusBar().showMessage(
                f"已导入: {len(self.project.image_files)} 张图片, "
                f"{sum(len(a) for a in self.project.frame_annotations.values())} 个标注, "
                f"{len(self.project.class_names)} 个类别"
            )

        self._previous_frame = 0

    def _setup_status_bar(self):
        """设置状态栏"""
        status = self.statusBar()
        status.showMessage("Ready")

        self._model_load_progress = make_model_load_progress_bar()
        status.addPermanentWidget(self._model_load_progress)

        self._device_label = QLabel("设备: CPU")
        self._device_label.setStyleSheet("color: #888888; padding: 2px 8px; border: 1px solid #444; border-radius: 3px;")
        status.addPermanentWidget(self._device_label)

    def _show_model_load_progress(self) -> None:
        bar = getattr(self, "_model_load_progress", None)
        if bar is not None:
            bar.setVisible(True)

    def _hide_model_load_progress(self) -> None:
        bar = getattr(self, "_model_load_progress", None)
        if bar is not None:
            bar.setVisible(False)

    # ==================== Video/Image Loading ====================

    def _open_video(self):
        """打开视频文件"""
        if not self._confirm_project_replacement():
            return
        meta = self.file_ctrl.open_video(parent_widget=self)
        if meta is None:
            return
        self.explorer.set_classes({})
        self.inspector.set_class_names({})
        path = meta["path"]
        self.state_manager._state.current_file = path
        self.state_manager._state.total_frames = meta["total_frames"]
        self.state_manager._state.fps = meta["fps"]
        self.state_manager.set_frame(0)
        self._frame_slider.setMaximum(meta["total_frames"] - 1)
        self._update_frame_label()
        self.statusBar().showMessage(
            f"Video: {path.name} | {meta['total_frames']} frames | "
            f"{meta['fps']:.2f} fps | {meta['width']}x{meta['height']}"
        )
        self._load_frame(0)

    def _open_images(self):
        """打开图片目录"""
        if not self._confirm_project_replacement():
            return
        meta = self.file_ctrl.open_image_dir(parent_widget=self)
        if meta is None:
            return
        self.explorer.set_classes({})
        self.inspector.set_class_names({})
        directory = meta["directory"]
        self.state_manager._state.current_file = directory
        self.state_manager._state.total_frames = len(meta["image_files"])
        self.state_manager.set_frame(0)
        self._frame_slider.setMaximum(len(meta["image_files"]) - 1)
        self._populate_file_tree(directory, meta["image_files"])
        self._update_frame_label()
        self.statusBar().showMessage(
            f"Directory: {directory.name} | {len(meta['image_files'])} images"
        )
        self._load_frame(0)

    def _populate_file_tree(self, directory: Path, files: list):
        """在 ExplorerPanel 的 file_tree 中显示文件列表"""
        self.explorer.file_tree.clear()
        root_item = self.explorer.file_tree.invisibleRootItem()

        dir_item = QTreeWidgetItem([f"📁 {directory.name}"])
        dir_item.setData(0, Qt.UserRole, directory)
        root_item.addChild(dir_item)

        for idx, file_path in enumerate(files):
            file_item = QTreeWidgetItem([f"🖼️ {file_path.name}"])
            file_item.setData(0, Qt.UserRole, file_path)
            file_item.setData(0, Qt.UserRole + 1, idx)  # Store index
            dir_item.addChild(file_item)

        dir_item.setExpanded(True)

        if not hasattr(self, '_file_tree_connected'):
            self.explorer.file_tree.itemDoubleClicked.connect(self._on_file_tree_double_click)
            self._file_tree_connected = True

    def _on_file_tree_double_click(self, item: QTreeWidgetItem, column: int):
        """处理文件树双击事件"""
        file_path = item.data(0, Qt.UserRole)
        if file_path is None:
            return

        if file_path.is_file():
            # Find index and load
            idx = item.data(0, Qt.UserRole + 1)
            if idx is not None:
                self.state_manager.set_frame(idx)
        elif file_path.is_dir():
            item.setExpanded(not item.isExpanded())

    # ==================== Frame Loading ====================

    def _load_frame(self, frame_index: int):
        """加载指定帧"""
        frame_index = max(0, min(frame_index, self.state_manager._state.total_frames - 1))

        img = None
        if self.project.video_capture is not None:
            # Video mode
            self.project.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = self.project.video_capture.read()
            if ret:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self._current_image_path = self.state_manager._state.current_file
        elif self.project.image_files:
            # Image directory mode
            if 0 <= frame_index < len(self.project.image_files):
                # Use np.fromfile + cv2.imdecode for better Unicode path support on Windows
                # cv2.imread() fails silently with Chinese/non-ASCII characters in paths
                img_path = str(self.project.image_files[frame_index])
                try:
                    img_array = np.fromfile(img_path, dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        self._current_image_path = self.project.image_files[frame_index]
                except Exception as e:
                    # Fallback to cv2.imread for simple paths
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        self._current_image_path = self.project.image_files[frame_index]

        if img is not None:
            height, width, channel = img.shape
            bytes_per_line = 3 * width
            # Make a copy of the data to ensure it stays valid
            img_copy = img.copy()
            qimage = QImage(img_copy.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()
            self.canvas.set_image(qimage)

            # Load annotations for this frame
            self._load_frame_annotations(frame_index)

            # Update inspector
            self._update_inspector()

    def _load_frame_annotations(self, frame_index: int):
        """加载指定帧的标注"""
        # Keep ProjectDocument frame index in sync with UI navigation
        self.project._current_frame_index = frame_index

        # Disable history recording when loading
        self.project.enable_history(False)

        # Clear current annotations
        self.project.current_annotations.clear()
        self.project.clear_selection()

        # Load from stored annotations
        if frame_index in self.project.frame_annotations:
            for ann_data in self.project.frame_annotations[frame_index]:
                if isinstance(ann_data, dict):
                    ann = Annotation.from_dict(ann_data)
                else:
                    ann = Annotation.from_dict(ann_data.to_dict())
                self.project.current_annotations.append(ann)

        self.project._next_id = max(
            [a.id for a in self.project.current_annotations] + [0]
        ) + 1

        # Re-enable history recording
        self.project.enable_history(True)

        self.project.current_annotations_changed.emit(self.project.visible_annotations)

    def _save_frame_annotations(self, frame_index: int):
        """保存指定帧的标注到内存"""
        self.project.frame_annotations[frame_index] = [
            ann.clone()
            for ann in self.project.current_annotations
        ]

    # ==================== Playback Controls ====================

    def _first_frame(self):
        """跳转到第一帧"""
        self.playback_ctrl.first_frame()

    def _prev_frame(self):
        """前一帧"""
        self.playback_ctrl.prev_frame()

    def _next_frame(self):
        """下一帧"""
        self.playback_ctrl.next_frame()

    def _last_frame(self):
        """跳转到最后一帧"""
        self.playback_ctrl.last_frame()

    def _toggle_playback(self):
        self.playback_ctrl.toggle_playback()
        if self.playback_ctrl.is_playing():
            self._play_btn.setText("⏸")
        else:
            self._play_btn.setText("▶")

    def _on_playback_tick(self):
        """播放定时器触发（由 PlaybackController 内部处理）"""
        pass

    def _on_slider_changed(self, value: int):
        """滑块值改变"""
        self.playback_ctrl.go_to_frame(value)

    def _on_frame_changed(self, frame_index: int):
        """帧改变处理"""
        # 取消未完成的 polygon 绘制
        if self.canvas._tool == Tool.POLYGON:
            self.canvas._cancel_polygon()

        # Save current annotations before switching
        if hasattr(self, '_previous_frame'):
            self._save_frame_annotations(self._previous_frame)

        self._previous_frame = frame_index
        # Keep ProjectDocument index aligned even if image decode fails below
        self.project._current_frame_index = frame_index

        self._frame_slider.blockSignals(True)
        self._frame_slider.setValue(frame_index)
        self._frame_slider.blockSignals(False)

        self._update_frame_label()
        self._load_frame(frame_index)
        if getattr(self, "detect_ctrl", None) is not None:
            self._refresh_detect_overlays()

    def _update_frame_label(self):
        """更新帧标签"""
        current = self.state_manager._state.current_frame
        total = self.state_manager._state.total_frames
        self._frame_label.setText(f"帧: {current + 1} / {total}")

    def _set_zoom(self, zoom: float):
        """设置缩放"""
        self.canvas.set_zoom(zoom)

    def _on_zoom_changed(self, zoom: float):
        """缩放改变处理"""
        self._zoom_label.setText(f"{int(zoom * 100)}%")

    def _on_tool_selected(self, tool_name: str):
        """Handle tool selection from the explorer panel."""
        try:
            tool = Tool(tool_name)
        except ValueError:
            return

        self.state_manager.set_tool(tool)

    def _on_tool_changed(self, tool: Tool):
        """Keep canvas, explorer, and toolbar state in sync."""
        self.canvas.set_tool(tool)
        self.explorer.set_selected_tool(tool.value)

        for action_tool, action in getattr(self, "_tool_actions", {}).items():
            action.setChecked(action_tool == tool)

        tool_labels = {
            Tool.SELECT: "选择",
            Tool.RECTANGLE: "手动标注",
            Tool.POLYGON: "多边形绘制",
            Tool.ERASER: "删除标注",
            Tool.SAM_CLICK: "SAM 点选",
        }
        self.statusBar().showMessage(f"当前工具: {tool_labels.get(tool, tool.value)}", 2000)

    # ==================== Annotation Handlers ====================

    def _on_annotation_created(self, class_id: int, bbox: tuple):
        self.ann_ctrl.add_annotation(class_id, bbox)
        self._save_frame_annotations(self.state_manager._state.current_frame)
        self._update_inspector()

    def _on_polygon_created(self, class_id: int, bbox: tuple, polygon: list):
        ann = self.ann_ctrl.add_annotation(class_id, bbox, polygon=polygon)
        self._save_frame_annotations(self.state_manager._state.current_frame)
        self._update_inspector()
        self.statusBar().showMessage(
            f"多边形标注已创建 [{ann.id}], {len(polygon)} 个顶点", 3000
        )

    def _on_annotation_modified(self, ann_id: int, new_bbox: tuple):
        self.ann_ctrl.modify_annotation(ann_id, new_bbox)
        self._save_frame_annotations(self.state_manager._state.current_frame)
        self._update_inspector()

    def _on_annotation_deleted(self, ann_id: int):
        self.ann_ctrl.delete_annotation(ann_id)
        self._save_frame_annotations(self.state_manager._state.current_frame)
        self._update_inspector()

    def _on_annotation_selected(self, ann_id: int):
        self.ann_ctrl.select_annotation(ann_id)
        self._update_inspector()

    def _on_annotations_changed(self, annotations: list):
        """标注列表改变"""
        self.canvas.update()

    def _on_inspector_selection(self, annotation):
        """Inspector 选中处理"""
        if annotation:
            self.ann_ctrl.select_annotation(annotation.id)
            self.canvas._hovered_annotation_id = annotation.id
        else:
            self.project.clear_selection()
            self.canvas._hovered_annotation_id = None
        self.canvas.update()

    def _on_class_changed(self, ann_id: int, new_class: int):
        """类别改变"""
        self.ann_ctrl.modify_annotation_class(ann_id, new_class)
        self._save_frame_annotations(self.state_manager._state.current_frame)
        self.canvas.update()

    def _on_class_selected(self, class_id: int):
        """类别选中"""
        self.canvas.set_selected_class_id(class_id)
        self.state_manager.set_class(class_id)

    def _on_history_changed(self):
        """历史记录改变"""
        history = self.project.get_history()
        current_index = self.project._history_index
        self.explorer.update_history(history, current_index)

    def _on_action_performed(self, description: str):
        """动作执行"""
        # Update status bar with action description
        self.statusBar().showMessage(description, 3000)

    def _on_class_added(self, name: str):
        """类别添加"""
        classes = self.explorer.get_classes()
        class_id = None
        for cid, cname in classes.items():
            if cname == name:
                class_id = cid
                break
        if class_id is None:
            class_id = len(self.project.class_names)
        self.project.class_names[class_id] = name
        self.ann_ctrl.add_class(name)
        self.inspector.set_class_names(self.project.class_names)
        self.statusBar().showMessage(f"已添加类别: {name}", 2000)

    def _on_class_renamed(self, class_id: int, new_name: str):
        """类别重命名"""
        self.project.class_names[class_id] = new_name
        self.ann_ctrl.rename_class(class_id, new_name)
        self.inspector.set_class_names(self.project.class_names)
        self.canvas.update()
        self.statusBar().showMessage(f"已重命名类别为: {new_name}", 2000)

    def _on_class_deleted(self, class_id: int):
        """类别删除 — 唯一路径：ProjectDocument.remove_class（级联清标注）。"""
        self.ann_ctrl.delete_class(class_id)
        self.inspector.set_class_names(self.project.class_names)
        self._update_inspector()
        self.canvas.update()

    def _ensure_class_registered(self, class_id: int, class_name: str | None = None) -> int:
        """Ensure every class ID used by annotations exists in the UI."""
        if class_id in self.project.class_names:
            return class_id

        name = display_class_name(class_name) or f"Class {class_id}"
        self.project.class_names[class_id] = name
        self.explorer.add_class(name, class_id)
        self.inspector.set_class_names(self.project.class_names)
        return class_id

    def _resolve_annotation_class_id(self, raw_label) -> int | None:
        """Map model output labels to stable integer class IDs."""
        class_id = coerce_class_id(raw_label)
        if class_id is not None:
            return self._ensure_class_registered(class_id)

        normalized_label = normalize_class_name(raw_label)
        if not normalized_label:
            return None

        for existing_id, existing_name in self.project.class_names.items():
            if normalize_class_name(existing_name) == normalized_label:
                return existing_id

        class_name = display_class_name(raw_label)
        if not class_name:
            return None

        next_id = max(self.project.class_names.keys(), default=-1) + 1
        return self._ensure_class_registered(next_id, class_name)

    def _add_auto_annotations(self, annotations: list) -> tuple[int, int]:
        """Add validated auto-annotation results to the current frame."""
        self.project.enable_history(False)
        added = 0
        skipped = 0
        annotations_data = []

        try:
            for annotation in annotations:
                raw_label = getattr(annotation, 'class_name', None)
                bbox = getattr(annotation, 'bbox', None)
                confidence = getattr(annotation, 'score', None)
                polygon = getattr(annotation, 'polygon', None)

                if raw_label is None and isinstance(annotation, tuple) and len(annotation) >= 3:
                    raw_label, bbox, confidence = annotation[:3]
                    polygon = annotation[3] if len(annotation) > 3 else None

                if not isinstance(bbox, tuple) or len(bbox) != 4:
                    skipped += 1
                    continue

                class_id = self._resolve_annotation_class_id(raw_label)
                if class_id is None:
                    skipped += 1
                    continue

                annotations_data.append((class_id, bbox, confidence, polygon))
                added += 1

            if annotations_data:
                self.project.enable_history(True)
                self.ann_ctrl.add_annotations_batch(annotations_data, record_history=True)
        finally:
            self.project.enable_history(True)

        return added, skipped

    # ==================== Selection Handlers ====================

    def _on_select_all_annotations(self):
        """全选标注"""
        all_ids = {ann.id for ann in self.project.visible_annotations}
        self.project.set_selection(all_ids)
        self.canvas.update()
        self._update_inspector()
        self.inspector.clear_detail_view()

    def _on_invert_selection(self):
        """反选标注"""
        all_ids = {ann.id for ann in self.project.visible_annotations}
        selected = self.project.selected_ids
        new_selection = all_ids - selected
        self.project.set_selection(new_selection)
        self.canvas.update()
        self._update_inspector()
        self.inspector.clear_detail_view()

    def _on_delete_selected_annotations(self):
        """删除选中的标注"""
        self.ann_ctrl.delete_selected()
        self._save_frame_annotations(self.state_manager._state.current_frame)
        self._update_inspector()
        self.statusBar().showMessage(f"已删除选中的标注", 2000)

    def _on_batch_delete(self, ids: set):
        """批量删除指定 ID 的标注（来自 Inspector 列表多选删除）"""
        if ids:
            self.project.delete_annotations(ids)
            self._save_frame_annotations(self.state_manager._state.current_frame)
            self._update_inspector()
            self.statusBar().showMessage(f"已删除 {len(ids)} 个标注", 2000)

    def _on_multi_selection_changed(self, ids: set):
        """多选变更（Shift/Ctrl 多选时同步选中状态）"""
        self.project.set_selection(ids)
        self.canvas.update()

    def _update_inspector(self):
        """更新 Inspector 面板"""
        # Get current file name
        current_file = self.state_manager._state.current_file
        file_name = "-"
        split_name = "-"
        current_frame = self.state_manager._state.current_frame
        if current_file:
            file_name = current_file.name
            if self.project.video_capture:
                split_name = "video"
            elif self.project.image_files:
                # 显示当前帧的 split 归属
                split_name = self.project.split_map.get(current_frame, "images")

        # Get image size
        size_str = "-"
        if self.canvas._image:
            size_str = f"{self.canvas._image.width()}x{self.canvas._image.height()}"

        # Update properties
        self.inspector.update_properties({
            'file': file_name,
            'split': split_name,
            'size': size_str,
            'count': len(self.project.current_annotations)
        })

        # Update annotation list
        self.inspector.update_annotation_list(self.project.current_annotations, self.project.class_names)

        # Sync list selection state
        self.inspector.sync_list_selection(self.project.selected_ids)

        # Update selected annotation if any
        selected_ids = self.project.selected_ids
        if selected_ids:
            for ann in self.project.current_annotations:
                if ann.id in selected_ids:
                    self.inspector.set_selected_annotation(ann)
                    break
        else:
            self.inspector.set_selected_annotation(None)

        # Update split stats
        self._update_split_stats_display()

    def _update_split_stats_display(self):
        """更新 Inspector 中的数据集划分统计"""
        if not self.project.split_map:
            self.inspector.update_split_stats(-1, 0, 0)
            return
        counts = {"train": 0, "val": 0, "test": 0}
        for s in self.project.split_map.values():
            counts[s] = counts.get(s, 0) + 1
        self.inspector.update_split_stats(counts["train"], counts["val"], counts["test"])

    # ==================== Save/Export ====================

    def _save(self):
        """Save project (.gsproj)"""
        self.file_ctrl.save_project(parent_widget=self)

    def _confirm_project_replacement(self) -> bool:
        """Return whether a destructive project replacement may continue."""
        if not self.project.is_modified:
            return True
        reply = QMessageBox.question(
            self,
            "未保存的修改",
            "当前项目有未保存的修改，是否保存？",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
        )
        if reply == QMessageBox.Save:
            return self.file_ctrl.save_project(parent_widget=self)
        return reply == QMessageBox.Discard

    def _open_project(self):
        """Handle open .gsproj"""
        if not self._confirm_project_replacement():
            return
        if self.file_ctrl.open_project(parent_widget=self):
            # State already loaded into self.project by FileController.open_project
            self.explorer.set_classes(self.project.class_names)
            self.inspector.set_class_names(self.project.class_names)
            if self.project.image_files:
                directory = self.project.image_files[0].parent
                self.state_manager._state.current_file = directory
                self.state_manager._state.total_frames = len(self.project.image_files)
                saved_frame = max(0, min(self.project._current_frame_index, len(self.project.image_files) - 1))
                self.state_manager.set_frame(saved_frame)
                self._frame_slider.setMaximum(len(self.project.image_files) - 1)
                self._update_frame_label()
                self._populate_file_tree(directory, self.project.image_files)
                self._load_frame(saved_frame)
            elif self.project.video:
                video = self.project.video
                path = Path(video["path"])
                total_frames = int(video.get("total_frames", 0) or 0)
                fps = float(video.get("fps", 30.0) or 30.0)
                self.state_manager._state.current_file = path
                self.state_manager._state.total_frames = total_frames
                self.state_manager._state.fps = fps
                saved_frame = max(0, min(self.project._current_frame_index, max(total_frames - 1, 0)))
                self.state_manager.set_frame(saved_frame)
                self._frame_slider.setMaximum(max(total_frames - 1, 0))
                self._update_frame_label()
                self.statusBar().showMessage(
                    f"Video: {path.name} | {total_frames} frames | "
                    f"{fps:.2f} fps | {video.get('width', 0)}x{video.get('height', 0)}"
                )
                if self.project.video_capture is not None:
                    self._load_frame(saved_frame)

    def _export_yolo_dataset(self):
        """导出完整的 YOLO 数据集"""
        if not self.project.frame_annotations and not self.project.image_files:
            QMessageBox.warning(self, "警告", "没有数据可导出")
            return
        result = self.export_ctrl.export_batch("yolo", parent_widget=self)
        if result:
            extra = result.get("skip_message") or ""
            QMessageBox.information(
                self,
                "导出成功",
                "YOLO 数据集已导出。\n"
                f"Train: {result.get('train', 0)} 张\n"
                f"Val: {result.get('val', 0)} 张\n"
                f"Test: {result.get('test', 0)} 张"
                + (f"\n{extra}" if extra else ""),
            )

    # ==================== Menu Actions ====================

    def _undo(self):
        """撤销"""
        self.ann_ctrl.undo()
        self._save_frame_annotations(self.state_manager._state.current_frame)
        self._update_inspector()

    def _redo(self):
        """重做"""
        self.ann_ctrl.redo()
        self._save_frame_annotations(self.state_manager._state.current_frame)
        self._update_inspector()

    def _delete_selected(self):
        """删除选中的标注"""
        self.ann_ctrl.delete_selected()
        self._save_frame_annotations(self.state_manager._state.current_frame)
        self._update_inspector()

    def _toggle_explorer(self):
        """切换 Explorer 面板"""
        if self.explorer_dock.isVisible():
            self.explorer_dock.hide()
        else:
            self.explorer_dock.show()

    def _toggle_inspector(self):
        """切换 Inspector 面板"""
        if self.inspector_dock.isVisible():
            self.inspector_dock.hide()
        else:
            self.inspector_dock.show()

    def _toggle_fullscreen(self):
        """切换全屏"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    # ==================== Export Functions ====================

    def _get_exporter(self, format_type: str):
        """获取指定格式的导出器（未知格式 raise，不回退 YOLO）"""
        return self.export_ctrl._get_exporter(format_type)

    def _export_current_frame(self, format_type: str):
        """导出当前帧标注"""
        if self.canvas._image is None and self._current_image_path is None:
            QMessageBox.warning(self, "警告", "未加载图片")
            return

        if not self.project.current_annotations:
            QMessageBox.warning(self, "警告", "当前帧没有标注")
            return

        # 选择输出目录
        output_dir = QFileDialog.getExistingDirectory(
            self, "选择导出目录", "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if not output_dir:
            return

        output_dir = Path(output_dir)

        # 获取图片路径
        if self._current_image_path and isinstance(self._current_image_path, Path):
            image_path = self._current_image_path
        elif self.project.image_files:
            idx = self.state_manager._state.current_frame
            if 0 <= idx < len(self.project.image_files):
                image_path = self.project.image_files[idx]
            else:
                image_path = self.project.image_files[0]
        else:
            QMessageBox.warning(self, "警告", "无法确定图片路径")
            return

        # 准备标注数据
        annotations = []
        for ann in self.project.current_annotations:
            if ann.visible:
                annotations.append({
                    'class_id': ann.class_id,
                    'bbox': ann.bbox,
                    'polygon': ann.polygon,
                    'confidence': ann.confidence,
                    'visible': ann.visible
                })

        # 导出（含格式校验；未知格式经 QMessageBox 提示，不回退 YOLO）
        try:
            exporter = self._get_exporter(format_type)
            exporter.set_classes(self.project.class_names)
            output_path = exporter.export_current_frame(annotations, image_path, output_dir)
            self.statusBar().showMessage(f"已导出到: {output_path}", 5000)
            QMessageBox.information(self, "导出成功", f"标注已导出到:\n{output_path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"导出时出错:\n{str(e)}")

    def _export_batch(self, format_type: str):
        """批量导出标注"""
        if not self.project.frame_annotations and not self.project.image_files:
            QMessageBox.warning(self, "警告", "没有可导出的标注")
            return
        result = self.export_ctrl.export_batch(format_type, parent_widget=self)
        if result:
            lines = ["批量导出完成。"]
            for split in ("train", "val", "test"):
                lines.append(
                    f"{split.capitalize()}: {result.get(split, 0)} 张"
                )
            QMessageBox.information(self, "导出成功", "\n".join(lines))

    # ==================== Mask Export ====================

    def _export_mask_current(self):
        """导出当前帧 Mask 图像"""
        if self.canvas._image is None and self._current_image_path is None:
            QMessageBox.warning(self, "警告", "未加载图片")
            return

        # 获取当前图片路径
        if self._current_image_path and isinstance(self._current_image_path, Path):
            current_file = self._current_image_path
        elif self.project.image_files:
            idx = self.state_manager._state.current_frame
            if 0 <= idx < len(self.project.image_files):
                current_file = self.project.image_files[idx]
            else:
                QMessageBox.warning(self, "警告", "无法确定图片路径")
                return
        else:
            QMessageBox.warning(self, "警告", "无法确定图片路径")
            return

        # 获取当前帧标注
        annotations = self._get_visible_annotation_dicts()
        has_polygon = any(a.get("polygon") for a in annotations)

        if not annotations:
            QMessageBox.warning(self, "警告", "当前帧没有标注")
            return

        if not has_polygon:
            reply = QMessageBox.question(
                self, "提示",
                "当前帧标注没有分割 mask 数据。\n是否继续导出（仅输出空白图像）？",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.No:
                return

        output_dir = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if not output_dir:
            return

        exporter = MaskExporter(output_dir)
        exporter.set_classes(self.project.class_names)

        try:
            results = exporter.export_single(annotations, current_file, Path(output_dir))
            if not results:
                QMessageBox.critical(
                    self, "导出失败",
                    "Mask 导出失败：未能写入任何输出文件。",
                )
                return
            count = len(results)
            png_count = len(list(Path(output_dir).rglob("*.png")))
            self.statusBar().showMessage(
                f"Mask 已导出 ({count} 个模式, {png_count} 个文件) → {output_dir}", 5000
            )
            QMessageBox.information(
                self, "导出成功",
                f"Mask 图像已导出到:\n{output_dir}\n\n共生成 {count} 个文件。",
            )
        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"导出 Mask 时出错:\n{str(e)}")

    def _export_mask_batch(self):
        """批量导出 Mask 图像"""
        if not self.project.frame_annotations and not self.project.image_files:
            QMessageBox.warning(self, "警告", "没有可导出的标注")
            return

        output_dir = QFileDialog.getExistingDirectory(self, "选择输出根目录")
        if not output_dir:
            return

        data = self._collect_export_data()
        if not data:
            QMessageBox.warning(self, "警告", "没有可导出的数据")
            return

        # 检测是否有 polygon 数据
        polygon_count = 0
        total_count = 0
        for _, anns in data:
            for a in anns:
                total_count += 1
                if a.get("polygon"):
                    polygon_count += 1
        if polygon_count == 0 and total_count > 0:
            reply = QMessageBox.question(
                self, "提示",
                f"共 {total_count} 个标注，但都没有分割 mask (polygon) 数据。\n"
                "请确认已使用 SAM3 进行分割标注。\n\n是否仍要导出（将生成空白图像）？",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.No:
                return

        exporter = MaskExporter(output_dir)
        exporter.set_classes(self.project.class_names)

        # 若无已有划分，弹出配置对话框
        split_map = self._get_or_configure_split(len(data))
        if split_map is None:
            return  # 用户取消

        try:
            success, total, split_counts = exporter.export_batch(
                data, Path(output_dir), split_map=split_map,
            )
            if success == 0:
                QMessageBox.critical(
                    self, "导出失败",
                    f"批量 Mask 导出失败：没有成功导出任何图片（0/{total}）。",
                )
                return
            # 验证导出结果
            png_count = len(list(Path(output_dir).rglob("*.png")))
            msg = f"批量导出完成: {success}/{total}\n"
            if split_counts:
                msg += (
                    f"\nTrain: {split_counts.get('train', 0)} | "
                    f"Val: {split_counts.get('val', 0)} | "
                    f"Test: {split_counts.get('test', 0)}"
                )
            msg += f"\n\n生成 PNG 文件: {png_count} 个"
            self.statusBar().showMessage(msg.replace("\n", " "), 5000)
            QMessageBox.information(self, "导出成功", msg)
        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"批量导出时出错:\n{str(e)}")

    # ==================== YOLO Segmentation Export ====================

    def _export_yolo_seg_dataset(self):
        """导出 YOLO 分割格式数据集（委托 ExportController）"""
        if not self.project.frame_annotations and not self.project.image_files:
            QMessageBox.warning(self, "警告", "没有数据可导出")
            return
        result = self.export_ctrl.export_batch("yolo_seg", parent_widget=self)
        if result:
            extra = result.get("skip_message") or ""
            QMessageBox.information(
                self,
                "导出成功",
                "YOLO 分割数据集已导出。\n"
                f"Train: {result.get('train', 0)} 张\n"
                f"Val: {result.get('val', 0)} 张\n"
                f"Test: {result.get('test', 0)} 张"
                + (f"\n{extra}" if extra else ""),
            )

    def _export_yolo_task(self, format_type: str, title: str):
        if not self.project.frame_annotations and not self.project.image_files:
            QMessageBox.warning(self, "警告", "没有数据可导出")
            return
        result = self.export_ctrl.export_batch(format_type, parent_widget=self)
        if result:
            extra = result.get("skip_message") or ""
            QMessageBox.information(
                self,
                "导出成功",
                f"{title} 数据集已导出。\n"
                f"Train: {result.get('train', 0)} 张\n"
                f"Val: {result.get('val', 0)} 张\n"
                f"Test: {result.get('test', 0)} 张"
                + (f"\n{extra}" if extra else ""),
            )

    # ==================== Export Helpers ====================

    def _get_or_configure_split(self, n: int) -> dict | None:
        """弹出划分配置对话框（始终询问，与 ExportController 一致）。

        返回 None 表示用户取消。
        """
        from src.gui.dialogs.split_config_dialog import SplitConfigDialog
        dialog = SplitConfigDialog(n, self)
        if dialog.exec() == QDialog.Accepted:
            split_map = dialog.compute_split(n)
            self.project.set_split_map(split_map)
            return split_map
        return None

    def _get_visible_annotation_dicts(self) -> list:
        """获取当前帧所有可见标注的字典列表"""
        return [
            {
                "class_id": ann.class_id,
                "bbox": ann.bbox,
                "polygon": ann.polygon,
                "confidence": ann.confidence,
                "visible": ann.visible,
                "kind": getattr(ann, "kind", "bbox"),
                "obb": getattr(ann, "obb", None),
                "keypoints": getattr(ann, "keypoints", None),
            }
            for ann in self.project.current_annotations
        ]

    def _collect_export_data(self) -> list:
        """收集批量导出数据 [(img_path, annotations), ...]"""
        data = []
        for idx, img_path in enumerate(self.project.image_files or []):
            annotations = self.project.frame_annotations.get(idx, [])
            std_anns = self._normalize_annotations(annotations)
            if std_anns:
                data.append((img_path, std_anns))
        return data

    def _normalize_annotations(self, annotations: list) -> list:
        """将 Annotation 对象或混合列表统一为字典列表"""
        result = []
        for ann in annotations:
            if isinstance(ann, dict):
                result.append(ann)
            elif hasattr(ann, "class_id"):
                result.append({
                    "class_id": ann.class_id,
                    "bbox": ann.bbox,
                    "polygon": getattr(ann, "polygon", None),
                    "confidence": getattr(ann, "confidence", 1.0),
                    "visible": getattr(ann, "visible", True),
                })
        return result

    # ==================== Detect Mode ====================

    def _set_app_mode(self, mode: AppMode) -> None:
        if not self._product.feature_detect_mode and mode is AppMode.DETECT:
            return
        self.detect_ctrl.set_mode(mode)

    def _on_app_mode_changed(self, mode_value: str) -> None:
        mode = AppMode(mode_value)
        self._apply_app_mode(mode)

    def _apply_app_mode(self, mode: AppMode) -> None:
        is_detect = mode is AppMode.DETECT and self._product.feature_detect_mode
        if self._mode_annotate_btn is not None:
            self._mode_annotate_btn.setChecked(not is_detect)
        if self._mode_detect_btn is not None:
            self._mode_detect_btn.setChecked(is_detect)

        self.canvas.set_overlay_visibility(
            show_annotations=not is_detect,
            show_detections=is_detect,
        )
        if hasattr(self, "detect_dock"):
            self.detect_dock.setVisible(is_detect)
            if is_detect:
                self.detect_dock.raise_()
            else:
                self.inspector_dock.raise_()

        # Disable drawing tools in Detect mode
        for tool, action in getattr(self, "_tool_actions", {}).items():
            action.setEnabled(not is_detect)
        if hasattr(self, "_sam_commit_action") and is_detect:
            self._sam_commit_action.setEnabled(False)

        self._refresh_detect_overlays()
        if is_detect:
            self.statusBar().showMessage("Detect 模式：结果仅预览，不写入工程", 4000)
        else:
            self.statusBar().showMessage("Annotate 模式", 2000)

    def _detect_frame_key(self) -> str:
        frame = self.state_manager._state.current_frame
        path = self._current_image_path or self.state_manager.state.current_file
        if path:
            return f"{path}#{frame}"
        return f"frame:{frame}"

    def _run_detect(self) -> None:
        """Open Detect dialog and store transient overlays."""
        if not self._product.feature_detect_mode:
            return
        if self.canvas._image is None:
            QMessageBox.warning(self, "警告", "未加载图片")
            return
        if self.model_ctrl.get_active_detector() is None:
            QMessageBox.warning(self, "警告", "请先加载检测模型")
            return

        self._set_app_mode(AppMode.DETECT)
        dialog = DetectDialog(
            image=self.canvas._image,
            grounding_model=self.model_ctrl.get_grounding_model(),
            model_ctrl=self.model_ctrl,
            parent=self,
            project_class_names=self.project.class_names,
        )
        if dialog.exec() != QDialog.Accepted:
            return

        results = dialog.get_results()
        w, h = self.canvas._image.width(), self.canvas._image.height()
        source = str(self._current_image_path or self.state_manager.state.current_file or "")
        self.detect_ctrl.store_results(
            frame_key=self._detect_frame_key(),
            results=results,
            prompt=dialog.get_prompt(),
            box_threshold=dialog.get_box_threshold(),
            text_threshold=dialog.get_text_threshold(),
            image_width=w,
            image_height=h,
            source_path=source,
        )
        self.statusBar().showMessage(f"Detect 预览：{len(results)} 个目标", 4000)

    def _refresh_detect_overlays(self) -> None:
        if not hasattr(self, "detect_ctrl"):
            return
        key = self._detect_frame_key()
        # Activate frame key without relying on signal re-entry
        if self.detect_ctrl.state.active_key != key:
            self.detect_ctrl.state.set_active_key(key)
        frame = self.detect_ctrl.state.get_frame(key)
        visible = self.detect_ctrl.state.visible_results(key) if frame else []
        score = self.detect_ctrl.state.score_filter
        self.canvas.set_detection_overlays(visible, score_filter=score)
        if hasattr(self, "detect_panel"):
            self.detect_panel.set_results(visible)
            self.detect_panel.set_score_filter(score)

    def _on_detect_score_filter(self, threshold: float) -> None:
        visible = self.detect_ctrl.state.visible_results()
        self.canvas.set_detection_overlays(visible, score_filter=threshold)
        if hasattr(self, "detect_panel"):
            self.detect_panel.set_results(visible)

    def _clear_detect_preview(self) -> None:
        self.detect_ctrl.clear(self._detect_frame_key())
        self.statusBar().showMessage("已清除 Detect 预览", 2000)

    def _export_detect_json(self) -> None:
        if not self.detect_ctrl.state.has_any():
            QMessageBox.information(self, "提示", "没有可导出的 Detect 结果")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出 Detect JSON", "detections.json", "JSON (*.json)"
        )
        if not path:
            return
        out = self.detect_ctrl.export_json(Path(path), tool_name=self._product.app_name)
        self.statusBar().showMessage(f"已导出: {out}", 4000)

    def _export_detect_csv(self) -> None:
        if not self.detect_ctrl.state.has_any():
            QMessageBox.information(self, "提示", "没有可导出的 Detect 结果")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出 Detect CSV", "detections.csv", "CSV (*.csv)"
        )
        if not path:
            return
        out = self.detect_ctrl.export_csv(Path(path))
        self.statusBar().showMessage(f"已导出: {out}", 4000)

    def _export_detect_overlay(self) -> None:
        frame = self.detect_ctrl.state.get_frame()
        if frame is None or not frame.results:
            QMessageBox.information(self, "提示", "当前帧没有 Detect 结果")
            return
        image_path = Path(frame.source_path) if frame.source_path else None
        if image_path is None or not image_path.is_file():
            # Fall back to current image path
            cur = self._current_image_path
            if isinstance(cur, Path) and cur.is_file():
                image_path = cur
            else:
                QMessageBox.warning(self, "警告", "无法定位源图片路径，无法导出叠加图")
                return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出叠加图", f"{image_path.stem}_detect.png", "PNG (*.png)"
        )
        if not path:
            return
        out = self.detect_ctrl.export_active_overlay(image_path, Path(path))
        if out is None:
            QMessageBox.warning(self, "失败", "叠加图导出失败")
            return
        self.statusBar().showMessage(f"已导出叠加图: {out}", 4000)

    def _convert_detect_to_annotations(self) -> None:
        """Commit visible Detect results into ProjectDocument (Annotate)."""
        results = self.detect_ctrl.state.visible_results()
        if not results:
            QMessageBox.information(self, "提示", "没有可转入的检测结果")
            return
        reply = QMessageBox.question(
            self,
            "转入标注",
            replace_warning_message(
                f"将用 {len(results)} 个 Detect 结果替换当前帧标注，并切换到 Annotate 模式。继续？",
                has_polygons=frames_have_polygons([self.project.current_annotations]),
            ),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self.project._sync_current_to_frame()
        self.project.current_annotations.clear()
        self.project.clear_selection()
        self.project.blockSignals(True)
        try:
            added, skipped = self._add_auto_annotations(results)
            self._save_frame_annotations(self.state_manager._state.current_frame)
        finally:
            self.project.blockSignals(False)
            self.project.current_annotations_changed.emit(self.project.visible_annotations)

        self._set_app_mode(AppMode.ANNOTATE)
        self._update_inspector()
        self.canvas.update()
        msg = f"已转入 {added} 个标注"
        if skipped:
            msg += f"，跳过 {skipped}"
        self.statusBar().showMessage(msg, 4000)

    def _process_video(self):
        """处理视频"""
        if self.project.video_capture is None or not self.project.video_capture.isOpened():
            QMessageBox.warning(self, "警告", "请先打开视频")
            return

        if not self.model_ctrl.is_grounding_loaded():
            QMessageBox.warning(self, "警告", "模型未加载")
            return

        video_path = str(self.state_manager.state.current_file)
        dialog = ProcessVideoDialog(
            video_path,
            self.model_ctrl.get_grounding_model(),
            self,
            model_ctrl=self.model_ctrl,
        )
        if dialog.exec() == QDialog.Accepted:
            results = dialog.get_results()
            # Product Detect: fold batch video detections into DetectionState for scrubbing
            if self._product.feature_detect_mode:
                for frame_idx, payload in results.get("annotations", {}).items():
                    auto_results = payload.get("auto_results") or []
                    if not auto_results:
                        continue
                    source = payload.get("image_path", "")
                    key = f"{self.state_manager.state.current_file}#{frame_idx}"
                    self.detect_ctrl.store_results(
                        frame_key=key,
                        results=auto_results,
                        prompt="",
                        box_threshold=0.0,
                        text_threshold=0.0,
                        image_width=int(payload.get("image_width") or 0),
                        image_height=int(payload.get("image_height") or 0),
                        source_path=source,
                    )
                self._set_app_mode(AppMode.DETECT)
                self._refresh_detect_overlays()
            self.statusBar().showMessage(
                f"Processing complete: {len(results['annotations'])} frames", 5000
            )

    def _auto_annotate(self):
        """自动标注图像（支持单张和批量模式）"""
        if self.model_ctrl.get_active_detector() is None:
            QMessageBox.warning(self, "警告", "请先加载检测模型 (GroundingDINO 或自定义 YOLO)")
            return

        # Check if we have multiple images to enable batch mode
        has_multiple = len(self.project.image_files) > 1 if self.project.image_files else False

        # Fix 3: batch mode does not need a canvas image; single mode still requires it
        if not has_multiple and self.canvas._image is None:
            QMessageBox.warning(self, "警告", "未加载图片")
            return

        # Open auto annotate dialog (auto-enable batch mode if multiple images)
        dialog = AutoAnnotateDialog(
            grounding_model=self.model_ctrl.get_grounding_model(),
            sam_model=self.model_ctrl.get_sam_model(),
            parent=self,
            batch_mode=has_multiple,
            image_files=self.project.image_files if has_multiple else None,
            model_ctrl=self.model_ctrl,
            project_class_names=self.project.class_names,
        )

        if dialog.exec() == QDialog.Accepted:
            # 记录自动标注会话开始
            session_start_time = datetime.now()
            session_log = {
                "session_time": session_start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "mode": "batch" if dialog.is_batch_mode() else "single",
                "parameters": {
                    "prompt": dialog.get_prompt(),
                    "box_threshold": dialog.get_box_threshold(),
                    "text_threshold": dialog.get_text_threshold()
                },
                "results": []
            }

            if dialog.is_batch_mode():
                # Batch mode - apply all annotations
                batch_results = dialog.get_batch_results()
                current_idx = self.state_manager._state.current_frame
                groups = [
                    list(self.project.current_annotations)
                    if frame_idx == current_idx
                    else list(self.project.frame_annotations.get(frame_idx, []))
                    for frame_idx in batch_results
                ]
                if frames_have_polygons(groups):
                    msg = replace_warning_message(
                        f"将替换 {len(batch_results)} 帧的已有标注。",
                        has_polygons=True,
                        polygon_frame_count=count_frames_with_polygons(groups),
                    )
                    if QMessageBox.question(
                        self, "自动标注", msg, QMessageBox.Yes | QMessageBox.No
                    ) != QMessageBox.Yes:
                        return
                total_added = 0
                total_skipped = 0

                for frame_idx, annotations in batch_results.items():
                    if frame_idx < len(self.project.image_files):
                        # Switch to this frame
                        self.state_manager.set_frame(frame_idx)

                        # Clear existing annotations for this frame
                        self.project.current_annotations.clear()
                        self.project.clear_selection()

                        # Add new annotations with validation
                        frame_added = 0
                        frame_skipped = 0
                        if annotations:
                            added, skipped = self._add_auto_annotations(annotations)
                            print(f"[Debug] 添加结果: added={added}, skipped={skipped}")
                            frame_added = added
                            frame_skipped = skipped
                            total_added += added
                            total_skipped += skipped

                        # 记录每帧结果
                        session_log["results"].append({
                            "frame_index": frame_idx,
                            "image_name": self.project.image_files[frame_idx].name if frame_idx < len(self.project.image_files) else f"frame_{frame_idx}",
                            "added": frame_added,
                            "skipped": frame_skipped,
                            "total_detected": len(annotations) if annotations else 0
                        })

                        self._save_frame_annotations(frame_idx)

                # 接收并存储数据集划分结果（经 set_split_map 标记 dirty）
                split_result = dialog.get_split_result()
                if split_result:
                    self.project.set_split_map(split_result)
                    session_log["split_result"] = split_result
                    self._update_split_stats_display()

                # 记录总体统计
                session_log["summary"] = {
                    "total_frames_processed": len(batch_results),
                    "total_annotations_added": total_added,
                    "total_skipped": total_skipped,
                    "split_applied": split_result is not None
                }

                # Update UI
                self._update_inspector()
                status_msg = f"批处理完成: {len(batch_results)} 张图片, {total_added} 个标注"
                if total_skipped:
                    status_msg += f"，跳过 {total_skipped} 条无效标签"
                if split_result:
                    counts = {"train": 0, "val": 0, "test": 0}
                    for s in split_result.values():
                        counts[s] = counts.get(s, 0) + 1
                    status_msg += f" | T:{counts['train']} V:{counts['val']} Te:{counts['test']}"
                self.statusBar().showMessage(status_msg, 5000)

                # Refresh current frame display
                current_frame = self.state_manager._state.current_frame
                self._on_frame_changed(current_frame)

            else:
                # Single image mode - add to current frame only
                annotations = dialog.get_annotations()
                session_log["results"].append({
                    "frame_index": self.state_manager._state.current_frame,
                    "image_name": str(self._current_image_path) if self._current_image_path else "current_frame",
                    "added": 0,
                    "skipped": 0,
                    "total_detected": len(annotations) if annotations else 0
                })

                if annotations:
                    if frames_have_polygons([self.project.current_annotations]):
                        if QMessageBox.question(
                            self,
                            "自动标注",
                            POLYGON_REPLACE_NOTE,
                            QMessageBox.Yes | QMessageBox.No,
                        ) != QMessageBox.Yes:
                            return
                    # Clear existing annotations before adding new ones (same as batch mode)
                    self.project._sync_current_to_frame()
                    self.project.current_annotations.clear()
                    self.project.clear_selection()

                    # Block signals during batch add to prevent multiple UI updates
                    self.project.blockSignals(True)
                    added = 0
                    skipped = 0
                    try:
                        added, skipped = self._add_auto_annotations(annotations)
                        self._save_frame_annotations(self.state_manager._state.current_frame)
                    finally:
                        # Unblock signals and emit single update
                        self.project.blockSignals(False)
                        self.project.current_annotations_changed.emit(
                            self.project.visible_annotations)

                    session_log["results"][0]["added"] = added
                    session_log["results"][0]["skipped"] = skipped
                    session_log["summary"] = {
                        "total_frames_processed": 1,
                        "total_annotations_added": added,
                        "total_skipped": skipped,
                        "split_applied": False
                    }

                    self._update_inspector()
                    self.canvas.update()
                    status_msg = f"已添加 {added} 个标注"
                    if skipped:
                        status_msg += f"，跳过 {skipped} 条无效标签"
                    self.statusBar().showMessage(status_msg, 3000)

            # 保存自动标注日志
            session_log["session_end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._auto_annotation_logs.append(session_log)

    def _refine_segmentation_current(self):
        """对当前帧已有框补分割（SAM3 按需加载，会话内常驻）。"""
        if self.canvas._image is None:
            QMessageBox.warning(self, "警告", "未加载图片")
            return
        anns = list(self.project.current_annotations)
        if not anns:
            QMessageBox.information(self, "提示", "当前图没有检测框，请先自动标注或手动画框。")
            return
        if not self.model_ctrl.ensure_sam3_loaded(self):
            self.statusBar().showMessage("SAM3 未加载，补分割已取消", 4000)
            return

        from .dialogs.auto_annotate_dialog import _qimage_to_pil

        pil = _qimage_to_pil(self.canvas._image)
        if pil is None:
            QMessageBox.warning(self, "警告", "无法读取当前图像")
            return

        pairs = [(a.id, a.bbox) for a in anns]
        prompt = _prompt_from_class_names(
            self.project.class_names, [a.class_id for a in anns]
        )
        sam = self.model_ctrl.get_sam_model()
        self._refine_expected_frame = self.state_manager._state.current_frame
        self.statusBar().showMessage("正在对当前图补分割...", 0)
        self._refine_worker = RefineSegmentationWorker(
            sam, pil, pairs, prompt=prompt, model_ctrl=self.model_ctrl
        )
        self._refine_worker.progress.connect(
            lambda msg: self.statusBar().showMessage(msg, 0)
        )
        self._refine_worker.error.connect(self._on_refine_current_error)
        self._refine_worker.finished.connect(self._on_refine_current_finished)
        self._refine_worker.start()

    def _on_refine_current_error(self, err: str):
        self._refine_expected_frame = None
        QMessageBox.critical(self, "补分割失败", err)
        self.statusBar().showMessage("补分割失败", 4000)

    def _on_refine_current_finished(self, results: list):
        expected = getattr(self, "_refine_expected_frame", None)
        current = self.state_manager._state.current_frame
        self._refine_expected_frame = None
        if expected is not None and current != expected:
            QMessageBox.warning(
                self,
                "补分割已丢弃",
                f"补分割期间已切换帧（期望 {expected}，当前 {current}），结果未写入以免写错图。\n"
                "请回到原图后重新补分割。",
            )
            self.statusBar().showMessage("补分割结果已丢弃（帧已切换）", 5000)
            return
        updated = self.ann_ctrl.apply_polygons_batch(results)
        self._update_inspector()
        self.canvas.update()
        self.statusBar().showMessage(f"补分割完成: 更新 {updated} 个多边形", 4000)

    def _refine_segmentation_batch(self):
        """批量：对所有已有框的帧补分割。"""
        if not self.project.image_files:
            QMessageBox.warning(self, "警告", "未加载任何图片。")
            return

        # Persist current frame first
        self.ann_ctrl.save_current_frame_to_project()

        jobs = []
        for frame_idx, path in enumerate(self.project.image_files):
            anns = self.project.frame_annotations.get(frame_idx, [])
            # frame may store Annotation objects
            pairs = []
            class_ids = []
            for a in anns:
                if hasattr(a, "bbox"):
                    pairs.append((a.id, a.bbox))
                    class_ids.append(a.class_id)
                elif isinstance(a, dict) and "bbox" in a:
                    pairs.append((a.get("id", -1), tuple(a["bbox"])))
                    class_ids.append(a.get("class_id", 0))
            if not pairs:
                continue
            jobs.append({
                "frame_idx": frame_idx,
                "image_path": path,
                "pairs": pairs,
                "prompt": _prompt_from_class_names(self.project.class_names, class_ids),
            })

        if not jobs:
            QMessageBox.information(self, "提示", "没有带检测框的图片可补分割。")
            return

        dialog = RefineSegmentationDialog(self.model_ctrl, jobs, parent=self)
        if dialog.exec() != QDialog.Accepted:
            return

        results_by_frame = dialog.get_results()
        total_updated = 0
        current_frame = self.state_manager._state.current_frame
        # Write polygons offline into frame_annotations (dict or Annotation).
        # Avoid change_frame here: it triggers UI reload and used to crash on dict storage.
        for frame_idx, pairs in results_by_frame.items():
            total_updated += self.ann_ctrl.apply_polygons_to_frame(
                frame_idx, pairs, record_history=True
            )

        # Refresh the visible frame from updated storage
        self._load_frame_annotations(current_frame)
        self._update_inspector()
        self.canvas.update()
        self.statusBar().showMessage(
            f"批量补分割完成: {len(results_by_frame)} 帧, {total_updated} 个多边形",
            5000,
        )

    def _open_image_resize(self):
        """Open standalone image resize preprocessing dialog."""
        dialog = ImageResizeDialog(self)
        dialog.exec()

    def _open_image_split(self):
        """Open standalone image tile-split preprocessing dialog."""
        dialog = ImageSplitDialog(self)
        dialog.exec()

    def _replace_class_labels(self):
        """Batch replace class labels dialog and execution."""
        if not self.project.image_files:
            QMessageBox.warning(self, "警告", "未加载任何图片。")
            return

        if not self.project.class_names:
            QMessageBox.information(self, "提示", "没有已注册的类别。")
            return

        total_annotations = sum(
            len(anns) for anns in self.project.frame_annotations.values()
        )
        if total_annotations == 0:
            QMessageBox.information(self, "提示", "当前没有任何标注。")
            return

        dialog = ReplaceClassDialog(self.project.class_names, self)
        if dialog.exec() != QDialog.Accepted:
            return

        source_class_id = dialog.get_source_class_id()
        target_name = dialog.get_target_class_name()
        current_only = dialog.is_current_image_only()

        if not target_name:
            QMessageBox.warning(self, "警告", "请输入目标类别名称。")
            return

        target_class_id = self._resolve_target_class_id(target_name)

        if source_class_id >= 0 and source_class_id == target_class_id:
            QMessageBox.warning(
                self, "警告",
                f"源类别和目标类别相同 (\"{target_name}\")。无需替换。"
            )
            return

        if current_only:
            self._replace_class_in_current_frame(source_class_id, target_class_id, target_name)
        else:
            self._replace_class_all_frames(source_class_id, target_class_id, target_name)

    def _resolve_target_class_id(self, target_name):
        """Find existing class_id for target_name, or create a new one."""
        target_lower = target_name.lower()
        for cid, cname in self.project.class_names.items():
            if cname.lower() == target_lower:
                return cid
        new_id = max(self.project.class_names.keys(), default=-1) + 1
        self.project.class_names[new_id] = target_name
        self.explorer.add_class(target_name, new_id)
        self.inspector.set_class_names(self.project.class_names)
        return new_id

    def _replace_class_in_current_frame(self, source_class_id, target_class_id, target_name):
        """Replace class labels in current frame only."""
        self.project.enable_history(False)

        replaced_count = 0
        for ann in self.project.current_annotations:
            if source_class_id < 0 or ann.class_id == source_class_id:
                ann.class_id = target_class_id
                replaced_count += 1

        self.project.enable_history(True)

        if replaced_count == 0:
            self.statusBar().showMessage("当前图片中没有匹配的标注。", 3000)
            return

        if source_class_id >= 0:
            self._maybe_remove_orphan_class(source_class_id)

        self._ensure_class_registered(target_class_id, target_name)

        self._save_frame_annotations(self.state_manager._state.current_frame)

        self.explorer.set_classes(self.project.class_names)
        self.inspector.set_class_names(self.project.class_names)
        self._update_inspector()
        self.project.current_annotations_changed.emit(
            self.project.visible_annotations
        )
        self.canvas.update()

        self.statusBar().showMessage(
            f"已替换 {replaced_count} 个标注的类别为 \"{target_name}\" (仅当前图片)",
            5000
        )

    def _replace_class_all_frames(self, source_class_id, target_class_id, target_name):
        """Replace class labels across all frames."""
        if source_class_id < 0:
            scope_desc = "所有类别"
        else:
            scope_desc = f"类别 \"{self.project.class_names.get(source_class_id, '未知')}\""

        reply = QMessageBox.warning(
            self, "确认全局替换",
            f"即将在所有 {len(self.project.image_files)} 张图片中，\n"
            f"将 {scope_desc} 的标注替换为 \"{target_name}\"。\n\n"
            f"此操作不可撤销。确定继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        current_frame = self.state_manager._state.current_frame
        self._save_frame_annotations(current_frame)

        total_replaced = 0
        frames_affected = 0

        for frame_idx, frame_data in self.project.frame_annotations.items():
            replaced = False
            normalized = []
            for raw_ann in frame_data:
                ann = self.project._coerce_annotation(raw_ann)
                if source_class_id < 0 or ann.class_id == source_class_id:
                    ann.class_id = target_class_id
                    total_replaced += 1
                    replaced = True
                normalized.append(ann)
            self.project.frame_annotations[frame_idx] = normalized
            if replaced:
                frames_affected += 1

        if total_replaced:
            self.project._mark_dirty()

        if source_class_id >= 0:
            self._maybe_remove_orphan_class(source_class_id)

        self._ensure_class_registered(target_class_id, target_name)

        self._load_frame_annotations(current_frame)

        self.explorer.set_classes(self.project.class_names)
        self.inspector.set_class_names(self.project.class_names)
        self._update_inspector()
        self.project.current_annotations_changed.emit(
            self.project.visible_annotations
        )
        self.canvas.update()

        self.statusBar().showMessage(
            f"已替换 {total_replaced} 个标注为 \"{target_name}\" "
            f"({frames_affected}/{len(self.project.image_files)} 张图片)",
            5000
        )

    def _maybe_remove_orphan_class(self, class_id):
        """Remove class_id from class_names if no annotations reference it."""
        if class_id not in self.project.class_names:
            return
        for frame_data in self.project.frame_annotations.values():
            for raw_ann in frame_data:
                if self.project._coerce_annotation(raw_ann).class_id == class_id:
                    return
        self.project.remove_class(class_id)
        self.explorer.remove_class(class_id)

    def _load_grounding_model(self):
        """加载 GroundingDINO，连接前断开旧信号以防重复绑定。"""
        from .controllers.model_controller import INFERENCE_BUSY_MESSAGE

        started = self.model_ctrl.load_grounding_dino(str(_get_project_root()))
        if not started:
            self.statusBar().showMessage(INFERENCE_BUSY_MESSAGE, 5000)
            QMessageBox.warning(self, "无法切换模型", INFERENCE_BUSY_MESSAGE)
            return
        model = self.model_ctrl.get_grounding_model()
        if model is None:
            return
        if model.is_loaded():
            self.statusBar().showMessage("GroundingDINO 已加载", 3000)
        else:
            try:
                model.loading_progress.disconnect(self._on_model_loading)
            except RuntimeError:
                pass
            try:
                model.loading_finished.disconnect(self._on_model_loaded)
            except RuntimeError:
                pass
            model.loading_progress.connect(self._on_model_loading)
            model.loading_finished.connect(self._on_model_loaded)
            self._show_model_load_progress()
            self.statusBar().showMessage("正在加载 GroundingDINO...", 0)

    def _unload_grounding_model(self):
        from .controllers.model_controller import INFERENCE_BUSY_MESSAGE

        if not self.model_ctrl.unload_grounding_dino():
            self.statusBar().showMessage(INFERENCE_BUSY_MESSAGE, 5000)
            QMessageBox.warning(self, "无法卸载", INFERENCE_BUSY_MESSAGE)
            return
        self.statusBar().showMessage("GroundingDINO 已卸载", 3000)

    def _load_sam_model(self):
        """加载 SAM3（异步），进度与结果走信号。"""
        from .controllers.model_controller import INFERENCE_BUSY_MESSAGE

        model = self.model_ctrl.get_sam_model()
        if model is None:
            from .models.sam3_model import SAM3Model
            model = SAM3Model()
            self.model_ctrl.set_sam_model(model)
        try:
            model.loading_progress.disconnect(self._on_model_loading)
        except RuntimeError:
            pass
        try:
            model.loading_finished.disconnect(self._on_sam3_loaded)
        except RuntimeError:
            pass
        model.loading_progress.connect(self._on_model_loading)
        model.loading_finished.connect(self._on_sam3_loaded)

        self._show_model_load_progress()
        self.statusBar().showMessage("正在加载 SAM3...", 0)
        started = self.model_ctrl.load_sam3(self)
        if not started:
            self._hide_model_load_progress()
            msg = (
                INFERENCE_BUSY_MESSAGE
                if self.model_ctrl.is_inference_busy
                else "SAM3 加载已取消"
            )
            self.statusBar().showMessage(msg, 4000)
            if self.model_ctrl.is_inference_busy:
                QMessageBox.warning(self, "无法切换模型", INFERENCE_BUSY_MESSAGE)
        # Completion message comes from _on_sam3_loaded

    def _on_sam3_loaded(self, success: bool):
        """SAM3 加载完成回调"""
        self._hide_model_load_progress()
        if success:
            self.statusBar().showMessage("SAM3 已加载", 3000)
        else:
            err = self.model_ctrl.get_last_sam_error() or "未知错误"
            self.statusBar().showMessage(f"SAM3 加载失败: {err}", 5000)

    def _unload_sam_model(self):
        from .controllers.model_controller import INFERENCE_BUSY_MESSAGE

        if not self.model_ctrl.unload_sam3():
            self.statusBar().showMessage(INFERENCE_BUSY_MESSAGE, 5000)
            QMessageBox.warning(self, "无法卸载", INFERENCE_BUSY_MESSAGE)
            return
        self.statusBar().showMessage("SAM3 已卸载", 3000)

    def _load_onnx_yolo_model(self):
        """Browse a local .onnx and load it as the active detector."""
        from .controllers.model_controller import INFERENCE_BUSY_MESSAGE
        from .models.detector_protocol import ONNX_RUNTIME_MISSING_MESSAGE
        from .models.onnx_yolo_model import OnnxYoloModel

        if not OnnxYoloModel.is_available():
            QMessageBox.warning(self, "无法加载 YOLO", ONNX_RUNTIME_MISSING_MESSAGE)
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 YOLO 检测 ONNX",
            "",
            "YOLO ONNX (*.onnx);;所有文件 (*.*)",
        )
        if not path:
            return
        model = OnnxYoloModel(path)
        self.model_ctrl.set_onnx_yolo_model(model)
        try:
            model.loading_progress.disconnect(self._on_model_loading)
        except RuntimeError:
            pass
        try:
            model.loading_finished.disconnect(self._on_onnx_yolo_loaded)
        except RuntimeError:
            pass
        model.loading_progress.connect(self._on_model_loading)
        model.loading_finished.connect(self._on_onnx_yolo_loaded)
        self._show_model_load_progress()
        self.statusBar().showMessage("正在加载自定义 YOLO...", 0)
        started = self.model_ctrl.load_onnx_yolo(path, self)
        if not started:
            self._hide_model_load_progress()
            msg = (
                INFERENCE_BUSY_MESSAGE
                if self.model_ctrl.is_inference_busy
                else ONNX_RUNTIME_MISSING_MESSAGE
            )
            self.statusBar().showMessage(msg, 4000)
            if self.model_ctrl.is_inference_busy:
                QMessageBox.warning(self, "无法切换模型", INFERENCE_BUSY_MESSAGE)
            elif not OnnxYoloModel.is_available():
                QMessageBox.warning(self, "无法加载 YOLO", ONNX_RUNTIME_MISSING_MESSAGE)

    def _on_onnx_yolo_loaded(self, success: bool):
        self._hide_model_load_progress()
        if success:
            self.statusBar().showMessage("自定义 YOLO 已加载", 3000)
            return
        model = self.model_ctrl.get_onnx_yolo_model()
        err = model.get_last_error() if model is not None and hasattr(model, "get_last_error") else "未知错误"
        self.statusBar().showMessage(f"YOLO 加载失败: {err}", 5000)

    def _unload_onnx_yolo_model(self):
        from .controllers.model_controller import INFERENCE_BUSY_MESSAGE

        if not self.model_ctrl.unload_onnx_yolo():
            self.statusBar().showMessage(INFERENCE_BUSY_MESSAGE, 5000)
            QMessageBox.warning(self, "无法卸载", INFERENCE_BUSY_MESSAGE)
            return
        self.statusBar().showMessage("自定义 YOLO 已卸载", 3000)

    def _show_model_status(self):
        status = self.model_ctrl.get_model_status()
        grounding_loaded = status["grounding_dino"]
        sam_loaded = status["sam3"]
        sam_error = None
        sam_model = self.model_ctrl.get_sam_model()
        if sam_model is not None and not sam_loaded:
            sam_error = sam_model.get_last_error()

        lines = [
            f"GroundingDINO: {'已加载' if grounding_loaded else '未加载'}",
        ]
        yolo_loaded = status.get("onnx_yolo", False)
        lines.append(f"自定义 YOLO: {'已加载' if yolo_loaded else '未加载'}")
        lines.append(f"SAM3: {'已加载' if sam_loaded else '未加载'}")
        if sam_error:
            lines.append(f"SAM3 错误: {sam_error}")

        QMessageBox.information(self, "模型状态", "\n".join(lines))

    def _show_shortcuts(self):
        """显示快捷键"""
        shortcuts_text = """
        <h3>键盘快捷键</h3>
        <table>
        <tr><td><b>Ctrl+O</b></td><td>打开视频</td></tr>
        <tr><td><b>Ctrl+Shift+O</b></td><td>打开图片目录</td></tr>
        <tr><td><b>Ctrl+S</b></td><td>保存</td></tr>
        <tr><td><b>Ctrl+Shift+P</b></td><td>打开项目</td></tr>
        <tr><td><b>←/→</b></td><td>上一帧/下一帧</td></tr>
        <tr><td><b>Space</b></td><td>播放/暂停</td></tr>
        <tr><td><b>Home/End</b></td><td>首帧/末帧</td></tr>
        <tr><td><b>Ctrl+Z</b></td><td>撤销</td></tr>
        <tr><td><b>Ctrl+Y</b></td><td>重做</td></tr>
        <tr><td><b>Delete</b></td><td>删除选中</td></tr>
        <tr><td><b>A</b></td><td>SAM 点选</td></tr>
        <tr><td><b>Ctrl+B</b></td><td>切换资源管理器</td></tr>
        <tr><td><b>Ctrl+Shift+B</b></td><td>切换检查器</td></tr>
        <tr><td><b>F11</b></td><td>全屏</td></tr>
        </table>
        """
        QMessageBox.about(self, "快捷键", shortcuts_text)

    def _show_about(self):
        """显示关于"""
        QMessageBox.about(self, "关于", self._product.about_html)

    def _export_logs(self):
        """导出会话日志"""
        # 收集日志数据
        log_data = {
            "export_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "session_info": {
                "current_file": str(self.state_manager._state.current_file) if self.state_manager._state.current_file else None,
                "total_frames": self.state_manager._state.total_frames,
                "current_frame": self.state_manager._state.current_frame,
                "device": self._device
            },
            "classes": self.project.class_names.copy(),
            "split_stats": {},
            "annotations_by_frame": {},
            "operation_history": [],
            "auto_annotation_sessions": self._auto_annotation_logs.copy()
        }

        # 统计数据集划分
        if self.project.split_map:
            split_counts = {"train": 0, "val": 0, "test": 0}
            for frame_idx, split in self.project.split_map.items():
                split_counts[split] = split_counts.get(split, 0) + 1
            log_data["split_stats"] = split_counts

        # 每帧标注统计
        for frame_idx, frame_data in self.project.frame_annotations.items():
            annotations = [
                self.project._coerce_annotation(ann) for ann in frame_data
            ]
            log_data["annotations_by_frame"][str(frame_idx)] = {
                "count": len(annotations),
                "split": self.project.split_map.get(frame_idx, "unknown"),
                "annotations": [
                    {
                        "id": ann.id,
                        "class_id": ann.class_id,
                        "class_name": self.project.class_names.get(
                            ann.class_id, "unknown"
                        ),
                        "confidence": ann.confidence,
                        "bbox": ann.bbox,
                        "visible": ann.visible,
                    }
                    for ann in annotations
                ]
            }

        # 操作历史记录
        history = self.project.get_history()
        for idx, action in enumerate(history):
            log_data["operation_history"].append({
                "index": idx,
                "type": action.type if hasattr(action, 'type') else "unknown",
                "description": action.description if hasattr(action, 'description') else str(action),
                "annotation_count": len(action.annotations) if hasattr(action, 'annotations') else 0
            })

        # 选择保存位置
        default_filename = f"annotation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出日志", default_filename,
            "JSON 文件 (*.json);;所有文件 (*)"
        )

        if not file_path:
            return

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)

            # 显示摘要
            total_annotations = sum(len(anns) for anns in self.project.frame_annotations.values())
            auto_sessions = len(self._auto_annotation_logs)
            auto_total_added = sum(session.get("summary", {}).get("total_annotations_added", 0) for session in self._auto_annotation_logs)

            summary = f"""
            <h3>日志导出成功</h3>
            <p><b>导出文件:</b> {Path(file_path).name}</p>
            <p><b>会话信息:</b></p>
            <ul>
            <li>总帧数: {log_data['session_info']['total_frames']}</li>
            <li>已标注帧数: {len(self.project.frame_annotations)}</li>
            <li>总标注数: {total_annotations}</li>
            <li>类别数: {len(self.project.class_names)}</li>
            <li>操作历史: {len(history)} 条</li>
            <li>自动标注会话: {auto_sessions} 次</li>
            <li>自动标注总数: {auto_total_added} 个</li>
            </ul>
            """
            QMessageBox.information(self, "导出成功", summary)
            self.statusBar().showMessage(f"日志已导出到: {file_path}", 5000)

        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"导出日志时出错:\n{str(e)}")

    def _on_project_dirty_changed(self, is_modified: bool):
        """Update window title to reflect dirty state."""
        title = self._product.window_title
        if is_modified:
            title = f"* {title}"
        self.setWindowTitle(title)

    def closeEvent(self, event):
        """Prompt to save before closing."""
        if self.project.is_modified:
            reply = QMessageBox.question(
                self, "未保存的修改",
                "当前项目有未保存的修改，是否保存？",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Save:
                if self.file_ctrl.save_project(parent_widget=self):
                    event.accept()
                else:
                    event.ignore()
            elif reply == QMessageBox.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
