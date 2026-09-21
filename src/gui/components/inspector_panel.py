from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QFormLayout, QComboBox, QPushButton,
    QMessageBox, QSizePolicy, QFrame, QApplication
)
from PySide6.QtCore import Qt, Signal, QItemSelectionModel
from typing import Optional


class InspectorPanel(QWidget):
    """右侧面板 - 属性、标注列表"""

    # Signals
    annotation_selected = Signal(object)  # Annotation or None
    annotation_deleted = Signal(int)  # annotation_id
    class_changed = Signal(int, int)  # annotation_id, new_class
    select_all_requested = Signal()  # request select all in main window
    invert_selection_requested = Signal()  # request invert selection in main window
    batch_delete_requested = Signal(set)  # set of annotation_ids
    multi_selection_changed = Signal(set)  # set of annotation_ids

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_annotation = None
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Properties Section
        layout.addWidget(QLabel("▼ 属性"))
        self.props_form = QWidget()
        props_layout = QFormLayout(self.props_form)
        props_layout.setContentsMargins(0, 0, 0, 0)

        self.file_label = QLabel("-")
        self.split_label = QLabel("-")
        self.size_label = QLabel("-")
        self.count_label = QLabel("-")

        props_layout.addRow("文件:", self.file_label)
        props_layout.addRow("数据集:", self.split_label)
        props_layout.addRow("尺寸:", self.size_label)
        props_layout.addRow("标注数:", self.count_label)

        layout.addWidget(self.props_form)

        # Selected Object Section
        layout.addWidget(QLabel("▼ 已选择"))
        self.selected_widget = QWidget()
        selected_layout = QFormLayout(self.selected_widget)
        selected_layout.setContentsMargins(0, 0, 0, 0)

        self.class_combo = QComboBox()
        self.class_combo.currentIndexChanged.connect(self._on_class_changed)

        self.conf_label = QLabel("-")
        self.bbox_label = QLabel("-")

        selected_layout.addRow("类别:", self.class_combo)
        selected_layout.addRow("置信度:", self.conf_label)
        selected_layout.addRow("边界框:", self.bbox_label)

        layout.addWidget(self.selected_widget)

        # Annotations List Section
        layout.addWidget(QLabel("▼ 标注列表"))
        self.annotation_list = QListWidget()
        self.annotation_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.annotation_list.itemClicked.connect(self._on_list_item_clicked)
        self.annotation_list.itemDoubleClicked.connect(self._on_list_item_double_clicked)
        layout.addWidget(self.annotation_list)

        # ── Dataset Split Stats ──────────────────────────────────────────────
        split_header = QLabel("▼ 数据集划分")
        layout.addWidget(split_header)

        split_stats_frame = QFrame()
        split_stats_frame.setFrameShape(QFrame.StyledPanel)
        split_stats_layout = QHBoxLayout(split_stats_frame)
        split_stats_layout.setContentsMargins(4, 2, 4, 2)
        split_stats_layout.setSpacing(0)

        # Train cell
        self._split_train_label = QLabel("Train: -")
        self._split_train_label.setAlignment(Qt.AlignCenter)
        self._split_train_label.setStyleSheet(
            "color: #4CAF50; font-weight: bold; padding: 2px 6px;"
        )
        split_stats_layout.addWidget(self._split_train_label)

        # separator
        sep1 = QLabel("|")
        sep1.setAlignment(Qt.AlignCenter)
        sep1.setStyleSheet("color: #555; padding: 0 2px;")
        split_stats_layout.addWidget(sep1)

        # Val cell
        self._split_val_label = QLabel("Val: -")
        self._split_val_label.setAlignment(Qt.AlignCenter)
        self._split_val_label.setStyleSheet(
            "color: #2196F3; font-weight: bold; padding: 2px 6px;"
        )
        split_stats_layout.addWidget(self._split_val_label)

        # separator
        sep2 = QLabel("|")
        sep2.setAlignment(Qt.AlignCenter)
        sep2.setStyleSheet("color: #555; padding: 0 2px;")
        split_stats_layout.addWidget(sep2)

        # Test cell
        self._split_test_label = QLabel("Test: -")
        self._split_test_label.setAlignment(Qt.AlignCenter)
        self._split_test_label.setStyleSheet(
            "color: #FF9800; font-weight: bold; padding: 2px 6px;"
        )
        split_stats_layout.addWidget(self._split_test_label)

        layout.addWidget(split_stats_frame)
        self._split_stats_frame = split_stats_frame
        # ────────────────────────────────────────────────────────────────────

        # List Actions - 包含全选、反选、删除按钮，大小相同
        list_action_layout = QHBoxLayout()
        list_action_layout.setSpacing(4)
        
        self.select_all_btn = QPushButton("全选")
        self.select_all_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.select_all_btn.clicked.connect(self._on_select_all)
        
        self.invert_btn = QPushButton("反选")
        self.invert_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.invert_btn.clicked.connect(self._on_invert)
        
        self.delete_btn = QPushButton("删除")
        self.delete_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.delete_btn.clicked.connect(self._on_delete_clicked)
        self.delete_btn.setEnabled(True)
        
        list_action_layout.addWidget(self.select_all_btn)
        list_action_layout.addWidget(self.invert_btn)
        list_action_layout.addWidget(self.delete_btn)
        layout.addLayout(list_action_layout)

    def _connect_signals(self):
        """连接信号"""
        # List selection changed
        self.annotation_list.itemSelectionChanged.connect(self._on_list_selection_changed)

    def set_class_names(self, class_names: dict):
        """设置类别名称列表"""
        self.class_combo.clear()
        for class_id, name in class_names.items():
            self.class_combo.addItem(name, class_id)

    def set_class_options(self, class_names: list):
        """设置类别选项"""
        self.class_combo.clear()
        for idx, name in enumerate(class_names):
            self.class_combo.addItem(name, idx)

    def update_properties(self, file_info: dict):
        """更新属性显示"""
        self.file_label.setText(file_info.get('file', '-'))
        self.split_label.setText(file_info.get('split', '-'))
        self.size_label.setText(file_info.get('size', '-'))
        self.count_label.setText(str(file_info.get('count', 0)))

    def update_annotation_list(self, annotations: list, class_names: dict = None):
        """更新标注列表"""
        self.annotation_list.clear()
        for ann in annotations:
            if class_names:
                class_name = class_names.get(ann.class_id, f"Class {ann.class_id}")
            else:
                class_name = f"Class {ann.class_id}"

            try:
                if hasattr(ann, 'bbox') and ann.bbox is not None:
                    x, y, w, h = ann.bbox
                else:
                    x = y = w = h = 0.0
            except (TypeError, ValueError) as e:
                # Handle malformed bbox
                x = y = w = h = 0.0

            item_text = f"{class_name} [{ann.id}] ({x:.2f}, {y:.2f}, {w:.2f}, {h:.2f})"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, ann)
            self.annotation_list.addItem(item)

    def set_selected_annotation(self, annotation):
        """设置选中的标注"""
        self._current_annotation = annotation
        # 删除按钮始终启用（如果列表有选中项或画布有选中）
        # 由外部控制启用状态
        self.delete_btn.setEnabled(annotation is not None or len(self.annotation_list.selectedItems()) > 0 or bool(self.annotation_list.count()))

        if annotation is None:
            self.conf_label.setText("-")
            self.bbox_label.setText("-")
            # Clear list selection
            self.annotation_list.clearSelection()
            return

        # Update selected info
        self.conf_label.setText(f"{annotation.confidence:.3f}")
        try:
            if hasattr(annotation, 'bbox') and annotation.bbox is not None:
                x, y, w, h = annotation.bbox
                self.bbox_label.setText(f"({x:.3f}, {y:.3f}, {w:.3f}, {h:.3f})")
            else:
                self.bbox_label.setText("-")
        except (TypeError, ValueError):
            self.bbox_label.setText("(invalid)")

        # Update class combo - handle missing class_id gracefully
        class_id = annotation.class_id
        class_found = False
        self.class_combo.blockSignals(True)
        try:
            for i in range(self.class_combo.count()):
                if self.class_combo.itemData(i) == class_id:
                    self.class_combo.setCurrentIndex(i)
                    class_found = True
                    break

            # If class not found in combo, add it temporarily
            if not class_found:
                self.class_combo.addItem(f"Class {class_id}", class_id)
                self.class_combo.setCurrentIndex(self.class_combo.count() - 1)
        finally:
            self.class_combo.blockSignals(False)

        # Highlight in list
        for i in range(self.annotation_list.count()):
            item = self.annotation_list.item(i)
            ann = item.data(Qt.UserRole)
            if ann and ann.id == annotation.id:
                self.annotation_list.setCurrentItem(item, QItemSelectionModel.NoUpdate)
                break

    def _on_list_item_clicked(self, item: QListWidgetItem):
        """列表项点击"""
        ann = item.data(Qt.UserRole)
        if ann:
            # 检查是否按住 Shift 或 Ctrl
            modifiers = QApplication.keyboardModifiers()
            if modifiers & (Qt.ShiftModifier | Qt.ControlModifier):
                # 多选模式：收集所有选中项 ID，发送 multi_selection_changed
                self._emit_multi_selection()
            else:
                # 单选模式：发送 annotation_selected
                self.annotation_selected.emit(ann)
    
    def _on_list_item_double_clicked(self, item: QListWidgetItem):
        """列表项双击 — 切换选中状态并同步"""
        if item.isSelected():
            item.setSelected(False)
        else:
            item.setSelected(True)
        self._emit_multi_selection()
    
    def _on_select_all(self):
        """全选按钮点击"""
        self.select_all_requested.emit()
    
    def _on_invert(self):
        """反选按钮点击"""
        self.invert_selection_requested.emit()

    def clear_detail_view(self):
        """清空详情面板，不改变列表选择状态（用于多选/全选场景）"""
        self._current_annotation = None
        self.conf_label.setText('-')
        self.bbox_label.setText('-')
        self.class_combo.blockSignals(True)
        self.class_combo.setCurrentIndex(-1)
        self.class_combo.blockSignals(False)

    def sync_list_selection(self, selected_ids: set):
        """同步列表选中状态（从 AnnotationState 到 QListWidget）"""
        self.annotation_list.blockSignals(True)
        try:
            for i in range(self.annotation_list.count()):
                item = self.annotation_list.item(i)
                ann = item.data(Qt.UserRole)
                item.setSelected(ann is not None and ann.id in selected_ids)
        finally:
            self.annotation_list.blockSignals(False)

    def _on_list_selection_changed(self):
        """列表选择改变 — 多选时更新当前标注但不改变 AnnotationState"""
        items = self.annotation_list.selectedItems()
        if items:
            ann = items[-1].data(Qt.UserRole)
            if ann:
                self.set_selected_annotation(ann)
        else:
            self.set_selected_annotation(None)

    def _on_class_changed(self, index: int):
        """类别改变"""
        if self._current_annotation is None or index < 0:
            return

        new_class_id = self.class_combo.itemData(index)
        if new_class_id is not None and new_class_id != self._current_annotation.class_id:
            self._current_annotation.class_id = new_class_id
            self.class_changed.emit(self._current_annotation.id, new_class_id)

    def _on_delete_clicked(self):
        """删除按钮点击 - 支持列表多选批量删除"""
        selected_items = self.annotation_list.selectedItems()
        if selected_items:
            # 收集所有选中项 ID，批量删除
            ids = set()
            for item in selected_items:
                ann = item.data(Qt.UserRole)
                if ann:
                    ids.add(ann.id)
            if ids:
                self.batch_delete_requested.emit(ids)
            self.annotation_list.clearSelection()
        elif self._current_annotation is not None:
            # 从画布选中的单个标注删除
            self.annotation_deleted.emit(self._current_annotation.id)
        self.set_selected_annotation(None)

    def _emit_multi_selection(self):
        """发送多选变更信号"""
        ids = set()
        for item in self.annotation_list.selectedItems():
            ann = item.data(Qt.UserRole)
            if ann:
                ids.add(ann.id)
        self.multi_selection_changed.emit(ids)

    def update_split_stats(self, train: int, val: int, test: int):
        """更新数据集划分统计显示
        
        Args:
            train: train 集图片数（-1 表示未设置）
            val:   val   集图片数
            test:  test  集图片数
        """
        if train < 0:
            self._split_train_label.setText("Train: -")
            self._split_val_label.setText("Val: -")
            self._split_test_label.setText("Test: -")
        else:
            self._split_train_label.setText(f"Train: {train} 张")
            self._split_val_label.setText(f"Val: {val} 张")
            self._split_test_label.setText(f"Test: {test} 张")

    def clear(self):
        """清空显示"""
        self.file_label.setText("-")
        self.split_label.setText("-")
        self.size_label.setText("-")
        self.count_label.setText("0")
        self.annotation_list.clear()
        self.set_selected_annotation(None)
