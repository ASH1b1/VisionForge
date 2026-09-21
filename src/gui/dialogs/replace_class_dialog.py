"""Batch replace class labels dialog."""
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel,
    QComboBox, QLineEdit, QRadioButton, QButtonGroup,
    QDialogButtonBox, QGroupBox
)
from PySide6.QtCore import Qt


class ReplaceClassDialog(QDialog):
    """Dialog to batch-replace annotation class labels."""

    ALL_CLASSES_KEY = -1  # sentinel for "-- All Classes --" option

    def __init__(self, class_names: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("批量替换类别标签")
        self.setMinimumWidth(380)
        self._class_names = class_names
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Info label
        info = QLabel(f"共 <b>{len(self._class_names)}</b> 个已注册类别")
        info.setWordWrap(True)
        layout.addWidget(info)

        # Source class selection
        source_group = QGroupBox("源类别 (替换来源)")
        source_layout = QVBoxLayout(source_group)
        self._source_combo = QComboBox()
        self._source_combo.addItem("-- 所有类别 --", self.ALL_CLASSES_KEY)
        for class_id, name in sorted(self._class_names.items()):
            self._source_combo.addItem(f"{name}  (ID: {class_id})", class_id)
        source_layout.addWidget(self._source_combo)
        layout.addWidget(source_group)

        # Target class input
        target_group = QGroupBox("目标类别 (替换为)")
        target_layout = QVBoxLayout(target_group)
        self._target_input = QLineEdit()
        self._target_input.setPlaceholderText("输入目标类别名称...")
        target_layout.addWidget(self._target_input)
        layout.addWidget(target_group)

        # Scope selection
        scope_group = QGroupBox("替换范围")
        scope_layout = QVBoxLayout(scope_group)
        self._scope_group = QButtonGroup(self)
        self._current_image_radio = QRadioButton("仅当前图片")
        self._all_images_radio = QRadioButton("所有图片 (全局)")
        self._current_image_radio.setChecked(True)
        self._scope_group.addButton(self._current_image_radio, 0)
        self._scope_group.addButton(self._all_images_radio, 1)
        scope_layout.addWidget(self._current_image_radio)
        scope_layout.addWidget(self._all_images_radio)
        layout.addWidget(scope_group)

        layout.addStretch()

        # OK/Cancel
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    # --- Public getters ---
    def get_source_class_id(self):
        """Return source class_id, or ALL_CLASSES_KEY (-1) for all classes."""
        return self._source_combo.currentData()

    def get_target_class_name(self):
        """Return the target class name string, stripped."""
        return self._target_input.text().strip()

    def is_current_image_only(self):
        """True if scope is current image, False for all images."""
        return self._current_image_radio.isChecked()
