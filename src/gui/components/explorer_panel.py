from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTreeWidget, QTreeWidgetItem, QListWidget, QListWidgetItem,
    QPushButton, QCheckBox, QFrame, QInputDialog, QMessageBox, QMenu
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon


class ExplorerPanel(QWidget):
    """左侧面板 - 文件树、类别、工具"""

    # Signals
    file_selected = Signal(object)  # Path
    class_toggled = Signal(int, bool)  # class_id, visible
    tool_selected = Signal(str)  # tool_name
    class_added = Signal(str)  # class_name
    class_renamed = Signal(int, str)  # class_id, new_name
    class_deleted = Signal(int)  # class_id
    class_selected = Signal(int)  # class_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self._class_names = {}  # class_id -> class_name
        self._next_class_id = 0
        self._selected_class_id = None
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # File Tree Section
        layout.addWidget(QLabel("📁 资源管理器"))
        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderHidden(True)
        self.file_tree.setIndentation(12)
        layout.addWidget(self.file_tree)

        separator1 = QFrame()
        separator1.setFrameShape(QFrame.HLine)
        separator1.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator1)

        # Classes Section
        layout.addWidget(QLabel("🎨 类别"))
        self.class_list = QListWidget()
        self.class_list.setMaximumHeight(150)
        self.class_list.setSelectionMode(QListWidget.SingleSelection)
        layout.addWidget(self.class_list)

        # Class buttons
        class_buttons = QHBoxLayout()
        self.add_class_btn = QPushButton("➕ 新建")
        self.add_class_btn.setMinimumWidth(65)
        self.add_class_btn.setToolTip("添加新类别")
        self.remove_class_btn = QPushButton("➖ 删除")
        self.remove_class_btn.setMinimumWidth(65)
        self.remove_class_btn.setToolTip("删除选中类别")
        self.rename_class_btn = QPushButton("✎ 重命名")
        self.rename_class_btn.setMinimumWidth(75)
        self.rename_class_btn.setToolTip("重命名选中类别")

        class_buttons.addWidget(self.add_class_btn)
        class_buttons.addWidget(self.rename_class_btn)
        class_buttons.addWidget(self.remove_class_btn)
        class_buttons.addStretch()
        layout.addLayout(class_buttons)

        # Current class indicator
        self.current_class_label = QLabel("当前: 无")
        self.current_class_label.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(self.current_class_label)

        separator2 = QFrame()
        separator2.setFrameShape(QFrame.HLine)
        separator2.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator2)

        # Tools Section
        layout.addWidget(QLabel("🔧 工具"))
        self.tool_list = QListWidget()
        tools = [
            ("✋", "选择", "V", "select"),
            ("⬛", "矩形标注", "R", "rectangle"),
            ("🔷", "多边形标注", "P", "polygon"),
            ("✂️", "删除标注", "E", "eraser"),
            ("✦", "SAM 点选", "A", "sam_click"),
        ]
        for icon, name, shortcut, tool_name in tools:
            item = QListWidgetItem(f"{icon} {name} ({shortcut})")
            item.setData(Qt.UserRole, tool_name)
            self.tool_list.addItem(item)
        layout.addWidget(self.tool_list)

        # History Section
        separator3 = QFrame()
        separator3.setFrameShape(QFrame.HLine)
        separator3.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator3)

        layout.addWidget(QLabel("历史记录"))
        self.history_list = QListWidget()
        self.history_list.setMaximumHeight(100)
        layout.addWidget(self.history_list)

        layout.addStretch()

    def _connect_signals(self):
        """连接信号"""
        self.add_class_btn.clicked.connect(self._add_class)
        self.remove_class_btn.clicked.connect(self._remove_class)
        self.rename_class_btn.clicked.connect(self._rename_class)
        self.class_list.itemSelectionChanged.connect(self._on_class_selection_changed)
        self.class_list.itemClicked.connect(self._on_class_item_clicked)
        self.class_list.itemChanged.connect(self._on_class_item_changed)
        self.tool_list.itemClicked.connect(self._on_tool_item_clicked)

    # ==================== Class Management ====================

    def set_classes(self, class_names: dict):
        """设置类别列表"""
        self._class_names = class_names.copy()
        self._refresh_class_list()
        if class_names:
            self._next_class_id = max(class_names.keys()) + 1
        else:
            self._next_class_id = 0

    def get_classes(self) -> dict:
        """获取类别列表"""
        return self._class_names.copy()

    def add_class(self, name: str, class_id: int = None) -> int:
        """添加类别"""
        if class_id is None:
            class_id = self._next_class_id
            self._next_class_id += 1

        self._class_names[class_id] = name
        self._refresh_class_list()
        return class_id

    def remove_class(self, class_id: int):
        """删除类别"""
        if class_id in self._class_names:
            del self._class_names[class_id]
            self._refresh_class_list()

    def rename_class(self, class_id: int, new_name: str):
        """重命名类别"""
        if class_id in self._class_names:
            self._class_names[class_id] = new_name
            self._refresh_class_list()

    def set_selected_class(self, class_id: int):
        """设置选中的类别"""
        self._selected_class_id = class_id
        for i in range(self.class_list.count()):
            item = self.class_list.item(i)
            if item.data(Qt.UserRole) == class_id:
                self.class_list.setCurrentItem(item)
                break

        if class_id is not None and class_id in self._class_names:
            self.current_class_label.setText(f"当前: {self._class_names[class_id]}")
        else:
            self.current_class_label.setText("当前: 无")

    def get_selected_class_id(self) -> int:
        """获取选中的类别 ID"""
        return self._selected_class_id

    def set_selected_tool(self, tool_name: str):
        """同步当前选中的工具"""
        for i in range(self.tool_list.count()):
            item = self.tool_list.item(i)
            if item.data(Qt.UserRole) == tool_name:
                self.tool_list.setCurrentItem(item)
                return

        self.tool_list.clearSelection()

    def _refresh_class_list(self):
        """刷新类别列表"""
        self.class_list.clear()
        for class_id, name in sorted(self._class_names.items()):
            item = QListWidgetItem(f"{self._get_class_icon(class_id)} {name}")
            item.setData(Qt.UserRole, class_id)
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            self.class_list.addItem(item)

    def _get_class_icon(self, class_id: int) -> str:
        """获取类别图标 (彩色圆点)"""
        # YOLO 风格颜色
        colors = [
            "🔴", "🟢", "🔵", "🟡", "🟣",
            "🟠", "⚫", "🟤", "🔘", "⚪"
        ]
        idx = class_id % len(colors)
        return colors[idx]

    def _add_class(self):
        """添加新类别"""
        class_name, ok = QInputDialog.getText(
            self, "添加类别",
            "输入类别名称:",
            text=f"class_{self._next_class_id}"
        )

        if ok and class_name.strip():
            class_id = self.add_class(class_name.strip())
            self.set_selected_class(class_id)
            self.class_added.emit(class_name.strip())

    def _remove_class(self):
        """删除选中的类别"""
        current_item = self.class_list.currentItem()
        if current_item is None:
            QMessageBox.information(self, "提示", "请选择要删除的类别")
            return

        class_id = current_item.data(Qt.UserRole)
        class_name = self._class_names.get(class_id, "未知")

        reply = QMessageBox.question(
            self, "确认删除",
            f"确定删除类别 '{class_name}'?\n\n该类别下的所有标注将被一并删除。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.remove_class(class_id)
            self.class_deleted.emit(class_id)
            if self._selected_class_id == class_id:
                self._selected_class_id = None
                self.current_class_label.setText("当前: 无")

    def _rename_class(self):
        """重命名选中的类别"""
        current_item = self.class_list.currentItem()
        if current_item is None:
            QMessageBox.information(self, "提示", "请选择要重命名的类别")
            return

        class_id = current_item.data(Qt.UserRole)
        old_name = self._class_names.get(class_id, "未知")

        new_name, ok = QInputDialog.getText(
            self, "重命名类别",
            "输入新的类别名称:",
            text=old_name
        )

        if ok and new_name.strip() and new_name.strip() != old_name:
            self.rename_class(class_id, new_name.strip())
            self.class_renamed.emit(class_id, new_name.strip())

    def _on_class_selection_changed(self):
        """类别选择改变"""
        current_item = self.class_list.currentItem()
        if current_item:
            class_id = current_item.data(Qt.UserRole)
            self._selected_class_id = class_id
            self.class_selected.emit(class_id)

            if class_id in self._class_names:
                self.current_class_label.setText(f"Current: {self._class_names[class_id]}")
        else:
            self._selected_class_id = None
            self.current_class_label.setText("当前: 无")

    def _on_class_item_clicked(self, item: QListWidgetItem):
        """类别项点击"""
        class_id = item.data(Qt.UserRole)
        self._selected_class_id = class_id
        self.class_selected.emit(class_id)

    def _on_class_item_changed(self, item: QListWidgetItem):
        """类别项编辑完成 (内联重命名)"""
        class_id = item.data(Qt.UserRole)
        text = item.text()
        # Extract name from icon + name format
        if " " in text:
            new_name = text.split(" ", 1)[1].strip()
        else:
            new_name = text.strip()

        if new_name and class_id in self._class_names:
            old_name = self._class_names[class_id]
            if new_name != old_name:
                self._class_names[class_id] = new_name
                self.class_renamed.emit(class_id, new_name)
                # Restore icon format with new name
                item.setText(f"{self._get_class_icon(class_id)} {new_name}")
        elif class_id in self._class_names:
            # If new_name is empty, restore the original name with icon
            original_name = self._class_names[class_id]
            item.setText(f"{self._get_class_icon(class_id)} {original_name}")
        else:
            # Fallback: restore icon format with empty/new class
            item.setText(f"{self._get_class_icon(class_id)} class_{class_id}")

    def _on_tool_item_clicked(self, item: QListWidgetItem):
        """工具项点击"""
        tool_name = item.data(Qt.UserRole)
        if tool_name:
            self.tool_selected.emit(tool_name)

    # ==================== Context Menu ====================

    def contextMenuEvent(self, event):
        """右键菜单"""
        item = self.class_list.itemAt(self.class_list.mapFromSelf(event.pos()))
        if item is not None:
            menu = QMenu(self)
            menu.addAction("添加类别", self._add_class)
            menu.addAction("重命名类别", self._rename_class)
            menu.addAction("删除类别", self._remove_class)
            menu.addSeparator()
            menu.addAction("设为当前类别", lambda: self._on_class_item_clicked(item))
            menu.exec(event.globalPos())
        else:
            menu = QMenu(self)
            menu.addAction("添加类别", self._add_class)
            menu.exec(event.globalPos())

    # ==================== History ====================

    def update_history(self, history: list, current_index: int):
        """更新历史记录列表"""
        self.history_list.clear()
        for idx, action in enumerate(reversed(history)):
            # Calculate actual index
            actual_idx = len(history) - 1 - idx

            # Display with indicator for current position
            prefix = "→ " if actual_idx == current_index else "  "
            item_text = f"{prefix}{action.description}"
            item = QListWidgetItem(item_text)

            # Style the current item
            if actual_idx == current_index:
                item.setForeground(Qt.gray)
            elif actual_idx > current_index:
                item.setForeground(Qt.darkGray)

            self.history_list.addItem(item)
