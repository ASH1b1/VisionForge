"""数据集划分配置对话框"""
import random
from typing import Dict, Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSpinBox, QCheckBox, QDialogButtonBox,
    QGroupBox, QSizePolicy
)
from PySide6.QtCore import Qt

from ..utils.split_utils import allocate_split_counts


class SplitConfigDialog(QDialog):
    """导出数据集时弹出，让用户配置 train/val/test 划分比例"""

    def __init__(self, total_count: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("数据集划分配置")
        self.setMinimumWidth(360)
        self._setup_ui(total_count)

    def _setup_ui(self, total_count: int):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # 信息标签
        info_label = QLabel(f"共 <b>{total_count}</b> 张图片，请设置划分比例：")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # 划分配置分组
        group = QGroupBox("划分比例")
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(8)

        # 预设按钮
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("预设:"))
        for label, values in [
            ("8:1:1", (80, 10, 10)),
            ("8:2:0", (80, 20, 0)),
            ("7:2:1", (70, 20, 10)),
            ("全部Train", (100, 0, 0)),
        ]:
            btn = QPushButton(label)
            btn.setFixedHeight(24)
            btn.clicked.connect(lambda _=False, v=values: self._apply_preset(*v))
            preset_row.addWidget(btn)
        preset_row.addStretch()
        group_layout.addLayout(preset_row)

        # 比例微调
        ratio_row = QHBoxLayout()
        ratio_row.addWidget(QLabel("Train:"))
        self._train_spin = QSpinBox()
        self._train_spin.setRange(0, 100)
        self._train_spin.setValue(80)
        self._train_spin.setSuffix("%")
        self._train_spin.setFixedWidth(80)
        self._train_spin.valueChanged.connect(self._on_value_changed)
        ratio_row.addWidget(self._train_spin)

        ratio_row.addWidget(QLabel("Val:"))
        self._val_spin = QSpinBox()
        self._val_spin.setRange(0, 100)
        self._val_spin.setValue(10)
        self._val_spin.setSuffix("%")
        self._val_spin.setFixedWidth(80)
        self._val_spin.valueChanged.connect(self._on_value_changed)
        ratio_row.addWidget(self._val_spin)

        ratio_row.addWidget(QLabel("Test:"))
        self._test_spin = QSpinBox()
        self._test_spin.setRange(0, 100)
        self._test_spin.setValue(10)
        self._test_spin.setSuffix("%")
        self._test_spin.setFixedWidth(80)
        self._test_spin.valueChanged.connect(self._on_value_changed)
        ratio_row.addWidget(self._test_spin)

        group_layout.addLayout(ratio_row)

        # 总和提示
        self._sum_label = QLabel("= 100% ✓")
        self._sum_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        group_layout.addWidget(self._sum_label)

        layout.addWidget(group)

        # 固定随机种子
        self._fixed_seed = QCheckBox("固定随机种子 (可复现)")
        self._fixed_seed.setChecked(True)
        layout.addWidget(self._fixed_seed)

        layout.addStretch()

        # 确定/取消按钮
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        self._ok_button = button_box.button(QDialogButtonBox.Ok)
        layout.addWidget(button_box)

    def _apply_preset(self, train: int, val: int, test: int):
        self._train_spin.blockSignals(True)
        self._val_spin.blockSignals(True)
        self._test_spin.blockSignals(True)
        self._train_spin.setValue(train)
        self._val_spin.setValue(val)
        self._test_spin.setValue(test)
        self._train_spin.blockSignals(False)
        self._val_spin.blockSignals(False)
        self._test_spin.blockSignals(False)
        self._on_value_changed()

    def _on_value_changed(self):
        total = (
            self._train_spin.value()
            + self._val_spin.value()
            + self._test_spin.value()
        )
        valid = total == 100
        if valid:
            self._sum_label.setText("= 100% ✓")
            self._sum_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            self._sum_label.setText(f"= {total}% ✗")
            self._sum_label.setStyleSheet("color: #f44336; font-weight: bold;")
        self._ok_button.setEnabled(valid)

    def compute_split(self, n: int) -> Dict[int, str]:
        """根据设定比例计算划分，返回 {index: "train"/"val"/"test"}"""
        if n == 0:
            return {}

        train_pct = self._train_spin.value() / 100.0
        val_pct = self._val_spin.value() / 100.0
        test_pct = self._test_spin.value() / 100.0

        indices = list(range(n))
        seed = 42 if self._fixed_seed.isChecked() else None
        rng = random.Random(seed)
        rng.shuffle(indices)

        n_train, n_val, _n_test = allocate_split_counts(n, train_pct, val_pct, test_pct)
        train_end = n_train
        val_end = n_train + n_val

        result: Dict[int, str] = {}
        for pos, idx in enumerate(indices):
            if pos < train_end:
                result[idx] = "train"
            elif pos < val_end:
                result[idx] = "val"
            else:
                result[idx] = "test"
        return result
