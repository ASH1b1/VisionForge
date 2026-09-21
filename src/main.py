import os
import sys

# PyInstaller --windowed (console=False) 模式下 sys.stdout / sys.stderr 为 None，
# transformers 依赖的 tqdm 在导入时调用 sys.stderr.isatty()，None 上会报 AttributeError。
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

from pathlib import Path


def _get_project_root() -> Path:
    """获取项目根目录，兼容 PyInstaller 打包（sys._MEIPASS）和源码运行（__file__）。"""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).parent.parent


project_root = _get_project_root()
sys.path.insert(0, str(project_root))

from src.runtime_env import (
    bootstrap_runtime_env,
    is_openmp_duplicate_error,
    openmp_user_message,
)

bootstrap_runtime_env()

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon

from src.product_config import get_product_config
from src.gui.main_window import MainWindow


def main():
    cfg = get_product_config()
    app = QApplication(sys.argv)
    app.setApplicationName(cfg.app_name)
    app.setApplicationVersion(cfg.app_version)
    app.setOrganizationName(cfg.organization)

    # 设置程序 logo
    logo_path = project_root / cfg.logo_filename
    if logo_path.exists():
        app.setWindowIcon(QIcon(str(logo_path)))

    window = MainWindow()
    window.show()

    try:
        sys.exit(app.exec())
    except Exception as e:
        if is_openmp_duplicate_error(str(e)):
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(None, "启动失败", openmp_user_message())
        raise


if __name__ == "__main__":
    main()
