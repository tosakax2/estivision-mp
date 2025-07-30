# estv/__main__.py

import sys

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont
import qdarkstyle

from estv.gui.main_window import MainWindow


def main() -> None:
    """ESTiVision アプリケーションを起動するエントリーポイント。"""
    # --- QApplicationの初期化
    app: QApplication = QApplication(sys.argv)

    # --- スタイルシートの適用
    style: str = qdarkstyle.load_stylesheet()
    app.setStyleSheet(style)

    # --- フォントの設定
    font: QFont = QFont("Arial", 10)
    app.setFont(font)

    # --- メインウィンドウの作成と表示
    window: MainWindow = MainWindow()
    window.show()

    # --- アプリケーションの実行
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
