# estv/gui/safe_widgets.py
"""ユーザー操作による誤動作を防ぐカスタムウィジェット群。"""

from PySide6.QtWidgets import QComboBox
from PySide6.QtGui import QWheelEvent


class SafeComboBox(QComboBox):
    """閉じている状態ではホイール回転を無視する ``QComboBox``。"""

    def wheelEvent(self, event: QWheelEvent) -> None:
        """ドロップダウンが開いていないときはホイールを無視する。"""
        if self.view().isVisible():     # ▼ が開いている？
            super().wheelEvent(event)   # 通常のホイール動作
        else:
            event.ignore()  # 何もしない
