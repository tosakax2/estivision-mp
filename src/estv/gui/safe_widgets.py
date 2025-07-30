# estv/gui/safe_widgets.py

from PySide6.QtWidgets import QComboBox
from PySide6.QtGui import QWheelEvent


class SafeComboBox(QComboBox):
    """ドロップダウンが開いていないときはホイールイベントを無視する ComboBox。"""

    def wheelEvent(self, event: QWheelEvent) -> None:
        """ドロップダウン表示時のみ既定動作。未表示なら無視。"""
        if self.view().isVisible():     # ▼ が開いている？
            super().wheelEvent(event)   # 通常のホイール動作
        else:
            event.ignore()  # 何もしない
