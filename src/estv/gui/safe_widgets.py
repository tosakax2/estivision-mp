# estv/gui/safe_widgets.py

from PySide6.QtWidgets import QComboBox
from PySide6.QtGui import QWheelEvent


class SafeComboBox(QComboBox):

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self.view().isVisible():     # ▼ が開いている？
            super().wheelEvent(event)   # 通常のホイール動作
        else:
            event.ignore()  # 何もしない
