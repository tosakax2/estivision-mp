import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QWheelEvent
from PySide6.QtCore import QPointF, QPoint, Qt
from estv.gui.safe_widgets import SafeComboBox


def create_event():
    return QWheelEvent(
        QPointF(),
        QPointF(),
        QPoint(),
        QPoint(0, 120),
        Qt.NoButton,
        Qt.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )


def test_wheel_event_ignored_when_closed():
    app = QApplication.instance() or QApplication([])
    combo = SafeComboBox()
    combo.addItems(["1", "2"])
    event = create_event()
    event.accept()
    combo.wheelEvent(event)
    assert not event.isAccepted()
