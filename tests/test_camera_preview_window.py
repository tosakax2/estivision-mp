import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from estv.gui.camera_preview_window import CameraPreviewWindow


class DummySignal:
    def connect(self, *args, **kwargs):
        pass

    def disconnect(self, *args, **kwargs):
        pass


class DummyCameraStreamManager:
    def __init__(self):
        self.q_image_ready = DummySignal()
        self.frame_ready = DummySignal()

    def set_exposure(self, device_id, value):
        pass

    def set_brightness(self, device_id, value):
        pass

    def stop_camera(self, device_id):
        pass


def test_calib_button_disabled_during_pose_estimation():
    app = QApplication.instance() or QApplication([])
    manager = DummyCameraStreamManager()
    window = CameraPreviewWindow(manager, device_id="0")

    assert window.calib_button.isEnabled()
    window.set_pose_estimation_enabled(True)
    app.processEvents()
    assert not window.calib_button.isEnabled()

    window.set_pose_estimation_enabled(False)
    app.processEvents()
    assert window.calib_button.isEnabled()

    window.close()
