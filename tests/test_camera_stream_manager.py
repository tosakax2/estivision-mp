import warnings

import estv.devices.camera_stream_manager as mgr_module
from estv.devices.camera_stream_manager import CameraStreamManager


class DummySignal:
    def connect(self, *args, **kwargs):
        pass

    def emit(self, *args, **kwargs):
        pass


class DummyStream:
    def __init__(self, device_index):
        self.frame_ready = DummySignal()
        self.q_image_ready = DummySignal()
        self.error = DummySignal()
        self.finished = DummySignal()

    def start(self):
        pass

    def stop(self):
        pass

    def wait(self):
        pass

    def deleteLater(self):
        pass


def test_cannot_start_more_than_three_cameras(monkeypatch):
    # Replace CameraStream with dummy implementation to avoid hardware access
    monkeypatch.setattr(mgr_module, "CameraStream", DummyStream)

    # Simple lookup that returns integer index for given id
    manager = CameraStreamManager(lambda cid: int(cid))

    manager.start_camera("0")
    manager.start_camera("1")

    assert len(manager.running_device_ids()) == 2

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        manager.start_camera("2")
        assert len(manager.running_device_ids()) == 2
        assert any("Cannot start" in str(warn.message) for warn in w)
