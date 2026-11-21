from pyllama.engine.device import detect_devices, select_device


def test_device_detection_shape():
    d = detect_devices()
    assert isinstance(d.cpu_available, bool)
    assert isinstance(d.gpu_available, bool)
    assert d.cpu_threads >= 1
    assert d.auto_device in {"cpu", "cuda"}


def test_select_cpu():
    assert select_device("cpu") == "cpu"
