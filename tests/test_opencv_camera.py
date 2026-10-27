"""The OpenCV camera driver and its plugin, without a webcam: a video file read by the real OpenCV, and a fake of
``cv2.VideoCapture`` for what a file cannot do (properties, failing reads, change of the frame size)"""
import importlib.util
import time

import cv2
import numpy as np
import pytest

try:
    available = (importlib.util.find_spec('pymodaq.utils.plugin_testing') is not None and
                 importlib.util.find_spec('pymodaq_plugins_utils.hardware.camera_base') is not None)
except ModuleNotFoundError:
    available = False
if not available:
    pytest.skip('Needs PyMoDAQ >= 5.3.2 and pymodaq_plugins_utils >= 5.3.1', allow_module_level=True)

from pymodaq.utils.plugin_testing import make_detector, grab_and_wait, wait_for_signal

try:
    from pymodaq_gui.plotting.items.roi import RoiInfo
except ImportError:
    from pymodaq_gui.plotting.utils.plot_utils import RoiInfo

from pymodaq_plugins_utils.hardware.camera_driver import CameraDriverBase

from pymodaq_plugins_opencv.hardware.opencv_camera import OpenCVCamera
from pymodaq_plugins_opencv.daq_viewer_plugins.plugins_2D.daq_2Dviewer_OpenCV import DAQ_2DViewer_OpenCV

WIDTH, HEIGHT = 64, 48
LEFT_RGB, RIGHT_RGB = (200, 100, 50), (20, 60, 220)  # colours of the two halves of the images


def make_frame(width=WIDTH, height=HEIGHT) -> np.ndarray:
    """What OpenCV gives: BGR"""
    frame = np.zeros((height, width, 3), np.uint8)
    frame[:, :width // 2] = LEFT_RGB[::-1]
    frame[:, width // 2:] = RIGHT_RGB[::-1]
    return frame


class FakeCapture:
    """Stands for ``cv2.VideoCapture``, with a few properties"""
    fail = False  # the read gives nothing, as an unplugged camera

    def __init__(self, source=0, *args):
        self.opened = source != 'missing'
        self.values = {cv2.CAP_PROP_FRAME_WIDTH: WIDTH, cv2.CAP_PROP_FRAME_HEIGHT: HEIGHT,
                       cv2.CAP_PROP_BRIGHTNESS: 128., cv2.CAP_PROP_GAIN: 10.}

    def isOpened(self):
        return self.opened

    def open(self, source):
        self.opened = True
        return True

    def read(self):
        time.sleep(0.005)  # a webcam gives a frame every few ms at most
        if self.fail:
            return False, None
        return True, make_frame(int(self.values[cv2.CAP_PROP_FRAME_WIDTH]), int(self.values[cv2.CAP_PROP_FRAME_HEIGHT]))

    def get(self, prop):
        return self.values.get(prop, -1.)

    def set(self, prop, value):
        if prop in self.values:
            self.values[prop] = value
            return True
        return False

    def getBackendName(self):
        return 'FAKE'

    def release(self):
        self.opened = False


@pytest.fixture
def fake_capture(monkeypatch):
    monkeypatch.setattr(cv2, 'VideoCapture', FakeCapture)


@pytest.fixture
def video(tmp_path):
    """A video file of the same images, read by the real OpenCV"""
    path = str(tmp_path / 'camera.avi')
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), 30., (WIDTH, HEIGHT))
    if not writer.isOpened():
        pytest.skip('This OpenCV cannot write video files')
    for _ in range(100):
        writer.write(make_frame())
    writer.release()
    return path


@pytest.fixture
def driver(fake_capture):
    camera = OpenCVCamera(0)
    yield camera
    camera.close()


@pytest.fixture
def camera(qapp, fake_capture):
    plugin = make_detector(DAQ_2DViewer_OpenCV)
    yield plugin
    plugin.close()


def acquire(driver: OpenCVCamera, nframes=1) -> np.ndarray:
    driver.setup_acquisition()
    driver.start_acquisition()
    assert driver.wait_for_frame(nframes=nframes, timeout=2)
    return driver.read_multiple_images()


def click_led(plugin, name: str):
    """What the user does: press the (checkable) button of the parameter, which the DAQ_Viewer sends to the plugin"""
    param = plugin.settings.child('roi', name)
    param.opts['checked'] = True
    param.activate()


def apply_roi(plugin, rows: slice, columns: slice):
    plugin.roi_select(RoiInfo.from_slices([rows, columns]))
    click_led(plugin, 'update_roi')
    click_led(plugin, 'apply_roi')


class TestVideoFile:
    """The driver with the real OpenCV"""
    def test_frames_are_rgb(self, video):
        camera = OpenCVCamera(video)
        try:
            assert camera.get_detector_size() == (WIDTH, HEIGHT)
            frames = acquire(camera, 2)
            assert frames.shape == (2, HEIGHT, WIDTH, 3) and frames.dtype == np.uint8
            assert frames[0, :, :WIDTH // 2].mean(axis=(0, 1)) == pytest.approx(LEFT_RGB, abs=8)  # JPEG compression
            assert frames[0, :, WIDTH // 2:].mean(axis=(0, 1)) == pytest.approx(RIGHT_RGB, abs=8)
        finally:
            camera.close()

    def test_a_file_that_cannot_be_opened_is_an_error(self, tmp_path):
        with pytest.raises(IOError):
            OpenCVCamera(str(tmp_path / 'nothing.avi'))


class TestDriver:
    def test_follows_the_driver_contract(self, driver):
        assert isinstance(driver, CameraDriverBase)
        not_implemented = [name for name in dir(CameraDriverBase)
                           if not name.startswith('_') and getattr(OpenCVCamera, name) is getattr(CameraDriverBase, name)
                           and name not in ('get_roi_limits', 'set_frame_format', 'get_pixel_size')]
        assert not_implemented == []

    def test_device_info(self, driver):
        assert driver.get_device_info().name == 'OpenCV camera 0 (FAKE)'

    def test_frames_are_given_once(self, driver):
        assert acquire(driver, 3).shape == (3, HEIGHT, WIDTH, 3)
        assert driver.read_multiple_images().shape[0] == 0
        frames, (first, last) = driver.read_multiple_images(return_rng=True)
        assert last == first

    def test_wait_returns_at_once_when_stopped(self, driver):
        driver.start_acquisition()
        driver.clear_acquisition()
        assert not driver.acquisition_in_progress()
        assert driver.wait_for_frame(timeout=1) is False

    def test_wait_times_out_when_the_camera_gives_nothing(self, driver):
        driver._capture.fail = True
        driver.start_acquisition()
        with pytest.raises(TimeoutError):
            driver.wait_for_frame(timeout=0.05)

    def test_roi(self, driver):
        driver.set_roi(hstart=40, hend=60, vstart=10, vend=30)
        assert driver.get_data_dimensions() == (20, 20)
        frames = acquire(driver)
        assert frames.shape == (1, 20, 20, 3)
        assert frames[0].mean(axis=(0, 1)) == pytest.approx(RIGHT_RGB)

    def test_binning_averages_the_pixels(self, driver):
        driver.set_roi(hstart=0, hend=WIDTH, vstart=0, vend=HEIGHT, hbin=2, vbin=4)
        assert driver.get_data_dimensions() == (12, 32)
        frames = acquire(driver)
        assert frames.shape == (1, 12, 32, 3) and frames.dtype == np.float32
        assert frames[0, 0, 0] == pytest.approx(LEFT_RGB)

    def test_wrong_roi_is_refused(self, driver):
        with pytest.raises(ValueError):
            driver.set_roi(hstart=0, hend=WIDTH + 1)
        with pytest.raises(ValueError):
            driver.set_roi(hstart=0, hend=3, hbin=4)

    def test_roi_change_forgets_the_frames_of_the_previous_one(self, driver):
        acquire(driver)
        driver.set_roi(hstart=0, hend=10, vstart=0, vend=10)
        assert driver.read_multiple_images().shape[0] == 0
        assert driver.read_newest_image() is None

    def test_properties_are_those_the_camera_supports(self, driver):
        properties = driver.get_properties()
        assert properties['PROP_BRIGHTNESS'] == (128, True)
        assert properties['PROP_FRAME_WIDTH'] == (WIDTH, True)
        assert 'PROP_GAIN' in properties
        assert 'PROP_ZOOM' not in properties
        assert not any(name.startswith('PROP_DC1394') for name in properties)

    def test_size_change_resets_the_roi(self, driver):
        driver.set_roi(hstart=10, hend=20, vstart=10, vend=20)
        assert driver.set_property('PROP_FRAME_WIDTH', 32) == 32
        assert driver.get_detector_size() == (32, HEIGHT)
        assert driver.get_roi() == (0, 32, 0, HEIGHT, 1, 1)
        assert acquire(driver).shape == (1, HEIGHT, 32, 3)


class TestPlugin:
    def test_snap_gives_a_gray_image_in_pixels(self, camera):
        dte = grab_and_wait(camera)
        assert dte[0].shape == (HEIGHT, WIDTH)
        assert dte[0][0][0, 0] == pytest.approx(0.299 * LEFT_RGB[0] + 0.587 * LEFT_RGB[1] + 0.114 * LEFT_RGB[2], abs=1)
        assert camera.x_axis.units == '' and camera.x_axis.size == WIDTH
        assert camera.settings['camera_name'] == 'OpenCV camera 0 (FAKE)'

    def test_without_conversion_the_channels_are_given(self, camera):
        camera.settings.child('color_conversion').setValue('None')
        dte = grab_and_wait(camera)
        assert dte[0].labels == ['Red', 'Green', 'Blue']
        assert [data[0, 0] for data in dte[0].data] == list(LEFT_RGB)

    def test_average(self, camera):
        assert grab_and_wait(camera, naverage=3)[0].shape == (HEIGHT, WIDTH)

    def test_settings_of_the_camera_are_listed(self, camera):
        cam_settings = camera.settings.child('cam_settings')
        assert {'PROP_FRAME_WIDTH', 'PROP_BRIGHTNESS', 'PROP_GAIN'} <= set(cam_settings.names)

    def test_exposure_and_buffer_are_not_shown(self, camera):
        assert not camera.settings.child('timing_opts', 'exposure_time').opts['visible']
        assert not camera.settings.child('buffer').opts['visible']

    def test_changing_a_property_is_sent_to_the_camera(self, camera):
        param = camera.settings.child('cam_settings', 'PROP_BRIGHTNESS')
        param.setValue(50)
        camera.commit_settings(param)
        assert camera.controller._capture.values[cv2.CAP_PROP_BRIGHTNESS] == 50

    def test_changing_the_frame_size_updates_the_image_and_the_axes(self, camera):
        param = camera.settings.child('cam_settings', 'PROP_FRAME_WIDTH')
        param.setValue(32)
        camera.commit_settings(param)
        dte = grab_and_wait(camera)
        assert dte[0].shape == (HEIGHT, 32)
        assert camera.x_axis.size == 32
        assert camera.settings['hdet'] == 32

    def test_roi_drawn_in_the_viewer_is_applied(self, camera):
        apply_roi(camera, rows=slice(10, 30), columns=slice(40, 60))
        assert camera.controller.get_roi() == (40, 60, 10, 30, 1, 1)
        dte = grab_and_wait(camera)
        assert dte[0].shape == (20, 20)
        assert camera.x_axis.offset == 40

    def test_binning(self, camera):
        for name in ('x_binning', 'y_binning'):
            camera.settings.child('roi', name).setValue(2)
            camera.commit_settings(camera.settings.child('roi', name))
        assert grab_and_wait(camera)[0].shape == (HEIGHT // 2, WIDTH // 2)
        assert camera.x_axis.scaling == 2

    def test_live_gives_images_until_stopped(self, camera):
        received = []
        camera.dte_signal.connect(received.append)
        wait_for_signal(camera.dte_signal, lambda: camera.grab_data(1, live=True), timeout_ms=3000)
        wait_for_signal(camera.dte_signal, lambda: None, timeout_ms=3000)
        camera.stop()
        assert len(received) >= 2
        assert not camera.controller.acquisition_in_progress()

    def test_roi_can_change_during_the_live_and_the_plugin_still_closes(self, qapp, fake_capture):
        plugin = make_detector(DAQ_2DViewer_OpenCV)
        try:
            wait_for_signal(plugin.dte_signal, lambda: plugin.grab_data(1, live=True), timeout_ms=3000)
            apply_roi(plugin, rows=slice(10, 30), columns=slice(40, 60))
            assert plugin.controller.get_roi() == (40, 60, 10, 30, 1, 1)
        finally:
            plugin.close()

    def test_close_releases_the_camera(self, qapp, fake_capture):
        plugin = make_detector(DAQ_2DViewer_OpenCV)
        capture = plugin.controller._capture
        plugin.close()
        assert not capture.isOpened()
