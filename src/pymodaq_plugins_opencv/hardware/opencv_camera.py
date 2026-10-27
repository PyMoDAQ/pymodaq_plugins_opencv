"""Driver of the cameras (webcams, video files) read with OpenCV, for ``CameraBase`` of ``pymodaq_plugins_utils``.

It follows ``pymodaq_plugins_utils.hardware.camera_driver.CameraDriverBase``, the methods being those of the pylablib
cameras. OpenCV cameras are simpler than most:

- there is no buffer to manage: ``VideoCapture.read`` blocks until the next frame, so ``wait_for_frame`` reads it and
  ``read_multiple_images`` gives what was read;
- the ROI and the binning are not done by the camera, the driver applies them to each frame (the sensor is the full
  frame, so it can change during the acquisition);
- the exposure is not set from here: its unit depends on the backend, change it with the properties of the camera
  (see :meth:`OpenCVCamera.get_properties`).

Frames are given as RGB, the order of OpenCV being BGR.
"""
import os
import sys
import threading
import time
from collections import deque
from types import SimpleNamespace
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np

from pymodaq_plugins_utils.hardware.camera_driver import CameraDriverBase

from pymodaq_plugins_opencv.hardware.opencv import OpenCVProp


def _keep_qt_of_the_binding():
    """The opencv-python package (not the headless one) points Qt to the platform plugins it ships, which makes the
    creation of a QApplication abort with the Qt of PyQt/PySide: restore the plugins of the Qt binding"""
    cv2_dir = os.path.dirname(os.path.abspath(cv2.__file__))
    for name in ('QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_QPA_FONTDIR'):
        if os.environ.get(name, '').startswith(cv2_dir):
            del os.environ[name]


_keep_qt_of_the_binding()


class OpenCVCamera(CameraDriverBase):
    """A camera read with ``cv2.VideoCapture``

    Parameters
    ----------
    source: int or str
        Index of the camera, or file (or url) opened by ``cv2.VideoCapture``
    """
    # the properties that change the size of the frame
    size_properties = ('PROP_FRAME_WIDTH', 'PROP_FRAME_HEIGHT')

    def __init__(self, source: Union[int, str] = 0):
        super().__init__()
        self.source = source
        self._capture_lock = threading.RLock()  # VideoCapture is not thread safe: reads and properties are exclusive
        self._lock = threading.Lock()  # protects the ROI and the frames
        self._capture = self._open()
        self._width, self._height = self._read_size()
        self._roi = (0, self._width, 0, self._height, 1, 1)
        self._running = threading.Event()
        self._frames: deque = deque()
        self._n_read = 0  # frames made since the start
        self._last_frame: Optional[np.ndarray] = None

    def _open(self) -> cv2.VideoCapture:
        if isinstance(self.source, int) and sys.platform.startswith('win'):
            capture = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        else:
            capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            capture.release()
            raise IOError(f'The OpenCV camera {self.source!r} cannot be opened')
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # the oldest frames of a buffer are late: keep only the last
        except cv2.error:
            pass
        return capture

    def _read_size(self) -> Tuple[int, int]:
        return (int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # --- sensor ----------------------------------------------------------------------------------------------------

    def get_device_info(self):
        with self._capture_lock:
            try:
                backend = self._capture.getBackendName()
            except (cv2.error, AttributeError):
                backend = 'unknown backend'
        return SimpleNamespace(name=f'OpenCV camera {self.source} ({backend})')

    def get_detector_size(self) -> Tuple[int, int]:
        return self._width, self._height

    def get_data_dimensions(self) -> Tuple[int, int]:
        hstart, hend, vstart, vend, hbin, vbin = self._roi
        return (vend - vstart) // vbin, (hend - hstart) // hbin

    def get_roi(self) -> Tuple[int, int, int, int, int, int]:
        return self._roi

    def set_roi(self, hstart: int = 0, hend: Optional[int] = None, vstart: int = 0, vend: Optional[int] = None,
                hbin: int = 1, vbin: int = 1):
        hend = self._width if hend is None else hend
        vend = self._height if vend is None else vend
        if not (0 <= hstart < hend <= self._width and 0 <= vstart < vend <= self._height):
            raise ValueError(f'The ROI {(hstart, hend, vstart, vend)} is not inside the {self._width}x{self._height} '
                             f'frame')
        if hbin < 1 or vbin < 1 or hend - hstart < hbin or vend - vstart < vbin:
            raise ValueError(f'The binning {(hbin, vbin)} does not fit in the ROI')
        with self._lock:
            self._roi = (hstart, hend, vstart, vend, hbin, vbin)
            self._frames.clear()  # they have the previous shape
            self._last_frame = None

    def set_exposure(self, exposure: float):
        """Not used: the unit of the exposure of OpenCV depends on the backend, see :meth:`set_property`"""

    # --- properties of the camera (brightness, gain, exposure...) ---------------------------------------------------

    def get_properties(self) -> Dict[str, Tuple[float, bool]]:
        """The properties of ``OpenCVProp`` that this camera supports: name -> (value, writable)

        Names are those of ``OpenCVProp`` without the ``CV_CAP_`` prefix, for instance ``PROP_BRIGHTNESS``.
        """
        properties = {}
        with self._capture_lock:
            for name in OpenCVProp.names():
                if OpenCVProp[name].value < 0:  # modes of the DC1394 registers, not properties
                    continue
                try:
                    value = int(self._capture.get(OpenCVProp[name].value))
                    if value == -1:  # not supported by the backend or the camera
                        continue
                    try:
                        writable = bool(self._capture.set(OpenCVProp[name].value, value))
                    except cv2.error:
                        writable = False
                except cv2.error:
                    continue
                properties[name[len('CV_CAP_'):]] = (value, writable)
        return properties

    def set_property(self, name: str, value: float) -> float:
        """Set a property (a name of :meth:`get_properties`) and return the value the camera took"""
        prop = OpenCVProp[f'CV_CAP_{name}'].value
        with self._capture_lock:
            self._capture.set(prop, value)
            taken = self._capture.get(prop)
            if name in self.size_properties:
                self._width, self._height = self._read_size()
                with self._lock:
                    self._roi = (0, self._width, 0, self._height, 1, 1)
                    self._frames.clear()
                    self._last_frame = None
        return taken

    def show_settings(self):
        """Open the window of settings of the driver of the camera (DirectShow on Windows)"""
        with self._capture_lock:
            self._capture.set(OpenCVProp['CV_CAP_PROP_SETTINGS'].value, 1)

    # --- acquisition -----------------------------------------------------------------------------------------------

    def setup_acquisition(self):
        with self._lock:
            self._frames.clear()

    def clear_acquisition(self):
        self._running.clear()
        self.setup_acquisition()

    def start_acquisition(self, nframes: int = 1):
        with self._capture_lock:
            if not self._capture.isOpened():
                self._capture.open(self.source)
        self._running.set()

    def acquisition_in_progress(self) -> bool:
        return self._running.is_set()

    def wait_for_frame(self, since: str = 'lastread', nframes: int = 1, timeout: float = 20.0):
        """Read ``nframes`` new frames, which blocks until the camera gives them

        Return ``False`` at once if the acquisition is stopped, as pylablib does, and raise ``TimeoutError`` if the
        camera gave nothing after ``timeout`` seconds.
        """
        deadline = time.perf_counter() + timeout
        n_new = 0
        while n_new < nframes:
            if not self._running.is_set():
                return False
            with self._capture_lock:
                ok, frame = self._capture.read()
            if ok:
                self._add_frame(frame)
                n_new += 1
            elif time.perf_counter() > deadline:
                raise TimeoutError('Timeout while waiting for a frame of the OpenCV camera')
            else:
                time.sleep(0.005)
        return True

    def _add_frame(self, frame: np.ndarray):
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with self._lock:
            hstart, hend, vstart, vend, hbin, vbin = self._roi
            frame = frame[vstart:vend, hstart:hend]
            if hbin > 1 or vbin > 1:  # mean of the pixels of a bin, the last incomplete ones are dropped
                height, width = (vend - vstart) // vbin, (hend - hstart) // hbin
                frame = frame[:height * vbin, :width * hbin]
                frame = frame.reshape((height, vbin, width, hbin) + frame.shape[2:]).mean(axis=(1, 3),
                                                                                         dtype=np.float32)
            self._frames.append(frame)
            self._last_frame = frame
            self._n_read += 1

    def read_multiple_images(self, rng=None, peek: bool = False, missing_frame: str = 'skip',
                             return_rng: bool = False):
        """The frames read and not given yet, as an array (nframes, height, width[, 3])"""
        with self._lock:
            frames = list(self._frames)
            last = self._n_read
            first = last - len(frames)
            if not peek:
                self._frames.clear()
        if frames:
            array = np.array(frames)
        else:
            array = np.zeros((0,) + self.get_data_dimensions())
        return (array, (first, last)) if return_rng else array

    def read_newest_image(self, peek: bool = False) -> Optional[np.ndarray]:
        with self._lock:
            if not peek:
                self._frames.clear()
            return self._last_frame

    def close(self):
        self.clear_acquisition()
        with self._capture_lock:
            self._capture.release()
