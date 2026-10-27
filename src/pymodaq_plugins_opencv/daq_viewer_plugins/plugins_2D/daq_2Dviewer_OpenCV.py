from pymodaq_utils.logger import set_logger, get_module_name
from pymodaq_gui.parameter import Parameter

from pymodaq.control_modules.viewer_utility_classes import comon_parameters, main

from pymodaq_plugins_utils.hardware.camera_base import CameraBase, cam_params

from pymodaq_plugins_opencv.hardware.opencv_camera import OpenCVCamera

logger = set_logger(get_module_name(__file__))


class DAQ_2DViewer_OpenCV(CameraBase):
    """Cameras read with OpenCV (webcams, any device handled by ``cv2.VideoCapture``)

    The acquisition, the live mode, the averaging, the ROI drawn in the viewer, the binning (both done on the image
    since a webcam has no ROI) and the FPS are those of ``CameraBase``. The images are RGB: select a colour
    conversion (default RGB2GRAY) or None to get the three channels.

    ``Cam. Settings`` lists the properties of the camera that OpenCV can read (brightness, gain, exposure...), the
    unit of some of them depends on the backend of OpenCV. Changing the width or the height there resets the ROI.
    """
    params = comon_parameters + \
             [{'title': 'Camera index:', 'name': 'camera_index', 'type': 'int', 'value': 0, 'default': 0, 'min': 0},
              {'title': 'Open Settings:', 'name': 'open_settings', 'type': 'bool', 'value': False},
              {'title': 'Cam. Settings:', 'name': 'cam_settings', 'type': 'group', 'children': []},
              ] + cam_params

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # OpenCV sets neither the exposure time nor the buffer of CameraBase, and a colour image is shown in gray
        self.settings.child('timing_opts', 'exposure_time').show(False)
        self.settings.child('buffer').show(False)
        self.settings.child('color_conversion').setOpts(value='RGB2GRAY', default='RGB2GRAY')

    def ini_detector_custom(self, controller=None):
        """Open the camera (master) or take the one of the master (slave)"""
        if self.is_master:
            self.controller = OpenCVCamera(self.settings['camera_index'])
        else:
            self.controller = controller
        self.add_camera_properties()

    def add_camera_properties(self):
        """Make a setting of each property that this camera supports"""
        group = self.settings.child('cam_settings')
        group.clearChildren()
        group.addChildren([
            {'title': name, 'name': name, 'type': 'float', 'value': value, 'readonly': not writable}
            for name, (value, writable) in self.controller.get_properties().items()])

    def commit_settings(self, param: Parameter):
        """Apply the consequences of a change of value in the detector settings

        Parameters
        ----------
        param: Parameter
            A given parameter (within detector_settings) whose value has been changed by the user
        """
        if param.name() == 'open_settings':
            if param.value():
                self.controller.show_settings()
        elif param.parent() is not None and param.parent().name() == 'cam_settings':
            self.set_camera_property(param)
        else:
            super().commit_settings(param)

    def set_camera_property(self, param: Parameter):
        """Set a property of the camera and show the value that it took"""
        resize = param.name() in self.controller.size_properties
        was_live = self.is_live
        if resize:  # the images get another shape
            self.stop()
        param.setValue(self.controller.set_property(param.name(), param.value()))
        if resize:
            self.get_set_main_parameters()
            self._prepare_view()
            if was_live:
                self.grab_data(self.Naverage, live=True)


if __name__ == '__main__':
    main(__file__, init=False)
