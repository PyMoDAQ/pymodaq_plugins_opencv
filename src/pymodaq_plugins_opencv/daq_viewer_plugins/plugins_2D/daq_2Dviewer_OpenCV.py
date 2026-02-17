from sys import platform

import numpy as np
import cv2

from pymodaq_utils.utils import ThreadCommand
from pymodaq.utils.data import DataFromPlugins, Axis, DataToExport
from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, comon_parameters, main
from pymodaq_gui.parameter import Parameter

from pymodaq_plugins_opencv.hardware.opencv import OpenCVProp


class DAQ_2DViewer_OpenCV(DAQ_Viewer_base):
    """
    """
    params = comon_parameters + \
             [{'title': 'Camera index:', 'name': 'camera_index', 'type': 'int', 'value': 0, 'default': 0, 'min': 0},
              {'title': 'Colors:', 'name': 'colors', 'type': 'list', 'value': 'gray', 'limits': ['gray', 'RGB']},
              {'title': 'Open Settings:', 'name': 'open_settings', 'type': 'bool', 'value': False},
              {'title': 'Cam. Settings:', 'name': 'cam_settings', 'type': 'group', 'children': [
                 #     {'title': 'Autoexposure:', 'name': 'autoexposure', 'type': 'bool', 'value': False},
                 #     {'title': 'Exposure:', 'name': 'exposure', 'type': 'int', 'value': 0},
             ]},
             ]

    def ini_attributes(self):
        #  TODO declare the type of the wrapper (and assign it to self.controller) you're going to use for easy
        #  autocompletion
        self.controller: cv2.VideoCapture = None

        # TODO declare here attributes you want/need to init with a default value

        self.x_axis = None
        self.y_axis = None

    def get_active_properties(self):
        props = OpenCVProp.names()
        self.additional_params = []
        for prop in props:
            try:
                ret = int(self.controller.get(OpenCVProp[prop].value))
                if ret != -1:
                    try:
                        ret_set = self.controller.set(OpenCVProp[prop].value, ret)
                    except:
                        ret_set = False
                    self.additional_params.append(
                        {'title': prop[7:], 'name': prop[7:], 'type': 'float', 'value': ret, 'readonly': not ret_set})
            except:
                pass
        self.settings.child('cam_settings').addChildren(self.additional_params)
        pass

    def commit_settings(self, param: Parameter):
        """Apply the consequences of a change of value in the detector settings

        Parameters
        ----------
        param: Parameter
            A given parameter (within detector_settings) whose value has been changed by the user
        """
        if param.name() == 'open_settings':
            if param.value():
                self.controller.set(OpenCVProp['CV_CAP_PROP_SETTINGS'].value, 1)
                # param.setValue(False)
        elif param.name() == 'colors':
            pass
        else:
            self.controller.set(OpenCVProp['CV_CAP_' + param.name()].value, param.value())
            val = self.controller.get(OpenCVProp['CV_CAP_' + param.name()].value)
            param.setValue(val)

    def ini_detector(self, controller=None):
        """Detector communication initialization

        Parameters
        ----------
        controller: (object)
            custom object of a PyMoDAQ plugin (Slave case). None if only one actuator/detector by controller
            (Master case)

        Returns
        -------
        info: str
        initialized: bool
            False if initialization failed otherwise True
        """
        if self.is_master:
            if 'win' in platform:
                self.controller = cv2.VideoCapture(self.settings['camera_index'], cv2.CAP_DSHOW)
            else:
                self.controller = cv2.VideoCapture(self.settings['camera_index'])
                  # to add settable settings to the param list (but driver builtin settings window is prefered (OpenCVProp['CV_CAP_PROP_SETTINGS'])
            self.get_active_properties()

        else:
            self.controller = controller


        self.x_axis = self.get_xaxis()
        self.y_axis = self.get_yaxis()

        self.dte_signal_temp.emit(DataToExport('myplugin',
                            data=[DataFromPlugins(name='Mock1',
                                                  data=[np.zeros((len(self.y_axis), len(self.x_axis)))],
                                                  dim='Data2D', labels=['opencv'],
                                                  axes=[self.x_axis, self.y_axis]), ]))

        info = "Whatever info you want to log"
        initialized = True
        return info, initialized

    def close(self):
        """Terminate the communication protocol"""
        self.controller.release()

    def get_xaxis(self) -> Axis:
        """ Get the current x_axis from the controller

        Returns
        -------
        Axis
        """
        Nx = int(self.controller.get(cv2.CAP_PROP_FRAME_WIDTH))  # property index corresponding to width
        return Axis('xaxis', data=np.linspace(0, Nx - 1, Nx), index=1)

    def get_yaxis(self) -> Axis:
        """ Get the current y_axis from the controller

        Returns
        -------
        Axis
        """
        Ny = int(self.controller.get(cv2.CAP_PROP_FRAME_HEIGHT))  # property index corresponding to width
        return Axis('yaxis', data=np.linspace(0, Ny - 1, Ny), index=0)

    def grab_data(self, Naverage=1, **kwargs):
        """Start a grab from the detector

        Parameters
        ----------
        Naverage: int
            Number of hardware averaging (if hardware averaging is possible, self.hardware_averaging should be set to
            True in class preamble and you should code this implementation)
        kwargs: dict
            others optionals arguments
        """
        if not self.controller.isOpened():
            self.controller.open(self.settings['camera_index'])

        ret, frame = self.controller.read()

        if ret:
            if self.settings['colors'] == 'gray':
                data_cam = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)]
                data_cam[0] = data_cam[0].astype(np.float32)
            else:
                if len(frame.shape) == 3:
                    data_cam = [frame[:, :, ind] for ind in range(frame.shape[2])]
                else:
                    data_cam = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)]

        else:
            data_cam = [np.zeros((len(self.y_axis), len(self.x_axis)))]
            self.emit_status(ThreadCommand('Update_Status', ['no return from the controller', 'log']))

        self.dte_signal.emit(DataToExport('OpenCV data',
                                          data=[
                            DataFromPlugins(name='OpenCV', data=data_cam, dim='Data2D')]))

    def stop(self):
        """Stop the current grab hardware wise if necessary"""
        pass


if __name__ == '__main__':
    main(__file__, init=False)
