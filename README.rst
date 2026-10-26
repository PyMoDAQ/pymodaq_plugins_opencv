pymodaq_plugins_opencv
######################

.. the following must be adapted to your developped package, links to pypi, github  description...

.. image:: https://img.shields.io/pypi/v/pymodaq_plugins_opencv.svg
   :target: https://pypi.org/project/pymodaq_plugins_opencv/
   :alt: Latest Version

.. image:: https://readthedocs.org/projects/pymodaq/badge/?version=latest
   :target: https://pymodaq.readthedocs.io/en/stable/?badge=latest
   :alt: Documentation Status

.. image:: https://github.com/PyMoDAQ/pymodaq_plugins_opencv/workflows/Upload%20Python%20Package/badge.svg
   :target: https://github.com/PyMoDAQ/pymodaq_plugins_opencv
   :alt: Publication Status

Set of PyMoDAQ plugins for camera recognised by the opencv library (webcams, ...)


Authors
=======

* Sebastien J. Weber  (sebastien.weber@cnrs.fr)


Instruments
===========

Below is the list of instruments included in this plugin

Viewer2D
++++++++

* **opencv**: control of cameras using the opencv library (mainly webcams)

  It is built on the camera base class of ``pymodaq_plugins_utils`` (``CameraBase``), with the driver
  ``hardware/opencv_camera.py`` that follows the camera driver interface of that package. So it has the same
  settings as the other camera plugins: live and snap, averaging, the ROI drawn in the viewer (*Update ROI from
  Viewer*, *Apply ROI*, *Clear ROI+Bin*), binning and FPS. A webcam has neither a hardware ROI nor binning: they are
  applied on each image (binning is the mean of the pixels).

  * *Camera index*: the camera to open (0 is usually the one built in the laptop).
  * *Color Conversion*: OpenCV gives colour images, converted by default to gray (``RGB2GRAY``). With ``None``
    the three channels (red, green, blue) are given.
  * *Cam. Settings*: the properties that OpenCV can read on the camera (brightness, gain, exposure...), the unit
    of some depends on the backend. Changing the frame width or height there resets the ROI. *Open Settings* shows
    the settings window of the driver of the camera (DirectShow, Windows).

  The exposure time and buffer settings of the camera base class are hidden: OpenCV does not set them in a portable
  way, use *Cam. Settings*.

.. if needed use this field

    Infos
    =====

    if needed for installation or other infos
