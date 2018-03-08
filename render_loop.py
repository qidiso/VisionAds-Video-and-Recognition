#!/usr/bin/env python
import webbrowser
import cv2

import numpy as np

from detect_picture_utils import Listener



def disable_vsync():
    import sys
    if sys.platform != 'darwin':
        return
    try:
        import ctypes
        import ctypes.util
        ogl = ctypes.cdll.LoadLibrary(ctypes.util.find_library("OpenGL"))
        v = ctypes.c_int(0)

        ogl.CGLGetCurrentContext.argtypes = []
        ogl.CGLGetCurrentContext.restype = ctypes.c_void_p

        ogl.CGLSetParameter.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        ogl.CGLSetParameter.restype = ctypes.c_int

        context = ogl.CGLGetCurrentContext()

        ogl.CGLSetParameter(context, 222, ctypes.pointer(v))
    except Exception as e:
        print("Unable to set vsync mode, using driver defaults: {}".format(e))

class RenderLoop(Listener):
    def __init__(self):
        Listener.__init__(self)

    def run(self):
        cv2.namedWindow('Video', cv2.WND_PROP_FULLSCREEN)
        disable_vsync()
        cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        frame = None
  
        while True:
            sender, data = self.recv()
            if sender == 'ad_feed':
                # play default advertise
                advertise_frame = data[0]
                cv2.imshow('Video', advertise_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
