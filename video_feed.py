#!/usr/bin/env python

from multiprocessing import Process
import time

import cv2
import imutils
from imutils.video import VideoStream

from detect_picture_utils import \
    log, \
    Broadcaster


class VideoFeed(Process, Broadcaster):
    """Video capture process"""

    def __init__(self):
        Process.__init__(self)
        Broadcaster.__init__(self, 'video_feed')

    def run(self):
        log('Live streaming...')
        # video_capture = VideoStream(src="rtsp://admin:admin@cyberlabsrio.ddns.net/cam/realmonitor?channel=5&subtype=0", usePiCamera=False).start(
        gst_str = ("v4l2src device=/dev/video{} ! " "video/x-raw, width=(int){}, height=(int){}, format=(string)RGB ! " "videoconvert ! appsink").format(1, 1920, 1080)
        print(gst_str)
   
        video_capture = VideoStream(src=gst_str, usePiCamera=False).start()
        while True:
            frame = video_capture.read()
            if frame is None:
                continue
            frame = imutils.resize(frame, width=min(256, frame.shape[1]))
            self.broadcast([frame])
            time.sleep(1.0 / 30.0)
            # time.sleep(1)
