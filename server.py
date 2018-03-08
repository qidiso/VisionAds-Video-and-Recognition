# main.py
import cv2
import imutils
from imutils.video import VideoStream

from multiprocessing import set_start_method
from multiprocessing import Process
from face_finder import FaceFinder
from video_feed import VideoFeed
# from render_loop import RenderLoop
from ad_feed import AdFeed
import os
import numpy as np

from detect_picture_utils import Listener
import time
import redis
import json 


class RenderLoop(Process,Listener):
    def __init__(self):
        Listener.__init__(self)
        Process.__init__(self)
 
    def run(self):
        myredis = redis.Redis()
        # cv2.namedWindow('Video', cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        while True:
            sender, data = self.recv()
            if sender == 'ad_feed':
                # play default advertise
                advertise_frame = data[0]
                cv2.imshow('Video'+str(os.getpid()), advertise_frame)
                ret, jpeg = cv2.imencode('.jpg', advertise_frame)
                myredis.publish('channel', jpeg.tobytes())
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break



def main():
    # Components
    render_loop = RenderLoop()
    video_feed = VideoFeed()
    face_finder = FaceFinder()
    ad_feed = AdFeed()

    video_feed.add_listener(face_finder) # face finder listens to video feed frames
    ad_feed.add_listener(render_loop)  # render loop listens to advertise feed
    face_finder.add_listener(ad_feed) # ad feed listens to face_finder
 
    video_feed.daemon = True
    face_finder.daemon = True
    ad_feed.daemon = True
    render_loop.daemon = True
 
    # start threads
    face_finder.start()
    video_feed.start()
    ad_feed.start()

    render_loop.run()

if __name__ == '__main__':
    main()
    