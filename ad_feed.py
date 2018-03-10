#!/usr/bin/env python

from multiprocessing import Process
from detect_picture_utils import \
    log, \
    Broadcaster, \
    Listener
import imutils
from imutils.video import VideoStream
from imutils import paths
import random 
import cv2
import time
import numpy as np 
from gui_face_finder import GUIFaceFinder

ADVERTISE_PATHS = "images/{}".format("unknow")
class Welcome:
    def __init__(self):
        self.font = cv2.freetype.createFreeType2()
        self.font.loadFontData(fontFileName='fonts/MarkerFelt.ttc', id=0)

    def render(self, frame, name,gender,age):
        h,w,_ = frame.shape
        font_size = 50
        half_font_size = font_size / 2
        if gender == "male":
            message = "Seja bem vindo à Brasil Brokers, {}!".format(name)
        else :
            message = "Seja bem vinda à Brasil Brokers, {}!".format(name)
        
        x_pos = int(w/2-half_font_size*0.8*(len(message)/2))
        y_pos = int(h/2-half_font_size)
        self.font.putText(
            img=frame,
            text=message,
            org=(x_pos, y_pos),
            fontHeight=font_size,
            color=(255, 255, 255),
            thickness=-1,
            line_type=cv2.LINE_AA,
            bottomLeftOrigin=False)


class AdFeed(Process, Broadcaster, Listener):
    """Face finder process"""

    def __init__(self):
        Process.__init__(self)
        Broadcaster.__init__(self, 'ad_feed')
        Listener.__init__(self)
        
        

    def run(self):
        FACE_RECOGNIZER = True 
        advertise_path = AdFeed.load_image(ADVERTISE_PATHS).replace("\ "," ")
        # video_capture = VideoStream(src=advertise_path, usePiCamera=False).start()
        video_capture = cv2.VideoCapture(advertise_path)
        log('Advertise streaming...')     
        blank_frame = None
        welcome = Welcome()
        while True:
            if not self.empty():
                sender, data = self.recv()
                log('Sender to AdFeed...{}'.format(sender)) 
                if sender == 'face_finder':
                    state = data[0]
                    if state == "NoMatch":
                        # broadcast frame of the face advertise 
                        url = AdFeed.load_image(data[1]).replace("\ "," ")
                        print("Found a face and gender open video :" +url)
                        ad_capture = cv2.VideoCapture(url)
                        while True:
                            grb, framex = ad_capture.read()
                            if framex is None:
                                break
                            framex = imutils.resize(framex, width=min(1280, framex.shape[1]))
                            self.broadcast([framex])
                            time.sleep(1.0 / 23.97)
                            if not self.empty() :
                                sender, data = self.recv()
                                if sender == 'face_finder':
                                    state = data[0]
                                    if data[0] == "Match" and FACE_RECOGNIZER:
                                        break

                    if state == "Match" and FACE_RECOGNIZER:
                        face = data[1]
                        people = data[2]
                        person = people[0]
                        gender = data[3]
                        age = data[4]
                        start_t = time.time()
                        elapse =  time.time() - start_t
                        while (elapse < 5):
                            if blank_frame is None:
                                blank_frame = np.zeros((720,1280,3), dtype=np.uint8)
                            frame = blank_frame.copy()
                            # GUIFaceFinder().draw(frame, face, people,person.imgFile, 0.99)
                            welcome.render(frame, person.name,gender,age)


                            self.broadcast([frame])
                            time.sleep(1.0 / 23.97)
                            elapse =  time.time() - start_t

                    while not self.empty():
                        self.recv()
            # log('Advertise streaming default...') 
            grb,frame = video_capture.read()
            # log("grab : {} ".format(grb))
            if frame is None:
                print("END OF FILE Advertise")
                advertise_path = AdFeed.load_image(ADVERTISE_PATHS).replace("\ "," ")
                video_capture = cv2.VideoCapture(advertise_path)
                grb,frame = video_capture.read()

            frame = imutils.resize(frame, width=min(1280, frame.shape[1]))
            # print("Broadcast Ad Feed")
            self.broadcast([frame])
            time.sleep(1.0 / 24.0)
  
    @staticmethod           
    def load_image(path):
        imagePaths = list(paths.list_files(path,validExts=(".jpg",".mp4")))
        random.shuffle(imagePaths)
        if imagePaths == []: print("need more gifs at :" + path)
        imagePath = imagePaths[0]
        return imagePath


  
