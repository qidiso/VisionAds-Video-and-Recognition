#!/usr/bin/env python

from multiprocessing import Process
# from gender_recognizer import GenderRecognizer
# from face_recognizer import FaceRecognizer
from detect_picture_utils import \
    log, \
    Broadcaster, \
    Listener
from imutils import paths
from imutils.face_utils import FaceAligner
# import dlib
import cv2
import imutils
from gui_face_finder import  \
    IdleState, \
    NoMatchState, \
    SearchingState, \
    MatchState
# from people import People

class FaceFinder(Process, Broadcaster, Listener):
    """Face finder process"""

    def __init__(self):
        Process.__init__(self)
        Broadcaster.__init__(self, 'face_finder')
        Listener.__init__(self)
        self.detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
 
    def run(self):
        import dlib
        from people import People
        from gender_recognizer_tf import GenderRecognizer
        from face_recognizer import FaceRecognizer
        self.gender_recognizer = GenderRecognizer()
        self.p = People("./known_people")
        self.face_recognizer = FaceRecognizer(self.p.people) #this is not ok , we should pass onlu self.p as object
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.fa = FaceAligner(self.predictor, desiredFaceWidth=160)
        self.state = IdleState()
        while True:         
            label, data = self.recv()
            if label == 'video_feed':
                frame = data[0]
                face= None
                aligned_image, face, rect_nums, XY = self.face_extractor(frame)
                if face is None:
                    continue

                face_locations, face_names , percent  = self.face_recognizer.recognize(face)
                in_the_frame =  self.__update_state(self.p.people, percent)
              
                if (self.state.__class__ == MatchState):
                    path,gender,age = self.gender_recognizer.recognize(aligned_image, face, rect_nums, XY)
                    self.broadcast(["Match",face,in_the_frame,gender,age]) 
                else:
                    path ,gender,age  = self.gender_recognizer.recognize(aligned_image, face, rect_nums, XY)
                    if path is None:
                        continue
                    print("FaceFinder path = "+path)
                    self.broadcast(["NoMatch",path])

    def face_extractor(self,frame):
        import dlib
        image = imutils.resize(frame, width=256)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # detect faces in the grayscale frame
        rects = self.detector.detectMultiScale(gray, scaleFactor=1.1, 
            minNeighbors=5, minSize=(15, 15),
            flags=cv2.CASCADE_SCALE_IMAGE)
        if len(rects) > 0:
            (x, y, w, h)  = rects[0]
            rect = dlib.rectangle(int(x), int(y), int(x + w),
                int(y + h))
            face = self.fa.align(image, gray, rect)
            return [face] , face,len(rects),[(x,y)]
        return [],None,0,[]

    def __update_state(self, people, percent):
        if percent < 0.8:
            if self.state.__class__ != IdleState:
                self.state = IdleState()
            return None
        else:
            in_the_frame = [p for p in people if p.in_the_frame]

            if len(in_the_frame) == 0:
                if percent > 0.8:
                    if self.state.__class__ != SearchingState:
                        self.state = SearchingState()
            elif in_the_frame[0].id is None:
                if self.state.__class__ != NoMatchState:
                    self.state = NoMatchState()
            else:
                if self.state.__class__ != MatchState:
                    self.state = MatchState()
            return in_the_frame