#!/usr/bin/env python

import base64
import json
import os
import pickle
import time

import cv2

import numpy as np

from detect_picture_utils import log, Singleton
from time_average import TimeAverage

hostname = os.environ.get('API_HOSTNAME') or 'localhost'


class UnknownPerson(Singleton):
    id = None
    imgFile = None
    name = 'unknown_person'
    encoding = None
    in_the_frame = False
# tempo alterado de 20 para 2
    i_am_recognized = TimeAverage(2)

    def set_state(self, state):
        self.i_am_recognized.update(int(state))
        self.in_the_frame = self.i_am_recognized.detected()
        # log('set state {} in frame: {} score: {}'.format(state, self.in_the_frame, self.i_am_recognized.score()))

    def entered_the_frame(self):
        self.set_state(True)

    def left_the_frame(self):
        self.set_state(False)

    def reset(self):
        self.i_am_recognized.reset()
        self.in_the_frame = False


class Person:
    id = None
    name = None
    project = None
    imgFile = None
    shape = None
    in_the_frame = False
    in_the_house = False
    checked_in = None
    checked_at = None

    def __init__(self):
        self.i_am_recognized = TimeAverage(20)

    @classmethod
    def fromfile(cls, filename):
        import face_recognition.api as face_recognition
        img = cv2.imread(filename)
        self = cls()
        try :
            self.name = os.path.splitext(os.path.basename(filename))[0]
            self.id = self.name
            self.encoding = face_recognition.face_encodings(img)[0]
            self.imgFile = cv2.resize(img, (100, 100))
            self.shape = self.imgFile.shape
        except Exception as e:
            print("Picture with problem {}".format(filename))

        return self

    @classmethod
    def fromimg(cls, img):
        import face_recognition.api as face_recognition
        self = cls()
        self.id = 'driver license match'
        self.name = 'driver license match'
        self.encoding = face_recognition.face_encodings(img)[0]
        self.imgFile = cv2.resize(img, (100, 100))
        self.shape = self.imgFile.shape
        return self

    @classmethod
    def fromjson(cls, obj):
        nparr = np.fromstring(
            base64.b64decode(obj['picture']['image']), np.uint8)

        self = cls()
        self.id = obj['id']
        self.name = obj['name']
        self.project = obj['project']
        self.in_the_house = obj['checked_in']
        self.encoding = pickle.loads(eval(obj['picture']['features']))
        self.imgFile = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        self.shape = self.imgFile.shape
        return self

    def __update_house_state(self):
        if not self.id:
            return
        log('********* update house state: {}'.format(not self.in_the_house))
        self.in_the_house = not self.in_the_house

    def set_state(self, state):
        if self.checked_at and (time.time() - self.checked_at < 35):
            log('{} taking a nap... ({})'.format(self.name,time.time() - self.checked_at))
            self.reset()
            return
        # log('({}) set state {} in frame: {} score: {}'.format(self.name, state, self.in_the_frame, self.i_am_recognized.score()))
        self.i_am_recognized.update(int(state))
        if self.i_am_recognized.detected():
            if not self.in_the_frame:
                self.__update_house_state()

            self.in_the_frame = True
            self.checked_at = time.time()
        else:
            self.in_the_frame = False

    def entered_the_frame(self):
        self.set_state(True)

    def left_the_frame(self):
        self.set_state(False)

    def reset(self):
        self.i_am_recognized.reset()
        self.in_the_frame = False

    def recognized(self):
        return self.i_am_recognized.detected()
