#!/usr/bin/env python

import cv2
import imutils

from fps import FPS
from people import People
from time_average import TimeAverageRenderer
from detect_picture_utils import log

class State:
    people = None

    def __draw_image(self, frame, x,y,w,h, img):
        frame[y:y+h,x:x+w] = img

    def update(self, people):
        self.people = people

    @staticmethod
    def __draw_border(frame, top, right, bottom, left, color=(0,255,0)):
        width = right - left
        height = bottom - top

        rect_len = int(width*0.2)
        cv2.rectangle(frame, (left,top), (left+rect_len,top+2), color, 2)
        cv2.rectangle(frame, (left,top), (left+2,top+rect_len), color, 2)

        cv2.rectangle(frame, (right-rect_len,top), (right,top+2), color, 2)
        cv2.rectangle(frame, (right,top), (right-2,top+rect_len), color, 2)

        cv2.rectangle(frame, (left,bottom), (left+rect_len,bottom-2), color, 2)
        cv2.rectangle(frame, (left,bottom), (left+2,bottom-rect_len), color, 2)

        cv2.rectangle(frame, (right-rect_len,bottom), (right,bottom+2), color, 2)
        cv2.rectangle(frame, (right,bottom), (right-2,bottom-rect_len), color, 2)

    def draw(self, frame, face):
        (roi_top, roi_right, roi_bottom, roi_left) = GUIFaceFinder.css_roi_for_frame_shape(frame.shape)

        h, w, _ = self.banner.shape
        pos = ( int((roi_right+roi_left-w)/2), int((roi_top-h)/2) )
        self.__draw_image(frame, *pos, w, h, self.banner)

        h, w, _ = face.shape
        pos = ( int((roi_right+roi_left-w)/2), int((roi_bottom+roi_top-h)/2) )
        self.__draw_image(frame, *pos, w, h, face)

        State.__draw_border(frame, roi_top, roi_right, roi_bottom, roi_left)

    def draw_matches(self, frame, face, border_color=(0,255,0)):
        (roi_top, roi_right, roi_bottom, roi_left) = GUIFaceFinder.css_roi_for_frame_shape(frame.shape)

        h, w, _ = self.banner.shape
        pos = ( int((roi_right+roi_left-w)/2), int((roi_top-h)/2) )
        self.__draw_image(frame, *pos, w, h, self.banner)

        padding = 8
        face_h, face_w, _ = face.shape
        face_pos = ( int((roi_right+roi_left)/2)-face_w-padding, int((roi_bottom+roi_top-face_h)/2) )
        self.__draw_image(frame, *face_pos, face_w, face_h, face)

        img = cv2.resize(self.rects, (face_w, face_h))
        h, w, _ = img.shape
        pos = ( int((roi_right+roi_left)/2)+padding, int((roi_bottom+roi_top-h)/2) )
        self.__draw_image(frame, *pos, w, h, img)

        face_left, face_top = face_pos
        face_right, face_bottom = face_left+face_w, face_top+face_h
        State.__draw_border(frame, face_top, face_right, face_bottom, face_left, border_color)

class IdleState(State):
    banner = cv2.imread('images/checkin1.jpg')

class SearchingState(State):
    banner = cv2.imread('images/checkin2.jpg')
    rects = None

    def draw(self, frame, face):
        super().draw_matches(frame, face)

class MatchState(State):
    banner = cv2.imread('images/checkin3.jpg')
    rects = cv2.imread('images/checkin5.jpg')

    def update(self, people):
        if people is not None and any(p for p in people if p.in_the_frame):
            super().update(people)

    def draw(self, frame, face):
        super().draw_matches(frame, face)
        People.draw_name(frame, [p.name for p in self.people if p.in_the_frame])
        People.draw_in_the_frame_picture(frame, [p for p in self.people if p.in_the_frame],face.shape)

class NoMatchState(State):
    banner = cv2.imread('images/checkin4.jpg')
    rects = cv2.imread('images/checkin6.jpg')

    def draw(self, frame, face):
        super().draw_matches(frame, face, (0,0,255))


class GUIFaceFinder:
    fps = FPS()
    state = IdleState()
    resized = False

    @staticmethod
    def css_roi_for_frame_shape(frame_shape):
        w_percent = 0.30
        h_percent = 0.15
        frame_h, frame_w, _ = frame_shape
        roi_top = int(frame_h * h_percent)
        roi_right = int(frame_w * (1 - w_percent))
        roi_bottom = int(frame_h * (1 - h_percent))
        roi_left = int(frame_w * w_percent)
        return (roi_top, roi_right, roi_bottom, roi_left)

    def __update_state(self, people, percent):
        if percent < 0.8:
            if self.state.__class__ != IdleState:
                self.state = IdleState()
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

    def draw(self, frame, face, people, driver_license_face, percent):
        if not self.resized:
            (roi_top, _roi_right, roi_bottom, _roi_left) = GUIFaceFinder.css_roi_for_frame_shape(frame.shape)
            padding = 32
            banners_height = roi_top-padding
            rects_height = roi_bottom-roi_top

            IdleState.banner = imutils.resize(IdleState.banner, height=banners_height)
            SearchingState.banner = imutils.resize(SearchingState.banner, height=banners_height)
            MatchState.banner = imutils.resize(MatchState.banner, height=banners_height)
            NoMatchState.banner = imutils.resize(NoMatchState.banner, height=banners_height)

            SearchingState.rects = imutils.resize(driver_license_face, height=rects_height)
            MatchState.rects = imutils.resize(MatchState.rects, height=rects_height)
            NoMatchState.rects = imutils.resize(NoMatchState.rects, height=rects_height)
            self.resized = True

        self.__update_state(people, percent)
        self.fps.draw(frame)
        TimeAverageRenderer.draw_for_percentage(frame, percent)
        self.state.update(people)
        self.state.draw(frame, face)
