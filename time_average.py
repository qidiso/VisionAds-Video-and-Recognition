#!/usr/bin/env python

import collections
import itertools

import cv2

class TimeAverage:
    SERIES_LENGTH = 20

    def __init__(self,SERIES_LENGTH):
        self.SERIES_LENGTH = SERIES_LENGTH
        self.reset()

    def score(self):
        return sum(self.series)

    def detected(self):
        return self.percent() > 0.8

    def reset(self):
        self.series = collections.deque(itertools.repeat(0, self.SERIES_LENGTH), self.SERIES_LENGTH)

    def percent(self):
        return (self.score() / self.series.maxlen + 1) / 2

    def update(self, delta):
        self.series.append(-1 if delta < 1 else +1)

class TimeAverageRenderer:
    def __init__(self, time_average):
        self.time_average = time_average

    def draw(self, frame):
        TimeAverageRenderer.draw_for_percentage(frame, self.time_average.percent())

    @staticmethod
    def draw_for_percentage(frame, percentage):
        padding = 8
        color = (0,255,0) if percentage > 0.8 else (0,0,255)
        width, height = int(frame.shape[1] * 0.2), int(frame.shape[0] * 0.025)
        progress = int(width * percentage)

        frame_h, frame_w, _ = frame.shape
        x, y = frame_w - width - padding, frame_h - height - padding
        cv2.rectangle(frame, (x-2,y-2), (x-2+width+4,y-2+height+4), color=(128,128,128), thickness=2)
        cv2.rectangle(frame, (x,y), (x+progress,y+height), color=color, thickness=-1)
