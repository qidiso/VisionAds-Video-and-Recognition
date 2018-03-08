#!/usr/bin/env python

import time

import cv2

class FPS:
    def __init__(self):
        self.frame_times = []
        self.start_t = time.time()

    def draw(self, frame):
        cv2.putText(frame, self.get(), (5, frame.shape[0] - 10), cv2.FONT_HERSHEY_DUPLEX, 0.75, (128,128,128), 1)

    def get(self):
        return "%.1f fps" % (self._fps(),)

    def _fps(self):
        end_t = time.time()
        time_taken = end_t - self.start_t
        self.start_t = end_t
        self.frame_times.append(time_taken)
        self.frame_times = self.frame_times[-20:]
        return len(self.frame_times) / sum(self.frame_times)
