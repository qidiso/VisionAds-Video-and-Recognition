#!/usr/bin/env python

from multiprocessing import current_process, Queue
import queue
import time

WINDOW_WIDTH = 600

def log(string):
    current_time = time.ctime()
    print("{0} [{1}] {2}".format(current_time, current_process().name, string))


class Listener:
    def __init__(self):
        self.queue = Queue(1)

    def send(self, data):
        try:
            self.queue.put_nowait(data)
        except queue.Full:
            pass

    def recv(self):
        return self.queue.get()

    def empty(self):
        return self.queue.empty()

class Broadcaster:
    def __init__(self, name):
        self.name = name
        self.listeners = []

    def add_listener(self, listener):
        self.listeners.append(listener)

    def broadcast(self, data):
        for listener in self.listeners:
            listener.send([self.name, data])

class Singleton(object):
    def __new__(cls, *p, **k):
        if not '_the_instance' in cls.__dict__:
            cls._the_instance = object.__new__(cls)
        return cls._the_instance
