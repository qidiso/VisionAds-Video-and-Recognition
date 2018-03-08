#!/usr/bin/env python
# -*- vim: set sw=4 ts=4 et: -*-
# python main.py 2> /dev/null


from __future__ import print_function
from multiprocessing import set_start_method
import click

from face_finder import FaceFinder
from video_feed import VideoFeed
from render_loop import RenderLoop
from ad_feed import AdFeed
def main():
    # Components
    render_loop = RenderLoop()
    video_feed = VideoFeed()
    face_finder = FaceFinder()
    ad_feed = AdFeed()

    # connecting the dots
    # video_feed.add_listener(render_loop) # render loop listens to video feed frames
    video_feed.add_listener(face_finder) # face finder listens to video feed frames
    ad_feed.add_listener(render_loop)  # render loop listens to advertise feed
    face_finder.add_listener(ad_feed) # ad feed listens to face_finder
    # video_feed.add_listener(ad_feed)
    # allow us to quit
    video_feed.daemon = True
    face_finder.daemon = True
    ad_feed.daemon = True
    # start threads
    face_finder.start()
    video_feed.start()
    ad_feed.start()

    render_loop.run()


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # set_start_method('spawn')  # for multiprocess cv2 to work
    main()
