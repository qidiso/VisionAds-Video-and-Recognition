# main.py
import cv2
from flask import Flask, render_template, Response
import imutils
import numpy as np 
import redis
import json 

app = Flask(__name__)
myredis = redis.Redis()


@app.route('/')
def index():
    return render_template('index.html')

def gen():

    pubsub = myredis.pubsub(ignore_subscribe_messages=True)
    pubsub.subscribe(['channel'])
    for item in pubsub.listen():
        frame = item["data"]
        yield (b'--frame\r\n'
        	b'Content-Type: image/jpeg\r\n\r\n' +frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, threaded=True)
