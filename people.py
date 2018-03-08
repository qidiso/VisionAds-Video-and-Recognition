import time
import os 
import person
import json 
import urllib
import re
import cv2 
import imutils
from person import Person,UnknownPerson

hostname = os.environ.get('API_HOSTNAME') or 'localhost'


# ------------------------------------------------------------------------------
class People:
    def __init__(self, known_people_folder=None):
        self.last_refreshed = time.time()
        self.people = []
        self.known_people_folder = known_people_folder
        self.load()

    def refresh(self):
        current_time = time.time()
        if current_time - self.last_refreshed < 60:
            return False
        self.last_refreshed = current_time
        self.load()
        return True

    def load(self):
        print('Loading people...' + str(time.ctime()))
        self.people = (self.__scan_known_people()
                       if self.known_people_folder
                       else self.__download_known_people())
        self.people.append(UnknownPerson())
        print('done.'+ str(time.ctime()))

    def __scan_known_people(self):
        images = [os.path.join(self.known_people_folder, f) for f in
                  os.listdir(self.known_people_folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]
        print('Extracting features of {} people...'.format(len(images)))
       
        return [Person.fromfile(file) for file in images]

    @staticmethod
    def __download_known_people():
        global hostname
        people_url = 'http://{0}:3000/visitors.json'.format(hostname)
        response = urllib.request.urlopen(people_url)
        str_response = response.read().decode('utf-8')
        obj = json.loads(str_response)
        return [Person.fromjson(o) for o in obj]
    
    @staticmethod
    def draw_name(frame, face_names):
        if len(face_names) < 1:
            return

        unknown = 'Desconhecido'
        name = face_names[0] if face_names[0] != UnknownPerson().name else unknown
        name = name.upper()
        frame_h, frame_w, _ = frame.shape
        padding = 16
        color = (0,255,0)
        fontFace = cv2.FONT_HERSHEY_DUPLEX
        fontScale = 1.0
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(name, fontFace, fontScale, thickness)
        cv2.putText(frame, name, (int((frame_w-text_w)/2), frame_h - padding), fontFace, fontScale, color, thickness)

    @staticmethod
    def draw_in_the_frame_picture(frame, people,shape):
        from gui_face_finder import GUIFaceFinder

        if len(people) < 1:
            return
        person = people[0]
        if person.imgFile is None:
            return

        (roi_top, roi_right, roi_bottom, roi_left) = GUIFaceFinder.css_roi_for_frame_shape(frame.shape)
        roi_h = roi_bottom - roi_top
        roi_w = roi_right - roi_left

        face = person.imgFile.copy()
        face = imutils.resize(face, width=shape[1])
        face_h, face_w, _ = face.shape

        x,y = 0, int(max(0, (face_h - roi_h)/2))
        w,h = roi_w, int(min(face_h, (face_h + roi_h)/2))
        face = face[y:y+h, x:x+w]

        padding = 8
        h, w, _ = face.shape
        x1,y1 = ( int((roi_right+roi_left)/2)+padding, int((roi_bottom+roi_top-h)/2) )

        frame[y1:y1+h, x1:x1+w] = face
        cv2.rectangle(frame, (x1,y1), (x1+w,y1+h), (0,255,0), 2)
