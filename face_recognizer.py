#!/usr/bin/env python

import face_recognition.api as face_recognition
from time_average import TimeAverage
from person import UnknownPerson
from detect_picture_utils import log

class FaceRecognizer:
    people = []
    known_face_encodings = []
    face_in_frame = TimeAverage(2)
    last_rects = []

    def __init__(self, people):
        self.set_people(people)

    def set_people(self, people):
        self.people = people
        self.known_face_encodings = [p.encoding for p in people if p.encoding is not None]

    def recognize(self, frame):
        import face_recognition.api as face_recognition

        present = []
        faces_rects = face_recognition.face_locations(frame)
        num_faces = len(faces_rects)
        self.face_in_frame.update(num_faces)
        if num_faces > 0:
            self.last_rects = faces_rects
        
        if self.face_in_frame.detected():
            # Find all the face encodings in the frame of video
            face_encodings = face_recognition.face_encodings(frame, self.last_rects)
            
            # Loop through each face in this frame of video
            for face_encoding in face_encodings:
                distances =  face_recognition.face_distance(self.known_face_encodings, face_encoding)
                dist_people = zip(distances, self.people)
                dist_people = sorted(dist_people, key=lambda distances: distances[0])
                if len(dist_people) > 0:
                    distance, person = dist_people[0]
                    if distance < 0.51 :
                        log("!!!!!!!!!!!!!!!!!!!!!! DISTANCE {} !!!!!!!".format(distance))
                        was_in_the_house = person.in_the_house
                        person.set_state(True)
                        
                        if person.recognized():
                            present.append(person)
                            UnknownPerson().reset()
                        # if was_in_the_house and not person.in_the_house:
                        #     present.append(person)
                        #     UnknownPerson().reset()
                    # no one known in the frame
                    else:
                        UnknownPerson().entered_the_frame()
        else:
            self.last_rects = []
            for person in self.people:
                person.reset()
            UnknownPerson().reset()
        return self.last_rects, present, self.face_in_frame.percent()

    # def recognize(self, frame):
    #     # import face_recognition.api as face_recognition

    #     present = []
    #     faces_rects = face_recognition.face_locations(frame)
    #     num_faces = len(faces_rects)
    #     self.face_in_frame.update(num_faces)
    #     if num_faces > 0:
    #         self.last_rects = faces_rects
        
    #     if self.face_in_frame.detected():
    #         # Find all the face encodings in the frame of video
    #         face_encodings = face_recognition.face_encodings(frame, self.last_rects)
            
    #         # Loop through each face in this frame of video
    #         for face_encoding in face_encodings:
    #             # See if the face is a match for the known face(s)
    #             matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.57)
                
    #             # DONT LIKE THIS METHOD WE SHOULD CHANGE IT ! 
    #             # debug (only UnknownUser): matches = [False for m in matches]
    #             for is_match, person in zip(matches, self.people):
    #                 was_in_the_house = person.in_the_house
    #                 person.set_state(is_match)
                    
    #                 if person.recognized():
    #                     present.append(person)
    #                     UnknownPerson().reset()
    #                 # if was_in_the_house and not person.in_the_house:
    #                 #     present.append(person)
    #                 #     UnknownPerson().reset()
    #             # no one known in the frame
    #             if True not in matches:
    #                 UnknownPerson().entered_the_frame()
    #     else:
    #         self.last_rects = []
    #         for person in self.people:
    #             person.reset()
    #         UnknownPerson().reset()
    #     return self.last_rects, present, self.face_in_frame.percent()