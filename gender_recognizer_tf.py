import argparse
import os
from detect_picture_utils import log
import cv2
import dlib
#import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import inception_resnet_v1

def draw_label(image, point, ages, genders,font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1, thickness=2):
    for i in range(len(point)):
        label = "{}, {}".format(int(ages[i]), "F" if genders[i] == 0 else "M")
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point[i]
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point[i], font, font_scale, (255, 255, 255), thickness)

class GenderRecognizer:
    def __init__(self):
        self.model_path = "./models"
        self.shape_detector="shape_predictor_68_face_landmarks.dat"
        cuda=True
        font_scale=1
        thickness=1
        #log('1')
        print('1')
        with tf.Graph().as_default():
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.4
            # session = tf.Session(config=config, ...)

            self.sess = tf.Session(config=config)
            self.images_pl = tf.placeholder(tf.float32, shape=[None, 160, 160, 3], name='input_image')
            images = tf.map_fn(lambda frame: tf.reverse_v2(frame, [-1]), self.images_pl) #BGR TO RGB
            images_norm = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), images)
            self.train_mode = tf.placeholder(tf.bool)
            age_logits, gender_logits, _ = inception_resnet_v1.inference(images_norm, keep_probability=0.8,
                                                                         phase_train=self.train_mode,
                                                                         weight_decay=1e-5)
            self.gender = tf.argmax(tf.nn.softmax(gender_logits), 1)
            age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
            self.age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
            self.sess.run(init_op)
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(self.model_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(self.sess, ckpt.model_checkpoint_path)
                print("restore and continue training!")
            else:
                pass
    
    def recognize(self, aligned_image, image, rect_nums, XY):
        print('2')
        ages, genders = self.eval(aligned_image, self.model_path)
        age = int(ages[0])
        if age < 39 :
            age = "adult"
        elif age >= 40 and age < 49:
            age = "senior"
        elif age >= 50:
            age = "mature" 

        gender = "female" if genders[0] == 0 else "male"
        log("[gender_recognizer_tf] Pessoa sexo: {} , idade: {}".format(gender,int(ages[0])))
        image_to_show_path_GA = "images/{}/{}".format(gender,age)
        return image_to_show_path_GA,gender,age
    
    def eval(self,aligned_images, model_path):
        aligned_images = [imutils.resize(aligned_images[0], width=160)]
        return self.sess.run([self.age, self.gender], feed_dict={self.images_pl: aligned_images, self.train_mode: False}) 
   
