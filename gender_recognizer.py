#!/usr/bin/env python
# import OpenCV before mxnet to avoid a segmentation fault
import cv2

from config import age_gender_deploy as deploy
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.preprocessing import MeanPreprocessor
from pyimagesearch.preprocessing import CropPreprocessor
from pyimagesearch.utils import AgeGenderHelper
import imutils
import pickle
import json
import numpy as np
import mxnet as mx
from imutils.face_utils import FaceAligner
from imutils import face_utils
from detect_picture_utils import log
import dlib
import os

class GenderRecognizer:
    # load the label encoders and mean files
    print("[INFO] loading label encoders and mean files...")
    ageLE = pickle.loads(open(deploy.AGE_LABEL_ENCODER, "rb").read())
    genderLE = pickle.loads(open(deploy.GENDER_LABEL_ENCODER, "rb").read())
    ageMeans = json.loads(open(deploy.AGE_MEANS).read())
    genderMeans = json.loads(open(deploy.GENDER_MEANS).read())

    # load the models from disk
    print("[INFO] loading models...")
    agePath = os.path.sep.join([deploy.AGE_NETWORK_PATH,
        deploy.AGE_PREFIX])
    genderPath = os.path.sep.join([deploy.GENDER_NETWORK_PATH,
        deploy.GENDER_PREFIX])
    ageModel = mx.model.FeedForward.load(agePath, deploy.AGE_EPOCH)
    genderModel = mx.model.FeedForward.load(genderPath,
        deploy.GENDER_EPOCH)

    # now that the networks are loaded, we need to compile them
    print("[INFO] compiling models...")
    ageModel = mx.model.FeedForward(ctx=[mx.gpu(0)],
        symbol=ageModel.symbol, arg_params=ageModel.arg_params,
        aux_params=ageModel.aux_params)
    genderModel = mx.model.FeedForward(ctx=[mx.gpu(0)],
        symbol=genderModel.symbol, arg_params=genderModel.arg_params,
        aux_params=genderModel.aux_params)

    # initialize the image pre-processors
    sp = SimplePreprocessor(width=256, height=256,
        inter=cv2.INTER_CUBIC)
    cp = CropPreprocessor(width=227, height=227, horiz=True)
    ageMP = MeanPreprocessor(ageMeans["R"], ageMeans["G"],
        ageMeans["B"])
    genderMP = MeanPreprocessor(genderMeans["R"], genderMeans["G"],
        genderMeans["B"])
    iap = ImageToArrayPreprocessor(dataFormat="channels_first")

   
    def __init__(self):
        # initialize dlib's face detector (HOG-based), then create the
        # the facial landmark predictor and face aligner
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(deploy.DLIB_LANDMARK_PATH)
        self.fa = FaceAligner(self.predictor)


    def recognize(self, face):
        # resize the face to a fixed size, then extract 10-crop
        # patches from it
        face = self.sp.preprocess(face)
        patches = self.cp.preprocess(face)
        # allocate memory for the age and gender patches
        agePatches = np.zeros((patches.shape[0], 3, 227, 227),
            dtype="float")
        genderPatches = np.zeros((patches.shape[0], 3, 227, 227),
            dtype="float")
        # loop over the patches
        for j in np.arange(0, patches.shape[0]):
            # perform mean subtraction on the patch
            agePatch = self.ageMP.preprocess(patches[j])
            genderPatch = self.genderMP.preprocess(patches[j])
            agePatch = self.iap.preprocess(agePatch)
            genderPatch = self.iap.preprocess(genderPatch)
            # update the respective patches lists
            agePatches[j] = agePatch
            genderPatches[j] = genderPatch
        # make predictions on age and gender based on the extracted
        # patches
        agePreds = self.ageModel.predict(agePatches)
        genderPreds = self.genderModel.predict(genderPatches)
        # compute the average for each class label based on the
        # predictions for the patches
        agePreds = agePreds.mean(axis=0)
        genderPreds = genderPreds.mean(axis=0)
        # visualize the age and gender predictions
        age = GenderRecognizer.visAge(agePreds, self.ageLE)
        gender = GenderRecognizer.visGender(genderPreds,self.genderLE)
        image_to_show_path_GA = "images/{}/{}".format(gender,age)
        
        return image_to_show_path_GA
      

    @staticmethod
    def visAge(agePreds, le):
        # initialize the canvas and sort the predictions according
        # to their probability
        idxs = np.argsort(agePreds)[::-1]
        # construct the text for the prediction
        #ageLabel = le.inverse_transform(j) # Python 2.7
        ageLabel = le.inverse_transform(idxs[0]).decode("utf-8")
        ageLabel = ageLabel.replace("_", "-")
        ageLabel = ageLabel.replace("-inf", "+")
        age = "{}".format(ageLabel)
        return age

    @staticmethod
    def visGender(genderPreds, le):
        idxs = np.argsort(genderPreds)[::-1]
        gender = le.inverse_transform(idxs[0])
        gender = "Male" if gender == 0 else "Female"
        text_gender = "{}".format(gender)
        return text_gender