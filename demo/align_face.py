# OLD USAGE
# python align_faces.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg
import os

# import the necessary packages
from imutils.face_utils import FaceAligner
from PIL import Image
import numpy as np
# import argparse
import imutils
import dlib
import cv2

if os.path.isdir('app/settings'):
    import app.settings.app_cfg as cfg
else:
    import settings as cfg

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("--shape-predictor", help="path to facial landmark predictor", default='shape_predictor_68_face_landmarks.dat')
# ap.add_argument("--input", help="path to input images", default='input_raw')
# ap.add_argument("--output", help="path to input images", default='input_aligned')
# args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(cfg.FP_DLIB_PREDICTOR)
fa = FaceAligner(predictor, desiredFaceWidth=256,
                 desiredLeftEye=(0.371, 0.480))


# Input: numpy array for image with RGB channels
# Output: (numpy array, face_found)
def align_face(img, expand):
    img = img[:, :, ::-1]  # Convert from RGB to BGR format
    img = imutils.resize(img, width=800)

    # detect faces in the grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 2)

    if len(rects) > 0:
        # align the face using facial landmarks
        x1, y1 = rects[0].tl_corner().x - expand, rects[0].tl_corner().y - expand
        x2, y2 = rects[0].br_corner().x + expand, rects[0].br_corner().y + expand
        expanded = dlib.rectangle(x1, y1, x2, y2)
        align_img = fa.align(img, gray, expanded)[:, :, ::-1]
        align_img = np.array(Image.fromarray(align_img).convert('RGB'))
        return align_img, True
    else:
        # No face found
        return None, False

# Input: img_path
# Output: aligned_img if face_found, else None
def align(img_path, expand=0):
    img = Image.open(img_path)
    img = img.convert('RGB')  # if image is RGBA or Grayscale etc
    img = np.array(img)
    x, face_found = align_face(img, expand)
    return x