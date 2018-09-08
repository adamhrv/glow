# override filepaths in demo files
from pathlib import Path
from os.path import join

DATA_STORE = '/data_store'
DIR_APP = join(DATA_STORE, 'apps/megapixels/')

# Glow
FP_GLOW_OPTIMIZED = join(DIR_APP, 'glow/graph_optimized.pb')
FP_GLOW_UNOPTIMIZED = join(DIR_APP, 'glow/graph_unoptimized.pb')
FP_GLOW_ATTR = join(DIR_APP, 'glow/attr.npy')
FP_GLOW_X = join(DIR_APP, 'glow/x.npy')
FP_GLOW_Y = join(DIR_APP, 'glow/y.npy')
FP_GLOW_Z = join(DIR_APP, 'glow/z_manipulate.npy')
USE_OPTIMIZED = True

# dlib
FP_DLIB_PREDICTOR = join(DIR_APP, 'dlib/shape_predictor_68_face_landmarks.dat')

# opencv
# should be preinstalled with pip install opencv-python==3.4.2