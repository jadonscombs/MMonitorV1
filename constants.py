import os

# -------- CONFIG ----------
CAM_INDEX = 0
FPS = 10  # lower fps reduces CPU and file size on Pi; adjust as needed
WIDTH = 640
HEIGHT = 360

PRE_SECONDS = 5         # X seconds before detection
POST_SECONDS = 5        # seconds to continue recording after last motion
DEBOUNCE_SECONDS = 10   # Y seconds minimum between clips
MIN_CONTOUR_AREA = 500  # tune to ignore small noise

S3_UPLOAD = True
S3_CONST = 's3'
S3_BUCKET = 's3-motion-cap-bucket'
S3_PREFIX = 'captures/'

OUT_DIR = '/tmp/motion_clips'  # local temporary store
os.makedirs(OUT_DIR, exist_ok=True)

OBJECT_DETECT_MODEL_PATH = "models/efficientdet-lite0.tflite"
# --------------------------

ISOFORMAT = '%Y%m%dT%H%M%SZ'
KB = 1024
MB = 1024 * 1024
