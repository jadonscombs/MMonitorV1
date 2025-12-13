"""
Lightweight motion recorder for Raspberry Pi + USB webcam.
Saves clips with X seconds before detection and Y seconds debounce between events.
Uploads completed clips to S3 and provides live feed accessibility via AWS Kinesis.

Dependencies:
  pip3 install numpy boto3 opencv-python imutils
  (on Pi, install opencv by apt if pip wheel fails)
"""

# flake8: noqa: F405

import logging
from logging.handlers import RotatingFileHandler

import cv2
import time
import datetime
import os
import argparse
import threading
from collections import deque
import boto3
from boto3 import Session
from botocore.exceptions import ClientError, NoCredentialsError
import uuid
import sys


# Import constants
from constants import *

# Logger config
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
log_handler = RotatingFileHandler(
    'monitor-app.log',
    maxBytes=10*KB,
    backupCount=5
)
log_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'
))

# Console logging
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] - %(message)s')
)

# Add log handlers
logger.addHandler(log_handler)
logger.addHandler(console_handler)


# AWS logging config
logging.getLogger('botocore').setLevel(logging.INFO)
logging.getLogger('boto3').setLevel(logging.INFO)


# Import and init object detectors
from detectors.person_detector import PersonDetector
person_detector = PersonDetector(OBJECT_DETECT_MODEL_PATH, threshold=0.6)

# Global S3 client
s3_client = None


# TODO: Add implementation of "Monitor" class
class Monitor:
    def __init__(self):
        pass


# create S3 client (uses env vars or ~/.aws/credentials or instance profile)
def init_s3_client(disable_s3: bool = False):
    if disable_s3:
        logger.info("S3 client DISABLED for this session")
        return None

    logger.info("S3 client ENABLED for this session")
    client = boto3.client(S3_CONST) if S3_UPLOAD else None
    if client is None:
        logger.warning("returned S3 client is NONE")
    else:
        try:
            # Check if client's allowed operations exist
            # - this implicitly confirms the client was initialized
            operations = client.meta.service_model.operation_names
            logger.info(f"S3 client has {len(operations)} allowed operations")
        except NoCredentialsError:
            logger.exception(
                "S3 connection test failed! "
                "AWS credentials could not be found."
            )
        except ClientError:
            logger.exception(
                "S3 client established but operation used during connection "
                "test is unauthorized for registered client!"
            )
        
        logger.info("OK - returned S3 client is not null")
    return client


# Define video encoding for OpenCV
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore


def upload_to_s3_local(filepath, bucket, key):
    try:
        s3_client.upload_file(filepath, bucket, key)
        logger.info(f"Uploaded {filepath} -> s3://{bucket}/{key}")
        return True
    except Exception:
        logger.exception("S3 upload failed!")
        return False
    

def uploader_thread(filepath, disable_s3: bool):
    if disable_s3:
        return

    if not S3_UPLOAD:
        return
    key = S3_PREFIX + os.path.basename(filepath)
    upload_to_s3_local(filepath, S3_BUCKET, key)
    # Optionally delete local file after upload:
    try:
        os.remove(filepath)
    except Exception:
        pass


def get_time_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime(ISOFORMAT)


def run_object_detectors_on_frame(frame):
    """
    Returns a filename suffix denoting whether the frame's associated recording
    captured a categorized object.

    If no recognized object detected, then ('', 0.0) is returned.
    Else, return (<object_suffix>, <object_confidence_score>).
    """
    person_score = person_detector.detect_person(frame)
    if person_score >= person_detector.threshold:
        return (person_detector.suffix, person_score)
    return ('', 0.0)


# Core logic
def start_monitor(disable_s3: bool = False):
    logger.info("Starting MMonitor Application...")

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    logger.info("CV2 VideoCapture initialized...")

    # Check if camera is accessible
    if not cap.isOpened():
        logger.error(f"ERROR: Cannot open camera '{CAM_INDEX}'")
        return
    else:
        logger.info("OK - Capture hardware is accessible")
    
    # MOG2 background subtractor
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=25,
        detectShadows=True
    )
    logger.info("OK - Created MOG2 background subractor")

    # Loop variables
    frame_buffer = deque(maxlen=PRE_SECONDS * FPS + 5)
    recording = False
    writer = None
    post_timer = None
    last_clip_end = 0.0

    logger.info("Starting capture. Ctrl+C to exit.")
    try:
        while True:
            start = time.time()
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame grab failed.")
                time.sleep(0.1)
                continue
            
            # grab timestamp closer to detection time
            now = time.time()

            # optionally resize for speed (already set)
            frame = cv2.resize(frame, (WIDTH, HEIGHT))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fgmask = bg_sub.apply(gray)

            # morphological ops to reduce noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel, iterations=1)

            # find contours
            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            motion_detected = False
            for c in contours:
                if cv2.contourArea(c) > MIN_CONTOUR_AREA:
                    motion_detected = True
                    break

            # push frame into pre-buffer
            frame_buffer.append(frame.copy())

            # handle detection
            if motion_detected:
                #logger.debug("motion detected!")

                # enforce debounce: only start new recording if enough time since last clip ended
                #
                # e.g., "if not yet recording, and at least <DEBOUNCE_SECONDS"
                # of stillness/no motion has passed since the last recording,
                # then prepare variables and data for a new recording"
                if not recording and (now - last_clip_end) >= DEBOUNCE_SECONDS:
                    # start writer and pre-fill with buffer
                    filename = f"motion_{get_time_now()}_{uuid.uuid4().hex[:8]}.mp4"
                    filepath = os.path.join(OUT_DIR, filename)
                    writer: cv2.VideoWriter = cv2.VideoWriter(filepath, fourcc, FPS, (frame.shape[1], frame.shape[0]))
                    # write pre-buffer
                    for bf in frame_buffer:
                        writer.write(bf)
                    recording = True
                    logger.info(f"Started recording: {filepath}")

                # reset/extend post-timer;
                # this only gets updated if more motion is detected before 
                # <POST_SECONDS> seconds since the initial recording start
                post_timer = now + POST_SECONDS

            # if recording, write current frame
            if recording and writer is not None:
                writer.write(frame)

            # check post-timer expiration
            #
            # logic: if recording True, and <post_timer> initialized (which is only initialized when motion is detected),
            #        and current timestamp is greater than the logged <post_timer>, then finalize recording
            if recording and post_timer is not None and now > post_timer:

                # finalize
                recording = False
                writer.release()
                writer = None
                last_clip_end = time.time()
                time_now = datetime.datetime.now(datetime.timezone.utc).isoformat()

                # run person detector on last frame of clip
                object_suffix, object_score = run_object_detectors_on_frame(frame)
                new_filename = filepath.replace(".mp4", f"{object_suffix}.mp4")
                os.rename(filepath, new_filename)
                filepath = new_filename
                logger.info(f"Finished recording at {time_now} UTC; person_score={object_score:.2f}")

                # upload in background thread
                t = threading.Thread(target=uploader_thread, args=(filepath, disable_s3))
                t.daemon = True
                t.start()

                # clear buffer so pre-buffer applies only new frames
                frame_buffer.clear()

            # throttle to target FPS
            elapsed = time.time() - start
            sleep_seconds = max(0, (1.0 / FPS) - elapsed)
            time.sleep(sleep_seconds)

    except KeyboardInterrupt:
        logger.warning("Stopping capture.")
    finally:
        if writer is not None:
            writer.release()
        cap.release()


# Monitor entrypoint
if __name__ == "__main__":
    # Temporary - later, use ArgParse
    disable_s3 = True if '--disable-s3' in sys.argv else False
    s3_client = init_s3_client(disable_s3=disable_s3)
    logger.info(f"[VERIFICATION] disable_s3={disable_s3}; s3_client={s3_client}")

    start_monitor(disable_s3=disable_s3)
