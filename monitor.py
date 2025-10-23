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
import uuid
import sys


# Import constants
from constants import *


# Logger config
logger = logging.getLogger()
log_handler = RotatingFileHandler(
    'mmonitor-app.log',
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

logger.addHandler(log_handler)
logger.addHandler(console_handler)


# create S3 client (uses env vars or ~/.aws/credentials or instance profile)
s3_client = boto3.client('s3') if S3_UPLOAD else None

# Define video encoding for OpenCV
fourcc = cv2.VideoWriter_fourcc(*'mp4v')


def upload_to_s3_local(filepath, bucket, key):
    try:
        s3_client.upload_file(filepath, bucket, key)
        logger.info(f"Uploaded {filepath} -> s3://{bucket}/{key}")
        return True
    except Exception:
        logger.exception("S3 upload failed!")
        return False
    

def uploader_thread(filepath):
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
    return datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')


# Core logic
def start_monitor():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    # Check if camera is accessible
    if not cap.isOpened():
        logger.error(f"ERROR: Cannot open camera '{CAM_INDEX}'")
        return
    
    # MOG2 background subtractor
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=25,
        detectShadows=True
    )

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

            # optionally resize for speed (already set)
            # frame = cv2.resize(frame, (WIDTH, HEIGHT))

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
            now = time.time()
            if motion_detected:
                # enforce debounce: only start new recording if enough time since last clip ended
                if not recording and (now - last_clip_end) >= DEBOUNCE_SECONDS:
                    # start writer and pre-fill with buffer
                    filename = f"motion_{get_time_now()}_{uuid.uuid4().hex[:8]}.mp4"
                    filepath = os.path.join(OUT_DIR, filename)
                    writer = cv2.VideoWriter(filepath, fourcc, FPS, (frame.shape[1], frame.shape[0]))
                    # write pre-buffer
                    for bf in frame_buffer:
                        writer.write(bf)
                    recording = True
                    logger.info(f"Started recording: {filepath}")
                # reset/extend post-timer
                post_timer = now + POST_SECONDS

            # if recording, write current frame
            if recording and writer is not None:
                writer.write(frame)

            # check post-timer expiration
            if recording and post_timer is not None and now > post_timer:
                # finalize
                recording = False
                writer.release()
                writer = None
                last_clip_end = time.time()
                logger.info(f"Finished recording at {datetime.datetime.utcnow().isoformat()} utc")

                # upload in background thread
                t = threading.Thread(target=uploader_thread, args=(filepath,))
                t.daemon = True
                t.start()

                # clear buffer so pre-buffer applies only new frames
                frame_buffer.clear()

            # throttle to target FPS
            elapsed = time.time() - start
            sleep = max(0, (1.0 / FPS) - elapsed)
            time.sleep(sleep)

    except KeyboardInterrupt:
        logger.warning("Stopping capture.")
    finally:
        if writer is not None:
            writer.release()
        cap.release()


# Monitor entrypoint
if __name__ == "__main__":
    start_monitor()
