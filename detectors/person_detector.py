# person_detector.py
import cv2
import time
import numpy as np
import logging
from util.check_os import is_raspberry_pi_robust
logger = logging.getLogger()

# Import correct version of tensorflow depending on host OS
if is_raspberry_pi_robust():
    import tflite_runtime.interpreter as tf  # type: ignore
else:
    import tensorflow as tf


class PersonDetector:
    def __init__(self, model_path: str, threshold: float = 0.5):
        logger.info(
            f"Using model path='{model_path}', threshold={threshold}"
        )

        self._set_tf_interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.threshold = threshold
        self.last_infer_time = 0
        self._suffix = "_person"

    def _set_tf_interpreter(self, model_path: str):
        try:
            if is_raspberry_pi_robust():
                self.interpreter = tf.Interpreter(model_path=model_path)
            else:
                self.interpreter = tf.lite.Interpreter(model_path=model_path)
        except Exception:
            logger.exception(
                'Error occurred while initializing PersonDetector'
            )

    @property
    def suffix(self):
        return self._suffix

    def detect_person(self, frame) -> float:
        """Returns person score [0,1]; 0 if none detected."""
        h, w = frame.shape[:2]
        inp = cv2.resize(frame, (300, 300))
        inp = np.expand_dims(inp, axis=0).astype(np.uint8)

        self.interpreter.set_tensor(self.input_details[0]['index'], inp)
        t0 = time.time()
        self.interpreter.invoke()
        self.last_infer_time = time.time() - t0

        boxes = self.interpreter.get_tensor(
            self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(
            self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(
            self.output_details[2]['index'])[0]

        best_person_score = 0.0
        for c, s in zip(classes, scores):
            if int(c) == 1 and s > best_person_score:  # COCO class 1 == person
                best_person_score = float(s)
        return best_person_score

    def is_person(self, frame) -> bool:
        """Returns True if model is confident that object is a person."""
        return self.detect_person(frame) >= self.threshold
