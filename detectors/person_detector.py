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
        logger.debug(f"interpreter input details: {self.input_details}")
        self.output_details = self.interpreter.get_output_details()
        logger.debug(f"interpreter output details: {self.output_details}")

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
        inp = cv2.resize(frame, (320, 320))
        inp = np.expand_dims(inp, axis=0).astype(np.uint8)

        self.interpreter.set_tensor(self.input_details[0]['index'], inp)
        t0 = time.time()
        self.interpreter.invoke()
        self.last_infer_time = time.time() - t0

        out_details = self.output_details

        boxes = None
        classes = None
        scores = None

        # Case 1: single structured 'detections' output (common for some EfficientDet-Lite exports)
        if len(out_details) == 1 and (
            'detect' in out_details[0]['name'].lower()
            or (hasattr(out_details[0]['shape'], '__len__') and int(out_details[0]['shape'][-1]) == 7)
        ):
            detections = self.interpreter.get_tensor(out_details[0]['index'])[0]
            # Expected shape: [N, 7] where columns often are [box(4), score, class, ...]
            if detections.ndim == 2 and detections.shape[1] >= 6:
                boxes = detections[:, 0:4]
                scores = detections[:, 4]
                classes = detections[:, 5].astype(int)
                logger.debug("Parsed structured 'detections' output; sample rows: %s", detections[:3].tolist())
            else:
                logger.error("Unexpected 'detections' tensor shape: %s", detections.shape)

        # Case 2: multi-output model (boxes, classes, scores) - be defensive about ordering
        elif len(out_details) >= 3:
            try:
                boxes = self.interpreter.get_tensor(out_details[0]['index'])[0]
                maybe1 = self.interpreter.get_tensor(out_details[1]['index'])[0]
                maybe2 = self.interpreter.get_tensor(out_details[2]['index'])[0]

                def is_prob_array(arr):
                    try:
                        return np.issubdtype(arr.dtype, np.floating) and arr.size > 0 and arr.max() <= 1.0 and arr.min() >= 0.0
                    except Exception:
                        return False

                # Heuristic: whichever of maybe1/maybe2 looks like scores (floats in [0,1]) is scores
                if is_prob_array(maybe1) and not is_prob_array(maybe2):
                    scores = maybe1
                    classes = maybe2.astype(int)
                elif is_prob_array(maybe2) and not is_prob_array(maybe1):
                    scores = maybe2
                    classes = maybe1.astype(int)
                else:
                    # Fallback assumption: maybe1 = classes, maybe2 = scores
                    classes = maybe1.astype(int)
                    scores = maybe2

                logger.debug("Used multi-output mapping; boxes.shape=%s scores.shape=%s classes.shape=%s", getattr(boxes, 'shape', None), getattr(scores, 'shape', None), getattr(classes, 'shape', None))
            except Exception:
                logger.exception("Error reading multi-output tensors; output_details=%s", out_details)

        else:
            logger.error("Unexpected TFLite outputs: %s", out_details)
            return 0.0

        # Ensure parsing produced usable arrays
        if classes is None or scores is None:
            logger.error("Could not parse model outputs into classes/scores. output_details=%s", out_details)
            return 0.0

        # Compute best person score defensively
        best_person_score = 0.0
        try:
            length = min(len(classes), len(scores))
        except Exception:
            logger.exception("Error getting lengths of classes/scores")
            return 0.0

        for c, s in zip(classes[:length], scores[:length]):
            try:
                if int(c) == 1 and float(s) > best_person_score:  # COCO class 1 == person
                    best_person_score = float(s)
            except Exception:
                # Ignore malformed entries
                continue

        return best_person_score

    def is_person(self, frame) -> bool:
        """Returns True if model is confident that object is a person."""
        return self.detect_person(frame) >= self.threshold
