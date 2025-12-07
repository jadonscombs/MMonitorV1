import logging
import os

logger = logging.getLogger()


def is_raspberry_pi_robust():
    # Check for /proc/device-tree/model
    if os.path.exists('/proc/device-tree/model'):
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read()
            if "Raspberry Pi" in model:
                return True

    # Check for RPi.GPIO module
    try:
        import RPi.GPIO as GPIO  # noqa: F401 # type: ignore
        return True
    except ImportError:
        pass

    return False


if is_raspberry_pi_robust():
    logger.info("Running on a Raspberry Pi (robust check).")
else:
    logger.info("Not running on a Raspberry Pi (robust check).")
