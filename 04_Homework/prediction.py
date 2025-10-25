from pathlib import Path
from time import time
from typing import Dict, Tuple, Optional

import numpy as np
from numpy import ndarray
import openvino as ov
from openvino import CompiledModel
from PIL import Image

# ---------------- Config ----------------
DEFAULT_SPEED = 75.0        # regular speed
SLOW_SPEED = 20.0           # speed after a 50Sign
SIGN_THRESHOLD = 0.75       # how certain the model should be before registering a sign

STOP_TIME_BUFFER = 2.0      # ignore repeated stop within this time
STOP_TIMEOUT = 3.0          # seconds to hold a full stop – DO NOT CHANGE!

DRIVE_MODEL_NAME = 'unequaled-skink-546_New_v4_2_v3_e30.onnx'
SIGN_MODEL_NAME = 'delightful-crane-77.onnx'

# ---------------- State ----------------
_last_detected_time: float = 0.0
_last_detected_sign: Optional[str] = None
_last_speed: float = DEFAULT_SPEED


# ---------------- Load ----------------
def load(model_dir: str) -> Tuple[CompiledModel, CompiledModel]:
    """This functions gets called every time the side button on the remote is pressed in self-driving mode.
    The function loads both models on the raspberry pi."""
    global _drive_input_name, _sign_input_name

    model_dir = Path(model_dir)

    drive_model_path = model_dir / DRIVE_MODEL_NAME
    sign_model_path = model_dir / SIGN_MODEL_NAME

    assert drive_model_path.exists(), f'Model does not exist: {drive_model_path}'
    assert sign_model_path.exists(), f'Model does not exist: {sign_model_path}'

    core = ov.Core()

    drive_model = core.read_model(drive_model_path)
    drive_model = core.compile_model(drive_model, 'CPU')

    sign_model = core.read_model(sign_model_path)
    sign_model = core.compile_model(sign_model, 'CPU')

    return drive_model, sign_model


# ---------------- Step ----------------
def step(img, models) -> tuple[float, float, Dict[str, float]]:
    """This function gets called for every image from the cars camera"""
    global _last_detected_time, _last_detected_sign, _last_speed
    global STOP_TIMEOUT, STOP_TIME_BUFFER

    drive_model, sign_model = models
    now = time()

    # enforce STOP hold
    if now < _last_detected_time + STOP_TIMEOUT:
        _last_detected_sign = None
        return 0.0, 0.0, {}

    drive_image, sign_image = img_to_tensor(img)

    angle = predict_angle(drive_model, drive_image)
    signs = predict_sign(sign_model, sign_image)

    chosen = resolve_sign(signs, now)
    speed = map_speed_to_sign(chosen, now)

    return angle, speed, signs


# ---------------- Inference ----------------

def predict_angle(drive_model: CompiledModel, img: ndarray) -> float:
    """Run drive model inference on an image"""
    out = drive_model(img)[0]
    return float(np.array(out).ravel()[0])


def predict_sign(sign_model: CompiledModel, img: ndarray) -> Dict[str, float]:
    """Run sign model inference on an image"""
    logits = sign_model(img)[0][0]
    probs = _softmax(logits.astype(np.float32))
    labels = ('50Sign', 'ClearSign', 'NoSign', 'StopSign')
    return {label: float(p) for label, p in zip(labels, probs)}


def resolve_sign(probs: Dict[str, float], now: float) -> str:
    """Handle the sign according to previous matches and the sign threshold"""
    global _last_detected_sign, _last_detected_time

    label, conf = max(probs.items(), key=lambda kv: kv[1])
    chosen = 'NoSign'

    # only trigger sign if there are 2 matches in a row
    if conf >= SIGN_THRESHOLD:
        if label == _last_detected_sign:
            chosen = label
        else:
            _last_detected_sign = label

    # ignore StopSign retriggers STOP_TIME_BUFFER seconds after restarting to drive
    if chosen == 'StopSign' and now < _last_detected_time + STOP_TIMEOUT + STOP_TIME_BUFFER:
        chosen = 'NoSign'

    return chosen


def map_speed_to_sign(sign: str, now: float) -> float:
    """Assign each sign the corresponding speed"""
    global _last_detected_time, _last_speed

    if sign == '50Sign':
        _last_speed = SLOW_SPEED
    elif sign == 'ClearSign':
        _last_speed = DEFAULT_SPEED
    elif sign == 'StopSign':
        _last_detected_time = now
    return _last_speed


# ---------------- Preprocessing ----------------

def img_to_tensor(img: Image.Image) -> tuple[ndarray, ndarray]:
    """Convert the images to tensors for the model input"""
    assert img.size == (320, 240), f'Expected image to be of size (320, 240), not {img.size}'

    # sign
    sign_arr = np.array(img, dtype=np.float32)  # [240, 320, 3]
    sign_arr *= 1.0 / 255.0
    sign_batched = np.transpose(sign_arr, (2, 0, 1))[None, ...]  # [1, 3, 240, 320]

    # drive
    drive_resized = img.resize((160, 120), resample=Image.Resampling.NEAREST)
    drive_arr = np.array(drive_resized, dtype=np.float32)  # [240, 320, 3]
    drive_arr *= 1.0 / 255.0
    drive_batched = np.transpose(drive_arr, (2, 0, 1))[None, ...]  # [1, 3, 84, 160]

    return drive_batched, sign_batched


def _softmax(x: np.ndarray) -> np.ndarray:
    """Converts the predicted logits to probabilities"""
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)
