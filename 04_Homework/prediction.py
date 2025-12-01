from collections import deque
from pathlib import Path
from time import time
from typing import Dict, Optional, Tuple

import numpy as np
import openvino as ov
from numpy import ndarray
from openvino import CompiledModel
from PIL import Image

# ---------------- Config ----------------
DEFAULT_SPEED = 75.0        # regular speed
SLOW_SPEED = 20.0           # speed after a 50Sign
SIGN_THRESHOLD = 0.75       # how certain the model should be before registering a sign

STOP_TIME_BUFFER = 2.0      # ignore repeated stop within this time
STOP_TIMEOUT = 3.0          # seconds to hold a full stop – DO NOT CHANGE!
SLOW_SPEED_MAX_DURATION = 500.0   # seconds to stay slow after 50Sign before returning to normal speed
SLOW_SPEED_MIN_DURATION = 0.0   # minimum seconds to stay slow after 50Sign before returning to normal speed
CONSECUTIVE_SIGN_THRESHOLD = 25  # number of consecutive detections required to confirm a sign
TOP_CROP=30

DRIVE_MODEL_NAME = 'DriveModel_v1.onnx'
SIGN_MODEL_NAME = 'SignModel.onnx'

MEM_SIZE = 12

IS_CAMEL_RACE = False

# ---------------- State ----------------
_last_detected_time: float = 0.0
_last_detected_sign: Optional[str] = None
_last_speed: float = DEFAULT_SPEED
_slow_speed_start_time: float = 0.0
last_confirmed_sign: Optional[str] = None
_consecutive_sign_count: int = 0
_consecutive_sign_type: Optional[str] = None

angle_history = deque(maxlen=MEM_SIZE)
_frame_counter: int = 2
_cached_signs: Dict[str, float] = {}


# ---------------- Load ----------------
def load(model_dir: str) -> Tuple[CompiledModel, CompiledModel]:
    """This functions gets called every time the side button on the remote is pressed in self-driving mode.
    The function loads both models on the raspberry pi."""
    global _drive_input_name, _sign_input_name, angle_history, _frame_counter, _cached_signs
    
    angle_history = deque([0.0]*MEM_SIZE, maxlen=MEM_SIZE)
    _frame_counter = 0
    _cached_signs = {}

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
    global STOP_TIMEOUT, STOP_TIME_BUFFER, angle_history, _frame_counter, _cached_signs

    drive_model, sign_model = models
    now = time()

    # enforce STOP hold
    if now < _last_detected_time + STOP_TIMEOUT:
        _last_detected_sign = None
        return 0.0, 0.0, {}

    drive_image, sign_image = img_to_tensor(img)

    angle_history_list = list(reversed(angle_history))
    if len(angle_history_list) < MEM_SIZE:
        angle_history_list += [0.0] * (MEM_SIZE - len(angle_history_list))
    else:
        angle_history_list = angle_history_list[:MEM_SIZE]

    angle_history_array = np.array(angle_history_list, dtype=np.float32).reshape(1, MEM_SIZE)

    angle = predict_angle(drive_model, drive_image, angle_history_array)
    
    angle_history.append(angle)
    
    # Run sign detection only every 3rd frame
    _frame_counter += 1
    if _frame_counter % 3 == 0 or IS_CAMEL_RACE:
        signs = predict_sign(sign_model, sign_image)
        _cached_signs = signs
    else:
        signs = _cached_signs

    chosen = resolve_sign(signs, now)
    if IS_CAMEL_RACE:
        speed = map_speed_to_sign_old(chosen, now)
    else:
        speed = map_speed_to_sign(chosen, now)

    return angle, speed, signs


# ---------------- Inference ----------------

def predict_angle(drive_model: CompiledModel, img: ndarray, angle_history: ndarray) -> float:
    """Run drive model inference on an image"""
    result = drive_model([img, angle_history])
    out = result[0]
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
    global _last_detected_time, _last_speed, last_confirmed_sign, _slow_speed_start_time
    global _consecutive_sign_count, _consecutive_sign_type

    # Check if we should automatically return to normal speed after SLOW_SPEED_DURATION
    if _slow_speed_start_time > 0 and now >= _slow_speed_start_time + SLOW_SPEED_MAX_DURATION:
        _last_speed = DEFAULT_SPEED
        _slow_speed_start_time = 0.0
        last_confirmed_sign = None

    # Count consecutive detections of the same sign
    if sign == _consecutive_sign_type:
        _consecutive_sign_count += 1
    else:
        _consecutive_sign_type = sign
        _consecutive_sign_count = 1

    # Only confirm sign after n consecutive detections
    confirmed_sign = None
    if _consecutive_sign_count >= CONSECUTIVE_SIGN_THRESHOLD:
        confirmed_sign = sign

    if last_confirmed_sign == confirmed_sign:
        return _last_speed
    
    if confirmed_sign == 'StopSign':
        _last_detected_time = now

    if last_confirmed_sign == '50Sign':
        _last_speed = SLOW_SPEED
        _slow_speed_start_time = now  # Start the slow speed timer
    elif last_confirmed_sign == 'ClearSign' and now >= _slow_speed_start_time + SLOW_SPEED_MIN_DURATION:
        _last_speed = DEFAULT_SPEED
        _slow_speed_start_time = 0.0  # Reset the timer

    last_confirmed_sign = confirmed_sign
    return _last_speed

# Old/basic version of this function, for camel race
def map_speed_to_sign_old(sign: str, now: float) -> float:
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
    drive_croped = drive_resized.crop((0, TOP_CROP, 160, 120))
    drive_arr = np.array(drive_croped, dtype=np.float32)  # [240, 320, 3]
    drive_arr *= 1.0 / 255.0
    drive_batched = np.transpose(drive_arr, (2, 0, 1))[None, ...]  # [1, 3, 84, 160]

    return drive_batched, sign_batched


def _softmax(x: np.ndarray) -> np.ndarray:
    """Converts the predicted logits to probabilities"""
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)
