from collections import deque
from pathlib import Path
from typing import Dict



import numpy as np
import openvino as ov
from numpy import ndarray
from openvino import CompiledModel
from PIL import Image

# ---------------- Config ----------------
SPEED = 75.0        # regular speed
DRIVE_MODEL_NAME = "DriveModel_v1.onnx"
MEM_SIZE = 12

angle_history = deque(maxlen=MEM_SIZE)

# ---------------- Load ----------------
def load(model_dir: str) -> CompiledModel:
    """This functions gets called every time the side button on the remote is pressed in self-driving mode"""
    global angle_history
    angle_history = deque([0.0]*MEM_SIZE, maxlen=MEM_SIZE)

    model_dir = Path(model_dir)

    drive_model_path = model_dir / DRIVE_MODEL_NAME

    assert drive_model_path.exists(), f'Model does not exist: {drive_model_path}'

    core = ov.Core()
    drive_model = core.read_model(drive_model_path)
    drive_model = core.compile_model(drive_model, 'CPU')
    return drive_model


# ---------------- Step ----------------
def step(img, model) -> tuple[float, float, Dict[str, float]]:
    """This function gets called for every image from the cars camera"""
    global angle_history

    drive_model = model
    drive_image, _ = img_to_tensor(img)

    angle_history_list = list(reversed(angle_history))
    if len(angle_history_list) < MEM_SIZE:
        angle_history_list += [0.0] * (MEM_SIZE - len(angle_history_list))
    else:
        angle_history_list = angle_history_list[:MEM_SIZE]

    angle_history_array = np.array(angle_history_list, dtype=np.float32).reshape(1, MEM_SIZE)

    angle = predict_angle(drive_model, drive_image, angle_history_array)

    angle_history.append(angle)

    return angle, SPEED, {}


# ---------------- Inference ----------------

def predict_angle(drive_model: CompiledModel, img: ndarray, angle_history: ndarray) -> float:
    """Run drive model inference on an image"""
    result = drive_model([img, angle_history])
    out = result[0]
    return float(np.array(out).ravel()[0])


# ---------------- Preprocessing ----------------

def img_to_tensor(img: Image.Image) -> tuple[ndarray, ndarray]:
    """Convert the images to tensors for the model input"""
    assert img.size == (320, 240), f'Expected image to be of size (320, 240), not {img.size}'

    # sign
    sign_batched = np.zeros((1, 3, 240, 320), dtype=np.float32)

    # drive
    drive_resized = img.resize((160, 120))
    drive_arr = np.array(drive_resized, dtype=np.float32)  # [240, 320, 3]
    drive_arr *= 1.0 / 255.0
    drive_batched = np.transpose(drive_arr, (2, 0, 1))[None, ...]  # [1, 3, 84, 160]

    return drive_batched, sign_batched
 