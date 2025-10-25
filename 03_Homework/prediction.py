from pathlib import Path
from typing import Dict

import numpy as np
from numpy import ndarray
import openvino as ov
from openvino import CompiledModel
from PIL import Image

# ---------------- Config ----------------
SPEED = 75.0        # regular speed
DRIVE_MODEL_NAME = 'unequaled-skink-546_New_v4_2_v3_e30.onnx'


# ---------------- Load ----------------
def load(model_dir: str) -> CompiledModel:
    """This functions gets called every time the side button on the remote is pressed in self-driving mode"""
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
    drive_model = model
    drive_image, _ = img_to_tensor(img)
    angle = predict_angle(drive_model, drive_image)
    return angle, SPEED, {}


# ---------------- Inference ----------------

def predict_angle(drive_model: CompiledModel, img: ndarray) -> float:
    """Run drive model inference on an image"""
    out = drive_model(img)[0]
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
