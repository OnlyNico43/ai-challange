from pathlib import Path
import json
from collections import OrderedDict
import random

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelSummary
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from aic.logger import DictLogger
from aic.runs import create_run_dir
from aic.helpers import disable_warnings
disable_warnings()

# Configure PyTorch to use the new TF32 API for better performance on newer GPUs
if torch.cuda.is_available():
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

print(f'GPU (Cuda) is available: {torch.cuda.is_available()}')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.mps.is_available() else torch.device('cpu')
print(f'Using device: {device}')

def get_image_paths(path: Path) -> list:
    return list(sorted(path.glob('**/*.jpg')))

def read_data_from_json(image_path: str | Path) -> tuple[float, float]:
    path = Path(image_path).with_suffix('.json')
    with open(path, 'r') as f:
        data = json.load(f)
    steering_angle = float(data['angle'])
    throttle = float(data['speed'])
    return steering_angle, throttle

import tempfile
p = Path(tempfile.gettempdir()) / 'test1-3'; p.mkdir(exist_ok=True)
img_path = p / 'test_01.jpg'
img_path.touch()
json_path = p / 'test_01.json'
json_path.write_text('{"angle": 17.805982737, "speed": 31}')

# Test
ret = read_data_from_json(str(img_path))
assert isinstance(ret, tuple), f'Die Funktion sollte ein Tuple zurückgeben, aber gibt {type(ret)} zurück'
assert len(ret) == 2, f'Das Tuple sollte genau 2 Elemente enthalten, aber enthält {len(ret)} Elemente'
assert all(isinstance(x, float) for x in ret), f'Die Elemente des Tuples sollten Zahlen von Typ float sein'
assert ret == (17.805982737, 31), f'Die Funktion sollte (17.805982737, 31.0) zurückgeben, gibt aber {ret} zurück'

# Cleanup
img_path.unlink()
print('Die Funktion funktioniert 👍')


image_dir_path = Path('../data/')

# Sollte der Pfad nicht existieren, wird hier eine Fehlermeldung ausgegeben
if not image_dir_path.exists():
    raise Exception(f"Fehler: Der Pfad {image_dir_path} existiert nicht 🛑")
else:
    print(f"Der Pfad {image_dir_path} wurde gefunden 👍")

image_paths = get_image_paths(image_dir_path)
assert len(image_paths) > 0, "Keine Bilder gefunden"
print(f"Es wurden {len(image_paths)} Bilder gefunden")

# der erste Pfad auswählen und ausgeben (wenn du einen anderen willst, ändere den Index)
some_image_path = image_paths[42]
print('Beispielbild:', some_image_path)

# Bild laden und anzeigen
img = Image.open(some_image_path)
print(f"Größe des Bildes (BxH): {img.size}")

# Dazugehörige Daten für Lenkung und Geschwindigkeit ausgeben
angle, speed = read_data_from_json(some_image_path)
print(f"Geschwindigkeit {speed:.2f}, Lenkung: {angle:.2f}")

sorted_image_paths = get_image_paths(image_dir_path)

from multiprocessing.pool import ThreadPool
with ThreadPool(20) as pool:
  vehicle_data = np.array(pool.map(read_data_from_json, image_paths))
all_angles = vehicle_data[:,0]
all_speeds = vehicle_data[:,1]

print(f"Kennzahlen zu Lenkung: min={all_angles.min():.2f}, max: {all_angles.max():.2f}, Durchschnitt: {all_angles.mean():.2f}")
print(f"Kennzahlen zu Geschwindigkeit: min={all_speeds.min():.2f}, max: {all_speeds.max():.2f}, Durchschnitt: {all_speeds.mean():.2f}")


import matplotlib.style as mplstyle
mplstyle.use('fast')
fig, (ax1, ax2) = plt.subplots(2,figsize=(10, 8))
ax1.plot(all_angles)
ax1.set_title("Angle")
ax1.minorticks_on()
ax2.plot(all_speeds)
ax2.set_title("Speed")
ax2.minorticks_on()
plt.close(fig)

from torchvision.transforms import v2

transformer = v2.Compose(
    [
        # Helligkeit, Kontrast, Sättigung und Farbton werden zufällig bis zum angegeben Wert geändert
        # Du kannst die Werte anpassen, wenn du willst (siehe Zusatzaufgabe)
        v2.ColorJitter(
            brightness=0.5,
            saturation=0.2,
            hue=0.1,
        ),
    v2.ColorJitter(brightness=0.5, saturation=0.2, hue=0.2),

    v2.RandomPerspective(distortion_scale=0.2, p=0.5),
    # v2.Resize((120, 160)),
    v2.ElasticTransform(alpha=100),
    v2.RandomInvert(),
    v2.RandomPosterize(bits=4, p=0.3),
    v2.RandomSolarize(threshold=128, p=0.3),
    v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ]
)


fig, axs = plt.subplots(1, 4, figsize=(3*4, 3))
for i, ax in enumerate(axs):
    # Zuerst das Originalbild
    if i == 0:
        ax.imshow(img)
        ax.set_title("Original")
        ax.axis('off')
        continue
    # Danach 3 zusätzlich augmentierte Bilder
    else:
        augmented_img = transformer(img)
        ax.imshow(augmented_img)
        ax.set_title(f"Augmented {i}")
        ax.axis('off')

plt.close(fig)




from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import pil_to_tensor
MEM_SIZE = 12

class ImageDataset(Dataset):
    def __init__(self, image_paths, sorted_image_paths, transform=None):
        self.image_paths = image_paths
        self.sorted_image_paths = sorted_image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path)
        # Wir werden das Bild etwas verkleinern, um Rechenzeit zu sparen (es sind immer noch genügend Informationen im Bild vorhanden)
        # Original ist das Bild BxH 320x240 Pixel gross, wir verkleinern es auf 16x120 Pixel
        image = image.resize((160, 120), resample=Image.Resampling.NEAREST)
        # Augmentation (wenn vorhanden)
        if self.transform:
            image = self.transform(image)
        image_array = (np.asarray(image) / 255.0).astype(np.float32).transpose(2, 0, 1)
        image_tensor = torch.tensor(image_array)
        angle, _ = read_data_from_json(path)
        angle_tensor = torch.tensor(angle, dtype=torch.float32).unsqueeze(0)

        last_angles = []

        data_index = self.sorted_image_paths.index(path)
        start_idx = max(0, data_index - MEM_SIZE)
        current_run = path.name.split('_')[0]
        start_idx_run = self.sorted_image_paths[start_idx].name.split('_')[0]
        if current_run != start_idx_run:
            start_idx = data_index - int(path.stem.split('_')[1])

        angle_paths = self.sorted_image_paths[start_idx:data_index]
        for angle_path in angle_paths:
            angle_value, _ = read_data_from_json(angle_path)
            last_angles.insert(0, angle_value)

        if len(last_angles) < MEM_SIZE:
            last_angles += [0.0] * (MEM_SIZE - len(last_angles)) 
        else:
            last_angles = last_angles[:MEM_SIZE]

        angle_history_tensor = torch.tensor(last_angles, dtype=torch.float32)

        return image_tensor, angle_tensor, angle_history_tensor

import random

# Wir werden 80% der Daten für das Training verwenden und 20% für das Validieren
random.shuffle(image_paths)
split_idx = int(len(image_paths) * 0.8)
train_image_paths = image_paths[:split_idx]
val_image_paths = image_paths[split_idx:]

# Wichtig: Augmentation nur für das Trainingsset aktivieren!
train_dataset = ImageDataset(train_image_paths, sorted_image_paths, transform=transformer)
val_dataset = ImageDataset(val_image_paths, sorted_image_paths, transform=None)

# Überprüfen wir mal die Grösse dieser Sets
print(len(train_dataset), len(val_dataset))

# Inhalt anschauen
image, angle, last_angles = train_dataset[0]
print(image.shape, image.dtype, angle, angle.dtype)


BATCH_SIZE = 128  # Increased for better GPU utilization
training_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    pin_memory=True,  # Speeds up CPU->GPU transfer
    persistent_workers=True  # Keeps workers alive between epochs
)
validation_loader = torch.utils.data.DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)
 
# Ein Batch aus dem DataLoader laden und dessen Shape anzeigen
images, labels, last_angles = next(iter(training_loader))

run_directory = create_run_dir('../runs/drive/')

class DriveModel(nn.Module):
    def __init__(self, lr=0.001):
        super().__init__()
        self.lr = lr

        # Convolutional layers for image processing
        self.conv_layers = nn.Sequential(
            OrderedDict(
                conv1 = nn.Conv2d(3, 18, 5, stride=2, padding=1),
                batch1 = nn.BatchNorm2d(18),
                relu1 = nn.ReLU(),

                conv2 = nn.Conv2d(18, 32, 5, stride=2, padding=1),
                batch2 = nn.BatchNorm2d(32),
                relu2 = nn.ReLU(),

                conv3 = nn.Conv2d(32, 64, 5, stride=2, padding=1),
                batch3 = nn.BatchNorm2d(64),
                relu3 = nn.ReLU(),

                conv4 = nn.Conv2d(64, 128, 3, stride=1, padding=1),
                batch4 = nn.BatchNorm2d(128),
                relu4 = nn.ReLU(),

                conv5 = nn.Conv2d(128, 64, 3, stride=1, padding=1),
                batch5 = nn.BatchNorm2d(64),
                relu5 = nn.ReLU(),

                one2one = nn.Conv2d(64, 1, 1, stride=1),
                flatten = nn.Flatten(1, -1),
            )
        )

        # Calculate the size of flattened conv output
        # For 160x120 input: after conv layers you get 266 features (based on your linear1)
        # conv_output_size = 266  # You can calculate this or determine it empirically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 120, 160)
            conv_output_size = self.conv_layers(dummy_input).shape[1]
        
        # Linear layers that take concatenated features
        self.linear_layers = nn.Sequential(
            nn.Linear(conv_output_size + MEM_SIZE, 16, bias=True),
            nn.ReLU(),
            # nn.Linear(128, 16),
            # nn.ReLU(),
            nn.Linear(16, 1, bias=True),
        )

    def forward(self, image, angle_history):
        # Process image through conv layers
        conv_features = self.conv_layers(image)
        
        # Concatenate conv features with angle history
        combined = torch.cat([conv_features, angle_history], dim=1)
        
        # Pass through linear layers
        output = self.linear_layers(combined)
        
        return output

import torch
from torch import nn
from pytorch_lightning import LightningModule

class CustomLightningDriveModel(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.lr = model.lr
        
    def forward(self, image, angle_history):
        return self.model(image, angle_history)
    
    def training_step(self, batch, batch_idx):
        images, angles, angle_histories = batch
        predictions = self(images, angle_histories)
        loss = nn.functional.mse_loss(predictions, angles)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, angles, angle_histories = batch
        predictions = self(images, angle_histories)
        loss = nn.functional.mse_loss(predictions, angles)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log('test_loss', loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x = batch[0]
        y_pred = self(x)
        return y_pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
       
        # Learning rate scheduler - reduces LR when validation loss plateaus
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )
       
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }


model = DriveModel()
lightning_model = CustomLightningDriveModel(model)

logger = DictLogger()

ModelSummary().on_fit_start(Trainer(), lightning_model)

# Hyperparameter definieren
EPOCHS = 100
LEARNING_RATE = 0.005

lightning_model.lr = LEARNING_RATE

# Trainer initialisieren mit GPU-Optimierungen
trainer = Trainer(
    max_epochs=EPOCHS,
    callbacks=[logger],
    precision='64-true',  # Mixed precision training for faster computation
    accelerator='auto',  # Automatically use available GPU
    devices=1,  # Use 1 GPU
    accumulate_grad_batches=2,  # Effective batch size = BATCH_SIZE * 2
    benchmark=True  # cuDNN benchmark mode for optimal algorithms
)

# Enable TensorFloat32 for faster matrix operations on newer GPUs with Tensor Cores
torch.set_float32_matmul_precision('high')  # Trade-off precision for performance on CUDA devices with Tensor Cores

# Das Training starten. Dieses dauert eine Weile...
trainer.fit(lightning_model, training_loader, validation_loader)

def plot_training(metrics: list[dict[str, list[float]]], save_to: str | Path = None):
    # 📝 Aufgabe: Plotte die Metriken
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Extract the metrics data
    train_losses = []
    val_losses = []
    
    for metric_dict in metrics:
        # Handle train loss
        if 'train_loss_step' in metric_dict:
            loss_value = metric_dict['train_loss_step']
            if hasattr(loss_value, 'item'):  # It's a tensor
                train_losses.append(loss_value.item())
            elif isinstance(loss_value, (list, tuple)):
                train_losses.extend([x.item() if hasattr(x, 'item') else x for x in loss_value])
            else:
                train_losses.append(float(loss_value))
                
        elif 'train_loss' in metric_dict:
            loss_value = metric_dict['train_loss']
            if hasattr(loss_value, 'item'):  # It's a tensor
                train_losses.append(loss_value.item())
            elif isinstance(loss_value, (list, tuple)):
                train_losses.extend([x.item() if hasattr(x, 'item') else x for x in loss_value])
            else:
                train_losses.append(float(loss_value))
            
        # Handle validation loss
        if 'val_loss' in metric_dict:
            loss_value = metric_dict['val_loss']
            if hasattr(loss_value, 'item'):  # It's a tensor
                val_losses.append(loss_value.item())
            elif isinstance(loss_value, (list, tuple)):
                val_losses.extend([x.item() if hasattr(x, 'item') else x for x in loss_value])
            else:
                val_losses.append(float(loss_value))
    
    # Plot the losses
    if train_losses:
        ax1.plot(train_losses, label='Train Loss', alpha=0.7)
    if val_losses:
        # Validation loss is typically logged less frequently, so we need to spread it out
        val_epochs = np.linspace(0, len(train_losses)-1, len(val_losses)) if train_losses else range(len(val_losses))
        ax1.plot(val_epochs, val_losses, label='Validation Loss', marker='o', markersize=3)
    
    ax1.set_title('Training and Validation Loss over Time')
    ax1.set_xlabel('Steps/Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    plt.tight_layout()
    
    # Save the plot if save_to is provided
    if save_to:
        plt.savefig(Path(save_to) / 'training_metrics.png', dpi=150, bbox_inches='tight')
    
    plt.close(fig)

# First, let's examine what's actually in the logger metrics
print("Available metrics keys:")
if logger.metrics:
    print(f"Number of metric dictionaries: {len(logger.metrics)}")
    for i, metric_dict in enumerate(logger.metrics[:3]):  # Show first 3 for inspection
        print(f"Metric dict {i} keys: {list(metric_dict.keys())}")

plot_training(logger.metrics, save_to=run_directory)

def predict(img_tensor, angle_history_tensor):
    model.eval()
    # Ensure model is in float32
    model.float()
    with torch.no_grad():
        y_pred = model(img_tensor.unsqueeze(0).float(), angle_history_tensor.unsqueeze(0).float())
    return y_pred.item()

# Testen wir unsere Funktion
image_tensor, angle_tensor, angle_history_tensor = train_dataset[0]
angle_pred = predict(image_tensor, angle_history_tensor)
print(f"Predicted angle: {angle_pred}")


def calculate_error(data: Dataset, save_to: str | Path = None) -> float:
    # 📝 Vervollständige die funktion:
    true_y = [angle.item() for _, angle, _ in data]
    pred_y = [predict(img, angle_hist) for img, _, angle_hist in data]
    errors = [abs(t - p) for t, p in zip(true_y, pred_y)]
    mean_error = sum(errors) / len(errors)
    #6. den Fehler zurückgibt und ihn als text file in `save_to` speichert.
    if save_to is not None:
        save_path = Path(save_to) / 'mean_error.txt'
        with open(save_path, 'w') as f:
            f.write(f"{mean_error:.4f}\n")
        
    return mean_error


error = calculate_error(val_dataset, save_to=run_directory)
print(f"Der durchschnittliche Fehler auf dem Validierung-Set ist {error:.1f}")

def visualize_angels(data: Dataset, save_to: str | Path = None):
    true_y = [angle.item() for _, angle, _ in data]

    n = min(100, len(true_y))
    true_y_plot = true_y[:n]
    pred_y_plot = [predict(image_tensor, angle_history) for image_tensor, _, angle_history in val_dataset][:n]
    sorted_idx = np.argsort(true_y_plot)
    true_y_plot = np.array(true_y_plot)[sorted_idx]
    pred_y_plot = np.array(pred_y_plot)[sorted_idx]

    plt.plot(true_y_plot, label="True")
    plt.plot(pred_y_plot, label="Predicted")
    plt.legend()
    plt.ylabel("Angle")
    plt.title("Visuelle Inspektion")
    
    if save_to:
        plt.savefig(Path(save_to) / 'angle_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()

visualize_angels(val_dataset, save_to=run_directory)

def visualize_drive_predictions(dataloader: DataLoader, save_to: str | Path = None):
    # 📝 Vervollständige die funktion:
    # 1. Select 6 random images from the dataloader
    dataset = dataloader if hasattr(dataloader, '__len__') else dataloader.dataset
    n_samples = min(6, len(dataset))
    
    # Randomly select indices
    random_indices = random.sample(range(len(dataset)), n_samples)
    
    # 2. Store images, true angles, and angle history
    images = []
    true_angles = []
    angle_histories = []
    
    for idx in random_indices:
        image_tensor, angle_tensor, angle_history_tensor = dataset[idx]
        images.append(image_tensor)
        true_angles.append(angle_tensor.item())
        angle_histories.append(angle_history_tensor)
    
    # 4. Make predictions for each image
    predicted_angles = []
    for image_tensor, angle_history in zip(images, angle_histories):
        pred_angle = predict(image_tensor, angle_history)
        predicted_angles.append(pred_angle)
    
    # 5. Plot the images with True, Pred, and Error values
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for i in range(n_samples):
        # Convert tensor to displayable format
        img_display = images[i].permute(1, 2, 0).numpy()
        
        true_angle = true_angles[i]
        pred_angle = predicted_angles[i]
        error = abs(true_angle - pred_angle)
        
        axes[i].imshow(img_display)
        axes[i].set_title(f'True: {true_angle:.2f}°\nPred: {pred_angle:.2f}°\nError: {error:.2f}°')
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # 6. Save the plot if save_to is provided
    if save_to:
        save_path = Path(save_to) / 'drive_predictions_visualization.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.close(fig)


visualize_drive_predictions(val_dataset, save_to=run_directory)

# Definieren, wo das Model gespeichert werden soll. Erstelle den Ordner, falls er noch nicht existiert.

model_path = Path(f'../models/drive/{run_directory.name}/DriveModel_v1.onnx')

# Zur Sicherheit prüfen, ob das Model bereits existiert, (lösche es, wenn du es überschreiben willst. Oder benenne es um)
assert not model_path.exists(), 'Das Model existiert bereits'
model_path.parent.mkdir(parents=True, exist_ok=True)

# Model speichern - mit beiden Inputs (Bild und Angle History)
example_image = train_dataset[0][0].unsqueeze(0)
example_angle_history = train_dataset[0][2].unsqueeze(0)
torch.onnx.export(
    model.to('cpu'),
    (example_image, example_angle_history),
    model_path,
    input_names=['image', 'angle_history'],
    output_names=['steering_angle'],
)
