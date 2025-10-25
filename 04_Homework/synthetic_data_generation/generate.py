import random
import click
from shutil import rmtree
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, UnidentifiedImageError, ImageDraw
from torchvision.transforms import v2
import numpy as np
from tqdm import tqdm
import torch
import time

def get_average_rgba(image):
    """Compute the average RGBA value of a Pillow image."""
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    np_image = np.array(image)
    avg_color = np.mean(np_image[:, :, :3], axis=(0, 1, 2))
    return avg_color

def add_gaussian_noise(image, mean=0, std=10):
    """Add Gaussian noise to a PIL image using PyTorch and return the noisy image."""
    tensor = v2.ToImage()(image)
    tensor = v2.ToDtype(torch.float32, scale=True)(tensor)
    noise = torch.randn_like(tensor) * (std / 255.0) + (mean / 255.0)
    noisy_tensor = torch.clamp(tensor + noise, 0.0, 1.0)
    return v2.ToPILImage()(noisy_tensor)

def apply_background_augmentations(
    background, gaussian_noise=False, noise_std=10, noise_prob=0.5, elastic_prob=0.5):
    """Apply augmentations to the background image and return the result (RGB)."""
    bg = background.convert("RGB")

    # Random flip augmentation
    if random.random() < 0.5:
        bg = bg.transpose(Image.FLIP_LEFT_RIGHT)

    jitter = v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.5)
    bg = jitter(bg)

    # Randomly apply Gaussian noise if enabled, with random intensity
    if gaussian_noise and random.random() < noise_prob:
        random_std = random.uniform(0, noise_std)
        bg = add_gaussian_noise(bg, std=random_std)
    return bg

def augment_data(
    background, label_path, labels, weights, gaussian_noise=False, noise_std=10, 
    elastic_transform=False, sign_visibility=1.0):
    """Augment a single background image, possibly adding a sign."""
    chosen_label = random.choices(labels, weights=weights)[0]

    # Always apply background augmentations (now with elastic)
    bg = apply_background_augmentations(background, gaussian_noise=gaussian_noise, noise_std=noise_std)
    bg_width, bg_height = bg.width, bg.height
    avg_rgba_background = get_average_rgba(bg)

    if chosen_label == "NoSign":
        return bg, chosen_label

    # Select a sign image
    class_folder = label_path / chosen_label
    sign_files = [
        sign_file for sign_file in class_folder.iterdir()
        if sign_file.suffix.lower() in (".png", ".jpg", ".jpeg")
    ]
    if not sign_files:
        return bg, "NoSign"

    sign_path = random.choice(sign_files)
    sign = Image.open(sign_path).convert("RGBA")

    # Sign augmentations
    sign = v2.RandomPerspective(distortion_scale=0.7, p=1.0)(sign)
    blur_radius = random.uniform(0, 2)
    sign = sign.filter(ImageFilter.BoxBlur(radius=blur_radius))

    # Resize sign
    ratio = sign.height / bg_height
    scale = random.uniform(0.35, 0.5 / ratio)
    new_size = (int(sign.width * scale), int(sign.height * scale))
    sign = sign.resize(new_size, Image.LANCZOS)

    # Elastic transform for sign
    if elastic_transform and random.random() < 0.5:
        elastic = v2.ElasticTransform(alpha=random.uniform(30, 50), sigma=random.uniform(4, 6))
        sign = elastic(sign)

    # Rotation
    angle = random.randint(-30, 30)
    sign = sign.rotate(angle, expand=True)

    # Match Brightness to Background, scaled by sign_visibility
    bg_brightness = (avg_rgba_background / 255) + 0.2
    sign = ImageEnhance.Brightness(sign).enhance(bg_brightness * sign_visibility)

    # Contrast, scaled by sign_visibility
    contrast_factor = random.uniform(0.5, 1.0) * sign_visibility
    sign = ImageEnhance.Contrast(sign).enhance(contrast_factor)

    # Color Temperature & Alpha, alpha scaled by sign_visibility
    sign_np = np.array(sign).astype(np.float32)
    temp_shift = random.uniform(-1, 1)
    temp_strength = 8 # Intensity of Temperature

    sign_np[..., 0] += temp_strength * max(0, temp_shift)
    sign_np[..., 2] += temp_strength * -min(0, temp_shift)
    # Adjust alpha for visibility
    sign_np[..., 3] = np.clip(sign_np[..., 3] * sign_visibility, 0, 255)
    sign_np = np.clip(sign_np, 0, 255)

    sign = Image.fromarray(sign_np.astype(np.uint8)).convert("RGBA")

    # Position
    max_x = bg.width - sign.width
    max_y = bg.height - sign.height
    if max_x > 0 and max_y > 0:
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        bg.paste(sign, (x, y), mask=sign)
    else:
        return bg, 'NoSign'

    return bg, chosen_label


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option('-i', '--input-path', required=True, type=click.Path(exists=True, file_okay=False),
              help="Directory with background images.", show_default=True)
@click.option('-o', '--output-path', required=True, type=click.Path(file_okay=False),
              help="Directory to save augmented images.", show_default=True)
@click.option('-n', '--num-augmentations', default=None, type=int,
              help="Augmentations per background image (ignored if --total-augmentations is set).", show_default=True)
@click.option('-t', '--total-augmentations', default=None, type=int,
              help="Total number of augmentations to generate (overrides --num-augmentations).", show_default=True)
@click.option('--seed', default=None, type=int, help="Random seed for reproducibility.", show_default=True)
@click.option('-g', '--gaussian-noise/--no-gaussian-noise', default=False,
              help="Randomly apply Gaussian noise to backgrounds.", show_default=True)
@click.option('--noise-std', default=10, type=int,
              help="Maximum standard deviation for Gaussian noise.", show_default=True)
@click.option('-e', '--elastic-transform/--no-elastic-transform', default=False,
              help="Randomly apply elastic transform to signs.", show_default=True)
@click.option('--sign-visibility', default=1.0, type=float,
              help="Global factor for sign visibility (1.0=normal, <1.0=harder, >1.0=easier).", show_default=True)
@click.option('--list-labels', is_flag=True, default=False,
              help="List available sign labels and exit.", show_default=True)
@click.option('--dry-run', is_flag=True, default=False,
              help="Show what would be done, but don't write files.", show_default=True)
def main(
    input_path, output_path, num_augmentations, total_augmentations, seed, gaussian_noise, 
    noise_std, elastic_transform, sign_visibility, list_labels, dry_run):

    """
    Generate augmented training data from backgrounds and signs.

    Example usage:
      python generate.py -i ./input -o ./output -n 5 -g --noise-std 20 --sign-visibility 0.8
    """

    start_time = time.time()

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    input_path = Path(input_path)
    output_path = Path(output_path)
    current_file = Path(__file__)
    label_path = current_file.parent / 'signs'

    labels = ['50Sign', 'ClearSign', 'StopSign', 'NoSign']
    weights = [1, 1, 1, 2]

    try:
        rmtree(output_path)
    except FileNotFoundError:
        pass
    output_path.mkdir(parents=True, exist_ok=True)

    background_files = [f for f in input_path.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")]
    num_backgrounds = len(background_files)

    # Determine augmentations per image
    if total_augmentations is not None:
        num_augmentations = total_augmentations // num_backgrounds
        remainder = total_augmentations % num_backgrounds
        augmentations_per_image = [num_augmentations] * num_backgrounds
        for i in range(remainder):
            augmentations_per_image[i] += 1
        total_tasks = total_augmentations
    else:
        if num_augmentations is None:
            num_augmentations = 1
        augmentations_per_image = [num_augmentations] * num_backgrounds
        total_tasks = num_backgrounds * num_augmentations

    print(f"\n🚀 Starting data augmentation...")
    print(f"   Input path:        {input_path}")
    print(f"   Output path:       {output_path}")
    print(f"   Background images: {num_backgrounds}")

    if total_augmentations is not None:
        print(f"   Total augmentations: {total_augmentations}")
    else:
        print(f"   Augmentations:     {num_augmentations} per image")
    print(f"   Total tasks:       {total_tasks}")
    print(f"   Gaussian noise:    {'ON' if gaussian_noise else 'OFF'}")
    print(f"   Elastic transform: {'ON' if elastic_transform else 'OFF'}")
    print(f"   Sign visibility:   {sign_visibility}\n")

    if list_labels:
        print("Available labels:", labels)
        return

    if dry_run:
        print("Dry run: No files will be written.")

    total_augmented = 0

    with tqdm(total=total_tasks, desc="Augmenting data", unit="img",
              dynamic_ncols=True, leave=True) as pbar:

        for idx, file in enumerate(background_files):
            for i in range(augmentations_per_image[idx]):

                try:
                    background = Image.open(file)
                    background = normalize_image(background)
                except UnidentifiedImageError:
                    print(f"⚠️ Skipping unreadable file: {file}")
                    continue

                augmented, label = augment_data(
                    background, label_path, labels, weights,
                    gaussian_noise=gaussian_noise, noise_std=noise_std,
                    elastic_transform=elastic_transform, sign_visibility=sign_visibility
                )

                if not dry_run:
                    class_output = (output_path / label)
                    class_output.mkdir(parents=False, exist_ok=True)

                    out_file = class_output / f"aug_{file.stem}_{i}_{random.randint(1000,9999)}.png"
                    augmented.save(out_file)
                    augmented.close()
                total_augmented += 1

                pbar.update(1)
                pbar.set_postfix({"Last Label": label})

            if not dry_run:
                background.close()

    print(f"\n🎉 Completed! Generated {total_augmented} augmented images in total.")
    print(f"   Saved in: {output_path.resolve()}\n")
    print(f"⏱️  Script ran for {time.time() - start_time:.2f} seconds.")

def normalize_image(image, width=380, height=240):
    """Resize the image to the given width and height."""
    return image.resize((width, height), Image.LANCZOS)

if __name__ == "__main__":
    main()
