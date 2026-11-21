import os
from pathlib import Path
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import albumentations as albu
from dotenv import load_dotenv
from segmentation_models_pytorch import Unet
from iglovikov_helper_functions.utils.image_utils import load_rgb
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image, rename_layers
load_dotenv()

HERE = Path(__file__).resolve().parent
#checkpoint produced by training.py (/opt/ml/model/best.ckpt after download)
WEIGHTS = Path(os.environ.get("WEIGHTS_PATH", HERE.parent / "best.ckpt"))
IMAGE = Path(os.environ.get("TEST_IMAGE_PATH", HERE / "Bg.jpeg"))
OUT_DIR = HERE.parent / "predictions"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(ckpt_path: Path) -> torch.nn.Module:
    model = Unet(encoder_name="timm-efficientnet-b3", classes=1, encoder_weights=None)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    #LightningModule stores the weights under "state_dict" with a "model." prefix
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = rename_layers(state_dict, {"model.": ""})
    model.load_state_dict(state_dict)
    return model.to(device).eval()


#must match val_aug / test_aug in config/settings.yaml
transform = albu.Compose([
    albu.LongestMaxSize(max_size=800, p=1),
    albu.PadIfNeeded(
        min_height=800,
        min_width=800,
        border_mode=cv2.BORDER_CONSTANT,
        fill=0,
        fill_mask=0,
        p=1,
    ),
    albu.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        max_pixel_value=255.0,
        p=1,
    ),
], p=1)


def predict(model: torch.nn.Module, image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    tensor = tensor_from_rgb_image(transform(image=image)["image"])
    tensor = torch.unsqueeze(tensor, 0).to(device)
    with torch.no_grad():
        logits = model(tensor)[0][0]
    mask = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.uint8)
    #network runs on the padded 800x800 canvas -> back to the original size
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


if __name__ == "__main__":
    if not WEIGHTS.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {WEIGHTS}\n"
            "download it from S3 (model.tar.gz -> best.ckpt) or set WEIGHTS_PATH"
        )
    model = load_model(WEIGHTS)
    image = load_rgb(IMAGE)
    mask = predict(model, image)

    overlay = image.copy()
    overlay[mask == 1] = (
        0.5 * image[mask == 1] + 0.5 * np.array([0, 255, 0], dtype=np.float32)
    ).astype(np.uint8)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.imsave(OUT_DIR / "mask.png", mask * 255, cmap="gray")
    plt.imsave(OUT_DIR / "overlay.png", overlay)
    print(f"saved {OUT_DIR / 'mask.png'} and {OUT_DIR / 'overlay.png'}")
