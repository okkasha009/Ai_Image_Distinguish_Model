from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, Tuple

import timm
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

MODEL_ARCH = "swin_base_patch4_window7_224"


class InvalidImageError(ValueError):
    """Raised when uploaded bytes cannot be decoded as a valid image."""


@dataclass
class ModelBundle:
    model: torch.nn.Module
    transform: transforms.Compose


def _build_transform() -> transforms.Compose:
    """Build the exact preprocessing pipeline used by the classifier."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def _extract_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    """Support both raw state dict and wrapped checkpoints with state_dict key."""
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported model checkpoint format.")

    # Handle checkpoints saved from DataParallel.
    return {
        (k.replace("module.", "", 1) if k.startswith("module.") else k): v
        for k, v in checkpoint.items()
    }


def load_model_bundle(model_path: Path) -> ModelBundle:
    """Load model and transform once (CPU only) for reuse across requests."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    device = torch.device("cpu")
    model = timm.create_model(MODEL_ARCH, pretrained=False, num_classes=1)
    model.to(device)

    checkpoint = torch.load(str(model_path), map_location=device)
    state_dict = _extract_state_dict(checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    return ModelBundle(model=model, transform=_build_transform())


def predict_from_bytes(image_bytes: bytes, bundle: ModelBundle) -> Tuple[str, float]:
    """Run binary inference on image bytes and return label + sigmoid probability."""
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            rgb_image = image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise InvalidImageError("Uploaded file is not a valid image.") from exc
    except (OSError, ValueError) as exc:
        raise InvalidImageError("Uploaded image is corrupted or unreadable.") from exc

    input_tensor = bundle.transform(rgb_image).unsqueeze(0)

    with torch.no_grad():
        output = bundle.model(input_tensor)
        probability = torch.sigmoid(output).item()

    label = "fake" if probability >= 0.5 else "real"
    return label, float(probability)
