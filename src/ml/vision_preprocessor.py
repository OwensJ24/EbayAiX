"""Local ResNet50 baseline classifier for the image intake pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50


@dataclass
class Prediction:
    label: str
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ClassificationResult:
    image_path: str
    predicted_category: str
    confidence: float
    top_k: list[Prediction]
    model_name: str
    weights_version: str
    device: str

    def to_dict(self) -> dict:
        return {
            **{k: v for k, v in asdict(self).items() if k != "top_k"},
            "top_k": [p.to_dict() for p in self.top_k],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class VisionPreprocessor:
    """Loads ResNet50 once and classifies images against it."""

    def __init__(self, top_k: int = 5, device: str | None = None) -> None:
        self.top_k = top_k
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")

        self.weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=self.weights).to(self.device).eval()
        self.transforms = self.weights.transforms()
        self.categories = self.weights.meta["categories"]

    def _load_image(self, image_path: str | Path) -> Image.Image:
        return Image.open(image_path).convert("RGB")

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        return self.transforms(image).unsqueeze(0).to(self.device)

    def _infer(self, batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.model(batch)
        return F.softmax(logits, dim=1)

    def _postprocess(self, probs: torch.Tensor, image_path: str) -> ClassificationResult:
        values, indices = torch.topk(probs[0], self.top_k)
        top_k = [
            Prediction(label=self.categories[idx], confidence=value.item())
            for value, idx in zip(values, indices)
        ]
        return ClassificationResult(
            image_path=str(image_path),
            predicted_category=top_k[0].label,
            confidence=top_k[0].confidence,
            top_k=top_k,
            model_name="resnet50",
            weights_version=str(self.weights),
            device=self.device,
        )

    def classify(self, image_path: str | Path) -> ClassificationResult:
        image = self._load_image(image_path)
        batch = self._preprocess(image)
        probs = self._infer(batch)
        return self._postprocess(probs, str(image_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify an image with a local ResNet50 baseline.")
    parser.add_argument("image_path", type=str, help="Path to an image file")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "mps"])
    args = parser.parse_args()

    preprocessor = VisionPreprocessor(top_k=args.top_k, device=args.device)
    result = preprocessor.classify(args.image_path)
    print(result.to_json())


if __name__ == "__main__":
    main()
