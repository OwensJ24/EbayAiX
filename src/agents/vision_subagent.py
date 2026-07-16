"""Claude-based Vision Subagent: structured product identification from an image."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Literal

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.ml.vision_preprocessor import ClassificationResult, VisionPreprocessor

load_dotenv()

_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

_SYSTEM_PROMPT = (
    "You are a product identification specialist for an e-commerce reselling pipeline. "
    "Examine the item photo and extract structured details for a resale listing. "
    "A local ResNet50 classifier has already produced a baseline category guess, provided "
    "as context below — treat it as a hint, and trust your own visual analysis over it "
    "if they disagree. Be conservative about condition: only claim 'New' if there is clear "
    "evidence (tags, packaging, no wear). If you cannot read a model number or brand, leave "
    "it null rather than guessing."
)


class ProductIdentification(BaseModel):
    item_name: str = Field(description="Clean, human-readable product title suitable for a listing")
    brand: str | None = Field(default=None, description="Manufacturer or brand name, if identifiable")
    model_number: str | None = Field(default=None, description="Model number or SKU visible on the item, if any")
    category: str = Field(description="Specific product category, e.g. 'Digital SLR Camera'")
    condition: Literal["New", "Like New", "Very Good", "Good", "Acceptable", "For Parts"]
    condition_notes: str = Field(description="Specific visible wear, damage, or missing parts supporting the condition rating")
    distinguishing_features: list[str] = Field(
        description="Visible features useful for identifying the exact variant: color, ports, markings, included accessories"
    )
    identification_confidence: Literal["high", "medium", "low"]


class VisionSubagent:
    """Wraps a Claude vision call that extracts strict structured product data."""

    def __init__(self, model: str = "claude-sonnet-5") -> None:
        self.model = model
        self.client = anthropic.Anthropic()

    def _encode_image(self, image_path: str | Path) -> tuple[str, str]:
        path = Path(image_path)
        media_type = _MEDIA_TYPES.get(path.suffix.lower(), "image/jpeg")
        data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
        return data, media_type

    def identify(
        self,
        image_path: str | Path,
        local_classification: ClassificationResult,
        user_provided_name: str | None = None,
    ) -> ProductIdentification:
        image_data, media_type = self._encode_image(image_path)
        local_context = json.dumps(local_classification.to_dict(), indent=2)

        prompt_text = f"Local ResNet50 classifier output:\n{local_context}\n\n"
        if user_provided_name:
            prompt_text += (
                f'The user has confirmed this item is: "{user_provided_name}". '
                "Use this to identify the specific brand, model number, and category as "
                "precisely as possible.\n\n"
            )
        prompt_text += "Identify this item."

        response = self.client.messages.parse(
            model=self.model,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_data}},
                    {"type": "text", "text": prompt_text},
                ],
            }],
            output_format=ProductIdentification,
        )
        return response.parsed_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local classifier + Claude Vision Subagent on an image.")
    parser.add_argument("image_path", type=str, help="Path to an image file")
    parser.add_argument("--model", type=str, default="claude-sonnet-5")
    args = parser.parse_args()

    local_result = VisionPreprocessor().classify(args.image_path)
    identification = VisionSubagent(model=args.model).identify(args.image_path, local_result)
    print(identification.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
