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

_PREVIEW_SYSTEM_PROMPT = (
    "You are a product identification specialist for an e-commerce reselling pipeline. "
    "Examine the item photo and give a quick best guess at what it is and what eBay "
    "category it belongs to — this is a fast first pass; a human will confirm or correct "
    "these before a second, more detailed analysis. A local ResNet50 classifier has "
    "already produced a baseline category guess, provided as context below — treat it as "
    "a hint, and trust your own visual analysis over it if they disagree."
)

_SYSTEM_PROMPT = (
    "You are a product identification specialist for an e-commerce reselling pipeline. "
    "The item's name and eBay category have already been confirmed by a human — treat "
    "them as ground truth, do not second-guess or change them. Your job now is to fill in "
    "everything else: brand, model number, condition, and a description. Be conservative "
    "about condition: only claim 'New' if there is clear evidence (tags, packaging, no "
    "wear). If you cannot read a model number or brand, leave it null rather than "
    "guessing.\n\n"
    "For content_description, write it like a real, simple eBay listing description — "
    "not a narrative about what the item is. The title is added separately and already "
    "covers that; do not repeat it. Structure: (1) If this is a functional/electronic "
    "item, confirm its functionality in one short line, e.g. 'Tested and working' — "
    "unless the condition is 'For Parts' or otherwise non-functional, in which case say "
    "so honestly instead (e.g. 'Sold for parts, not tested' or 'Not functional'). Omit "
    "this line entirely for items where a working/non-working claim doesn't apply (e.g. "
    "clothing, books, purely decorative items). (2) List relevant specifics as short, "
    "plain facts — model number, style, size, material, color, included accessories — "
    "whatever applies. Keep it brief and factual, not descriptive prose. Do NOT mention "
    "wear, damage, defects, or condition here; that belongs only in condition_notes."
)


class ItemPreview(BaseModel):
    item_name: str = Field(description="Clean, human-readable product title suitable for a listing")
    category: str = Field(description="Specific product category guess, e.g. 'Digital SLR Camera'")


class ProductIdentification(BaseModel):
    item_name: str = Field(description="Clean, human-readable product title suitable for a listing")
    brand: str | None = Field(default=None, description="Manufacturer or brand name, if identifiable")
    model_number: str | None = Field(default=None, description="Model number or SKU visible on the item, if any")
    category: str = Field(description="Specific product category, e.g. 'Digital SLR Camera'")
    category_id: str | None = Field(default=None, description="Confirmed eBay Taxonomy categoryId, set after user confirmation")
    condition: Literal["New", "Like New", "Very Good", "Good", "Acceptable", "For Parts"]
    condition_notes: str = Field(description="Specific visible wear, damage, or missing parts supporting the condition rating")
    content_description: str = Field(
        description=(
            "eBay listing body text (the title is added separately — do not repeat it): a "
            "short functionality confirmation line when applicable (e.g. 'Tested and "
            "working', or an honest non-functional equivalent), followed by brief specifics "
            "(model number, style, size, material, color, included accessories). No "
            "condition/wear commentary — that belongs only in condition_notes."
        )
    )
    distinguishing_features: list[str] = Field(
        description="Visible features useful for identifying the exact variant: color, ports, markings, included accessories"
    )
    identification_confidence: Literal["high", "medium", "low"]


class VisionSubagent:
    """Wraps Claude vision calls that extract structured product data, in two phases:
    a cheap preview() guess, and a fuller identify() analysis run only after a human
    confirms the preview's name/category."""

    def __init__(self, model: str = "claude-sonnet-5") -> None:
        self.model = model
        self.client = anthropic.Anthropic()

    def _encode_image(self, image_path: str | Path) -> tuple[str, str]:
        path = Path(image_path)
        media_type = _MEDIA_TYPES.get(path.suffix.lower(), "image/jpeg")
        data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
        return data, media_type

    def preview(self, image_path: str | Path, local_classification: ClassificationResult) -> ItemPreview:
        image_data, media_type = self._encode_image(image_path)
        local_context = json.dumps(local_classification.to_dict(), indent=2)
        prompt_text = f"Local ResNet50 classifier output:\n{local_context}\n\nWhat is this item, and what eBay category does it belong to?"

        response = self.client.messages.parse(
            model=self.model,
            max_tokens=256,
            system=_PREVIEW_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_data}},
                    {"type": "text", "text": prompt_text},
                ],
            }],
            output_format=ItemPreview,
        )
        return response.parsed_output

    def identify(
        self,
        image_path: str | Path,
        local_classification: ClassificationResult,
        confirmed_item_name: str,
        confirmed_category: str,
    ) -> ProductIdentification:
        image_data, media_type = self._encode_image(image_path)
        local_context = json.dumps(local_classification.to_dict(), indent=2)

        prompt_text = (
            f"Local ResNet50 classifier output:\n{local_context}\n\n"
            f'The user has confirmed this item is: "{confirmed_item_name}", '
            f'in eBay category: "{confirmed_category}". '
            "Analyze the photo for everything else: brand, model number, condition, and description."
        )

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
        identification = response.parsed_output
        # Defensive — don't trust the model to echo the confirmed values back exactly.
        identification.item_name = confirmed_item_name
        identification.category = confirmed_category
        return identification


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local classifier + Claude Vision Subagent on an image.")
    parser.add_argument("image_path", type=str, help="Path to an image file")
    parser.add_argument("--model", type=str, default="claude-sonnet-5")
    args = parser.parse_args()

    local_result = VisionPreprocessor().classify(args.image_path)
    subagent = VisionSubagent(model=args.model)

    # No human in the loop for this standalone CLI — chain preview straight into
    # identify() using its own guesses as "confirmed", unlike the web app's flow where a
    # user reviews/edits them in between.
    preview = subagent.preview(args.image_path, local_result)
    print("Preview:", preview.model_dump_json(indent=2))

    identification = subagent.identify(args.image_path, local_result, preview.item_name, preview.category)
    print(identification.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
