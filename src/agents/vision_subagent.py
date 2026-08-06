"""Claude-based Vision Subagent: structured product identification from one or more
photos of the same item."""

from __future__ import annotations

import argparse
import base64
from pathlib import Path
from typing import Literal

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

_MULTI_IMAGE_NOTE = (
    "You may be given multiple photos of the same item — examine all of them together; "
    "a detail visible in one (e.g. a tag, label, or box) may not be visible in another."
)

_PREVIEW_SYSTEM_PROMPT = (
    "You are a product identification specialist for an e-commerce reselling pipeline. "
    "Examine the item photo(s) and give a quick best guess at what it is and what eBay "
    "category it belongs to — this is a fast first pass; a human will confirm or correct "
    f"these before a second, more detailed analysis. {_MULTI_IMAGE_NOTE}"
)

_CLUSTER_SYSTEM_PROMPT = (
    "You are sorting a batch of resale photos for an e-commerce reselling pipeline. The "
    "photos are numbered 0 to N-1, in the order given. Each photo shows exactly one "
    "physical item for resale, but the SAME item may appear in more than one photo — "
    "different angles, a tag or label, packaging — and those photos are not necessarily "
    "consecutive. Group the photo numbers by distinct physical item: every number from 0 "
    "to N-1 must appear in exactly one group, with no number skipped or duplicated. If "
    "every photo shows the same single item, return one group containing every number. "
    "For each group, give the same kind of quick best-guess item name and eBay category "
    "you'd give for a single item on a fast first pass — a human will confirm or correct "
    "these before a second, more detailed per-item analysis."
)

_SYSTEM_PROMPT = (
    "You are a product identification specialist for an e-commerce reselling pipeline. "
    "The item's name and eBay category have already been confirmed by a human — treat "
    "them as ground truth, do not second-guess or change them. Your job now is to fill in "
    "everything else: brand, model number, condition, and a description. Be conservative "
    "about condition: only claim 'New' if there is clear evidence (tags, packaging, no "
    f"wear). If you cannot read a model number or brand, leave it null rather than "
    f"guessing. {_MULTI_IMAGE_NOTE}\n\n"
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
    "wear, damage, defects, or condition here; that belongs only in condition_notes.\n\n"
    "For distinguishing_features, be thorough: this is what downstream eBay category "
    "item-specifics resolution draws from, so note every visible detail that could match "
    "one of eBay's required specifics for this category (size, color, material, style, "
    "type, connectivity, etc.) — check tags, labels, and packaging across all provided "
    "photos, not just the main shot."
)


class ItemPreview(BaseModel):
    item_name: str = Field(description="Clean, human-readable product title suitable for a listing")
    category: str = Field(description="Specific product category guess, e.g. 'Digital SLR Camera'")


class ItemCluster(BaseModel):
    photo_indices: list[int] = Field(
        description="0-based indices of the photos (in the order provided) that show this one physical item"
    )
    item_name: str = Field(description="Clean, human-readable product title suitable for a listing")
    category: str = Field(description="Specific product category guess, e.g. 'Digital SLR Camera'")


class BatchClusterResult(BaseModel):
    items: list[ItemCluster]


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
    """Wraps Claude vision calls that extract structured product data from one or more
    photos, in two phases: a cheap preview() guess, and a fuller identify() analysis run
    only after a human confirms the preview's name/category."""

    def __init__(self, model: str = "claude-sonnet-5") -> None:
        self.model = model
        self.client = anthropic.Anthropic()

    def _image_blocks(self, images: list[tuple[bytes, str]]) -> list[dict]:
        return [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64.standard_b64encode(image_bytes).decode("utf-8"),
                },
            }
            for image_bytes, media_type in images
        ]

    def preview(self, images: list[tuple[bytes, str]]) -> ItemPreview:
        content = self._image_blocks(images)
        content.append({"type": "text", "text": "What is this item, and what eBay category does it belong to?"})

        response = self.client.messages.parse(
            model=self.model,
            max_tokens=256,
            system=_PREVIEW_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
            output_format=ItemPreview,
        )
        return response.parsed_output

    def cluster_items(self, images: list[tuple[bytes, str]]) -> BatchClusterResult:
        """Groups a batch of photos by distinct physical item — the first pass over a
        multi-item upload, before any per-item confirm/analyze happens. `images` should
        be small/cheap thumbnails, not full eBay-quality images (see
        CLUSTERING_THUMBNAIL_SIZE in app.py) — a batch can be up to MAX_BATCH_IMAGES
        photos, and this sends all of them in one message.
        """
        content = self._image_blocks(images)
        content.append({
            "type": "text",
            "text": f"Group these {len(images)} photos (numbered 0 to {len(images) - 1}) by distinct physical item.",
        })

        response = self.client.messages.parse(
            model=self.model,
            max_tokens=4096,
            system=_CLUSTER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
            output_format=BatchClusterResult,
        )
        return response.parsed_output

    def identify(
        self,
        images: list[tuple[bytes, str]],
        confirmed_item_name: str,
        confirmed_category: str,
        required_aspects: list[dict] | None = None,
    ) -> ProductIdentification:
        prompt_text = (
            f'The user has confirmed this item is: "{confirmed_item_name}", '
            f'in eBay category: "{confirmed_category}". '
            "Analyze the photo(s) for everything else: brand, model number, condition, and description."
        )
        if required_aspects:
            aspect_lines = []
            for aspect in required_aspects:
                values = aspect.get("values") or []
                values_note = f" (eBay's suggested values include: {', '.join(values[:15])})" if values else ""
                aspect_lines.append(f"- {aspect['name']}{values_note}")
            prompt_text += (
                "\n\neBay requires the following item specifics for this category — look across all "
                "provided photos (including any tags, labels, or packaging) for each one, and note it "
                "explicitly in distinguishing_features, using eBay's own suggested wording above when it "
                "matches what you see (do not fabricate a value you can't actually determine):\n"
                + "\n".join(aspect_lines)
            )

        content = self._image_blocks(images)
        content.append({"type": "text", "text": prompt_text})

        response = self.client.messages.parse(
            model=self.model,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
            output_format=ProductIdentification,
        )
        identification = response.parsed_output
        # Defensive — don't trust the model to echo the confirmed values back exactly.
        identification.item_name = confirmed_item_name
        identification.category = confirmed_category
        return identification


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Claude Vision Subagent on an image.")
    parser.add_argument("image_path", type=str, help="Path to an image file")
    parser.add_argument("--model", type=str, default="claude-sonnet-5")
    args = parser.parse_args()

    path = Path(args.image_path)
    media_type = _MEDIA_TYPES.get(path.suffix.lower(), "image/jpeg")
    images = [(path.read_bytes(), media_type)]

    subagent = VisionSubagent(model=args.model)

    # No human in the loop for this standalone CLI — chain preview straight into
    # identify() using its own guesses as "confirmed", unlike the web app's flow where a
    # user reviews/edits them in between.
    preview = subagent.preview(images)
    print("Preview:", preview.model_dump_json(indent=2))

    identification = subagent.identify(images, preview.item_name, preview.category)
    print(identification.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
