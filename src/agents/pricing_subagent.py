"""Claude-based Pricing/Research Subagent: agentic web search for resale pricing."""

from __future__ import annotations

import argparse
import json
from typing import Literal

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.agents.vision_subagent import ProductIdentification, VisionSubagent
from src.ml.vision_preprocessor import VisionPreprocessor

load_dotenv()

_SYSTEM_PROMPT = (
    "You are a resale pricing analyst. Given a structured item identification, use the "
    "web_search tool to research what this item (or the closest reasonable match) actually "
    "sold for recently — prefer real sold/completed listing prices (eBay sold listings, "
    "completed auction results) over active asking prices, since asking prices overstate "
    "realistic resale value. Search for a few different phrasings if the first search is "
    "unproductive (e.g. with/without brand, with/without model number). "
    "Once you've gathered enough comparable data, respond with ONLY a single raw JSON object "
    "(no markdown code fences, no commentary) matching exactly this shape:\n"
    "{\n"
    '  "suggested_price": <number, your single best-estimate price in USD>,\n'
    '  "price_range_low": <number>,\n'
    '  "price_range_high": <number>,\n'
    '  "comparable_listings": [\n'
    '    {"title": <string>, "price": <number>, "condition": <string or null>, '
    '"sold_date": <string or null>, "source_url": <string or null>}\n'
    "  ],\n"
    '  "reasoning": <string explaining how you arrived at the price, referencing the comps>,\n'
    '  "pricing_confidence": <"high", "medium", or "low">\n'
    "}\n"
    "If you can't find solid comps, say so in reasoning, make your best judgment from whatever "
    "market context you do have, and set pricing_confidence to \"low\"."
)


class ComparableListing(BaseModel):
    title: str
    price: float
    condition: str | None = None
    sold_date: str | None = None
    source_url: str | None = None


class PricingRecommendation(BaseModel):
    suggested_price: float = Field(description="Single best-estimate resale price in USD")
    price_range_low: float
    price_range_high: float
    comparable_listings: list[ComparableListing]
    reasoning: str
    pricing_confidence: Literal["high", "medium", "low"]


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            raise
        return json.loads(text[start : end + 1])


class PricingSubagent:
    """Wraps a Claude call with the web_search tool to research resale pricing."""

    def __init__(self, model: str = "claude-sonnet-5", max_searches: int = 5) -> None:
        self.model = model
        self.max_searches = max_searches
        self.client = anthropic.Anthropic()

    def research_price(self, identification: ProductIdentification) -> PricingRecommendation:
        item_context = identification.model_dump_json(indent=2)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=_SYSTEM_PROMPT,
            tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": self.max_searches}],
            messages=[{
                "role": "user",
                "content": (
                    f"Item to price:\n{item_context}\n\n"
                    "Research recent resale prices and respond with the pricing JSON."
                ),
            }],
        )

        text = next((block.text for block in response.content if block.type == "text"), None)
        if text is None:
            raise RuntimeError("Pricing Subagent returned no text content to parse")

        data = _extract_json(text)
        return PricingRecommendation.model_validate(data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the local classifier + Vision Subagent + Pricing Subagent on an image."
    )
    parser.add_argument("image_path", type=str, help="Path to an image file")
    parser.add_argument("--model", type=str, default="claude-sonnet-5")
    args = parser.parse_args()

    local_result = VisionPreprocessor().classify(args.image_path)
    identification = VisionSubagent(model=args.model).identify(args.image_path, local_result)
    print("--- Identification ---")
    print(identification.model_dump_json(indent=2))

    pricing = PricingSubagent(model=args.model).research_price(identification)
    print("\n--- Pricing ---")
    print(pricing.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
