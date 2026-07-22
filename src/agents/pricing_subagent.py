"""Claude-based Pricing/Research Subagent: eBay sold-listing research for resale pricing."""

from __future__ import annotations

import argparse
import json
import logging
import urllib.parse
from typing import Literal

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.agents.vision_subagent import ProductIdentification, VisionSubagent
from src.ml.vision_preprocessor import VisionPreprocessor

load_dotenv()

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a resale pricing analyst. Research pricing using ONLY eBay — never cite or "
    "rely on any other marketplace or site.\n\n"
    "You've been given one or more eBay SOLD-listings search URLs (using eBay's own "
    "LH_Sold=1&LH_Complete=1 completed-items filter) for this item. Use the web_fetch tool "
    "to retrieve each one and read the actual sold listing titles, prices, and dates shown — "
    "this is the most reliable source of real transaction data, since it's eBay's own sold-"
    "items filter rather than a general search that might surface active listings instead.\n\n"
    "If those URLs return no useful results (e.g. zero matches, or nothing resembling this "
    "item), you may use the web_search tool (also restricted to ebay.com) to try a few "
    "additional phrasings — still include \"sold\" in those queries and still prefer sold/"
    "completed data over active listings wherever you find it.\n\n"
    "Only fall back to active eBay listings if you genuinely cannot find comparable sold "
    "data. If you do, say so explicitly in your reasoning and set pricing_confidence no "
    "higher than \"medium\" (usually \"low\"), since asking prices overstate realistic resale "
    "value. Mark every comparable listing's listing_type as \"sold\" or \"active\" so this "
    "distinction is explicit, not just implied by your reasoning.\n\n"
    "Once you've gathered enough comparable data, respond with ONLY a single raw JSON object "
    "(no markdown code fences, no commentary) matching exactly this shape:\n"
    "{\n"
    '  "suggested_price": <number, your single best-estimate price in USD>,\n'
    '  "price_range_low": <number>,\n'
    '  "price_range_high": <number>,\n'
    '  "comparable_listings": [\n'
    '    {"title": <string>, "price": <number>, "listing_type": "sold" or "active", '
    '"condition": <string or null>, "sold_date": <string or null>, "source_url": <string or null>}\n'
    "  ],\n"
    '  "reasoning": <string explaining how you arrived at the price, referencing the comps>,\n'
    '  "pricing_confidence": <"high", "medium", or "low">\n'
    "}\n"
    "If you can't find solid eBay sold comps at all, say so in reasoning, make your best "
    "judgment from whatever eBay market context you do have, and set pricing_confidence to "
    '"low".'
)


def _build_sold_search_url(query: str) -> str:
    params = urllib.parse.urlencode({"_nkw": query, "LH_Sold": "1", "LH_Complete": "1"})
    return f"https://www.ebay.com/sch/i.html?{params}"


def _build_search_queries(identification: ProductIdentification) -> list[str]:
    queries: list[str] = []
    if identification.brand and identification.model_number:
        queries.append(f"{identification.brand} {identification.model_number}")
    if identification.brand and not identification.item_name.lower().startswith(identification.brand.lower()):
        queries.append(f"{identification.brand} {identification.item_name}")
    queries.append(identification.item_name)
    queries.append(identification.category)

    # Dedupe while preserving order, drop anything that collapsed to empty/whitespace.
    # Capped at 2: each query becomes a web_fetch call, and each fetched eBay search
    # page can be huge, so keeping the URL count low matters as much as max_content_tokens.
    seen: set[str] = set()
    deduped: list[str] = []
    for q in queries:
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            deduped.append(q)
    return deduped[:2]


class ComparableListing(BaseModel):
    title: str
    price: float
    listing_type: Literal["sold", "active"] = Field(
        description="Whether this was an actual sold/completed eBay listing or a currently-active asking price"
    )
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
    """Wraps a Claude call that reads eBay's own sold-listings pages to research pricing."""

    def __init__(
        self,
        model: str = "claude-sonnet-5",
        max_searches: int = 2,
        max_content_tokens: int = 3000,
    ) -> None:
        self.model = model
        self.max_searches = max_searches
        self.max_content_tokens = max_content_tokens
        self.client = anthropic.Anthropic()

    def research_price(self, identification: ProductIdentification) -> PricingRecommendation:
        item_context = identification.model_dump_json(indent=2)
        queries = _build_search_queries(identification)
        sold_urls = [_build_sold_search_url(q) for q in queries]
        sold_urls_text = "\n".join(f"- {url}" for url in sold_urls)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=_SYSTEM_PROMPT,
            tools=[
                {
                    "type": "web_fetch_20260209",
                    "name": "web_fetch",
                    "max_uses": len(sold_urls) + 1,
                    # Each fetched eBay search page can be huge (dozens of listing
                    # cards, sponsored content, scripts-as-text) — without this cap
                    # a single request can balloon into hundreds of thousands of
                    # input tokens, since server-side tool results stay in context
                    # for every subsequent step within the same request.
                    "max_content_tokens": self.max_content_tokens,
                    "allowed_domains": ["ebay.com"],
                },
                {
                    "type": "web_search_20260209",
                    "name": "web_search",
                    "max_uses": self.max_searches,
                    "allowed_domains": ["ebay.com"],
                },
            ],
            messages=[{
                "role": "user",
                "content": (
                    f"Item to price:\n{item_context}\n\n"
                    f"eBay sold-listings search URLs to fetch:\n{sold_urls_text}\n\n"
                    "Research recent resale prices and respond with the pricing JSON."
                ),
            }],
        )

        logger.info(
            "PricingSubagent.research_price token usage: input=%d output=%d",
            response.usage.input_tokens,
            response.usage.output_tokens,
        )

        # Take the LAST text block, not the first: with multiple tool-call rounds
        # (web_fetch, possibly web_search too), Claude often emits an early text
        # block like "Let me check eBay's sold listings..." before any tool use —
        # the actual JSON-only final answer is whatever text block comes last.
        text_blocks = [block.text for block in response.content if block.type == "text"]
        if not text_blocks:
            raise RuntimeError("Pricing Subagent returned no text content to parse")

        data = _extract_json(text_blocks[-1])
        return PricingRecommendation.model_validate(data)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

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
