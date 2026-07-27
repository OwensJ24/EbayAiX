FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

# glibc's malloc creates multiple memory arenas per process by default, and freed
# memory within a non-primary arena often isn't returned to the OS — a well-known
# cause of RSS ratcheting upward across requests (rather than resetting) in
# containerized Python/PyTorch apps specifically. Capping to a single arena keeps all
# allocations in one pool that malloc can actually release back to the OS, which
# matters a lot on Render's free-tier 512MB limit.
ENV MALLOC_ARENA_MAX=1

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY . .
RUN uv sync --frozen

# Bake the ResNet50 weights into the image so a cold start (Render's free
# tier spins down after inactivity) doesn't need a ~100MB download first.
RUN uv run python -c "from src.ml.vision_preprocessor import VisionPreprocessor; VisionPreprocessor()"

EXPOSE 8000

# --forwarded-allow-ips='*' trusts X-Forwarded-Proto from whatever connects to this
# container. Safe here specifically because the container has no other direct public
# exposure — the only thing that can reach it is Render's own edge proxy, which
# terminates TLS and forwards plain HTTP internally. Without this, uvicorn's default
# trusted-proxy list (127.0.0.1 only) doesn't match Render's actual internal proxy IP,
# so request.base_url silently resolves to http:// instead of https:// — which broke
# eBay image uploads outright, since eBay's Inventory API requires (and silently drops
# images from) any imageUrls value that isn't HTTPS. Confirmed via a local repro before
# and after this flag.
CMD uv run uvicorn src.web.app:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'
