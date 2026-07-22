FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY . .
RUN uv sync --frozen

# Bake the ResNet50 weights into the image so a cold start (Render's free
# tier spins down after inactivity) doesn't need a ~100MB download first.
RUN uv run python -c "from src.ml.vision_preprocessor import VisionPreprocessor; VisionPreprocessor()"

EXPOSE 8000

CMD uv run uvicorn src.web.app:app --host 0.0.0.0 --port ${PORT:-8000}
