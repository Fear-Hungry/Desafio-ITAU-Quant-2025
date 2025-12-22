# syntax=docker/dockerfile:1

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_HOME=/opt/poetry \
    POETRY_VERSION=1.8.3 \
    POETRY_NO_INTERACTION=1 \
    PROJECT_ROOT=/workspace \
    PYTHONPATH=/workspace/src

ENV PATH="$POETRY_HOME/bin:$PATH"

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    nodejs \
    npm \
    pkg-config \
    ca-certificates \
    libopenblas-dev \
    libblas-dev \
    liblapack-dev \
    libgomp1 \
    gfortran \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Codex CLI (so it is available without relying on postCreate networking)
RUN npm install -g @openai/codex \
    && npm cache clean --force

RUN curl -sSL https://install.python-poetry.org | python3 - \
    && poetry config virtualenvs.create false \
    && poetry config installer.max-workers 8

# Set working directory to workspace (standard for this project)
WORKDIR $PROJECT_ROOT

# Copy dependency files first for caching
COPY pyproject.toml poetry.lock* ./

# Install dependencies (including dev)
RUN poetry install --with dev --no-interaction --no-ansi --no-root

# Copy source code and project files
COPY src ./src
COPY configs ./configs
COPY tests ./tests
COPY README.md Makefile LICENSE ./
COPY docs ./docs

# Install the project itself (keeps dependency layer cacheable)
RUN poetry install --only-root --no-interaction --no-ansi

# Default command
CMD ["bash"]
