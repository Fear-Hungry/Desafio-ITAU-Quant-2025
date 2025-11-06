# Use Python 3.11 official image (compatible with pyproject.toml requirements)
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    git \
    curl \
    ca-certificates \
    openssl \
    build-essential \
    nodejs \
    npm \
    && update-ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_VERSION=1.8.0
ENV POETRY_HOME="/opt/poetry"
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
ENV POETRY_NO_INTERACTION=1
ENV PATH="$POETRY_HOME/bin:$PATH"

RUN curl -sSL https://install.python-poetry.org | python3 - \
    && poetry --version

# Set working directory
WORKDIR /workspace

# Copy only dependency files first (for better caching)
COPY pyproject.toml poetry.lock* /workspace/

# Install dependencies (will be run in postCreateCommand for devcontainer)
# RUN poetry install --no-root --no-interaction

# Default command
CMD ["/bin/bash"]
