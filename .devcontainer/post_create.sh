#!/usr/bin/env bash
set -euo pipefail

# Ensure Poetry and Codex binaries are on PATH
if [ -d "/opt/poetry/bin" ] && ! echo "$PATH" | tr ':' '\n' | grep -Fx "/opt/poetry/bin" >/dev/null; then
    export PATH="/opt/poetry/bin:$PATH"
fi

CODEX_BIN="${HOME}/.codex/bin"
if [ -d "${CODEX_BIN}" ] && ! echo "$PATH" | tr ':' '\n' | grep -Fx "${CODEX_BIN}" >/dev/null; then
    export PATH="${CODEX_BIN}:$PATH"
fi

if command -v codex >/dev/null 2>&1; then
    echo "[post-create] Codex CLI detected at $(command -v codex)."
else
    echo "[post-create] Codex CLI not found in PATH; installing via npm..."
    if command -v npm >/dev/null 2>&1 && npm install -g @openai/codex; then
        echo "[post-create] Codex CLI installed globally with npm."
    else
        echo "[post-create] npm install failed; attempting download via install script..."
        if curl -fsSL https://cli.codex.cloud/install.sh | bash; then
            echo "[post-create] Codex CLI installed via bootstrap script."
        else
            echo "[post-create] WARNING: Failed to provision Codex CLI (network/TLS issue)." >&2
            echo "[post-create]          Please install manually inside the container if needed." >&2
        fi
    fi
fi

# Ensure Poetry exists (should already be baked into image, but guard just in case)
if ! command -v poetry >/dev/null 2>&1; then
    echo "[post-create] Poetry not found; attempting to install..."
    if curl -fsSL https://install.python-poetry.org | python3 -; then
        export PATH="${HOME}/.local/bin:$PATH"
        echo "[post-create] Poetry installed to ${HOME}/.local/bin."
    else
        echo "[post-create] ERROR: Failed to install Poetry; please install manually." >&2
        exit 1
    fi
fi

# Install project dependencies with Poetry
echo "[post-create] Installing Poetry dependencies..."
poetry install
