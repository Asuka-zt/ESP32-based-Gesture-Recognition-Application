#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/esp32-gesture-mouse-uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/tmp/esp32-gesture-mouse-python}"

uv sync --all-extras
uv run ruff check .
uv run pytest

if [[ ! -f firmware/include/wifi_secrets.h ]]; then
  cp firmware/include/wifi_secrets.example.h firmware/include/wifi_secrets.h
fi
uv tool run platformio run --project-dir firmware

