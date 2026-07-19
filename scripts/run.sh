#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "已创建 .env，请确认 ESP32 地址和鼠标控制开关。"
fi

exec uv run gesture-mouse

