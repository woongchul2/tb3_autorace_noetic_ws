#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER_NAME="custom-autorace-noetic"
LAYOUT_FILE="$WORKSPACE_DIR/docker/terminator_layout.json"

if ! command -v terminator >/dev/null 2>&1; then
  echo "오류: 호스트에 Terminator가 설치되어 있지 않습니다." >&2
  echo "설치 명령: sudo apt install terminator" >&2
  exit 1
fi

cd "$WORKSPACE_DIR"

export LOCAL_UID="${LOCAL_UID:-$(id -u)}"
export LOCAL_GID="${LOCAL_GID:-$(id -g)}"

xhost +local:docker >/dev/null
docker compose -f compose.noetic.yaml up -d autorace

if ! docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null | grep -qx true; then
  echo "오류: 컨테이너를 시작하지 못했습니다: $CONTAINER_NAME" >&2
  exit 1
fi

terminator --no-dbus --geometry=1200x800 --title="AutoRace Noetic" \
  --config-json="$LAYOUT_FILE" >/dev/null 2>&1 &

echo "AutoRace Terminator 6분할 창을 열었습니다."
