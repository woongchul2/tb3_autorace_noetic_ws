#!/usr/bin/env bash
set -e

PANEL_NAME="${1:-}"

source /opt/ros/noetic/setup.bash

if [[ -f /workspace/devel/setup.bash ]]; then
  source /workspace/devel/setup.bash
else
  echo "안내: devel/setup.bash가 없습니다. ./docker/build_workspace.sh로 먼저 빌드하세요." >&2
fi

cd /workspace
if [[ -n "$PANEL_NAME" ]]; then
  export PS1="[$PANEL_NAME] \u@\h:\w\$ "
else
  export PS1="\u@\h:\w\$ "
fi
exec bash --norc -i
