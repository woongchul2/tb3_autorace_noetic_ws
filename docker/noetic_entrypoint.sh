#!/usr/bin/env bash
set -e

source /opt/ros/noetic/setup.bash

if [ -f /workspace/devel/setup.bash ]; then
  source /workspace/devel/setup.bash
fi

exec "$@"

