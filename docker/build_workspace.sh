#!/usr/bin/env bash
set -e

cd /workspace

rosdep check --from-paths src --ignore-src --rosdistro noetic \
  --skip-keys="librealsense2 python-enum34 python-numpy"

catkin_make \
  -DROS_EDITION=ROS1 \
  -Drealsense2_DIR=/usr/local/lib/cmake/realsense2

echo
echo "Build complete. Run: source /workspace/devel/setup.bash"
