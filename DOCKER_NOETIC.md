# ROS Noetic Docker 명령

## 환경 시작

역할: Docker 컨테이너와 ROS용 Terminator 6분할 창을 시작합니다.

```bash
cd ~/tb3_autorace_noetic_ws
./docker/start_sim_terminator.sh
```

## 로봇 모델과 센서 단독 확인

역할: Xacro 로봇 모델을 빌드하고 GUI에서 확인합니다.

```bash
cd /workspace
./docker/build_workspace.sh
source devel/setup.bash
roslaunch custom_autorace_description description.launch use_gui:=true
```

역할: 미션 제어기 없이 기본 AutoRace 코스와 Gazebo 센서를 실행하고 터널 원통 배치를 실행마다 A/B/C 순환 변경합니다.

```bash
roslaunch custom_autorace_description gazebo_autorace.launch
```

역할: 별도 코스 이미지를 원본 모델 수정 없이 적용합니다.

```bash
roslaunch custom_autorace_description gazebo_autorace.launch \
  course_texture:=/workspace/maps/changed_course.png
```

역할: 터널 원통을 재현 가능한 B 배치로 실행합니다.

```bash
roslaunch custom_autorace_description gazebo_autorace.launch \
  tunnel_obstacle_layout:=layout_b
```

역할: Gazebo 실제 위치 기반 world odometry를 사용합니다.

```bash
roslaunch custom_autorace_description gazebo_autorace.launch odometry_source:=world
```

역할: EKF 없이 encoder odometry만 시험합니다.

```bash
roslaunch custom_autorace_description gazebo_autorace.launch fuse_imu:=false
```

## 통합 자동주행

역할: 공식 시작 자세에서 전체 미션, 센서, AMCL, EKF와 RViz를 실행하고 터널 원통 배치를 실행마다 A/B/C 순환 변경합니다.

```bash
roslaunch custom_autorace_bringup gazebo.launch
```

역할: 신호등 대기 없이 카메라 방향지시판 판독부터 실행합니다.

```bash
roslaunch custom_autorace_bringup gazebo.launch \
  wait_for_green:=false forced_direction:=0
```

## 교차로 회귀

역할: 왼쪽 경로를 강제해 교차로 진입부터 차선 복귀까지 실행합니다.

```bash
roslaunch custom_autorace_bringup gazebo.launch \
  wait_for_green:=false forced_direction:=2
```

역할: 오른쪽 경로를 강제해 교차로 진입부터 차선 복귀까지 실행합니다.

```bash
roslaunch custom_autorace_bringup gazebo.launch \
  wait_for_green:=false forced_direction:=3
```

## 수동주행과 센서 시험

역할: 모든 자동 `/cmd_vel` 제어기를 끄고 Gazebo 센서만 실행합니다.

```bash
roslaunch custom_autorace_bringup gazebo.launch \
  drive_lane:=false intersection_mission:=false obstacle_mission:=false \
  parking_mission:=false zigzag_mission:=false \
  level_crossing_mission:=false tunnel_mission:=false wait_for_green:=false
```

역할: 키보드 텔레옵을 실행합니다.

```bash
roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
```

## 실행 옵션

역할: RViz 없이 실행합니다.

```bash
roslaunch custom_autorace_bringup gazebo.launch rviz:=false
```

역할: 표지판 검출기 없이 실행합니다.

```bash
roslaunch custom_autorace_bringup gazebo.launch detect_signs:=false
```

역할: EKF 없이 raw `/odom`과 odom TF를 사용합니다.

```bash
roslaunch custom_autorace_bringup gazebo.launch fuse_imu:=false
```

역할: 후진 부호 검증용 world odometry를 사용합니다.

```bash
roslaunch custom_autorace_bringup gazebo.launch odometry_source:=world fuse_imu:=false
```

역할: encoder와 IMU를 결합한 전진 odometry를 시험합니다.

```bash
roslaunch custom_autorace_bringup gazebo.launch odometry_source:=encoder fuse_imu:=true
```

역할: 개별 미션 제어기를 비활성화합니다.

```bash
roslaunch custom_autorace_bringup gazebo.launch obstacle_mission:=false
roslaunch custom_autorace_bringup gazebo.launch parking_mission:=false
roslaunch custom_autorace_bringup gazebo.launch zigzag_mission:=false
roslaunch custom_autorace_bringup gazebo.launch level_crossing_mission:=false
roslaunch custom_autorace_bringup gazebo.launch tunnel_mission:=false
```

역할: 로봇과 AMCL의 초기 자세를 지정합니다.

```bash
roslaunch custom_autorace_bringup gazebo.launch \
  x_pos:=0.8 y_pos:=-1.747 yaw_pos:=0.0
```

## 장애물 단독 회귀

역할: 공사 구간 직전에서 장애물 미션과 차선 복귀를 실행합니다.

```bash
roslaunch custom_autorace_bringup gazebo.launch \
  x_pos:=1.6375 y_pos:=-0.05 yaw_pos:=1.5707963 \
  wait_for_green:=true detect_signs:=false intersection_mission:=false \
  parking_mission:=false zigzag_mission:=false level_crossing_mission:=false \
  tunnel_mission:=false \
  mission_zone_config:=$(rospack find custom_autorace_bringup)/config/mission_zones_obstacle_test_gazebo.yaml
```

## 주차 단독 회귀

역할: 우측 장애물로 좌측 빈 주차공간 선택과 복귀를 실행합니다.

```bash
roslaunch custom_autorace_bringup gazebo.launch \
  gui:=false rviz:=false x_pos:=1.20 y_pos:=1.735 yaw_pos:=3.14159265 \
  odometry_source:=world \
  wait_for_green:=false detect_signs:=false \
  intersection_mission:=false obstacle_mission:=false zigzag_mission:=false \
  level_crossing_mission:=false tunnel_mission:=false \
  parking_obstacle_x:=0.23 \
  mission_zone_config:=$(rospack find custom_autorace_bringup)/config/mission_zones_parking_test_gazebo.yaml
```

역할: 좌측 장애물로 우측 빈 주차공간 선택과 복귀를 실행합니다.

```bash
roslaunch custom_autorace_bringup gazebo.launch \
  gui:=false rviz:=false x_pos:=1.20 y_pos:=1.735 yaw_pos:=3.14159265 \
  odometry_source:=world \
  wait_for_green:=false detect_signs:=false \
  intersection_mission:=false obstacle_mission:=false zigzag_mission:=false \
  level_crossing_mission:=false tunnel_mission:=false \
  parking_obstacle_x:=0.73 \
  mission_zone_config:=$(rospack find custom_autorace_bringup)/config/mission_zones_parking_test_gazebo.yaml
```

## 지그재그 단독 회귀

역할: 주차 뒤 직선에서 지그재그 경로 추종과 도색 여유를 시험합니다.

```bash
roslaunch custom_autorace_bringup gazebo.launch \
  gui:=false rviz:=false x_pos:=0.30 y_pos:=1.75 yaw_pos:=3.14159265 \
  odometry_source:=world wait_for_green:=false detect_signs:=false \
  intersection_mission:=false obstacle_mission:=false parking_mission:=false zigzag_mission:=true \
  level_crossing_mission:=false tunnel_mission:=false \
  mission_zone_config:=$(rospack find custom_autorace_bringup)/config/mission_zones_zigzag_test_gazebo.yaml
```

## 차단봉 단독 회귀

역할: 차단봉 닫힘·정지·개방·차선 재개를 시험합니다.

```bash
roslaunch custom_autorace_bringup gazebo.launch \
  gui:=false rviz:=false x_pos:=-1.60 y_pos:=1.25 yaw_pos:=0.0 \
  odometry_source:=world wait_for_green:=false detect_signs:=false \
  mission_models_initial_state:=6 \
  intersection_mission:=false obstacle_mission:=false \
  parking_mission:=false zigzag_mission:=false level_crossing_mission:=true tunnel_mission:=false \
  mission_zone_config:=$(rospack find custom_autorace_bringup)/config/mission_zones_level_crossing_test_gazebo.yaml
```

## 터널 단독 회귀

역할: 터널 직전에서 costmap, Hybrid A*와 차선 복귀를 중간 검증합니다.

```bash
roslaunch custom_autorace_bringup gazebo.launch \
  gui:=false rviz:=true x_pos:=-1.75 y_pos:=0.18 yaw_pos:=-1.57079632679 \
  odometry_source:=world wait_for_green:=false mission_models:=false \
  intersection_mission:=false obstacle_mission:=false parking_mission:=false \
  zigzag_mission:=false level_crossing_mission:=false tunnel_mission:=true \
  tunnel_obstacle_layout:=layout_a \
  mission_zone_config:=$(rospack find custom_autorace_bringup)/config/mission_zones_tunnel_test_gazebo.yaml
```

역할: 터널 직전에서 90도 회전한 B 기둥 배치와 Hybrid A* 경로를 실행합니다.

```bash
roslaunch custom_autorace_bringup gazebo.launch \
  gui:=false rviz:=true x_pos:=-1.75 y_pos:=0.18 yaw_pos:=-1.57079632679 \
  odometry_source:=world wait_for_green:=false mission_models:=false \
  intersection_mission:=false obstacle_mission:=false parking_mission:=false \
  zigzag_mission:=false level_crossing_mission:=false tunnel_mission:=true \
  tunnel_obstacle_layout:=layout_b \
  mission_zone_config:=$(rospack find custom_autorace_bringup)/config/mission_zones_tunnel_test_gazebo.yaml
```

역할: 터널 직전에서 180도 회전한 C 기둥 배치와 Hybrid A* 경로를 실행합니다.

```bash
roslaunch custom_autorace_bringup gazebo.launch \
  gui:=false rviz:=true x_pos:=-1.75 y_pos:=0.18 yaw_pos:=-1.57079632679 \
  odometry_source:=world wait_for_green:=false mission_models:=false \
  intersection_mission:=false obstacle_mission:=false parking_mission:=false \
  zigzag_mission:=false level_crossing_mission:=false tunnel_mission:=true \
  tunnel_obstacle_layout:=layout_c \
  mission_zone_config:=$(rospack find custom_autorace_bringup)/config/mission_zones_tunnel_test_gazebo.yaml
```

이 단독 실행은 공식 시작점 통합 완료 판정으로 사용하지 않습니다.

## 일반 차선 rolling CommonPath 회귀

역할: 출발선부터 교차로 gate까지 0.28 m/s 경로 추종 결과를 기록합니다.

```bash
roslaunch custom_autorace_bringup gazebo_lane_path_test.launch \
  gui:=false rviz:=false result_file:=/tmp/lane_path_result.json
```

역할: 0.20 m/s 속도 프로파일을 비교 기록합니다.

```bash
roslaunch custom_autorace_bringup gazebo_lane_path_test.launch \
  gui:=false rviz:=false test_velocity:=0.20 \
  result_file:=/tmp/lane_path_020.json
```

## 주행 정지·재개

역할: 모든 자동주행 제어기를 정지합니다.

```bash
rosservice call /control/lane_following "data: false"
```

역할: 수동 정지를 해제하고 주행을 재개합니다.

```bash
rosservice call /control/lane_following "data: true"
```

## 미션 상태 확인

역할: 현재 미션과 순차 진행 상태를 확인합니다.

```bash
rostopic echo -n 1 /mission/current
rostopic echo -n 1 /mission/state
rostopic echo -n 1 /mission/sequence_index
```

## 교차로 상태 확인

역할: 교차로 arm, 센서 등록 ready, gate, 표지판, 경로와 차선 복귀를 확인합니다.

```bash
rostopic echo /mission/current
rostopic echo -n 1 /mission/arm/intersection
rostopic echo -n 1 /mission/ready/intersection
rostopic echo /mission/enable/intersection
rostopic echo /detect/signs
rostopic echo /intersection/direction
rostopic echo /intersection/state
rostopic echo -n 1 /intersection/generated_path
rostopic echo -n 1 /control/lane_path
rostopic echo /intersection/diagnostics
rostopic echo /detect/lane_boundaries
rqt_image_view /detect/image_signs
```

## 장애물 상태 확인

역할: 장애물 arm, LiDAR 등록 ready, gate, 경로, 상태와 안전 여유를 확인합니다.

```bash
rostopic echo -n 1 /mission/arm/obstacle
rostopic echo -n 1 /mission/ready/obstacle
rostopic echo /mission/enable/obstacle
rostopic echo /obstacle/state
rostopic echo /obstacle/planner_status
rostopic echo -n 1 /obstacle/local_path
rostopic echo /obstacle/diagnostics
```

## 주차 상태 확인

역할: 주차 arm, LiDAR 등록 ready, 공간 선택, 경로, 진단과 `/cmd_vel` 제어권을 확인합니다.

```bash
rostopic echo -n 1 /mission/arm/parking
rostopic echo -n 1 /mission/ready/parking
rostopic echo /mission/enable/parking
rostopic echo /odom
rostopic echo /parking/occupancy_points
rostopic echo /parking/selected_space
rostopic echo /parking/state
rostopic echo -n 1 /parking/planned_path/left
rostopic echo -n 1 /parking/planned_path/right
rostopic echo /parking/diagnostics
rostopic echo /control/lane_path_diagnostics
rostopic echo /control/max_vel
rostopic echo /cmd_vel
rostopic info /cmd_vel
rosservice info /control/lane_mission_handoff
```

## 지그재그 상태 확인

역할: 지그재그 arm, 카메라 곡선 등록 ready, 경로, 도색 여유와 제어권을 확인합니다.

```bash
rostopic echo -n 1 /mission/arm/zigzag
rostopic echo -n 1 /mission/ready/zigzag
rostopic echo /mission/enable/zigzag
rostopic echo /odom
rostopic echo /zigzag/state
rostopic echo -n 1 /zigzag/path
rostopic echo -n 1 /control/lane_path
rostopic echo /zigzag/diagnostics
rostopic echo /detect/lane_boundaries
rostopic echo /control/max_vel
rostopic echo /control/manual_stop
rostopic echo /cmd_vel
rosservice info /control/lane_mission_handoff
```

## 차단봉 상태 확인

역할: 차단봉 arm, 고정 landmark 등록 ready, LiDAR 판정과 제어권을 확인합니다.

```bash
rostopic echo -n 1 /mission/arm/level_crossing
rostopic echo -n 1 /mission/ready/level_crossing
rostopic echo /mission/enable/level_crossing
rostopic echo /level_crossing/landmark_pose
rostopic echo /level_crossing/state
rostopic echo /level_crossing/barrier_down
rostopic echo /level_crossing/diagnostics
rostopic echo /scan_mid360_raw
rostopic echo /cmd_vel
rostopic info /cmd_vel
rosservice info /control/lane_mission_handoff
```

## 터널 상태 확인

역할: 터널 arm, portal 등록 ready, costmap, Hybrid A* 경로와 제어권을 확인합니다.

```bash
rostopic echo -n 1 /mission/arm/tunnel
rostopic echo -n 1 /mission/ready/tunnel
rostopic echo /mission/enable/tunnel
rostopic echo /tunnel/state
rostopic echo -n 1 /tunnel/path
rostopic echo -n 1 /tunnel/costmap
rostopic echo /tunnel/diagnostics
rostopic echo /control/lane_path_diagnostics
rostopic echo /control/max_vel
rostopic echo /cmd_vel
rostopic info /cmd_vel
rosservice info /control/lane_mission_handoff
```

## 카메라와 토픽 확인

역할: D405 1280x720@30 입력과 30 Hz 인지 파이프라인을 실행합니다.

```bash
roslaunch custom_autorace_bringup hardware.launch \
  start_opencr:=false start_lidar:=false \
  projection_config:=/absolute/path/to/measured_d405_projection.yaml
```

역할: 카메라 입력과 인지 단계별 영상을 확인합니다.

```bash
rqt_image_view /camera/color/image_raw
rqt_image_view /camera/image_rect_color
rqt_image_view /camera/image_rect_color/compressed
rqt_image_view /camera/image_projected
rqt_image_view /camera/image_projected_compensated
rqt_image_view /detect/image_lane
rqt_image_view /detect/image_signs
```

역할: 카메라, 인지, 센서와 제어 토픽의 주기를 확인합니다.

```bash
rostopic hz /camera/color/image_raw
rostopic hz /camera/image_rect_color
rostopic hz /camera/image_rect_color/compressed
rostopic hz /camera/image_projected
rostopic hz /camera/image_projected_compensated
rostopic hz /detect/lane
rostopic hz /detect/lane_boundaries
rostopic hz /detect/lane_centerline
rostopic hz /detect/signs
rostopic hz /detect/image_traffic_light/compressed
rostopic hz /scan_mid360_raw
rostopic hz /odometry/filtered
rostopic hz /odom
rostopic hz /cmd_vel
```

역할: 실행 중인 ROS 노드와 토픽을 확인합니다.

```bash
rosnode list
rostopic list
```

역할: 기본 통합 bringup을 실행하고 RViz에서 계획 경로, Gazebo 실제 위치와 EKF 궤적을 실시간 표시합니다.

```bash
roslaunch custom_autorace_bringup gazebo.launch
```

역할: 기본 통합 bringup의 RViz 주행 궤적을 초기화합니다.

```bash
rosservice call /trajectory/reset
```

## Docker 셸, 빌드와 테스트

역할: 실행 중인 Noetic 컨테이너 셸에 진입합니다.

```bash
docker exec -it custom-autorace-noetic bash
```

역할: 전체 워크스페이스를 빌드하고 overlay를 적용합니다.

```bash
cd /workspace
./docker/build_workspace.sh
source devel/setup.bash
```

역할: workspace AMCL이 선택됐는지 확인합니다.

```bash
rospack find amcl
```

역할: 실행 중인 AMCL odometry 모델을 확인합니다.

```bash
rosparam get /amcl/odom_model_type
```

역할: 전체 workspace의 등록 시험을 실행하고 결과를 집계합니다.

```bash
cd /workspace
./docker/build_workspace.sh
source devel/setup.bash
catkin_make run_tests
catkin_test_results build/test_results
```

역할: 터널과 통합 launch 관련 회귀만 빠르게 실행합니다.

```bash
cd /workspace
source /opt/ros/noetic/setup.bash
source devel/setup.bash
nosetests3 -q \
  src/custom_autorace_bringup/test/test_tunnel_costmap.py \
  src/custom_autorace_bringup/test/test_tunnel_planner.py \
  src/custom_autorace_bringup/test/test_tunnel_controller.py \
  src/custom_autorace_bringup/test/test_gazebo_launch_wiring.py
```

## 종료

역할: Docker Compose 서비스를 정지합니다.

```bash
cd ~/tb3_autorace_noetic_ws
docker compose -f compose.noetic.yaml stop autorace
```
