# ROS Noetic Docker 실행 명령

## 시뮬레이션 환경 시작

역할: Docker 컨테이너와 ROS용 Terminator 6분할 창 시작.

```bash
cd ~/tb3_autorace_noetic_ws
./docker/start_sim_terminator.sh
```

## 교차로·장애물·주차·지그재그·차단봉·터널 자동주행

역할: signed world odom을 EKF로 융합하고 신호등 출발, 교차로·장애물·주차·지그재그·라이다 차단봉·Hybrid A* 터널, 센서·AMCL·RViz 실행.

```bash
roslaunch custom_autorace_bringup gazebo.launch
```

역할: 실제 카메라 방향지시판 판독, 반원 진입, 영상 반원 추종, AMCL 지도 자세 기반 제어권 회수와 공통 탈출 경로 실행.

```bash
roslaunch custom_autorace_bringup gazebo.launch \
  wait_for_green:=false forced_direction:=0
```

## 교차로 회귀

역할: 왼쪽 방향을 강제하고 반원 진입, 영상 반원 추종, AMCL 지도 자세 기반 제어권 회수와 공통 탈출 경로 회귀.

```bash
roslaunch custom_autorace_bringup gazebo.launch \
  wait_for_green:=false forced_direction:=2
```

역할: 오른쪽 방향을 강제하고 반원 진입, 영상 반원 추종, AMCL 지도 자세 기반 제어권 회수와 공통 탈출 경로 회귀.

```bash
roslaunch custom_autorace_bringup gazebo.launch \
  wait_for_green:=false forced_direction:=3
```

## 수동주행과 센서 시험

역할: 자동 `/cmd_vel` 제어기를 끄고 Gazebo 센서만 실행.

```bash
roslaunch custom_autorace_bringup gazebo.launch \
  drive_lane:=false intersection_mission:=false obstacle_mission:=false \
  parking_mission:=false zigzag_mission:=false \
  level_crossing_mission:=false tunnel_mission:=false wait_for_green:=false
```

역할: 키보드 텔레옵 실행.

```bash
roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
```

## 실행 옵션

역할: RViz 없이 실행.

```bash
roslaunch custom_autorace_bringup gazebo.launch rviz:=false
```

역할: 표지판 검출기 없이 실행.

```bash
roslaunch custom_autorace_bringup gazebo.launch detect_signs:=false
```

역할: EKF 없이 `/odom`과 raw odom TF를 쓰는 Gazebo 진단 실행.

```bash
roslaunch custom_autorace_bringup gazebo.launch fuse_imu:=false
```

역할: 후진 부호 검증용 `/odom`과 raw odom TF 진단 실행.

```bash
roslaunch custom_autorace_bringup gazebo.launch odometry_source:=world fuse_imu:=false
```

역할: 전진 전용 encoder+IMU EKF odometry 비교 시험.

```bash
roslaunch custom_autorace_bringup gazebo.launch odometry_source:=encoder fuse_imu:=true
```

역할: 장애물 미션 제어기 없이 실행.

```bash
roslaunch custom_autorace_bringup gazebo.launch obstacle_mission:=false
```

역할: 주차 미션 제어기 없이 실행.

```bash
roslaunch custom_autorace_bringup gazebo.launch parking_mission:=false
```

역할: 지그재그 미션 제어기 없이 실행.

```bash
roslaunch custom_autorace_bringup gazebo.launch zigzag_mission:=false
```

역할: 라이다 차단봉 미션 제어기 없이 실행.

```bash
roslaunch custom_autorace_bringup gazebo.launch level_crossing_mission:=false
```

역할: Hybrid A* 터널 미션 제어기 없이 실행.

```bash
roslaunch custom_autorace_bringup gazebo.launch tunnel_mission:=false
```

역할: 로봇과 AMCL 초기 자세 지정.

```bash
roslaunch custom_autorace_bringup gazebo.launch \
  x_pos:=0.8 y_pos:=-1.747 yaw_pos:=0.0
```

## 장애물 단독 회귀

역할: 센서 준비 중 차선을 정지하고 공사 구간 직전 자세와 장애물 전용 순서로 실행.

```bash
roslaunch custom_autorace_bringup gazebo.launch \
  x_pos:=1.6375 y_pos:=-0.05 yaw_pos:=1.5707963 \
  wait_for_green:=true detect_signs:=false intersection_mission:=false \
  parking_mission:=false zigzag_mission:=false level_crossing_mission:=false \
  tunnel_mission:=false \
  mission_zone_config:=$(rospack find custom_autorace_bringup)/config/mission_zones_obstacle_test_gazebo.yaml
```

## 주차 단독 회귀

역할: 우측 주차면 장애물 고정, AMCL gate·LiDAR 좌측 빈 칸 선택·중앙 판정 자세의 좌측 칸 방향 제자리회전·직선 주차/즉시 후진·북쪽 제자리회전·복귀점에서 지그재그 회전 시작점까지의 단일 quintic·이동 중 차선 인계를 회귀.

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

역할: 좌측 주차면 장애물 고정, AMCL gate·LiDAR 우측 빈 칸 선택·중앙 판정 자세의 우측 칸 방향 제자리회전·직선 주차/즉시 후진·북쪽 제자리회전·복귀점에서 지그재그 회전 시작점까지의 단일 quintic·이동 중 차선 인계를 회귀.

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

역할: 주차 뒤 직선의 현재 odom 진행 위치에서 적응 인계하고 lookahead·곡률 선행 감속과 실제 도색 여유를 회귀.

```bash
roslaunch custom_autorace_bringup gazebo.launch \
  gui:=false rviz:=false x_pos:=0.30 y_pos:=1.75 yaw_pos:=3.14159265 \
  odometry_source:=world wait_for_green:=false detect_signs:=false \
  intersection_mission:=false obstacle_mission:=false parking_mission:=false zigzag_mission:=true \
  level_crossing_mission:=false tunnel_mission:=false \
  mission_zone_config:=$(rospack find custom_autorace_bringup)/config/mission_zones_zigzag_test_gazebo.yaml
```

## 차단봉 단독 회귀

역할: 차단봉 직전 자세에서 라이다 닫힘 확인·정지·개방 확인·차선 재개·구역 이탈을 회귀.

```bash
roslaunch custom_autorace_bringup gazebo.launch \
  gui:=false rviz:=false x_pos:=-1.40 y_pos:=1.25 yaw_pos:=0.0 \
  odometry_source:=world wait_for_green:=false detect_signs:=false \
  mission_models_initial_state:=6 \
  intersection_mission:=false obstacle_mission:=false \
  parking_mission:=false zigzag_mission:=false level_crossing_mission:=true tunnel_mission:=false \
  mission_zone_config:=$(rospack find custom_autorace_bringup)/config/mission_zones_level_crossing_test_gazebo.yaml
```

## 터널 단독 회귀

역할: 터널 입구 직전 자세에서 AMCL map-odom anchor를 고정하고 frozen 입구·출구 `CommonPath` 사이의 미션 전용 LiDAR costmap·전진 Hybrid A*와 공유 `/control/lane_path_diagnostics` 기반 차선 handoff를 중간 회귀하며 공식 시작 통합 완료 판정에는 사용하지 않음.

```bash
roslaunch custom_autorace_bringup gazebo.launch \
  gui:=false rviz:=true x_pos:=-1.75 y_pos:=0.18 yaw_pos:=-1.57079632679 \
  odometry_source:=world wait_for_green:=false mission_models:=false \
  intersection_mission:=false obstacle_mission:=false parking_mission:=false \
  zigzag_mission:=false level_crossing_mission:=false tunnel_mission:=true \
  mission_zone_config:=$(rospack find custom_autorace_bringup)/config/mission_zones_tunnel_test_gazebo.yaml
```

## 일반 차선 rolling CommonPath 회귀

역할: 0.28 m/s 카메라 rolling `CommonPath`의 공통 직사각형 sweep·`PathFollower`·13항목 진단을 출발선→교차로 gate에서 JSON으로 측정.

```bash
roslaunch custom_autorace_bringup gazebo_lane_path_test.launch \
  gui:=false rviz:=false result_file:=/tmp/lane_path_result.json
```

역할: 카메라 rolling 경로의 단계별 속도 회귀 실행.

```bash
roslaunch custom_autorace_bringup gazebo_lane_path_test.launch \
  gui:=false rviz:=false test_velocity:=0.20 \
  result_file:=/tmp/lane_path_020.json
```

## 주행 정지·재개

역할: 모든 자동주행 제어기의 수동 정지 활성화.

```bash
rosservice call /control/lane_following "data: false"
```

역할: 수동 정지 해제와 주행 재개.

```bash
rosservice call /control/lane_following "data: true"
```

## 미션 상태 확인

역할: 현재 미션, 순차 상태와 인덱스 확인.

```bash
rostopic echo -n 1 /mission/current
rostopic echo -n 1 /mission/state
rostopic echo -n 1 /mission/sequence_index
```

## 교차로 상태 확인

역할: AMCL과 미션 지도 자세 확인.

```bash
rostopic echo /amcl_pose
rostopic echo /mission/map_pose
```

역할: 교차로 gate, 조기 방향지시판 관찰창과 AMCL 지도 자세 확인.

```bash
rostopic echo /mission/current
rostopic echo /mission/enable/intersection
rostopic echo /mission/inside/intersection
rostopic echo /mission/inside/intersection_direction_observation
```

역할: 방향지시판 판독, 교차로 상태, 반원 진입 경로, 별도 탈출 경로와 최종 두 차선 확인.

```bash
rostopic echo /detect/signs
rostopic echo /intersection/direction
rostopic echo /intersection/state
rostopic echo -n 1 /intersection/generated_path
rostopic echo /intersection/diagnostics
rostopic echo /detect/lane_boundaries
rqt_image_view /detect/image_signs
```

## 장애물 상태 확인

역할: 장애물 gate·구역 이탈, 단일 곡률 연속 경로의 상태와 안전 여유 확인.

```bash
rostopic echo /mission/enable/obstacle
rostopic echo /mission/inside/obstacle
rostopic echo /mission/clear/obstacle
rostopic echo /obstacle/state
rostopic echo /obstacle/planner_status
rostopic echo -n 1 /obstacle/local_path
rostopic echo /obstacle/diagnostics
```

## 주차 상태 확인

역할: AMCL gate·보정량 진단, local odom 직선 주차/복귀와 두 제자리회전 sweep, 좌·우 점유 판정, 차선→주차 무정지 인계 및 공통 13항목 진단으로 검증된 rolling CommonPath 인계 확인.

```bash
rostopic echo /mission/enable/parking
rostopic echo /mission/map_pose
rostopic echo /odom
rostopic echo /parking/occupancy_points
rostopic echo /parking/selected_space
rostopic echo /parking/state
rostopic echo /parking/diagnostics
rostopic echo /control/lane_path_diagnostics
rostopic echo /control/max_vel
rostopic echo /cmd_vel
rostopic info /cmd_vel
rosservice info /control/lane_mission_handoff
```

## 지그재그 상태 확인

역할: 지그재그 gate, 현재 odom 투영 경로와 실제 도색 침범·바깥 여유·정지거리 예측 진단 확인.

```bash
rostopic echo /mission/enable/zigzag
rostopic echo /mission/inside/zigzag
rostopic echo /mission/clear/zigzag
rostopic echo /mission/map_pose
rostopic echo /odom
rostopic echo /zigzag/state
rostopic echo -n 1 /zigzag/path
rostopic echo /zigzag/diagnostics
rostopic echo /detect/lane_boundaries
rostopic echo /control/max_vel
rostopic echo /control/manual_stop
rostopic echo /cmd_vel
rosservice info /control/lane_mission_handoff
```

## 차단봉 상태 확인

역할: 차단봉 gate·라이다 판정·정지/통과 상태와 `/cmd_vel` 제어권 인계 확인.

```bash
rostopic echo /mission/enable/level_crossing
rostopic echo /mission/inside/level_crossing
rostopic echo /mission/clear/level_crossing
rostopic echo /level_crossing/state
rostopic echo /level_crossing/barrier_down
rostopic echo /level_crossing/diagnostics
rostopic echo /scan_mid360_raw
rostopic echo /cmd_vel
rostopic info /cmd_vel
rosservice info /control/lane_mission_handoff
```

## 터널 상태 확인

역할: 터널 gate·AMCL 자세·LiDAR 동적 costmap·Hybrid A* 경로·상태·공유 차선 진단과 `/cmd_vel` 제어권 인계 확인.

```bash
rostopic echo /mission/enable/tunnel
rostopic echo /mission/inside/tunnel
rostopic echo /mission/clear/tunnel
rostopic echo /mission/map_pose
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

## 카메라·토픽 확인

역할: 실물 D405 1280x720@30 입력에서 차선은 매 프레임 처리하고 temporal state는 3프레임 주기로 갱신하는 신호등·표지판 검출 파이프라인 실행.

```bash
roslaunch custom_autorace_bringup hardware.launch \
  start_opencr:=false start_lidar:=false \
  projection_config:=/absolute/path/to/measured_d405_projection.yaml
```

역할: 원본·투영·차선·표지판 영상 확인.

```bash
rqt_image_view /camera/color/image_raw
rqt_image_view /camera/image_rect_color
rqt_image_view /camera/image_rect_color/compressed
rqt_image_view /camera/image_projected
rqt_image_view /camera/image_projected_compensated
rqt_image_view /detect/image_lane
rqt_image_view /detect/image_signs
```

역할: 카메라 입력·중간 영상·30 Hz 검출 출력과 주요 센서·제어 토픽 주기 확인.

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

역할: 실행 중인 ROS 노드와 토픽 확인.

```bash
rosnode list
rostopic list
```

역할: RViz 주행 궤적 초기화.

```bash
rosservice call /trajectory/reset
```

## Docker 셸과 빌드

역할: 실행 중인 Noetic 컨테이너 셸 진입.

```bash
docker exec -it custom-autorace-noetic bash
```

역할: workspace AMCL `diff-signed` 모델을 포함한 전체 워크스페이스 빌드.

```bash
cd /workspace
catkin_make
source devel/setup.bash
```

역할: 빌드 뒤 apt 패키지가 아니라 workspace AMCL overlay가 선택되는지 확인.

```bash
cd /workspace
source devel/setup.bash
rospack find amcl
```

역할: 통합 launch 실행 중 Gazebo용 AMCL이 짧은 후진 부호 보존 모델을 사용하는지 확인.

```bash
rosparam get /amcl/odom_model_type
```

역할: AMCL odom model·bringup·원본 카메라·description 테스트 실행 및 결과 확인.

```bash
cd /workspace
catkin_make run_tests_amcl_gtest_amcl_odom_model_test \
  run_tests_custom_autorace_bringup run_tests_turtlebot3_autorace_camera \
  run_tests_custom_autorace_description
catkin_test_results build/test_results/amcl
catkin_test_results build/test_results/custom_autorace_bringup
catkin_test_results build/test_results/turtlebot3_autorace_camera
catkin_test_results build/test_results/custom_autorace_description
```

역할: 터널 동적 costmap, 전진 Hybrid A*, 제어 상태·handoff와 통합 launch wiring 회귀 실행.

```bash
cd /workspace
source devel/setup.bash
python3 -m pytest -q \
  src/custom_autorace_bringup/test/test_tunnel_costmap.py \
  src/custom_autorace_bringup/test/test_tunnel_planner.py \
  src/custom_autorace_bringup/test/test_tunnel_controller.py \
  src/custom_autorace_bringup/test/test_gazebo_launch_wiring.py
```

## 종료

역할: 자동 실행한 Docker 시뮬레이션 프로세스 종료.

```bash
cd ~/tb3_autorace_noetic_ws
./docker/stop_sim_terminator.sh
```

역할: Docker Compose 서비스 정지.

```bash
cd ~/tb3_autorace_noetic_ws
docker compose -f compose.noetic.yaml stop autorace
```
