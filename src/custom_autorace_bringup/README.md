# Custom AutoRace hardware bringup

대상 하드웨어는 OpenCR, Intel RealSense D405, Livox Mid-360입니다.

Ubuntu 24.04 호스트에서는 저장소 루트의 `compose.noetic.yaml`로 Ubuntu 20.04/ROS Noetic 컨테이너를 사용합니다. Noetic 기본 RealSense SDK는 D405 지원 이전 버전이므로 Dockerfile이 `librealsense 2.55.1`을 별도로 빌드하고, `src/realsense-ros`의 ROS1 래퍼를 사용합니다.

## 경로 공통화 상태

Intersection, Obstacle, Parking, Zigzag의 실행 경로 자료형·추종기·속도 제한·완료
판정·직사각형 swept-footprint 검사는 [PATH_FOLLOWING.md](PATH_FOLLOWING.md)에 정리한 공통 구현을
사용합니다. 일반 차선도 `/detect/lane_centerline`을 차량 좌표의 rolling
`CommonPath`로 변환하고 검출된 도색선을 `PathSafety`로 실은 뒤 같은
`SweptFootprintValidator`와 `PathFollower`를 사용합니다. Intersection 반원도 이
카메라 rolling 경로를 사용합니다. 고정 미션 경로 공통화 범위는 위 네 미션이며,
차단봉·Tunnel Hybrid A*의 제어식은 제외합니다. Tunnel은 입·출구 portal과 Hybrid A*
결과를 `CommonPath` 자료형으로만 보관하고 자체 추종·충돌 검사를 유지하며, 출구에서는
일반 차선 제어기의 공통 `/control/lane_path_diagnostics`로 안전한 인계를 확인합니다.

현재 주차 경로는 중앙 판정 자세에서 LiDAR로 빈 칸을 선택한 직후 그 자리에서
제자리회전하고, 직선 주차와 즉시 후진 뒤 다시 제자리회전하는 구조입니다. 실제 복귀점에서
지그재그 회전 시작점까지는 하나의 zero-end-curvature quintic으로 연결합니다. 아래의
분기별 회전점 및 원호 기반 Gazebo 결과는 각 당시 구현의 검증 이력이며, 현재 중앙
제자리회전 경로의 완료 근거가 아닙니다. Level Crossing과 Tunnel을 비활성화한 통합
기록도 프로젝트 전체 완주 검증으로 보지 않습니다. Intersection의 검증 범위는 이 문서의
교차로 절과 [PATH_FOLLOWING.md](PATH_FOLLOWING.md)에서 변경별로 구분해 기록합니다.

### 2026-09-13 현재 통합 검증

새 Gazebo를 공식 시작 자세에서 실행한 `0.28 m/s` adaptive-lookahead `run23`은 카메라 신호 출발 뒤
`Intersection → Obstacle → Parking → Zigzag → Level Crossing → Tunnel`을 순서대로
완료하고 결승선의 비대칭 직사각형 footprint까지 통과했습니다. 최종 `result.json`의
통합 판정 항목은 `22/22` 모두 참이고,
출발부터 결승선까지는 `280.472 s`였습니다. 실제 카메라가 LEFT 교차로를 선택했고
LiDAR가 LEFT 주차면을 선택했습니다. 주행 구간의
`/detect/lane_centerline`은 `8,322`개, 평균 `29.672 Hz`, 최대 간격 `57 ms`였고,
각 제어기의 `/cmd_vel` 소유권은 중복 없이 순서대로 인계됐습니다. 전체 자동 회귀
중 `custom_autorace_bringup`의 `504`개와 전체 workspace의 `557`개도
오류·실패·건너뜀 없이 통과했습니다. 원시 bag과 판정 JSON,
미션별 오차·여유는 [PATH_FOLLOWING.md](PATH_FOLLOWING.md)에 기록합니다.

이 결과는 Gazebo 공식 시작점 통합 검증입니다. 실제 장착 D405의 원근 보정,
OpenCR 응답과 실제 도색에서의 30 Hz 주행은 별도 실차 검증이 남아 있습니다.

## 실행 전 사용자 입력

1. `config/MID360_config.json`에서 호스트 Ethernet 주소(기본 `192.168.1.5`)와 실제 Mid-360 주소(기본 예시 `192.168.1.12`)를 수정합니다.
2. `custom_autorace_description/urdf/robot_parameters.xacro`에서 LiDAR와 카메라의 실제 렌즈/스캔 중심 장착 좌표를 입력합니다.
3. 카메라가 여러 대라면 `camera_serial_no` launch 인자로 D405 시리얼 번호를 지정합니다.
4. Mid-360의 장착 높이와 장애물 높이에 맞춰 `livox_mid360.launch`의 `min_height`, `max_height`를 트랙에서 조정합니다.

## 토픽 변환

- Mid-360 `/livox/lidar` (`PointCloud2`) → `/scan_mid360_raw` (`LaserScan`)
- D405 `/camera/color/image_raw` (1280x720) → 중앙 4:3 crop + 320x240 →
  `/camera/image_rect_color` (`Image`, 차선 경로)와
  `/camera/image_rect_color/compressed` (`CompressedImage`, 신호등·보정 도구)

카메라 어댑터는 한 번 보정·축소한 같은 프레임을 raw와 compressed로 발행합니다.
차선의 `image_projection`과 `image_compensation`은 raw 연결을 사용해 JPEG
왕복 없이 처리하고, 기존 보정 도구와 신호등 검출은 compressed 연결을 사용합니다.

## 실행

```bash
roslaunch custom_autorace_bringup hardware.launch
```

`hardware.launch`는 OpenCR, D405, Mid-360, `robot_state_publisher`와 하나의 카메라
검출 파이프라인을 실행합니다. D405 30 Hz 원본에서 한 어댑터가 만든 영상으로
투영·보상·차선 검출을 순서대로 처리하고, 같은 카메라에서 신호등과 표지판 검출도
매 프레임 수행합니다. 이 실물 launch에는 차선 제어기나 `/cmd_vel` 발행자가 없으므로
보정 전 검출 시험이 차량을 움직이지 않습니다. AMCL, 미션 구역 매니저와 미션 제어기는
실물용 지도·센서·경로 YAML이 준비되기 전까지 포함하지 않습니다.

기본 `projection_config`는
`config/d405_projection_uncalibrated.yaml`이며 실행 배선을 확인하기 위한 미보정
템플릿입니다. Gazebo 투영값을 재사용하지 않습니다. 장착된 D405에서 측정한 파일을
다음처럼 명시해야 차선 형상이나 먼 시야를 주행 판단에 사용할 수 있습니다.

```bash
roslaunch custom_autorace_bringup hardware.launch \
  projection_config:=/absolute/path/to/measured_d405_projection.yaml
```

Gazebo에서는 센서 어댑터까지 포함한 bringup launch를 사용합니다.

```bash
roslaunch custom_autorace_bringup gazebo.launch
```

기본 실행은 `map_server`와 AMCL도 함께 실행합니다. AMCL은
`/scan_mid360_raw`와 Gazebo diff-drive의 raw `odom → base_footprint` TF를 이용해 `map → odom`을
보정합니다. 미션 구역 매니저는 TF의 `map → base_footprint`를 0.10초마다 직접
조회하고, 같은 자세 표본으로 `/mission/map_pose`와 모든 polygon 신호를 계산합니다.
통합 Gazebo launch는 diff-drive encoder 모드가 후진 속도의 부호를 잃는 문제를
피하기 위해 `odometry_source:=world fuse_imu:=true`를 기본으로 사용합니다. signed
world pose/twist를 EKF 입력으로 쓰고 EKF가 `/odometry/filtered`와 유일한
`odom → base_footprint` TF를 발행합니다. `fuse_imu:=false` 진단에서는 미션 제어기의
입력도 자동으로 `/odom`으로 바뀝니다. `odometry_source:=encoder fuse_imu:=true`는
전진 진단용이고, 실물 OpenCR odometry에는 이 설정을 사용하지 않습니다.

장애물·차단봉·터널 제어기는 `angle_min`과 `angle_increment`가 그대로인
`/scan_mid360_raw`를 직접 사용합니다. scan 배열을 정면 인덱스 기준으로
재정렬하지 않습니다.

이 명령은 AutoRace 카메라 투영·차선 검출, 안전 차선 주행 제어기,
교차로·장애물·주차·지그재그·차단봉·터널 미션 제어기, 원본 미션 환경과
`/workspace/autorace_sim.rviz` 설정의 RViz를 함께 실행합니다. 원본 미션 환경은
출발 신호등을 빨강→노랑→초록으로 전환하고, 삼거리의 좌·우 방향 표지판 중 하나를
무작위로 선택하며 차단기와 주차 장애물을 배치합니다. 별도의 차선 제어 launch나
`rqt_image_view` 터미널은 필요하지 않습니다. 차량은 빨강과 노랑 동안 정지하고,
카메라 신호등 검출기가 초록을 연속 검출하여 `/detect/traffic_light`에 `2`를 발행한
뒤에만 차선 주행을 시작합니다. 이 신호등 대기·출발은 순차 미션의 전처리이며, 현재
자동 제어가 연결된 순서는
`intersection → obstacle → parking → zigzag → level_crossing → tunnel`입니다.
물리 코스의 순서도 `신호등 → 교차로 → 장애물 → 주차 → 지그재그 → 차단봉 →
터널`입니다. RViz에는
`/camera/color/image_raw` 원본 영상, `/detect/image_lane/compressed` 차선 검출 영상,
`/detect/image_traffic_light/compressed` 신호등 검출 영상과 청록색
`/obstacle/local_path`에 한 번 확정한 C2 clamped-cubic slalom과 접속부의
heading·curvature가 연속(G2)인 quintic 출구 경로가 표시됩니다. 지그재그의
lookahead 기준 경로는 `/zigzag/path`, 터널의 실시간 점유 격자와 Hybrid A* 경로는
각각 `/tunnel/costmap`, `/tunnel/path`로 표시됩니다.

차량만 정지하거나 주행을 재개하려면 다른 컨테이너 터미널에서 다음 서비스를
호출합니다. 정지 상태에서도 Gazebo, 센서, 검출 및 RViz는 계속 실행됩니다.

```bash
# 즉시 정지
rosservice call /control/lane_following false

# 수동 정지 해제(현재 제어권 소유자가 재개)
rosservice call /control/lane_following true
```

수동 정지는 `/control/manual_stop`으로 미션 제어기에도 전파됩니다. 미션이
`/cmd_vel`을 소유하는 중에 정지하면 0 명령을 보내고 상태 전이와 센서 판정을
멈춥니다. 재개할 때 기존 미션 제어권과 상태를 유지하고 최신 센서 입력으로 계속합니다.

유효한 `/detect/lane_centerline`이 0.5초 동안 들어오지 않으면 마지막 조향으로
추측 주행하지 않고 0 속도를 발행합니다. 다음 정상 카메라 표본이 오면 즉시 일반 차선
주행을 계속합니다. 처음부터 차량을 움직이지 않고 센서와 RViz만 확인하려면
`drive_lane:=false`를 사용합니다.

정상 입력에서는 30 Hz로 들어온 각 현재 프레임마다 가까운·중간·먼 중심 표본과 곡률을
다시 계산해 짧은 rolling `CommonPath`를 교체합니다. 이전 프레임의 경로 형상을 새
관측처럼 재사용하지 않으며, 직선은 YAML의 cruise 속도까지 가속하고 앞쪽 곡률이 커지면
보이는 경로 전체에서 만든 공통 속도 profile이 현재 위치까지 감속을 미리 전파합니다.

Gazebo 투영 사다리꼴의 위쪽은 기존 입력 영상 `y=176`에서 `y=110`까지 확장했고,
bird-eye 영상의 `0.60 m` 전방까지 8개 행에서 가까운·중간·먼 차선을 같은 프레임으로
표본화합니다. 이 수치는 Gazebo 카메라에 대해 검증한 값입니다. 실제 D405는 렌즈와 장착
자세가 다르므로 이 사다리꼴을 복사하지 않고, 제공된 미보정 템플릿을 장착 상태에서 다시
측정해야 합니다.

차선 영상은 `turtlebot3_autorace_detect/nodes/detect_lane` 한 노드에서 한 번만
HSV 분리합니다. 같은 프레임에서 가까운 곳부터 먼 곳까지의 표본 행,
중심·노란선·흰선 x, 유효 여부와 신뢰도를 `/detect/lane_centerline`으로
발행합니다. `/detect/lane`과 `/detect/lane_boundaries`도 같은 마스크에서 발행하므로
일반 차선과 교차로가 서로 다른 선을 중복 판단하지 않습니다. 검출 면적·경계 표본 행·
연결 branch 조건은 `turtlebot3_autorace_detect/param/lane/lane.yaml`에서 조정합니다.
주행 모드에서는 제어에 쓰지 않는 1000×600 오버레이/JPEG 디버그 영상을 기본
비활성화해 구독자나 기록기가 30 Hz 중심선 계산을 막지 않게 했습니다. 카메라 보정
모드에서는 이 디버그 영상을 기존처럼 매 프레임 발행합니다.

`safe_lane_controller.py`는 카메라 촬영 시각에 맞춘 odom 자세에서 표본을 미터
단위로 변환하고, 차량 위치와 관측 구간을 연결한 rolling `CommonPath`를 odom에
고정합니다. 선택한 같은 프레임의 노란색·흰색 경계는 도색 폭과 점 사이 간격을 포함한
연속 경계로 경로에 함께 고정합니다. 공통 검증기는 위치추정·추종 오차를 더한 실제
비대칭 직사각형을 경로점 사이와 현재 반응시간·완전 정지 영역까지 검사합니다. 예상
침범이 없으면 명목 속도를 유지하고, 예상될 때만 공통 안전 속도로 감속하거나 정지합니다.
곡률 feed-forward와 방향·횡오차 feedback은 공통 `PathFollower`가 계산합니다.
`/control/lane_path_diagnostics`도 다른 경로 제어기와 같은 13개 진행률·오차·최소
여유·명령 배열입니다. 일반 차선 제어권이 활성인 동안은 이 제어기만 `/cmd_vel`을
발행합니다.

주행 설정은 `config/lane_controller.yaml`의 `lane_path/control`과
`lane_path/safety`에 있습니다. 현재
Gazebo profile은 직선 `cruise_velocity=0.28 m/s`, 최종 상한
`maximum_velocity=0.30 m/s`이며, 공통 추종기가 곡률·최대 횡가속도·최대 각속도와
선형·각가속도 한계로 코너 전에 자동 감속합니다. YAML 저장 후
`gazebo.launch`를 다시 실행하면 적용되며 다시 빌드할 필요는 없습니다.

조향 목표거리는 현재 측정 선속도 `v`에 따라
`clamp(0.065 + 0.35·|v|, 0.08, 0.16) m`로 증가합니다. 따라서 정지 부근은
`0.08 m`, `0.20 m/s`에서는 `0.135 m`, `0.26 m/s`에서는 `0.156 m`,
`0.28 m/s`에서는 상한이 적용된 `0.160 m` 앞을 봅니다. 차량 바로 아래의 카메라 사각
구간은 현재 프레임에서 처음 보이는 중심점의
위치와 접선까지 C1 Hermite로 연결하므로, 목표점이 연결부에 있어도 미래 차선 형상을
반영합니다. 곡률 속도 profile은 보이는 경로 전체에서 뒤로 감속을 전파하고 현재 진행점에서
읽기 때문에, 조향점을 멀리 옮기는 것과 코너 선제 감속을 서로 중복시키지 않습니다.

### 일반 차선 rolling CommonPath 회귀

전용 launch는 출발선부터 교차로 진입 전 gate까지 카메라 rolling
`CommonPath`를 추종합니다. `safe_lane_controller` 하나만 `/cmd_vel`을 발행하고,
별도 제어 경로는 만들지 않은 채 기록기가 실제 Gazebo 자세의 소요시간·도색
여유·추종 오차를 JSON으로 저장합니다.

```bash
roslaunch custom_autorace_bringup gazebo_lane_path_test.launch \
  gui:=false rviz:=false result_file:=/tmp/lane_path_result.json
```

Gazebo에서 `0.28 m/s` profile의 반복 안전 회귀와 공식 시작점 통합 완주를
통과했습니다. 전용 회귀의 속도 차이는 동일한 출발선→교차로 gate 조건끼리만
비교하며, 분기가 달랐던 전체 완주 시간끼리는 속도 효과로 비교하지 않습니다. 실물 D405
투영 보정값·OpenCR 응답·실제 도색에서의 30 Hz 주행도 아직 검증하지 않았습니다.

현재 C1 카메라 경로 생성기를 고정하고 매번 새 Gazebo에서 측정한 결과는 다음과
같습니다. 침범과 여유는 도색 바깥이 아니라 허용 주행면을 기준으로 한 signed 값입니다.

| 조향 목표/직선 속도 | gate 시간 | 최대 안쪽 침범 | 최소 바깥 여유 | 경로 오차 p95 | 판정 |
|---|---:|---:|---:|---:|---|
| 고정 `0.065 m` / `0.26 m/s` | `12.338 s` | `9.538 mm` | `12.778 mm` | `16.468 mm` | 성공·안전 |
| adaptive `0.08–0.16 m` / `0.20 m/s` | `12.522 s` | `0 mm` | `27.676 mm` | `16.271 mm` | 성공·안전 |
| adaptive `0.08–0.16 m` / `0.22 m/s` | `12.412 s` | `0 mm` | `28.736 mm` | `15.500 mm` | 성공·안전 |
| adaptive `0.08–0.16 m` / `0.24 m/s` | `11.760 s` | `0 mm` | `29.706 mm` | `16.869 mm` | 성공·안전 |
| adaptive `0.08–0.16 m` / `0.26 m/s`, 2회 | `11.475–12.002 s` | `0 mm` | `28.080–28.322 mm` | `13.998–18.283 mm` | `2/2` 성공·안전 |
| adaptive `0.08–0.16 m` / `0.26 m/s`, 최신 교차 2회 | `11.654–11.909 s` | `0 mm` | `28.552–28.650 mm` | `14.598–16.635 mm` | `2/2` 성공·안전 |
| adaptive `0.08–0.16 m` / `0.28 m/s`, 4회 | `11.607–11.807 s` | `0 mm` | `28.996–29.673 mm` | `14.094–16.175 mm` | `4/4` 성공·안전 |
| adaptive `0.08–0.16 m` / `0.30 m/s`, 1회 | `11.776 s` | `0 mm` | `28.970 mm` | `14.289 mm` | 통과했으나 미채택 |

같은 `0.26 m/s`에서 첫 직접 비교는 `12.338 → 12.002 s`, `0.336 s(2.7%)`
단축됐습니다. adaptive 속도 단계의 `0.20 → 0.26 m/s` 비교는
`12.522 → 11.475 s`, `1.047 s(8.4%)` 단축됐습니다. 다만 경로 접합 방식까지 달랐던
이전 close-target 기록은 `10.836 s`로 현재 adaptive 최선보다 `0.639 s` 빨랐으므로,
현재 구현이 모든 과거 구현보다 빠르다고 해석하지 않습니다. 최대 목표를 `0.18 m`로
늘린 후보도 `12.218 s`로 느려서 채택하지 않았습니다.

같은 날 순서를 섞어 다시 측정한 `0.26 m/s` 2회 평균은 `11.781 s`, `0.28 m/s`
4회 평균은 `11.699 s`로 `0.082 s(0.70%)` 짧았습니다. 두 설정 모두 침범 없이
전 회차를 통과했습니다. `0.30 m/s`는 `11.776 s`로 시간 이득이 없었고 순간 최대
횡가속도가 `0.1599 m/s²`까지 증가해 채택하지 않았습니다. 따라서 기본 cruise만
`0.28 m/s`로 올리고 최대 상한, 곡률 감속, `0.160 m` lookahead 상한은 유지합니다.
원시 JSON은
[`diagnostics/lane_speed_adaptive_20260912`](../../diagnostics/lane_speed_adaptive_20260912)와
[`diagnostics/lane_speed_adaptive_20260913`](../../diagnostics/lane_speed_adaptive_20260913)에
보존했습니다.

#### 2026-09-07 이전 구현의 역사적 A/B 결과

아래 결과는 현재 rolling `CommonPath` 전환 전, 당시의 카메라 PD와 고정 측량
경로 추종 구현을 비교한 기록입니다. 현재 제어기의 검증 근거로 사용하지 않습니다.
`0.450 m`로 줄인 시험 PNG를 Gazebo 바닥에 스테이징한 당시 1회 시험에서는
두 방식 모두 안전 기준을 통과했고, 고정 측량 경로의 gate 시간은 `8.979 s`,
카메라 PD는 `11.207 s`로 전자가 `19.9%` 짧았습니다.

2026-09-07에 매번 Gazebo를 새로 시작해 각 방식 11회씩 시험했습니다. 구성은 명목 자세
5회, `y ±10 mm`와 `yaw ±2°`의 결합 오차 4회, 허용 시작영역 가까이의 위치·방위 오차
2회입니다. 성공 기준은 gate 완주와 도색 안쪽 침범 `16 mm` 이하, 바깥 도색 경계 여유
`5 mm` 이상이었습니다.

| 당시 제어 방식 | 성공 | gate 시간 중앙값 | 전체 범위 | 최대 안쪽 침범 | 최소 바깥 여유 | 경로 오차 p95 중앙값 |
|---|---:|---:|---:|---:|---:|---:|
| 카메라 PD, 0.20 m/s | 11/11 | 11.863 s | 11.836–11.954 s | 15.44 mm | 7.29 mm | 35.84 mm |
| 카메라 PD, 0.22 m/s | 11/11 | 11.396 s | 11.377–11.609 s | 15.42 mm | 7.33 mm | 35.93 mm |
| 고정 측량 경로 lookahead, 0.22 m/s | 11/11 | 9.781 s | 9.666–9.971 s | 1.44 mm | 22.56 mm | 2.69 mm |

33회 핵심 원시 지표는 `test/lane_lookahead_ab_20260907.csv`에 보존했습니다.
당시 고정 측량 경로의 중앙값은 `0.20 m/s` PD보다 `17.6%`, 동일 최고속도
PD보다 `14.2%` 짧았고, 속도만 `0.22 m/s`로 높인 PD 효과는 `3.9%`였습니다.
고정 측량 경로의 gate 사이 실제 이동거리 중앙값은 `1.809 m`, 두 PD는 약
`1.871 m`였습니다. 이 수치들은 당시 고정 구간에서의 비교만 나타냅니다.

RViz, 차선 검출 또는 주행 제어를 끄거나 다른 RViz 설정을 지정하려면 다음 인자를
사용합니다.

```bash
roslaunch custom_autorace_bringup gazebo.launch rviz:=false
roslaunch custom_autorace_bringup gazebo.launch detect_lane:=false
roslaunch custom_autorace_bringup gazebo.launch detect_signs:=false
roslaunch custom_autorace_bringup gazebo.launch drive_lane:=false
roslaunch custom_autorace_bringup gazebo.launch mission_models:=false
roslaunch custom_autorace_bringup gazebo.launch wait_for_green:=false
roslaunch custom_autorace_bringup gazebo.launch intersection_mission:=false
roslaunch custom_autorace_bringup gazebo.launch obstacle_mission:=false
roslaunch custom_autorace_bringup gazebo.launch parking_mission:=false
roslaunch custom_autorace_bringup gazebo.launch zigzag_mission:=false
roslaunch custom_autorace_bringup gazebo.launch level_crossing_mission:=false
roslaunch custom_autorace_bringup gazebo.launch tunnel_mission:=false
roslaunch custom_autorace_bringup gazebo.launch odometry_source:=encoder fuse_imu:=true
roslaunch custom_autorace_bringup gazebo.launch mission_zones:=false localization:=false intersection_mission:=false obstacle_mission:=false parking_mission:=false zigzag_mission:=false level_crossing_mission:=false tunnel_mission:=false
roslaunch custom_autorace_bringup gazebo.launch x_pos:=0.8 y_pos:=-1.747 yaw_pos:=0.0
roslaunch custom_autorace_bringup gazebo.launch mission_zone_config:=/workspace/src/custom_autorace_bringup/config/다른구역.yaml
roslaunch custom_autorace_bringup gazebo.launch map_file:=/workspace/maps/갱신지도.yaml
roslaunch custom_autorace_bringup gazebo.launch rviz_config:=/workspace/다른설정.rviz
```

교차로·장애물·주차·지그재그·차단봉·터널 제어기는 AMCL polygon 순차 gate를 진입 조건으로
사용합니다.
장애물 제어기는 polygon의 최신 `inside`와 `clearance` 값도 복귀와 완료 조건으로
사용합니다. `mission_zones:=false`이면 제어기는 각각 진입 대기 상태에 머무르므로
localization 없이 센서만 시험할 때는 여섯 미션 제어기도 함께
끕니다. 기본 `intersection → obstacle → parking → zigzag → level_crossing → tunnel`
sequence의 각 단계는
설정된 외부 `COMPLETE`가 필요합니다.

## AMCL 미션 구역과 순차 상태 머신

Gazebo의 AMCL 지도는 `maps/autorace_gazebo.yaml`이며 라이다가 실제로 반사하는
코스 고정 구조만 포함합니다. AMCL은 이 사전 작성 지도와 scan을 맞춰
`map → odom`을 추정하는 위치추정기이며, 주행 중 장애물을 지도에 추가하는 SLAM이
아닙니다. 터널 planning 범위의 static layer에는 입구·출구와 네 벽만 남겨 있고,
Gazebo world의 원통 장애물 세 개는 위치가 알려지지 않은 경기 조건을 만들기 위해
의도적으로 PGM에서 빼었습니다. 바닥 차선은 카메라에는 보이지만 라이다에는 보이지
않으므로 점유지도에 넣지 않았습니다. 지도 원점과 해상도는 각각
`[-2.5, -2.5]`, `0.02 m/cell`입니다. 지도에 포함되는 Gazebo 구조물을 바꾸면 다음
명령으로 PGM을 다시 생성할 수 있습니다.

```bash
python3 src/custom_autorace_bringup/tools/generate_autorace_gazebo_map.py
```

AMCL 파라미터는 `config/amcl_gazebo.yaml`, 미션 순서와 polygon은
`config/mission_zones_gazebo.yaml`에서 조정합니다. 이 workspace는 ROS Navigation
AMCL 1.17.3을 `src/navigation/amcl`에 overlay하고, 기존 `diff` 모델은 그대로 둔 채
`diff-signed` 모델을 추가합니다. 이 모델은 1 cm 미만 이동에서 `atan2`를 생략하는
기존 제자리회전 보호 동작을 유지하면서 이전 odom 방위에 대한 종방향 투영으로 후진을
판별합니다. 따라서 주차 `BACK_OUT`의 짧은 후진 odom 갱신도 전진으로 반사되지 않습니다.
`config/amcl_gazebo.yaml`이 이 모델을 선택하며, 소스를 받은 뒤 `catkin_make`해야
workspace AMCL 실행 파일이 사용됩니다. 현재 순서는 다음과 같습니다.

```text
intersection → obstacle → parking → zigzag → level_crossing → tunnel
```

아직 차례가 오지 않은 구역에 AMCL 위치가 들어가도 해당 미션은 활성화되지 않습니다.
예상 구역 안쪽으로 `enter_margin` 이상 들어오면 ACTIVE가 됩니다. 모든 미션의 완료는
각 컨트롤러가 설정된 외부 토픽에 `COMPLETE`를 발행할 때만 처리합니다. 현재 설정은
`intersection`이 `/intersection/state`, `obstacle`이 `/obstacle/state`, `parking`이
`/parking/state`, `zigzag`가 `/zigzag/state`, `level_crossing`이
`/level_crossing/state`, `tunnel`이 `/tunnel/state`를 사용합니다. 각 전용 제어기가 실제
동작과 최종 자세를 확인한 뒤 `COMPLETE`를 발행하므로 순서 매니저가 위치만으로 미션을
건너뛰지 않습니다. 신호등은 이 sequence에 들어가기 전 출발 허가를 담당합니다.

매니저는 각 미션에 대해 설정된 `inside_topic`과 `clearance_topic`을 0.10초
주기로 발행합니다. `inside`는 로봇 기준점이 polygon 내부인지, `clear`는 로봇
기준점이 설정된 footprint 여유까지 포함해 밖으로 나갔는지를 나타냅니다. 현재
Gazebo 설정의 보조 region은 교차로 방향지시판 조기 관측에만 사용하며, 반원 끝은
최신 AMCL 지도 자세로 직접 판정합니다.

교차로 방향지시판 관측 region과 순차 mission gate polygon은 분리되어 있습니다.
관측 region의 동쪽 경계는 `x=1.72 m`이고 자체 `inside_margin=0.03 m`가 적용되어,
서쪽으로 주행할 때 `x=1.69 m`부터 방향 검출을 받을 수 있습니다. 이 좁은 AMCL
구역 안에서는 영상 면적 3% 이상의 표지판을 최대 1.20초 간격으로 30 Hz의 9회 같은 방향인지
확인합니다. 방향을 미리 저장해도 상태는 `WAIT_INTERSECTION`이고 일반 차선 추종을
유지합니다.
이후 별도 mission gate가 `x=1.62 m`에서 열리면 저장한 방향을 사용하며, 미확정이면
그때부터 제한시간이 있는 `SEARCH_DIRECTION`을 시작합니다. 따라서 조기 관측 시간이
35초 탐색 제한이나 5초 진입 제한을 소모하지 않습니다. 실제 제어권은 계속 측량
진입점 `x=1.395 m`의 법선 평면 0.200 m 전까지 유지합니다. 진입면에서 제어권을 받은
뒤 교차로 제어기는 0 속도를 유지하고, 서비스
호출 뒤 새 `/odometry/filtered` 표본을 한 번 확인합니다. 그 최신 EKF 자세에서 선택한
반원 입구까지 직접 cubic을 생성해 공통 추종기로 실행합니다. 반원은 카메라 차선 추종을
유지하며, 반원 끝에서는 최신
`/mission/map_pose`와 선택 방향 5차 Bezier prefix 시작점의 직접 거리로 제어권 회수
시점을 판정합니다. 회수 뒤 최신 AMCL `map → odom`을 다시 고정해 방향별 prefix와 공통
5차 Bezier 탈출 경로를 공통 추종기로 실행합니다.

주요 상태와 gate는 다음 토픽에서 확인합니다.

```bash
rostopic echo /amcl_pose
rostopic echo /mission/map_pose
rostopic echo /mission/state
rostopic echo /mission/current
rostopic echo /mission/sequence_index
rostopic echo /mission/detected_zone
rostopic echo /mission/enable/intersection
rostopic echo /mission/inside/intersection
rostopic echo /mission/inside/intersection_direction_observation
rostopic echo /mission/clear/intersection_observation  # mission polygon 진단 전용
rostopic echo /mission/enable/obstacle
rostopic echo /mission/inside/obstacle
rostopic echo /mission/clear/obstacle
rostopic echo /mission/enable/parking
rostopic echo /mission/inside/parking
rostopic echo /mission/enable/zigzag
rostopic echo /mission/inside/zigzag
rostopic echo /mission/clear/zigzag
rostopic echo /mission/enable/level_crossing
rostopic echo /mission/inside/level_crossing
rostopic echo /mission/clear/level_crossing
rostopic echo /mission/enable/tunnel
rostopic echo /mission/inside/tunnel
rostopic echo /mission/clear/tunnel
rostopic echo /intersection/diagnostics
rostopic echo /obstacle/state
rostopic echo /obstacle/diagnostics
rostopic echo /parking/state
rostopic echo /parking/diagnostics
rostopic echo /zigzag/state
rostopic echo -n 1 /zigzag/path
rostopic echo /zigzag/diagnostics
rostopic echo /level_crossing/state
rostopic echo /level_crossing/barrier_down
rostopic echo /level_crossing/diagnostics
rostopic echo /tunnel/state
rostopic echo -n 1 /tunnel/path
rostopic echo -n 1 /tunnel/costmap
rostopic echo /tunnel/diagnostics
```

`/mission/state`는 `WAITING_FOR_POSE`, `SEEKING`, `ACTIVE`, `COMPLETE` 중 하나입니다.
RViz의 `Mission Zones`에는 완료 구역을 회색, 현재 예상 구역을 주황색 또는 활성
초록색으로 표시합니다. `/mission/detected_zone`에는 현재 자세가 들어간 mission
polygon이 표시됩니다. 새 순차 시험은 bringup을 다시 실행해 처음부터 시작합니다.

실물에서는 Gazebo PGM과 polygon 좌표를 사용하지 않습니다. 먼저 Mid-360으로 실제
경기장의 고정 구조만 담은 정적 점유지도를 별도로 작성하고 `amcl_localization.launch`의 `map_file`,
`initial_pose_*`, `amcl_config` 인자를 실물용 파일로 교체한 뒤, 같은 지도 좌표계에서
미션 polygon, 교차로 고정 경로와 도색 경계, 터널 입구·출구를 각각 측량해야 합니다.
페인트 라인이나 위치가 바뀌는 장애물은 AMCL 지도에 넣지 않습니다. 터널 장애물은
주행 중 Mid-360 동적 costmap으로만 추적합니다.

### 교차로 미션

교차로는 다음 7단계로 동작합니다.

1. AMCL 관찰 polygon 안에서 카메라 방향 표지판으로 LEFT 또는 RIGHT를 선택합니다.
2. 확정한 방향을 유지한 채 일반 차선 추종으로 계속 접근합니다. 최신 AMCL map 자세가
   측량된 `map_entry_start` 법선 평면의 0.200 m 전을 통과했을 때만 교차로 제어기가
   `/cmd_vel`을 인수합니다.
3. 인수 직후 교차로 제어기가 zero 명령을 유지하고, 서비스 호출 뒤에 도착한 새
   `/odometry/filtered` 표본을 한 번 확인합니다. 그 최신 자세에서 AMCL
   `map → odom`을 고정하고 선택된 반원 입구까지 하나의 cubic을 직접 만듭니다.
   양 끝 접선 길이는 실제 시작점과 목표점 사이 거리의 `0.50`, `0.25`배입니다.
   선택 방향 반대쪽 누적 회전이 `2°`를 넘거나 전체 회전량이 `120°`를 넘으면
   움직이기 전에 실패시키고, 공통 sweep 검사를 통과한 경로만 실행합니다.
4. 진입 경로의 공통 완료 판정 직후 일반 차선 제어기에 `/cmd_vel`을 바로 넘깁니다.
   반원은 카메라가 매 프레임 만든 rolling `CommonPath`와 공통 `PathFollower`로
   주행하며, 교차로 제어기는 이 구간에 별도 조향 명령을 만들지 않습니다.
5. 최신 `/mission/map_pose`와 선택 방향 5차 Bezier prefix의 첫 점 사이 직접 거리가
   `exit_takeover_max_distance` 안에 들면 교차로 제어기가
   `/control/lane_mission_handoff`로 제어권을 회수합니다. 별도 arc-end polygon이나
   누적 region 표본은 사용하지 않습니다. `PREPARE_EXIT_PATH`에서 0.10초 정지하고
   인계 뒤 새 `/odometry/filtered` 표본을 기다립니다.
6. 이 시점의 최신 AMCL `map → odom`을 다시 고정해 정지 후 실제 자세에서 선택된
   방향별 5차 Bezier branch의 40% 지점까지 C1 connector를 만듭니다. connector,
   선택 branch suffix와 `(0.600, -0.750)`에서 시작하는 공통 5차 Bezier를 하나의
   `CommonPath`로 결합합니다. 전체 raster cell 경계와 비대칭 footprint sweep를
   검증한 뒤 같은 공통 추종기로 출구 `(0.250, -0.300, 90°)`까지 주행합니다.
7. 출구 경로의 공통 완료 판정 직후 일반 차선 제어권을 먼저 돌려주고, 이어지는
   `VERIFY_FINAL_LANE`에서는 교차로 제어기가 `/cmd_vel`을 발행하지 않은 채 동일 HSV
   mask의 노란선·흰선 경계만 관찰합니다. 폭·중심·방위 조건이 30 Hz의 서로 다른 영상
   9프레임에서 확인되면 `COMPLETE`를 발행합니다. 같은 mask에서 파생된 `/detect/lane`
   중심을 다시 확인하지 않습니다.

관찰 polygon은 방향 선택을 시작하는 gate일 뿐 반원 끝 전환이나 완전 탈출 조건이
아닙니다. 반원 끝 제어권 회수는 선택된 고정 prefix 시작점까지의 map 거리로, 완전
탈출은 실제 출구의 두 차선 사이에 들어간 것을 카메라로 확인한 뒤에만 성립합니다.
전체 제어권 순서는 다음과 같습니다.

```text
일반 차선 제어기(방향 확인·AMCL 진입면 접근)
→ 교차로 제어기(진입 경로)
→ 일반 차선 제어기(선택 반원)
→ 교차로 제어기(별도 탈출 경로)
→ 일반 차선 제어기(다음 코스)
```

진입은 제어권 인계 뒤 0 속도를 유지하면서 새로 받은 EKF 자세와 YAML의 방향별 반원
입구를 직접 잇는 cubic입니다. 이때 최신 `map → odom`을 한 번 고정하며 실행 중 AMCL
갱신으로 경로를 움직이지 않습니다. 탈출은 YAML에 측량된 방향별 branch와 공통 map
곡선이며, 탈출 branch의 인계 불연속만 명시적 C1
connector로 잇고 선택된 측량 suffix와 고정 목표를 보존합니다. Ground Truth, A* 경로
생성과 실행 중 재정렬은 사용하지 않습니다. 진입과 탈출 모두 공통
`RasterCellBoundary`, 비대칭 footprint, `PathFollower`를 사용합니다.

주요 상태 순서는 다음과 같습니다. 경로 끝의 위치·방위·종단 통과는 공통
`GoalTolerance`가 함께 판정하므로 별도 회전 정렬 상태를 두지 않습니다.

```text
WAIT_INTERSECTION → (조기 미확정일 때만 SEARCH_DIRECTION) → WAIT_ENTRY_HANDOFF
→ PREPARE_ENTRY_PATH → FOLLOW_ENTRY_PATH → FOLLOW_ARC_LANE
→ PREPARE_EXIT_PATH → FOLLOW_EXIT_PATH → VERIFY_FINAL_LANE → COMPLETE
```

방향 표지판은 일반 차선 제어가 소유권을 가진 상태에서 최대 1.20초 간격의 동일 방향
관측 9프레임으로 확정합니다. 표지판을 화면 중앙에 맞추기 위한 별도 회전은 하지 않으며,
방향 확정 뒤 첫 미션
움직임도 AMCL 진입면까지는 같은 차선 제어가 이어갑니다. 진입면에서 제어권을 한 번
인계하면 짧은 zero barrier와 인계 뒤 새 EKF 자세 확인을 거친 뒤 경로를 만듭니다.
고정 진입·탈출 경로와 카메라 rolling 반원·출구 경로 모두 공통 추종기를 사용하며,
교차로 제어기는 두 카메라 주행 구간에서 `/cmd_vel`을 발행하지 않습니다.

다음 수치는 현재 직접 cubic 적용 전의 비교 기준이며 최신 코드의 검증 결과가 아닙니다.
최종 adaptive 인계 경로를 넣기 전 Intersection-only Gazebo 기준은
RIGHT가 총 `31.140 s`(진입 `5.852 s`, 탈출 `11.364 s`, 최종 위치·방향
오차 `0.020 m`/`0.0°`), LEFT가 총 `28.738 s`(진입 `4.004 s`, 탈출
`11.715 s`, 최종 위치·방향 오차 `0.024 m`/`0.1°`)였다. 양쪽 모두
`COMPLETE`하고 일반 차선 제어권을 반환했다. LEFT의 최소 raw line 여유는
`5.441 mm`였다. RIGHT 탈출은 카메라 반원이 넘겨준 시작 자세의 raw overlap
`4.831 mm`를 명시적 `line_overlap_allowance` 수렴 구간에서만 허용한 후
0으로 수렴했습니다. 당시 공통화 코드의 공식 시작점 통합 주행에서는 실제 카메라가
RIGHT를 3프레임으로 확정했고 `137.570→177.749 s`, 총 `40.179 s`에
`COMPLETE`했습니다.

방향 확인과 AMCL 진입면 인계를 분리했지만 명목 경로의 20% 지점으로 먼저 연결하던
이전 코드는 Gazebo와 관련 노드를 매회 새로 시작하고 공식 시작 자세에서
`forced_direction=0`으로 4회 연속 검증했다. 실제 카메라 선택은
`RIGHT, LEFT, LEFT, RIGHT`였고 네 번 모두 `COMPLETE`했다. 총 교차로 시간은 각각
`38.977, 38.256, 34.247, 37.747 s`, 진입 경로 시간은
`13.984, 12.946, 9.347, 13.143 s`, 탈출 경로 시간은
`11.855, 12.155, 11.715, 11.504 s`였다. 이 기록은 현재의 actual-pose direct
cubic과 인계 뒤 새 EKF 표본 barrier에 대한 검증 결과로 사용하지 않습니다.

현재 actual-pose 직접 cubic과 `PREPARE_ENTRY_PATH`는 Gazebo와 관련 노드를 방향마다
새로 시작하고 공식 시작 자세에서 `forced_direction=0`으로 LEFT와 RIGHT를 각각 1회
검증했습니다. RIGHT는 진입 경로 `10.323 s`, 교차로 제어 활성 뒤 `COMPLETE`까지
`34.836 s`였고, 생성 경로의 순회전은 `-68.386°`, 반대 누적 회전은 `0°`였습니다.
실제 진입 명령 207개도 전부 우회전이었고 실제 yaw의 반대 excursion은 `0°`였습니다.
LEFT는 진입 `7.378 s`, 전체 `32.490 s`, 생성 경로 순회전 `+110.360°`, 반대 회전
`0°`였으며 진입 명령 148개가 모두 좌회전이었습니다. 두 주행 모두 인계 뒤 새 EKF
자세와 경로 시작점의 차이는 `0 m`, 진입·탈출의 종단 line allowance는 `0`, 안전 정지·
wrong-turn·timeout·`FAILED`는 0회였습니다. `COMPLETE` 뒤 일반 차선의 첫 nonzero 명령은
RIGHT `0.098 s`, LEFT `0.008 s` 뒤에 나왔습니다. 이는 공식 시작점부터 교차로 종료와
차선 복귀까지의 양방향 `1/1` 결과이며, 변경 뒤 결승선까지의 전체 코스는 다시 검증하지
않았습니다.

조기 방향 관찰 변경 뒤에도 GUI와 RViz를 켠 기본 통합 launch를 매회 새로 시작하고
`forced_direction=0`으로 양방향을 다시 검증했습니다. RIGHT 주행에서는 첫 후보를
ordered gate보다 `0.432 s` 먼저 저장하고 gate 뒤 `0.166 s`에 3회 확정했으며,
교차로 제어 시간 `35.081 s` 뒤 `COMPLETE`, 다시 `0.048 s` 뒤 일반 차선 명령이
복귀했습니다. LEFT 주행에서는 관찰 region이 gate보다 `1.004 s` 먼저 열렸고,
gate 뒤 들어온 RIGHT 단발 후보를 다음 LEFT가 초기화한 뒤 LEFT 3회를 확정했습니다.
gate부터 확정까지 `0.716 s`, 교차로 제어 시간은 `32.657 s`, `COMPLETE` 뒤 일반
차선 명령 복귀는 `0.014 s`였습니다. 두 주행 모두 공식 시작 자세와 실제 카메라 판독을
사용해 교차로 완전 탈출과 차선 복귀까지 수행했고 `FAILED`나 timeout은 없었습니다.
변경 뒤 결승선까지의 전체 코스는 다시 검증하지 않았습니다. 이 당시 패키지 등록 시험은
`378 tests`, 오류·실패·건너뜀 0이었습니다. 현재 시험 수와 통합 결과는 문서 상단의
2026-09-13 검증을 따릅니다.

반원 차선 추종의 전용 속도 상한은 `0.10 → 0.12 m/s`로 올렸습니다. 일반 차선의
전역 감속식과 제어 게인, 진입·탈출 경로 속도는 바꾸지 않았습니다. GUI와 RViz를 켠
기본 통합 launch를 매회 새로 시작하고 `forced_direction=0`의 실제 카메라 판독으로
RIGHT 3회와 LEFT 3회를 공식 시작 자세에서 검증했습니다. `/clock` 기준 반원 평균
시간은 RIGHT `12.195 → 11.020 s`(`9.63%` 단축), LEFT
`12.401 → 11.085 s`(`10.62%` 단축)이었고, 실제 평균 속도는 각각 약
`0.0713 m/s`, `0.0701 m/s`였습니다. 여섯 주행 모두 반원 중 zero 명령,
`FAILED`, timeout 없이 방향별 탈출 경로와 최종 차선 복귀를 거쳐 `COMPLETE`했습니다.
차선 중심 오차의 95% 값은 RIGHT `121.5 px`, LEFT `119.8 px`, 최대 검출 간격은
`0.186 s`로 허용 중심 오차 `250 px`와 watchdog `0.5 s` 안이었습니다. 최대
`|v·ω|`는 `0.0202 m/s²`로 설정된 횡가속 기준 `0.08 m/s²`보다 낮았습니다.
속도 변경 뒤 결승선까지의 전체 코스는 다시 검증하지 않았습니다.

진입·탈출 중 현재 활성 경로 하나만 `/intersection/generated_path`에 발행합니다. 경기장 색 경계는
`/intersection/course_map`, 확정 방향과 상태는 `/intersection/direction`과
`/intersection/state`에서 확인합니다. 일반 차선 제어기는 미션 소유 중에도 최신 유효
차선 관측을 캐시합니다. 제어권 반환 시 캐시가 0.5초 이내이면 다음 카메라 프레임 전
불필요한 0 속도 삽입을 막고, 오래된 표본이면 새 정상 표본이 올 때까지 0 속도를 유지합니다.

재현 시험에서 `2=좌회전`, `3=우회전`이며 실제 경기에서는 `forced_direction` 기본값
`0`으로 카메라 판독 결과를 사용합니다. `forced_direction`은 방향 결과만 고정하며
관찰 polygon과 사용 가능한 방향 표지판 검출 조건은 그대로 확인합니다.

```bash
roslaunch custom_autorace_bringup gazebo.launch wait_for_green:=false forced_direction:=2
roslaunch custom_autorace_bringup gazebo.launch wait_for_green:=false forced_direction:=3
```

### 장애물 미션

교차로 제어기가 `/intersection/state=COMPLETE`를 발행해야 순차 상태 머신이 다음
`obstacle` polygon의 `/mission/enable/obstacle` gate를 엽니다. 장애물 제어기는
`ACQUIRING` 동안 차선 제어기가 `/cmd_vel`을 계속 소유한 채 최고속도만 `0.09 m/s`로
낮춥니다. 공식 시작점 기록에서 진입 차선은 map x 약 `1.36 m`까지 직선이었습니다.
polygon의 서쪽 경계는 `1.28 m`이고 공통 `enter_margin: 0.03 m`가 적용되므로 gate는
약 `x=1.31 m`, 측정된 코너 약 `50 mm` 전에 열립니다. 이전 `x=1.08 m` 경계에서
직선 주행까지 일찍 `0.09 m/s`로 제한하던 구간은 사용하지 않습니다. map의 코스 방위
`90°`와 로봇 방위 차이가 `12°` 이내가 될 때까지는 카메라와 LiDAR를 장애물 직선의
자료로 해석하지 않고 일반 차선 주행으로 코너를 마칩니다. 정렬 뒤 AMCL 자세를 초기값으로
사용해 새 LiDAR scan마다 측량된 장애물 표면과 정합을 시도합니다. inlier 수와 RMS가
설정값을 만족하면 코스 좌표계와 카메라로 확인한 주행 corridor를 odom에 고정합니다.
현재 자세부터 남은 경로와 연결 구간의 직사각형 sweep까지 안전해야
`/control/lane_mission_handoff`를 호출하고 장애물 제어기가 단독으로 `/cmd_vel`을
소유합니다. 정합 또는 경로 검사가 실패한 scan에서는 제어권을 가져오지 않고 다음
scan으로 다시 검사합니다.

경로는 측량된 세 장애물을 한 번에 지나는 **clamped cubic spline**과 접속점의
heading·curvature를 맞춘 quintic 출구 곡선을 연결한 하나의 경로입니다. spline
내부는 C2이고 두 곡선의 접속부는 위치·진행방향·곡률이 연속인 G2입니다.
로컬 `+x`는 map `+y`, 로컬 `+y`는 map `-x`이고, map anchor는
`(1.6375, 0.0200)`입니다. AMCL은 미션 polygon 판정과 LiDAR 정합의 종·횡방향
초기값까지만 담당합니다. 정합이 끝나면 당시 로봇 자세와 코스 진행률·횡방향 위치를
odom에 latch하고, 이후 남은 경로 suffix와 장애물 위치는 이 고정 코스 좌표계에서
계산합니다. 따라서 장애물 사이의 수 mm 여유를 AMCL의 순간 보정 오차나 map→odom
점프로 움직이지 않습니다.

카메라 선은 코스 template 자체를 이동시키지 않습니다. `WAIT_GATE`와 `ACQUIRING`에서
노란선과 흰선이 모두 유효하고 그 중심이 측량 corridor와 `4 mm` 이내로 일치할 때만
odom 기준 corridor 중심을 갱신합니다. 한 선만 보이거나 간격·방위 검사를 통과하지
못한 표본은 사용하지 않습니다. LiDAR 정합도 선처럼 보이는 점을 제외하고 센서
위치에서 실제로 보이는 직사각형 면만 비교한 뒤 inlier 수와 RMS를 검사합니다.
따라서 앞면을 보이지 않는 뒷면에 붙여 진행률을 약 `0.10 m` 잘못 이동시키는 정합을
사용하지 않습니다.

`template/lateral_scale: 0.9950`으로 정한 단일 경로를 시작할 때 측량 장애물 전체와
hard line bounds에 대해 검사합니다. LiDAR 정합 뒤에는 현재 진행률부터 남은 suffix를
한 번만 확정하며 장애물마다 경로를 다시 만들지 않습니다. 주행 중에는 매 LiDAR
scan마다 현재 위치부터 앞 `0.45 m`의 swept footprint를 다시 검사합니다. 현재 자세나
도색선 경계를 추종할 수 없으면 즉시 `FAILED`, LiDAR 장애물이 경로를 막는 상태가
3 scan 연속이면 `FAILED`로 정지합니다. 대체 경로나 부분 안전 경로를 만들지 않으며,
한 번 확정한 경로를 주행 중에 다시 생성하지 않습니다.

명목 template 전체 sweep에는 `obstacle_padding: 0.014 m`를 적용합니다. 이미 그
여유로 설계된 경로를 live scan에서 다시 팽창하지 않도록
`validation_obstacle_padding: 0.0 m`를 사용하고, 도색선에는 계속
`validation_line_margin: 0.009 m`를 적용합니다.

속도 profile은 경로 곡률에 따라 선속도를 낮추고 `0.85 rad/s` 각속도,
`0.25/0.80 m/s²` 가·감속과 전체 yaw 가속도
`d(v·curvature)/dt ≤ 0.55 rad/s²`를 함께 제한합니다. 그래서 좁은 통로에서
제어기의 slew limiter가 갑작스러운 곡률 변화를 대신 처리하지 않습니다.

#### 직사각형 footprint와 곡률 연속 경로를 선택한 근거

Gazebo에서 측정한 비대칭 footprint는 base_link 기준 앞 `67.645 mm`, 뒤
`118.073 mm`, 반폭 `90.3 mm`입니다. 장애물 padding `14 mm`와 line margin
`9 mm`를 포함하면 두 번째 장애물 옆 `225.0 mm` 통로에 필요한 직사각형 폭은
`203.6 mm`입니다. 반면 가장 먼 뒤쪽 모서리로 만든 외접원의 반지름은
`148.645 mm`이고 같은 여유를 적용한 필요 폭은 `320.290 mm`입니다. 원 모델은 실제로
통과 가능한 길을 막혔다고 오판하므로, 모든 경로 표본에서 로봇 방위에 맞춰 회전한
네 모서리와 장애물 점·도색선 사이의 거리를 직접 검사합니다.

기존 4-piece cubic Bezier 후보는 조각 연결부에서 곡률이 튀어 같은 속도에서 조향이
갑자기 변했습니다. 현재 구현에는 그 후보 선택 코드가 남아 있지 않습니다. 장애물
구간은 clamped cubic으로 한 번에 계산하고, 마지막 좌회전만 접속점의 heading과
curvature를 맞춘 quintic으로 잇습니다. 최종 경로는 길이 `2.204808 m`, 최대 곡률
`6.2289 1/m`, 예상 주행시간 `31.548 s`, 최대 yaw 가속도 `0.5489 rad/s²`입니다.
실제 `course.png` 도색선 중심을 사용한 회귀 검사에서 `9 mm` line margin 적용 후
흰선과 노란선의 최소 여유는 각각 `5.1435 mm`, `10.8083 mm`였습니다.

상태 전이는 다음과 같습니다.

```text
일반 차선 제어기 소유
WAIT_GATE → ACQUIRING
                ├─ AMCL polygon + LiDAR 정합 seed
                ├─ 직사각형 표면 정합
                ├─ odom course-frame latch
                └─ cubic+quintic suffix 검사 → cmd_vel handoff
                                      ↓
장애물 제어기 소유                   AVOIDING
                                      ↓
                                 REJOINING
                                      ↓
                            lane handoff → COMPLETE

기동 전 명목 sweep 실패 ───────────────────→ 노드 초기화 실패
실시간 충돌·추종·handoff·조기 경로 끝 ─────→ FAILED (정지)
```

최신 `/mission/inside/obstacle=false`를 받으면 `AVOIDING`에서 `REJOINING`으로
전환합니다. `REJOINING`에서도 장애물 제어기가 경로 끝까지 `/cmd_vel`을 소유합니다.
미션 매니저가 `base_footprint` 기준점이 polygon 경계 밖 `0.11 m` 이상이라고
`/mission/clear/obstacle=true`를 발행하고, 남은 경로가
`handoff_remaining_distance: 0.04 m` 이하일 때 차선 제어기로 제어권을 넘긴 뒤
`/obstacle/state=COMPLETE`를 발행합니다. `0.11 m`는 출구 yaw `180°`에서 polygon
바깥 방향으로 놓이는 직사각형 반폭 `0.0903 m`, line margin `0.009 m`, 측정된 AMCL
오차 여유 `0.0107 m`를 합친 값입니다. 통합 시험에서는 실제 직사각형이 polygon을
`38.9 mm` 완전히 벗어난 뒤 완료됐습니다. 경로가 먼저 끝나거나 실시간 충돌·추종·
제어권 인계 검사에 실패하면 `FAILED`에서 정지합니다. 확정 뒤 일시적인
scan/odometry 수신 지연만으로는 실패시키지 않고, 새 scan이 올 때 전방 검사를
갱신합니다.

주요 조정값은 `config/obstacle_mission_gazebo.yaml`에 있습니다.

- `course`: map 코스 방위, 진입 방위 허용값, 카메라 pixel/m와 두 선 보정 허용 오차
- `footprint`: 비대칭 직사각형 크기, 명목 장애물 padding, live padding과 hard line margin
- `planner`: 연속 unsafe scan 수, 현재 자세 추종 허용 오차와 collision sampling 간격
- `template`: map anchor, 세 장애물, cubic knot·끝기울기·단일 횡 scale, quintic 출구,
  live 검사 거리·속도와 AMCL seed 기반 LiDAR 정합 범위·inlier·RMS
- `scan`: 사용 거리, 군집·downsample·median filter와 LiDAR 장착 오프셋
- `control`: 추종 속도, lookahead, 선·각 가속도, 제어 주기와 handoff 남은 거리
- `timeouts`: 경로를 확정하기 전 scan과 odometry freshness

경로 정합·확정 상태와 안전 여유는 다음 토픽에서 확인합니다.
`/obstacle/planner_status`는 `TEMPLATE_READY`, `ALIGNING`, `PATH_REJECTED`,
`PATH_COMMITTED`, `FAILED`, `COMPLETE` 중 하나입니다. `/obstacle/diagnostics`는
`PATH_FOLLOWING.md`에 정의한 공통 13개 진단 배열입니다.

```bash
rostopic echo /obstacle/state
rostopic echo /obstacle/planner_status
rostopic echo -n 1 /obstacle/local_path
rostopic echo /obstacle/diagnostics
```

교차로를 건너뛰고 장애물 코스만 회귀 시험할 때는 전용 sequence와 그에 맞는 초기
자세를 함께 사용합니다. `wait_for_green:=true`로 센서가 준비되는 동안 차선을 멈춘 뒤
시뮬레이터의 초록 신호 검출로 주행을 시작합니다. 일반 완주 시험에는 이 설정을
사용하지 않습니다.

```bash
roslaunch custom_autorace_bringup gazebo.launch \
  x_pos:=1.6375 y_pos:=-0.05 yaw_pos:=1.5707963 \
  wait_for_green:=true detect_signs:=false intersection_mission:=false \
  parking_mission:=false zigzag_mission:=false tunnel_mission:=false \
  mission_zone_config:=$(rospack find custom_autorace_bringup)/config/mission_zones_obstacle_test_gazebo.yaml
```

#### 장애물 진입 gate 속도 검증(2026-09-10)

이전 서쪽 경계 `x=1.08 m`에서는 공식 시작 주행의 직선 구간부터 장애물 gate가 열려
실제 회전을 시작하기 약 `0.20 m` 전부터 차선 제어 속도가 `0.09 m/s`로 제한됐다.
경계를 `x=1.28 m`로 옮긴 뒤 공통 `enter_margin: 0.03 m`를 포함한 활성 위치를
`x≈1.31 m`로 만들었다. 새 상태·경로·제어기는 추가하지 않았으며, 직선은 기존 차선
속도를 사용하고 기록된 회전 시작점 `x≈1.36 m` 약 `50 mm` 전부터 기존 진입 제한만
적용한다.

변경 뒤 새 Gazebo를 공식 시작 자세와 기본 통합 launch로 실행했다. 실제 녹색 신호와
카메라 RIGHT 판독을 거쳐 Intersection은 `176.336 s`, Obstacle은
`185.672 s`의 `ACQUIRING`부터 `216.024 s`의 `COMPLETE`까지 사람 개입 없이 진행했고,
차선 제어권 반환 뒤 다음 Parking gate가 `218.651 s`에 열렸다. 같은 공식 시작·실제
RIGHT 기준 이전 기록보다 Obstacle 활성 시간은 `32.431→30.352 s`, 즉
`2.079 s(6.4%)` 짧아졌다. Intersection 완료부터 Obstacle 완료까지는
`40.850→39.688 s`, 즉 `1.162 s(2.8%)` 단축됐다. 최소 raw line 여유는
`3.391 mm`, obstacle 여유는 `4.245 mm`였고 `FAILED`나 충돌 로그는 없었다.
현재 반복 검증 횟수는 `1/1`이며, 이 결과는 Parking 이후 전체 완주 완료를 뜻하지 않는다.

### 주차 미션

주차 제어기는 표지판을 사용하지 않습니다. 장애물 미션 뒤의 순차 AMCL gate가 열리면
먼저 일반 차선 제어기에 `0.09 m/s` 제한만 전달하고 `PREPARE_APPROACH` 동안 제어권과
주행을 그대로 유지합니다. gate 뒤에 새로 발행된 제한속도 이하 `/cmd_vel`과 새 odom을
받은 뒤, 그 이동 중 자세에서 적응 진입곡선을 생성·검증합니다. 경로가 안전한 경우에만
`/control/lane_mission_handoff`로 차선 제어기를 끄고 `/cmd_vel`을 단독으로 소유합니다.
첫 주차 명령은 마지막 차선 명령과 선속도·각속도를 똑같이 내며 다음 주기부터 가속도
한도 안에서 적응 경로 명령으로 수렴하므로 진입 준비를 위해 멈추지 않습니다.

Gazebo의 측량 경로는 다음과 같습니다.

1. 제어권 인계 직전의 최신 이동 odom 자세에서 측량 회전 시작점
   `(0.6950, 1.7425, 180°)`까지 매 실행마다 zero-end-curvature quintic 연결곡선을
   생성합니다. 양 끝 접선 길이는
   남은 전진거리의 `0.20`배로 정하고, 현재 횡·방위 오차를 곡선 안에서 연속적으로
   수렴시킵니다. 연결곡선의 footprint sweep이 입구 도색의 실제 기울기와 안전 여유를
   만족할 때 속도를 끊지 않고, 변위 `0.195 m`인 고정 quintic 전진 좌회전으로 바로
   이어집니다. 곡선 끝 `(0.5000, 1.5475, -90°)`에서도 정지하지 않고 중앙 판정 자세
   `(0.5000, 0.6920, -90°)`까지 직진한 뒤 LiDAR 판정을 위해 정지합니다.
2. 최신 `/scan_mid360_raw`의 원래 `angle_min`, `angle_increment`와 LiDAR 장착
   오프셋을 사용해 scan 점을 AMCL 지도에 투영합니다. 좌·우 측량 ROI 가운데 한 곳은
   점유되고 다른 한 곳은 비어 있다는 판정이 새 scan 3개에서 연속될 때만 빈 주차면을
   확정합니다.
3. 빈 주차면을 확정하면 위치를 다시 옮기지 않고 중앙 판정 자세에서 선속도 0으로 바로
   90° 제자리회전합니다. 좌측 빈 칸이면 `0°`, 우측 빈 칸이면 `180°`를 바라봅니다.
   이어 좌측 `x=0.7741 m` 또는 우측 `x=0.2275 m`까지 직선으로 전진하여 footprint
   전체를 선택한 점선 안쪽에 둡니다.
4. `PARK_IN` 종단은 서로 다른 최신 odom 2개로 확인합니다. `PARKED`는 같은 callback에서
   AMCL 진단 checkpoint만 기록하며 별도 유지시간 없이 즉시 저장한 실제 회전점까지
   같은 직선을 후진합니다. `BACK_OUT`을 최신 odom 2개로 확인하면 그 자리에서 선속도
   0으로 북쪽 `90°`까지 다시 제자리회전합니다.
5. 북쪽 제자리회전을 마친 실제 복귀점에서 지그재그 회전 시작점
   `(0.4960, 1.6000, 90°)`까지 매 실행마다 하나의 zero-end-curvature quintic을
   생성합니다. 별도의 중앙 복귀 connector나 직선 조각은 두지 않습니다. 여기서 정지하지
   않고 변위 `0.150 m`, 접선 길이 `0.055 m`인 zero-end-curvature quintic 좌회전으로
   `(0.3460, 1.7500, 180°)`까지 진행한 다음, 같은 방위의 `0.120 m` 정렬 tail을
   `(0.2260, 1.7500, 180°)`까지 이어갑니다. 이전 반지름 `0.075 m` 원호보다 최대
   곡률이 `13.333 → 11.261 1/m`로 낮고 두 quintic의 접속 위치·방위·곡률과 실행
   속도가 연속이므로 경로 사이 정지가 없습니다.
6. 직선 tail에서 `x=0.310 m`를 지난 뒤 카메라 입력으로 생성·검증된 rolling
   `CommonPath`가 9 frame 확인되면, 주차 제어기가 마지막 안전 전진 명령과
   `0.10 m/s` 제한을 먼저 발행하고 움직이는 상태로 일반 차선 제어기에 `/cmd_vel`을
   넘깁니다. 한쪽 경계만 보여도 공통 경로 생성기가 유효 경로로 판정하면 사용할 수
   있습니다. 차선 제어기는 같은 서쪽 직선을 이어 주행하며 `x=0.08--0.19 m` 완료
   영역에서 새 유효 경로 9 frame을 확인하면 `/parking/state=COMPLETE`가 됩니다.
   tail 끝까지 시야가 확보되지 않은 경우에만 기존 정지 확인 상태를 사용합니다. 일반 속도 한도는
   `0.10 m/s`로 유지하여 다음 순차 gate가 열린 지그재그 제어기가 첫 굴곡 전에
   제어권을 받게 합니다.

AMCL은 순차 gate와 정지한 공간 판정 자세의 LiDAR 점을 지도 ROI로 투영하는 데
사용합니다. 인계 순간의 `map → odom` 관계를 한 번 고정하므로 이후 짧은 주차 경로는
local odom으로 추종하며 AMCL 보정으로 움직이는 목표가 되지 않습니다. Gazebo 프로필도
`route/odom_aligned: false`로 이 경로를 검증합니다. `PARKED`는 진단 checkpoint일 뿐
상태 유지 조건이 아니며 `PREPARE_EXIT` 상태는 사용하지 않습니다. `PARK_IN`과
`BACK_OUT`은 동일한 실제 odom 회전점을 공유합니다. 두 제자리회전은 선속도 0, 전체
정적 sweep과 최신 LiDAR sweep, 목표 방위와 실제 각속도 정지 및 서로 다른 최신 odom
2개를 확인합니다. 회전 중 발생한 실제 위치 변화는 진단값으로 기록하고, 매 주기 현재
odom 자세부터 남은 회전 sweep을 다시 검사합니다. 따라서 안전한 미끄러짐을 임의의
누적 거리로 실패 처리하지 않으며 이후 직선 경로도 측정된 실제 회전 종료점에서
생성합니다. 회전 중 장애물 여유
`rotation_obstacle_margin`은 `0`이므로 가상의 추가 간격 없이 실제 footprint가 장애물에
접촉하는 경우만 차단합니다. 도색과 지도 경계에는 기존 line·localization·tracking
여유를 그대로 적용합니다.
진입 연결곡선은 고정 AMCL x/y/yaw 시작 박스나 고정 오차 허용창을 기다리는 대신 실제
인계 odom 자세에서 매번 다시 만듭니다. 순차 gate가 미션 시점을 결정하고, 경로의 전진
단조성, 차체 footprint의 도색 여유, 곡률과 곡률 변화율이 실제 진입 가능 여부를
결정합니다. 위치 오차만으로는 진입을 막지 않지만, 목표를 이미 지나쳤거나 도색을
침범하는 실제 자세, 오래된 입력처럼 물리적으로 안전한 경로를 만들 수 없는 경우에는
정지합니다. 저장한 실제 odom 자세를 후진 목표로 재사용하므로 들어간 점선 입구로
그대로 나옵니다. 중앙 제자리회전과 후진 뒤 북쪽 제자리회전은 실제 도색 픽셀, 비대칭
로봇 footprint와 주차된 Burger 충돌 형상에 대해 검사합니다. 해당 상태에서 필요한
AMCL, odom, scan 또는 영상이 오래되면 주차 제어권을 유지한 채 0 속도를 발행합니다.
단 AMCL은 미션 시작과 주차면 선택처럼 실제로 사용하는 상태에서만 필수입니다.
odom 최신값과 scan-odom 동기화 이력은 swept-footprint 계산이 잡는 제어 lock과 분리해
콜백 도착 시 바로 저장합니다. 따라서 정상 주기의 센서가 긴 안전 계산 뒤 callback queue에서
기다리다가 거짓 stale로 판정되지 않습니다. 실제 stale 입력은 예측 주행 없이 즉시 0을
명령하고, 다음 동작은 명령값뿐 아니라 odom 선속도·각속도도 정지 임계값 아래인 경우에만
시작합니다. 주차면 판정 scan도 lock이 겹치면 최신 한 장을 보관해 다음 제어 주기에서
처리하되 `SELECT_SPACE` 시작 전 scan은 확인 횟수에 포함하지 않습니다.
진입 직선과 연속 곡선도 실제 도색 픽셀 및 두 주차 표지판 충돌 상자에 대해 연속
sweep으로 검사합니다. 이미 낮은 여유로 인계된 경우 연결곡선이 그 여유를 더 줄이지
않는지도 검사합니다.
`LEAVE_AISLE`의 `parking_exit` 경계는 두 aisle 선을 무한히 연장하지 않습니다. 실제
texture에서 끝나는 solid arm과 그 사이의 유한한 출구 개구를 union으로 표현해, 합법적인
점선 개구는 통과시키고 실제 solid arm을 가로지를 때만 공통 검증기가 거부합니다.
두 주차면이 모두 비었거나 모두 점유된 것처럼 보이는 scan, 경로 제한시간 초과 또는
제어권 인계 실패는 임의 복구 없이 `FAILED`로 정지합니다.

`/cmd_vel` 소유권과 상태 순서는 다음과 같습니다.

```text
일반 차선 제어기 소유
WAIT_GATE → PREPARE_APPROACH
            └─ 제한속도 반영 새 cmd + 새 odom으로 적응 연결곡선 검증
                                                       ↓ 같은 비영 속도로 cmd_vel 인계
주차 제어기 소유                        APPROACH → TURN_IN → ENTER_AISLE
                                       → SELECT_SPACE → TURN_TO_SPACE
                                                          중앙 제자리회전
                                       → PARK_IN → BACK_OUT
                                                    └ 유지시간 없음
                                       → TURN_TO_EXIT → LEAVE_AISLE
                                             제자리회전       ↓ 단일 quintic·무정지 접속
                                       → TURN_TO_ZIGZAG ── 유효 CommonPath 9 frame
                                                       │      ↓ 이동 중 인계
일반 차선 제어기 소유                                 ├─→ JOIN_ZIGZAG → COMPLETE
                                                       └ tail 끝 시야 누락
                                                          → VERIFY_ZIGZAG_LANE
                                                          → JOIN_ZIGZAG → COMPLETE

검증·제어권 인계·제한시간 실패 ───────────────────────────────→ FAILED (정지)
```

현재 순차 설정에서 parking 다음은 zigzag입니다. 주차 제어기는 서쪽 정렬 tail을 움직이며
검증된 rolling CommonPath와 Gazebo 측량 odom 자세를 확인해 차선 제어기에 넘기고, 짧은 직선에서
`COMPLETE`합니다. 지그재그 제어기는 별도 순차 gate에서 현재 진행 위치를 측량 경로에
투영하여 이어받습니다.

주요 상태와 판정값은 다음 토픽에서 확인합니다. 점유 배열은 `[좌측 점 수, 우측 점 수]`
순서입니다.

```bash
rostopic echo /mission/enable/parking
rostopic echo /mission/map_pose
rostopic echo /parking/state
rostopic echo /parking/selected_space
rostopic echo /parking/occupancy_points
rostopic echo /parking/diagnostics
rostopic echo /control/lane_path_diagnostics
```

주차만 반복 시험할 때는 전용 sequence와 동일한 시작 자세를 사용합니다.
`parking_obstacle_x`는 제어 입력이 아니라 Gazebo 환경의 고정 회귀 조건이며,
`0.23`은 우측 장애물(좌측 주차), `0.73`은 좌측 장애물(우측 주차)입니다. 일반 경기의
기본값은 `random`입니다. 주차 제어기는 표지판 검출 여부와 무관하므로 아래 명령은
AMCL gate부터 지그재그 차선 인계까지 전체 경로를 회귀합니다.

```bash
roslaunch custom_autorace_bringup gazebo.launch \
  gui:=false rviz:=false x_pos:=1.20 y_pos:=1.735 yaw_pos:=3.14159265 \
  odometry_source:=world \
  wait_for_green:=false detect_signs:=false \
  intersection_mission:=false obstacle_mission:=false zigzag_mission:=false \
  level_crossing_mission:=false tunnel_mission:=false \
  parking_obstacle_x:=0.23 \
  mission_zone_config:=$(rospack find custom_autorace_bringup)/config/mission_zones_parking_test_gazebo.yaml

roslaunch custom_autorace_bringup gazebo.launch \
  gui:=false rviz:=false x_pos:=1.20 y_pos:=1.735 yaw_pos:=3.14159265 \
  odometry_source:=world \
  wait_for_green:=false detect_signs:=false \
  intersection_mission:=false obstacle_mission:=false zigzag_mission:=false \
  level_crossing_mission:=false tunnel_mission:=false \
  parking_obstacle_x:=0.73 \
  mission_zone_config:=$(rospack find custom_autorace_bringup)/config/mission_zones_parking_test_gazebo.yaml
```

#### 중앙 즉시 제자리회전 최종 검증(2026-09-10)

현재 구현은 `POSITION_FOR_PARKING`이나 좌·우 분기별 안전 회전점으로 가는 이동 상태를
사용하지 않습니다. `SELECT_SPACE`에서 새 LiDAR scan 3개로 빈 칸을 확정한 뒤 같은 중앙
자세에서 곧바로 `TURN_TO_SPACE`로 전환합니다. 공식 시작 자세와 기본 통합 launch에서
신호등 대기, Intersection, Obstacle, Parking, Zigzag를 모두 순서대로 거치는 새 Gazebo
시험을 양쪽 주차 조건에 각각 수행했습니다.

- 우측 빈 칸 run `parking_direct_official_right_anchor_JL8k64`은 Intersection
  `171.308 s`, Obstacle `211.192 s`에 완료됐습니다. Parking은
  `PREPARE_APPROACH 213.848 s → COMPLETE 276.699 s`, 총 `62.851 s`였고,
  `RIGHT` 확정 `234.068 s → TURN_TO_SPACE 234.092 s`로 별도 위치 이동 없이
  `24 ms` 뒤 같은 자리에서 회전을 시작했습니다. Zigzag도 `293.528 s`에
  `COMPLETE`했습니다.
- 좌측 빈 칸 run `parking_direct_official_left_final_CNF4xJ`은 Intersection
  `174.366 s`, Obstacle `214.062 s`에 완료됐습니다. Parking은
  `PREPARE_APPROACH 216.684 s → COMPLETE 278.851 s`, 총 `62.167 s`였고,
  `LEFT` 확정 `236.934 s → TURN_TO_SPACE 236.954 s`로 `20 ms` 뒤 회전을
  시작했습니다. Zigzag도 `296.050 s`에 `COMPLETE`했습니다.

두 실행 모두 `PARK_IN → BACK_OUT → TURN_TO_EXIT`을 중단 없이 완료했고
`FAILED`가 없었습니다. Gazebo physics contact 전체 기록에서 바닥
`course::course_link` 접촉을 제외해 검사한 로봇-표지판·로봇-주차 장애물 접촉은 양쪽
모두 `0`건입니다. 좌측 기록은
`diagnostics/parking_direct_official_left_final_CNF4xJ/official_left_final.bag`, 우측
기록은 `diagnostics/parking_direct_official_right_anchor_JL8k64/official_right_anchor.bag`
입니다. 주차 controller, 주차 기하, 공통 경로 추종, Zigzag 회귀 `167`개도 모두
통과했습니다. 이 검증은 요청 범위인 공식 시작부터 주차 및 바로 뒤 Zigzag 완료까지의
완료 근거이며, 이후 차단봉·터널·결승선까지 프로젝트 전체 완주를 새로 판정한 기록은
아닙니다.

#### 이전 분기별 회전점 제자리회전 구현 검증 기록(보관)

아래 결과는 공간 선택 뒤 `POSITION_FOR_PARKING`에서 좌·우 분기별 회전점으로 이동하던
이전 구현의 기록이며, 현재 중앙 판정 자세 제자리회전 경로의 완료 근거가 아닙니다.
당시 주차 완료 유지시간과 `PREPARE_EXIT`를 제거하고
`POSITION_FOR_PARKING → TURN_TO_SPACE → PARK_IN → BACK_OUT → TURN_TO_EXIT → LEAVE_AISLE`
순서로 바꾼 코드에서 새 Gazebo 전용 회귀를 양쪽 분기 각각 수행했습니다.
좌측 빈 칸 run `29ff3b5e-ac70-11f1-89ff-22316c0401e8`은
`PREPARE_APPROACH 118.235 s → COMPLETE 212.835 s`로 `94.600 s`, 오른쪽 빈 칸 run
`81daf5f2-ac70-11f1-81a5-22316c0401e8`은 `118.053 s → 214.057 s`로 `96.004 s`에
사람 개입 없이 완료했습니다. 두 실행 모두 `PARKED` 진단과 `BACK_OUT` 상태 전환의
간격은 `0.001 s`였고, `FAILED`나 충돌 로그가 없었습니다.

진입 회전점의 짧은 연결부는 점유된 반대편 Burger 여유 때문에 tangent `0.020 m`를
유지했습니다. 장애물 여유가 큰 출구 연결부만 별도 `0.030 m`로 완만하게 하자
`LEAVE_AISLE`이 좌/우 각각 `21.747 s`, `21.752 s`가 되어 직전 같은 경로의
`30.501 s`, `32.563 s`보다 각각 `8.754 s`, `10.811 s` 짧아졌습니다. ±8 mm·±1°
시작오차 전 조합을 11 mm 확장 footprint로 검사한 최소 여유는 실제 도색
`44.018 mm`, 점유 Burger `17.420 mm`였습니다.

주차 뒤와 같은 서쪽 진행 자세의 Zigzag 전용 run
`f8452be4-ac71-11f1-a3a2-22316c0401e8`은 `17.960 s`에 `COMPLETE`했고 최대 경로오차는
`4.6 mm`였습니다. 따라서 지그재그 자체는 기존 측량 경로·라인 인계 구성으로 주행
가능합니다. 이후 직선 인계 속도와 센서 callback 경로를 수정한 당시 코드에서는 공식
시작점부터 Intersection, Obstacle, LEFT Parking, Zigzag 차선 복귀까지 통합 주행을
완료했습니다. 상세 시간과 실패 표본은 아래 `주차→지그재그 직선 인계` 검증 절에
기록합니다.

이 단계 트리의 주차 controller `65`개와 패키지 전체 `365`개 회귀는 모두 통과했습니다.

#### 이전 원호 기반 구현 검증 기록(보관)

아래 기록은 `ARC_IN`, `PREPARE_EXIT`, `ARC_OUT`과 주차 유지시간을 사용하던 이전
구현의 이력이며, 현재 제자리회전 경로의 완료 근거가 아닙니다.

2026-09-07 이전 구현의 최신 공식 시작 주행은 오른쪽 빈 칸에서 `BACK_OUT`까지 마친 뒤
`PREPARE_EXIT`에서 실패했습니다. 안전한 실제 odom 복귀점 `x=0.3908 m`과 raw AMCL
`x=0.3522 m`를 서로 다른 좌표계인데도 직접 비교해 `46.6 mm`가 되었고, 고정
`45 mm` 조건을 넘은 것이 원인이었습니다. 그보다 앞선 좌·우 성공 주행은 개별 성공
사례이지 이 실패를 무효화하는 반복 성공 근거로 보지 않습니다.

수정 뒤에는 공식 시작점의 전체 `gazebo.launch`를 매번 새로 실행하고 장애물만
`x=0.73`으로 고정해 같은 오른쪽 빈 칸 분기를 2회 연속 재시험했습니다. 첫 실행(run
`c890a736-aaa3-11f1-b977-22316c0401e8`)은 교차로와 장애물 미션을 순서대로 거쳐 주차
gate가 `204.698 s`에 열렸고, 길이 `0.281 m`, 횡보정 `-20 mm`, 계획 최소 여유
`10 mm`인 적응 진입으로 오른쪽 칸을 선택했습니다. `BACK_OUT 240.423 s`,
`PREPARE_EXIT 246.422 s`, `ARC_OUT 246.523 s`로 문제 지점의 정지는 `0.101 s`였으며,
AMCL 왕복 보정 차이 `13.0 mm / 1.10°`는 진단만 기록했습니다. 주차는 `279.122 s`,
지그재그는 `297.722 s`에 `COMPLETE`였습니다. 두 번째 실행(run
`cbf3916c-aaa4-11f1-a084-22316c0401e8`)도 `PREPARE_EXIT 246.437 s`에서 `0.103 s` 뒤
`ARC_OUT`으로 전환했고 AMCL 보정 차이는 `14.1 mm / 1.20°`였습니다. 주차
`279.137 s`, 지그재그 `297.695 s`에 각각 `COMPLETE`였으며 두 실행 모두 `FAILED`,
충돌 로그가 없었습니다. 이 수정 뒤 공식 시작 통합 성공은 현재 같은 오른쪽 분기 2회이며,
반대쪽 분기 반복 재현성은 후속 주행으로 계속 확인해야 합니다.
주차 단위 회귀 46개와 패키지 전체 회귀 117개도 통과했습니다.

2026-09-08에는 출차 중 위치 이동의 근본 원인을 AMCL `diff` odom 모델에서 재현했습니다.
원본 모델은 1 cm 미만 이동에서 방위 계산을 생략하면서 이동량은 항상 양수로 두므로,
`ARC_OUT`의 짧은 후진 갱신을 전진으로 적용했습니다. `update_min_a`를 `0.05`에서
`0.10 rad`로 높인 임시 A/B 시험은 좌·우 주차 모두 성공하고 `ARC_OUT`부터
`LEAVE_AISLE`까지 map 이동량과 raw odom 이동량의 차이를 각각 `0.614 mm`,
`2.372 mm`로 줄였지만, 회전으로 발생하는 관측 갱신 주기를 낮추는 경로 의존 우회이므로
채택하지 않았습니다.

대신 `update_min_a: 0.05`를 유지하고 workspace AMCL의 `diff-signed` 모델을 적용했습니다.
주차 전용 새 프로세스 시험에서 좌측 빈 칸과 우측 빈 칸을 각각 사람 개입 없이
`COMPLETE`했고 같은 출차 이동량 차이는 `2.484 mm`, `4.468 mm`였습니다. 이어 공식
시작 자세에서 기본 미션 구성의 통합 launch를 주차 장애물 `x=0.23`으로 고정해 새로
실행한 run
`d38a8098-aad1-11f1-b162-22316c0401e8`은 녹색 신호 뒤 교차로, 장애물, 좌측 주차,
지그재그, 차단봉을 순서대로 통과해 `/mission/state=COMPLETE`가 되었습니다. 이 주행의
출차 이동량 차이는 `3.259 mm`, yaw 변화량 차이는 `0.003 deg`로, 이전 약 `30 cm`의
반대 방향 이동은 사라졌습니다. `FAILED`와 충돌은 없었고, 녹색 신호부터 설정된 마지막
미션 완료까지 `216.774 s`였습니다. AMCL 회귀 8개, bringup 회귀 144개, 카메라 회귀
8개도 모두 통과했습니다. 이 실행 당시 sequence의 마지막은 차단봉이었으므로, 새로
추가한 터널 제어와 결승선 통과까지 검증한 결과는 아닙니다.

2026-09-09에는 주차 속도 값을 일괄적으로 더하는 대신 먼저 각 구간의 목표 속도를
`0.10 m/s`로 올리고 한계 검사를 수행했습니다. 첫 후보의 `approach_velocity: 0.10`은
공식 주행에서 관측한 진입 연결 자세를 사용한 정적 검사에서 각가속도가
`0.55944 rad/s²`가 되어 설정 한도 `0.55 rad/s²`를 넘었으므로 채택하지 않았습니다.
기존 `0.08`과 첫 후보 `0.10`의
중간값인 `0.09 m/s`는 검사를 통과했고, 차선 인계 속도도 같은 `0.09 m/s`로 맞췄습니다.
최종 설정은 접근·차선 인계만 `0.09 m/s`, 통로·주차·후진·원호·진입 회전·복귀·순항
목표는 `0.10 m/s`입니다. 원호와 짧은 주차 직선은 공통 속도 profile이 곡률 및
가감속 거리로 자동 제한하므로, 기록된 실제 최고 목표 속도는 주차 원호 약
`0.0558 m/s`, 주차·후진 직선 약 `0.054--0.060 m/s`, 마지막 복귀 원호
`0.0375 m/s`였습니다.

새 Gazebo 프로세스로 수행한 주차 전용 회귀는 오른쪽 빈 칸 `93.455 s`, 왼쪽 빈 칸
`92.613 s`에 각각 `PREPARE`부터 `COMPLETE`까지 사람 개입 없이 성공했습니다. 이어
공식 시작 자세에서 녹색 신호, 교차로, 장애물을 순서대로 거치는 통합 launch를 양쪽
주차 분기에 각각 새로 실행했습니다. 오른쪽 빈 칸은 run
`058ed57e-ac4b-11f1-8879-22316c0401e8`에서 주차 `93.809 s`, 지그재그
`18.599 s`였고, 왼쪽 빈 칸은 run `464c5d6a-ac4c-11f1-928d-22316c0401e8`에서
주차 `97.163 s`, 지그재그 `18.565 s`였습니다. 같은 왼쪽 분기의 변경 전 주차
`109.927 s`와 비교하면 `12.764 s`, `11.61%` 단축됐습니다. 두 주행 모두 후진,
출차 원호, 직선 차선 복귀와 지그재그 차선 제어권 반환까지 완료했고 `FAILED` 및 충돌
로그가 없었습니다. 지그재그의 최대 경로 오차는 각각 `8.4 mm`, `6.4 mm`였습니다.

최종 설정의 공식 시작 통합 시도는 총 4회였습니다. 이 중 2회는 주차 gate 전에 기존
교차로 진입 원호 추종 timeout으로 끝나 주차 속도 평가에는 도달하지 못했고, 주차에
도달한 나머지 2회는 양쪽 분기를 각각 완수했습니다. 주차 형상 회귀 7개와 패키지 전체
회귀 331개는 최종 설정에서 모두 통과했습니다. 병렬 전체 회귀 첫 실행에서는 지그재그
계산시간 표본 하나가 `62.6 ms`로 `50 ms` 예산을 넘었지만, 해당 30개 검사를 3회
단독 반복하고 전체 331개를 직렬 재실행한 결과는 모두 통과했습니다. 이 통합 시험은
주차 변경의 차선 복귀와 바로 뒤 지그재그까지를 검증한 것이며, 차단봉·터널·결승선은
비활성화했으므로 프로젝트 전체 완주 검증으로 보지는 않습니다.

같은 날 직선 구간을 더 빠르게 하기 위해 통로·주차·후진·순항 속도 상한을
`0.20 m/s`까지 올렸습니다. 접근과 차선 인계는 각가속도 한계를 지키기 위해
`0.09 m/s`, 진입·출차 원호와 복귀는 검증된 `0.10 m/s`를 그대로 유지했습니다.
먼저 `0.20`, `0.15`, `0.125`, `0.1125 m/s`를 이분 탐색 방식으로 주차 전용
회귀했으며 모든 후보가 주차와 후진, 출차, 차선 복귀까지 완료했습니다. 다만 최초
후보들은 직선 정지 sweep에서 거의 0인 여러 yaw rate를 동일한 궤적으로 반복 검사해
LiDAR 표본이 많은 순간 제어 주기를 놓쳤고, 속도를 낮춰도 시간이 개선되지 않았습니다.

공통 swept-footprint 검사기는 이미 직선으로 취급하는 `1e-9 rad/s` 이하 yaw rate를
하나로 합치고, 고정·실시간 장애물의 최근접 거리 계산과 footprint 확장을 한 번만
수행하도록 정리했습니다. 저장된 인접 pose가 검사 간격 이내이면 같은 끝점과 같은
허용량만 직접 검사합니다. 4,000개 실시간 장애물 점을 넣은 `0.20 m/s` 완전정지
검사의 변경 전 계산시간은 진입 약 `69.3 ms`, 출차 약 `64.6 ms`였고, 변경 뒤에는
각각 median/max `25.218/26.400 ms`, `21.695/23.009 ms`로 20 Hz 예산 안에
들었습니다. 안전 여유와 unsafe 판정 결과는 변경 전후가 같았습니다.

최종 `0.20 m/s` 설정의 새 주차 전용 회귀는 오른쪽 빈 칸 run
`30ad3966-ac56-11f1-b8bc-22316c0401e8`에서 `89.450 s`, 왼쪽 빈 칸 run
`8f7dd608-ac56-11f1-9a1c-22316c0401e8`에서 `89.700 s`에 각각 사람 개입 없이
완료됐습니다. 공식 시작점 통합 주행도 새 Gazebo 프로세스로 양쪽을 검증했습니다.
오른쪽 run `5e6e5d82-ac59-11f1-9aba-22316c0401e8`은 녹색 신호, 교차로, 장애물,
오른쪽 주차와 지그재그를 순서대로 완료했고 주차 `90.350 s`, 지그재그 `18.512 s`로
기존 `0.10 m/s`의 `93.809 s`, `18.599 s`보다 각각 `3.459 s`, `0.087 s`
짧았습니다. 왼쪽 run `fa3eb858-ac57-11f1-aabb-22316c0401e8`도 같은 순서로
왼쪽 주차와 지그재그를 완료했고 주차 `92.650 s`, 지그재그 `18.463 s`로 기존
`97.163 s`, `18.565 s`보다 각각 `4.513 s`, `0.102 s` 짧았습니다.

두 공식 주행 모두 후진 중 정지 실패, `FAILED`, 충돌 없이 `BACK_OUT`부터
`LEAVE_AISLE`, 직선 차선 복귀와 지그재그 제어권 반환까지 완료했습니다. 각 주행에서
일회성 stale scan 정지는 2회였으며 기존 `0.10 m/s` 공식 주행과 같은 횟수이고,
후진 구간에서는 발생하지 않았습니다. 짧은 거리와 `0.03 m/s²` 가감속 제한 때문에
실제 최고 목표 속도는 진입 직선 약 `0.137 m/s`, 주차·후진 약
`0.054--0.060 m/s`, 출차 직선 약 `0.170--0.172 m/s`였습니다. 장애물 위치
`x=0.23`으로 왼쪽 분기를 선택한 앞선 두 공식 시도는 주차 gate 전에 기존 교차로
왼쪽 원호 경계/timeout으로 끝났으므로 주차 실패나 성공 표본에 포함하지 않았습니다.
최종 트리의 패키지 전체 회귀 335개는 모두 통과했습니다. 차단봉·터널·결승선은 이
통합 launch에서 비활성화했으므로 이번 결과도 프로젝트 전체 완주 검증은 아닙니다.

#### 주차→지그재그 직선 인계 속도 검증(2026-09-10, 기존 출구 원호)

직선 인계가 느려진 원인은 두 가지였습니다. 주차 완료 뒤 차선 제어는 약 `0.10 m/s`로
주행하고 있었지만 Zigzag의 `entry_velocity_cap: 0.06`이 gate가 열릴 때 속도를 다시
`0.06 m/s`로 낮췄고, Zigzag follower도 그 낮은 속도에서 다시 가속했습니다. 두 미션의
인계 상한을 `0.10 m/s`로 맞춰 직선에서 불필요한 감속이 없게 했습니다.

또한 주차의 swept-footprint 계산 중 하나의 controller lock이 scan과 odom callback까지
기다리게 해, 원본 토픽은 정상 주기인데 controller 내부 timestamp만 stale이 되는 경우가
있었습니다. scan-odom 안전 이력과 최신 odom snapshot을 별도 짧은 lock에서 먼저 저장하고
control tick이 최신 snapshot 하나를 적용하도록 바꿨습니다. gate edge보다 먼저 도착한
in-flight odom은 projection이 끝나기 전에도 gate 기준값에 포함됩니다. 주차면 scan은 main
lock이 겹치면 최신 한 장을 다음 control tick에서 처리합니다. 실제로 센서가 stale인 경우는
추정 주행하지 않고 즉시 0을 명령하며, segment 전이는 odom의 선속도와 각속도까지 정지한
뒤에만 허용합니다.

반지름 `0.075 m` 출구 원호를 사용하던 당시 공식 시작점 LEFT 통합 run
`b43d2848-acd5-11f1-9cda-22316c0401e8`은 Intersection과 Obstacle을 순서대로 거쳐
Parking `PREPARE_APPROACH 221.251 s → COMPLETE 313.975 s`, `92.724 s`에 완료했고,
`LEAVE_AISLE 278.556 s → TURN_TO_ZIGZAG 298.026 s`는 `19.470 s`였습니다. 이어
Zigzag `ACQUIRING 314.076 s → FOLLOWING 314.214 s → COMPLETE 331.790 s`로,
경로 획득 `0.138 s`, 전체 `17.714 s`에 차선 제어권을 반환했습니다. Parking과
Zigzag의 stale 입력, `FAILED`, 충돌 또는 unsafe 로그는 없었습니다. Zigzag 최대
위치·횡오차는 `5.32 mm`, 최소 바깥 도색 여유는 `2.92 mm`, 최소 장애물 여유는
`25.56 mm`였습니다.

인계 직전과 직후 `/control/max_vel`은 모두 `0.10 m/s`였고 실제 `/cmd_vel`은 Parking
완료 뒤 차선 제어기의 마지막 `0.097915 m/s`에서 Zigzag의 첫 명령
`0.100355 m/s`로 이어졌습니다. 이전 `0.06 m/s` 설정의 공식 LEFT 기록에서는 같은
구간이 `0.09833 → 0.05890 m/s`로 떨어지고 Zigzag 첫 추종 명령이 `0.04200 m/s`였습니다.
cap을 `0.10 m/s`로 맞춘 뒤에도 첫 구현은 경로 획득 중 controller lock을 `0.343 s`
잡아 odom callback을 밀었고, stale 감속으로 `0.097563 → 0.078973 m/s`가 됐습니다.

고정 측량선과 texture 도색 경계는 노드 시작 시 한 번 전체 sweep해 확정합니다. 경로를
map에서 odom으로 강체 변환해도 이 고정 여유는 변하지 않으므로, 미션 진입 때는 한
스냅샷의 LiDAR 장애물만 전체 후보 경로에 다시 검사합니다. 입력 캡처와 최종 commit만
짧게 잠그고 sweep은 lock 밖에서 수행합니다. 완료 뒤에는 캡처 입력과 최신 입력의
신선도, gate·수동정지·shutdown, odom frame을 다시 확인하고 최신 odom 자세로 합류 오차를
계산합니다. 후보마다 역변환을 고정해 다른 run의 좌표가 섞이지 않게 하며, 실제 handoff
완료 시각부터 속도 제한 시간을 계산합니다. 이 변경으로 경로 획득은 `0.138 s`로
`0.205 s` 짧아졌고 stale 경고는 `1 → 0`건, 첫 명령은 `0.078973 → 0.100355 m/s`가
됐습니다. Zigzag 완료시간도 직전 최종 run의 `17.931 s`보다 `0.217 s` 짧았습니다.
주행별 초기 자세와 앞선 미션 편차가 있으므로 전체 시간 차이는 순수 알고리즘 A/B로
해석하지 않고, 인계 순간의 caller별 명령과 stale 경고 제거를 채택 근거로 사용합니다.

출구 C2 tangent를 현재 `0.030 m`보다 키우는 `.035`와 `.040 m` 이상 후보도 실제
50 mm lookahead 규칙으로 계산했지만, 최종 run 시작 자세에서 예상 `LEAVE_AISLE` 시간이
각각 `0.311 s`, `0.775 s` 늘었습니다. 최대 곡률도 커지므로 경로 형상은 변경하지
않았습니다. 앞선 callback 수정 검증의 첫 LEFT 시도 한 번은 진입 자세 편차로 적응
`APPROACH` 경로의 도색 여유가 `-5.1 mm`가 되어 안전하게 `FAILED`했고, 재시도는
완료됐습니다. 이번 Zigzag 획득 수정의 최종 LEFT run은 첫 시도에 위와 같이
완료됐습니다. RIGHT 재시도 2회는 모두 Parking gate 전의 기존 Intersection 실패로
끝나 주차 결과에 포함하지 않았습니다. 따라서 이 단계에서는 공식 LEFT 성공을
확인했지만 양쪽 분기 반복 성공과 차단봉·터널·결승선까지의 전체 완주는 아직 남아
있었습니다.

이 단계 코드의 주차 controller `67`개, Zigzag `35`개, 패키지 전체 `369`개 회귀는
각각 오류 없이 통과했습니다.

#### 출구 연속 좌회전·이동 중 인계 검증(2026-09-10)

기존 기록을 다시 분석한 결과 `TURN_TO_ZIGZAG 298.026 s → VERIFY_ZIGZAG_LANE
312.276 s`의 `14.250 s` 가운데 마지막 `8.653 s` 동안 선속도 명령이 0이었고,
작은 방위 오차만 천천히 줄이고 있었습니다. 반지름 `0.075 m` 원호를 변위
`0.150 m`, 접선 `0.055 m`인 zero-end-curvature quintic과 `0.120 m` 서쪽 정렬
tail로 바꾸고, tail의 유효 두 선 3 frame을 이동 중 확인해 차선 제어기에 바로
넘기도록 수정했습니다. 북쪽 접근과 좌회전은 양 끝 곡률 0 및 같은 실행 속도로
접속하며, 정지 확인 상태는 tail 끝까지 시야가 없는 경우에만 사용합니다.

정적 sweep에서 `LEAVE_AISLE` 선 여유는 `2.110 mm`, 좌회전+tail 선 여유는
`18.884 mm`, 고정 표지판 여유는 `29.603 mm`였습니다. 실제 texture의 raw 여유는
각각 `15.773 mm`, `14.149 mm`였고 좌회전은 footprint를 추가로 `11 mm` 확장해도
`2.517 mm`가 남았습니다. 최대 곡률은 기존 `13.333 1/m`보다 15.5% 낮은
`11.261 1/m`입니다.

공식 시작 자세에서 신호등 대기, 실제 카메라 방향 판독, Intersection, Obstacle,
Parking, Zigzag, Level Crossing, Tunnel을 모두 켜고 새 Gazebo 통합 주행을 양쪽
주차 분기에서 각각 수행했습니다.

- 우측 장애물·LEFT 주차 run `8ff9e6a6-ad1b-11f1-9bb4-22316c0401e8`은
  `LEAVE_AISLE 273.906 s → TURN_TO_ZIGZAG 289.689 s → JOIN_ZIGZAG 295.816 s
  → Parking COMPLETE 297.083 s`였습니다. 좌회전 상태는 `6.127 s`였고 전체
  `LEAVE_AISLE→COMPLETE`는 기존 `35.419 s`에서 `23.177 s`로
  `12.242 s(34.6%)` 짧아졌습니다.
- 좌측 장애물·RIGHT 주차 run `c089ec98-ad1c-11f1-9394-22316c0401e8`은
  `LEAVE_AISLE 267.988 s → TURN_TO_ZIGZAG 283.591 s → JOIN_ZIGZAG 289.672 s
  → Parking COMPLETE 290.942 s`였고 좌회전 `6.081 s`, 해당 전체 구간
  `22.954 s`였습니다.

원본 기록은 각각
`diagnostics/parking_zigzag_transition_20260910_continuous_left_official_run1/official_left.bag`과
`diagnostics/parking_zigzag_transition_20260910_continuous_right_official_run1/official_right.bag`입니다.

두 실행 모두 `VERIFY_ZIGZAG_LANE` 없이 이동 중 인계했습니다. LEFT/RIGHT의 마지막
주차 명령 `0.08172/0.08071 m/s`는 각각 `64/87 ms` 뒤 차선 제어 명령
`0.09689/0.09828 m/s`로 이어졌고, 좌회전 상태의 모든 123/122개 명령과 실제 odom
표본이 양의 선속도를 유지했습니다. 이어 Zigzag 첫 명령도 `0.10197/0.10232 m/s`로
이어졌으며 Zigzag는 각각 `16.635/16.675 s`, 최대 경로 오차 `7.5/8.5 mm`로
완료했습니다. RIGHT 기록의 실시간 진단 최소 여유는 `LEAVE_AISLE` 선 `1.652 mm`,
좌회전 선 `16.639 mm`, 동적 장애물 `7.711 mm`였습니다.

두 실행은 모두 Tunnel 뒤 `/mission/state=COMPLETE`까지 도달했습니다. 첫 Intersection
`ACTIVE`부터 전체 완료까지 각각 `298.085 s`, `296.239 s`였고 목표 구간에는 stale,
`FAILED`, unsafe, collision 또는 ERROR/FATAL 로그가 없었습니다. 기록 bag에는 별도
contact sensor가 없으므로 collision은 상태와 rosout 기준입니다. 당시 코드에서 주차
controller `67`개, Zigzag `35`개, 패키지 전체 `369`개 회귀도 오류 없이
통과했습니다.

### 지그재그 미션

주차 제어기가 westbound 진입부에서 `/parking/state=COMPLETE`를 발행하면 순서상 겹쳐 둔
거친 AMCL polygon이 다음 10 Hz 표본에서 `/mission/enable/zigzag`를 엽니다. 일반 차선
제어기는 이 짧은 인계 동안 `0.10 m/s`로 계속 주행합니다. 지그재그 제어기는
`ACQUIRING`에서 최신 `/mission/map_pose`와 최대 `40 ms` 차이인 `/odom` 표본을 이력에서
골라 입력 동기와 AMCL gate를 확인합니다. 고정 map x/y 시작 상자는 사용하지 않으며,
현재 odom 자세를 전체 측량 경로에 투영해 가장 가까운 진행 index부터 남은 suffix를
추종합니다. 경로 합류 오차 `18 mm`, 접선 방위 `7°`는 위치 gate가 아니라 현재 제어기가
안전하게 인수할 수 있는 추종 한계입니다. 인계 전에는 공통 swept validator가 전체
경로의 최신 LiDAR 장애물 여유를 검사합니다. 측량 기준선과 실제 픽셀 도색은 노드 시작
때 전체 경로를 한 번 검사하며, 강체 변환으로 보존되는 그 결과를 미션 진입 때 재사용합니다.
LiDAR sweep 동안 sensor callback은 계속 갱신되고, sweep을 마친 뒤 최신 odom으로 합류
가능 여부를 다시 확인한 다음에만 차선 제어권을 넘겨받습니다.

Gazebo의 `route/odom_aligned: true`에서는 texture 측량 경로와 raw world odom이 같은
좌표계이므로 AMCL 보정으로 경로를 이동하지 않습니다. 실물 프로필의 `false`에서는
동기화한 map/odom 한 쌍으로 실물 측량 경로를 odom에 한 번 고정합니다. 어느 모드든
`/control/lane_mission_handoff`를 호출할 때 인계 직전 `/cmd_vel`을 이어받습니다.

`config/zigzag_mission_gazebo.yaml`에는 본 경로용 22개 knot와 형상만 만드는 마지막 guide
tail knot 1개가 있습니다. 전체 C2 clamped cubic spline을 `2 mm` 간격으로 만든 다음
마지막 `0.125 m` guide tail은 주행 경로에서 잘라냅니다. 실제 주행 경로는 981개 표본,
길이 `1.895242 m`, 끝점 약 `(-1.392331, 1.748927, 177.34°)`, 최대 곡률
`5.0991 1/m`입니다. guide tail 덕분에 끝점 곡률은 약 `-0.2382 1/m`로 낮아져 직선 차선
제어로 자연스럽게 넘길 수 있습니다. 미션 중에는 AMCL 보정으로 경로를 다시 움직이거나
새 경로를 만들지 않고 고정된 odom 경로 하나만 추종합니다.

공통 `PathFollower`는 현재 경로 위치보다 `0.065 m` 앞의 목표점을 선택합니다.
lookahead feedback 곡률에 측량 경로의 목표점 곡률을 `0.25` 비율로 섞어 코너 방향을
선행 제어하고, 현재점부터 lookahead 목표까지의 속도 profile 최솟값을 사용합니다. 각 경로 표본의
속도는 최고 `0.14 m/s`, 각속도 `0.80 rad/s`, 횡가속도 `0.035 m/s²` 한도에서 곡률에
따라 정합니다. 이어 `0.12 m/s²` 감속 한도의 backward pass가 코너의 낮은 허용 속도를
앞쪽 표본으로 전파하므로 코너에 들어간 뒤가 아니라 진입 전에 감속합니다. 코너를 지난
뒤에는 `0.04 m/s²` 가속 한도로 다시 속도를 올립니다. 속도 profile은
`omega = velocity × curvature`의 변화도 계산해 계획 각가속도를 `0.80 rad/s²` 안으로
제한합니다. 실시간 감속이 걸리면 선속도와 각속도를 같은 비율로 줄여 감속 때문에 주행
곡률이 커지지 않게 하며, 조향이 아직 따라오지 못하면 선속도를 더 낮춥니다.

YAML의 sparse 곡선은 노란색·흰색 stripe를 선택하는 측량 기준이며, 실제 안쪽·바깥쪽
도색 경계는 Gazebo `course.png`의 픽셀 면적에서 추출합니다. 앞 `67.645 mm`, 뒤
`118.073 mm`, 반폭 `90.3 mm`인 비대칭 footprint의 모든 가장자리와 경로점 사이를
검사합니다. 이 footprint는 `x=-0.525` 부근의 날카로운 두 코너에서 안쪽 측량선을
침범하지 않고는 통과할 수 없으므로 `inner_paint_blocking: false`로 둡니다. 실제 차로
이탈을 나타내는 바깥 texture paint edge와 LiDAR 장애물 접촉은 공통 swept validator의
차단 조건으로 유지합니다. 현재 고정 spline은 이 정책으로 활성화되고 주행됩니다.

고정 경로는 활성화 전에 바깥 도색 경계까지 전체 sweep하고, 주행 중에는 공통 validator가
현재 자세부터 `0.10 s` 반응 구간과 완전 정지까지 비대칭 footprint를 연속 sweep합니다.
바깥 texture paint edge는 접촉(`0 m`)부터 unsafe이며, 위치추정·추종 오차는 확장된
footprint에 이미 포함하므로 추가 거리 margin을 중복 적용하지 않습니다. 첫 예상 접촉이
정지거리보다 앞이면 공통 속도 제한으로 감속하고, 현재 정지 sweep가 접촉하면 같은
`FOLLOWING` 상태에서 정지합니다. source stamp에 맞춘 실제 LiDAR 장애물점도 동일 검사에
포함합니다.

남은 경로가 `35 mm` 이하일 때 기존 카메라 누적값을 버리고 출구 차선 확인을 새로
시작합니다. 노란선·흰선이 함께 유효한 관측뿐 아니라 일반 차선 제어기와 같은 중심
offset을 적용한 한쪽 경계 관측도 사용할 수 있습니다. 남은 거리 `5 mm`, 끝점 위치 오차
`50 mm`, 방위 오차 `10°` 안에서 새 유효 영상 6개가 확인되면 먼저 일반 차선 제어에
`0.07 m/s` 제한으로 제어권을 넘깁니다. 영상 확인이
늦으면 `VERIFY_EXIT`에서 정지한 채 최대 2초 기다립니다. 인계 후 `JOINING_LANE`에서는
일반 차선 제어가 출구 방향으로 최소 `35 mm` 주행하고 새 유효 차선 영상 6개를 다시 확인해야
`0.30 m/s`를 복원하고 `/zigzag/state=COMPLETE`를 발행합니다. 합류 확인 제한은 1.5초입니다.

상태와 `/cmd_vel` 소유권은 다음과 같습니다. 출구 조건을 주행 중 이미 만족하면
`VERIFY_EXIT`는 생략될 수 있습니다.

```text
일반 차선 제어기 소유(0.10 m/s)  WAIT_GATE → ACQUIRING
                                                   ↓ cmd_vel 인계
지그재그 제어기 소유                            FOLLOWING
                                                   ↓
                                             VERIFY_EXIT
                                                   ↓ 0.07 m/s lane handoff
일반 차선 제어기 소유                         JOINING_LANE
                                                   ↓ 35 mm + 새 차선 확인
                                             COMPLETE (0.30 m/s)

입력·추종·출구 확인·handoff 실패 ─→ FAILED (정지)
도색 바깥 경계·LiDAR 침범 예상 ─→ 공통 validator 감속/정지
```

상태는 `WAIT_GATE`, `ACQUIRING`, `FOLLOWING`, `VERIFY_EXIT`, `JOINING_LANE`, `COMPLETE`,
`FAILED` 중 하나입니다. 입력은 `/mission/enable/zigzag`, `/mission/map_pose`, `/odom`,
`/detect/lane_boundaries`, `/control/manual_stop`, `/cmd_vel`이고, 제어 중에는 `/cmd_vel`과
`/control/max_vel`을 발행하며 `/control/lane_mission_handoff` 서비스로 단독 소유권을
인계합니다. 인계 응답이 불확실한 실패에서는 소유권 회수를 다시 요청하고,
회수 서비스도 실패하면 `/control/lane_following`으로 차선 제어를 정지합니다.
`/zigzag/diagnostics` 배열은 `PATH_FOLLOWING.md`의 공통 13개 값만 발행합니다.

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

지그재그만 시뮬레이션할 때는 전용 sequence와 주차 완료 뒤의 직선 시작 자세를 함께
사용합니다.

```bash
roslaunch custom_autorace_bringup gazebo.launch \
  gui:=false rviz:=false x_pos:=0.30 y_pos:=1.75 yaw_pos:=3.14159265 \
  odometry_source:=world wait_for_green:=false detect_signs:=false \
  intersection_mission:=false obstacle_mission:=false parking_mission:=false \
  zigzag_mission:=true tunnel_mission:=false \
  mission_zone_config:=$(rospack find custom_autorace_bringup)/config/mission_zones_zigzag_test_gazebo.yaml
```

현재 공통화 코드의 Zigzag 전용 Gazebo 회귀는
`ACQUIRING → FOLLOWING → VERIFY_EXIT → JOINING_LANE → COMPLETE`를 `25.956 s`에
통과하고 일반 차선 제어권을 반환했다. 최대 경로 오차는 `4.5 mm`, 최대 방향 오차는
`3.41°`, 공통 validator의 최소 line 여유는 `3.5155 mm`, 최소 LiDAR 장애물 여유는
약 `30.2 mm`였다. 별도의 이전 공통화 기준 공식 시작점 통합 주행에서는 앞선 Intersection,
Obstacle, Parking LEFT를 차례로 통과한 뒤 Zigzag가 `331.282→350.061 s`, 총
`18.779 s`에 `COMPLETE`했고 실제 `/safe_lane_controller`가 `/cmd_vel` 소유권을
회수했다. 공통화 대상 네 미션의 상세 통합 측정값은 `PATH_FOLLOWING.md`에 기록한다.

### 라이다 차단봉 미션

`level_crossing_lidar_controller.py`는 순차 AMCL gate
`/mission/enable/level_crossing`이 열린 동안에만 `/scan_mid360_raw`를 검사합니다.
각 점의 방향은 scan 배열 위치를 정면으로 가정하지 않고
`angle_min + index × angle_increment`로 계산합니다. 유효 거리의 점을 `base_scan`
전방 좌표로 투영한 뒤, 차로 폭 방향으로 넓고 진행 방향 두께가 얇은 연속 군집만
차단봉으로 인정합니다. 따라서 같은 구역의 폭 `0.12 m` 고정 정지 표지판은 현재
Gazebo 설정의 최소 폭 `0.18 m`를 통과하지 못합니다.

상태는 `WAIT_GATE → APPROACH → STOPPED → PASSING → COMPLETE` 순서입니다. 시작할 때
봉이 보이지 않는 scan은 개방으로 처리하지 않습니다. 닫힌 봉을 3개 scan에서 연속
확인한 뒤 `/control/lane_mission_handoff`로 차선 제어를 먼저 끄고 즉시 0 속도를
발행합니다. 정지 중에는 0 속도를 계속 유지하며, scan이 오래되거나 끊기면 출발하지
않습니다. 봉이 사라진 상태를 5개 scan에서 연속 확인하면 차선 제어권을 반환하여 즉시
주행하고, AMCL이 차단봉 polygon 바깥 여유를 확인하면 `COMPLETE`가 됩니다.

검출 거리·폭·군집 간격·닫힘/열림 확인 횟수와 scan timeout은
`config/level_crossing_mission_gazebo.yaml`에 있습니다. `/level_crossing/diagnostics`의
배열 순서는 `[검출 여부, 전방 거리, 횡방향 폭, 진행방향 두께, 점 수, 닫힘 확인 수,
열림 확인 수, 유효 scan 점 수]`입니다. 유효 거리점이 설정 개수보다 적은 scan은 개방
확인에 사용하지 않으며, 통과 중 봉이 다시 내려오면 같은 닫힘 확인 뒤 다시 정지합니다.
`/control/manual_stop` 중에는 확인 횟수와 상태 전이를 멈춥니다. 실물에서는 Gazebo 값을
복사하지 않고 실제 Mid-360 장착 상태와
차단봉 폭·정지거리를 측정한 별도 YAML을 사용해야 합니다.

Gazebo의 센서는 실제 Mid-360 점군과 달리 센서 높이의 수평 ray 한 층만 만듭니다.
원본 차단봉 asset의 중복 pose를 제거했고, 닫힌 모델에는 실제 `PointCloud2 → LaserScan`
높이 구간 집계를 모사하는 얇은 비가시 LiDAR collision을 추가했습니다. 렌더링되는 봉과
실제 충돌 형상은 기존 `0.30 × 0.02 × 0.05 m` 크기를 유지합니다.

2026-09-08 반복 시험에서는 기존 최대 검출 거리 `0.60 m`가 Gazebo 차단봉 개방
타이머의 위치 조건보다 먼저 정지를 확정하는 것을 확인했습니다. 이전 공식 주행 3회의
세 번째 닫힘 확인 위치는 world x `-1.505, -1.482, -1.480 m`였지만 타이머는
`x > -1.45 m`에서만 시작하므로, 정지 뒤 시간이 지나도 봉이 열리지 않을 수 있었습니다.
Gazebo 프로필의 `maximum_forward_distance`를 `0.45 m`로 줄여 이 조건 안쪽까지
접근하도록 조정했습니다. 이 값은 실물 프로필에 적용하지 않습니다.

수정 뒤 공식 시작점에서 `parking_obstacle_x:=0.23`과 `0.73`을 각각 고정한 새 통합
주행은 모두 교차로, 장애물, 주차, 지그재그를 순서대로 마친 뒤 차단봉까지
`COMPLETE`했습니다. 왼쪽 빈 칸 주행은 세 번째 검출이 `305.727 s`, world
`x=-1.368 m`, 거리 `0.373 m`였고 `305.778 s`에 정지했습니다. 닫힌 `up_bar`가
`315.921 s`에 실제로 사라진 뒤 `316.379 s`에 출발하고 `320.975 s`에
완료했습니다. 오른쪽 빈 칸 주행은 세 번째 검출이 `305.910 s`, world
`x=-1.373 m`, 거리 `0.367 m`였고 `305.941 s`에 정지했습니다. `up_bar`가
`316.089 s`에 사라진 뒤 `316.540 s`에 출발하고 `321.237 s`에 완료했습니다.
두 주행의 위치 트리거 여유는 약 `77~82 mm`였고, 정지 뒤 이동까지 반영한 로봇 전면과
봉의 여유는 약 `0.25 m`였습니다. 정지 구간의 `/cmd_vel` 212개와 213개는 모두
차단봉 제어기 한 노드가 발행한 0 속도였습니다. 별도의 첫 통합 시도 1회는 앞선 교차로
방향 표지 탐색 제한시간 초과로 차단봉에 도달하지 못했으므로 차단봉 성공 횟수에
포함하지 않습니다.

차단봉 직전 명목 자세를 새 Gazebo에서 2회 더 반복한 보조 회귀도 모두
`APPROACH → STOPPED → PASSING → COMPLETE`였습니다. 세 번째 검출 거리는
`0.337 m`와 `0.325 m`였고, 두 실행 모두 실제 `up_bar` 제거 전 출발은 0건이며 제거
약 `0.49~0.55 s` 뒤 5개 개방 scan을 확인하고 출발했습니다.

거리 변경은 사선에서 닫힌 봉을 잠깐 놓치는 별도 문제를 해결하지 않습니다. 강제로
`y=1.17 m, yaw=+10°`에서 시작한 전용 회귀에서는 가까운 ROI 경계에 군집 폭이 잘려
실제 개방 전 `PASSING`이 재현됐습니다. `y=1.33 m, yaw=-10°` 조건도 이전 반복에서
군집 진행방향 두께가 `0.09 m`를 넘으며 같은 오류가 재현됐고, raw scan 재판정에서
`0.45 m` 변경의 영향이 없었습니다. 정상 공식 접근 2회에서는 발생하지 않았으므로
현재 주 경로는 LiDAR 단독으로 유지하고 카메라 융합은 추가하지 않았습니다. 공식 접근
분산에서 이 오류가 재현되면 정지 판정의 필수 AND 조건이 아니라, 닫힌 빨강·흰색 봉이
영상에 남아 있는 동안 출발을 막는 veto로 검증해야 합니다.

차단봉 직전 원인 분석용 회귀는 다음처럼 실행합니다. 이 시작점 시험은 중간 검증이며,
최종 완료 판정에는 공식 시작점에서 앞선 미션을 순서대로 통과한 통합 주행이 필요합니다.

```bash
roslaunch custom_autorace_bringup gazebo.launch \
  gui:=false rviz:=false x_pos:=-1.40 y_pos:=1.25 yaw_pos:=0.0 \
  odometry_source:=world wait_for_green:=false detect_signs:=false \
  mission_models_initial_state:=6 \
  intersection_mission:=false obstacle_mission:=false \
  parking_mission:=false zigzag_mission:=false level_crossing_mission:=true \
  tunnel_mission:=false \
  mission_zone_config:=$(rospack find custom_autorace_bringup)/config/mission_zones_level_crossing_test_gazebo.yaml
```

### 터널 미션

`tunnel_mission_controller.py`는 차선 영상이 사라지기 전의 순차 AMCL gate
`/mission/enable/tunnel`에서 시작합니다. 입구에서는 AMCL 자세와 같은 시각의 odom
자세로 `map <- odom` 변환을 한 번 계산해 고정합니다. 이후 제어 자세는 이 고정 변환에
새 odom을 투영해 구하며, 터널 안의 미등록 장애물이 AMCL scan matching을 끌어도
미션 좌표계는 움직이지 않습니다. 이후의 AMCL 자세는 제어에 사용하지 않고 고정
anchor와의 위치·방위 차이만 진단으로 기록합니다.

제어권을 받은 뒤에는 현재 자세에서 측량한 staging pose까지의 zero-end-curvature
connector와 입구 안쪽까지의 직선을 하나의 frozen `CommonPath`로 만들고, 실제 비대칭
footprint가 벽과 포털을 통과하는지 확인한 뒤 저속으로 연속 추종합니다. 입구 뒤쪽까지
완전히 들어간 후 새 내부 LiDAR scan 3개를 받아야 Hybrid A* 계획을 시작합니다. 이
`CommonPath` 사용은 자료형 재사용이며 Tunnel의 추종기·costmap 충돌 검사·재계획은
미션 전용 구현입니다.

터널 제어에서 AMCL의 역할은 사전에 측량한 입구와 고정 벽에 대한 입구 전역
위치추정까지입니다.
AMCL은 LiDAR 장애물을 발견해 지도를 만들어 가는 모듈이 아니며, 실행 중 `/map`을
수정하지 않습니다. 전체 Gazebo 정적 지도에는 다른 구간의 고정 구조도 있지만, 터널
내부 planning layer는 장애물 위치를 넣지 않은 벽-only 지도입니다. Gazebo world에
있는 원통 세 개도 PGM에서는 의도적으로 제외해 매 주행 전에 위치를 모르는 조건을
재현합니다.

같은 `/scan_mid360_raw`는 AMCL과 별개로 터널 제어기의 동적 costmap에 투영됩니다.
각 scan의 측정 timestamp에 맞는 `odom <- scan_frame` TF를 조회하고, 입구에서 고정한
`map <- odom`과 합성해 endpoint를 map 좌표로 변환합니다. 고정 벽 layer는 바꾸지 않고
ray별 free 공간과 endpoint 장애물만 별도 layer에 mark/clear하며, Gazebo 설정에서는
같은 관측을 2회 확인한 cell만 변경합니다. 확인된 endpoint는 값 `100`인 치명 셀로
유지합니다. 그 주위 `60 mm`는 `64/43/21`의 세 단계 회피 비용으로만 표시하며, 정적
벽에는 이 띠를 적용하지 않습니다. 이 값들은 occupied threshold `65`보다 작아서 새로
보인 원통 측면의 여유 띠가 정지한 로봇 아래 생겨도 현재의 안전한 시작 자세를 막지
않습니다. Gazebo에서는 공식 출발 주행에서 확인한 AMCL·격자 오차를 반영해 정적 셀에서
`100 mm` 이내인 endpoint를 고정 벽의 중복 반사로 제외합니다. 가장 가까운 원통 표면과
정적 layer의 실측 간격은 `440 mm`여서 이 제외 범위 밖에 남습니다. gate가 열린 뒤 최소
3개 scan으로 초기 layer를 확인하고 이 두 layer를 한
`/tunnel/costmap`에 발행합니다. 따라서 이것은 온라인 장애물 회피용 지도 갱신이지
AMCL 지도 생성이나 SLAM이 아닙니다.

계획기는 입구 안쪽의 현재 `(x, y, yaw)`부터 동쪽 포털 내부의 고정 staging 목표
`(-0.180, -1.748373, 0°)`까지 전진 전용 Hybrid A*를 실행합니다. 빈 공간에서는
직선에 가까운 최단 경로를 선호하고, 직선 사이에 동적 장애물이 있으면
직진과 좌·우 각 2단계 곡률의 5개 primitive를 조합해 우회합니다. 후진, 제자리회전 또는
미리 저장한 장애물 우회 경로는 사용하지 않습니다. 각 primitive 전체에서 앞
`67.645 mm`, 뒤 `118.073 mm`, 반폭 `90.3 mm`, padding `10 mm`인 비대칭
직사각형 footprint를 방향까지 포함해 sweep합니다. 좁은 출구를 원형 반경으로
inflation하면 통로를 잘못 닫을 수 있으므로 static 원형 inflation은 사용하지 않습니다.
위의 동적 `60 mm` 비용은 각 primitive의 `20 mm` 이하 간격 전체 sweep에서 패딩된 실제
footprint가 접촉한 최대 단계값을 거리 적분해 경로 비용에 더합니다. endpoint 자체와
static/unknown 셀은 계속 치명 장애물이므로 비용과 맞바꿔 통과할 수 없습니다. 입구·터널
내부·출구 포털의 YAML keep-in 영역은 로봇 기준점에만 적용하여 북동·남서 외벽 바깥
우회를 막고, AMCL 지도나 동적 costmap에는 가상의 벽을 추가하지 않습니다. 계획 결과는
`/tunnel/path`에 발행합니다.

주행 중 LiDAR layer가 바뀔 때마다 현재 위치부터 출구까지 남은 경로 전체를 다시
검사합니다. sampled pose만 확인하지 않고 각 primitive 사이를 비대칭 직사각형
footprint로 연속 sweep하므로, 한 번 관측된 먼 장애물이 이후 안정된 cell로 남아 있어도
누락하지 않습니다. 새 치명 셀이 남은 경로와 겹치면 즉시 정지합니다. `21/43/64`
안전띠는 계획에 사용한 layer를 baseline으로 보관한 뒤, 이후 값이 증가한 cell만 exact
footprint로 검사합니다. 최초 접촉 station을 저장해 매 제어 주기 현재 station과
비교하고, 남은 경로거리 `0.50 m` 안으로 들어오면 먼저 0 속도로 정지한 뒤
`PLANNING`으로 돌아가 현재 자세에서 Hybrid A*를 다시 실행합니다. 따라서 멀리 보인
장애물이 한동안 같은 cell로 남아 있어도 접근 시점을 놓치지 않고, 이미 계획 비용에
반영한 안전띠 때문에 scan마다 다시 계획하지도 않습니다. 정지 중 이미 생긴 안전띠
안에서 새 계획을 시작해야 할 때는 현재 footprint와 겹친 새 증가분의 연속 시작
prefix를 완전히 벗어나는 것까지만 허용합니다. 기록된 정상 탈출 최대 `0.320 m`에
`30 mm`를 더한 `0.35 m`를 넘거나, 경로 끝까지 한 번도 벗어나지 못하거나, 한 번
벗어난 뒤 새 증가분에 다시 닿는 경로는 채택하지 않습니다. 안전띠는 치명 장애물이
아니라 아직 가려진 표면을 위한 조기 재계획 신호입니다.

`FOLLOWING`의 각 제어 주기에는 odom으로 측정한 실제 속도와 다음에 요청할 속도를 각각
현재 자세에서 고정 반응시간만큼 진행시킨 뒤, 설정된 선속도·각속도 감속 한계로 완전히
멈출 때까지 footprint sweep을 검사합니다. 정지한 최종 자세에서는 추가 `10 mm` 직선
공간 여유도 검사합니다. 둘 중 하나라도 충돌 없이 정지할 수 없으면 해당 주행 명령을
내지 않습니다.

Hybrid A*가 내부 staging 목표에 도달하면 `ALIGNING_EXIT`에서 선속도 0으로 출구 방위를
`1°` 안까지 맞춥니다. 이 각속도도 현재 측정 운동과 요청 운동의 반응·완전정지
footprint sweep을 모두 통과한 경우에만 발행합니다. 이어 그 실제 횡위치를 보존한
측량 직선 출구를 frozen `CommonPath`로 만들고, footprint 뒤쪽이 동쪽 포털을
`10 mm` 이상 완전히 지난 뒤 정지합니다. `VERIFY_EXIT`는 일반 차선 제어기가 이미
생성·검증한 공통 13항목 진단에서 유효 경로와 양수 line 여유를 새 표본 6개로 확인한
뒤 제어권을 인계합니다. `JOINING_LANE`은 40 mm 이상 주행하고 같은 안전 진단을 새로
6개 확인해야 완료합니다.

상태와 `/cmd_vel` 소유권은 다음과 같습니다.

```text
일반 차선 제어기 소유              WAIT_GATE → ACQUIRING (0.04 m/s 제한)
                                                ↓ cmd_vel 인계 후 즉시 정지
터널 제어기 소유                         ALIGNING_ENTRY → ENTERING
                                                ↓ 입구 완전 통과 + fresh 내부 scan 3개
                                              PLANNING
                                                ↓
                                              FOLLOWING (최대 0.075 m/s)
                                                ↓
                                              ALIGNING_EXIT (정지·1° 방위 정렬)
                                                ↓
                                              EXITING (측량 직선 CommonPath)
                                                ↓ rear portal 여유 10 mm + 공통 lane 진단 6개
                                              VERIFY_EXIT
                                                ↓ cmd_vel 인계
일반 차선 제어기 소유                         JOINING_LANE (0.06 m/s)
                                                ↓ 40 mm 주행 + 공통 lane 진단 6개 재확인
                                              COMPLETE (0.30 m/s)

지도·pose·odom·scan·계획·추종·출구 확인·handoff 실패 ─→ FAILED (정지)
```

상태는 `WAIT_GATE`, `ACQUIRING`, `ALIGNING_ENTRY`, `ENTERING`, `PLANNING`,
`FOLLOWING`, `ALIGNING_EXIT`, `EXITING`, `VERIFY_EXIT`, `JOINING_LANE`,
`COMPLETE`, `FAILED` 중 하나입니다. 입력은 `/map`,
`/scan_mid360_raw`, `/mission/map_pose`, `/odometry/filtered`,
`/mission/enable/tunnel`, `/control/lane_path_diagnostics`, `/detect/signs`,
`/control/manual_stop`입니다. 출력은 `/cmd_vel`,
`/control/max_vel`, `/tunnel/state`, `/tunnel/path`, `/tunnel/costmap`,
`/tunnel/diagnostics`이며 `/control/lane_mission_handoff` 서비스로 차선 제어기와
단독 소유권을 인계합니다. 터널 경고판 검출은 gate를 대신하지 않고
`sign_seen/confidence` 진단으로만 기록합니다. `/mission/clear/tunnel`은 mission zone
manager가 발행하는 구역 진단 신호이며 터널 제어기는 구독하지 않습니다. 출구 확인은
고정한 map 좌표의 rear-portal 여유와 일반 차선의 최신 공통 경로 진단을 함께 사용합니다.

`/tunnel/diagnostics` 배열은 `[미션 경과시간, 마지막 계획시간, 확장 node 수,
costmap version, scan 갱신 수, 경로 index, 남은 거리, 위치 오차, 방위 오차(deg),
계획 시도 수, 재계획 수, 터널 경고판 검출 여부, 경고판 최고 confidence,
현재 AMCL과 고정 anchor의 위치 차이(m), 방위 차이(deg)]`의 15개 값 순서입니다.
다음 토픽으로 gate부터 차선 복귀까지 확인합니다.

```bash
rostopic echo /mission/enable/tunnel
rostopic echo /mission/inside/tunnel
rostopic echo /mission/clear/tunnel
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

터널 입구부터 출구 뒤 차선 복귀까지만 확인하는 Gazebo 전용 회귀는 다음과 같습니다.

```bash
roslaunch custom_autorace_bringup gazebo.launch \
  gui:=false rviz:=true x_pos:=-1.75 y_pos:=0.18 yaw_pos:=-1.57079632679 \
  odometry_source:=world wait_for_green:=false mission_models:=false \
  intersection_mission:=false obstacle_mission:=false parking_mission:=false \
  zigzag_mission:=false level_crossing_mission:=false tunnel_mission:=true \
  mission_zone_config:=$(rospack find custom_autorace_bringup)/config/mission_zones_tunnel_test_gazebo.yaml
```

이 명령은 터널 직전 자세와 tunnel-only sequence를 사용하는 원인 분석·중간 회귀입니다.
앞선 미션과 터널 경고판 접근을 건너뛰므로 여기서 `COMPLETE`가 되어도 터널 미션의
최종 완료 판정은 아닙니다. 최종 판정은 새 Gazebo와 관련 노드를 공식 시작 자세에서
실행한 통합 주행으로 수행했습니다. 교차로, 장애물, LiDAR가 선택한 주차면, 지그재그,
차단봉을
순서대로 통과한 뒤 터널 진입부터 출구 정렬, 새 차선 확인과 `/cmd_vel` 반환까지 사람
개입 없이 `COMPLETE`가 됐습니다. 이전 `run15`의 터널 구간은 `73.371 s`, 계획 2회,
재계획 1회였습니다. 현재 기준인 `run23`의 터널 구간은 `73.117 s`였고 완료 시 차선
제어기가 `/cmd_vel`을 소유했습니다. 같은 주행은 이후 결승선 footprint까지 통과했고
최종 `result.json`의 통합 판정 `22/22`를 만족했습니다.

## 통합 표지판 인식

표지판 영상은 `sign_detector.py` 한 노드만 구독합니다. 검출 결과는 공통 메시지
`custom_autorace_bringup/TrafficSign`으로 `/detect/signs`에 발행하며, 교차로
컨트롤러는 필요한 `sign_type`을 선택해 이 typed 토픽을 직접 구독합니다. 주차
제어기는 이 토픽을 구독하지 않고 AMCL 순차 gate와 정지 판정 자세의 LiDAR 지도 ROI
투영을 사용하며, 좁은 주차·출차 경로는 local odom에 고정합니다.
`detect_signs`, `intersection_mission`, `obstacle_mission`, `parking_mission`,
`zigzag_mission`, `level_crossing_mission`, `tunnel_mission` launch 인자는 서로 독립적이므로 개별 미션
컨트롤러를 꺼도 공통 표지판
인식은 계속 실행할 수 있습니다. `detect_signs:=false`여도 AMCL gate를 사용하는 주차,
지그재그, 차단봉과 터널 미션의 시작과 수행에는 영향이 없지만, 터널 경고판 검출
진단은 기록되지 않습니다.

```text
/camera/color/image_raw
  -> sign_detector.py
  -> /detect/signs
       |- INTERSECTION_WARNING -> 검출·진단용
       |- DIRECTION + LEFT/RIGHT -> intersection_mission_controller.py
       |- CONSTRUCTION_WARNING   -> 검출·진단용(장애물 gate는 AMCL polygon)
       |- TUNNEL_WARNING         -> tunnel_mission_controller.py 진단
       |- PARKING                -> 검출·진단용(주차 gate는 AMCL polygon)
       `- LEVEL_CROSSING_WARNING -> 검출·진단용(차단봉 gate는 AMCL polygon)
```

현재 교차로 경고판, 좌·우 방향지시판, 공사, 터널, 철도건널목 표지판은 ROBOTIS
기준 이미지와 비교합니다. 컨테이너의 OpenCV 4.2에는 원본 검출기가 사용한
SIFT가 없으므로 ORB 특징점 매칭과 RANSAC homography로 크기·원근 변화에 대응합니다.
방향지시판이 가까워져 원형 테두리가 화면 위로 잘려도 판독할 수 있도록, 같은 통합
노드에서 파란 원형 영역과 흰 화살표의 좌우 픽셀 비대칭도 함께 검사합니다. 관련 HSV,
크기, 원형도와 비대칭 임계값은 `config/sign_detector.yaml`의
`classifiers/direction_geometry`에서 조정합니다.
Gazebo 주차 표지판은 현재 world가 사용하는 정확한 texture를 `package://` URI로
불러오며, 1% polygon 근사에서 꼭짓점 8개 미만인 사각 패널은 방향 표지로 분류하지
않습니다. 해당 Gazebo asset이 없으면 검출기는 명시적으로 초기화에 실패합니다.
`stop.png`는 철도건널목 미션의 진입 표지판으로 매핑됩니다. 템플릿 파일과 종류별
최소 매칭 수는 `config/sign_detector.yaml`의 `classifiers/template/entries`에서
관리합니다. 새로운 표지판은 같은 위치에 템플릿을 등록하고 `TrafficSign.msg`에
종류를 추가하며, 별도 카메라 구독 노드를 만들지 않습니다.

개별 센서만 시험할 수 있습니다.

```bash
roslaunch custom_autorace_bringup hardware.launch start_opencr:=false start_lidar:=false
roslaunch custom_autorace_bringup hardware.launch start_opencr:=false start_camera:=false
```

카메라 intrinsic과 원본 `turtlebot3_autorace_camera`의
`image_projection`/`image_compensation`은 실제 장착이 끝난 후 다시 보정해야 합니다.
`config/intersection_mission.yaml`, `config/obstacle_mission_gazebo.yaml`,
`config/parking_mission_gazebo.yaml`,
`config/zigzag_mission_gazebo.yaml`, `config/level_crossing_mission_gazebo.yaml`,
`config/tunnel_mission_gazebo.yaml`도
시뮬레이션 전용입니다. 실물에서는 코스
방위·카메라 픽셀/거리 비율·경계 중심, Mid-360 외부 파라미터, 방향성 footprint와
padding, 속도·가속도 한계를 D405·Mid-360·OpenCR 장착 상태에서 따로 측정한 YAML로
교체해야 합니다. Mid-360 scan timestamp와 odom TF의 시간 동기 및 주행 중 scan
motion distortion 영향도 실물 속도에서 따로 확인해야 합니다. Gazebo 지도·AMCL
파라미터·mission polygon도 실제 경기장용 파일을
사용해야 하며, 교차로 관찰창·고정 진입/탈출 곡선·도색 경계, 주차 접근·정지 checkpoint,
주차면 ROI, 지그재그 spline과 양쪽 도색 경계, 차단봉 gate와 LiDAR ROI, 터널의 벽-only
static layer·입구·출구를 같은 실물 지도에서 측량해야 합니다. 터널 안의 위치가 바뀌는
장애물은 이 static layer에 넣지 않습니다.
시뮬레이션 값을 그대로 복사하면 안 됩니다.

Gazebo bird-eye 영상의 수평 중심은 `config/gazebo_projection.yaml`의
`output_shift_x`로 조절합니다. 양수는 투영 결과를 오른쪽으로, 음수는 왼쪽으로
평행이동합니다. `center_x`는 원본 사다리꼴의 원근 형상을 결정하므로 수평 중심을
맞추는 용도로 변경하지 않습니다. YAML을 저장한 뒤 `gazebo.launch`만 재실행하면
적용되며 다시 빌드할 필요는 없습니다.

## 실물 확인 순서

```bash
roslaunch custom_autorace_bringup hardware.launch \
  start_opencr:=false start_lidar:=false
rostopic hz /camera/color/image_raw
rostopic hz /camera/image_rect_color
rostopic hz /camera/image_rect_color/compressed
rostopic hz /camera/image_projected
rostopic hz /camera/image_projected_compensated
rostopic hz /detect/lane_centerline
rostopic hz /detect/image_traffic_light/compressed
rostopic hz /detect/signs

roslaunch custom_autorace_bringup livox_mid360.launch
rostopic hz /livox/lidar
rostopic echo -n 1 /scan_mid360_raw
```

`/scan_mid360_raw`는 원래 각도 메타데이터와 range 순서를 유지합니다. 각 range의
방향은 `angle_min + index × angle_increment`로 계산하며 배열을 재정렬하지 않습니다.
