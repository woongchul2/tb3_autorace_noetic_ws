# Custom AutoRace bringup

OpenCR, Intel RealSense D405, Livox Mid-360을 사용하는 ROS Noetic AutoRace
통합 패키지입니다. Ubuntu 24.04 호스트에서는 저장소 루트의
`compose.noetic.yaml`로 Ubuntu 20.04/ROS Noetic 컨테이너를 실행합니다.

실행·빌드·회귀·토픽 확인 명령은 저장소 루트의
[`DOCKER_NOETIC.md`](../../DOCKER_NOETIC.md)를 단일 기준으로 사용합니다. 이 문서는
현재 런타임 구조와 미션별 책임만 설명합니다.

## 현재 검증 상태

현재 작업트리는 연결 직선의 절대 길이와 AMCL polygon에 의존하지 않는
mission-local adaptive registration으로 전환되었습니다. 2026-09-18 `run20`은 변경 후
새 Gazebo와 통합 launch를 시작하고 공식 자세 `(0.800, -1.747, 0°)`에서 출발해
`Intersection → Obstacle → Parking → Zigzag → Level Crossing → Tunnel`을 순서대로
완료하고 결승선 footprint를 통과했습니다.

| 항목 | 결과 |
|---|---|
| 미션 결과 | 6개 모두 `COMPLETE`, `FAILED` 0건 |
| 출발부터 결승선 | `283.053 s` |
| 선택 조건 | Intersection LEFT, Parking LEFT, Tunnel layout B |
| 미션 수행시간 | `25.972 / 35.125 / 57.886 / 16.222 / 14.386 / 75.283 s` |
| Parking 후진 | `6.303 s`, 최저 명령 `-0.1034 m/s` |
| 제어권 | 기대한 순서로 인계, `/cmd_vel` 발행자 교차 0건 |
| 결승 자세 | `(1.0335, -1.7459, 0.032°)`, footprint 전체 허용 구간 안 |

`run20` bag 분석은 22개 중 19개 항목을 통과했습니다. 나머지 3개는 주행 실패가 아니라
recorder를 `/use_sim_time` 설정 전에 시작해 생긴 기록 시각 burst와 첫 pose 지연,
`/detect/lane_centerline` 녹화 누락입니다. 실제 연속 pose의 최대 이동은 `0.283 mm`였고,
미션 순서·진단·결승·제어권 검사는 모두 통과했습니다. 두 번째 공식 시작 주행은
Intersection RIGHT, Parking RIGHT, Tunnel layout C 조건에서 6개 미션과 결승을 다시
완료했습니다. 이 두 번째 bag은 출발 뒤 녹화를 시작했으므로 공식 시작 계측값으로
사용하지 않습니다.

현재 ROS Noetic 자동 회귀는 등록된 29개 bringup test 파일의 `669/669`와 description
`7/7`, 합계 `676/676`이 실패·오류 없이 통과했고 두 ROS 패키지 build도 통과했습니다.
실제 장착 D405 원근 보정, 실제 도색의 30 Hz 차선 주행, Mid-360 motion distortion,
OpenCR 응답과 실물 공식 시작점 통합 주행은 아직 검증하지 않았습니다.
미션별 오차와 안전 여유는 [공통 경로 문서](PATH_FOLLOWING.md), 과거 구현과 속도
비교는 [검증 이력](docs/VALIDATION_HISTORY.md)에 분리해 기록합니다.

## 런타임 구조

신호등은 순차 미션 전에 출발을 허가하고, 일반 구간과 Intersection 반원은 카메라가
매 프레임 만든 rolling `CommonPath`를 추종합니다. Intersection, Obstacle, Parking,
Zigzag의 고정·측량 경로도 같은 `CommonPath`, `SweptFootprintValidator`,
`PathFollower`를 사용합니다.

```text
다음 미션 arm
  → 해당 미션 센서로 mission-local 경로 선택·정렬
  → CommonPath 생성
  → 현재 실행할 CommonPath 전체 swept-footprint 검증
  → source stamp와 세대 번호가 맞는 ready
  → enable 뒤 공통 PathFollower 실행
  → 완료 후 일반 차선 제어권 반환
```

연결 직선이 짧아지거나 길어져도 `arm` 상태에서 일반 차선 주행을 계속하고, 각 미션의
직접 관측 가능한 형상으로 미션 좌표계를 odom에 맞춘 뒤 실제 진입 창에서만 제어권을
넘깁니다. 등록과 전체 경로 검증이 끝나기 전에는 미션 제어기가 속도 제한이나
`/cmd_vel`을 소유하지 않습니다.

```text
카메라 신호등 출발
  → 일반 차선 rolling CommonPath
  → Intersection
  → Obstacle
  → Parking
  → Zigzag
  → Level Crossing
  → Tunnel
  → 결승선
```

활성 제어기 하나만 `/cmd_vel`을 직접 발행합니다. 별도 relay는 없으며
`/control/lane_mission_handoff` 서비스로 일반 차선 제어기와 미션 제어기가 소유권을
주고받습니다. `/control/lane_following`은 수동 정지·재개 서비스이고,
`/control/manual_stop`이 현재 미션 제어기에도 정지 상태를 전달합니다.

| 구간 | 경로 결정 | 정렬·기준 frame | 실행 주체 |
|---|---|---|---|
| 일반 차선 | 현재 카메라 중심선 | 촬영 시각 odom에 고정 | 일반 차선 제어기 + 공통 추종기 |
| Intersection | 카메라 LEFT/RIGHT | source stamp의 AMCL `map→odom` 방향과 방향 표지 거리·베어링으로 `intersection_local→odom` 고정 | 입·출구는 미션 제어기, 반원은 일반 차선 제어기 |
| Obstacle | 측량된 단일 spline | 서로 독립인 LiDAR 장벽 면으로 `obstacle_local→odom` 고정 | 미션 제어기 + 공통 추종기 |
| Parking | LiDAR로 빈 좌·우 공간 선택 | 고정 주차 표지의 비평행 면으로 `parking_local→odom` 고정 | 미션 제어기 + 공통 전진·후진 추종기 |
| Zigzag | 측량된 단일 spline | rolling 카메라 곡선으로 station·SE(2) 확인; Gazebo는 검증된 identity | 미션 제어기 + 공통 추종기 |
| Level Crossing | LiDAR 차단봉 판정 | source-stamped 고정 landmark로 crossing plane 고정 | 전용 정지·재출발 제어기 |
| Tunnel | LiDAR 동적 costmap + Hybrid A* | 종방향 벽·직교 입구벽·끝점으로 tunnel template 고정 | 전용 계획·추종·충돌 검사 |

공통 경로 자료형, 속도 제한, 완료 판정과 직사각형 sweep의 세부 인터페이스는
[`PATH_FOLLOWING.md`](PATH_FOLLOWING.md)에 있습니다.

## 센서와 위치추정

### 카메라

D405 원본 `1280×720@30 Hz`는 한 어댑터에서 중앙 4:3 crop과 `320×240` 축소를 한 뒤
raw와 compressed 출력으로 나뉩니다. 차선 투영·보상은 raw 연결을 사용하고, 신호등과
보정 도구는 compressed 연결을 사용합니다. 차선 검출기는 한 HSV mask에서 중심선과
노란선·흰선 경계를 함께 계산하므로 제어기마다 같은 영상을 다시 분리하지 않습니다.

Gazebo 투영은 원본 영상의 위쪽을 `y=176`에서 `y=110`까지 확장해 bird-eye 기준
약 `0.60 m` 전방까지 표본화합니다. 이 값은 Gazebo 카메라 전용입니다. 장착 위치와
렌즈가 다른 실제 D405에는 `d405_projection_uncalibrated.yaml`을 보정한 별도 파일이
필요합니다.

`run21`의 `/detect/lane_centerline` header 기준 전체 발행은 8,278건/281.537초,
`29.399 Hz`였고 중앙 간격 `33 ms`, p95 `35 ms`, 최대 `70 ms`, 중복·역행 stamp 0건이었습니다.
경계가 하나 이상 유효한 메시지는 7,712건, `27.389 Hz`였으며 명시적 empty 관측은
566건이었습니다. 이는 Gazebo의 30 Hz 처리 주기 검증이며 실제 D405 검증을 대신하지
않습니다.

### LiDAR

Mid-360 `PointCloud2`는 `/scan_mid360_raw`로 변환됩니다. 미션 제어기는 LaserScan의
`angle_min`과 `angle_increment`를 그대로 사용하며 정면 인덱스 기준으로 배열을
재정렬하지 않습니다.

### Odometry와 AMCL 진단

통합 Gazebo launch는 후진 부호를 보존하기 위해 signed world pose/twist를 EKF 입력으로
쓰고, EKF가 `/odometry/filtered`와 유일한 `odom → base_footprint` TF를 발행합니다.
AMCL은 정적 지도와 scan으로 `map → odom`을 보정합니다. 미션 매니저가 polygon이나
AMCL 좌표로 enable gate를 열지는 않습니다. `/mission/map_pose`와 Gazebo polygon
신호는 위치추정 비교, RViz와 회귀 계측용 진단일 뿐입니다.

AMCL은 동적 장애물을 지도에 추가하는 SLAM이 아닙니다. 페인트와 위치가 바뀌는 장애물은
정적 지도에서 제외하며, Obstacle·Parking·Tunnel 제어기가 실시간 LiDAR로 별도
검사합니다. 주차 후진의 짧은 변위를 보존하기 위해 workspace AMCL은 `diff-signed`
odom 모델을 사용합니다.

## 일반 차선 주행

유효한 각 카메라 프레임에서 가까운·중간·먼 중심 표본과 곡률을 다시 계산해 rolling
`CommonPath`를 교체합니다. 표본은 촬영 시각 odom 자세로 변환하며, 첫 관측 위치까지의
카메라 사각 구간은 현재 차량 자세에서 C1 Hermite connector로 연결합니다. 같은 프레임의
도색 경계도 `PathSafety`에 고정합니다.

조향 lookahead는 측정 선속도 `v`에 따라 다음처럼 증가합니다.

```text
clamp(0.065 + 0.35·|v|, 0.08, 0.16) m
```

`0.20/0.26/0.28 m/s`에서 각각 `0.135/0.156/0.160 m`입니다. 곡률 속도 profile은
보이는 경로 전체의 제한 속도를 현재 위치까지 뒤로 전파하므로 곡선에 들어가기 전에
감속합니다. 현재 Gazebo profile은 직선 cruise `0.28 m/s`, 최종 상한 `0.30 m/s`입니다.

유효한 중심선이 `0.5 s` 동안 없으면 마지막 조향으로 추측 주행하지 않고 정지합니다.
다음 정상 표본이 들어오면 현재 제어권 상태에 맞춰 재개합니다.

## 순차 미션 매니저

기본 순서는 다음과 같습니다.

```text
intersection → obstacle → parking → zigzag → level_crossing → tunnel
```

매니저는 다음 미션 하나에 세대 번호가 있는 `Header` arm을 보냅니다. 컨트롤러는 그
세대에서 얻은 source-stamped 센서 표본으로 등록하고, 진입할 남은 경로의 전체 sweep가
통과한 경우에만 같은 세대의 ready를 발행합니다. fresh ready가 들어오면 enable을 열고,
미션의 `COMPLETE` 뒤에만 다음 미션을 arm합니다. polygon 안팎과 연결 직선 길이는 이
상태 전환에 사용하지 않습니다. 신호등은 이 sequence 밖에서 최초 출발만 허가합니다.

실물에서는 Gazebo PGM이나 polygon을 활성화 기준으로 옮길 필요가 없습니다. 각 미션의
내부 형상과 고정 landmark만 mission-local 좌표로 보정하고, 연결 직선에서는 다음
미션이 ready가 될 때까지 일반 차선 주행을 유지합니다.

## 미션별 동작

### Intersection

Intersection 제어기는 다음 판단만 소유합니다.

1. arm 세대에서 카메라 방향 표지판의 LEFT/RIGHT를 30 Hz 표본 9개로 확정합니다.
2. source stamp의 AMCL `map→odom`에서 방향을, 방향 표지의 세로 크기와
   가로 베어링에서 평행이동을 구해 `intersection_local→odom`을 고정합니다.
   AMCL 평행이동은 사용하지 않으므로 앞 직선 길이가 바뀌어도 로컬 미션 형상은
   같은 장애물 기준에 정렬됩니다.
3. 일반 차선 제어를 유지한 채 로컬 진입면까지 접근하고, 선택한 진입 경로의 전체
   직사각형 sweep가 통과하면 ready를 발행합니다.
4. enable과 제어권 인수 뒤 선택된 반원 입구까지 cubic 진입 경로를 실행합니다.
5. 진입 완료 직후 차선 제어권을 돌려주고 반원 rolling 경로를 주행합니다.
6. 선택 방향의 탈출 prefix에 가까워지면 제어권을 회수하고, 처음 고정한 같은 변환으로
   방향별 branch와 공통 출구 `CommonPath`를 실행합니다.
7. 출구 완료 직후 차선 제어권을 반환하고 새 영상 경계로 최종 차선을 확인합니다.

고정 진입·탈출과 카메라 반원은 모두 공통 validator와 follower를 사용합니다.
AMCL 방향과 표지 기준 평행이동은 제어권 인수 전에 고정하고 미션 실행 중에는
다시 갱신하지 않습니다. `/mission/map_pose`는 진단용이며 Ground Truth나 A*로 교차로
경로를 만들지 않습니다.

주요 상태는 다음과 같습니다.

```text
WAIT_INTERSECTION → SEARCH_DIRECTION(필요할 때만) → WAIT_ENTRY_HANDOFF
→ PREPARE_ENTRY_PATH → FOLLOW_ENTRY_PATH → FOLLOW_ARC_LANE
→ PREPARE_EXIT_PATH → FOLLOW_EXIT_PATH → VERIFY_FINAL_LANE → COMPLETE
```

### Obstacle

Obstacle은 미리 측량한 하나의 clamped cubic spline과 곡률 연속 quintic 출구만
사용합니다. arm 뒤 서로 독립인 LiDAR 장벽 전면을 source stamp의 odom과 맞춰 완전한
`obstacle_local→odom` SE(2)를 구합니다. 한 면만 보이거나 평행 정보만 있어 퇴화한
등록은 ready가 되지 않습니다. 카메라 경계는 로컬 template을 이동시키지 않고 주행
corridor가 맞는지 확인합니다.

제어권을 받기 전 남은 경로 전체의 직사각형 sweep을 검사하고, 주행 중에는 실시간
LiDAR와 라인 여유를 공통 validator로 다시 검사합니다. 별도 우회 경로나 실행 중 경로
재생성은 없습니다. 경로 목표와 로컬 종단 통과를 확인한 뒤 일반 차선으로 반환합니다.

```text
WAIT_GATE → ACQUIRING → AVOIDING → REJOINING → COMPLETE
```

### Parking

Parking은 선택과 상태 전환을 소유하고, 모든 이동 구간은 전진 또는 후진
`CommonPath`로 실행합니다.

1. arm 뒤 고정 주차 표지의 비평행 두 면을 LiDAR로 확인해
   `parking_local→odom`을 고정합니다.
2. 일반 차선 제어를 유지한 채 로컬 진입면으로 이동합니다. 실제 인계 자세의 adaptive
   connector와 선택 전 공통 entry quarter-turn 두 `CommonPath`의 전체 sweep가 통과하면
   ready를 보내고, enable 뒤 이를 실행합니다.
3. 중앙 판정 자세에서 정지하고, 동시각 LiDAR 점을 local 좌·우 ROI에 투영해 빈 공간을
   새 scan 3개로 확정합니다.
4. 같은 자리에서 선택 공간 방향으로 제자리회전한 뒤 직선 주차합니다.
5. 별도 유지시간 없이 저장한 실제 회전점까지 같은 직선을 후진합니다.
6. 북쪽으로 제자리회전하고 실제 복귀점부터 지그재그 진입부까지 quintic 경로를
   실행합니다.
7. 움직이는 동안 검증된 rolling 차선 경로가 확인되면 일반 차선으로 인계하고, 새
   영상과 진행거리를 확인한 뒤 완료합니다.

bringup 직후 RViz의 `Parking Left Planned Path`와
`Parking Right Planned Path`에 좌·우 주차 후보 경로가 모두 표시됩니다. 빈 공간은
중앙 판정 자세의 LiDAR 결과로만 선택하므로, 이 두 선은 선택 전에 고정 경로
형상과 양쪽 분기를 확인하기 위한 명목 계획입니다. 실제 진입 connector는 인계
자세에 맞춰 새로 생성됩니다.

고정한 mission-local 변환은 진입·빈 공간 ROI·주차·후진·복귀·handoff 경계 모두에
동일하게 적용합니다. 두 제자리회전도 실제 비대칭 footprint와 최신 LiDAR sweep을
검사합니다. 출구는 실제
도색의 두 solid arm 끝과 그 사이 개구를 유한 경계로 표현해 합법적인 개구를 막지
않습니다.

```text
WAIT_GATE → PREPARE_APPROACH → APPROACH → TURN_IN → ENTER_AISLE
→ SELECT_SPACE → TURN_TO_SPACE → PARK_IN → BACK_OUT
→ TURN_TO_EXIT → LEAVE_AISLE → TURN_TO_ZIGZAG
→ JOIN_ZIGZAG 또는 VERIFY_ZIGZAG_LANE → COMPLETE
```

### Zigzag

Zigzag는 YAML의 고정 knot로 만든 단일 C2 spline을 사용합니다. arm 뒤 rolling 카메라
경로의 곡선 형상을 측량 spline과 맞춰 고유 station을 확인합니다.
직선처럼 방향 정보가 부족하거나 여러 station이 같은 정도로 맞는 표본은 거부합니다.
Gazebo의 `route/odom_aligned: true`는 이 곡선 확인 뒤에만 identity 변환을 사용하고,
실물용 별도 profile은 이를 `false`로 두고 확인된 완전한 `zigzag_local→odom` SE(2)를
사용해야 합니다. 이 실물 profile과 현장 보정값은 아직 없습니다. 현재 위치를 전체
경로에 투영해 가장 가까운 진행 index부터 남은 suffix를 추종합니다.

bringup 직후 RViz의 `Zigzag Surveyed Path`에는 YAML로 만든 전체 측량 계획 경로가
표시됩니다. 지그재그 진입에 성공하면 같은 `/zigzag/path` 표시를 그 시점에 odom으로
고정한 실제 실행 경로로 교체하므로, 주행 중 보이는 선이 제어기가 추종하는 경로입니다.
Gazebo의 제어용 raw `/odom` 수치는 map/world와 정렬되지만 TF의 `odom`은 EKF가
소유하므로, 표시 메시지는 원래 측량 좌표계인 `map`을 유지해 시작점 오프셋을 중복
적용하지 않습니다.

공통 follower는 곡률 feed-forward, lookahead feedback과 미리 뒤로 전파한 속도
profile을 사용합니다. 공통 validator는 바깥 도색 경계와 동시각 LiDAR 장애물을 현재
반응·완전 정지 영역까지 검사합니다. 끝에서 새 rolling 차선 경로를 확인한 뒤 저속으로
인계하고, 진행거리와 새 영상 확인 후 일반 속도를 복원합니다.

```text
WAIT_GATE → ACQUIRING → FOLLOWING → VERIFY_EXIT → JOINING_LANE → COMPLETE
```

### Level Crossing

Level Crossing은 공통 경로 추종 대상이 아닙니다. enable 뒤 LiDAR 점을 실제
각도 메타데이터로 투영하고, 차로 폭 방향으로 넓고 진행 방향으로 얇은 연속 군집만
차단봉으로 인정합니다. arm 뒤 source-stamped 고정 landmark로 crossing frame과 통과
평면을 먼저 고정하고 ready를 보냅니다. `0.60 m`부터 닫힌 봉을 연속 확인하되
`0.45 m` 이하까지 접근한 뒤 제어권을 받아 정지합니다. 새 odometry로 실제 정지를
확인하고 그 자세에서 닫힌 봉을 다시 관측한 뒤에만 개방 확인을 시작합니다. 정상
scan에서 개방을 5회 연속
확인해야 차선 제어권을 반환합니다. scan이나 odometry가 오래되면 출발하지 않고,
통과 중 봉이 다시 내려오면 다시 정지합니다. 처음부터 열린 경우에는 차선 제어권을
유지하며, 실제 비대칭 footprint의 뒤끝이 로컬 통과 평면을 지난 뒤 완료합니다.

현재 Gazebo course에는 고정 차단봉 지주가 없어 움직이는 봉 군집을 등록 기준으로 쓰는
Gazebo 전용 fallback이 켜져 있습니다. 실물 설정에서는 이 fallback을 끄고
`/level_crossing/landmark_pose`에 source stamp가 있는 고정 landmark 자세를 발행해야
합니다. 이 실물 publisher와 현장 보정은 아직 launch에 구현되어 있지 않습니다.

```text
WAIT_GATE → APPROACH → STOPPED → PASSING → COMPLETE
```

### Tunnel

Tunnel의 Hybrid A*와 전용 추종·충돌 검사는 공통화 대상이 아닙니다. 입구에서 동시각
LiDAR가 본 종방향 벽, 직교 입구벽과 두 끝점을 mission-local tunnel template에 맞춰
template←odom 변환을 고정하고, 이후 odom을 이 고정 template 좌표로 변환해 제어합니다.
한 벽만 보이는 퇴화 표본은 ready가 되지 않습니다. 입·출구 connector와 계획 결과는
`CommonPath` 자료형에 담지만 공통
`PathFollower`나 `SweptFootprintValidator`를 호출하지 않습니다.

LiDAR는 정적 벽-only 지도와 별도인 동적 costmap을 갱신합니다. 전진 전용 Hybrid A*는
비대칭 직사각형 footprint를 primitive 사이까지 검사하며, 새 장애물이 남은 경로를
막으면 정지 후 재계획합니다. 정상 출구에서는 Hybrid A* 끝점부터 출구 밖까지의
zero-end-curvature connector를 미리 결합해 순항 속도를 유지합니다. 이 connector까지 전체
footprint 검사를 통과한 계획만 주행에 사용하며 정지 yaw 정렬 fallback은 없습니다. 출구
주행 중 일반 차선 제어기의 최신 공통 경로를 미리 확인하고, rear footprint가 portal을
통과하면 zero 명령 없이 곧바로 제어권을 반환합니다.

```text
WAIT_GATE → ACQUIRING → ALIGNING_ENTRY → ENTERING → PLANNING
→ FOLLOWING → EXITING → JOINING_LANE → COMPLETE
```

## 표지판 인식

`sign_detector.py` 한 노드가 `/detect/signs` typed 결과를 발행합니다. 현재 통합 주행
allowlist는 다음 세 템플릿뿐입니다.

| 미션 상태 | 활성 템플릿 | 사용처 |
|---|---|---|
| Intersection | `direction_left`, `direction_right` | 경로 분기 선택 |
| Tunnel | `tunnel_warning` | 진단 기록만 수행, arm/ready 대체 안 함 |
| mission 정보가 없는 단독 실행 | 위 세 템플릿 | 카메라 단독 시험 |

Obstacle, Parking, Level Crossing의 진입은 각 미션의 arm/ready와 해당 센서 등록을
사용합니다. 설정 파일에 남아 있는 다른 템플릿 entry는 현재 allowlist에서 선택되지
않으며 통합 주행 판단에 사용되지 않습니다. 방향 표지는 ORB/RANSAC과 파란 원형 영역의
흰 화살표 비대칭을 함께 검사합니다.

## 설정 파일 역할

| 파일 | 역할 |
|---|---|
| `config/lane_controller.yaml` | rolling 경로, 공통 추종·안전과 일반 차선 속도 |
| `config/mission_zones_gazebo.yaml` | 미션 순서, arm/ready/enable 계약과 선택적 polygon 진단 |
| `config/intersection_mission.yaml` | 표지·AMCL 방향 등록, 방향 경로와 속도 |
| `config/obstacle_mission_gazebo.yaml` | 측량 spline, 장벽 면 등록과 안전 여유 |
| `config/parking_mission_gazebo.yaml` | 고정 표지 등록, 공간 ROI와 전진·후진 구간 |
| `config/zigzag_mission_gazebo.yaml` | spline knot, rolling 곡선 등록과 속도 profile |
| `config/level_crossing_mission_gazebo.yaml` | 고정 landmark, 차단봉 LiDAR ROI와 확인 횟수 |
| `config/tunnel_mission_gazebo.yaml` | portal 등록, costmap, Hybrid A*와 전용 제어 |
| `config/sign_detector.yaml` | Gazebo 표지판 allowlist와 검출 임계값 |
| `config/sign_detector_d405.yaml` | 실물 카메라용 초기 표지판 임계값 |

Gazebo 전용 좌표·경계·속도는 실물에 복사하지 않습니다. 조정 가능한 속도·거리·각도·
가속도·timeout은 해당 환경 YAML을 기준으로 관리합니다.

## 실물 NUC 적용 순서

1. NUC에서 저장소를 clone하고 [`DOCKER_NOETIC.md`](../../DOCKER_NOETIC.md)의 환경 시작,
   build, `hardware.launch` 순서로 센서와 인지를 확인합니다. 현재 `hardware.launch`는
   드라이버·인지용이며 차선·미션 제어기 전체를 시작하는 실물 통합 launch가 아닙니다.
2. 정지 상태에서 D405 `1280×720@30 Hz`, Mid-360, OpenCR odometry와 TF가 모두 들어오는지
   확인하고 source timestamp와 odom 보간 오차를 기록합니다.
3. 장착된 D405 intrinsic, 중앙 crop과 bird-eye 사다리꼴을 보정합니다. 실제 도색에서
   중심선·경계·rolling `CommonPath`가 30 Hz로 안정된 뒤에만 속도를 올립니다.
4. Mid-360 외부 파라미터·높이 필터, 실제 비대칭 footprint, OpenCR 전·후진 부호·바퀴
   반지름·가감속 응답을 측정해 실물 YAML을 만듭니다.
5. 연결 직선 전체나 미션의 절대 map 위치를 다시 재지 않습니다. 미션 내부 형상과 등록
   기준만 보정합니다: Intersection 방향 표지와 접근 차선, Obstacle의 독립 장벽 면,
   Parking 고정 표지의 비평행 면·좌우 ROI, Zigzag spline과 카메라 곡선, Tunnel 입구벽과
   종방향 벽입니다.
6. 바퀴 주행 전에는 Gazebo 전용 파일을 복사하지 말고, 실물 odometry·실물 YAML과 같은
   arm/ready 매니저 및 제어기들을 연결하는 실물 통합 launch를 작성해 토픽을 점검합니다.
   현재 저장소에는 이 실물 통합 launch가 없습니다.
7. Level Crossing은 고정 지주나 표식 자세를 source-stamped
   `/level_crossing/landmark_pose`로 발행하는 실물 producer와 현장 보정값을 먼저
   구현해야 합니다. 현재 launch에는 이 producer가 없으며 움직이는 봉 fallback은
   Gazebo에서만 사용합니다.
8. 바퀴를 띄우거나 저속으로 각 미션의 arm→ready 등록을 먼저 확인합니다. 시작 위치와
   연결 직선 길이를 바꿔도 같은 local 경로가 고정되고, 퇴화 표본에서는 ready가 나오지
   않는지 확인합니다.
9. 미션 직전 저속 회귀로 선택·전체 sweep·전진/후진·차선 반환을 확인한 뒤,
   `Intersection → Obstacle → Parking → Zigzag → Level Crossing → Tunnel` 순서로 앞선
   미션을 하나씩 포함합니다. 단독 회귀는 최종 완료 판정이 아닙니다.
10. 마지막에는 새로 시작한 공식 launch와 공식 시작 자세에서 사람 개입 없이 결승선까지
   반복 주행하고, 미션별 시간·최대 추종 오차·최소 라인/장애물 여유·실패 횟수를 같은
   조건의 기준선과 비교합니다.

현재 adaptive 코드의 Gazebo 공식 시작점 통합 주행은 완료했습니다. 실제
D405·Mid-360·OpenCR 보정, 실물 통합 launch 작성과 공식 시작점 반복 주행은 남아
있습니다.

실측 기구값은
[`custom_autorace_description/HARDWARE_PARAMETERS.md`](../custom_autorace_description/HARDWARE_PARAMETERS.md),
펌웨어 보정은
[`firmware/custom_autorace_core/README.md`](../../firmware/custom_autorace_core/README.md)를
기준으로 합니다.
