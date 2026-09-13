# Custom AutoRace bringup

OpenCR, Intel RealSense D405, Livox Mid-360을 사용하는 ROS Noetic AutoRace
통합 패키지입니다. Ubuntu 24.04 호스트에서는 저장소 루트의
`compose.noetic.yaml`로 Ubuntu 20.04/ROS Noetic 컨테이너를 실행합니다.

실행·빌드·회귀·토픽 확인 명령은 저장소 루트의
[`DOCKER_NOETIC.md`](../../DOCKER_NOETIC.md)를 단일 기준으로 사용합니다. 이 문서는
현재 런타임 구조와 미션별 책임만 설명합니다.

## 현재 검증 상태

2026-09-13 `run23`은 새 Gazebo의 공식 시작 자세에서 카메라 신호로 출발해
`Intersection → Obstacle → Parking → Zigzag → Level Crossing → Tunnel`을 순서대로
완료하고 결승선 footprint를 통과했습니다.

| 항목 | 결과 |
|---|---|
| 통합 판정 | `22/22 PASS`, 실패 항목 0 |
| 출발부터 결승선 | `280.472 s` |
| 선택 분기 | Intersection LEFT, Parking LEFT |
| 차선 중심선 | 평균 `29.672 Hz`, 최대 출력 간격 `57 ms` |
| 제어권 | `/cmd_vel` 중복 발행 없이 순차 인계 |
| 당시 자동 회귀 | bringup `504`, 전체 workspace `557`, 실패·오류·건너뜀 0 |

이 결과는 Gazebo 공식 시작점 통합 검증입니다. 실제 장착 D405 원근 보정, 실제 도색의
30 Hz 차선 주행, Mid-360 motion distortion, OpenCR 응답은 아직 검증하지 않았습니다.
미션별 오차와 안전 여유는 [공통 경로 문서](PATH_FOLLOWING.md), 과거 구현과 속도
비교는 [검증 이력](docs/VALIDATION_HISTORY.md)에 분리해 기록합니다.

## 런타임 구조

신호등은 순차 미션 전에 출발을 허가하고, 일반 구간과 Intersection 반원은 카메라가
매 프레임 만든 rolling `CommonPath`를 추종합니다. Intersection, Obstacle, Parking,
Zigzag의 고정·측량 경로도 같은 `CommonPath`, `SweptFootprintValidator`,
`PathFollower`를 사용합니다.

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
| Intersection | 카메라 LEFT/RIGHT | 진입·탈출 시점 AMCL map→odom 고정 | 입·출구는 미션 제어기, 반원은 일반 차선 제어기 |
| Obstacle | 측량된 단일 spline | LiDAR 장애물 면으로 course→odom 고정 | 미션 제어기 + 공통 추종기 |
| Parking | LiDAR로 빈 좌·우 공간 선택 | 진입 시점 AMCL map→odom 고정, 짧은 구간은 odom | 미션 제어기 + 공통 전진·후진 추종기 |
| Zigzag | 측량된 단일 spline | Gazebo odom-aligned, 실물 AMCL map→odom | 미션 제어기 + 공통 추종기 |
| Level Crossing | LiDAR 차단봉 판정 | 경로 정렬 없음 | 전용 정지·재출발 제어기 |
| Tunnel | LiDAR 동적 costmap + Hybrid A* | 입구 AMCL map→odom anchor 고정 | 전용 계획·추종·충돌 검사 |

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

### LiDAR

Mid-360 `PointCloud2`는 `/scan_mid360_raw`로 변환됩니다. 미션 제어기는 LaserScan의
`angle_min`과 `angle_increment`를 그대로 사용하며 정면 인덱스 기준으로 배열을
재정렬하지 않습니다.

### Gazebo odometry와 AMCL

통합 Gazebo launch는 후진 부호를 보존하기 위해 signed world pose/twist를 EKF 입력으로
쓰고, EKF가 `/odometry/filtered`와 유일한 `odom → base_footprint` TF를 발행합니다.
AMCL은 정적 지도와 scan으로 `map → odom`을 보정합니다. 미션 구역 매니저는 같은
`map → base_footprint` 자세로 순차 polygon과 `/mission/map_pose`를 계산합니다.

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

아직 차례가 아닌 미션은 해당 polygon에 들어가도 활성화되지 않습니다. 각 미션은
자기 제어기가 실제 동작과 최종 자세를 확인한 뒤 `COMPLETE`를 발행해야 다음 단계로
넘어갑니다. 신호등은 이 sequence 밖에서 최초 출발만 허가합니다.

실물에서는 Gazebo PGM과 polygon 좌표를 재사용하지 않습니다. Mid-360으로 고정 구조만
담은 지도를 만들고, 같은 map 좌표계에서 미션 polygon과 고정 경로·도색 경계를 다시
측량해야 합니다.

## 미션별 동작

### Intersection

Intersection 제어기는 다음 판단만 소유합니다.

1. 조기 관찰 polygon에서 카메라 방향 표지판의 LEFT/RIGHT를 확정합니다.
2. 일반 차선 제어를 유지한 채 측량 진입면까지 접근합니다.
3. 제어권 인수 뒤 새 EKF 자세에서 선택된 반원 입구까지 cubic 진입 경로를 만듭니다.
4. 진입 완료 직후 차선 제어권을 돌려주고 반원 rolling 경로를 주행합니다.
5. 선택 방향의 탈출 prefix에 가까워지면 제어권을 회수합니다.
6. 최신 AMCL map→odom을 다시 고정해 connector, 방향별 branch와 공통 출구를 하나의
   `CommonPath`로 실행합니다.
7. 출구 완료 직후 차선 제어권을 반환하고 새 영상 경계로 최종 차선을 확인합니다.

고정 진입·탈출과 카메라 반원은 모두 공통 validator와 follower를 사용합니다. 실행 중
AMCL 갱신으로 확정 경로를 움직이지 않으며 Ground Truth나 A*로 교차로 경로를 만들지
않습니다.

주요 상태는 다음과 같습니다.

```text
WAIT_INTERSECTION → SEARCH_DIRECTION(필요할 때만) → WAIT_ENTRY_HANDOFF
→ PREPARE_ENTRY_PATH → FOLLOW_ENTRY_PATH → FOLLOW_ARC_LANE
→ PREPARE_EXIT_PATH → FOLLOW_EXIT_PATH → VERIFY_FINAL_LANE → COMPLETE
```

### Obstacle

Obstacle은 미리 측량한 하나의 clamped cubic spline과 곡률 연속 quintic 출구만
사용합니다. AMCL은 진입 gate와 LiDAR 정합 초기값만 제공하고, 실제 장애물 면 정합이
성공하면 course 좌표계를 odom에 한 번 고정합니다. 카메라 경계는 course template을
이동시키지 않고 주행 corridor가 맞는지 확인합니다.

제어권을 받기 전 남은 경로 전체의 직사각형 sweep을 검사하고, 주행 중에는 실시간
LiDAR와 라인 여유를 공통 validator로 다시 검사합니다. 별도 우회 경로나 실행 중 경로
재생성은 없습니다. 경로 끝과 AMCL 출구 여유를 확인한 뒤 일반 차선으로 반환합니다.

```text
WAIT_GATE → ACQUIRING → AVOIDING → REJOINING → COMPLETE
```

### Parking

Parking은 선택과 상태 전환을 소유하고, 모든 이동 구간은 전진 또는 후진
`CommonPath`로 실행합니다.

1. 순차 gate 뒤 일반 차선 속도를 제한한 상태로 이동하며 실제 인계 자세에서 진입
   connector를 만듭니다.
2. 중앙 판정 자세에서 정지하고, 동시각 LiDAR 점을 map의 좌·우 ROI에 투영해 빈 공간을
   새 scan 3개로 확정합니다.
3. 같은 자리에서 선택 공간 방향으로 제자리회전한 뒤 직선 주차합니다.
4. 별도 유지시간 없이 저장한 실제 회전점까지 같은 직선을 후진합니다.
5. 북쪽으로 제자리회전하고 실제 복귀점부터 지그재그 진입부까지 quintic 경로를
   실행합니다.
6. 움직이는 동안 검증된 rolling 차선 경로가 확인되면 일반 차선으로 인계하고, 새
   영상과 진행거리를 확인한 뒤 완료합니다.

AMCL map→odom 관계는 진입 때 한 번 고정되며 짧은 주차 구간은 odom 기준으로 유지됩니다.
두 제자리회전도 실제 비대칭 footprint와 최신 LiDAR sweep을 검사합니다. 출구는 실제
도색의 두 solid arm 끝과 그 사이 개구를 유한 경계로 표현해 합법적인 개구를 막지
않습니다.

```text
WAIT_GATE → PREPARE_APPROACH → APPROACH → TURN_IN → ENTER_AISLE
→ SELECT_SPACE → TURN_TO_SPACE → PARK_IN → BACK_OUT
→ TURN_TO_EXIT → LEAVE_AISLE → TURN_TO_ZIGZAG
→ JOIN_ZIGZAG 또는 VERIFY_ZIGZAG_LANE → COMPLETE
```

### Zigzag

Zigzag는 YAML의 고정 knot로 만든 단일 C2 spline을 사용합니다. Gazebo에서는
`route/odom_aligned: true`로 측량 경로와 odom을 직접 맞추고, 실물에서는 동시각
AMCL map/odom 자세로 map 경로를 odom에 한 번 고정합니다. 현재 위치를 전체 경로에
투영해 가장 가까운 진행 index부터 남은 suffix를 추종합니다.

공통 follower는 곡률 feed-forward, lookahead feedback과 미리 뒤로 전파한 속도
profile을 사용합니다. 공통 validator는 바깥 도색 경계와 동시각 LiDAR 장애물을 현재
반응·완전 정지 영역까지 검사합니다. 끝에서 새 rolling 차선 경로를 확인한 뒤 저속으로
인계하고, 진행거리와 새 영상 확인 후 일반 속도를 복원합니다.

```text
WAIT_GATE → ACQUIRING → FOLLOWING → VERIFY_EXIT → JOINING_LANE → COMPLETE
```

### Level Crossing

Level Crossing은 공통 경로 추종 대상이 아닙니다. 순차 gate 안에서 LiDAR 점을 실제
각도 메타데이터로 투영하고, 차로 폭 방향으로 넓고 진행 방향으로 얇은 연속 군집만
차단봉으로 인정합니다. 닫힌 봉을 연속 확인한 뒤 제어권을 받아 정지하며, 충분한 새
scan에서 개방을 확인한 뒤 차선 제어권을 반환합니다. scan이 오래되면 출발하지 않고,
통과 중 봉이 다시 내려오면 다시 정지합니다.

```text
WAIT_GATE → APPROACH → STOPPED → PASSING → COMPLETE
```

### Tunnel

Tunnel의 Hybrid A*와 전용 추종·충돌 검사는 공통화 대상이 아닙니다. 입구에서 동시각
AMCL과 odom으로 map←odom anchor를 고정하고, 이후 odom을 이 고정 map 좌표로 변환해
제어합니다. 입·출구 connector와 계획 결과는 `CommonPath` 자료형에 담지만 공통
`PathFollower`나 `SweptFootprintValidator`를 호출하지 않습니다.

LiDAR는 정적 벽-only 지도와 별도인 동적 costmap을 갱신합니다. 전진 전용 Hybrid A*는
비대칭 직사각형 footprint를 primitive 사이까지 검사하며, 새 장애물이 남은 경로를
막으면 정지 후 재계획합니다. 출구 뒤에는 일반 차선 제어기의 최신 공통 경로 진단과
portal 여유를 확인하고 제어권을 반환합니다.

```text
WAIT_GATE → ACQUIRING → ALIGNING_ENTRY → ENTERING → PLANNING
→ FOLLOWING → ALIGNING_EXIT → EXITING → VERIFY_EXIT
→ JOINING_LANE → COMPLETE
```

## 표지판 인식

`sign_detector.py` 한 노드가 `/detect/signs` typed 결과를 발행합니다. 현재 통합 주행
allowlist는 다음 세 템플릿뿐입니다.

| 미션 상태 | 활성 템플릿 | 사용처 |
|---|---|---|
| Intersection | `direction_left`, `direction_right` | 경로 분기 선택 |
| Tunnel | `tunnel_warning` | 진단 기록만 수행, gate 대체 안 함 |
| mission 정보가 없는 단독 실행 | 위 세 템플릿 | 카메라 단독 시험 |

Obstacle, Parking, Level Crossing의 진입은 AMCL 순차 gate와 해당 센서 판단을
사용합니다. 설정 파일에 남아 있는 다른 템플릿 entry는 현재 allowlist에서 선택되지
않으며 통합 주행 판단에 사용되지 않습니다. 방향 표지는 ORB/RANSAC과 파란 원형 영역의
흰 화살표 비대칭을 함께 검사합니다.

## 설정 파일 역할

| 파일 | 역할 |
|---|---|
| `config/lane_controller.yaml` | rolling 경로, 공통 추종·안전과 일반 차선 속도 |
| `config/mission_zones_gazebo.yaml` | 미션 순서와 AMCL polygon |
| `config/intersection_mission.yaml` | 방향 판정, 진입·탈출 경로와 속도 |
| `config/obstacle_mission_gazebo.yaml` | 측량 spline, LiDAR 정합과 안전 여유 |
| `config/parking_mission_gazebo.yaml` | 공간 ROI, 전진·후진 구간과 상태 판정 |
| `config/zigzag_mission_gazebo.yaml` | spline knot, 정렬 방식과 속도 profile |
| `config/level_crossing_mission_gazebo.yaml` | 차단봉 LiDAR ROI와 확인 횟수 |
| `config/tunnel_mission_gazebo.yaml` | costmap, Hybrid A*, portal과 전용 제어 |
| `config/sign_detector.yaml` | Gazebo 표지판 allowlist와 검출 임계값 |
| `config/sign_detector_d405.yaml` | 실물 카메라용 초기 표지판 임계값 |

Gazebo 전용 좌표·경계·속도는 실물에 복사하지 않습니다. 조정 가능한 속도·거리·각도·
가속도·timeout은 해당 환경 YAML을 기준으로 관리합니다.

## 실물 적용 전 남은 검증

- 장착된 D405 intrinsic, crop과 bird-eye 사다리꼴 재보정
- 실제 조명·도색에서 중심선과 경계 `30 Hz` 유지 확인
- 실제 Mid-360의 외부 파라미터, 높이 필터, timestamp/odom 동기 확인
- 주행 중 Mid-360 motion distortion이 정합과 stopping sweep에 미치는 영향 확인
- OpenCR의 전·후진 부호, 유효 바퀴 반지름, 가·감속 응답 측정
- 실제 경기장 고정 구조 지도와 미션 polygon 재작성
- 같은 map frame에서 Intersection 경로, Obstacle spline, Parking ROI·경로,
  Zigzag spline, Level Crossing ROI, Tunnel portal·벽을 재측량
- 실물용 footprint 여유와 미션별 속도 profile 조정
- 공식 시작 자세에서 사람 개입 없이 전체 순서와 결승 통과 반복 검증

실측 기구값은
[`custom_autorace_description/HARDWARE_PARAMETERS.md`](../custom_autorace_description/HARDWARE_PARAMETERS.md),
펌웨어 보정은
[`firmware/custom_autorace_core/README.md`](../../firmware/custom_autorace_core/README.md)를
기준으로 합니다.
