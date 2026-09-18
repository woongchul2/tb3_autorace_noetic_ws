# Gazebo 검증 이력

이 문서는 현재 운용 설명에서 분리한 검증 기록입니다. 맨 위에는 2026-09-18
mission-local adaptive registration의 현재 검증 범위를 기록하고, 그 아래
2026-09-17 및 이전 결과는 각 당시 코드와 설정에만 해당하는 과거 기준선으로
보존합니다. 결승선까지의 완료 판정은 상위 [`README.md`](../README.md)에 구분해
기록합니다.

원시 bag과 임시 분석 결과는 저장소에 포함하지 않습니다. 숫자는 당시 기록 문서에서
옮긴 자체 완결 요약이며, 서로 다른 방향·주차 분기나 다른 시작 조건의 시간은 엄밀한
A/B로 비교하지 않습니다.

## 2026-09-18 Mission-local adaptive registration

현재 코드는 순차 AMCL polygon gate를 `arm → sensor-relative registration → ready →
enable` 계약으로 교체했습니다. Intersection은 source stamp의 AMCL 방향과 방향
표지 높이·베어링, Obstacle은 서로 독립인 LiDAR 장벽 면, Parking은 고정 표지의 비평행 면,
Zigzag는 rolling 카메라 곡선으로 각각 mission-local 경로를 odom에 고정합니다.
Level Crossing은 고정 landmark 통과 평면, Tunnel은 직교하는 입구·종방향 벽과 끝점을
같은 ordered readiness에 사용합니다. 선택적 AMCL polygon은 진단일 뿐 활성화 조건이
아닙니다.

ROS Noetic 컨테이너의 `CMakeLists.txt`에 등록된 Python test 29개 파일을 최종
aggregate 대상으로 사용했습니다. 아래 범위를 모두 포함하며 시험 파일을 제외하지
않았습니다.

| 검증 범위 | 결과 |
|---|---|
| 공통 local SE(2), 퇴화·outlier·시간 연속성, rolling 곡선 station | 포함 |
| 공통 경로 투영, 전진·후진 추종, 곡률 속도 제한, 비대칭 직사각형 sweep | 포함 |
| 6개 미션 arm/ready/enable 세대·source stamp와 launch/config 배선 | 포함 |
| Intersection 양방향, Obstacle 장벽 등록, Parking 좌·우·후진, Zigzag 곡선 등록 | 포함 |
| Level Crossing 통과 평면과 Tunnel portal 등록·costmap·Hybrid A* | 포함 |
| 최종 aggregate·build·diff | bringup `669/669`, description `7/7`, 합계 `676/676 PASS`; fail 0, error 0; 두 패키지 build와 `git diff --check` PASS |

### 공식 시작점 통합 `run20`

새 Gazebo와 공식 통합 launch를 시작하고 공식 자세 `(0.800, -1.747, 0°)`에서 카메라
신호로 출발했습니다. 순간이동, 수동 gate, 미션 비활성화와 사람 개입 없이
Intersection LEFT, Parking LEFT, Tunnel layout B를 선택해 모든 미션과 결승선을
통과했습니다.

| 구간 | ACTIVE→COMPLETE |
|---|---:|
| Intersection | `25.972 s` |
| Obstacle | `35.125 s` |
| Parking | `57.886 s` |
| Zigzag | `16.222 s` |
| Level Crossing | `14.386 s` |
| Tunnel | `75.283 s` |
| 출발→결승 footprint | `283.053 s` |

결승 base pose는 `(1.033509, -1.745858, 0.03248°)`였고 비대칭 footprint 전체가
허용 y 구간 `[-1.86, -1.64] m` 안에 있었습니다. 6개 미션은 모두 `COMPLETE`,
`FAILED`와 manual stop은 0건이었습니다. Parking 후진은 `6.303 s`, 음수 명령 127개,
최저 `-0.103447 m/s`였습니다. `/cmd_vel` 발행자는 기대한 순서로만 바뀌었고 발행자
교차는 없었으며 14개 인계 공백은 `3.582–121.699 ms`였습니다. 공통 경로 오차와
최소 여유는 [`PATH_FOLLOWING.md`](../PATH_FOLLOWING.md)에 기록합니다.

bag 분석은 `19/22`였습니다. 실패한 세 항목은 recorder를 `/use_sim_time=true`보다
먼저 시작해 생긴 첫 pose `2.411 s` 지연과 record-time callback burst,
`/detect/lane_centerline` 녹화 누락입니다. 연속 pose의 실제 최대 이동은 `0.283 mm`,
최대 yaw 변화는 `0.054°`여서 순간이동 증거는 없었습니다. 물리 contact/bumper 토픽은
기록하지 않았으므로 충돌 자체를 독립 판정하지 않고 mission 상태와 clearance를
근거로 사용했습니다.

### Tunnel 등록 보정과 반복 주행

앞선 `run19`의 Tunnel 실패는 동적 원통이 아니라 등록 landmark midpoint를 입구 plane에
직접 맞추면서 생긴 약 `65 mm` 종방향 편향이 출구 고정벽을 장애물로 오투영한 것이
원인이었습니다. 검출 landmark용 `registration.reference_pose`와 실제 clearance용
`entry_portal_plane_y`를 분리했습니다. `run19` bag 재투영에서 출구 ghost는 사라지고
실제 원통 3개는 유지됐으며, `2.498 m` 전체 경로의 비대칭 footprint sweep가
통과했습니다.

최종 코드의 `run20` Tunnel은 계획 2회·재계획 1회로 layout B를 완료했습니다. 별도
공식 시작 반복 `run21`도 Intersection RIGHT, Parking RIGHT, Tunnel layout C에서 6개
미션을 모두 완료했고 Tunnel은 계획 2회·재계획 1회, `69.660 s`였습니다. Tunnel 완료
`3.539 s` 뒤 결승 footprint를 통과했습니다. 미션별 ACTIVE→COMPLETE는
`24.955 / 34.657 / 56.321 / 16.184 / 14.756 / 69.660 s`였습니다. `run21` bag은
출발 뒤 녹화를 시작했으므로 공식 시작 시간 비교 자료로 사용하지 않습니다.

`run21`의 `/detect/lane_centerline` 전체 발행은 header stamp 기준 8,278건/281.537초,
`29.399 Hz`였고 interval median/p95/max는 `33/35/70 ms`였습니다. 정상 크기의 배열에서
하나 이상의 실제 경계가 유효한 메시지는 7,712건, `27.389 Hz`였고 명시적 empty 관측은
566건이었습니다. header와 bag receipt time 모두 중복·역행 stamp는 0건이었습니다.
이 bag은 정상 종료 index가 없는 `.bag.active`의 마지막 완전 chunk까지만 임시 복사본을
색인해 분석했으며 원본은 수정하지 않았습니다.

Gazebo 공식 시작점 통합 검증은 완료했지만 실제 D405·Mid-360·OpenCR 검증은 남아
있습니다. 현재 `hardware.launch`는 센서와 인지만 시작하며 실물 통합 미션 launch와
`/level_crossing/landmark_pose` 고정 landmark producer는 아직 없습니다. 따라서
실물 주행 완료로 판정하지 않습니다.

## 2026-09-17 Level Crossing 거리 기반 정지·재확인

차단봉 군집을 `0.60 m`부터 3회 연속 확인해 접근 상태로 전환하고, 측정 거리가
`0.45 m` 이하일 때만 제어권을 받아 정지하도록 검출 거리와 정지 거리를 분리했습니다.
정지 명령 뒤에는 새 odometry 2개로 실제 선속도 `0.03 m/s`, 각속도 `0.10 rad/s`
이하를 확인하고, 그 정지 자세에서 내려온 차단봉을 다시 확인해야 합니다. 이후 정상
LiDAR scan에서 차단봉 없음이 5회 연속 확인된 경우에만 차선 제어권을 돌려줍니다.
정지 뒤 움직임이나 odometry 단절은 이 확인을 초기화합니다.

새 Gazebo·ROS에서 `x=-1.60 m` 차단봉 단독 회귀를 당시 최종 코드로 3회 반복했습니다.

| 회차 | 최초 닫힘 확인 | 정지 명령 거리 | 최종 정지 자세 재확인 | 결과 |
|---:|---:|---:|---:|---|
| 1 | `0.544 m` | `0.432 m` | `0.423 m` | `PASSING→COMPLETE` |
| 2 | `0.532 m` | `0.438 m` | `0.428 m` | `PASSING→COMPLETE` |
| 3 | `0.532 m` | `0.425 m` | `0.408 m` | `PASSING→COMPLETE` |

2·3회차에는 정지 직후 잔류 움직임으로 첫 확인이 취소됐고, 실제로 다시 정지한 뒤
차단봉을 재관측해 통과했습니다. 세 회 모두 차단봉 상승 전에는 0 속도를 유지했고,
상승 뒤 차선 제어권 복귀와 zone 이탈 완료까지 사람의 개입이 없었습니다.

같은 당시 코드의 공식 시작 자세 통합 주행에서는 카메라 `RIGHT`, Parking `LEFT`로
Intersection, Obstacle, Parking, Zigzag, Level Crossing, Tunnel을 순서대로 모두
완료했습니다. Level Crossing은 `0.489 m`에서 최초 확인, `0.439 m`에서 정지 명령,
실제 정지 자세 `0.423 m`에서 재확인했고 `PASSING→COMPLETE` 뒤 차선 제어권을
반환했습니다. 순간이동, 수동 gate, 미션 비활성화와 사람 개입은 사용하지 않았습니다.

bringup 자동 회귀 `520 tests`는 오류·실패·건너뜀 0으로 통과했습니다. 이 결과는
Gazebo의 2D LiDAR proxy 기준이며 실제 Mid-360 반복시험을 대신하지 않습니다. 당시에는
LiDAR 단독 판정을 유지하고, 실물 정지 자세의 point cloud에서 반복 누락이나 오검출이
확인될 때 카메라 교차 확인을 추가합니다.

## 2026-09-17 Intersection 단일 polygon 통합

별도 `intersection_direction_observation` region과 토픽을 제거하고, Intersection
mission polygon의 동쪽 경계를 `x=1.65→1.72 m`로 확장했습니다. 이제 하나의 순차
AMCL gate 안에서만 표지판을 판독하며, 실제 `/cmd_vel` 인수는 기존의 별도 측량
진입면에서 결정합니다. 탈출은 polygon이 아니라 선택한 반원 끝에 대한 AMCL map-pose
투영으로 계속 판단합니다.

새 Gazebo·ROS의 기본 통합 launch, 공식 시작 자세, `forced_direction=0`, GUI·RViz를
사용해 카메라 LEFT와 RIGHT를 각각 1회 검증했습니다. 순간이동, 수동 gate, 강제 방향,
사람 개입은 사용하지 않았습니다.

| 실제 카메라 방향 | gate 개방 | 방향 확정 | 방향 확정 지연 | `COMPLETE` | 완료 뒤 일반 차선 첫 명령 |
|---|---:|---:|---:|---:|---:|
| LEFT | `15.509 s` | `16.988 s` | `1.479 s` | `51.722 s` | `0.002 s` |
| RIGHT | `13.877 s` | `14.475 s` | `0.598 s` | `46.619 s` | `0.026 s` |

두 주행 모두
`WAIT_INTERSECTION → SEARCH_DIRECTION → WAIT_ENTRY_HANDOFF → PREPARE_ENTRY_PATH →
FOLLOW_ENTRY_PATH → FOLLOW_ARC_LANE → PREPARE_EXIT_PATH → FOLLOW_EXIT_PATH →
VERIFY_FINAL_LANE → COMPLETE` 순서를 지켰습니다. 움직이는 `/cmd_vel` 발행자도
`lane → intersection → lane → intersection → lane` 순서였고, `FAILED`, 완료 후
Intersection 제어기의 잔여 주행 명령, 제거한 보조 관찰 토픽은 없었습니다. 공통 경로
최대 위치 오차는 LEFT `6.803 mm`, RIGHT `6.591 mm`였습니다. 자동 회귀는 bringup
`507 tests`, 오류·실패·건너뜀 0으로 통과했습니다. 이 단독 검증 직후에는 전체 코스를
다시 주행하지 않았고, 이후 위 Level Crossing 절의 공식 통합에서 전체 미션 순서를
재검증했습니다.

## Adaptive registration 이전의 공통화 기록

| 검증 | 도달 상태 | 당시 판정 |
|---|---|---|
| Intersection actual-pose 직접 cubic | 공식 자세에서 카메라 RIGHT·LEFT 각 1회. 진입 `10.323/7.378 s`, 전체 `34.836/32.490 s`; 반대 누적 회전과 반대 조향 명령 0 | 양방향 완료·차선 복귀 |
| 이전 공통화 등록 시험 | `350 tests`, 오류·실패·건너뜀 0 | 통과 |
| 네 미션 공식 시작 통합 | 카메라 RIGHT, Parking LEFT. Intersection `177.749 s`, Obstacle `218.599 s`, Parking `331.192 s`, Zigzag `350.061 s`; `350.104 s` 차선 명령 재개 | 공통화 네 미션 완료 |
| Obstacle gate 축소 통합 | Intersection 뒤 Obstacle `185.672→216.024 s`; 다음 Parking gate `218.651 s` | 목표 미션까지 완료 |
| Intersection 20% connector | 실제 카메라 `RIGHT, LEFT, LEFT, RIGHT`; 총 `38.977, 38.256, 34.247, 37.747 s` | `4/4` 완료 |
| Intersection-only 이전 기준 | RIGHT `31.140 s`, LEFT `28.738 s` | 양방향 완료 |
| Obstacle 전용 | 최대 경로 오차 `4.8088 mm`, 최소 line/obstacle 여유 `3.39056/4.24547 mm` | 완료 |
| Parking LEFT 전용 | 최대 위치·횡오차 `8.7069/6.8366 mm`, 최소 line/obstacle 여유 `0.98794/2.11462 mm` | 완료 |
| Parking RIGHT 전용 | 최대 위치·횡오차 `11.2714/6.8695 mm`, 최소 line/obstacle 여유 `0.67022/1.43997 mm` | 완료 |
| Zigzag 전용 | `25.956 s`, 최대 경로·방향 오차 `4.5 mm/3.41°`, 최소 line/LiDAR 여유 `3.5155/~30.2 mm` | 완료 |

위 네 미션 공식 통합 기록의 활성 상태부터 완료까지 시간은 Intersection `40.179 s`,
Obstacle `32.431 s`, Parking `109.926 s`, Zigzag `18.779 s`였습니다. 당시 공통 진단의
최대 오차와 최소 raw 여유는 다음과 같습니다.

| 미션 | 최대 위치 오차 | 최대 절대 횡오차 | 최소 raw line | 최소 raw obstacle | 최소 raw map |
|---|---:|---:|---:|---:|---:|
| Intersection | `23.483 mm` | `23.483 mm` | `-54.117 mm` | 경계 없음 | `484.418 mm` |
| Obstacle | `19.400 mm` | `19.400 mm` | `3.391 mm` | `4.245 mm` | 경계 없음 |
| Parking LEFT | `9.517 mm` | `7.013 mm` | `1.093 mm` | `-7.882 mm` | `126.377 mm` |
| Zigzag | `20.521 mm` | `4.636 mm` | `2.871 mm` | `22.348 mm` | 경계 없음 |

Intersection 음수 line 값은 시작 overlap allowance 적용 전 raw 값입니다. Parking 음수
obstacle 값은 live LiDAR를 포함한 예측 sweep의 일시적 최소값이며, 당시
`motion_safety`가 감속·정지 판단에 사용했습니다. 이 집계만으로 고정점과 live 점,
route sweep과 stopping sweep을 구분할 수는 없습니다.

## 일반 차선 속도와 lookahead

### Rolling CommonPath 속도 단계

당시 C1 카메라 경로 생성기를 고정하고 새 Gazebo에서 측정했던 결과입니다. 침범과 여유는
허용 주행면 기준 signed 값입니다.

| 조향 목표/직선 속도 | gate 시간 | 최대 안쪽 침범 | 최소 바깥 여유 | 경로 오차 p95 | 판정 |
|---|---:|---:|---:|---:|---|
| 고정 `0.065 m` / `0.26 m/s` | `12.338 s` | `9.538 mm` | `12.778 mm` | `16.468 mm` | 성공 |
| adaptive `0.08–0.16 m` / `0.20 m/s` | `12.522 s` | `0 mm` | `27.676 mm` | `16.271 mm` | 성공 |
| adaptive / `0.22 m/s` | `12.412 s` | `0 mm` | `28.736 mm` | `15.500 mm` | 성공 |
| adaptive / `0.24 m/s` | `11.760 s` | `0 mm` | `29.706 mm` | `16.869 mm` | 성공 |
| adaptive / `0.26 m/s`, 2회 | `11.475–12.002 s` | `0 mm` | `28.080–28.322 mm` | `13.998–18.283 mm` | `2/2` |
| adaptive / `0.26 m/s`, 교차 2회 | `11.654–11.909 s` | `0 mm` | `28.552–28.650 mm` | `14.598–16.635 mm` | `2/2` |
| adaptive / `0.28 m/s`, 4회 | `11.607–11.807 s` | `0 mm` | `28.996–29.673 mm` | `14.094–16.175 mm` | `4/4` |
| adaptive / `0.30 m/s`, 1회 | `11.776 s` | `0 mm` | `28.970 mm` | `14.289 mm` | 통과, 미채택 |

같은 `0.26 m/s` 첫 직접 비교는 `12.338→12.002 s`, `0.336 s(2.7%)` 단축됐습니다.
adaptive `0.20→0.26 m/s`는 `12.522→11.475 s`, `1.047 s(8.4%)` 단축됐습니다.
순서를 섞은 재측정에서 `0.26 m/s` 평균 `11.781 s`, `0.28 m/s` 평균 `11.699 s`로
차이는 `0.082 s(0.70%)`였습니다. `0.30 m/s`는 시간 이득이 없고 순간 횡가속도가
`0.1599 m/s²`까지 올라 미채택했습니다. 최대 lookahead `0.18 m` 후보도
`12.218 s`로 느려 미채택했습니다.

경로 접합 방식까지 달랐던 더 오래된 close-target 기록은 `10.836 s`였으므로 해당
adaptive 구현이 그보다 오래된 모든 구현보다 빠르다고 해석하지 않습니다.

### 2026-09-07 이전 카메라 PD 비교

rolling `CommonPath` 전환 전 카메라 PD와 고정 측량 경로를 매번 새 Gazebo에서 11회씩
비교했습니다. 명목 자세 5회, `y ±10 mm`와 `yaw ±2°` 조합 4회, 허용 시작영역 가장자리
2회였습니다.

| 당시 제어 방식 | 성공 | gate 시간 중앙값 | 전체 범위 | 최대 안쪽 침범 | 최소 바깥 여유 | 경로 오차 p95 중앙값 |
|---|---:|---:|---:|---:|---:|---:|
| 카메라 PD, `0.20 m/s` | `11/11` | `11.863 s` | `11.836–11.954 s` | `15.44 mm` | `7.29 mm` | `35.84 mm` |
| 카메라 PD, `0.22 m/s` | `11/11` | `11.396 s` | `11.377–11.609 s` | `15.42 mm` | `7.33 mm` | `35.93 mm` |
| 고정 측량 lookahead, `0.22 m/s` | `11/11` | `9.781 s` | `9.666–9.971 s` | `1.44 mm` | `22.56 mm` | `2.69 mm` |

당시 측량 경로 중앙값은 `0.20 m/s` PD보다 `17.6%`, 같은 최고속도 PD보다 `14.2%`
짧았습니다. PD의 속도만 `0.20→0.22 m/s`로 높인 효과는 `3.9%`였습니다. 이 결과는
후속 rolling 경로 성능과 직접 비교하지 않습니다.

## Intersection 변경 이력

### 실제 인계 자세 직접 cubic

명목 경로의 20% 지점으로 연결하던 구현을 실제 인계 자세에서 반원 입구까지 직접 잇는
cubic으로 교체했습니다. 공식 시작 자세와 실제 카메라 방향을 사용해 양쪽을 각 1회
검증했습니다.

| 방향 | 진입 | 전체 | 순방향 누적 회전 | 반대 회전 | 완료 뒤 차선 첫 명령 |
|---|---:|---:|---:|---:|---:|
| RIGHT | `10.323 s` | `34.836 s` | `-68.386°` | `0°` | `0.098 s` 뒤 |
| LEFT | `7.378 s` | `32.490 s` | `+110.360°` | `0°` | `0.008 s` 뒤 |

인계 뒤 새 EKF 자세와 경로 첫 점의 차이는 두 방향 모두 `0 m`였고, 안전 정지,
wrong-turn, timeout과 `FAILED`는 없었습니다. 이 단계에서는 결승선까지 다시 검증하지
않았습니다.

### 조기 방향 관찰

방향 관찰창과 미션 gate를 분리한 뒤 GUI·RViz 기본 통합 launch에서 RIGHT와 LEFT를
각 1회 확인했습니다. RIGHT는 ordered gate보다 `0.432 s` 먼저 후보를 저장하고 gate
뒤 `0.166 s`에 확정했으며, LEFT는 관찰창이 gate보다 `1.004 s` 먼저 열리고 gate 뒤
`0.716 s`에 확정했습니다. 교차로 제어 시간은 `35.081/32.657 s`, 완료 뒤 일반 차선
명령 복귀는 `0.048/0.014 s`였습니다.

### 반원 속도 상한

반원 차선 추종 상한을 `0.10→0.12 m/s`로 올리고 실제 카메라 판독으로 RIGHT 3회와
LEFT 3회를 검증했습니다. 반원 평균 시간은 RIGHT `12.195→11.020 s(9.63%)`, LEFT
`12.401→11.085 s(10.62%)`였습니다. 여섯 주행 모두 탈출과 차선 복귀까지 완료했으며
zero 명령, timeout, `FAILED`는 없었습니다. 이 단계 역시 결승선까지 재검증하지 않았습니다.

## Obstacle 변경 이력

### 진입 gate 축소

서쪽 경계를 `x=1.08→1.28 m`로 옮겨 공통 `enter_margin=0.03 m`를 포함한 활성 위치를
`x≈1.31 m`로 만들었습니다. 직선 구간의 불필요한 `0.09 m/s` 제한을 줄인 변경입니다.

공식 시작점에서 Intersection 뒤 Obstacle 활성 시간은 `32.431→30.352 s`,
`2.079 s(6.4%)` 단축됐습니다. Intersection 완료부터 Obstacle 완료까지는
`40.850→39.688 s`, `1.162 s(2.8%)` 단축됐습니다. 최소 raw line/obstacle 여유는
`3.391/4.245 mm`였고 실패와 충돌 로그는 없었습니다. 당시 반복은 `1/1`이며 Parking
이후 완료 근거는 아닙니다.

## Parking 변경 이력

### 중앙 판정 자세 즉시 제자리회전

분기별 회전점 이동을 제거하고 LiDAR가 공간을 선택한 중앙 자세에서 바로 회전하도록
바꿨습니다. 공식 시작점에서 앞선 미션과 Zigzag까지 양쪽 주차 조건을 각각 검증했습니다.

| 빈 공간 | Parking 구간 | 선택→회전 | Zigzag 완료 시각 | 비바닥 contact |
|---|---:|---:|---:|---:|
| RIGHT | `62.851 s` | `24 ms` | `293.528 s` | 0 |
| LEFT | `62.167 s` | `20 ms` | `296.050 s` | 0 |

두 실행 모두 `PARK_IN → BACK_OUT → TURN_TO_EXIT`을 완료했고 `FAILED`가 없었습니다.
이 검증은 주차와 바로 뒤 Zigzag까지의 당시 완료 근거이며 이후 미션은 포함하지 않았습니다.

### 분기별 회전점 구현

이전 `POSITION_FOR_PARKING` 구현은 좌·우 전용 주차 회귀에서 `94.600/96.004 s`에
완료했습니다. `PARKED→BACK_OUT` 간격은 각 `0.001 s`였고 실패나 충돌은 없었습니다.
출구 tangent를 완만하게 하자 `LEAVE_AISLE`이 `21.747/21.752 s`로 직전
`30.501/32.563 s`보다 `8.754/10.811 s` 짧아졌습니다. 그 뒤 구현은 이 상태를
사용하지 않습니다.

### 원호 기반 구현과 속도 조정

원호 기반 초기 공식 주행 한 번은 안전한 odom 복귀점과 raw AMCL을 직접 비교해
`46.6 mm` 오차가 `45 mm` 조건을 넘으며 `PREPARE_EXIT`에서 실패했습니다. 좌표계가
다른 두 값을 완료 gate로 사용한 것이 원인이었습니다.

수정 뒤 같은 RIGHT 분기를 공식 시작에서 2회 반복해 주차와 Zigzag까지 완료했습니다.
두 실행의 `PREPARE_EXIT→ARC_OUT` 정지는 `0.101/0.103 s`, AMCL 왕복 차이는
`13.0 mm/1.10°`, `14.1 mm/1.20°`였습니다.

주차 목표 속도를 `0.10 m/s`로 올린 첫 접근 후보는 계획 각가속도
`0.55944 rad/s²`로 `0.55 rad/s²` 한도를 넘어 미채택했고, 접근과 차선 인계에는
`0.09 m/s`를 채택했습니다. 양쪽 공식 통합에서 주차는 `93.809/97.163 s`, Zigzag는
`18.599/18.565 s`였으며, 같은 LEFT 분기의 기존 주차보다 `12.764 s(11.61%)`
단축됐습니다.

직선 구간 상한을 `0.20 m/s`로 올리고 sweep 계산을 최적화한 뒤 4,000개 live 점의
완전정지 검사 시간은 진입 median/max `25.218/26.400 ms`, 출차
`21.695/23.009 ms`였습니다. 양쪽 공식 통합에서 주차는 `90.350/92.650 s`로 직전보다
`3.459/4.513 s` 짧았습니다. 짧은 거리와 가감속 한계 때문에 실제 주차·후진 최고
목표 속도는 약 `0.054–0.060 m/s`였습니다.

### Parking→Zigzag 인계

Zigzag 진입 cap을 `0.06→0.10 m/s`로 맞추고 센서 snapshot 저장을 긴 sweep lock과
분리했습니다. LEFT 공식 run에서 Zigzag 경로 획득은 `0.138 s`, stale 경고는
`1→0`, 인계 명령은 `0.078973→0.100355 m/s`가 됐습니다. Zigzag 완료 시간은 직전
run보다 `0.217 s` 짧았습니다.

출구 원호를 zero-end-curvature quintic과 서쪽 정렬 tail로 바꾸고 이동 중 인계하도록
한 양쪽 공식 통합 결과는 다음과 같습니다.

| 주차 | `LEAVE_AISLE→COMPLETE` | 좌회전 상태 | 마지막 주차→차선 명령 | Zigzag |
|---|---:|---:|---:|---:|
| LEFT | `23.177 s` | `6.127 s` | `0.08172→0.09689 m/s` | `16.635 s` |
| RIGHT | `22.954 s` | `6.081 s` | `0.08071→0.09828 m/s` | `16.675 s` |

LEFT의 해당 구간은 기존 `35.419 s`보다 `12.242 s(34.6%)` 짧아졌습니다. 두 실행은
Tunnel 뒤 전체 미션 완료까지 도달했지만 당시 기록은 별도 contact sensor 없이 상태와
rosout으로만 충돌을 확인했습니다.

### 진입과 출구 좌회전 형상 공통화

주차 진입과 Parking→Zigzag 좌회전은 `offset=0.195 m`, `tangent=0.078 m`, 101개
표본의 같은 zero-end-curvature quarter-turn 생성기를 사용합니다. 출구 곡선 시작은
`x=0.493 m`, 곡선 끝은 `x=0.298 m`, 정렬 tail 끝은 `x=0.226 m`입니다. 두 곡선을
각자의 시작 좌표계로 옮긴 회귀에서 최대 차이는 `1.1e-15`였습니다.

새 Gazebo를 각각 시작하고 공식 시작 자세에서 카메라 교차로, 장애물, Parking,
Zigzag를 순서대로 수행한 결과입니다. 장애물 `x=0.23/0.73 m`로 빈 공간 LEFT와
RIGHT를 각각 만들었고 두 실행 모두 Parking과 Zigzag가 `COMPLETE`였습니다.

| 빈 공간 | Parking | `LEAVE_AISLE→COMPLETE` | 공통 출구 좌회전 | Zigzag | 출구 좌회전 line/obstacle 여유 |
|---|---:|---:|---:|---:|---:|
| LEFT | `56.214 s` | `15.302 s` | `5.778 s` | `16.511 s` | `18.2/15.7 mm` |
| RIGHT | `54.565 s` | `15.252 s` | `5.786 s` | `16.620 s` | `18.2/3.2 mm` |

첫 LEFT 실행은 완료 구간의 카메라 선행조향이 `5.80→6.25°`가 되면서 이전 6° 확인
조건을 벗어나 원인을 드러냈습니다. Zigzag의 독립 진입 한도와 같은 7°로 맞춘 뒤
위 양쪽 공식 주행을 새로 수행했습니다. 실행 중 `FAILED`, `ERROR`, 충돌 관련 rosout은
없었습니다. 이 기록은 Zigzag 종료 뒤 차선 복귀까지의 완료 근거이며 결승선까지의
전체 완료 근거는 아닙니다.

## Level Crossing 변경 이력

Gazebo 차단봉의 최대 검출 거리를 `0.60→0.45 m`로 줄여 world의 개방 timer가 시작되는
`x>-1.45 m` 안까지 접근하도록 했습니다. 양쪽 주차 조건의 공식 통합에서 차단봉은 모두
완료했습니다. 세 번째 닫힘 확인 거리는 `0.373/0.367 m`, 실제 봉 제거 뒤 출발까지는
약 `0.46/0.45 s`였습니다. 정지 구간 명령 212개와 213개는 모두 차단봉 제어기의 0
속도였습니다.

직전 자세 보조 회귀 2회도 완료했지만, `y=1.17 m, yaw=+10°`와
`y=1.33 m, yaw=-10°`의 강제 사선 시작에서는 ROI에 군집 폭이나 두께가 잘리며 실제
개방 전 `PASSING`이 재현됐습니다. 정상 공식 접근에서는 재현되지 않았으며 당시 주
경로의 LiDAR 단독 판정은 유지했습니다.

## Tunnel 변경 이력

공통화 중간 `run12`는 출구의 이전 two-line 확인에서 실패했습니다. 안전성이 계산된
rolling lane path 진단으로 인계를 확인하도록 바꾼 `run15`는 Tunnel `73.371 s`, 계획
2회와 재계획 1회로 완료하고 결승선을 통과했습니다. 같은 조건의 이전 네 미션 시간과
차이는 모두 `0.5 s` 이내였고, Parking 회전 직후 non-finite 진단은 `2→0`개가 됐습니다.

당시 전체 코스 기준 `run23` Tunnel은 `73.117 s`였고 완료 시 일반 차선 제어기가
`/cmd_vel`을 소유했습니다. 이 결과의 전체 판정은 상위 README에 기록했습니다.

## 해석 원칙

- 미션 직전 시작, 비활성화된 앞선 미션, 수동 gate, 순간이동은 원인 분석용입니다.
- 최종 미션 완료는 새 Gazebo의 공식 시작 자세에서 앞선 미션을 순서대로 통과한
  경우에만 인정합니다.
- 선택 방향·주차 분기·경로 형상까지 다른 전체 시간은 속도 효과로 직접 비교하지
  않습니다.
- 실제 D405·Mid-360·OpenCR 결과는 위 Gazebo 기록으로 대체하지 않습니다.
