# 공통 경로 주행 구조

공통 경로 적용 범위는 `Intersection`, `Obstacle`, `Parking`, `Zigzag`와 일반 카메라
차선입니다. Level Crossing의 정지·재출발과 Tunnel의 Hybrid A* 추종·충돌 검사는
대상에서 제외합니다. Tunnel은 입·출구와 계획 결과를 `CommonPath` 자료형에 담지만
공통 follower와 validator를 사용하지 않습니다.

실행·회귀·진단 명령은 저장소 루트의
[`DOCKER_NOETIC.md`](../../DOCKER_NOETIC.md), 전체 런타임과 미션 상태는
[`README.md`](README.md), 과거 A/B 결과는
[`docs/VALIDATION_HISTORY.md`](docs/VALIDATION_HISTORY.md)를 참조합니다.

## 이전 전 중복 조사

| 미션 | 경로·정렬 | 이전 추종·속도 | 이전 완료 판정 | 안전 검사 | 유지한 미션 판단 |
|---|---|---|---|---|---|
| Intersection | map 경로를 미션 코드에서 odom으로 변환 | 최근접점 Pure Pursuit, 구간별 상수 속도 | 목표 원과 종단 평면 직접 검사 | 도색 raster 중심선 검사 | 카메라 방향 선택, AMCL 정렬, 반원 차선 추종, 입·출구 인계 |
| Obstacle | 측량 spline을 AMCL seed와 LiDAR 면으로 정합 | 자체 lookahead·곡률 조향·limiter | 경로 인덱스와 출구 AMCL 여유 검사 | 별도 직사각형 checker | 단일 고정 경로, LiDAR 면 정합, 실시간 장애물·라인 판단 |
| Parking | AMCL 고정 좌표와 portal 목표를 상태별 생성 | Pure Pursuit, cross-track, 고정 반경 arc | 상태마다 거리·각도·선 통과 검사 | 상태별 footprint/portal 검사 | 빈 공간 선택, 전진·후진 상태와 확인 순서 |
| Zigzag | spline을 odom-aligned 또는 map→odom 변환 | 자체 lookahead·곡률 FF/FB·limiter | 목표 거리·방향·출구 구역 검사 | 자체 segment sweep | 고정 spline, 환경별 정렬, 진입·출구와 차선 인계 |

## 공통 계산 경로

각 미션 컨트롤러는 경로 선택과 센서 기반 정렬까지만 소유합니다. 실행 경로를 odom에
한 번 고정한 뒤 모든 계산은 다음 순서를 따릅니다.

```text
미션 경로 선택기
  → 미션 경로 정렬기
  → CommonPath 생성
  → SweptFootprintValidator
  → PathFollower
  → 활성 제어기의 /cmd_vel 발행
  → 일반 차선 제어권 반환
```

일반 차선은 30 Hz 카메라의 각 유효 프레임에서 짧은 `CommonPath`를 새로 만들고,
촬영 시각 odom 자세와 같은 프레임의 도색 경계를 함께 고정합니다. 첫 관측점까지는 현재
차량 자세와 접선을 시작 조건으로 하는 C1 Hermite connector를 사용합니다.

제어 tick에서는 `PathFollower.calculate_tracking(pose)`를 한 번 호출하고 그
`TrackingResult`를 안전, 완료, 명령과 진단 계산에 그대로 전달합니다. 따라서 한 tick의
최근접점, lookahead와 오차가 서로 다르게 재계산되지 않습니다.

## 라이브러리 인터페이스

`src/custom_autorace_bringup/path_following.py`가 다음 기능을 한 번만 구현합니다.

- `path_from_xy`, `path_from_poses`: 위치 또는 자세열에서 `CommonPath` 생성
- `RigidTransform2D`, `freeze_path_in_odom`: map·미션 경로를 odom에 고정
- `project_to_path`, `sample_path`: 선분 투영, 시작 인덱스, 최근접점과 lookahead 계산
- `SpeedProfile`, `build_speed_profile`: 곡률·각속도·횡가속도와 가감속 기반 속도 profile
- `calculate_tracking`, `PathFollower`: 곡률 feed-forward, 방향·횡오차 feedback,
  전진·후진과 명령 제한
- `goal_status`: 목표 위치·몸체 방향·종단 평면 통과를 결합한 완료 판정
- `SweptFootprintValidator`: 비대칭 직사각형의 경로점 사이, 전체 경로와 정지 영역 sweep
- `motion_safety`: 예상 침범까지의 거리에 따른 안전 속도와 정지 판단
- `PathDiagnostics`: 진행률, 오차, 속도, 곡률, 최소 여유와 최종 명령

## `CommonPath` 형식

| 필드 | 의미 |
|---|---|
| `x`, `y` | 실행 frame의 경로점 |
| `heading` | 각 점의 로봇 몸체 방향 |
| `curvature` | 누적 거리 기준 곡률 |
| `station` | 실행 순서대로 증가하는 누적 거리 |
| `speed` | 각 점의 부호 없는 목표 속도 |
| `direction` | `+1` 전진, `-1` 후진 |
| `feedforward_scale` | 시험으로 확인된 지점별 곡률 보정, 기본 `1` |
| `line_overlap_allowance` | 시작 footprint의 기존 line 겹침을 station에 따라 0으로 수렴시키는 envelope |
| `frame_id` | 고정된 실행 frame |
| `goal_tolerance` | 목표 위치·방향·종단 통과 허용오차 |
| `safety` | 라인·지도 경계, 고정 장애물, 위치추정·추종 여유 |
| `line_clearance`, `obstacle_clearance`, `map_clearance` | 전체 정적 sweep의 최소 여유 |

후진도 경로점은 실제 실행 순서로 저장하며 `heading`은 이동 접선이 아니라 몸체 방향입니다.
한 경로 구간의 `direction`은 모두 같아야 합니다. Parking처럼 방향이 바뀌면 현재 구간을
감속하고 odom 정지를 확인한 뒤 새 `CommonPath`로 follower를 reset합니다.

## 추종과 속도 제한

공통 follower는 경로 곡률 feed-forward에 방향 오차와 횡방향 오차 feedback을 더합니다.
lookahead 목표와 그 사이의 속도 profile을 함께 사용하며, 다음 제한을 적용합니다.

- 지점별 목표 속도와 미션별 최고 선속도
- 최고 각속도
- 최대 횡가속도
- 선속도 가속·감속
- 각속도 가속
- 곡률과 각가속도 제한을 미리 만족시키기 위한 forward/backward 속도 전파

`SpeedProfile.minimum_velocity`는 정상 profile의 명목 하한입니다. 곡률·각속도·횡가속도
한계와 이를 준비하는 감속 구간, 안전 감속·정지는 이 명목 하한보다 우선할 수 있습니다.

차량 공통 기본식과 게인은 공유하고, 미션 차이는 주로 YAML의 다음 속도 항목으로
표현합니다.

| 항목 | 역할 |
|---|---|
| `cruise_velocity` | 직선 순항 속도 |
| `minimum_velocity` | 정상 profile의 명목 하한 |
| `entry_velocity`, `exit_velocity` | 경로 시작·종단 속도 |
| `maximum_angular_velocity` | 각속도 상한 |
| `maximum_lateral_acceleration` | 곡률 기반 선속도 상한 |
| `linear_acceleration`, `linear_deceleration` | 선속도 slew 제한 |
| `angular_acceleration` | 각속도 slew 제한 |

Intersection은 진입·탈출 profile을 분리하기 위해 `path_`와 `exit_path_` 접두어를
사용합니다. 각 컨트롤러는 별도 속도 공식을 만들지 않고 같은 profile builder와
follower 제한을 사용합니다.

## 안전 검사

footprint는 `front`, `rear`, `half_width`가 서로 다른 실제 직사각형입니다. 저장된
경로점만 검사하지 않고 이동 거리와 회전각 기준으로 자세를 보간해 경로점 사이까지
sweep합니다.

- 고정 라인과 지도 경계에는 line margin 적용
- 장애물에는 obstacle margin 적용
- 두 경우 모두 위치추정 오차와 추종 허용오차 포함
- 고정 경로는 활성화 전에 전체 sweep 검증
- 제어 중에는 실제 자세부터 반응시간과 완전 정지까지의 sweep 검증
- Obstacle·Parking·Zigzag는 sensor source stamp에 정렬된 LiDAR 점 포함

첫 예상 접촉이 정지거리 안으로 들어오면 안전 속도를 낮추고, 현재 운동 상태에서 완전히
정지하는 sweep가 침범하면 0 속도를 지시합니다. 예상 침범이 없으면 명목 profile을
그대로 유지합니다. 불변 경계의 전체 sweep는 중복 계산하지 않지만 현재 정지 영역에서는
생략하지 않습니다.

Parking 탈출 경계는 두 aisle 선을 무한 연장하지 않습니다. 실제 texture에서 끝나는
solid arm과 그 사이의 paint-free opening을 합성해 합법적인 개구는 통과시키고 실제
도색 arm을 가로지르는 경로만 거부합니다.

## 미션별 책임

| 미션 | 선택기 | 정렬기 | 경로 생성 | 공통 실행 | 완료와 인계 |
|---|---|---|---|---|---|
| Intersection | 카메라 LEFT/RIGHT | 진입·탈출 때 최신 AMCL map→odom | 실제 인계 자세의 진입 cubic, 방향 branch와 공통 출구; 반원은 rolling lane path | 모든 구간 validator와 follower 공유 | 진입 뒤 반원에 반환, 탈출 뒤 최종 차선에 반환 |
| Obstacle | 측량된 단일 spline | AMCL seed 뒤 LiDAR 장애물 면으로 course→odom | 남은 spline suffix와 곡률 연속 출구 | 고정 라인·course와 live LiDAR 검사 | 종단과 출구 여유 확인 뒤 반환 |
| Parking | LiDAR 좌·우 ROI로 빈 공간 선택 | 인계 시점 map→odom 고정, 실행은 local odom | 진입·주차·후진·복귀를 방향별 여러 구간으로 생성 | 이동과 제자리회전의 전체·정지 sweep | rolling 차선 확인 뒤 이동 중 반환 |
| Zigzag | YAML 고정 spline | Gazebo identity, 실물 동시각 AMCL map→odom | 현재 투영점 이후 suffix | 바깥 도색·live LiDAR 검사 | 종단과 새 차선 확인 뒤 저속 반환 |

Level Crossing과 Tunnel의 상태·제어 책임은 [`README.md`](README.md)에 설명합니다.

## 공통 진단 배열

대상 미션의 `*/diagnostics`와 일반 차선의 `/control/lane_path_diagnostics`는 다음 순서의
`Float64MultiArray` 13개 값을 발행합니다.

| 인덱스 | 값 |
|---:|---|
| 0 | 진행률 `0..1` |
| 1 | 남은 경로 거리 |
| 2 | 최근접 경로 위치 오차 |
| 3 | 부호 있는 횡방향 오차 |
| 4 | 방향 오차 |
| 5 | 경로점 목표 속도 |
| 6 | lookahead 목표 인덱스 |
| 7 | 명령 곡률 |
| 8 | 최소 라인 여유 |
| 9 | 최소 장애물 여유 |
| 10 | 최소 지도 경계 여유 |
| 11 | 제한 후 선속도 명령 |
| 12 | 제한 후 각속도 명령 |

Tunnel 고유 `/tunnel/diagnostics`는 이 배열이 아닙니다. Tunnel은 출구 인계를 위해
일반 차선의 fresh 공통 진단에서 유효 rolling 경로와 양수 line 여유를 확인합니다.

## 현재 검증 범위

`run23`은 Gazebo 공식 시작점에서 일반 차선과 네 공통화 대상 미션을 거쳐 차단봉,
Tunnel과 결승선까지 완료했습니다. malformed 또는 non-finite 공통 진단 표본은 네
미션 모두 0개였습니다.

| 미션 | 수행시간 | 최대 위치 오차 | 최대 절대 횡오차 | 최소 raw line 여유 | 최소 obstacle 여유 | 최소 map 여유 |
|---|---:|---:|---:|---:|---:|---:|
| Intersection | `33.909 s` | `7.003 mm` | `7.003 mm` | `-9.422 mm`* | 경계 없음 | `312.704 mm` |
| Obstacle | `27.610 s` | `5.419 mm` | `5.418 mm` | `3.391 mm` | `4.245 mm` | `118.879 mm` |
| Parking LEFT | `62.944 s` | `6.231 mm` | `6.231 mm` | `3.100 mm` | `8.867 mm` | `126.049 mm` |
| Zigzag | `16.792 s` | `16.542 mm` | `4.690 mm` | `2.857 mm` | `30.610 mm` | `98.227 mm` |

\* Intersection의 raw 음수값은 시작 자세의 명시적 `line_overlap_allowance` 적용 전
값입니다. 허용량은 시작 station에서 0으로 수렴하며 validator의 유효 판정은 전 구간
PASS였습니다. 같은 run에서 Parking 후진은 `7.491 s`, 음수 명령 151개, 최저
`-0.08234 m/s`로 확인했습니다.

당시 등록된 bringup `504`개와 전체 workspace `557`개 시험도 오류·실패·건너뜀 없이
통과했습니다. 이 수치는 해당 코드 상태의 검증 기록이며 현재 작업트리의 자동 보증으로
해석하지 않습니다.

실제 D405·Mid-360·OpenCR과 실제 경기장에서는 아직 공식 시작점 통합 주행을 수행하지
않았습니다. 실물 보정과 전체 검증이 끝날 때까지 완료 범위는 Gazebo로 한정합니다.
