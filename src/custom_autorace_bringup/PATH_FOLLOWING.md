# 공통 경로 주행 구조

미션 경로 공통화 적용 범위는 `Intersection`, `Obstacle`, `Parking`,
`Zigzag`이다. 일반 차선은 `/detect/lane_centerline`에서 매 프레임 만든
camera rolling `CommonPath`를 30 Hz로 공통 `PathFollower`가 추종하고,
`safe_lane_controller`만 차선 제어 중 `/cmd_vel`을 발행한다. Intersection 반원도
카메라가 매 프레임 만든 rolling `CommonPath`를 같은 공통 추종기로 실행하며,
차단봉과 Tunnel은 이 공통 추종·검증 대상이 아니다. 다만 Tunnel은 portal 경로와
Hybrid A* 결과를 `CommonPath` 형식에 담고, 탈출 인계 때 일반 차선이 발행하는 공통
`/control/lane_path_diagnostics`를 사용한다.

## 이전 전 중복 조사

| 미션 | 경로·정렬 | 이전 추종·속도 | 이전 완료 판정 | 안전 검사 | 유지할 미션 판단 |
|---|---|---|---|---|---|
| Intersection | map 노드 경로를 직접 삼각함수로 odom 변환 | 최근접 점 기반 Pure Pursuit, 구간별 상수 속도, 별도 종단 감속 | 목표 원과 종단 평면을 직접 검사 | 도색 raster에서 중심선만 검사 | 카메라 LEFT/RIGHT 선택, AMCL 정렬, 반원 카메라 추종, 입·출구 인계 순서 |
| Obstacle | 측량 spline을 AMCL seed와 LiDAR 장애물 면으로 직접 정합 | 자체 lookahead·곡률 조향·속도 limiter | 경로 인덱스·출구 AMCL 여유를 각각 검사 | 별도 직사각형 checker와 LiDAR/카메라 corridor 검사 | 하나의 고정 경로, LiDAR 면 정합, 실시간 장애물·라인 판정, 재계획 없음 |
| Parking | AMCL 고정 좌표와 실시간 portal 목표를 상태별 생성 | 진입 Pure Pursuit, 직선 cross-track, 고정 반경 arc의 세 공식 | 상태마다 거리·각도·선 통과 조건을 별도 계산 | 일부 상태별 footprint/portal 검사 | 빈 좌·우 공간 선택, portal 재사용, 진입·주차·후진·복귀 상태와 확인 순서 |
| Zigzag | 측량 spline을 Gazebo odom-aligned 또는 map→odom으로 직접 변환 | 자체 lookahead·곡률 FF/FB·명령 limiter | 목표 거리·방향·출구 구역을 별도 검사 | 도색 경계에 자체 segment sweep | 고정 spline, Gazebo/실물 정렬 방식, 진입·출구 확인과 차선 인계 |

## 공통 인터페이스

각 미션 컨트롤러는 선택과 센서 기반 정렬까지만 소유한다. 정렬을 한 번 확정한 뒤
`CommonPath`를 만들고 다음 한 계산 경로를 사용한다.

```text
미션 경로 선택기 → 미션 경로 정렬기 → CommonPath
                  → SweptFootprintValidator → PathFollower
                  → 활성 미션 컨트롤러의 /cmd_vel 발행 → 차선 제어권 반환
```

일반 차선은 미션 선택·정렬 표에 포함하지 않지만, 별도 추종 공식을
두지 않는다. 30 Hz 카메라의 각 프레임 표본을 촬영 시각 odom에 고정한 rolling
`CommonPath`로
만들고, 선택된 같은 프레임 도색 경계를 `PathSafety`로 함께 고정한다. 이후 위와 같은
`SweptFootprintValidator`의 전체 직사각형·반응·제동 sweep과 `PathFollower`의 곡률
feed-forward, 방향·횡오차 feedback, 속도·가속도 제한을 사용한다.

일반 차선의 첫 관측 위치와 접선은 현재 차량 자세에서 시작하는 C1 Hermite connector로
연결한다. 조향 lookahead는 현재 선속도 `v`에 따라
`clamp(0.065 + 0.35·|v|, 0.08, 0.16) m`로 정하며, `0.20/0.26/0.28 m/s`에서 각각
`0.135/0.156/0.160 m`이다. 마지막 값에는 상한이 적용된다. 속도 선택은 별도의
현재-station preview를 사용한다. 보이는 경로의
곡률 제한과 선·각 감속 한계가 미리 뒤로 전파된 지점별 profile을 읽으므로, 먼 조향점을
직접 속도 상한으로 오용하지 않으면서 다가오는 곡선에는 선제 감속한다. 같은 source stamp의
중복·역순 카메라 입력은 시간 필터와 경로를 두 번 갱신하지 않도록 검출기 입구에서 버린다.

Tunnel의 `CommonPath` 사용은 자료 형식 재사용에 한정한다. portal connector·직선과
Hybrid A* 결과의 자세, 곡률, station, 속도 profile을 이 container에 저장하지만,
Tunnel은 자체 `calculate_tracking`·command limiter와 costmap/직사각형 충돌 검사를
계속 사용하며 공통 `PathFollower`나 `SweptFootprintValidator`를 호출하지 않는다.
탈출 때만 30 Hz rolling lane path의 공통 13항목 진단을 읽어 안전한 차선 경로를
확인한 뒤 제어권을 반환한다.

`path_following.py`가 다음을 한 번만 구현한다.

- `path_from_xy`, `path_from_poses`: 좌표 또는 몸체 자세 열에서 heading,
  curvature, station, 속도 profile과 안전 메타데이터가 든 `CommonPath` 생성
- `RigidTransform2D`, `freeze_path_in_odom`: map/미션 경로를 odom에 한 번 고정
- `project_to_path`, `sample_path`: 선분 투영, 시작 인덱스, 최근접점과 lookahead
- `SpeedProfile`, `build_speed_profile`: 곡률·각속도·횡가속도와 선형 가감속 기반 지점 속도
- `calculate_tracking`, `PathFollower`: 곡률 feed-forward와 방향·횡오차 feedback,
  전진/후진, 선속도·각속도·가속도 제한
- `goal_status`: 목표 위치·몸체 방향·가까운 종단 평면 통과를 함께 쓰는 완료 판정
- `SweptFootprintValidator`: 비대칭 직사각형의 경로점 사이 sweep, 전체 경로 여유,
  실제 자세에서 반응시간과 완전 정지까지의 sweep
- `motion_safety`: 실제 침범이 예측될 때만 안전 속도를 낮추고, 현재 정지 영역이
  안전하지 않을 때 0 속도를 지시
- `PathDiagnostics`: 진행률, 오차, 목표 속도, 곡률, 최소 라인·장애물·지도 여유와
  최종 명령

제어 tick에서는 `PathFollower.calculate_tracking(pose)`를 한 번만 호출한다. 반환된
불변 `TrackingResult`를 `motion_safety(..., tracking=...)`,
`goal_status(..., tracking=...)`, `command(..., tracking=...)`에 그대로 넘긴다. 따라서
명령과 안전 판정, 완료 판정 및 진단이 같은 투영 위치와 lookahead 목표를 사용한다.

`SpeedProfile.minimum_velocity`는 정상 주행 속도 프로파일의 명목 하한이다. 따라서
`entry_velocity`와 `exit_velocity`도 이 값보다 작게 설정할 수 없다. 다만 곡률에 따른
최대 각속도·최대 횡가속도, 각가속도와 그 제한을 미리 만족하기 위한 감속 구간은 안전
상한이므로 명목 하한보다 낮아질 수 있다. 안전 감속·정지 속도 제한 역시 이 하한보다
우선한다.

## 미션별 책임 매핑

| 미션 | 경로 선택기 | 정렬기 | `CommonPath` 생성 | 공통 검증·추종 | 차선 인계 |
|---|---|---|---|---|---|
| Intersection | AMCL 관찰 polygon에서 카메라 방향 표지판의 LEFT/RIGHT를 먼저 확정하고, 측량 `map_entry_start` 법선 평면 직전까지 차선 제어를 유지한 뒤 진입 cubic과 탈출 branch 선택 | 진입 제어권 인수 뒤 zero를 유지하며 새 EKF 표본을 기다리고, 그 실제 자세에서 선택된 반원 입구까지 chord 비례 접선 `0.50/0.25`의 단일 cubic을 만든다. 카메라 반원 종료 뒤에는 정지 후 자세에서 선택된 측량 탈출 branch의 40% 지점까지 C1 connector를 만든다. 두 시점의 최신 AMCL `map→odom` 관계를 각각 한 번 저장 | `_generate_map_entry_path`는 직접 cubic 하나를, `_generate_exit_path`는 connector·선택 branch suffix·공통 출구를 각각 하나의 `path_from_xy` 경로로 변환. 반원과 출구 이후 차선은 카메라 표본을 rolling `CommonPath`로 변환 | 고정 진입·탈출과 카메라 반원·출구 차선 모두 공통 `SweptFootprintValidator`와 `PathFollower`를 적용한다. 카메라 구간의 경계는 `safe_lane_controller`가 선택된 현재 영상의 도색에서 생성한다 | 진입 `goal_status` 직후 차선 제어에 즉시 반환하고 반원 끝에서 회수한다. 탈출 `goal_status` 직후 다시 반환하며 `VERIFY_FINAL_LANE`은 경계만 확인하고 `/cmd_vel`을 발행하지 않음 |
| Obstacle | `CourseSplinePlanner`의 미리 측량한 하나의 spline 중 현재 진행 이후 구간 선택; 실시간 장애물로 별도 우회 경로를 만들지 않음 | LiDAR 장애물 면 `align_pose`로 course 진행·횡 오프셋을 구한 뒤 `RigidTransform2D.from_pose_pair`로 course→odom 고정 | `CourseSplinePlanner.plan`이 고정 spline 나머지를 `CommonPath`로 반환하고 `freeze_path_in_odom`으로 실행 frame에 고정 | 고정 course/line과 LiDAR point를 공통 validator의 전체 sweep·`motion_safety`로 검사한 뒤 `PathFollower` | 공통 종단 판정과 출구 확인 후 동일 서비스로 반환 |
| Parking | 정지 상태의 LiDAR 점 개수로 빈 LEFT/RIGHT 주차면 선택 | 진입 직전 AMCL map 자세와 동시각 odom 자세로 `route_transform`을 고정하고, 검증된 실제 portal 자세를 후진·복귀에 재사용 | 진입 connector/curve, aisle, 주차, 후진, 복귀 arc/straight를 `_common_path_from_poses`→`path_from_poses`로 생성; 방향은 `+1/-1` | 각 segment 생성 시 공통 전체 sweep, 주행 tick에 공통 `motion_safety`와 `_common_path_command`. 탈출 경계는 도색선을 무한 직선으로 연장하지 않고 측량한 두 solid arm 끝과 그 사이의 유한한 개구를 union으로 검사 | 복귀 종단과 rolling 지그재그 차선 확인 후 반환 |
| Zigzag | YAML knot에서 생성한 하나의 고정 spline `map_path` 선택 | Gazebo는 `route_odom_aligned=true`로 identity 고정, 실물은 동시각 AMCL map/odom 자세를 `freeze_path_in_odom`에 전달 | `build_zigzag_path`가 속도 profile이 든 `CommonPath`를 생성하고 콘트롤러가 측량 corridor/paint `PathSafety`를 결합 | 활성화 전 공통 전체 sweep, 주행 tick에 line/LiDAR `motion_safety`와 `PathFollower` | 공통 종단에서 fresh 카메라 차선 6프레임을 확인하고 저속 lane join으로 넘긴 뒤, join 중 다시 6프레임을 확인해 완료 |

각 콘트롤러의 YAML이 `SpeedProfile`의 `cruise_velocity`,
`minimum_velocity`, `entry_velocity`, `exit_velocity`,
`maximum_angular_velocity`, `maximum_lateral_acceleration`,
`linear_acceleration`, `linear_deceleration`, `angular_acceleration`에 대응한다.
Intersection은 진입·탈출 profile을 구분하기 위해 같은 값에 `path_`·`exit_path_`
접두어를 사용한다. 미션 코드의 별도 속도 공식이 아니라 모두
`build_speed_profile`과 `PathFollower`의 제한을 적용한다.

## `CommonPath` 형식

| 필드 | 의미 |
|---|---|
| `x`, `y` | 실행 frame의 경로점 |
| `heading` | 각 점에서의 로봇 몸체 방향 |
| `curvature` | 누적 거리 기준 곡률 |
| `station` | 실행 순서대로 증가하는 누적 거리 |
| `speed` | 각 점의 부호 없는 목표 속도 |
| `direction` | `+1` 전진, `-1` 후진 |
| `feedforward_scale` | 실차 시험으로 확인된 경우에만 쓰는 지점별 조향 보정(기본 1) |
| `line_overlap_allowance` | 이전 제어 구간에서 이미 발생한 시작 footprint의 line overlap을 station에 따라 0으로 수렴시키는 명시적 envelope |
| `frame_id` | 고정된 경로 기준 frame |
| `goal_tolerance` | 목표 위치·방향·종단 통과 허용오차 |
| `safety` | 라인·지도 경계, 고정 장애물, 위치추정·추종 여유 |
| `line_clearance`, `obstacle_clearance`, `map_clearance` | 전체 경로 정적 검증의 최소 여유 |

후진 경로도 점은 실제 실행 순서로 저장한다. `heading`은 이동 접선이 아니라 로봇
몸체 방향이고, `direction=-1`일 때 공통 추종기가 이동 접선을 일관되게 계산한다.
한 `CommonPath` 실행 구간의 `direction`은 모두 같아야 한다. Parking처럼 방향을
바꿀 때는 현재 구간의 감속과 odom 정지를 확인한 뒤 새 전진 또는 후진 `CommonPath`를
`PathFollower.reset`으로 시작한다. 혼합 방향 배열은 가감속 제한을 건너뛸 수 없도록
경로 생성 시 거부한다.

## 안전 검사 규칙

footprint는 `front`, `rear`, `half_width`가 서로 다른 실제 직사각형이다. 라인과
지도에는 line margin을, 장애물에는 obstacle margin을 적용하고 두 경우 모두
localization error와 tracking tolerance를 더한다. 저장된 경로점만 검사하지 않고
이동 거리와 회전각 기준으로 자세를 보간한다.

고정 경로는 활성화 전에 전체 sweep를 한 번 검증한다. 제어 tick에서는 실제 자세부터
반응시간과 완전 정지까지의 sweep를 항상 검사하고, LiDAR 안전 입력을 쓰는
Obstacle·Parking·Zigzag는 sensor source stamp에 정렬된 실제 점도 포함한다. 불변 경계에
대한 명목 경로 전체 sweep는 중복 계산하지 않지만, stopping sweep에서는 생략하지 않는다.
첫 예상 접촉까지 반응거리와 제동거리가 들어가도록 속도를 제한하고, 현재의 횡오차와
방향오차를 보존한 완전 정지 sweep가 침범하면 같은 주행 상태에서 감속해 정지한다.

Parking 탈출은 두 aisle 경계를 무한히 연장한 가상 닫힌 모서리를 사용하지 않는다.
`_parking_exit_union_clearance`가 실제 texture에서 끝나는 solid arm의 유한한 범위와 그
사이의 paint-free opening을 함께 표현한다. 따라서 합법적인 탈출 개구는 통과시키되
실제 solid arm을 가로지르는 경로는 그대로 거부한다.

## 공통화 이후 실제 검증 범위

아래 표의 첫 두 행은 현재 기준이고 나머지는 비교를 위해 보존한 역사적 기록이다.
역사적 Intersection 결과는 현재 구현의 완료 판정으로 사용하지 않는다. 역사적 전용
시작점 회귀도 미션 단위 검증일 뿐 현재 공식 시작점 통합 완료를 뜻하지 않는다.

| 검증 | 실제 도달 상태 | 판정 |
|---|---|---|
| 최종 공식 시작점 통합 run23 | 새 Gazebo의 공식 시작점에서 `0.28 m/s` adaptive lookahead 코드로 출발해 실제 카메라가 선택한 LEFT Intersection부터 `Obstacle → Parking LEFT → Zigzag → Level Crossing → Tunnel`까지 순서대로 완료하고 Tunnel 뒤 결승선 footprint 통과까지 확인. 최종 [result.json](../../diagnostics/official_full_camera_rolling_common_swept_adaptive_20260913_run23/result.json)의 판정 항목 `22/22`가 모두 참이고 실패 항목 `0`. 출발부터 결승선까지 `280.472 s`; 전체 주행 구간 `/detect/lane_centerline`은 `29.672 Hz`, 최대 출력 간격 `57 ms`, 중복·역순 source stamp `0` | `COMPLETE, 22/22 PASS` |
| 최신 빌드·등록 시험 | `custom_autorace_bringup`에 등록된 `504 tests`와 전체 workspace의 `557 tests`가 모두 통과. `0 errors`, `0 failures`, `0 skipped` | `OK` |
| 역사적: Intersection actual-pose 직접 cubic | 방향마다 Gazebo와 관련 노드를 새로 시작하고 공식 자세, `forced_direction=0`, 실제 카메라로 검증. RIGHT는 진입 `10.323 s`/전체 `34.836 s`, 경로 순회전 `-68.386°`, 반대 회전과 반대 명령 `0`; LEFT는 진입 `7.378 s`/전체 `32.490 s`, 순회전 `+110.360°`, 반대 회전과 반대 명령 `0`. 두 방향 모두 인계 뒤 fresh EKF 자세와 경로 시작점 차이 `0 m`, 진입·탈출 endpoint allowance `0`, 안전 정지·wrong-turn·timeout·`FAILED` `0회`, 최종 차선 제어권 반환까지 사람 개입 없이 완료. 당시 변경 뒤 결승선까지는 다시 검증하지 않았음 | `LEFT 1/1, RIGHT 1/1 COMPLETE (역사적)` |
| 이전 공통화 빌드·등록 시험 | `catkin_make` 성공, 당시 패키지 등록 시험 `350 tests` 실행(`0 errors`, `0 failures`, `0 skipped`) | `OK (이전 기준)` |
| 이전 공식 시작점 통합 `gazebo.launch`(공통화 대상 4개 미션) | 새 Gazebo의 odom `(0.8000038, -1.7469952, 0.0063°)`에서 출발해 `forced_direction=0`인 실제 카메라가 선택한 RIGHT branch로 Intersection `177.749 s`, Obstacle `218.599 s`, LiDAR가 선택한 Parking LEFT(후진 포함) `331.192 s`, Zigzag `350.061 s`에 차례로 `COMPLETE`. `350.104 s`에 실제 `/safe_lane_controller`의 다음 `/cmd_vel`도 확인했다. 사람 개입, 순간이동, 수동 gate 발행은 없었다. 시작 odom부터 Zigzag 완료까지 `233.452 s`, 차선 명령 재개까지 `233.495 s`였다. | `COMPLETE (이전 기준)` |
| 이전 Obstacle 진입 gate 축소 뒤 공식 시작점 통합 | 새 Gazebo의 공식 자세에서 실제 녹색 신호와 카메라 RIGHT를 거쳐 Intersection `176.336 s` 완료 후 Obstacle `185.672→216.024 s` 완료, 차선 제어권 반환과 다음 Parking gate `218.651 s`까지 사람 개입 없이 진행했다. 같은 공식 시작·실제 RIGHT 기준의 이전 Obstacle 활성 시간 `32.431 s`에서 `30.352 s`로 `2.079 s(6.4%)`, Intersection 완료부터 Obstacle 완료까지 `40.850 s`에서 `39.688 s`로 `1.162 s(2.8%)` 단축됐다. 최소 raw line/obstacle 여유는 각각 `3.391/4.245 mm`였고 `FAILED`·충돌 로그는 없었다. | `1/1 COMPLETE (이전 기준)` |
| 이전 20% connector Intersection 공식 시작점 반복 | 방향 관찰과 AMCL 진입면 인계를 분리했지만 진입은 고정 경로의 20% 지점에 접속하던 코드의 기록. 새 Gazebo에서 실제 카메라 `RIGHT, LEFT, LEFT, RIGHT`를 연속 실행했고 네 번 모두 진입·반원·별도 탈출·최종 차선 복귀까지 사람 개입 없이 완료. 총 시간 `38.977, 38.256, 34.247, 37.747 s`; 진입 안전 정지와 timeout `0회`. 현재 live-pose direct cubic 검증 횟수에는 포함하지 않음 | `4/4 COMPLETE (이전 기준)` |
| 이전 RIGHT Intersection-only 기준 | adaptive 인계 경로를 넣기 전 공통 추종 구조의 전용 회귀. 총 `31.140 s`, 진입 `5.852 s`, 탈출 `11.364 s`, 최종 위치 오차 `0.020 m`, 방향 오차 `0.0°` | `COMPLETE (역사적)` |
| 이전 LEFT Intersection-only 기준 | 같은 이전 전용 회귀. 총 `28.738 s`, 진입 `4.004 s`, 탈출 `11.715 s`, 최종 위치 오차 `0.024 m`, 방향 오차 `0.1°` | `COMPLETE (역사적)` |
| 역사적: Obstacle 전용 | LiDAR 면 정렬 뒤 `ACQUIRING → AVOIDING → REJOINING → COMPLETE`, 차선 제어권 반환. 최대 경로 오차 `4.8088 mm`, 최소 line 여유 `3.39056 mm`, 최소 obstacle 여유 `4.24547 mm`, 최종 진행률 `98.09%` | `COMPLETE (역사적)` |
| 역사적: Parking LEFT 전용 | 진입·주차·후진·복귀의 공통 경로 구간을 모두 실행하고 차선 제어권 반환. 최대 위치 오차 `8.7069 mm`, 최대 횡오차 `6.8366 mm`, 최소 line 여유 `0.98794 mm`, 최소 obstacle 여유 `2.11462 mm` | `COMPLETE (역사적)` |
| 역사적: Parking RIGHT 전용 | LEFT와 같은 상태 전이와 차선 제어권 반환. 최대 위치 오차 `11.2714 mm`, 최대 횡오차 `6.8695 mm`, 최소 line 여유 `0.67022 mm`, 최소 obstacle 여유 `1.43997 mm` | `COMPLETE (역사적)` |
| 역사적: Zigzag 전용 | `ACQUIRING → FOLLOWING → VERIFY_EXIT → JOINING_LANE → COMPLETE`, 차선 제어권 반환. `25.956 s`, 최대 경로 오차 `4.5 mm`, 최대 방향 오차 `3.41°`, 최소 공통 line 여유 `3.5155 mm`, 최소 LiDAR 여유 약 `30.2 mm` | `COMPLETE (역사적)` |

run23의 공식 시작점 통합 주행에서 공통 13항목 진단을 active부터 `COMPLETE`까지 집계한
현재 기준값은 다음과 같다. 네 미션 모두 malformed 또는 non-finite core 표본은 0개였다.

| 현재 run23 미션 | 수행시간 | 최대 위치 오차 | 최대 절대 횡오차 | 최소 raw line 여유 | 최소 obstacle 여유 | 최소 map 여유 |
|---|---:|---:|---:|---:|---:|---:|
| Intersection | `33.909 s` | `7.003 mm` | `7.003 mm` | `-9.422 mm`* | 경계 없음(`∞`) | `312.704 mm` |
| Obstacle | `27.610 s` | `5.419 mm` | `5.418 mm` | `3.391 mm` | `4.245 mm` | `118.879 mm` |
| Parking LEFT | `62.944 s` | `6.231 mm` | `6.231 mm` | `3.100 mm` | `8.867 mm` | `126.049 mm` |
| Zigzag | `16.792 s` | `16.542 mm` | `4.690 mm` | `2.857 mm` | `30.610 mm` | `98.227 mm` |

\* Intersection의 raw 음수값은 진입·탈출 시작 자세에 명시한 station별
`line_overlap_allowance` 적용 전 수치다. 허용량은 시작 겹침에서 0으로 수렴하며 공통
validator 판정은 전 구간 PASS였다. 현재 run23 Parking의 line·obstacle·map raw 여유는
모두 양수다. run23의 나머지 미션 시간은 Level Crossing `16.616 s`, Tunnel
`73.117 s`다. Parking 후진은 `7.491 s` 동안 음수 명령 151개, 최저
`-0.08234 m/s`로 확인됐다.

같은 공식 시작 자세와 RIGHT 주차 조건으로 기록한 직전 후보 run12를 같은 offline
analyzer에 넣은 회귀 비교는 다음과 같다. run12도 공통화 중간 후보이므로 이 표는 기존
legacy 추종기 대비 성능 A/B가 아니라 마지막 수정의 반복성·회귀 확인이다.

| 미션 | run12 | run15 | 차이(run15-run12) |
|---|---:|---:|---:|
| Intersection | `31.551 s` | `32.035 s` | `+0.484 s` |
| Obstacle | `27.795 s` | `27.627 s` | `-0.168 s` |
| Parking RIGHT | `63.966 s` | `63.954 s` | `-0.012 s` |
| Zigzag | `16.516 s` | `16.648 s` | `+0.132 s` |
| Level Crossing | `16.703 s` | `16.716 s` | `+0.013 s` |
| Tunnel | 출구의 이전 two-line 확인에서 `FAILED` | `73.371 s`, `COMPLETE` | 결승선까지 복구 |

run12 Parking에는 회전 시작 직후 non-finite 진단 2개가 있었으나 run15은 0개다. 앞의
네 미션 시간 편차는 `0.5 s` 이내였고, Tunnel은 raw 두 선 재판독 대신 안전성이 이미
계산된 rolling path 진단을 사용하면서 정상 인계·완료·결승 통과로 바뀌었다.

아래 이전 공식 통합 표는 같은 공식 출발 자세에서 얻은 역사 기준이지만 선택 방향과
주차 분기가 run15와 같지 않으므로 엄밀한 동일-분기 A/B로 해석하지 않는다. 현재 구조의
반복 재현성·실물 성능 비교에는 동일한 방향과 주차 조건을 고정한 추가 run이 필요하다.

위 이전 공식 시작점 통합 주행에서 각 제어기의 첫 활성 상태부터 `COMPLETE`까지 걸린 시간은
Intersection `40.179 s`(`137.570→177.749 s`), Obstacle `32.431 s`
(`186.168→218.599 s`), Parking `109.926 s`(`221.266→331.192 s`), Zigzag
`18.779 s`(`331.282→350.061 s`)였다. 당시 교차로 제어기 로그는 실제 카메라가
RIGHT를 3프레임으로 확정하고 이전 adaptive RIGHT 진입·탈출 경로를 생성한 것도
기록했다.
이 수치는 네 미션을 모두 통과한 당시 기준 run의 기록이며, 위의 Obstacle gate 축소
run은 Obstacle 완료와 다음 차선 복귀까지만 검증한 당시 최신 목표 미션 결과다.

| 이전 공식 통합 미션 | 최대 위치 오차 | 최대 절대 횡오차 | 최소 raw line 여유 | 최소 raw obstacle 여유 | 최소 raw map 여유 |
|---|---:|---:|---:|---:|---:|
| Intersection | `23.483 mm` | `23.483 mm` | `-54.117 mm` | 경계 없음(`∞`) | `484.418 mm` |
| Obstacle | `19.400 mm` | `19.400 mm` | `3.391 mm` | `4.245 mm` | 경계 없음(`∞`) |
| Parking LEFT | `9.517 mm` | `7.013 mm` | `1.093 mm` | `-7.882 mm` | `126.377 mm` |
| Zigzag | `20.521 mm` | `4.636 mm` | `2.871 mm` | `22.348 mm` | 경계 없음(`∞`) |

다음 설명도 위 이전 공식 통합 run에 해당한다. 이 표의 여유는 공통 13값 진단에서
주행 전체의 최솟값을 취한 **raw signed clearance**다. `∞`는 해당 미션의
`PathSafety`에 그 종류의 경계가 없었다는 뜻이다.
Intersection의 음수 line 값은 실제 pose의 침범량이 아니라 전체 계획 sweep에서 얻은
allowance 적용 전 최소값이다. 안전 판정에는 최대 `60 mm`에서 station에 따라 0으로
수렴하는 명시적 `line_overlap_allowance`가 각 동일 sample에 더해졌고 전체 sweep의 유효
여유는 양수였다. 다만 모니터가 raw 최소점과 그 지점 allowance를 함께 저장하지 않아 정확한
유효 최솟값은 복원할 수 없다. 활성화 순간의 별도 로그에 기록된 raw overlap은 진입
`29.4 mm`, 탈출 `2.4 mm`이며, station 0 allowance를 적용한 유효 여유는 각각
`30.6 mm`, `22.6 mm`다. Parking의 음수 obstacle 값은 runtime LiDAR를
포함한 route/정지 sweep 중 한 obstacle point가 확장 footprint 안에 들어왔음을 뜻하는
일시적 predicted intrusion의 raw 최솟값이다. Parking 구현은 선택한 빈 공간의 LiDAR
점을 별도로 제외하지 않으므로 이 값에 임의의 제외 여유를 더하지 않는다. 공통
`motion_safety`가 이 signed 값을 속도 제한·정지 판단에 사용했고, 주행은 `FAILED`나
충돌 보고 없이 완료했다. 이 집계만으로 그 순간이 route 예측 감속인지 정지 sweep의
0 속도 판단인지, 또는 고정점과 live 점 중 어느 입력인지는 분리할 수 없다.

이전 Intersection LEFT 전용 회귀의 최소 raw line 여유는 `5.441 mm`였다.
RIGHT 탈출은 카메라 반원이 넘겨준 자세에서 raw overlap `4.831 mm`로 시작했고,
`CommonPath.line_overlap_allowance`가 명시한 탈출 수렴 구간 안에서만 이를 허용한 후 0으로
수렴했다. 위 이전 전용 회귀로 네 미션의 공통 경로 생성·검증·추종과 완료 뒤
차선 인계를 확인했다.

현재 공통 추종 구조는 run23에서 네 대상 미션과 일반 차선 구간을 포함한 공식 시작점
통합 기준으로 검증됐고, 같은 run이 Level Crossing·Tunnel까지 여섯 미션과 결승선을
완료했다. 다만 Level Crossing과 Tunnel의 완료는 프로젝트 전체 통합의 근거이지 공통
`PathFollower`·`SweptFootprintValidator` 적용의 근거는 아니다. 특히 Tunnel의
`CommonPath` container와 lane 진단 재사용도 공통화 적용 범위를 넓히지 않는다. 위의
나머지 기록은 당시 구현 비교용이며, 방향·주차 분기별 반복 재현성 비교는 별도 후속
검증 범위다. 실제 장착 D405의 원근 보정, Mid-360 motion distortion, OpenCR 응답과
실제 도색에서의 30 Hz 경로 주행은 아직 검증하지 않았다.

공통 단위 테스트와 네 미션 회귀는 `CMakeLists.txt`의
`catkin_add_nosetests(...)`에 등록되어 있다. 정상 handoff와 수동 정지에서는 현재
`/cmd_vel` 소유자만 정지 명령을 발행하며, 새 relay나 중복 경로 계산 노드는 없다.

## 진단 배열

각 대상 미션의 `*/diagnostics`와 일반 차선의
`/control/lane_path_diagnostics`는 아래 순서의 공통 13개 값을
`Float64MultiArray`로 발행한다.

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

Tunnel 고유의 `/tunnel/diagnostics`는 이 공통 배열이 아니다. Tunnel은 탈출 확인과
저속 lane join 완료를 위해 `safe_lane_controller`의
`/control/lane_path_diagnostics`를 구독하며, 유효한 rolling `CommonPath`의 양수 line
여유가 든 fresh 결과를 확인한다.

새 `/cmd_vel` relay는 없다. 서비스 기반 미션 인계로 활성화된 제어기 하나만
`/cmd_vel`을 발행한다. 네 공통화 대상과 일반 차선은 공통 라이브러리가 반환한
`Twist`를 사용하고, Level Crossing과 Tunnel은 각각의 비공통 제어 계산을 사용한다.
