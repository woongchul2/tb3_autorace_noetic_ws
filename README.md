# TurtleBot3 AutoRace Noetic

고정된 AutoRace 경기장에서 TurtleBot3가 규정을 지키며 모든 미션을 순서대로
완수하고, 반복 주행의 안정성을 유지하면서 결승선까지의 시간을 줄이는 ROS Noetic
workspace다. 프로젝트 고유 제어기와 설정은 `src/custom_autorace_bringup`, 로봇·센서
구성은 `src/custom_autorace_description`에 있다.

## 빠른 실행

### Docker / Gazebo

호스트에 Docker Compose v2, Terminator와 X11이 필요하다.

```bash
git clone https://github.com/woongchul2/tb3_autorace_noetic_ws.git
cd tb3_autorace_noetic_ws
./docker/start_sim_terminator.sh
```

열린 Terminator 패널에서 빌드 후 통합 launch를 실행한다.

```bash
./docker/build_workspace.sh
source /workspace/devel/setup.bash
roslaunch custom_autorace_bringup gazebo.launch
```

`gazebo.launch`가 Gazebo와 전체 미션을 함께 시작하므로 Gazebo를 별도로 실행하지 않는다.

### 실물 로봇

현재 `hardware.launch`는 OpenCR, D405, Mid-360과 인지만 실행한다. AMCL, 차선·미션
제어기와 실물용 지도는 포함하지 않으며 `/cmd_vel`을 발행하지 않는다.

실행 전에 다음 항목이 필요하다.

- [`firmware/custom_autorace_core`](firmware/custom_autorace_core)를 OpenCR에 업로드
- [`d405_projection_uncalibrated.yaml`](src/custom_autorace_bringup/config/d405_projection_uncalibrated.yaml)과
  [`lane_detector_d405_uncalibrated.yaml`](src/custom_autorace_bringup/config/lane_detector_d405_uncalibrated.yaml)을
  별도 측정 파일로 복사한 뒤 장착 상태에서 보정
- [`MID360_config.json`](src/custom_autorace_bringup/config/MID360_config.json)의
  호스트·LiDAR IP를 실제 값으로 변경
- 컨테이너에 D405 USB와 `/dev/ttyACM0` 전달

기본 `compose.noetic.yaml`에는 USB·serial 장치 전달이 설정되어 있지 않다.
보정 파일을 만든 뒤 ROS Noetic 셸에서 센서·인지 bringup을 실행한다.

```bash
roslaunch custom_autorace_bringup hardware.launch \
  projection_config:=/workspace/src/custom_autorace_bringup/config/d405_projection_measured.yaml \
  lane_detector_config:=/workspace/src/custom_autorace_bringup/config/lane_detector_d405_measured.yaml
```

실물 전체 자동주행에는 실물 odom·AMCL 설정, 각 미션 형상의 sensor-relative 등록값과
이를 묶는 실물 통합 launch가 추가로 필요하다. 연결 직선 전체를 다시 측량할 필요는
없으며 Gazebo용 좌표와 미션 YAML은 실물에 사용하지 않는다.

## 현재 검증 기준

2026-09-18의 현재 adaptive registration 코드 공식 시작점 통합 **run20**에서 실제
카메라가 선택한 Intersection LEFT부터 Obstacle, Parking LEFT, Zigzag,
Level Crossing, Tunnel layout B를 순서대로 완료한 뒤 결승선 footprint를 통과했다.

- 출발 명령부터 결승선 통과까지: **283.053 s**
- 6개 미션 `COMPLETE`, `FAILED` 0건, `/cmd_vel` 발행자 교차 0건
- 반대 조건 반복: Intersection RIGHT, Parking RIGHT, Tunnel layout C도 전 미션·결승 완료
- 자동 회귀: bringup **669개**, description **7개**, 합계 **676개**가
  `0 errors`, `0 failures`로 통과

`run20` bag 종합 판정은 recorder 시각 설정과 차선 토픽 녹화 누락 때문에 `19/22`였고,
미션 순서·공통 진단·제어권·결승 검사는 모두 통과했다. 실제 D405·Mid-360·OpenCR
주행은 아직 검증하지 않았다. 상세 조건, 미션별 시간과 여유 값은
[검증 이력](src/custom_autorace_bringup/docs/VALIDATION_HISTORY.md)을 따른다.

## 문서 지도

- [실행·빌드·시험 명령](DOCKER_NOETIC.md): 사용자가 실행할 명령과 각 명령의 역할
- [공통 경로 주행 구조](src/custom_autorace_bringup/PATH_FOLLOWING.md): 경로 형식,
  검증기·추종기 구조와 회귀 기록
- [Bringup 패키지](src/custom_autorace_bringup/README.md): launch, 설정과 노드 구성
- [로봇 설명 패키지](src/custom_autorace_description/README.md): URDF와 센서 구성
- [실물 하드웨어 파라미터](src/custom_autorace_description/HARDWARE_PARAMETERS.md):
  D405, Mid-360, OpenCR 기준값
- [Upstream 출처](UPSTREAM_SOURCES.md): monorepo에 포함한 외부 소스의 기준 revision
- [진단 자료 보존 기준](DIAGNOSTICS_RETENTION.md): 로컬 원시 기록과 요약 파일의
  보존 범위
- [작업 원칙](AGENTS.md): 구현 및 최종 검증 기준
- [Notion 작성 규칙](NOTION_GUIDE.md): AutoRace 관련 Notion 문서 규칙

README에는 최초 실행만 두고, 나머지 실행 옵션과 시험 명령은 `DOCKER_NOETIC.md`를
단일 기준으로 유지한다.

## 진단 자료 방침

ROS bag을 포함한 원시 진단 자료는 크기가 크고 실행마다 생성되므로 Git 저장소에
포함하지 않는다. 로컬 `diagnostics/` 또는 별도 보관소에 원본을 유지하고, 재현에 필요한
조건과 최종 지표만 추적되는 문서에 남긴다. 따라서 저장소를 clone해도 run20·run21의
원시 bag은 내려받아지지 않는다.
