# Custom AutoRace description

ROBOTIS 원본을 수정하지 않고 실측 로봇과 Gazebo 센서를 구성하는 ROS Noetic Xacro 패키지입니다.

## 기준 파일

- 실측값, 계산 근거와 미확정 항목: [`HARDWARE_PARAMETERS.md`](HARDWARE_PARAMETERS.md)
- URDF/Xacro 실행값: [`urdf/robot_parameters.xacro`](urdf/robot_parameters.xacro)
- 비대칭 주행 footprint: [`../custom_autorace_bringup/config/navigation_footprint.yaml`](../custom_autorace_bringup/config/navigation_footprint.yaml)
- 기어비와 회전 방향 보정: [`../../firmware/custom_autorace_core`](../../firmware/custom_autorace_core)

하드웨어 수치는 `HARDWARE_PARAMETERS.md`에서 한 번만 설명합니다. 값을 변경할 때는 원장과 Xacro 또는 펌웨어 실행값을 함께 갱신합니다.

## 실행 방식

Xacro 확인, 센서 단독 Gazebo, 코스 이미지 교체와 odometry 비교 명령은
[`DOCKER_NOETIC.md`](../../DOCKER_NOETIC.md#로봇-모델과-센서-단독-확인)를 사용합니다.
이 문서에는 각 옵션의 동작만 설명합니다.

선택한 PNG는 SHA 기반 `/tmp` 모델에 스테이징되어 gzserver와 gzclient가 같은 모델을 사용합니다. 기본값 `course_texture:=from_model`은 원본 코스 텍스처를 사용합니다.

기본값 `tunnel_obstacle_layout:=auto`는 새 `roslaunch`마다 세 원통 배치를
`layout_a`, `layout_b`, `layout_c` 순서로 한 단계씩 바꿉니다. 기존 기본 배치가
`layout_a`였으므로 상태가 없는 환경의 첫 자동 실행은 `layout_b`입니다. gzclient는
gzserver가 선택해 연 월드를 그대로 표시합니다. 회귀 시험에서
`tunnel_obstacle_layout:=layout_a|layout_b|layout_c`를 지정하면 각각 0도, 90도,
180도로 고정되어 재현할 수 있습니다. `layout_a`는 원본 월드와 같으며, 나머지
배치는 원본을 수정하지 않고 SHA 기반 임시 월드에 스테이징됩니다.

## Odometry와 진단 토픽

Gazebo 기본값은 바퀴 회전 기반 `encoder` odometry와 encoder 전진속도·IMU 방향·각속도를 결합한 EKF입니다. Gazebo raw odom TF는 끄고 EKF만 `odom -> base_footprint`를 발행합니다.

| 토픽 | 역할 |
|---|---|
| `/camera/color/image_raw` | 가상 카메라 원본 영상 |
| `/scan_mid360_raw` | Mid-360 형식의 Gazebo 2D scan 근사 |
| `/imu` | 가상 IMU |
| `/odom` | EKF 입력용 raw odometry |
| `/odometry/filtered` | EKF 자세·속도와 6×6 공분산 |
| `/ground_truth/path` | Gazebo 실제 주행 경로 |
| `/filtered/path` | encoder+IMU EKF 추정 경로 |
| `/trajectory/comparison` | 실제·EKF 경로와 2-sigma 위치 공분산 |

통합 `custom_autorace_bringup gazebo.launch`는 RViz와 세 궤적 토픽을 기본으로
실행합니다. RViz Fixed Frame은 `map`이며 궤적의 `odom` frame은 기존 TF로
변환됩니다. `/trajectory/comparison`의 빨간 점은 Gazebo 실제 위치, 초록색은 EKF
경로, 노란색 타원은 현재 EKF 위치 공분산입니다. Ground Truth는 비교 전용이며 TF를
발행하지 않습니다. 실행과 초기화 명령은
[`DOCKER_NOETIC.md`](../../DOCKER_NOETIC.md#카메라와-토픽-확인)에 있습니다.

`fuse_imu:=false`에서는 TF가 끊기지 않도록 Gazebo raw odom TF가 다시 활성화되며 EKF TF와 동시에 발행되지 않습니다.

## 실물 적용 전 확인

- 저장소의 커스텀 OpenCR 스케치를 실물 보드에 업로드합니다.
- 최종 장착 상태에서 카메라 intrinsic, projection과 compensation을 보정합니다.
- 공통 costmap은 `navigation_footprint.yaml`의 비대칭 footprint를 사용합니다.
- 센서 잡음, 공분산, 속도와 가속도는 실물 로그와 반복 주행으로 조정합니다.

고정부 visual은 Onshape STL을 `base_link`에 맞춰 사용하고, collision은 계산량과 안정성을 위해 실측 외곽의 단순 box로 유지합니다. 세부 수치와 계산 근거는 하드웨어 원장을 참조하십시오.
