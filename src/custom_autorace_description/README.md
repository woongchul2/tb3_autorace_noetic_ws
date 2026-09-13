# Custom AutoRace description

ROBOTIS 원본을 수정하지 않고 사용하는 파라미터형 ROS Noetic Xacro 패키지입니다.

## 주요 파라미터 상태

`urdf/robot_parameters.xacro`의 기구 실측값은 입력 완료했습니다. 단위는 m, kg, rad, Hz입니다.

1. `wheel_radius`, `wheel_width`, `wheel_separation`은 실측값 입력 완료
2. 차체 collision 크기·중심과 Onshape 고정부 질량·COM·관성은 입력 완료
3. Mid-360 점군 원점의 `lidar_x/y/z`, `roll/pitch/yaw`는 입력 완료
4. D405 장착점·광학 중심과 `roll/pitch/yaw`는 입력 완료
5. LiDAR 범위·샘플 수·주기와 카메라 수평 FOV·해상도·FPS는 초기 시뮬레이션값 적용
6. 전체 수평 외곽 footprint와 보수적 반경 `0.149 m`는 입력 완료

좌표는 `base_link` 기준으로 +x 전방, +y 왼쪽, +z 위입니다. 각도는 rad이며 roll/pitch/yaw 순서입니다.

## 확인 명령

```bash
cd ~/tb3_autorace_noetic_ws
catkin_make
source devel/setup.bash
roslaunch custom_autorace_description description.launch use_gui:=true
```

Gazebo AutoRace 맵:

```bash
roslaunch custom_autorace_description gazebo_autorace.launch
```

별도 컬러 코스 PNG를 원본 모델 수정 없이 Gazebo 바닥과 카메라에 적용:

```bash
roslaunch custom_autorace_description gazebo_autorace.launch \
  course_texture:=/workspace/maps/changed_course.png
```

선택한 PNG는 SHA 기반 `/tmp` 모델에 스테이징되며 gzserver와 gzclient가 같은 모델을
사용합니다. 기본값 `course_texture:=from_model`은 원본 월드를 그대로 실행합니다.

가상 센서의 원본 토픽은 `/camera/color/image_raw`, `/scan_mid360_raw`, `/imu`,
`/odom`입니다. Gazebo에서는 기본적으로 바퀴 회전을 적분하는 `encoder` 오도메트리와
encoder 전진속도+IMU 방향/각속도를 결합한 EKF를 함께 실행합니다.

```text
/ground_truth/path       Gazebo 실제 경로(nav_msgs/Path)
/filtered/path           encoder+IMU EKF 경로(nav_msgs/Path)
/odometry/filtered       EKF 자세·속도와 계산된 6x6 공분산(nav_msgs/Odometry)
/trajectory/comparison   Ground Truth·EKF 경로와 2-sigma 위치 공분산(MarkerArray)
```

RViz의 Fixed Frame을 `odom`으로 설정하고 `MarkerArray` Display에서
`/trajectory/comparison`만 선택하면 다음 색으로 겹쳐 보입니다.

```text
빨간 점: `/gazebo/model_states`에서 직접 읽은 실제 시뮬레이션 위치 표식
초록색: encoder+IMU EKF
노란색 타원: 현재 EKF 위치 공분산의 2-sigma 범위
```

기본 실행에서는 Gazebo diff-drive의 raw odom TF를 끄고 EKF가 유일하게
`odom -> base_footprint` TF를 발행합니다. 따라서 RViz의 RobotModel과 LiDAR는
초록색 `/odometry/filtered` 자세를 따릅니다. 빨간색 Ground Truth는 시뮬레이션
비교용이며 TF를 제어하지 않습니다. Raw `/odom`은 EKF 입력으로만 유지하고 RViz에는
별도 궤적으로 표시하지 않습니다.

수치 공분산은 `/odometry/filtered`의 `pose.covariance`와 `twist.covariance`에서
확인할 수 있습니다. 시뮬레이션 초기 표준편차는 encoder 전진속도 `0.02 m/s`,
IMU 출력 `0.002`로 두었으며, 실물 로그를 수집한 뒤 반드시 다시 추정해야 합니다.
경로를 초기화하려면 다음 서비스를 호출합니다.

```bash
rosservice call /trajectory/reset
```

기존처럼 Gazebo 실제 위치 기반 `/odom`이 필요한 경우에만 다음 인자를 사용합니다.

```bash
roslaunch custom_autorace_description gazebo_autorace.launch odometry_source:=world
```

EKF를 제외하고 encoder odom만 시험하려면 다음 인자를 사용합니다.

```bash
roslaunch custom_autorace_description gazebo_autorace.launch fuse_imu:=false
```

이 모드에서는 TF 트리가 끊기지 않도록 Gazebo raw odom TF가 자동으로 다시
활성화되며, EKF TF와 동시에 발행되지는 않습니다.

## Xacro 밖에서 반드시 별도로 처리할 항목

- **2:1 증속 기어**: 저장소의 `firmware/custom_autorace_core`에 `cmd_vel -> 모터 RPM`과 `encoder tick -> wheel rotation` 보정을 모두 반영했습니다. 실물 OpenCR에는 이 커스텀 스케치를 업로드해야 합니다.
- 실물 카메라는 intrinsic calibration을 새로 하고, 설치 위치/각도 변경 후 AutoRace projection 및 compensation도 다시 보정합니다.
- Navigation용 실제 외곽 측정은 완료했습니다. `custom_autorace_bringup/config/navigation_footprint.yaml`의 비대칭 footprint를 공통 costmap 설정에 불러오고, 최대 속도/가속도는 실물 주행으로 조정합니다.

고정부 visual은 Onshape Assembly 원점 기준 Medium STL(약 9.4 MB, 197,026 triangles)을 `base_link`로 보정해 사용합니다. Collision은 성능과 안정성을 위해 실측 외곽의 단순 box를 별도로 유지합니다.
