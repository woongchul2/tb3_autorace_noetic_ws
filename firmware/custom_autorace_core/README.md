# Custom AutoRace OpenCR firmware

ROBOTIS OpenCR의 `turtlebot3_burger/turtlebot3_core` 예제를 복사해 만든 ROS 1 Noetic용 커스텀 스케치입니다. 원본 라이브러리와 `vendor/OpenCR` 참조 복사본은 수정하지 않습니다.

기준 OpenCR 저장소 commit: `68ec75d8a400949580ecf263e0105ea9743b878e`

## 확정된 기구 값

| 항목 | 값 |
|---|---:|
| 바퀴 지름 | `0.0955 m` |
| 바퀴 반지름 | `0.04775 m` |
| 타이어 고무 폭 | `0.0113 m` |
| 고무 바깥쪽-바깥쪽 | `0.1595 m` |
| 바퀴 중심 간격 | `0.1482 m` |
| 외부 기어 | `2:1 증속` |
| 모터 회전 / 바퀴 회전 크기 | `0.5` |
| 직접 맞물린 외접 평기어 방향 | 반대 방향, 부호 `-1` |

중심 간격은 `159.5 mm - 11.3 mm = 148.2 mm`로 계산했습니다.

## 펌웨어 보정 방식

- 명령: `cmd_vel`에서 계산한 바퀴 각속도에 속도비 `0.5`와 외접기어 방향 `-1`을 적용해 XL430 목표 속도로 변환합니다. 기존 모터 드라이버 API에는 부호를 포함한 등가 반지름을 전달합니다.
- 엔코더: 모터축 tick 각도에 속도배율 `2`와 외접기어 방향 `-1`을 적용해 실제 바퀴 회전각과 `/odom` 이동거리를 계산합니다.
- 바퀴 joint state와 OpenCR 버튼 주행시험 거리도 같은 계수로 보정했습니다.
- 이론상 61 motor rpm 기준 최대 직선속도는 약 `0.610 m/s`입니다. 실제 AutoRace 주행 제한은 ROS 주행 파라미터에서 더 낮게 설정해야 합니다.

수정할 핵심 상수는 `turtlebot3_burger.h`의 아래 네 개입니다.

```cpp
#define WHEEL_RADIUS          0.04775f
#define WHEEL_SEPARATION      0.1482f
#define TURNING_RADIUS        (WHEEL_SEPARATION / 2.0f)
#define GEAR_SPEEDUP          2.0f
#define GEAR_DIRECTION        -1.0f
```

톱니 수 자체는 더 필요하지 않습니다. 현재 2:1 증속의 회전수 비 크기는 `0.5`이고, 직접 맞물린 외접 평기어 한 쌍은 회전 방향을 뒤집으므로 `GEAR_DIRECTION = -1.0f`입니다. 아이들러나 추가 기어단 때문에 최종 바퀴가 모터와 같은 방향으로 도는 구조라면 이 값만 `+1.0f`로 바꿉니다.

## 빌드와 업로드

ROS Noetic과 Gazebo는 Ubuntu 20.04 Docker에서 실행하지만, OpenCR 펌웨어 업로드는 ROS 배포판과 무관하므로 Ubuntu 24.04 호스트의 Arduino IDE에서 해도 됩니다.

1. ROBOTIS e-Manual의 OpenCR Arduino IDE 설치 절차에 따라 OpenCR 보드 패키지를 설치합니다.
2. Arduino IDE에서 이 디렉터리의 `custom_autorace_core.ino`를 엽니다.
3. Board를 `OpenCR Board`, Port를 연결된 `/dev/ttyACM*`로 선택합니다.
4. Verify 후 Upload합니다.
5. 업로드 뒤 OpenCR을 다시 연결하고 ROS Noetic에서 `roslaunch custom_autorace_bringup hardware.launch`를 실행합니다.

포트 권한 오류가 나면 사용자를 `dialout` 그룹에 추가한 뒤 다시 로그인합니다.

```bash
sudo usermod -aG dialout "$USER"
```

업로드 전에 로봇을 받침대 위에 올려 바퀴가 공중에 뜨도록 하고, 비상 정지를 위해 배터리 전원을 바로 끌 수 있게 준비합니다.

## 실물 검증 순서

1. 작은 양수 `/cmd_vel` 명령에서 두 바퀴가 전진 방향인지 확인합니다. 반대로 돌면 기어비는 그대로 두고 `GEAR_DIRECTION`의 부호만 바꿉니다.
2. `0.10 m` 직선 명령 후 실제 이동거리와 `/odom`을 비교합니다.
3. 실제 거리가 다르면 기어비를 다시 추정하기 전에 하중 상태의 유효 바퀴 반지름을 먼저 보정합니다.
4. 제자리 회전 오차는 `WHEEL_SEPARATION`, 직선 거리 오차는 `WHEEL_RADIUS`를 중심으로 보정합니다.

센서 위치와 볼캐스터 위치는 `custom_autorace_description/urdf/robot_parameters.xacro`에 반영했습니다. 전체 수평 외곽으로 계산한 보수적 반경 `0.149 m`는 이 펌웨어의 `ROBOT_RADIUS`와 description 파라미터에 함께 반영했습니다.
