# Upstream 소스 기준

이 저장소는 재현성과 단순한 배포를 위해 외부 ROS 저장소를 submodule이 아닌 일반
디렉터리로 포함한 **monorepo snapshot**이다. 아래 SHA는 2026-09-13에 snapshot을 만든
시점의 upstream 기준 revision이다.

| 포함 경로 | Upstream | Branch | 기준 SHA | Snapshot 상태 |
|---|---|---|---|---|
| `vendor/OpenCR` | [ROBOTIS-GIT/OpenCR](https://github.com/ROBOTIS-GIT/OpenCR.git) | `master` | `68ec75d8a400949580ecf263e0105ea9743b878e` | upstream working tree |
| `src/turtlebot3` | [ROBOTIS-GIT/turtlebot3](https://github.com/ROBOTIS-GIT/turtlebot3.git) | `noetic` | `4ae959ea6a52415c90bf752d2b76e3e28f8a87e2` | upstream working tree |
| `src/turtlebot3_msgs` | [ROBOTIS-GIT/turtlebot3_msgs](https://github.com/ROBOTIS-GIT/turtlebot3_msgs.git) | `noetic` | `76e78b0a34e07cf1dd16dafdc54c44f35c5b83eb` | upstream working tree |
| `src/realsense-ros` | [realsenseai/realsense-ros](https://github.com/realsenseai/realsense-ros.git) | `ros1-legacy` | `debadfaaa4a21c667c076ad2abacf69f9427c825` | upstream working tree |
| `src/livox_ros_driver2` | [Livox-SDK/livox_ros_driver2](https://github.com/Livox-SDK/livox_ros_driver2.git) | `master` | `4a1def929e5b59c7a8122d19fce6efba581ce9f7` | upstream working tree |
| `src/turtlebot3_simulations` | [ROBOTIS-GIT/turtlebot3_simulations](https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git) | `noetic` | `e9d809ca8e3bf889c0275e4103b15a341ffab888` | 기준 SHA 위에 경기장·차단봉 수정 포함 |
| `src/turtlebot3_autorace_2020` | [ROBOTIS-GIT/turtlebot3_autorace_2020](https://github.com/ROBOTIS-GIT/turtlebot3_autorace_2020.git) | `noetic` | `367a24a7228fe9dcd339af50f70b8672df3cc0cc` | 기준 SHA 위에 카메라·검출·미션 연동 수정 포함 |

마지막 두 행의 SHA는 해당 디렉터리가 upstream과 완전히 동일하다는 뜻이 아니라,
프로젝트 변경을 적용하기 전의 **base commit**을 뜻한다. 프로젝트 변경까지 포함한 현재
파일 상태는 루트 저장소의 commit이 기준이다. 이 구조에는 `.gitmodules`가 없으며
`git submodule update`를 실행하지 않는다.

## 원래 Git 메타데이터 백업

snapshot 전의 일곱 nested `.git` 디렉터리와 비어 있던 이전 루트 `.git`은 다음 로컬
경로에 이동해 보관했다.

`/home/sj/.codex/backups/tb3_autorace_noetic_ws_nested_git_20260913`

이 백업은 monorepo나 원격 GitHub 저장소에 포함되지 않는다. upstream 이력 확인 또는
기존 중첩 저장소 복구가 필요할 때만 사용하며, 일상적인 개발·빌드의 기준은 루트
저장소다.
