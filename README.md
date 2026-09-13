# TurtleBot3 AutoRace Noetic

고정된 AutoRace 경기장에서 TurtleBot3가 규정을 지키며 모든 미션을 순서대로
완수하고, 반복 주행의 안정성을 유지하면서 결승선까지의 시간을 줄이는 ROS Noetic
workspace다. 프로젝트 고유 제어기와 설정은 `src/custom_autorace_bringup`, 로봇·센서
구성은 `src/custom_autorace_description`에 있다.

## 현재 검증 기준

2026-09-13의 공식 시작점 통합 **run23**에서 실제 카메라가 선택한 LEFT 교차로부터
Obstacle, Parking LEFT, Zigzag, Level Crossing, Tunnel을 순서대로 완료한 뒤 결승선
footprint 통과까지 확인했다.

- 오프라인 통합 판정: **22/22 PASS**, 실패 항목 0
- 출발 명령부터 결승선 통과까지: **280.472 s**
- 등록 시험: `custom_autorace_bringup` **504개**, 전체 workspace **557개**가 각각
  `0 errors`, `0 failures`, `0 skipped`로 통과

여기서 `504/557`은 성공 비율이 아니라 패키지 범위와 전체 workspace 범위에서 실행한
두 테스트 모음의 개수다. 상세한 조건, 미션별 시간과 여유 값은
[공통 경로 주행 구조 및 검증 기록](src/custom_autorace_bringup/PATH_FOLLOWING.md)을 따른다.

## 문서 지도

- [실행·빌드·시험 명령](DOCKER_NOETIC.md): 사용자가 실행할 명령과 각 명령의 역할
- [공통 경로 주행 구조](src/custom_autorace_bringup/PATH_FOLLOWING.md): 경로 형식,
  검증기·추종기 구조와 회귀 기록
- [Bringup 패키지](src/custom_autorace_bringup/README.md): launch, 설정과 노드 구성
- [로봇 설명 패키지](src/custom_autorace_description/README.md): URDF와 센서 구성
- [실물 하드웨어 파라미터](src/custom_autorace_description/HARDWARE_PARAMETERS.md):
  D405, Mid-360, OpenCR 기준값
- [Upstream 출처](UPSTREAM_SOURCES.md): monorepo에 포함한 외부 소스의 기준 revision
- [작업 원칙](AGENTS.md): 구현 및 최종 검증 기준
- [Notion 작성 규칙](NOTION_GUIDE.md): AutoRace 관련 Notion 문서 규칙

실행 명령은 여러 문서에 복제하지 않고 `DOCKER_NOETIC.md`를 단일 기준으로 유지한다.

## 진단 자료 방침

ROS bag을 포함한 원시 진단 자료는 크기가 크고 실행마다 생성되므로 Git 저장소에
포함하지 않는다. 로컬 `diagnostics/` 또는 별도 보관소에 원본을 유지하고, 재현에 필요한
조건과 최종 지표만 추적되는 문서에 남긴다. 따라서 저장소를 clone해도 run23의 원시
bag은 내려받아지지 않는다.
