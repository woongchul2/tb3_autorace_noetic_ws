# 진단 자료 보존 기준

`diagnostics/`는 Git에 포함하지 않는 로컬 실행 자료다. 공간 정리 시 아래 네 기록은
bag과 결과 파일을 포함한 디렉터리 전체를 보존한다.

- `official_full_camera_rolling_common_swept_adaptive_20260913_run21`
- `official_full_camera_rolling_common_swept_adaptive_20260913_run23`
- `parking_zigzag_transition_20260910_continuous_left_official_run1`
- `parking_zigzag_transition_20260910_continuous_right_official_run1`

그 밖의 기록에서는 `*.json`과 `*.csv` 요약만 보존한다. bag, 미완성 bag, 로그,
프레임 이미지는 삭제할 수 있다. 삭제된 원시 자료는 GitHub에서 복구할 수 없으므로,
추가 기록을 장기 보존해야 할 때는 이 목록에 먼저 등록한다.

2026-09-14 정리에서는 위 기준으로 48.63 GiB의 중간 원시 자료를 제거하고,
최종 통합 주행과 Parking 좌·우 전환 기록 약 192 MiB 및 모든 JSON 요약을 남겼다.
