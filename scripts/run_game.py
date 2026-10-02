#!/usr/bin/env python3
"""정면, 왼쪽 사선, 왼쪽 측면 스쿼트 영상을 시점별로 녹화한다.

각 시점에서 촬영을 시작한 뒤 스쿼트 10회를 마칠 때마다 완료 버튼을 누른다.
세 시점 녹화가 끝나면 영상 저장 버튼으로 한 폴더에 내보낸다.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_trainer.squat_recorder import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
