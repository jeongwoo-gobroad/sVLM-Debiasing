# (c) 2026 Jeongwoo Kim from KNU CSE, All rights reserved
# sVLM Inference-time Debiasing Research
"""Atomic progress files that tolerate short Windows sharing locks."""
import json
import os
from pathlib import Path
import time


def write_status(path, state, *, required=True, attempts=20):
    path=Path(path)
    temp=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temp.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    for attempt in range(attempts):
        try:
            temp.replace(path)
            return True
        except PermissionError:
            # 진행 창이나 동기화가 잠깐 파일을 읽어도 추론 결과는 버리지 않는다.
            if attempt+1==attempts:
                if required:raise
                return False
            time.sleep(min(.05*(attempt+1),.25))
