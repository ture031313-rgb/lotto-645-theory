#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
호기(추첨기) 모듈 — 최소 정직 버전.

2026-09-28 검증 결론:
- 호기별 번호 특성(프로파일)은 통계적으로 유의미하지 않음 (p>0.05, 별도 영수증)
- 번호만으로 호기 역추적 불가 (월별/3개월 주기 구조 없음, 별도 영수증)
→ 호기별 번호 프로파일링(M1/M2/M3형)은 삭제.
→ 남는 것: 호기 교체 타이밍 추적(일정 팩트, 번호 예측 아님).
   배정 데이터가 없으면 모듈 전체가 대기 상태.
"""
import os
import pandas as pd

HOGI_CSV = os.path.expanduser("~/workspace/lotto/hogi_836_1243.csv")
LOG = []


def log(msg):
    LOG.append(f"[Hogi] {msg}")


def load_hogi_map(csv_path=HOGI_CSV):
    """회차->추첨기 매핑 로드. 파일 없으면 None."""
    if not os.path.exists(csv_path):
        log(f"STANDBY — 호기 배정 파일 없음: {csv_path}")
        return None
    h = pd.read_csv(csv_path)
    assert {"회차", "추첨기"} <= set(h.columns), f"컬럼 오류: {list(h.columns)}"
    assert set(h["추첨기"].unique()) <= {1, 2, 3}, "추첨기는 1/2/3만 허용"
    m = dict(zip(h["회차"].astype(int), h["추첨기"].astype(int)))
    log(f"{len(m)}개 회차 호기 배정 로드")
    return m


def handover_watch(df, hogi_map, target_draw):
    """호기 교체 타이밍만 추적. 번호 특성은 다루지 않음.
    - target 회차의 호기가 직전 회차와 다르면 교체 주간으로 표기.
    - 월별 로테이션 가설(1·4·7·10월=1호기)은 미검증이므로 사용하지 않음."""
    if target_draw not in hogi_map:
        log(f"{target_draw}회 호기 미확정 — 교체 타이밍 추적 불가")
        return {"known": False}
    cur = hogi_map[target_draw]
    prev_draws = [d for d in hogi_map if d < target_draw]
    prev = hogi_map[max(prev_draws)] if prev_draws else None
    handover = prev is not None and prev != cur
    log(f"{target_draw}회: {cur}호기" + (" (교체 주간)" if handover else " (유지)"))
    return {"known": True, "machine": cur,
            "handover_week": handover,
            "note": "교체 주간 표기는 일정 정보일 뿐, 번호 예측에 사용하지 않음"}


def run_hogi_agents(df, target_draw, draw_date=None):
    hogi_map = load_hogi_map()
    if hogi_map is None:
        return {"available": False,
                "reason": f"호기 배정 파일 없음: {HOGI_CSV}",
                "log": LOG}
    hw = handover_watch(df, hogi_map, target_draw)
    return {"available": True, "handover": hw, "log": LOG,
            "note": "호기별 번호 프로파일링은 검증 결과 유의미하지 않아 삭제됨 (2026-09-28)"}
