#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
사용법:
  python3 run_pipeline.py predict --draw 1244
      → 1244회 추첨 전에 실행. 1243회까지 데이터로 6개 에이전트 + 호기 모듈을 돌리고
        10개 게임(7 primary + 3 hedge)을 생성해 타임스탬프 예측 파일로 고정한다.
        (사전 등록: 추첨 전 변경 불가능한 기록)
  python3 run_pipeline.py backtest --draw 1243
      → 과거 회차 검증용. 해당 회차 이전 데이터만으로 파이프라인을 돌리고
        실제 당첨번호와 대조한다. look-ahead 없음.

데이터 경로:
  XLSX: ~/workspace/user/files/645_408.xlsx (837~1243회)
  BONUS: ~/workspace/lotto/bonus_836_1243.csv
"""
import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agents import (load_data, agent1_phase_zlema, agent2_quotas, agent3_transition,
                    agent4_score, agent5_hedge, agent6_kalman_audit,
                    generate_games, LOG)
from hogi import run_hogi_agents, LOG as HOGI_LOG

XLSX = os.path.expanduser("~/workspace/user/files/645_408.xlsx")
BONUS = os.path.expanduser("~/workspace/lotto/bonus_836_1243.csv")
RECEIPTS = os.path.expanduser("~/workspace/receipts")


def run(target_draw, write_receipt):
    df = load_data(XLSX, BONUS, max_draw=target_draw)
    a1 = agent1_phase_zlema(df)
    a2 = agent2_quotas(df)
    a3 = agent3_transition(df)
    a5 = agent5_hedge(df)
    a6 = agent6_kalman_audit(df)
    hg = run_hogi_agents(df, target_draw)
    LOG.extend(HOGI_LOG)
    games = generate_games(df, seed=target_draw)
    result = {
        "target_draw": target_draw,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_through": int(df["회차"].iloc[-1]),
        "agent1": a1, "agent2": a2,
        "agent3": {k: v for k, v in a3.items()},
        "agent5": a5, "agent6": a6, "hogi": hg,
        "games": games,
        "execution_log": LOG,
    }
    if write_receipt:
        os.makedirs(RECEIPTS, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(RECEIPTS, f"prediction-{target_draw}-{stamp}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        result["receipt_path"] = path
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["predict", "backtest"])
    ap.add_argument("--draw", type=int, required=True)
    args = ap.parse_args()

    if args.mode == "predict":
        r = run(args.draw, write_receipt=True)
        print(f"예측 고정 완료 → {r['receipt_path']}")
    else:
        r = run(args.draw, write_receipt=False)
        df = load_data(XLSX, BONUS)
        actual = df[df["회차"] == args.draw]
        if len(actual):
            a = sorted(actual["numbers"].iloc[0])
            print(f"실제 {args.draw}회 당첨번호: {a}")
            for i, g in enumerate(r["games"], 1):
                hit = len(set(g["numbers"]) & set(a)) if g["numbers"] else 0
                print(f"  게임 {i:2d} [{g['kind']:7s}] {g['numbers']} → 적중 {hit}개")
        else:
            print(f"{args.draw}회 실측 데이터 없음 (미래 회차는 predict 사용)")
    # 에이전트 요약
    print(f"\n[Agent1] {r['agent1']['phase']} | 목표 CM {r['agent1']['target_cm_band']}")
    print(f"[Agent2] primary quotas: {r['agent2']['primary_quotas']}")
    print(f"[Agent3] 전이망 풀 {len(r['agent3']['pool'])}개, 의무 {r['agent3']['mandate']}개/게임")
    print(f"[Agent5] " + (" | ".join(r['agent5']['guidance']) if r['agent5']['guidance'] else "결빙 없음"))
    print(f"[Agent6] CM 추정 {r['agent6']['final_cm_estimate']}, 혁신 평균 {r['agent6']['mean_innovation']}")
    print(f"[Hogi] {'활성' if r['hogi']['available'] else '비활성: ' + r['hogi']['reason']}")


if __name__ == "__main__":
    main()
