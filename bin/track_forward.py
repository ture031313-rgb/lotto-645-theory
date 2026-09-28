#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
순방향 추적 프로토콜 — 추첨 전 예측 고정 → 추첨 후 채점 → 원장 누적.

사용법:
  python3 track_forward.py lock --draw 1244
      파이프라인으로 예측 생성 → SHA256 해시와 함께 원장에 'locked' 기록
  python3 track_forward.py score --draw 1244 --numbers 1,2,3,4,5,6 --bonus 7
      고정된 예측과 실제 추첨 결과 대조 → 게임별 적중 수 → 원장에 'scored' 기록
  python3 track_forward.py ledger
      원장 요약 출력

원칙:
- lock은 추첨 전에만 가능. 추첨 후 lock 시도는 거부.
- 예측 파일 자체는 수정 불가. 채점은 별도 원장에만 기록.
- 적중 판정: 6개=1등, 5+보너스=2등, 5개=3등, 4개=4등, 3개=5등.
"""
import argparse, json, os, hashlib, subprocess, sys
from datetime import datetime, timezone, timedelta

BASE = os.path.expanduser("~/workspace/lotto/forward")
LEDGER = os.path.join(BASE, "ledger.json")
KST = timezone(timedelta(hours=9))
# 회차별 추첨일 (토요일). 필요시 추가.
DRAW_DATES = {1244: "2026-10-03", 1245: "2026-10-10", 1246: "2026-10-17"}

def now_kst():
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

def load_ledger():
    if os.path.exists(LEDGER):
        return json.load(open(LEDGER))
    return {"entries": []}

def save_ledger(l):
    os.makedirs(BASE, exist_ok=True)
    json.dump(l, open(LEDGER, "w"), ensure_ascii=False, indent=2)

def cmd_lock(draw):
    l = load_ledger()
    if any(e["draw"] == draw and e["status"] == "locked" for e in l["entries"]):
        print(f"{draw}회는 이미 고정됨 — 중복 lock 거부"); return 1
    dd = DRAW_DATES.get(draw)
    if dd and datetime.now(KST).strftime("%Y-%m-%d") >= dd:
        print(f"{draw}회 추첨일({dd}) 이후 — lock 거부"); return 1
    r = subprocess.run([sys.executable, os.path.expanduser(
        "~/workspace/skills/lotto-645-agents/bin/run_pipeline.py"),
        "predict", "--draw", str(draw)], capture_output=True, text=True)
    if r.returncode != 0:
        print("파이프라인 실패:\n", r.stderr[-500:]); return 1
    line = [x for x in r.stdout.splitlines() if x.startswith("예측 고정 완료")]
    if not line:
        print("예측 파일 경로를 찾지 못함"); return 1
    path = line[0].split("→")[1].strip()
    h = sha256(path)
    l["entries"].append({
        "draw": draw, "status": "locked",
        "locked_at": now_kst(), "file": path, "sha256": h,
        "draw_date": dd,
    })
    save_ledger(l)
    print(f"{draw}회 고정 완료\n  파일: {path}\n  SHA256: {h}\n  시각: {now_kst()}")
    return 0

def cmd_score(draw, numbers, bonus):
    l = load_ledger()
    ent = next((e for e in l["entries"] if e["draw"] == draw and e["status"] == "locked"), None)
    if not ent:
        print(f"{draw}회 고정 기록 없음 — score 불가"); return 1
    actual = set(numbers); b = bonus
    pred = json.load(open(ent["file"]))
    games = pred["games"]
    res = []
    for g in games:
        gn = set(g["numbers"]); m = len(gn & actual)
        rank = None
        if m == 6: rank = 1
        elif m == 5 and b in gn: rank = 2
        elif m == 5: rank = 3
        elif m == 4: rank = 4
        elif m == 3: rank = 5
        res.append({"numbers": sorted(gn), "match": m, "rank": rank})
    hits = [r for r in res if r["rank"]]
    ent["status"] = "scored"
    ent["scored_at"] = now_kst()
    ent["actual"] = {"numbers": sorted(actual), "bonus": b}
    ent["results"] = res
    ent["summary"] = f"{len(hits)}/10 게임 적중, 최고 {min([r['rank'] for r in hits])}등" if hits else "10게임 전부 낙첨"
    save_ledger(l)
    print(f"{draw}회 채점 완료: {ent['summary']}")
    for i, r in enumerate(res):
        mark = f" ★{r['rank']}등" if r["rank"] else ""
        print(f"  게임{i+1}: {r['numbers']} → {r['match']}개 적중{mark}")
    return 0

def cmd_ledger():
    l = load_ledger()
    print(f"=== 순방향 추적 원장 ({len(l['entries'])}건) ===")
    for e in l["entries"]:
        s = f"{e['draw']}회 [{e['status']}] 고정 {e['locked_at']}"
        if e["status"] == "scored":
            s += f" → {e['summary']}"
        print(s)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("lock"); a.add_argument("--draw", type=int, required=True)
    b = sub.add_parser("score"); b.add_argument("--draw", type=int, required=True)
    b.add_argument("--numbers", required=True); b.add_argument("--bonus", type=int, required=True)
    sub.add_parser("ledger")
    args = ap.parse_args()
    if args.cmd == "lock": sys.exit(cmd_lock(args.draw))
    elif args.cmd == "score":
        sys.exit(cmd_score(args.draw, list(map(int, args.numbers.split(","))), args.bonus))
    else: cmd_ledger()
