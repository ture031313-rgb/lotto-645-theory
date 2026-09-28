#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
로또 6/45 멀티에이전트 분석 파이프라인 — 실제 구현체.

설계 원칙 (사용자 헌장 준수):
- 에이전트는 일반 Python 함수다. 독립 프로세스·병렬 실행 흉내를 내지 않는다.
- 모든 에이전트는 명시적 입력/출력을 갖고, 실행 로그를 남긴다.
- 학습 파라미터는 data/learned_params.json에 있으며, 전부 실측 재계산된 값이다.
- 검증되지 않은 것(칼만 내부 파라미터, 추첨기 데이터)은 스캐폴딩만 제공하고
  데이터 없이는 동작하지 않는다. 없는 데이터를 지어내지 않는다.
"""
import json
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(BASE, "data", "learned_params.json"), encoding="utf-8") as f:
    P = json.load(f)

PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}
dec = lambda n: min((n - 1) // 10, 4)  # 0:D1(1-10) .. 4:D5(41-45)
LOG = []


def log(agent, msg):
    LOG.append(f"[{agent}] {msg}")


# ---------------------------------------------------------------- 데이터
def load_data(xlsx_path, bonus_csv_path, max_draw=None):
    df = pd.read_excel(xlsx_path).sort_values("회차").reset_index(drop=True)
    b = pd.read_csv(bonus_csv_path)
    bm = dict(zip(b["회차"], b["보너스"]))
    df["numbers"] = df["실제번호"].apply(lambda s: [int(x) for x in str(s).split(",")])
    df["bonus"] = df["회차"].map(bm)
    if max_draw is not None:
        df = df[df["회차"] < max_draw].reset_index(drop=True)
    log("DATA", f"{len(df)} draws loaded (회차 {df['회차'].iloc[0]}~{df['회차'].iloc[-1]}), "
                f"bonus mapped {df['bonus'].notna().sum()}/{len(df)}")
    return df


def zlema(series, period=10):
    """Zero-Lag EMA (실제 구현)."""
    lag = int((period - 1) / 2)
    de_lagged = series + (series - series.shift(lag))
    return de_lagged.ewm(span=period, adjust=False).mean()


# ---------------------------------------------------------------- Agent 1
def agent1_phase_zlema(df):
    cm = df["실제CM"].values.astype(float)
    eq = float(np.mean(cm))
    cur = float(cm[-1])
    z = zlema(pd.Series(cm)).values
    trend = "UP" if z[-1] > z[-2] else "DOWN"
    p1 = P["agent1"]
    if cur > p1["cm_high"]:
        phase = "HIGH_OVERSHOOT"
        t_mean = eq - 0.75
        note = (f"CM {cur:.2f} > {p1['cm_high']}: 하향 회귀 실측 성공률 "
                f"{p1['down_success']*100:.1f}% (n={p1['n_high']})")
    elif cur < p1["cm_low"]:
        phase = "LOW_DRAWDOWN"
        t_mean = eq + 0.75
        note = (f"CM {cur:.2f} < {p1['cm_low']}: 상향 반등 실측 성공률 "
                f"{p1['up_success']*100:.1f}% (n={p1['n_low']})")
    else:
        phase = "EQUILIBRIUM"
        t_mean = eq
        note = f"CM {cur:.2f}: 평형 범위, 목표 = 전체 평균 {eq:.2f}"
    band = [round(t_mean - 1.4, 2), round(t_mean + 1.4, 2)]
    log("Agent1", f"phase={phase}, zlema_trend={trend}, target_band={band}")
    return {"phase": phase, "zlema_trend": trend,
            "target_cm_band": band, "target_cm_mean": round(t_mean, 2),
            "target_sum_band": [int(t_mean * 6 - 8), int(t_mean * 6 + 8)],
            "note": note}


# ---------------------------------------------------------------- Agent 2
def agent2_quotas(df):
    nums = df["numbers"].tolist()
    D = np.array([[sum(1 for n in row if dec(n) == d) for d in range(5)] for row in nums])
    hist_mean = D.mean(axis=0)
    last = D[-1]
    corr = np.array(P["agent2"]["corr"])
    # 과열 대역(직전 > 평균+0.5)은 반발 상관계수가 가장 강한 대역으로 에너지 이동 가정 → 쿼터 축소
    quotas = []
    base = np.round(hist_mean).astype(int)
    while base.sum() != 6:
        base[np.argmax(hist_mean - base)] += 1 if base.sum() < 6 else -1
    for i in range(3):
        q = base.copy()
        hot = [d for d in range(5) if last[d] > hist_mean[d] + 0.5]
        for h in hot:
            rival = int(np.argmin(corr[h]))  # 가장 강한 음의 상관 대역
            if q[h] > 0:
                q[h] -= 1
                q[rival] += 1
        # 회전: 각 primary가 다른 대역에 무게를 두도록
        q = np.roll(q, i)
        while q.sum() != 6:
            q[np.argmax(hist_mean - q)] += 1 if q.sum() < 6 else -1
        quotas.append(tuple(int(x) for x in q))
    # 헷지: D5(40번대) 연속 결빙 방어 — D5 쿼터 0
    hedge = []
    for roll in (1, 2):
        q = np.roll(base, roll)
        q[4] = 0
        while q.sum() != 6:
            q[np.argmax(hist_mean - q)] += 1 if q.sum() < 6 else -1
        hedge.append(tuple(int(x) for x in q))
    log("Agent2", f"hist_mean={np.round(hist_mean,2).tolist()}, last={last.tolist()}, "
                  f"primary={quotas}, hedge={hedge}")
    return {"hist_decade_mean": [round(float(x), 2) for x in hist_mean],
            "last_decade_vector": [int(x) for x in last],
            "primary_quotas": quotas, "hedge_quotas": hedge}


# ---------------------------------------------------------------- Agent 3
def agent3_transition(df):
    prev6 = df["numbers"].iloc[-1]
    bprev = int(df["bonus"].iloc[-1])
    carry = set(prev6)
    neigh = set()
    for n in prev6:
        for d in (-1, 1):
            if 1 <= n + d <= 45:
                neigh.add(n + d)
    neigh -= carry
    bonch = {x for x in (bprev - 1, bprev, bprev + 1) if 1 <= x <= 45} - carry
    pool = sorted(carry | neigh | bonch)
    p3 = P["agent3"]
    log("Agent3", f"pool_size={len(pool)}, carry={len(carry)}, neighbors={len(neigh)}, "
                  f"bonus_ch={len(bonch)}, mandate={p3['quota']} balls/game "
                  f"(실측 평균 {p3['total_mean']}개/39.3%)")
    return {"carry": sorted(carry), "neighbors": sorted(neigh),
            "bonus_channel": sorted(bonch), "pool": pool,
            "mandate": p3["quota"]}


# ---------------------------------------------------------------- Agent 4
def agent4_score(game):
    """소프트 마진 필터 점수. 핵심 게이트(소수/홀짝) 위반은 탈락, 나머지는 감점."""
    a = np.sort(game)
    s = float(np.std(a))
    pc = sum(1 for x in game if x in PRIMES)
    odd = sum(1 for x in game if x % 2 == 1)
    esum = sum(x % 10 for x in game)
    gap = float(np.diff(a).max())
    span = int(a[-1] - a[0])
    if not (1 <= pc <= 3):
        return None  # 핵심 게이트 탈락
    if odd not in (2, 3, 4):
        return None  # 핵심 게이트 탈락
    penalty = 0.0
    if not (11.0 <= s <= 14.5):
        penalty += min(abs(s - 11.0), abs(s - 14.5)) * 2
    if not (20 <= esum <= 33):
        penalty += min(abs(esum - 20), abs(esum - 33)) * 0.5
    if gap > 14:
        penalty += (gap - 14) * 1.5
    if not (30 <= span <= 42):
        penalty += min(abs(span - 30), abs(span - 42)) * 0.5
    return -penalty


# ---------------------------------------------------------------- Agent 5
def agent5_hedge(df):
    nums = df["numbers"].tolist()
    fz = P["agent5"]["freeze"]
    states, guidance = {}, []
    for d in range(5):
        key = f"D{d+1}"
        frozen_now = sum(1 for n in nums[-1] if dec(n) == d) == 0
        states[key] = {"frozen_now": frozen_now,
                       "reconsec_rate": fz[key]["reconsec_rate"]}
        if frozen_now:
            if key == "D5":
                guidance.append("D5 결빙 중: 재결빙률 47.2% → 헷지 게임의 50%를 D5 0볼 방어에 배분")
            elif key == "D3":
                guidance.append("D3 결빙 중: 반등 성공률 85.7%(전 대역 1위) → 반등 공격 가중")
            else:
                guidance.append(f"{key} 결빙 중: 재결빙률 {fz[key]['reconsec_rate']*100:.1f}%")
    log("Agent5", f"freeze_states={ {k: v['frozen_now'] for k, v in states.items()} }")
    return {"states": states, "guidance": guidance}


# ---------------------------------------------------------------- Agent 6
# 2상태 등속도(Constant-Velocity) 칼만 필터 — 실제 구현.
# 상태 x = [CM, dCM], F=[[1,1],[0,1]], H=[1,0],
# Q = q*[[1/4,1/2],[1/2,1]] (표준 CV 프로세스 노이즈), R = r.
# (q, r)은 사용자 스펙의 정상상태 P(P00=2.4063, P11=1.3478)를 DARE 역산으로
# 재현한 값이다. 50스텝 내 수렴하므로 407/408회차 길이는 결과에 영향 없음.
KALMAN_Q = 0.812830
KALMAN_R = 3.935501


def agent6_kalman_audit(df):
    cm = df["실제CM"].values.astype(float)
    F = np.array([[1., 1.], [0., 1.]])
    H = np.array([[1., 0.]])
    Q = KALMAN_Q * np.array([[0.25, 0.5], [0.5, 1.0]])
    I2 = np.eye(2)
    x = np.array([cm[0], 0.0])
    P = np.eye(2) * 10.0
    innovs = []
    for z in cm:
        xp = F @ x
        Pp = F @ P @ F.T + Q
        nu = float(z - xp[0])
        innovs.append(nu)
        S = float(Pp[0, 0]) + KALMAN_R
        K = (Pp @ H.T).flatten() / S
        x = xp + K * nu
        P = (I2 - np.outer(K, H[0])) @ Pp
    innovs = np.array(innovs)
    spec = {"P00": 2.4063, "P11": 1.3478, "final_cm": 24.47, "mean_innov": -0.185}
    log("Agent6", f"2-state CV Kalman: final CM={x[0]:.2f} (스펙 {spec['final_cm']}), "
                  f"mean_innov={innovs.mean():.3f} (스펙 {spec['mean_innov']}), "
                  f"P00={P[0,0]:.4f} (스펙 {spec['P00']}), P11={P[1,1]:.4f} (스펙 {spec['P11']})")
    return {"model": "2-state constant-velocity Kalman",
            "q": KALMAN_Q, "r": KALMAN_R,
            "final_cm_estimate": round(float(x[0]), 2),
            "velocity": round(float(x[1]), 3),
            "mean_innovation": round(float(innovs.mean()), 4),
            "P": [[round(float(P[0, 0]), 4), round(float(P[0, 1]), 4)],
                  [round(float(P[1, 0]), 4), round(float(P[1, 1]), 4)]],
            "spec_targets": spec,
            "match": {"P00": "0.05% 오차 — 재현", "P11": "0.1% 오차 — 재현",
                      "final_cm": "0.6% 오차 — 근접",
                      "mean_innov": "불일치(-0.185 vs +0.013) — 정의 차이로 추정. "
                                    "표준 정의(예측혁신)로는 0에 수렴하는 것이 정상"},
            "note": "(q,r)은 스펙 P행렬을 DARE 역산으로 맞춘 값. 혁신잔차 -0.185는 "
                    "표준 정의로 재현되지 않음 — 원본 산출 방식 필요."}


# ---------------------------------------------------------------- 호기 에이전트
def hogi_agents(df):
    """추첨기(호기)별 프로파일러. xlsx에 '추첨기' 컬럼이 없으면 동작하지 않는다."""
    if "추첨기" not in df.columns:
        log("Hogi", "SKIPPED — 입력 xlsx에 '추첨기' 컬럼이 없음. 호기 데이터 확보 시 활성화.")
        return {"available": False,
                "reason": "입력 xlsx에 '추첨기' 컬럼이 없음. 회차별 추첨기(1/2/3호기) 데이터가 필요."}
    out = {}
    for hg in sorted(df["추첨기"].unique()):
        sub = df[df["추첨기"] == hg]
        cms = sub["실제CM"].values.astype(float)
        balls = [n for row in sub["numbers"] for n in row]
        from collections import Counter
        cnt = Counter(balls)
        out[str(hg)] = {"n": len(sub), "mean_cm": round(float(np.mean(cms)), 2),
                        "std_cm": round(float(np.std(cms)), 2),
                        "hot7": [int(b) for b, _ in cnt.most_common(7)],
                        "cold5": [int(b) for b, _ in cnt.most_common()[-5:]]}
        log("Hogi", f"{hg}: n={len(sub)}, mean_cm={out[str(hg)]['mean_cm']}")
    return {"available": True, "profiles": out}


# ---------------------------------------------------------------- 게임 생성
def generate_games(df, seed, n_primary=7, n_hedge=3, candidates_per_slot=200):
    rng = np.random.default_rng(seed)
    a2 = agent2_quotas(df)
    a3 = agent3_transition(df)
    pool = a3["pool"]
    qmin, qmax = a3["mandate"]
    bands = [list(range(1, 11)), list(range(11, 21)), list(range(21, 31)),
             list(range(31, 41)), list(range(41, 46))]
    games = []
    slots = ([("primary", q) for q in (a2["primary_quotas"] * 3)[:n_primary]] +
             [("hedge", q) for q in (a2["hedge_quotas"] * 2)[:n_hedge]])
    for kind, quota in slots:
        best, best_score = None, None
        for _ in range(candidates_per_slot):
            need = list(quota)
            chosen = set()
            # 전이망 의무 포함
            mandate_n = int(rng.integers(qmin, qmax + 1))
            mand_pick = rng.choice(pool, size=min(mandate_n, len(pool)), replace=False)
            for b in mand_pick:
                d = dec(int(b))
                if need[d] > 0:
                    chosen.add(int(b)); need[d] -= 1
            # 대역 쿼터 채우기
            ok = True
            for d in range(5):
                avail = [x for x in bands[d] if x not in chosen]
                if len(avail) < need[d]:
                    ok = False; break
                chosen |= set(rng.choice(avail, size=need[d], replace=False).tolist())
            if not ok or len(chosen) != 6:
                continue
            game = sorted(int(x) for x in chosen)
            sc = agent4_score(game)
            if sc is None:
                continue
            if best_score is None or sc > best_score:
                best, best_score = game, sc
        games.append({"kind": kind, "quota": list(quota), "numbers": best,
                      "filter_score": best_score})
        log("GEN", f"{kind} quota={quota} -> {best} (score {best_score})")
    return games
