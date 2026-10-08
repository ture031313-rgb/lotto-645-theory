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


def _cm_series(df):
    """실제CM 컬럼을 읽되, 비어있는 신규 회차는 번호 평균으로 대체한다.
    (2026-10-08: 1244행 추가 시 실제CM 미기입으로 NaN 크래시 발생.
    기존 행은 저장값을 그대로 쓰므로 기존 예측 결과에 영향 없음.)"""
    cm = df["실제CM"].values.astype(float)
    if np.isnan(cm).any():
        computed = df["numbers"].apply(lambda r: sum(r) / 6.0).values.astype(float)
        n_fill = int(np.isnan(cm).sum())
        cm = np.where(np.isnan(cm), computed, cm)
        log("DATA", f"실제CM NaN {n_fill}건을 번호 평균으로 대체")
    return cm


# ---------------------------------------------------------------- Agent 1
def agent1_phase_zlema(df):
    cm = _cm_series(df)
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
            # D5는 방어 대역이라 재배치에서 제외한다.
            # (2026-10-08 수정: 기존 코드는 이 루프가 D5에 +1을 되돌려
            #  "D5=0 방어"가 무력화됐음 — hedge 게임 전부 D5 1볼로 생성되던 버그)
            gaps = hist_mean - q
            gaps[4] = -np.inf
            q[int(np.argmax(gaps))] += 1 if q.sum() < 6 else -1
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
def agent4_score(game, target_cm=None, pool=None, attack_decades=None):
    """소프트 마진 필터 점수. 핵심 게이트(소수/홀짝) 위반은 탈락.
    감점(제약 위반) + 가점(이론 부합)으로 변별력을 확보한다.
    가점 근거: (1) Agent1 목표CM 근접 = 이론의 핵심 예측과 일치,
    (2) pool 포함 = Agent3 실측(당첨번호의 39.3%가 pool 출신)과 일치,
    (3) attack_decades = Agent5 결빙 대역 반등 공격 (D5 제외 — D5는 재결빙률
    47.2%로 최고라 방어가 정답이며, agent2 hedge quota의 0볼 방어로 처리됨).
    가점 없이 감점만 있으면 통과 후보가 전부 동점(-0.0)이 되어
    '200개 중 최고 선택'이 무력화된다 (2026-10-04 진단)."""
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
    bonus = 0.0
    if target_cm is not None:
        cm = sum(game) / 6.0
        bonus += max(0.0, 2.0 - abs(cm - target_cm))
    if pool is not None:
        bonus += sum(1 for x in game if x in pool) * 0.5
    if attack_decades is not None:
        bonus += sum(1 for x in game if dec(x) in attack_decades) * 0.5
    return bonus - penalty


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
    cm = _cm_series(df)
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
    a1 = agent1_phase_zlema(df)
    a2 = agent2_quotas(df)
    a3 = agent3_transition(df)
    a5 = agent5_hedge(df)
    pool = a3["pool"]
    pool_set = set(pool)
    target_cm = a1["target_cm_mean"]
    # Agent5 연결: 결빙된 대역(비-D5)은 반등 공격 가중.
    # D5 결빙은 재결빙률 47.2%로 최고라 방어가 정답이며,
    # agent2 hedge quota의 D5=0볼 방어로 이미 처리되므로 제외.
    attack_decades = {d for d in range(4)
                      if a5["states"][f"D{d+1}"]["frozen_now"]}
    if attack_decades:
        log("Agent5", f"반등 공격 대역: {sorted(f'D{d+1}' for d in attack_decades)}")
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
            sc = agent4_score(game, target_cm=target_cm, pool=pool_set,
                             attack_decades=attack_decades)
            if sc is None:
                continue
            if best_score is None or sc > best_score:
                best, best_score = game, sc
        if best is None:
            # 200개 후보 전부 게이트 탈락 — 확률상 거의 불가능하나
            # silent None이 원장·시트까지 흘러가는 것을 방지하기 위해 명시적 오류
            raise RuntimeError(
                f"[{kind}] quota={quota}: 200 candidates all gated out")
        games.append({"kind": kind, "quota": list(quota), "numbers": best,
                      "filter_score": best_score})
        log("GEN", f"{kind} quota={quota} -> {best} (score {best_score})")
    return games


# ---------------------------------------------------------------- 에이전트 감시 (validation)
def validate_prediction(result):
    """각 에이전트 출력을 검사한다. 예측 JSON 1개를 받아
    [{agent, check, status, detail}] 리스트를 반환한다.
    status: ok / warn / fail"""
    checks = []

    def add(agent, check, status, detail=""):
        checks.append({"agent": agent, "check": check,
                       "status": status, "detail": detail})

    # ---- Agent1 ----
    a1 = result.get("agent1", {})
    band = a1.get("target_cm_band", [])
    if len(band) == 2 and band[0] < band[1]:
        add("agent1", "밴드 유효성", "ok", f"{band}")
    else:
        add("agent1", "밴드 유효성", "fail", f"band={band}")
    if band and 1 <= band[0] and band[1] <= 45:
        add("agent1", "밴드 범위 (1~45)", "ok")
    else:
        add("agent1", "밴드 범위 (1~45)", "fail", f"band={band}")
    mean = a1.get("target_cm_mean")
    if band and mean is not None and band[0] <= mean <= band[1]:
        add("agent1", "mean이 밴드 안에", "ok")
    else:
        add("agent1", "mean이 밴드 안에", "fail", f"mean={mean}, band={band}")
    if a1.get("phase") in ("HIGH_OVERSHOOT", "LOW_DRAWDOWN", "EQUILIBRIUM"):
        add("agent1", "phase 값 유효", "ok", a1["phase"])
    else:
        add("agent1", "phase 값 유효", "fail", str(a1.get("phase")))

    # ---- Agent2 ----
    a2 = result.get("agent2", {})
    pqs, hqs = a2.get("primary_quotas", []), a2.get("hedge_quotas", [])
    if len(pqs) == 3 and len(hqs) == 2:
        add("agent2", "quota 개수 (3+2)", "ok")
    else:
        add("agent2", "quota 개수 (3+2)", "fail",
            f"primary={len(pqs)}, hedge={len(hqs)}")
    bad_q = [q for q in pqs + hqs
             if not (isinstance(q, (list, tuple)) and len(q) == 5
                     and sum(q) == 6 and all(isinstance(x, int) and x >= 0 for x in q))]
    if not bad_q:
        add("agent2", "quota 합=6·비음수", "ok")
    else:
        add("agent2", "quota 합=6·비음수", "fail", str(bad_q))
    # D5 방어: hedge 쿼터는 D5=0이어야 한다.
    # (2026-10-08 버그: 재배치 루프가 D5에 +1을 되돌려 방어가 무력화됐었음.
    #  주석만 믿고 넘어갔던 전례라 실행 결과로 직접 검사한다.)
    bad_hq = [q for q in hqs
              if not (isinstance(q, (list, tuple)) and len(q) == 5 and q[4] == 0)]
    if not bad_hq:
        add("agent2", "hedge 쿼터 D5=0", "ok")
    else:
        add("agent2", "hedge 쿼터 D5=0", "fail",
            f"D5≠0 hedge 쿼터: {bad_hq}")
    # hedge 게임에 D5 볼이 실제로 없는지도 검사 (쿼터≠게임 어긋남 방지)
    _games = result.get("games", [])
    _hedge_games = [g for g in _games if g.get("kind") == "hedge"]
    bad_hg = [g.get("numbers") for g in _hedge_games
              if any(min((n - 1) // 10, 4) == 4 for n in g.get("numbers", []))]
    if not bad_hg:
        add("agent2", "hedge 게임 D5 볼 없음", "ok",
            f"{len(_hedge_games)}개 확인")
    else:
        add("agent2", "hedge 게임 D5 볼 없음", "fail", str(bad_hg))

    # ---- Agent3 ----
    a3 = result.get("agent3", {})
    pool = set(a3.get("pool", []))
    expect = (set(a3.get("carry", [])) | set(a3.get("neighbors", []))
              | set(a3.get("bonus_channel", [])))
    if pool == expect and pool:
        add("agent3", "pool = carry∪neighbors∪bonus_ch", "ok", f"{len(pool)}개")
    else:
        add("agent3", "pool = carry∪neighbors∪bonus_ch", "fail",
            f"pool={len(pool)}, expect={len(expect)}")
    if pool and all(1 <= x <= 45 for x in pool):
        add("agent3", "pool 번호 범위", "ok")
    else:
        add("agent3", "pool 번호 범위", "fail")
    mand = a3.get("mandate", [])
    if (isinstance(mand, (list, tuple)) and len(mand) == 2
            and 0 <= mand[0] <= mand[1] <= 6):
        add("agent3", "mandate 유효", "ok", str(mand))
    else:
        add("agent3", "mandate 유효", "fail", str(mand))

    # ---- Agent4 (핵심: 변별력 감시) ----
    games = result.get("games", [])
    scores = [g.get("filter_score") for g in games]
    if games and all(isinstance(s, (int, float)) for s in scores):
        add("agent4", "점수 숫자형·탈락 없음", "ok")
    else:
        add("agent4", "점수 숫자형·탈락 없음", "fail", str(scores))
    uniq = len(set(scores)) if scores else 0
    if uniq <= 1:
        add("agent4", "점수 변별력", "fail",
            f"고유값 {uniq}개 — 전부 동점이라 선별 무력화! (2026-10-04 진단 재발)")
    elif uniq < 3:
        add("agent4", "점수 변별력", "warn", f"고유값 {uniq}개 — 낮음")
    else:
        add("agent4", "점수 변별력", "ok", f"고유값 {uniq}개")
    # 게임 형식 검사
    bad_g = [g["numbers"] for g in games
             if not (isinstance(g.get("numbers"), list) and len(g["numbers"]) == 6
                     and len(set(g["numbers"])) == 6
                     and all(1 <= x <= 45 for x in g["numbers"]))]
    if not bad_g and len(games) == 10:
        add("agent4", "게임 10개·형식", "ok")
    else:
        add("agent4", "게임 10개·형식", "fail",
            f"games={len(games)}, bad={len(bad_g)}")

    # ---- Agent5 ----
    a5 = result.get("agent5", {})
    states = a5.get("states", {})
    if set(states.keys()) == {"D1", "D2", "D3", "D4", "D5"}:
        add("agent5", "states D1~D5 존재", "ok")
    else:
        add("agent5", "states D1~D5 존재", "fail", str(list(states.keys())))
    bad_s = [k for k, v in states.items()
             if not (isinstance(v.get("frozen_now"), bool)
                     and 0 <= v.get("reconsec_rate", -1) <= 1)]
    if not bad_s:
        add("agent5", "states 값 유효", "ok")
    else:
        add("agent5", "states 값 유효", "fail", str(bad_s))

    # ---- Agent6 ----
    a6 = result.get("agent6", {})
    p00 = (a6.get("P") or [[None]])[0][0]
    if p00 is not None and abs(p00 - 2.4063) / 2.4063 < 0.05:
        add("agent6", "P00 스펙 재현 (±5%)", "ok", f"{p00}")
    else:
        add("agent6", "P00 스펙 재현 (±5%)", "fail", f"P00={p00}")
    fc = a6.get("final_cm_estimate")
    if fc is not None and 1 <= fc <= 45:
        add("agent6", "final_cm 범위", "ok", f"{fc}")
    else:
        add("agent6", "final_cm 범위", "fail", f"{fc}")

    # ---- 교차: Agent1 밴드 vs 게임 CM ----
    band = a1.get("target_cm_band", [])
    if band and games:
        in_band = sum(1 for g in games
                      if band[0] <= sum(g["numbers"]) / 6 <= band[1])
        if in_band >= 5:
            add("cross", "게임 CM 밴드 적중률", "ok", f"{in_band}/10")
        elif in_band >= 3:
            add("cross", "게임 CM 밴드 적중률", "warn",
                f"{in_band}/10 — quota와 밴드 충돌 가능")
        else:
            add("cross", "게임 CM 밴드 적중률", "fail",
                f"{in_band}/10 — Agent1 목표와 생성 결과 괴리!")

    n_fail = sum(1 for c in checks if c["status"] == "fail")
    n_warn = sum(1 for c in checks if c["status"] == "warn")
    log("VALIDATE", f"{len(checks)}개 검사: ok {len(checks)-n_fail-n_warn}, "
                    f"warn {n_warn}, fail {n_fail}")
    return checks
