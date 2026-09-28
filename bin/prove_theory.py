#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
유준열 이론 입증 스크립트 — 검증 가능한 명제를 실데이터로 검정하고
입증/부분입증/미입증/반증 판정을 내린다.
데이터: ~/workspace/user/files/645_408.xlsx (837~1243회, 407행)
실행: python3 prove_theory.py
"""
import pandas as pd, numpy as np
from math import comb, erf, sqrt

XLSX = "/home/hatch/workspace/user/files/645_408.xlsx"
PRIMES = {2,3,5,7,11,13,17,19,23,29,31,37,41,43}

defPhi = lambda z: 0.5*(1+erf(z/sqrt(2)))
def binom_p(k, n, p=0.5):
    # 양측 p-value (정규근사)
    z = (k - n*p)/sqrt(n*p*(1-p))
    return 2*(1-defPhi(abs(z)))

df = pd.read_excel(XLSX).sort_values("회차").reset_index(drop=True)
nums = np.array([list(map(int, str(r).split(","))) for r in df["실제번호"]])
n = len(nums)
cms = nums.mean(axis=1)
sums = nums.sum(axis=1)
# 5대역: D1 1-10, D2 11-20, D3 21-30, D4 31-40, D5 41-45
EDGES = [(1,10),(11,20),(21,30),(31,40),(41,45)]
band = np.array([[[np.sum((x>=a)&(x<=b)) for a,b in EDGES] for x in nums]])
band = band[0]

R = []
def verdict(name, claim, method, result, v):
    R.append((name, claim, method, result, v))

print("="*70); print("유준열 이론 입증 리포트"); print("="*70)

# T1: 대역 간 상관계수 전부 음수
corrs = np.corrcoef(band.T)
neg = [corrs[i,j] for i in range(5) for j in range(i+1,5)]
v = "입증" if all(c < 0 for c in neg) else "반증"
verdict("T1 슬롯총량·음의상관", "5대역 10개 쌍 상관계수 전부 음수",
        f"n={n} 피어슨 상관계수", f"범위 [{min(neg):.3f}, {max(neg):.3f}], 전부 음수 {all(c<0 for c in neg)}", v)
print(f"[T1] 상관계수 범위 [{min(neg):.3f}, {max(neg):.3f}] → {v}")

# T2: CM 평균회귀
hi = np.where(cms[:-1] > 26)[0]; lo = np.where(cms[:-1] < 20)[0]
hi_hit = np.sum(cms[hi+1] < cms[hi]); lo_hit = np.sum(cms[lo+1] > cms[lo])
p1, p2 = binom_p(hi_hit, len(hi)), binom_p(lo_hit, len(lo))
v = "입증" if (p1 < 1e-6 and p2 < 1e-6) else "미입증"
verdict("T2 CM 평균회귀", "CM>26→하락, CM<20→상승",
        "이항검정 vs 50%",
        f"고점 {hi_hit}/{len(hi)}={hi_hit/len(hi)*100:.1f}% (p={p1:.2e}), 저점 {lo_hit}/{len(lo)}={lo_hit/len(lo)*100:.1f}% (p={p2:.2e})", v)
print(f"[T2] 고점하락 {hi_hit}/{len(hi)} ({hi_hit/len(hi)*100:.1f}%), 저점상승 {lo_hit}/{len(lo)} ({lo_hit/len(lo)*100:.1f}%) → {v}")

# T3: 결빙 탄성 반등 (5대역 + 10번호구간)
for label, edges in [("5대역", EDGES), ("10번호구간", [(1,10),(11,20),(21,30),(31,40)])]:
    tot=hit=0; nxt=[]
    for a,b in edges:
        c = np.array([np.sum((x>=a)&(x<=b)) for x in nums])
        for t in range(n-1):
            if c[t]==0:
                tot+=1; nxt.append(c[t+1])
                if c[t+1]>=1: hit+=1
    v = "입증" if hit/tot > 0.75 else "부분입증"
    verdict(f"T3 결빙반등({label})", "결빙 대역은 직후 회차에 1개 이상 출현",
            f"{tot}건", f"반등률 {hit/tot*100:.1f}%, 직후 평균 {np.mean(nxt):.2f}개", v)
    print(f"[T3-{label}] 반등률 {hit/tot*100:.1f}% ({hit}/{tot}) → {v}")

# T4: 전이 사슬 (이월 + 이웃±1 + 보너스채널)
bonus = pd.read_csv("/home/hatch/workspace/lotto/bonus_836_1243.csv")
bmap = dict(zip(bonus["회차"].astype(int), bonus["보너스"].astype(int)))
draws = df["회차"].astype(int).values
tot_cap, cnt23 = [], 0
for i in range(1, n):
    prev = set(nums[i-1]); cur = set(nums[i])
    carry = len(cur & prev)
    neigh = len([x for x in cur - prev
                 if any(abs(x-p) == 1 for p in prev)])
    pb = bmap.get(int(draws[i-1]))
    bch = len([x for x in cur - prev
               if pb is not None and abs(x-pb) <= 1
               and not any(abs(x-p) == 1 for p in prev)])
    tot = carry + neigh + bch
    tot_cap.append(tot)
    if 2 <= tot <= 3: cnt23 += 1
p23 = cnt23/len(tot_cap)
v = "입증" if 2.0 <= np.mean(tot_cap) <= 2.8 and p23 > 0.6 else "부분입증" if p23 > 0.5 else "미입증"
verdict("T4 전이 사슬", "이월+이웃+보너스 전이망이 평균 2~3개 포획",
        f"{len(tot_cap)}건", f"평균 {np.mean(tot_cap):.2f}개, 2~3개 비율 {p23*100:.1f}%", v)
print(f"[T4] 전이망 평균 {np.mean(tot_cap):.2f}개, 2~3개 {p23*100:.1f}% → {v}")

# T5: 갈튼보드 중심 수렴
mu, sd = cms.mean(), cms.std(ddof=1)
skew = ((cms-mu)**3).mean()/sd**3
# 이론: 평균 23.0, CM 표준편차 = sqrt(168.67/6*39/44) ≈ 4.99
v = "입증" if abs(mu-23.0) < 0.5 and abs(skew) < 0.3 else "부분입증"
verdict("T5 갈튼보드 수렴", "CM 분포가 23 중심 정규형에 수렴",
        f"n={n}", f"평균 {mu:.2f} (이론 23.0), SD {sd:.2f} (이론 4.99), 왜도 {skew:.2f}", v)
print(f"[T5] 평균 {mu:.2f}, SD {sd:.2f}, 왜도 {skew:.2f} → {v}")

# T6: 화음 필터 (경험 분포)
pr = np.array([sum(x in PRIMES for x in r) for r in nums])
odd = np.array([sum(x % 2 == 1 for x in r) for r in nums])
span = nums.max(axis=1)-nums.min(axis=1)
gaps = np.array([max(np.diff(sorted(r))) for r in nums])
c_pr = np.mean((pr>=1)&(pr<=3)); c_odd = np.mean((odd>=2)&(odd<=4))
c_span = np.mean((span>=33)&(span<=42)); c_gap = np.mean(gaps<=14)
v = "입증" if min(c_pr,c_odd,c_span,c_gap) > 0.75 else "부분입증"
verdict("T6 화음 필터", "소수1~3·홀짝균형·스팬33~42·최대간격≤14가 대다수",
        f"n={n}", f"소수 {c_pr*100:.1f}%, 홀짝 {c_odd*100:.1f}%, 스팬 {c_span*100:.1f}%, 간격 {c_gap*100:.1f}%", v)
print(f"[T6] 소수{c_pr*100:.1f}% 홀짝{c_odd*100:.1f}% 스팬{c_span*100:.1f}% 간격{c_gap*100:.1f}% → {v}")

# T7: 호기 지문 (반증 기록)
verdict("T7 호기 지문", "기계별 고유 번호 특성 존재",
        "2026-09-28 별도 검증 (z-test)",
        "30번대평균 p≈0.103, 결빙률 p≈0.32, 총합 p≈0.36 — 전부 유의미하지 않음", "반증")
print("[T7] 호기별 차이 유의미하지 않음 → 반증 (별도 영수증 참조)")

# T8: 반전 주기 1.47 (정의 의존)
ep=[]
for t in range(n):
    if cms[t] > 26:
        for k in range(1, n-t):
            if cms[t+k] < 23.14: ep.append(k); break
    elif cms[t] < 20:
        for k in range(1, n-t):
            if cms[t+k] > 23.14: ep.append(k); break
v = "미입증"  # 정의가 명세서에 없어 재현 불가
verdict("T8 반전주기 1.47", "극단 이탈 후 평균 1.47회차 만에 반대편 회귀",
        f"{len(ep)}건 (정의: 장기평균 23.14 반대편 돌파)",
        f"실측 {np.mean(ep):.2f}회차 — 명세서에 정의 없어 재현 불가", v)
print(f"[T8] 실측 {np.mean(ep):.2f}회차 (주장 1.47) → {v}")

print("="*70)
print("판정 요약")
for name, claim, method, result, v in R:
    print(f"  [{v}] {name}: {result}")
print("="*70)

import json
with open("/home/hatch/workspace/receipts/proof_theory_20260928.json","w") as f:
    json.dump([{"명제":a,"주장":b,"방법":c,"결과":d,"판정":e} for a,b,c,d,e in R],
              f, ensure_ascii=False, indent=2)
print("JSON 저장: ~/workspace/receipts/proof_theory_20260928.json")
