#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
표본 분할(out-of-sample) 입증 — 앞부분으로 규칙을 발견하고 뒷부분에서 검증.
발견 구간: 837~1040회 (204행) / 검증 구간: 1041~1243회 (203행)
"""
import pandas as pd, numpy as np
from math import erf, sqrt

XLSX = "/home/hatch/workspace/user/files/645_408.xlsx"
bonus = pd.read_csv("/home/hatch/workspace/lotto/bonus_836_1243.csv")
bmap = dict(zip(bonus["회차"].astype(int), bonus["보너스"].astype(int)))
Phi = lambda z: 0.5*(1+erf(z/sqrt(2)))
binom_p = lambda k, n, p=0.5: 2*(1-Phi(abs((k-n*p)/sqrt(n*p*(1-p))))) if n*p*(1-p) > 0 else 1.0

df = pd.read_excel(XLSX).sort_values("회차").reset_index(drop=True)
nums = np.array([list(map(int, str(r).split(","))) for r in df["실제번호"]])
draws = df["회차"].astype(int).values
cms = nums.mean(axis=1)
EDGES = [(1,10),(11,20),(21,30),(31,40),(41,45)]

disc = draws <= 1040
val = draws > 1040

def band_counts(X):
    return np.array([[np.sum((x>=a)&(x<=b)) for a,b in EDGES] for x in X])

def reversion_rate(c, idx, hi=26, lo=20):
    r_hi = [i for i in idx if c[i] > hi and i+1 < len(c)]
    r_lo = [i for i in idx if c[i] < lo and i+1 < len(c)]
    h = sum(c[i+1] < c[i] for i in r_hi); l = sum(c[i+1] > c[i] for i in r_lo)
    return (h, len(r_hi), l, len(r_lo))

def rebound_rate(X, edges):
    tot=hit=0
    for a,b in edges:
        c = np.array([np.sum((x>=a)&(x<=b)) for x in X])
        for t in range(len(X)-1):
            if c[t]==0:
                tot += 1
                if c[t+1] >= 1: hit += 1
    return hit, tot

def transition_capture(X, dnums):
    caps=[]
    for j in range(1, len(X)):
        prev, cur = set(X[j-1]), set(X[j])
        carry = len(cur & prev)
        neigh = len([x for x in cur-prev if any(abs(x-p)==1 for p in prev)])
        pb = bmap.get(int(dnums[j-1]))
        bch = len([x for x in cur-prev if pb is not None and abs(x-pb)<=1
                   and not any(abs(x-p)==1 for p in prev)])
        caps.append(carry+neigh+bch)
    return np.array(caps)

di = np.where(disc)[0]; vi = np.where(val)[0]
print("="*68); print("표본 분할 입증: 발견 837~1040회(n=204) → 검증 1041~1243회(n=203)"); print("="*68)

# A. CM 평균회귀
h1,n1,l1,m1 = reversion_rate(cms, di); h2,n2,l2,m2 = reversion_rate(cms, vi)
print(f"\n[A] CM 평균회귀")
print(f"  발견: 고점하락 {h1}/{n1}={h1/n1*100:.1f}%, 저점상승 {l1}/{m1}={l1/m1*100:.1f}%")
print(f"  검증: 고점하락 {h2}/{n2}={h2/n2*100:.1f}% (p={binom_p(h2,n2):.2e}), 저점상승 {l2}/{m2}={l2/m2*100:.1f}% (p={binom_p(l2,m2):.2e})")
va = h2/n2 > 0.70 and l2/m2 > 0.70 and binom_p(h2,n2) < 0.01 and binom_p(l2,m2) < 0.01
print(f"  → {'입증' if va else '미입증'}")

# B. 결빙 반등 (10번호 구간)
b1 = rebound_rate(nums[di], EDGES[:4]); b2 = rebound_rate(nums[vi], EDGES[:4])
print(f"\n[B] 결빙 반등 (10번호 구간)")
print(f"  발견: {b1[0]}/{b1[1]}={b1[0]/b1[1]*100:.1f}%")
print(f"  검증: {b2[0]}/{b2[1]}={b2[0]/b2[1]*100:.1f}%")
vb = b2[0]/b2[1] > 0.75
print(f"  → {'입증' if vb else '미입증'}")

# C. 전이 사슬
c1 = transition_capture(nums[di], draws[di]); c2 = transition_capture(nums[vi], draws[vi])
p1 = np.mean((c1>=2)&(c1<=3)); p2 = np.mean((c2>=2)&(c2<=3))
print(f"\n[C] 전이 사슬 포획")
print(f"  발견: 평균 {c1.mean():.2f}개, 2~3개 {p1*100:.1f}%")
print(f"  검증: 평균 {c2.mean():.2f}개, 2~3개 {p2*100:.1f}%")
vc = 2.0 <= c2.mean() <= 2.8 and p2 > 0.60
print(f"  → {'입증' if vc else '미입증'}")

# D. 대역 음의 상관 (양쪽 구간 모두 전부 음수인가)
neg1 = [np.corrcoef(band_counts(nums[di]).T)[i,j] < 0 for i in range(5) for j in range(i+1,5)]
neg2 = [np.corrcoef(band_counts(nums[vi]).T)[i,j] < 0 for i in range(5) for j in range(i+1,5)]
print(f"\n[D] 대역 음의 상관")
print(f"  발견 구간 전부 음수: {all(neg1)}, 검증 구간 전부 음수: {all(neg2)}")
vd = all(neg1) and all(neg2)
print(f"  → {'입증' if vd else '미입증'}")

# E. 갈튼보드 수렴
mu2, sd2 = cms[vi].mean(), cms[vi].std(ddof=1)
sk2 = ((cms[vi]-mu2)**3).mean()/sd2**3
print(f"\n[E] 갈튼보드 중심 수렴 (검증 구간)")
print(f"  평균 {mu2:.2f} (이론 23.0), SD {sd2:.2f} (이론 4.99), 왜도 {sk2:.2f}")
ve = abs(mu2-23.0) < 0.5 and abs(sk2) < 0.3
print(f"  → {'입증' if ve else '미입증'}")

print("\n"+"="*68)
ok = sum([va,vb,vc,vd,ve])
print(f"종합: {ok}/5 입증 — " + ("이론의 5개 기둥이 표본 분할에서도 안정적으로 재현됨" if ok==5 else "일부 기둥 불안정"))
print("="*68)
print("정직 고지: 이는 규칙의 안정성 입증이지, 당첨번호 예측 우위 입증이 아님.")
