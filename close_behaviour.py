# -*- coding: utf-8 -*-
"""Do coordinate neighbours behave alike? -- with the paper's own statistic.

Two mistakes in the first pass are fixed here.

1. The statistic. The paper's 0.88 / 0.96 are a standardised mean difference
   (Table 8): for each model, the mean behavioural distance to the remaining
   models minus the mean distance to its five nearest neighbours, divided by
   the standard deviation of the behavioural distance over all pairs, then
   averaged over models, with a Wilcoxon signed-rank test paired by model. A
   Spearman correlation over pairs is a different quantity and was reported in
   its place. With nine agents, five neighbours out of eight others is
   degenerate, so k = 2 and k = 3 are reported.

2. The sampling. Each agent used to roll out on its own random stream, so two
   agents holding the same policy still diverged: replicates of one agent sat
   0.44-0.58 apart in normalised edit distance, which is the width of the
   whole between-agent range. The rollouts read here instead share one uniform
   per (task, step) and invert each agent's CDF, so an identical policy gives
   an identical sequence.

Run close.py roll_crn cross_crn first.
"""
import itertools, json, os, sys
import numpy as np
from scipy.stats import spearmanr, wilcoxon
import close, close_report as cr

D = close.D
NAMES, HA, TA = close.NAMES, close.HA, close.TA
PAIRS = list(itertools.combinations(NAMES, 2))
SUB = "_crn" if "--crn" in sys.argv else ""
RD, CD = f"rollouts{SUB}", f"cross{SUB}"


def acts(n):
    return [[s["action"] for s in t["steps"]]
            for t in json.load(open(D(RD, f"{n}.json")))]


def behaviour():
    A = {n: acts(n) for n in NAMES}
    out = {}
    for i, j in PAIRS:
        ed = np.mean([cr._edit(x, y) for x, y in zip(A[i], A[j])])
        jac = []
        for x, y in zip(A[i], A[j]):
            gx = {(x[k], x[k + 1]) for k in range(len(x) - 1)}
            gy = {(y[k], y[k + 1]) for k in range(len(y) - 1)}
            jac.append(1 - (len(gx & gy) / len(gx | gy) if gx | gy else 1.0))
        out[(i, j)] = dict(edit=float(ed), gram2=float(np.mean(jac)),
                           first=float(np.mean([x[0] != y[0] for x, y in zip(A[i], A[j])])))
    return out


def measured():
    cr_ = {(h, o): [np.array(v) for v in json.load(open(D(CD, f"{h}__{o}.json")))]
           for h in close.HARNESS for o in NAMES}
    lens = {o: [len(t["steps"]) for t in json.load(open(D(RD, f"{o}.json")))]
            for o in NAMES}
    pol = {(n, o): [close.agent.normalise(v, TA[n]) for v in cr_[(HA[n], o)]]
           for n in NAMES for o in NAMES}
    K = {}
    for i in NAMES:
        for j in NAMES:
            pi, pj = pol[(i, i)], pol[(j, i)]
            tot, k = [], 0
            for L in lens[i]:
                s = 0.0
                for _ in range(L):
                    p, q = pi[k], pj[k]
                    s += float((p * (np.log(p + 1e-300) - np.log(q + 1e-300))).sum())
                    k += 1
                tot.append(s)
            K[(i, j)] = float(np.mean(tot))
    return K, {o: float(np.mean(v)) for o, v in lens.items()}


def coord_dist(key="norm", center=True):
    step, _ = cr.load_coords(key, center)
    return {p: cr.dist(step[p[0]], step[p[1]]) for p in PAIRS}


def effect(dist_by_pair, beh, metric, k):
    """The paper's standardised mean difference, averaged over agents."""
    d = {p: beh[p][metric] for p in PAIRS}
    get = lambda i, j: d[(i, j)] if (i, j) in d else d[(j, i)]
    sd = np.std(list(d.values()), ddof=1)
    per = []
    for i in NAMES:
        others = sorted([j for j in NAMES if j != i],
                        key=lambda j: dist_by_pair.get((i, j), dist_by_pair.get((j, i))))
        near, far = others[:k], others[k:]
        per.append((np.mean([get(i, j) for j in far])
                    - np.mean([get(i, j) for j in near])) / sd)
    per = np.array(per)
    try:
        p = wilcoxon(per, alternative="two-sided").pvalue
    except Exception:
        p = float("nan")
    return float(per.mean()), float(p), per


if __name__ == "__main__":
    tag = "共有乱数" if SUB else "独立乱数（旧）"
    print(f"=== {tag}  ロールアウト: close/{RD}")
    B = behaviour()
    K, lens = measured()
    sym = {p: K[p] + K[(p[1], p[0])] for p in PAIRS}
    tbar = float(np.mean(list(lens.values())))
    succ = {n: sum(t["success"] for t in json.load(open(D(RD, f"{n}.json"))))
            for n in NAMES}
    print(f"平均軌跡長 {tbar:.2f}  成功 {sum(succ.values())}/{len(NAMES)*close.N_EVAL}")

    for m in ("edit", "gram2", "first"):
        v = np.array([B[p][m] for p in PAIRS])
        print(f"{m:<6} 最小{v.min():.3f} 中央{np.median(v):.3f} 最大{v.max():.3f} "
              f"標準偏差{v.std(ddof=1):.3f}")

    dn = coord_dist("norm", True)
    dr = coord_dist("raw", True)

    rows = []
    for name, dd in (("norm座標", dn), ("raw座標", dr), ("実測KL", sym)):
        for m in ("edit", "gram2"):
            for k in (2, 3):
                e, p, _ = effect(dd, B, m, k)
                rows.append({"近さの基準": name, "挙動の指標": m, "近傍k": k,
                             "効果量": f"{e:.3f}", "p(Wilcoxon)": f"{p:.3f}"})
    cr.table(f"論文と同じ統計量: 標準化平均差（近傍k体 対 残り、{len(NAMES)}体で平均）",
             rows, ["近さの基準", "挙動の指標", "近傍k", "効果量", "p(Wilcoxon)"])
    print("  論文（ALFWorld・100モデル・近傍5体）: edit 0.88 / 2-gram 0.96")

    rows = []
    for m in ("edit", "gram2", "first"):
        b = [B[p][m] for p in PAIRS]
        r1 = spearmanr([dn[p] for p in PAIRS], b)
        r2 = spearmanr([sym[p] for p in PAIRS], b)
        rows.append({"挙動の指標": m, "norm座標距離": f"{r1[0]:.3f}", "p": f"{r1[1]:.3f}",
                     "実測KL": f"{r2[0]:.3f}", "p ": f"{r2[1]:.3f}"})
    cr.table("参考: 36ペアの Spearman 相関（論文の統計量とは別物）", rows,
             ["挙動の指標", "norm座標距離", "p", "実測KL", "p "])

    rows = []
    for p in sorted(PAIRS, key=lambda p: sym[p])[:8]:
        rows.append({"ペア": f"{p[0]}|{p[1]}", "対称KL": f"{sym[p]:.3f}",
                     "norm距離": f"{dn[p]:.3f}", "edit": f"{B[p]['edit']:.3f}",
                     "gram2": f"{B[p]['gram2']:.3f}", "群": cr.group(*p)})
    cr.table("実測KL の近い順（上位8ペア）", rows,
             ["ペア", "対称KL", "norm距離", "edit", "gram2", "群"])

    json.dump({"behaviour": {f"{i}|{j}": B[(i, j)] for i, j in PAIRS},
               "sym_kl": {f"{i}|{j}": sym[(i, j)] for i, j in PAIRS},
               "norm_dist": {f"{i}|{j}": dn[(i, j)] for i, j in PAIRS},
               "effect": {f"{n}|{m}|{k}": effect(d, B, m, k)[0]
                          for n, d in (("norm", dn), ("raw", dr), ("kl", sym))
                          for m in ("edit", "gram2") for k in (2, 3)},
               "success": succ},
              open(D(f"behaviour{SUB or '_indep'}.json"), "w"),
              ensure_ascii=False, indent=1)
