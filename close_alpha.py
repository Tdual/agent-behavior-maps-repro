# -*- coding: utf-8 -*-
"""The alpha = 0 embedding, against the paper's alpha = 1 coordinate.

Amari's alpha-representation embeds a distribution as p^((1-alpha)/2):

    alpha =  1  ->  log p      the paper's coordinate
    alpha =  0  ->  sqrt(p)
    alpha = -1  ->  p

At alpha = 0 the map p -> 2*sqrt(p) is an isometry from the Fisher-Rao metric
on the simplex onto the sphere of radius 2, so no Taylor expansion and no
locality assumption are involved:

    ||sqrt(p) - sqrt(q)||^2 = 2(1 - BC),   BC = sum_a sqrt(p_a q_a)
    Fisher-Rao geodesic     = 2 arccos(BC)
    locally   KL ~ 2||sqrt(p) - sqrt(q)||^2
    globally  ||sqrt(p) - sqrt(q)||^2 <= KL        (a lower bound, never an
                                                    overestimate)

Three estimators of the symmetric trajectory KL are compared with the paper's.
Since K_ij ~ 2 T_bar * mean_t ||.||^2 and the Hellinger term is symmetric, the
symmetric prediction carries a factor 4.

The fourth entry is not an embedding at all. The forward pass already produces
a score for every admissible action; the paper keeps one of them and recovers
the rest with a bound. Keeping the whole policy vector instead gives the exact
KL under the reference distribution for every pair, at the same number of
forward passes.

All of this reads the files that are already on disk. No model is loaded.
"""
import itertools, json, sys
import numpy as np
from scipy.stats import pearsonr, spearmanr
import close, close_report as cr, close_behaviour as cb

NAMES, HA, TA = close.NAMES, close.HA, close.TA
PAIRS = list(itertools.combinations(NAMES, 2))
EPS = 1e-300


def ref_policies():
    """pi_i(.|h) at every reference step, per agent. Admissible sets are shared."""
    C = {h: json.load(open(close.D("coords", f"{h}.json"))) for h in close.HARNESS}
    out = {}
    for n in NAMES:
        c = C[HA[n]]
        out[n] = [close.agent.normalise(v, TA[n]) for v in c["vecs"]]
    M = len(out[NAMES[0]])
    return out, M


def stats(per_pair, sym):
    e = np.array([per_pair[p] for p in PAIRS])
    s = np.array([sym[p] for p in PAIRS])
    if e.std() == 0:
        return "—", "—", "—", "—"
    return (f"{pearsonr(e,s)[0]:.3f}", f"{spearmanr(e,s)[0]:.3f}",
            f"{np.median(e/s):.2f}", f"{(e/s).min():.2f}–{(e/s).max():.2f}")


if __name__ == "__main__":
    K, lens = cb.measured()                   # on-policy, shared-random rollouts
    sym = {p: K[p] + K[(p[1], p[0])] for p in PAIRS}
    tbar = float(np.mean(list(lens.values())))
    P, M = ref_policies()
    print(f"参照ステップ {M}  エージェント {len(NAMES)}体  ペア {len(PAIRS)}組  "
          f"平均軌跡長 {tbar:.2f}")

    hell, arc, exact = {}, {}, {}
    for i, j in PAIRS:
        bc = np.array([float(np.sqrt(a * b).sum()) for a, b in zip(P[i], P[j])])
        bc = np.clip(bc, 0.0, 1.0)
        hell[(i, j)] = 4.0 * tbar * float(np.mean(2.0 * (1.0 - bc)))
        arc[(i, j)] = 4.0 * tbar * float(np.mean(np.arccos(bc) ** 2))
        kl = 0.0
        for a, b in zip(P[i], P[j]):
            kl += float((a * (np.log(a + EPS) - np.log(b + EPS))).sum())
            kl += float((b * (np.log(b + EPS) - np.log(a + EPS))).sum())
        exact[(i, j)] = tbar * kl / M

    # the paper's alpha = 1 coordinates, step version, centred
    a1 = {}
    for key in ("raw", "norm"):
        step, _ = cr.load_coords(key, True)
        a1[key] = {p: tbar * cr.dist(step[p[0]], step[p[1]]) ** 2 for p in PAIRS}

    rows = []
    for name, d in (("α=1 log（論文の raw）", a1["raw"]),
                    ("α=1 log（許容集合で正規化）", a1["norm"]),
                    ("α=0 √（Hellinger）", hell),
                    ("α=0 √（Fisher–Rao 測地線）", arc),
                    ("埋め込みなし・参照分布上の厳密KL", exact)):
        pr, sp, lv, rng = stats(d, sym)
        rows.append({"方法": name, "Pearson": pr, "Spearman": sp,
                     "水準比(中央値)": lv, "水準比の範囲": rng})
    cr.table("実測の対称軌跡KL（共有乱数・on-policy）に対する当てはまり", rows,
             ["方法", "Pearson", "Spearman", "水準比(中央値)", "水準比の範囲"])

    # degeneracy on temperature-only pairs
    rows = []
    for p in sorted([p for p in PAIRS if cr.group(*p) == "温度のみ"], key=lambda p: sym[p]):
        rows.append({"ペア": f"{p[0]} / {p[1]}",
                     "実測対称KL": f"{sym[p]:.4f}",
                     "α=1 raw": f"{a1['raw'][p]:.2e}",
                     "α=1 norm": f"{a1['norm'][p]:.4f}",
                     "α=0 √": f"{hell[p]:.4f}",
                     "厳密KL": f"{exact[p]:.4f}"})
    cr.table("温度だけが違う6ペア（論文の raw は原理的に区別できない）", rows,
             ["ペア", "実測対称KL", "α=1 raw", "α=1 norm", "α=0 √", "厳密KL"])

    # tail sensitivity: how much of each quantity comes from the low-probability actions
    print("\n裾の寄与（各ステップで、実測方策の確率が下位半分の行動が占める割合の中央値）")
    sh, sk = [], []
    for i, j in PAIRS:
        fh, fk = [], []
        for a, b in zip(P[i], P[j]):
            o = np.argsort(a)[: len(a) // 2]
            t = 2.0 * (1.0 - float(np.sqrt(a * b).sum()))
            th = float((np.sqrt(a[o]) - np.sqrt(b[o])) ** 2 @ np.ones(len(o)))
            k = float((a * (np.log(a + EPS) - np.log(b + EPS))).sum())
            tk = float((a[o] * (np.log(a[o] + EPS) - np.log(b[o] + EPS))).sum())
            if t > 1e-12:
                fh.append(th / t)
            if abs(k) > 1e-12:
                fk.append(tk / k)
        sh.append(np.median(fh)); sk.append(np.median(fk))
    print(f"  α=0 √ の距離に占める裾の割合   中央値 {np.median(sh):.3f}")
    print(f"  KL に占める裾の割合             中央値 {np.median(sk):.3f}")
    print(f"  参考: α=1 log は ε（裾の最大対数比）が中央値 10.97 で、値が裾で非有界")

    # Separate embedding error from reference mismatch.
    #
    # Every Q-based quantity approximates the exact KL under Q. How well that
    # exact quantity itself predicts the on-policy KL is therefore a ceiling
    # for anything computed from the reference set. Any method that beats the
    # ceiling is not approximating KL_Q.
    rows = []
    for name, d in (("α=1 log（論文の raw）", a1["raw"]),
                    ("α=1 log（許容集合で正規化）", a1["norm"]),
                    ("α=0 √（Hellinger）", hell),
                    ("α=0 √（Fisher–Rao 測地線）", arc)):
        pr, sp, lv, rng = stats(d, exact)
        rows.append({"方法": name, "Pearson": pr, "Spearman": sp,
                     "水準比(中央値)": lv, "水準比の範囲": rng})
    cr.table("参照分布上の厳密KL に対する当てはまり（埋め込みの誤差だけを見る）", rows,
             ["方法", "Pearson", "Spearman", "水準比(中央値)", "水準比の範囲"])
    pr, sp, _, _ = stats(exact, sym)
    print(f"  参照分布上の厳密KL が on-policy の実測KL を予測する力: "
          f"Pearson {pr} / Spearman {sp}")
    print("  これが参照集合から計算するあらゆる量の上限であり、"
          "超える方法は KL_Q を近似していない")

    # bound direction
    viol = sum(1 for p in PAIRS if hell[p] > sym[p])
    print(f"\n下限の向きの確認: α=0 √ の推定が実測KLを上回ったペア {viol}/{len(PAIRS)}")
    print(f"  （理論は ‖√p−√q‖² ≤ KL。係数4を掛けた推定なので上回ることはあり得る）")

    json.dump({"sym_kl": {f"{i}|{j}": sym[p] for p in PAIRS for i, j in [p]},
               "hellinger": {f"{i}|{j}": hell[p] for p in PAIRS for i, j in [p]},
               "arc": {f"{i}|{j}": arc[p] for p in PAIRS for i, j in [p]},
               "exact_ref_kl": {f"{i}|{j}": exact[p] for p in PAIRS for i, j in [p]},
               "a1_raw": {f"{i}|{j}": a1["raw"][p] for p in PAIRS for i, j in [p]},
               "a1_norm": {f"{i}|{j}": a1["norm"][p] for p in PAIRS for i, j in [p]}},
              open(close.D("alpha.json"), "w"), ensure_ascii=False, indent=1)
