# -*- coding: utf-8 -*-
"""Does the map survive a change of reference set?

Three reference sets, the same nine agents (one LLM, six harnesses, four
temperatures):

  setA  the original: 12 oracle + 12 random trajectories, seeds 0-23
  setB  the same recipe, seeds 5000-5023
  setC  seeds 0-23 but every trajectory from the random generator

For each set the 36 pair distances are recomputed in the three coordinates,
together with the exact per-step symmetric KL on that set's own states. Two
questions: does each coordinate keep its ranking of the pairs, and does the
quantity it is supposed to track move with it.
"""
import itertools, json, os
import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.spatial import procrustes
import close, viz, agent

HERE = os.path.dirname(os.path.abspath(__file__))
D = lambda *p: os.path.join(HERE, *p)
NAMES, HA, TA = close.NAMES, close.HA, close.TA
PAIRS = list(itertools.combinations(NAMES, 2))
EPS = 1e-300
SETS = {"A": lambda h: close.D("coords", f"{h}.json"),
        "B": lambda h: D("coords_B", f"{h}.json"),
        "C": lambda h: D("coords_C", f"{h}.json")}


def load(tag):
    C = {h: json.load(open(SETS[tag](h))) for h in close.HARNESS}
    pol = {n: [agent.normalise(v, TA[n]) for v in C[HA[n]]["vecs"]] for n in NAMES}
    idx = C["h4"]["idx"]
    M = len(idx)
    raw = {n: np.array([C[HA[n]]["vecs"][k][idx[k]] for k in range(M)]) for n in NAMES}
    z = {}
    z["α=1 raw"] = {n: (raw[n] - raw[n].mean()) / np.sqrt(M) for n in NAMES}
    nv = {n: np.array([np.log(pol[n][k][idx[k]] + EPS) for k in range(M)]) for n in NAMES}
    z["α=1 norm"] = {n: (nv[n] - nv[n].mean()) / np.sqrt(M) for n in NAMES}
    z["α=0 √π"] = {n: np.concatenate([np.sqrt(v) for v in pol[n]]) / np.sqrt(M)
                   for n in NAMES}
    kl = {}
    for i, j in PAIRS:
        s = []
        for p, q in zip(pol[i], pol[j]):
            s.append(float((p * np.log((p + 1e-12) / (q + 1e-12))).sum()
                           + (q * np.log((q + 1e-12) / (p + 1e-12))).sum()))
        kl[(i, j)] = float(np.mean(s))
    dd = {k: {p: float(np.linalg.norm(v[p[0]] - v[p[1]])) for p in PAIRS}
          for k, v in z.items()}
    return dd, kl, M


if __name__ == "__main__":
    S = {}
    for t in SETS:
        if all(os.path.exists(SETS[t](h)) for h in close.HARNESS):
            S[t] = load(t)
            print(f"set{t}: {S[t][2]}ステップ")

    rows = []
    for t, (dd, kl, _) in S.items():
        for k, d in dd.items():
            pe = pearsonr([d[p] for p in PAIRS], [kl[p] for p in PAIRS])[0]
            sp = spearmanr([d[p] for p in PAIRS], [kl[p] for p in PAIRS])[0]
            rows.append({"参照集合": f"set{t}", "座標": k,
                         "Pearson": f"{pe:.3f}", "Spearman": f"{sp:.3f}",
                         "距離の中央": f"{np.median([d[p] for p in PAIRS]):.3f}",
                         "KLの中央": f"{np.median([kl[p] for p in PAIRS]):.3f}"})
    import close_report as cr
    cr.table("参照集合ごとの、座標距離 対 その集合上の対称KL（36ペア）", rows,
             ["参照集合", "座標", "Pearson", "Spearman", "距離の中央", "KLの中央"])

    if "A" in S and len(S) > 1:
        rows = []
        for t in [x for x in S if x != "A"]:
            for k in S["A"][0]:
                a = np.array([S["A"][0][k][p] for p in PAIRS])
                b = np.array([S[t][0][k][p] for p in PAIRS])
                m = a > 0
                rows.append({"比較": f"setA 対 set{t}", "座標": k,
                             "Pearson": f"{pearsonr(a, b)[0]:.3f}",
                             "Spearman": f"{spearmanr(a, b)[0]:.3f}",
                             "距離の比(中央)": f"{np.median(b[m] / a[m]):.3f}"
                             + ("" if m.all() else f"（{m.sum()}組）")})
            a = np.array([S["A"][1][p] for p in PAIRS])
            b = np.array([S[t][1][p] for p in PAIRS])
            rows.append({"比較": f"setA 対 set{t}", "座標": "対称KL(実測)",
                         "Pearson": f"{pearsonr(a, b)[0]:.3f}",
                         "Spearman": f"{spearmanr(a, b)[0]:.3f}",
                         "距離の比(中央)": f"{np.median(b / a):.3f}"})
        cr.table("参照集合を替えたとき、ペアの距離はどれだけ動くか", rows,
                 ["比較", "座標", "Pearson", "Spearman", "距離の比(中央)"])

        rows = []
        for t in [x for x in S if x != "A"]:
            for k in S["A"][0]:
                DA = np.array([[S["A"][0][k].get((i, j), S["A"][0][k].get((j, i), 0.0))
                                if i != j else 0.0 for j in NAMES] for i in NAMES])
                DB = np.array([[S[t][0][k].get((i, j), S[t][0][k].get((j, i), 0.0))
                                if i != j else 0.0 for j in NAMES] for i in NAMES])
                XA = viz.mds(DA)[0]
                XB = viz.mds(DB)[0]
                _, _, dis = procrustes(XA, XB)
                rows.append({"比較": f"setA 対 set{t}", "座標": k,
                             "Procrustes 不一致": f"{dis:.3f}"})
        cr.table("2次元に落とした地図の食い違い（0 なら同じ配置）", rows,
                 ["比較", "座標", "Procrustes 不一致"])
    json.dump({t: {"dist": {k: {f"{i}|{j}": v[(i, j)] for i, j in PAIRS}
                            for k, v in S[t][0].items()},
                   "kl": {f"{i}|{j}": S[t][1][(i, j)] for i, j in PAIRS},
                   "M": S[t][2]} for t in S},
              open(D("refs2.json"), "w"), ensure_ascii=False, indent=1)
