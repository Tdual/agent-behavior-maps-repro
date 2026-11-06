# -*- coding: utf-8 -*-
"""The six LLMs under common random numbers: behaviour vs the three coordinates.

Run 1 let every agent draw its own random stream, so two agents holding the
same policy still produced different trajectories. Here one uniform per
(task, step) is shared and inverted through each agent's CDF, so an identical
policy gives an identical sequence and the noise floor is zero.

Reads rollouts_crn/ and cross_crn/ (written by crn6.py) and the reference
coordinates of coords_full/ and close/coords/ (written by alpha14.py).
"""
import itertools, json, os
import numpy as np
from scipy.stats import spearmanr, pearsonr, wilcoxon
import pipeline as pl, close, close_report as cr, models, alpha14, agent

D = pl.D
NAMES = [n for n, _ in models.MODELS]
PAIRS = list(itertools.combinations(NAMES, 2))
EPS = 1e-300


def acts(sub):
    return {n: [[s["action"] for s in t["steps"]]
                for t in json.load(open(D(f"rollouts{sub}", f"{n}.json")))]
            for n in NAMES}


def behaviour(sub):
    A = acts(sub)
    out = {}
    for i, j in PAIRS:
        ed = np.mean([cr._edit(x, y) for x, y in zip(A[i], A[j])])
        jac = []
        for x, y in zip(A[i], A[j]):
            gx = {(x[k], x[k + 1]) for k in range(len(x) - 1)}
            gy = {(y[k], y[k + 1]) for k in range(len(y) - 1)}
            jac.append(1 - (len(gx & gy) / len(gx | gy) if gx | gy else 1.0))
        out[(i, j)] = dict(edit=float(ed), gram2=float(np.mean(jac)),
                           first=float(np.mean([x[0] != y[0]
                                                for x, y in zip(A[i], A[j])])))
    return out


def measured(sub):
    """Trajectory KL of j against i, over i's own rollout histories."""
    pol = {(s, o): [np.asarray(v) for v in
                    json.load(open(D(f"cross{sub}", f"{s}__{o}.json")))]
           for s in NAMES for o in NAMES}
    lens = {o: [len(t["steps"]) for t in
                json.load(open(D(f"rollouts{sub}", f"{o}.json")))] for o in NAMES}
    K = {}
    for i in NAMES:
        for j in NAMES:
            pi, pj, tot, k = pol[(i, i)], pol[(j, i)], [], 0
            for L in lens[i]:
                s = 0.0
                for _ in range(L):
                    p, q = pi[k], pj[k]
                    s += float((p * (np.log(p + EPS) - np.log(q + EPS))).sum())
                    k += 1
                tot.append(s)
            K[(i, j)] = float(np.mean(tot))
    return K, {o: float(np.mean(v)) for o, v in lens.items()}


# ------------------------------------------------- reference coordinates

def coord_raw():
    raw = alpha14.raw_vectors()
    M = len(raw[NAMES[0]])
    return {n: (raw[n] - raw[n].mean()) / np.sqrt(M) for n in NAMES}


def coord_norm():
    P, out = alpha14.policies(), {}
    for n in NAMES:
        idx = alpha14.chosen_index(n)
        v = np.array([np.log(P[n][k][idx[k]] + EPS) for k in range(len(idx))])
        out[n] = (v - v.mean()) / np.sqrt(len(v))
    return out


def coord_a0():
    P, out = alpha14.policies(), {}
    for n in NAMES:
        M = len(P[n])
        out[n] = np.concatenate([np.sqrt(v) for v in P[n]]) / np.sqrt(M)
    return out


def dmat(Z):
    return {p: float(np.linalg.norm(Z[p[0]] - Z[p[1]])) for p in PAIRS}


def effect(dist_by_pair, beh, metric, k):
    d = {p: beh[p][metric] for p in PAIRS}
    get = lambda i, j: d[(i, j)] if (i, j) in d else d[(j, i)]
    sd = np.std(list(d.values()), ddof=1)
    per = []
    for i in NAMES:
        others = sorted([j for j in NAMES if j != i],
                        key=lambda j: dist_by_pair.get((i, j), dist_by_pair.get((j, i))))
        per.append((np.mean([get(i, j) for j in others[k:]])
                    - np.mean([get(i, j) for j in others[:k]])) / sd)
    per = np.array(per)
    try:
        p = wilcoxon(per, alternative="two-sided").pvalue
    except Exception:
        p = float("nan")
    return float(per.mean()), float(p)


if __name__ == "__main__":
    DZ = {"α=1 raw": dmat(coord_raw()), "α=1 norm": dmat(coord_norm()),
          "α=0 √π": dmat(coord_a0())}
    res = {}
    for sub, tag in (("_crn", "共有乱数"), ("", "独立乱数（1回目）")):
        if not os.path.isdir(D(f"rollouts{sub}")):
            continue
        B = behaviour(sub)
        K, lens = measured(sub)
        sym = {p: K[p] + K[(p[1], p[0])] for p in PAIRS}
        succ = {n: sum(t["success"] for t in
                       json.load(open(D(f"rollouts{sub}", f"{n}.json")))) for n in NAMES}
        print(f"\n=== {tag}   平均軌跡長 {np.mean(list(lens.values())):.2f}  "
              f"成功 {sum(succ.values())}/{len(NAMES)*pl.N_EVAL}")
        rows = [{"指標": m,
                 "最小": f"{min(B[p][m] for p in PAIRS):.3f}",
                 "中央": f"{np.median([B[p][m] for p in PAIRS]):.3f}",
                 "最大": f"{max(B[p][m] for p in PAIRS):.3f}"}
                for m in ("edit", "gram2", "first")]
        cr.table("挙動の距離（15ペア）", rows, ["指標", "最小", "中央", "最大"])

        rows = []
        for name, dd in list(DZ.items()) + [("実測KL", sym)]:
            for m in ("edit", "gram2"):
                r = spearmanr([dd[p] for p in PAIRS], [B[p][m] for p in PAIRS])
                e2, p2 = effect(dd, B, m, 2)
                rows.append({"近さの基準": name, "挙動の指標": m,
                             "Spearman": f"{r[0]:.3f}", "p": f"{r[1]:.3f}",
                             "効果量k=2": f"{e2:.3f}", "p(W)": f"{p2:.3f}"})
        cr.table("座標の近さは挙動の近さを当てるか", rows,
                 ["近さの基準", "挙動の指標", "Spearman", "p", "効果量k=2", "p(W)"])

        rows = []
        for name, dd in DZ.items():
            pe = pearsonr([dd[p] for p in PAIRS], [sym[p] for p in PAIRS])
            sp = spearmanr([dd[p] for p in PAIRS], [sym[p] for p in PAIRS])
            rows.append({"座標": name, "Pearson": f"{pe[0]:.3f}",
                         "Spearman": f"{sp[0]:.3f}"})
        cr.table("座標距離 対 実測の軌跡KL（15ペア）", rows,
                 ["座標", "Pearson", "Spearman"])
        res[tag] = {"behaviour": {f"{i}|{j}": B[(i, j)] for i, j in PAIRS},
                    "sym_kl": {f"{i}|{j}": sym[(i, j)] for i, j in PAIRS},
                    "success": succ, "len": lens}
    res["coord"] = {k: {f"{i}|{j}": v[(i, j)] for i, j in PAIRS}
                    for k, v in DZ.items()}
    json.dump(res, open(D("crn6.json"), "w"), ensure_ascii=False, indent=1)
