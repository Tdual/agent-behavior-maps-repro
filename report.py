# -*- coding: utf-8 -*-
"""Coordinate distances vs. measured trajectory KL, and vs. measured behaviour.

With six agents there are only 15 pairs, so the paper's "five nearest
neighbours" effect size is replaced by rank correlation over all pairs, which is
the right analogue at this size.
"""
import itertools, json, os
import numpy as np
from scipy.stats import pearsonr, spearmanr
import models

HERE = os.path.dirname(os.path.abspath(__file__))
D = lambda *p: os.path.join(HERE, *p)
NAMES = [n for n, _ in models.MODELS]
VERBS = ["go", "open", "close", "examine", "take", "put", "heat", "cool", "clean"]


# -------------------------------------------------------- coordinates

def load_coords(key="raw", center=True):
    """Returns step-level and trajectory-level coordinate vectors per agent."""
    step, traj = {}, {}
    for n in NAMES:
        z = np.load(D("coords", f"{n}.npz"))
        v, tid = z[key].astype(float), z["tid"]
        s = v - v.mean() if center else v.copy()
        g = np.array([v[tid == t].sum() for t in range(tid.max() + 1)])
        t_ = g - g.mean() if center else g.copy()
        step[n], traj[n] = s, t_
    return step, traj


def dist(a, b):
    return float(np.linalg.norm(a - b) / np.sqrt(len(a)))


def kl_estimates(step, traj, tbar):
    """Scaled squared distances, as the paper's bound relates them to KL."""
    es, et = {}, {}
    for i, j in itertools.combinations(NAMES, 2):
        es[(i, j)] = tbar * dist(step[i], step[j]) ** 2
        et[(i, j)] = dist(traj[i], traj[j]) ** 2
    return es, et


# ----------------------------------------------------- measured KL

def measured_kl():
    """K_ij over agent i's own visited histories, in closed form."""
    pol = {}
    for s in NAMES:
        for o in NAMES:
            pol[(s, o)] = [np.array(p) for p in
                           json.load(open(D("cross", f"{s}__{o}.json")))]
    lens = {o: [len(t["steps"]) for t in json.load(open(D("rollouts", f"{o}.json")))]
            for o in NAMES}
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


# ------------------------------------------------------- behaviour

def acts(name):
    return [[st["action"] for st in t["steps"]]
            for t in json.load(open(D("rollouts", f"{name}.json")))]


def _edit(a, b):
    n, m = len(a), len(b)
    d = list(range(m + 1))
    for i in range(1, n + 1):
        prev, d[0] = d[0], i
        for j in range(1, m + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1,
                                   prev + (a[i - 1] != b[j - 1]))
    return d[m] / max(n, m, 1)


def behaviour():
    A = {n: acts(n) for n in NAMES}
    out = {}
    for i, j in itertools.combinations(NAMES, 2):
        ed = np.mean([_edit(x, y) for x, y in zip(A[i], A[j])])
        jac = []
        for x, y in zip(A[i], A[j]):
            gx = {(x[k], x[k + 1]) for k in range(len(x) - 1)}
            gy = {(y[k], y[k + 1]) for k in range(len(y) - 1)}
            jac.append(1 - (len(gx & gy) / len(gx | gy) if gx | gy else 1.0))
        first = np.mean([x[0] != y[0] for x, y in zip(A[i], A[j])])
        def hist(seqs):
            c = np.zeros(len(VERBS))
            for s in seqs:
                for a in s:
                    c[VERBS.index(a.split()[0])] += 1
            return c / max(c.sum(), 1)
        p, q = hist(A[i]), hist(A[j])
        m = (p + q) / 2
        kl = lambda u, v: float((u * (np.log(u + 1e-300) - np.log(v + 1e-300))).sum())
        js = np.sqrt(max(0.5 * kl(p, m) + 0.5 * kl(q, m), 0))
        out[(i, j)] = dict(edit=float(ed), gram2=float(np.mean(jac)),
                           first=float(first), typejs=float(js))
    return out


# ---------------------------------------------------------- report

def table(title, rows, head):
    w = [max(len(str(r[k])) for r in [dict(zip(head, head))] + rows) for k in head]
    print(f"\n{title}")
    print("  " + "  ".join(h.rjust(x) for h, x in zip(head, w)))
    for r in rows:
        print("  " + "  ".join(str(r[h]).rjust(x) for h, x in zip(head, w)))


if __name__ == "__main__":
    K, lens = measured_kl()
    pairs = list(itertools.combinations(NAMES, 2))
    sym = np.array([K[(i, j)] + K[(j, i)] for i, j in pairs])
    tbar = float(np.mean(list(lens.values())))
    print(f"エージェント {len(NAMES)}体  ペア {len(pairs)}組  平均軌跡長 {tbar:.2f}")
    print("自己KL（0であるべき）: " + "  ".join(f"{n}={K[(n,n)]:.2e}" for n in NAMES))

    rows = []
    for key in ("raw", "norm"):
        for cen in (True, False):
            step, traj = load_coords(key, cen)
            es, et = kl_estimates(step, traj, tbar)
            for ver, est in (("step", es), ("traj", et)):
                e = np.array([est[p] for p in pairs])
                rows.append({
                    "座標": key, "中心化": "有" if cen else "無", "版": ver,
                    "Pearson": f"{pearsonr(e, sym)[0]:.3f}",
                    "Spearman": f"{spearmanr(e, sym)[0]:.3f}",
                    "水準比(中央値)": f"{np.median(e / sym):.1f}",
                })
    table("座標距離 対 実測の対称軌跡KL（15ペア）", rows,
          ["座標", "中心化", "版", "Pearson", "Spearman", "水準比(中央値)"])

    B = behaviour()
    step, traj = load_coords("raw", True)
    es, _ = kl_estimates(step, traj, tbar)
    coord = np.array([es[p] for p in pairs])
    dsize = np.array([abs(np.log(models.PARAMS[i]) - np.log(models.PARAMS[j]))
                      for i, j in pairs])
    lbar = {n: float(np.load(D("coords", f"{n}.npz"))["raw"].mean()) for n in NAMES}
    dlbar = np.array([abs(lbar[i] - lbar[j]) for i, j in pairs])
    rows = []
    for m in ("edit", "gram2", "first", "typejs"):
        b = np.array([B[p][m] for p in pairs])
        rows.append({"挙動の指標": m,
                     "座標距離": f"{spearmanr(coord, b)[0]:.3f}",
                     "実測KL": f"{spearmanr(sym, b)[0]:.3f}",
                     "サイズ差": f"{spearmanr(dsize, b)[0]:.3f}",
                     "平均対数尤度差": f"{spearmanr(dlbar, b)[0]:.3f}"})
    table("挙動の近さとの Spearman 相関（15ペア）", rows,
          ["挙動の指標", "座標距離", "実測KL", "サイズ差", "平均対数尤度差"])

    print("\n座標距離の近い順（上位6ペア、raw・中心化あり・step版）")
    for p in sorted(pairs, key=lambda p: es[p])[:6]:
        same = "同族" if models.FAMILY[p[0]] == models.FAMILY[p[1]] else "別族"
        print(f"  {p[0]:<12}{p[1]:<12} 推定{es[p]:8.3f}  実測KL{K[(p[0],p[1])]+K[(p[1],p[0])]:8.3f}  "
              f"編集距離{B[p]['edit']:.3f}  {same}")
    json.dump({"sym_kl": {f"{i}|{j}": K[(i,j)]+K[(j,i)] for i,j in pairs},
               "est_step": {f"{i}|{j}": es[(i,j)] for i,j in pairs},
               "behaviour": {f"{i}|{j}": B[(i,j)] for i,j in pairs},
               "tbar": tbar, "success": {n: sum(t["success"] for t in
                   json.load(open(D("rollouts", f"{n}.json")))) for n in NAMES}},
              open(D("results.json"), "w"), ensure_ascii=False, indent=1)
