# -*- coding: utf-8 -*-
"""Does the coordinate distance track the trajectory KL when the pairs are close?

Reports the same quantities as report.py, over three groups of pairs:

  温度のみ      same harness, different temperature -- the `raw` coordinate is
                identical by construction, so distance is exactly 0
  ハーネスのみ  different harness, temperature 1.0 -- both conventions see it
  混在          different harness and different temperature

and, separately, over the subset whose measured epsilon is below a threshold,
which is the quantity the paper's bound is stated in.
"""
import itertools, json, os
import numpy as np
from scipy.stats import pearsonr, spearmanr
import close

D = close.D
NAMES, HA, TA = close.NAMES, close.HA, close.TA
PAIRS = list(itertools.combinations(NAMES, 2))
VERBS = ["go", "open", "close", "examine", "take", "put", "heat", "cool", "clean"]


def group(i, j):
    if HA[i] == HA[j]:
        return "温度のみ"
    if TA[i] == TA[j] == 1.0:
        return "ハーネスのみ"
    return "混在"


# -------------------------------------------------------- coordinates

def load_coords(key, center):
    C = {h: json.load(open(D("coords", f"{h}.json"))) for h in close.HARNESS}
    step, traj = {}, {}
    for n in NAMES:
        c = C[HA[n]]
        idx, tid = c["idx"], np.array(c["tid"])
        if key == "raw":
            v = np.array([c["vecs"][k][idx[k]] for k in range(len(idx))])
        else:
            v = np.array([np.log(close.agent.normalise(c["vecs"][k], TA[n])[idx[k]])
                          for k in range(len(idx))])
        s = v - v.mean() if center else v.copy()
        g = np.array([v[tid == t].sum() for t in range(tid.max() + 1)])
        step[n] = s
        traj[n] = g - g.mean() if center else g
    return step, traj


def dist(a, b):
    return float(np.linalg.norm(a - b) / np.sqrt(len(a)))


# ------------------------------------------------------- measured KL

def measured():
    """K[i][j], per-step symmetric KL, and epsilon, all over i's own histories."""
    cr = {(h, o): [np.array(v) for v in json.load(open(D("cross", f"{h}__{o}.json")))]
          for h in close.HARNESS for o in NAMES}
    lens = {o: [len(t["steps"]) for t in json.load(open(D("rollouts", f"{o}.json")))]
            for o in NAMES}
    pol = {(n, o): [close.agent.normalise(v, TA[n]) for v in cr[(HA[n], o)]]
           for n in NAMES for o in NAMES}
    K, eps, perstep = {}, {}, {}
    for i in NAMES:
        for j in NAMES:
            pi, pj = pol[(i, i)], pol[(j, i)]
            tot, k, e, ps = [], 0, 0.0, []
            for L in lens[i]:
                s = 0.0
                for _ in range(L):
                    p, q = pi[k], pj[k]
                    lp, lq = np.log(p + 1e-300), np.log(q + 1e-300)
                    s += float((p * (lp - lq)).sum())
                    e = max(e, float(np.abs(lp - lq).max()))
                    ps.append(float((p * (lp - lq)).sum()))
                    k += 1
                tot.append(s)
            K[(i, j)] = float(np.mean(tot))
            eps[(i, j)] = e
            perstep[(i, j)] = ps
    return K, eps, perstep, {o: float(np.mean(v)) for o, v in lens.items()}


# --------------------------------------------------------- behaviour

def acts(n):
    return [[s["action"] for s in t["steps"]]
            for t in json.load(open(D("rollouts", f"{n}.json")))]


def _edit(a, b):
    m = len(b)
    d = list(range(m + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, m + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1,
                                   prev + (a[i - 1] != b[j - 1]))
    return d[m] / max(len(a), m, 1)


def behaviour():
    A = {n: acts(n) for n in NAMES}
    out = {}
    for i, j in PAIRS:
        ed = np.mean([_edit(x, y) for x, y in zip(A[i], A[j])])
        jac = []
        for x, y in zip(A[i], A[j]):
            gx = {(x[k], x[k + 1]) for k in range(len(x) - 1)}
            gy = {(y[k], y[k + 1]) for k in range(len(y) - 1)}
            jac.append(1 - (len(gx & gy) / len(gx | gy) if gx | gy else 1.0))
        out[(i, j)] = dict(edit=float(ed), gram2=float(np.mean(jac)),
                           first=float(np.mean([x[0] != y[0] for x, y in zip(A[i], A[j])])))
    return out


def table(title, rows, head):
    w = [max(len(str(r[k])) for r in [dict(zip(head, head))] + rows) for k in head]
    print(f"\n{title}")
    print("  " + "  ".join(h.rjust(x) for h, x in zip(head, w)))
    for r in rows:
        print("  " + "  ".join(str(r[h]).rjust(x) for h, x in zip(head, w)))


def corr(est, sym, sub):
    e = np.array([est[p] for p in sub]); s = np.array([sym[p] for p in sub])
    if len(sub) < 3 or e.std() == 0:
        return "—", "—", "—", "—"
    pr, pp = pearsonr(e, s); sr, sp = spearmanr(e, s)
    return f"{pr:.3f}", f"{pp:.3f}", f"{sr:.3f}", f"{sp:.3f}"


if __name__ == "__main__":
    K, eps, perstep, lens = measured()
    sym = {p: K[p] + K[(p[1], p[0])] for p in PAIRS}
    symeps = {p: max(eps[p], eps[(p[1], p[0])]) for p in PAIRS}
    tbar = float(np.mean(list(lens.values())))
    print(f"エージェント {len(NAMES)}体（LLMは1つ、ハーネスだけが違う）  "
          f"ペア {len(PAIRS)}組  平均軌跡長 {tbar:.2f}")
    print("自己KL: " + "  ".join(f"{n}={K[(n,n)]:.1e}" for n in NAMES))

    allps = np.array([v for p in PAIRS for v in perstep[p]])
    print(f"\n実測ε（1ステップの対数比の最大値）  中央値 {np.median(list(symeps.values())):.3f}  "
          f"95%点 {np.percentile(list(symeps.values()),95):.3f}  "
          f"最大 {max(symeps.values()):.3f}")
    print(f"1ステップあたりの非対称KL  中央値 {np.median(allps):.4f}  "
          f"95%点 {np.percentile(allps,95):.4f}")
    print(f"対称軌跡KL  最小 {min(sym.values()):.4f}  "
          f"中央値 {np.median(list(sym.values())):.4f}  最大 {max(sym.values()):.4f}")

    # ---- main table, by group
    rows = []
    for key in ("raw", "norm"):
        for cen in (True, False):
            step, traj = load_coords(key, cen)
            for ver in ("step", "traj"):
                V = step if ver == "step" else traj
                est = {p: (tbar * dist(V[p[0]], V[p[1]]) ** 2 if ver == "step"
                           else dist(V[p[0]], V[p[1]]) ** 2) for p in PAIRS}
                for gname, sub in [("全体", PAIRS)] + [
                        (g, [p for p in PAIRS if group(*p) == g])
                        for g in ("ハーネスのみ", "混在", "温度のみ")]:
                    pr, pp, sr, sp = corr(est, sym, sub)
                    lv = np.array([est[p] for p in sub]) / np.array([sym[p] for p in sub])
                    rows.append({"座標": key, "中心化": "有" if cen else "無",
                                 "版": ver, "ペア群": gname, "n": len(sub),
                                 "Pearson": pr, "p": pp, "Spearman": sr,
                                 "p ": sp, "水準比": f"{np.median(lv):.2f}"})
    table("座標距離 対 実測の対称軌跡KL", rows,
          ["座標", "中心化", "版", "ペア群", "n", "Pearson", "p", "Spearman", "p ", "水準比"])

    # ---- epsilon thresholds, the quantity the bound is stated in
    rows = []
    for key in ("raw", "norm"):
        step, traj = load_coords(key, True)
        est = {p: tbar * dist(step[p[0]], step[p[1]]) ** 2 for p in PAIRS}
        for th in (0.25, 0.5, 1.0, 2.0, 1e9):
            sub = [p for p in PAIRS if symeps[p] <= th and est[p] > 0]
            pr, pp, sr, sp = corr(est, sym, sub)
            rows.append({"座標": key, "ε上限": ("なし" if th > 1e8 else f"{th}"),
                         "n": len(sub), "Pearson": pr, "p": pp,
                         "Spearman": sr, "p ": sp})
    table("ε で絞ったときの相関（step版・中心化あり・座標距離>0のペアのみ）", rows,
          ["座標", "ε上限", "n", "Pearson", "p", "Spearman", "p "])

    # ---- the raw blind spot
    step, _ = load_coords("raw", True)
    stepn, _ = load_coords("norm", True)
    rows = []
    for p in sorted([p for p in PAIRS if group(*p) == "温度のみ"],
                    key=lambda p: sym[p]):
        rows.append({"ペア": f"{p[0]} / {p[1]}",
                     "raw距離": f"{dist(step[p[0]], step[p[1]]):.2e}",
                     "norm距離": f"{dist(stepn[p[0]], stepn[p[1]]):.4f}",
                     "実測対称KL": f"{sym[p]:.4f}", "ε": f"{symeps[p]:.3f}"})
    table("温度だけが違うペア（raw 座標は原理的に区別できない）", rows,
          ["ペア", "raw距離", "norm距離", "実測対称KL", "ε"])

    # ---- behaviour
    B = behaviour()
    stepn, _ = load_coords("norm", True)
    estn = {p: tbar * dist(stepn[p[0]], stepn[p[1]]) ** 2 for p in PAIRS}
    rows = []
    for m in ("edit", "gram2", "first"):
        b = np.array([B[p][m] for p in PAIRS])
        r1 = spearmanr([estn[p] for p in PAIRS], b)
        r2 = spearmanr([sym[p] for p in PAIRS], b)
        rows.append({"挙動の指標": m,
                     "norm座標距離": f"{r1[0]:.3f}", "p": f"{r1[1]:.3f}",
                     "実測KL": f"{r2[0]:.3f}", "p ": f"{r2[1]:.3f}"})
    table("挙動の近さとの Spearman 相関（36ペア）", rows,
          ["挙動の指標", "norm座標距離", "p", "実測KL", "p "])

    print("\n実測KL の近い順（上位8ペア、norm・step版）")
    for p in sorted(PAIRS, key=lambda p: sym[p])[:8]:
        print(f"  {p[0]:<9}{p[1]:<9} 実測{sym[p]:7.4f}  推定{estn[p]:7.4f}  "
              f"ε{symeps[p]:6.3f}  {group(*p)}")

    json.dump({"sym_kl": {f"{i}|{j}": sym[(i, j)] for i, j in PAIRS},
               "eps": {f"{i}|{j}": symeps[(i, j)] for i, j in PAIRS},
               "est_norm_step": {f"{i}|{j}": estn[(i, j)] for i, j in PAIRS},
               "group": {f"{i}|{j}": group(i, j) for i, j in PAIRS},
               "behaviour": {f"{i}|{j}": B[(i, j)] for i, j in PAIRS},
               "tbar": tbar,
               "success": {n: sum(t["success"] for t in
                                  json.load(open(D("rollouts", f"{n}.json"))))
                           for n in NAMES}},
              open(D("results.json"), "w"), ensure_ascii=False, indent=1)
