# -*- coding: utf-8 -*-
"""All 14 agents at alpha = 0, and the LLM-versus-harness question settled.

The 2D maps raised a disagreement: in the paper's alpha = 1 coordinate the nine
harness variants of one LLM sit farther apart than the six separate LLMs, while
the measured on-policy KL said the opposite. Those two numbers came from
different rollouts (independent random numbers over 10 steps versus shared ones
over 8), so they were not comparable.

With the full policy vector for all 14 agents the question needs no rollout at
all. On the shared 193 reference steps the per-step KL is a closed-form sum, so
the same quantity -- the KL between two policies, averaged over one fixed set
of histories -- can be put next to each coordinate distance. Everything below
is measured on that one set of states.
"""
import itertools, os
import numpy as np
from scipy.stats import pearsonr, spearmanr
import alpha14, close, models, viz

EPS = 1e-300
LL = [n for n, _ in models.MODELS]
HH = [n for n in close.NAMES if n != "h4"]
NAMES = LL + HH


def kl_step(P, i, j):
    """Mean symmetric per-step KL over the reference states."""
    out = []
    for p, q in zip(P[i], P[j]):
        p = np.asarray(p); q = np.asarray(q)
        out.append(float((p * np.log((p + EPS) / (q + EPS))).sum()
                         + (q * np.log((q + EPS) / (p + EPS))).sum()))
    return float(np.mean(out))


if __name__ == "__main__":
    P = alpha14.policies()
    M = len(P[NAMES[0]])
    pairs = list(itertools.combinations(NAMES, 2))
    print(f"参照ステップ {M}　エージェント {len(NAMES)}体　ペア {len(pairs)}組\n")

    # ---- the three distance conventions, all on the same states
    chosen = {}
    for n in NAMES:
        c = alpha14.chosen_index(n)
        chosen[n] = np.array([np.log(P[n][k][c[k]] + EPS) for k in range(M)])
    rawv = alpha14.raw_vectors()

    D0 = viz.dmat_hellinger(P, NAMES)
    D1 = viz.dmat({n: viz.centre(chosen[n]) for n in NAMES}, NAMES)
    DR = viz.dmat({n: viz.centre(rawv[n]) for n in NAMES}, NAMES)
    K = {(a, b): kl_step(P, a, b) for a, b in pairs}

    iL = [NAMES.index(n) for n in LL]
    iH = [NAMES.index(n) for n in HH] + [NAMES.index("qwen05-it")]
    gl = list(itertools.combinations(iL, 2))
    gh = list(itertools.combinations(iH, 2))
    kv = lambda idx: [K[(NAMES[a], NAMES[b])] if (NAMES[a], NAMES[b]) in K
                      else K[(NAMES[b], NAMES[a])] for a, b in idx]

    rows = [("座標距離 &alpha;=1 raw", DR), ("座標距離 &alpha;=1 norm", D1),
            ("座標距離 &alpha;=0 &radic;", D0)]
    print(f"{'量':<26}{'LLM 6体(15組)':>16}{'ハーネス9体(36組)':>20}{'LLM/ハーネス':>14}")
    for lab, Dm in rows:
        a = np.median([Dm[i, j] for i, j in gl])
        b = np.median([Dm[i, j] for i, j in gh])
        print(f"{lab:<26}{a:>16.3f}{b:>20.3f}{a/b:>14.2f}倍")
    a, b = np.median(kv(gl)), np.median(kv(gh))
    print(f"{'1ステップ対称KL（同一状態）':<22}{a:>16.3f}{b:>20.3f}{a/b:>14.2f}倍")

    # ---- which convention orders the 91 pairs correctly
    kall = np.array(kv(list(itertools.combinations(range(len(NAMES)), 2))))
    print("\n91ペア全体で、実測KL（同一状態・閉形式）をどれだけ順序づけられるか")
    for lab, Dm in rows:
        d = np.array([Dm[i, j] for i, j in
                      itertools.combinations(range(len(NAMES)), 2)])
        pr = pearsonr(d ** 2, kall)[0]
        sp = spearmanr(d, kall)[0]
        print(f"  {lab:<26} 距離^2 vs KL  Pearson {pr:.3f}　順位 Spearman {sp:.3f}")

    # ---- the 14-agent alpha=0 map
    X, share, sp, pr = viz.mds(D0)
    col = [viz.FAM[models.FAMILY[n]] for n in LL] + [viz.GRP[viz.grp(n)] for n in HH]
    p = viz.svg(X, NAMES, col,
                "14エージェントの地図（&alpha;=0 &radic;&pi;・Hellinger）",
                f"2次元が保つ配置の割合 {share*100:.1f}%　"
                f"距離の再現 Pearson {pr:.3f} / Spearman {sp:.3f}",
                "map14_a0.svg", W=720, H=480,
                legend=[("SmolLM2", viz.FAM["SmolLM2"]), ("Qwen2.5", viz.FAM["Qwen2.5"]),
                        ("Pythia", viz.FAM["Pythia"]), ("窓", viz.GRP["窓"]),
                        ("プロンプト", viz.GRP["プロンプト"]), ("温度", viz.GRP["温度"])],
                D=D0)
    print(f"\n14体・α=0: 2次元の割合 {share*100:.1f}%  r={pr:.3f} ρ={sp:.3f}  -> {p}")
