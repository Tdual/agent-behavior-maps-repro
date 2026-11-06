# -*- coding: utf-8 -*-
"""Does the harness-composition result survive the alpha = 0 embedding?

The paper's composition table (Appendix C) lives in the linear space the
log-likelihood coordinate spans: a displacement is a difference of coordinate
vectors, and the claim is that two harness changes add. The alpha = 0
embedding is sqrt(pi), which lies on a product of unit spheres -- one per
reference step -- so there is no reason in advance for differences there to
add. This measures it.

Coordinate for agent i:

    z_i = M^(-1/2) * concat_t sqrt(pi_i(. | h_t))      over the admissible set

so ||z_i - z_j||^2 = mean_t ||sqrt(p) - sqrt(q)||^2 = mean_t 2(1 - BC), which
is the distance the map9/map14 alpha = 0 figures use. Centering is not needed
and cannot matter: v and v_hat are both differences, so any origin common to
the agents cancels.

No forward pass. Run close.py coords_compose first.
"""
import json, numpy as np
import close, close_compose as cc

BASE = "h4"


def z0(h):
    """The alpha = 0 coordinate: sqrt of the policy, concatenated over steps."""
    c = json.load(open(close.D("coords", f"{h}.json")))
    M = len(c["vecs"])
    return np.concatenate([np.sqrt(close.agent.normalise(v)) for v in c["vecs"]]) / np.sqrt(M)


def triple(dc, dp):
    cos = float(dc @ dp / (np.linalg.norm(dc) * np.linalg.norm(dp)))
    r = float(np.linalg.norm(dc) / np.linalg.norm(dp))
    re_ = float(np.linalg.norm(dp - dc) / np.linalg.norm(dc))
    return cos, r, re_


if __name__ == "__main__":
    print("ハーネスの変位の合成を &alpha;=0 で測る（前向き計算なし）\n")
    print(f"{'組合せ':<22}{'座標':>10}{'cos':>9}{'ノルム比':>10}{'相対誤差':>10}{'|実測Δ|':>10}")
    out = {}
    for key, get in (("α=1 norm", lambda h: cc.vec(h, "norm", True)),
                     ("α=1 raw", lambda h: cc.vec(h, "raw", True)),
                     ("α=0 √π", z0)):
        b, rows = get(BASE), []
        for comb, A, B in close.TESTS:
            dc = get(comb) - b
            dp = (get(A) - b) + (get(B) - b)
            cos, r, re_ = triple(dc, dp)
            rows.append((cos, r, re_))
            print(f"{comb+' = '+A+' + '+B:<22}{key:>10}{cos:>9.3f}{r:>10.3f}"
                  f"{re_:>10.3f}{np.linalg.norm(dc):>10.3f}")
        m = np.median(np.array(rows), axis=0)
        out[key] = m
        print(f"{'  中央値（論文の方式）':<22}{key:>10}{m[0]:>9.3f}{m[1]:>10.3f}{m[2]:>10.3f}{'—':>10}\n")

    print("まとめ（4件の中央値）")
    print(f"{'座標':<12}{'cos':>9}{'ノルム比':>10}{'相対誤差':>10}")
    for k, m in out.items():
        print(f"{k:<12}{m[0]:>9.3f}{m[1]:>10.3f}{m[2]:>10.3f}")
    print(f"{'論文 τ-bench':<12}{0.90:>9.2f}{0.98:>10.2f}{0.46:>10.2f}")
    print(f"{'論文 ALFWorld':<12}{0.63:>9.2f}{0.89:>10.2f}{0.97:>10.2f}")
