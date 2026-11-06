# -*- coding: utf-8 -*-
"""Does a two-component harness change equal the sum of its parts?

The base point is h4 (window 4, base system prompt). One component moves the
history window, the other the system prompt. For each combined harness the
displacement from h4 is compared with the sum of the two single-component
displacements -- the paper's harness-composition table (Appendix C).

The paper's conventions, which the first version of this script got wrong in
two places:

  v  = z(A+B) - z(base)      the measured composed displacement
  v^ = dA + dB               the sum of the single-component displacements
  cos      = cos(v^, v)
  norm     = ||v|| / ||v^||  measured over predicted. The first version
                             computed the reciprocal, so an overshooting sum
                             was reported as 1.28 instead of 0.78.
  rel.err. = ||v^ - v|| / ||v||   The paper states the identity
                             ||e||^2/||v||^2 = 1 + r^-2 - 2c/r for its square,
                             and the figures it reports (0.46, 0.97) match the
                             unsquared ratio.
  pooling  = the median over cases, not one distance over the concatenated
             vectors. Both are printed here.

Coordinates are the centered step version. The per-component mean shift in
nats per step is computed on the uncentered coordinates, because centering
removes exactly the quantity that shift measures.

Run close.py coords_compose first.
"""
import json, numpy as np
import close, close_report as cr

BASE = "h4"


def vec(h, key, center):
    c = json.load(open(close.D("coords", f"{h}.json")))
    idx = c["idx"]
    if key == "raw":
        v = np.array([c["vecs"][k][idx[k]] for k in range(len(idx))])
    else:
        v = np.array([np.log(close.agent.normalise(c["vecs"][k])[idx[k]])
                      for k in range(len(idx))])
    return v - v.mean() if center else v


if __name__ == "__main__":
    for key in ("raw", "norm"):
        for center in (True, False):
            rows, dc_all, dp_all = [], [], []
            b = vec(BASE, key, center)
            for comb, A, B in close.TESTS:
                dc = vec(comb, key, center) - b
                dp = (vec(A, key, center) - b) + (vec(B, key, center) - b)
                cos = float(dc @ dp / (np.linalg.norm(dc) * np.linalg.norm(dp)))
                r = float(np.linalg.norm(dc) / np.linalg.norm(dp))
                re_ = float(np.linalg.norm(dp - dc) / np.linalg.norm(dc))
                rows.append({
                    "組合せ": f"{comb} = {A} + {B}",
                    "cos": f"{cos:.3f}", "ノルム比": f"{r:.3f}",
                    "相対誤差": f"{re_:.3f}",
                    "|実測Δ|": f"{np.linalg.norm(dc):.2f}"})
                dc_all.append((cos, r, re_, dc, dp))
            med = lambda i: float(np.median([x[i] for x in dc_all]))
            rows.append({"組合せ": "まとめて（4件の中央値・論文の方式）",
                         "cos": f"{med(0):.3f}", "ノルム比": f"{med(1):.3f}",
                         "相対誤差": f"{med(2):.3f}", "|実測Δ|": "—"})
            dc = np.concatenate([x[3] for x in dc_all])
            dp = np.concatenate([x[4] for x in dc_all])
            rows.append({
                "組合せ": "参考: 4件連結",
                "cos": f"{float(dc @ dp/(np.linalg.norm(dc)*np.linalg.norm(dp))):.3f}",
                "ノルム比": f"{np.linalg.norm(dc)/np.linalg.norm(dp):.3f}",
                "相対誤差": f"{np.linalg.norm(dp-dc)/np.linalg.norm(dc):.3f}",
                "|実測Δ|": f"{np.linalg.norm(dc):.2f}"})
            cr.table(f"ハーネスの変位の合成（{key}座標・中心化{'有' if center else '無'}）",
                     rows, ["組合せ", "cos", "ノルム比", "相対誤差", "|実測Δ|"])
    print("\n論文（2成分・全ケースの中央値）: "
          "τ-bench 0.90 / 0.98 / 0.46、ALFWorld 0.63 / 0.89 / 0.97")
    print("（cos が1、ノルム比が1、相対誤差が0なら完全に加法的。"
          "ノルム比が1未満なら和が実測を過大評価している）")

    # per-component mean shift, the paper's nats-per-step table.
    # Uncentered: centering removes exactly this quantity.
    print("\n成分ごとの平均座標シフト（raw・中心化なし・1ステップあたり nats、h4 基準）")
    b = vec(BASE, "raw", False)
    for h in ("h2", "h1", "h0", "n4", "s4", "n2", "s2", "n0", "s0"):
        d = vec(h, "raw", False) - b
        print(f"  {h:<4} {d.mean():+7.3f}")
    print("  論文（τ-bench・12モデルの中央値）: "
          "retrieve +9.6 / summarize +3.5 / replan -4.2 / think -6.2")
