# -*- coding: utf-8 -*-
"""Is the alpha = 0 coordinate a vector space? Measure it instead of assuming.

The alpha = 1 coordinate lives in a linear space: it is a log-likelihood, and
differences of it add. The alpha = 0 coordinate does not. Each reference step
contributes sqrt(pi), which satisfies ||sqrt(pi)|| = 1, so every agent sits on
a product of unit spheres and a difference of two agents is a chord, not a
tangent vector. Chords on a curved surface do not add.

Two things are measured here.

  1. the geometry itself: the norm of every coordinate, and the per-step angle
     theta_r = arccos(BC_r) between the base agent and each other agent, which
     is how far the displacement reaches around the sphere

  2. composition, three ways
       chord     v = z(A+B) - z(base),  v_hat = (z(A) - z(base)) + (z(B) - z(base))
       tangent   the same differences taken through the Fisher-Rao log map at
                 the base point, blockwise (the product manifold's own log map)
       alpha = 1 the paper's coordinate, for reference

No forward pass: every policy vector is already stored.
"""
import json, numpy as np
import close, compose2 as c2

BASE, EPS = "h4", 1e-300


def pol(h, T=1.0):
    """The policy vectors over the admissible set, one per reference step."""
    c = json.load(open(close.D("coords", f"{h}.json")))
    return [np.asarray(close.agent.normalise(v, T), dtype=float) for v in c["vecs"]]


def z_chord(h, T=1.0):
    P = pol(h, T)
    return np.concatenate([np.sqrt(p) for p in P]) / np.sqrt(len(P))


def z_tangent(h, T, base):
    """Fisher-Rao log map at `base`, applied per reference step."""
    P, B, out = pol(h, T), base, []
    for p, b in zip(P, B):
        sp, sb = np.sqrt(p), np.sqrt(b)
        bc = float(np.clip(sp @ sb, -1.0, 1.0))
        th = np.arccos(bc)
        s = np.sin(th)
        out.append((sp - bc * sb) * (th / s if s > 1e-12 else 1.0))
    return np.concatenate(out) / np.sqrt(len(P))


def triple(dc, dp):
    cos = float(dc @ dp / (np.linalg.norm(dc) * np.linalg.norm(dp)))
    return cos, float(np.linalg.norm(dc) / np.linalg.norm(dp)), \
        float(np.linalg.norm(dp - dc) / np.linalg.norm(dc))


if __name__ == "__main__":
    basepol = pol(BASE, 1.0)
    M = len(basepol)
    print(f"参照ステップ M = {M}\n")

    print("座標のノルム（√π を繋いで √M で割ったもの）")
    agents = sorted({(h, t) for _, hc, parts in
                     [(k, hc, p) for k, hc, p in c2.CASES] for h, t in [hc] + parts})
    for h, t in agents[:4]:
        print(f"  {h}@T{t}  ‖z‖ = {np.linalg.norm(z_chord(h, t)):.12f}")
    print(f"  （全 {len(agents)} 体の ‖z‖ の最大のずれ: "
          f"{max(abs(np.linalg.norm(z_chord(h, t)) - 1) for h, t in agents):.2e}）\n")

    print("基準 h4 から見た1ステップあたりの角度 θ_r = arccos(BC_r)（ラジアン）")
    print(f"{'エージェント':<14}{'中央値':>9}{'最大':>9}{'弦/弧':>9}")
    for h, t in agents:
        th = []
        for p, b in zip(pol(h, t), basepol):
            th.append(np.arccos(np.clip(np.sqrt(p) @ np.sqrt(b), -1, 1)))
        th = np.array(th)
        ch = np.linalg.norm(z_chord(h, t) - z_chord(BASE, 1.0))
        tg = np.linalg.norm(z_tangent(h, t, basepol))
        print(f"{h+'@T'+str(t):<14}{np.median(th):>9.3f}{th.max():>9.3f}"
              f"{ch / tg if tg > 0 else float('nan'):>9.4f}")

    print("\n合成（基準 h4、CASES は compose2.py と同じ 16 件）")
    print(f"{'座標':<16}{'種類':<14}{'件数':>5}{'cos':>9}{'ノルム比':>10}{'相対誤差':>10}")
    out = {}
    for label, get in (
            ("α=1 norm", lambda h, t: c2.vec_norm(h, t)),
            ("α=0 弦", lambda h, t: z_chord(h, t)),
            ("α=0 接空間", lambda h, t: z_tangent(h, t, basepol))):
        b, rows = get(BASE, 1.0), {}
        for kind, (ch, ct), parts in c2.CASES:
            dc = get(ch, ct) - b
            dp = sum((get(h, t) - b) for h, t in parts)
            rows.setdefault(kind, []).append(triple(dc, dp))
        for k, v in rows.items():
            m = np.median(np.array(v), axis=0)
            out[(label, k)] = [float(x) for x in m]
            print(f"{label:<16}{k:<14}{len(v):>5}{m[0]:>9.3f}{m[1]:>10.3f}{m[2]:>10.3f}")
    json.dump({f"{a}|{b}": v for (a, b), v in out.items()},
              open("tangent.json", "w"), ensure_ascii=False, indent=1)
