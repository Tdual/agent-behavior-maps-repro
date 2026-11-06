# -*- coding: utf-8 -*-
"""Two-dimensional maps of the agents, by classical MDS.

The paper lays its 100 models out with t-SNE. t-SNE needs a perplexity below
the number of points and is stochastic, so at 9-15 agents it says nothing.
Classical MDS (principal coordinates) is the right tool at this size: it is
deterministic, it minimises the distortion of the given distances in two
dimensions, and it reports how much of the configuration those two dimensions
actually carry.

Both runs scored the same 193 reference steps, so the six separate LLMs and
the nine harness variants of one LLM go on a single map. Agent `h4` and agent
`qwen05-it` are by construction the same agent -- same weights, window 4, same
system prompt, temperature 1 -- which is checked here.

Output: viz/*.svg, self-contained, no plotting library.
"""
import itertools, json, os
import numpy as np
from scipy.stats import spearmanr
import close, models

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "viz")

FAM = {"SmolLM2": "#1f6fb4", "Qwen2.5": "#b4531f", "Pythia": "#2e7d32"}
GRP = {"温度": "#840791", "窓": "#1f6fb4", "プロンプト": "#c08000", "複合": "#2e7d32"}


# ------------------------------------------------------------ coordinates

def llm_coords(key):
    """The six separate LLMs, BASE harness. One value per reference step."""
    out = {}
    for n, _ in models.MODELS:
        z = np.load(os.path.join(HERE, "coords", f"{n}.npz"))
        out[n] = z[key].astype(float)
    return out


def harness_vecs():
    """Full policy vector at every reference step, per harness agent."""
    C = {h: json.load(open(close.D("coords", f"{h}.json"))) for h in close.HARNESS}
    pol, chosen = {}, {}
    for n in close.NAMES:
        c = C[close.HA[n]]
        p = [close.agent.normalise(v, close.TA[n]) for v in c["vecs"]]
        pol[n] = p
        chosen[n] = np.array([np.log(p[k][c["idx"][k]]) for k in range(len(p))])
    raw = {n: np.array([C[close.HA[n]]["vecs"][k][C[close.HA[n]]["idx"][k]]
                        for k in range(len(C[close.HA[n]]["idx"]))])
           for n in close.NAMES}
    return pol, chosen, raw


def centre(v):
    return v - v.mean()


def dmat(vecs, names):
    M = len(vecs[names[0]])
    D = np.zeros((len(names), len(names)))
    for a, b in itertools.combinations(range(len(names)), 2):
        d = np.linalg.norm(vecs[names[a]] - vecs[names[b]]) / np.sqrt(M)
        D[a, b] = D[b, a] = d
    return D


def dmat_hellinger(pol, names):
    M = len(pol[names[0]])
    D = np.zeros((len(names), len(names)))
    for a, b in itertools.combinations(range(len(names)), 2):
        bc = np.array([float(np.sqrt(x * y).sum())
                       for x, y in zip(pol[names[a]], pol[names[b]])])
        v = float(np.mean(2.0 * (1.0 - np.clip(bc, 0, 1))))
        D[a, b] = D[b, a] = np.sqrt(v)
    return D


# -------------------------------------------------------------------- MDS

def mds(D):
    n = len(D)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)
    o = np.argsort(w)[::-1]
    w, V = w[o], V[:, o]
    pos = np.clip(w, 0, None)
    share = pos[:2].sum() / pos.sum() if pos.sum() > 0 else 0.0
    X = V[:, :2] * np.sqrt(pos[:2])
    d2 = np.array([np.linalg.norm(X[a] - X[b])
                   for a, b in itertools.combinations(range(n), 2)])
    d1 = np.array([D[a, b] for a, b in itertools.combinations(range(n), 2)])
    return X, float(share), float(spearmanr(d1, d2)[0]), float(np.corrcoef(d1, d2)[0, 1])


# -------------------------------------------------------------------- SVG

def merge(D, labels, colors, tol=1e-9):
    """Agents at distance zero are the same point. Draw them once.

    The alpha = 1 raw coordinate cannot see temperature at all, so several
    agents land on one point exactly. Overplotting them hides the fact.
    """
    n = len(labels)
    seen, groups = set(), []
    for a in range(n):
        if a in seen:
            continue
        g = [a] + [b for b in range(a + 1, n) if D[a, b] <= tol]
        seen.update(g)
        groups.append(g)
    keep = [g[0] for g in groups]
    lab = []
    for g in groups:
        if len(g) == 1:
            lab.append(labels[g[0]])
        else:
            lab.append(" ＝ ".join(labels[k] for k in g) + f"（{len(g)}体が同一点）")
    return keep, lab, [colors[g[0]] for g in groups]


def place(px, py, labels, W, H):
    """Greedy label placement: try offsets until the box is free."""
    boxes, out = [], []
    CH, LH = 6.0, 13.0                       # character width, line height
    cand = [(10, 4, "start"), (-10, 4, "end"), (10, -9, "start"), (-10, -9, "end"),
            (10, 16, "start"), (-10, 16, "end"), (0, -12, "middle"), (0, 20, "middle"),
            (10, 28, "start"), (-10, 28, "end"), (0, -24, "middle"), (0, 32, "middle")]
    order = sorted(range(len(labels)), key=lambda i: -len(labels[i]))
    res = [None] * len(labels)
    for i in order:
        w = len(labels[i]) * CH
        best = None
        for dx, dy, anc in cand:
            x = px[i] + dx
            x0 = x if anc == "start" else (x - w if anc == "end" else x - w / 2)
            box = (x0, py[i] + dy - LH + 3, x0 + w, py[i] + dy + 3)
            if box[0] < 2 or box[2] > W - 2 or box[1] < 46 or box[3] > H - 4:
                continue
            hit = any(not (box[2] < b[0] or box[0] > b[2] or
                           box[3] < b[1] or box[1] > b[3]) for b in boxes)
            hit = hit or any(box[0] - 4 < px[k] < box[2] + 4 and
                             box[1] - 4 < py[k] < box[3] + 4 for k in range(len(labels)))
            if not hit:
                best = (dx, dy, anc, box)
                break
        if best is None:
            dx, dy, anc = cand[0]
            x0 = px[i] + dx
            best = (dx, dy, anc, (x0, py[i] + dy - LH + 3, x0 + w, py[i] + dy + 3))
        boxes.append(best[3])
        res[i] = best[:3]
    return res


ENT = {"&alpha;": "\u03b1", "&radic;": "\u221a", "&pi;": "\u03c0",
       "&beta;": "\u03b2", "&times;": "\u00d7"}


def svg(X, labels, colors, title, sub, path, W=700, H=460, pad=86, legend=None, D=None):
    """SVG is XML: HTML entity names are not defined there, so they are
    substituted for the characters themselves before writing."""
    for k, v in ENT.items():
        title, sub = title.replace(k, v), sub.replace(k, v)
    if D is not None:
        keep, labels, colors = merge(D, labels, colors)
        X = X[keep]
    x, y = X[:, 0].copy(), X[:, 1].copy()
    s = min((W - 2 * pad) / max(x.max() - x.min(), 1e-9),
            (H - 2 * pad) / max(y.max() - y.min(), 1e-9))
    cx, cy = (x.max() + x.min()) / 2, (y.max() + y.min()) / 2
    px = W / 2 + (x - cx) * s
    py = (H + 36) / 2 - (y - cy) * s
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" '
         f'style="max-width:{W}px;display:block;margin:0 auto;'
         f'font-family:Hiragino Sans,Noto Sans CJK JP,sans-serif">',
         f'<rect width="{W}" height="{H}" fill="#fff"/>',
         f'<text x="{W/2}" y="22" font-size="13.5" font-weight="bold" fill="#16213e" '
         f'text-anchor="middle">{title}</text>',
         f'<text x="{W/2}" y="39" font-size="10" fill="#666" '
         f'text-anchor="middle">{sub}</text>']
    for i, lab in enumerate(labels):
        o.append(f'<circle cx="{px[i]:.1f}" cy="{py[i]:.1f}" r="5.5" fill="{colors[i]}" '
                 f'fill-opacity="0.85" stroke="#fff" stroke-width="1.2"/>')
    for i, (dx, dy, anc) in enumerate(place(px, py, labels, W, H)):
        o.append(f'<text x="{px[i]+dx:.1f}" y="{py[i]+dy:.1f}" font-size="10.5" '
                 f'fill="#333" text-anchor="{anc}">{labels[i]}</text>')
    if legend:
        lx = 12
        for name, c in legend:
            o.append(f'<circle cx="{lx+5}" cy="{H-12}" r="4.5" fill="{c}"/>')
            o.append(f'<text x="{lx+14}" y="{H-8}" font-size="10" fill="#555">{name}</text>')
            lx += 16 + len(name) * 11
    o.append("</svg>")
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, path), "w", encoding="utf-8").write("\n".join(o))
    return os.path.join(OUT, path)


def grp(n):
    if n.startswith("h4-T"):
        return "温度"
    if n in ("h4", "h2", "h1", "h0"):
        return "窓"
    if n in ("n4", "s4"):
        return "プロンプト"
    return "複合"


if __name__ == "__main__":
    pol, chosen, hraw = harness_vecs()
    lraw, lnorm = llm_coords("raw"), llm_coords("norm")

    # --- consistency: h4 and qwen05-it must be the same agent
    d = np.abs(hraw["h4"] - lraw["qwen05-it"])
    print(f"整合性: h4 と qwen05-it は同一のエージェント（同じ重み・窓4・同じプロンプト・温度1）")
    print(f"  raw 座標の最大差 {d.max():.3e}   平均差 {d.mean():.3e}")
    assert d.max() < 1e-6, "別物になっている"

    # --- combined map, 14 distinct agents (h4 == qwen05-it, kept once)
    LL = [n for n, _ in models.MODELS]
    HH = [n for n in close.NAMES if n != "h4"]
    names = LL + HH
    for key, vr, vh, tag in (("raw", lraw, hraw, "raw"), ("norm", lnorm, chosen, "norm")):
        V = {n: centre(vr[n]) for n in LL}
        V.update({n: centre(vh[n]) for n in HH})
        D = dmat(V, names)
        X, share, sp, pr = mds(D)
        col = [FAM[models.FAMILY[n]] for n in LL] + [GRP[grp(n)] for n in HH]
        lab = [n for n in LL] + [n for n in HH]
        p = svg(X, lab, col,
                f"14エージェントの地図（&alpha;=1 {tag}・中心化・古典的MDS）",
                f"2次元が保つ配置の割合 {share*100:.1f}%　"
                f"距離の再現 Pearson {pr:.3f} / Spearman {sp:.3f}",
                f"map14_{tag}.svg", W=720, H=480,
                legend=[("SmolLM2", FAM["SmolLM2"]), ("Qwen2.5", FAM["Qwen2.5"]),
                        ("Pythia", FAM["Pythia"]), ("窓", GRP["窓"]),
                        ("プロンプト", GRP["プロンプト"]), ("温度", GRP["温度"])], D=D)
        print(f"\n14体・α=1 {tag}: 2次元の割合 {share*100:.1f}%  "
              f"距離の再現 r={pr:.3f} ρ={sp:.3f}  -> {p}")
        # how tight is the harness cluster against the LLM spread
        iL = [names.index(n) for n in LL]
        iH = [names.index(n) for n in HH] + [names.index("qwen05-it")]
        dl = [D[a, b] for a, b in itertools.combinations(iL, 2)]
        dh = [D[a, b] for a, b in itertools.combinations(iH, 2)]
        print(f"  LLM同士の距離 中央値 {np.median(dl):.3f}　"
              f"ハーネス違い同士 中央値 {np.median(dh):.3f}　"
              f"比 {np.median(dl)/np.median(dh):.1f}倍")

    # --- 9 harness agents: alpha=1 norm vs alpha=0
    for tag, D, title in (
            ("a1", dmat({n: centre(chosen[n]) for n in close.NAMES}, close.NAMES),
             "9エージェント（&alpha;=1 log・許容集合で正規化・中心化）"),
            ("a0", dmat_hellinger(pol, close.NAMES),
             "9エージェント（&alpha;=0 &radic;・Hellinger）")):
        X, share, sp, pr = mds(D)
        col = [GRP[grp(n)] for n in close.NAMES]
        p = svg(X, close.NAMES, col, title,
                f"2次元が保つ配置の割合 {share*100:.1f}%　"
                f"距離の再現 Pearson {pr:.3f} / Spearman {sp:.3f}",
                f"map9_{tag}.svg", W=660, H=440,
                legend=[("窓", GRP["窓"]), ("プロンプト", GRP["プロンプト"]),
                        ("温度", GRP["温度"])], D=D)
        print(f"\n9体・{tag}: 2次元の割合 {share*100:.1f}%  "
              f"距離の再現 r={pr:.3f} ρ={sp:.3f}  -> {p}")
