# -*- coding: utf-8 -*-
"""The paper's layout: t-SNE on the centered coordinates.

Figure 1(a) of the paper is "a t-SNE layout of the centered coordinates with
perplexity 15" over 100 LLMs. Perplexity must stay below the number of points,
so 15 is unreachable at 14 agents; the sweep below runs every perplexity that
is available (2 to 12) and ten seeds each, and reports what changes.

Two things are measured rather than asserted:

  seed stability   Procrustes disparity between the layouts of two seeds at the
                   same perplexity (0 = identical up to rotation/scale/reflection)
  distance fidelity Spearman correlation between the planar distances and the
                   original coordinate distances (t-SNE does not optimise this;
                   it is reported because the question is whether the picture
                   can be read as a map)

Classical MDS is run on the same matrix as the reference point.
"""
import itertools, os
import numpy as np
from scipy.spatial import procrustes
from scipy.stats import spearmanr
from sklearn.manifold import TSNE
import alpha14, close, models, viz

LL = [n for n, _ in models.MODELS]
HH = [n for n in close.NAMES if n != "h4"]


def coords_raw(names):
    """The paper's coordinate: centered action log-likelihood, scaled M^(-1/2)."""
    raw = alpha14.raw_vectors()
    M = len(raw[names[0]])
    return np.array([(raw[n] - raw[n].mean()) / np.sqrt(M) for n in names])


def run(Z, perp, seed, init="pca"):
    return TSNE(n_components=2, perplexity=perp, init=init,
                random_state=seed, max_iter=2000).fit_transform(Z)


def fidelity(Z, X):
    n = len(Z)
    pr = list(itertools.combinations(range(n), 2))
    d0 = np.array([np.linalg.norm(Z[a] - Z[b]) for a, b in pr])
    d1 = np.array([np.linalg.norm(X[a] - X[b]) for a, b in pr])
    return float(spearmanr(d0, d1)[0])


def dmat(Z, names):
    n = len(names)
    D = np.zeros((n, n))
    for a, b in itertools.combinations(range(n), 2):
        D[a, b] = D[b, a] = np.linalg.norm(Z[a] - Z[b])
    return D


def sweep(Z, names, tag, title, colors, legend):
    print(f"\n=== {tag}: {len(names)}体 ===")
    print(f"{'perplexity':>11}{'種ごとの Procrustes 差':>24}{'距離の順位 Spearman':>22}")
    best = None
    for perp in range(2, len(names)):
        Xs = [run(Z, perp, s) for s in range(10)]
        dis = [procrustes(Xs[a], Xs[b])[2]
               for a, b in itertools.combinations(range(10), 2)]
        fid = [fidelity(Z, X) for X in Xs]
        print(f"{perp:>11}{np.median(dis):>24.3f}{np.median(fid):>22.3f}"
              f"   （差 {min(dis):.3f}–{max(dis):.3f}　ρ {min(fid):.3f}–{max(fid):.3f}）")
        if best is None or np.median(fid) > best[2]:
            best = (perp, Xs[0], float(np.median(fid)), float(np.median(dis)))
    perp, X, fid, dis = best
    print(f"  採用: perplexity {perp}（距離の順位 Spearman が最大）")
    p = viz.svg(X, names, colors,
                f"{title}（t-SNE・perplexity {perp}・seed 0）",
                f"距離の順位 Spearman {fid:.3f}（10種の中央値）　"
                f"種どうしの Procrustes 差 中央値 {dis:.3f}",
                f"tsne_{tag}.svg", W=720, H=480, legend=legend,
                D=dmat(Z, names))
    print(f"  -> {p}")
    return perp, X


if __name__ == "__main__":
    names14 = LL + HH
    Z14 = coords_raw(names14)
    col14 = ([viz.FAM[models.FAMILY[n]] for n in LL]
             + [viz.GRP[viz.grp(n)] for n in HH])
    leg14 = [("SmolLM2", viz.FAM["SmolLM2"]), ("Qwen2.5", viz.FAM["Qwen2.5"]),
             ("Pythia", viz.FAM["Pythia"]), ("窓", viz.GRP["窓"]),
             ("プロンプト", viz.GRP["プロンプト"]), ("温度", viz.GRP["温度"])]
    sweep(Z14, names14, "14", "14エージェントの地図（&alpha;=1 raw・中心化）", col14, leg14)

    # the paper's own setting: one harness, the LLM varied
    Z6 = coords_raw(LL)
    col6 = [viz.FAM[models.FAMILY[n]] for n in LL]
    leg6 = [("SmolLM2", viz.FAM["SmolLM2"]), ("Qwen2.5", viz.FAM["Qwen2.5"]),
            ("Pythia", viz.FAM["Pythia"])]
    sweep(Z6, LL, "6", "6つの LLM・BASE ハーネス（論文 図1(a) と同じ設定）", col6, leg6)

    # the seeds only matter when t-SNE is not initialised from the PCA
    print("\n=== init=\"random\" にすると種で絵が変わる（14体） ===")
    print(f"{'perplexity':>11}{'Procrustes 差 中央値':>22}{'最大':>8}{'ρ 中央値':>10}")
    for perp in (3, 6, 9, 12):
        Xs = [run(Z14, perp, s, init="random") for s in range(10)]
        dis = [procrustes(a, b)[2] for a, b in itertools.combinations(Xs, 2)]
        fid = [fidelity(Z14, X) for X in Xs]
        print(f"{perp:>11}{np.median(dis):>22.3f}{max(dis):>8.3f}{np.median(fid):>10.3f}")

    # classical MDS on the same matrices, for comparison
    print("\n=== 比較: 同じ座標を古典的MDS で落とす ===")
    for tag, Z, names in (("14", Z14, names14), ("6", Z6, LL)):
        D = viz.dmat({n: Z[i] * np.sqrt(len(Z[i])) for i, n in enumerate(names)}, names)
        X, share, sp, pr = viz.mds(D)
        print(f"  {tag}体: 2次元が保つ割合 {share*100:.1f}%　"
              f"距離の順位 Spearman {sp:.3f}（決定論的、種による揺れなし）")
