# -*- coding: utf-8 -*-
"""Render the article's display formulas from TeX to inline SVG.

matplotlib's mathtext is the engine: no TeX installation, no network at view
time, and the glyphs come out as paths, so the page needs no maths font.

Two things are done to the SVG matplotlib writes, so it can live inside the
article next to the hand-authored figures of `figs.py`:

  * every id is prefixed with the formula's name -- several of these fragments
    share one document, and `<use href="#id">` resolves to the first id in
    document order, so unprefixed glyph ids would silently draw the wrong
    glyphs;
  * `fill="currentColor"` on the root, so the page decides the ink and the
    formulas follow the light and dark themes like the rest of the figures.

`--png` also writes 2x rasters, which is what note.com needs.
"""
import io, os, re, sys

import matplotlib
matplotlib.use("svg")
matplotlib.rcParams["svg.fonttype"] = "path"
matplotlib.rcParams["mathtext.fontset"] = "cm"
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "viz")
SIZE = 15.5

# name -> (TeX, aria-label)
EQS = {
    # --- step version ---------------------------------------------------
    "step": (r"z^{S}_i \;=\; M^{-1/2}\left(\ell_i - \bar{\ell}_i\right),"
             r"\qquad M \;=\; \sum_r T_r",
             "ステップ版の座標は、中心化した1ステップごとの対数尤度を参照ステップ総数の平方根で割る"),
    "stepd": (r"D_S \;=\; \|z^{S}_i - z^{S}_j\| \;=\; "
              r"\sqrt{\frac{1}{M}\sum_{r,t} \left(\ell_{i,rt} - \bar{\ell}_i "
              r"- \ell_{j,rt} + \bar{\ell}_j\right)^2}",
              "ステップ版の距離は、中心化した対数尤度の差の二乗平均平方根"),
    # --- trajectory version ---------------------------------------------
    "traj": (r"A_{i,r} \;=\; \sum_{t<T_r} \ell_{i,rt},\qquad "
             r"z^{\tau}_i \;=\; N^{-1/2}\left(A_i - \bar{A}_i\right)",
             "軌跡版は、1本の軌跡の対数尤度を合計してから中心化し、軌跡本数の平方根で割る"),
    "trajd": (r"D_{\tau} \;=\; \|z^{\tau}_i - z^{\tau}_j\|",
              "軌跡版の距離は、軌跡座標の差のノルム"),
    "est": (r"\frac{\bar{T}}{2}\,D_S^{\,2} \qquad\qquad \frac{1}{2}\,D_{\tau}^{\,2}",
            "軌跡KLの推定量は、ステップ版が平均手数の半分倍の距離二乗、軌跡版が距離二乗の半分"),
    # --- the target the estimates are compared against -------------------
    "kl": (r"\mathrm{KL}(i\,\|\,j) \;=\; \sum_t \sum_a \pi_i(a\mid h_t)\,"
           r"\log \frac{\pi_i(a\mid h_t)}{\pi_j(a\mid h_t)}",
           "軌跡KLは、各履歴での許容行動についての和をステップに沿って足したもの"),
    "ksym": (r"K_{\mathrm{sym}} \;=\; \mathrm{KL}(i\,\|\,j) + \mathrm{KL}(j\,\|\,i)",
             "対称KLは両方向の和"),
    # --- alpha = 0 -------------------------------------------------------
    "hel": (r"\|\sqrt{p}-\sqrt{q}\|^2 \;=\; 2\left(1 - \mathrm{BC}\right),"
            r"\qquad \mathrm{BC} \;=\; \sum_a \sqrt{p_a\,q_a}",
            "ヘリンジャー距離の二乗はバッタチャリヤ係数で書ける"),
    "geo": (r"d_{\mathrm{FR}} \;=\; 2\,\arccos\left(\mathrm{BC}\right)",
            "Fisher-Rao 測地線距離はバッタチャリヤ係数の逆余弦の2倍"),
    "bounds": (r"\mathrm{KL} \;\approx\; 2\|\sqrt{p}-\sqrt{q}\|^2 "
               r"\qquad\qquad \|\sqrt{p}-\sqrt{q}\|^2 \;\leq\; \mathrm{KL}",
               "局所では KL はヘリンジャー距離の二乗の2倍に近く、大域では常にその下限"),
    # --- scoring details -------------------------------------------------
    "pi": (r"\pi_i(a\mid h) \;=\; \sum_{v\,:\,\mathcal{N}(\mathrm{decode}(v))=a}\;"
           r"\prod_k q_i^{\mathcal{G}}(v_k \mid h, v_{<k})",
           "行動の確率は、その行動に正規化されるトークン列すべての和"),
    "mask": (r"q_i^{\mathcal{G}}(v_k \mid h, v_{<k}) \;=\; "
             r"\frac{q_i(v_k \mid C_i(h), v_{<k})}"
             r"{\sum_{v \in \mathcal{V}(h,\,v_{<k})} q_i(v \mid C_i(h), v_{<k})}",
             "文法マスクは、各位置で許される語の上で確率を割り直す"),
}


def render(name, tex, label, png=False):
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.text(0, 0, "$" + tex + "$", fontsize=SIZE)
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.02,
                transparent=True)
    if png:
        fig.savefig(os.path.join(OUT, f"eq_{name}.png"), format="png",
                    bbox_inches="tight", pad_inches=0.02, transparent=True, dpi=220)
    plt.close(fig)
    return clean(buf.getvalue(), name, label)


def clean(s, name, label):
    s = s[s.index("<svg"):]                      # drop the prolog and doctype
    s = re.sub(r"<metadata>.*?</metadata>", "", s, flags=re.S)
    s = re.sub(r'<g id="patch_1">.*?</g>', "", s, flags=re.S)   # white backdrop
    # Namespace every id, and every reference to one.
    ids = set(re.findall(r'id="([^"]+)"', s))
    for i in sorted(ids, key=len, reverse=True):
        s = s.replace(f'id="{i}"', f'id="{name}-{i}"')
        s = s.replace(f'href="#{i}"', f'href="#{name}-{i}"')
    s = s.replace("<svg ", f'<svg role="img" aria-label="{label}" '
                           f'fill="currentColor" ', 1)
    # The pt width/height stay: they give the fragment its intrinsic size, and
    # the page caps it with `max-width: 100%`.
    return re.sub(r"\n\s*\n", "\n", s).strip()


def main():
    png = "--png" in sys.argv
    os.makedirs(OUT, exist_ok=True)
    for name, (tex, label) in EQS.items():
        svg = render(name, tex, label, png)
        p = os.path.join(OUT, f"eq_{name}.svg")
        open(p, "w", encoding="utf-8").write(svg)
        vb = re.search(r'viewBox="([^"]+)"', svg).group(1)
        print(f"{os.path.basename(p):18s} {len(svg):6d} B  viewBox {vb}")


if __name__ == "__main__":
    main()
