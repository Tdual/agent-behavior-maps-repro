# -*- coding: utf-8 -*-
"""Hand-authored inline SVG figures for the article.

Separate from viz.py, which draws the measured maps from the coordinate files.
These are explanatory diagrams: they carry no measurement of their own, only
the mechanism the prose argues about. Every stroke is currentColor so the page
decides the ink; the one literal hue (ACC) is reserved for the single element
that carries the claim in each figure.

Writes viz/fig_*.svg. No data, no forward pass.
"""
import math, os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "viz")
ACC = "#1c4f8c"
FONT = "Hiragino Sans,Noto Sans CJK JP,sans-serif"
MONO = "ui-monospace,Menlo,monospace"


def head(w, h, label):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'role="img" aria-label="{label}" '
            f'style="width:100%;height:auto;display:block;font-family:{FONT}">'
            f'<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0 0 L10 5 L0 10 z" fill="currentColor"/></marker>'
            f'<marker id="ara" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0 0 L10 5 L0 10 z" fill="{ACC}"/></marker></defs>')


def txt(x, y, s, size=12.5, anchor="middle", weight="normal", fill="currentColor",
        family=None, op=1.0):
    f = f' font-family="{family}"' if family else ""
    o = f' opacity="{op}"' if op != 1.0 else ""
    return (f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
            f'font-weight="{weight}" fill="{fill}"{f}{o}>{s}</text>')


def box(x, y, w, h, lines, size=12.5, sub=10.5, dash=False, acc=False):
    st = ACC if acc else "currentColor"
    d = ' stroke-dasharray="5 4"' if dash else ""
    o = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="none" '
         f'stroke="{st}" stroke-width="{2 if acc else 1.4}"{d}/>']
    n = len(lines)
    gap = sub + 5.5
    y0 = y + h / 2 - (n - 1) * gap / 2 + 4.5
    for i, s in enumerate(lines):
        big = i == 0
        o.append(txt(x + w / 2, y0 + i * gap, s, size if big else sub,
                     weight="600" if big else "normal",
                     fill=ACC if (acc and big) else "currentColor",
                     op=1.0 if big else 0.72))
    return "".join(o)


def arrow(x1, y1, x2, y2, label=None, lx=None, ly=None, acc=False, size=10.5):
    st = ACC if acc else "currentColor"
    m = "url(#ara)" if acc else "url(#ar)"
    o = [f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{st}" '
         f'stroke-width="1.4" marker-end="{m}"/>']
    if label:
        o.append(txt(lx if lx is not None else (x1 + x2) / 2,
                     ly if ly is not None else (y1 + y2) / 2 - 7,
                     label, size, op=0.78))
    return "".join(o)


def write(name, s):
    open(os.path.join(OUT, name), "w", encoding="utf-8").write(s + "</svg>")
    print(name, len(s))


# ---------------------------------------------------------------- 1 pipeline
def pipeline():
    W, H = 880, 340
    o = [head(W, H, "共通の参照軌跡を2体のエージェントに教師強制で採点させ、"
                    "対数尤度を並べて中心化した座標の差で距離を測る流れ")]
    TOP, BOT, MID = 74, 214, 144
    o.append(box(8, MID - 52, 128, 104,
                 ["参照軌跡", "全員で共通", "M = 193 ステップ"]))
    for y, who in ((TOP, "i"), (BOT, "j")):
        o.append(box(214, y - 40, 150, 80,
                     [f"エージェント {who}", "LLM + ハーネス"]))
        o.append(box(430, y - 40, 162, 80,
                     [f"&#8467;<tspan dy='3' font-size='9'>{who}</tspan>"
                      f"<tspan dy='-3'> = log &pi;</tspan>", "記録された行動の", "対数尤度を M 個"]))
        o.append(box(658, y - 40, 112, 80,
                     [f"z<tspan dy='3' font-size='9'>{who}</tspan>", "座標"]))
    o.append(arrow(140, MID - 26, 210, TOP + 16, None))
    o.append(arrow(140, MID + 26, 210, BOT - 16, None))
    o.append(txt(175, MID + 4, "同じ履歴と行動", 10, op=0.78))
    for y in (TOP, BOT):
        o.append(arrow(368, y, 426, y))
        o.append(arrow(596, y, 654, y))
    o.append(txt(397, TOP - 12, "教師強制", 10.5, op=0.78))
    o.append(txt(397, TOP + 22, "で採点", 10.5, op=0.78))
    o.append(txt(397, BOT - 12, "教師強制", 10.5, op=0.78))
    o.append(txt(397, BOT + 22, "で採点", 10.5, op=0.78))
    o.append(txt(625, TOP - 12, "中心化", 10.5, op=0.78))
    o.append(txt(625, TOP + 22, "&times; M<tspan dy='-5' font-size='8'>&minus;1/2</tspan>",
                 10.5, op=0.78))
    o.append(txt(625, BOT - 12, "中心化", 10.5, op=0.78))
    o.append(txt(625, BOT + 22, "&times; M<tspan dy='-5' font-size='8'>&minus;1/2</tspan>",
                 10.5, op=0.78))
    o.append(box(790, MID - 48, 84, 96,
                 ["&#8214;z<tspan dy='3' font-size='9'>i</tspan>"
                  "<tspan dy='-3'>&minus;z</tspan><tspan dy='3' font-size='9'>j</tspan>"
                  "<tspan dy='-3'>&#8214;</tspan>", "距離", "= 地図"], acc=True))
    o.append(arrow(774, TOP + 10, 802, MID - 50, acc=True))
    o.append(arrow(774, BOT - 10, 802, MID + 50, acc=True))
    o.append(txt(W / 2, 324, "ロールアウトは1本も走らせない。前向き計算だけで座標が決まる。",
                 11, op=0.62))
    return "".join(o)


# ---------------------------------------------------------------- 2 tail
def tail():
    W, H = 700, 372
    L, R, T, B = 86, 596, 44, 262
    o = [head(W, H, "確率が 0 に近づくとき log p は発散し、"
                    "&radic;p は 0 に収束することを示す曲線")]
    lo, hi = -6.0, 0.0          # log10 p
    ylo, yhi = -14.0, 2.0

    def X(l10):
        return L + (l10 - lo) / (hi - lo) * (R - L)

    def Y(v):
        return B - (v - ylo) / (yhi - ylo) * (B - T)

    o.append(f'<rect x="{X(-6)}" y="{T}" width="{X(-3) - X(-6)}" height="{B - T}" '
             f'fill="{ACC}" opacity="0.07"/>')
    o.append(txt((X(-6) + X(-3)) / 2, T + 18,
                 "一度も選ばれない行動がここに入る", 10.5, fill=ACC))
    for v in (2, 0, -4, -8, -12):
        o.append(f'<line x1="{L}" y1="{Y(v)}" x2="{R}" y2="{Y(v)}" '
                 f'stroke="currentColor" stroke-width="1" opacity="0.13"/>')
        o.append(txt(L - 10, Y(v) + 4, f"{v}", 10, anchor="end", op=0.55))
    o.append(f'<line x1="{L}" y1="{Y(0)}" x2="{R}" y2="{Y(0)}" '
             f'stroke="currentColor" stroke-width="1" opacity="0.4"/>')
    for e in range(-6, 1):
        o.append(f'<line x1="{X(e)}" y1="{B}" x2="{X(e)}" y2="{B + 5}" '
                 f'stroke="currentColor" stroke-width="1" opacity="0.4"/>')
        lab = "1" if e == 0 else f"10<tspan dy='-5' font-size='8'>{e}</tspan>"
        o.append(txt(X(e), B + 20, lab, 10, op=0.62))
    o.append(txt((L + R) / 2, B + 46, "その行動に方策が置く確率 p", 11.5, op=0.78))
    o.append(txt(L - 10, T - 12, "座標の値", 11.5, anchor="end", op=0.78))

    pts_log, pts_sq = [], []
    for i in range(241):
        l10 = lo + (hi - lo) * i / 240
        p = 10 ** l10
        pts_log.append(f"{X(l10):.1f},{Y(math.log(p)):.1f}")
        pts_sq.append(f"{X(l10):.1f},{Y(math.sqrt(p)):.1f}")
    o.append(f'<polyline points="{" ".join(pts_log)}" fill="none" '
             f'stroke="currentColor" stroke-width="2" stroke-dasharray="7 4"/>')
    o.append(f'<polyline points="{" ".join(pts_sq)}" fill="none" '
             f'stroke="currentColor" stroke-width="2"/>')
    o.append(txt(X(-0.6), Y(-9.3), "log p（&alpha;=1・論文の座標）", 11.5,
                 anchor="end", weight="600"))
    o.append(txt(R, Y(1.5), "&radic;p（&alpha;=0）", 11.5,
                 anchor="end", weight="600"))
    o.append(f'<line x1="{L + 24}" y1="344" x2="{L + 56}" y2="344" '
             f'stroke="currentColor" stroke-width="2"/>')
    o.append(txt(L + 62, 348, "実線 &radic;p", 10.5, anchor="start", op=0.72))
    o.append(f'<line x1="{L + 190}" y1="344" x2="{L + 222}" y2="344" '
             f'stroke="currentColor" stroke-width="2" stroke-dasharray="7 4"/>')
    o.append(txt(L + 228, 348, "破線 log p", 10.5, anchor="start", op=0.72))
    return "".join(o)


# ---------------------------------------------------------------- 3 ratio
def ratio():
    W, H = 760, 300
    L, R, T = 232, 712, 82
    o = [head(W, H, "4つの座標について、推定した距離の2乗と実測KLの比を"
                    "中央値と範囲で比べたチャート")]
    rows = [("<tspan font-weight='600'>&alpha;=1 log</tspan> 論文の raw",
             3.54, 0.00, 5.54),
            ("<tspan font-weight='600'>&alpha;=1 log</tspan> 許容集合で正規化",
             3.69, 2.06, 14.31),
            ("<tspan font-weight='600'>&alpha;=0 &radic;</tspan> Hellinger",
             0.82, 0.66, 1.00),
            ("<tspan font-weight='600'>&alpha;=0 &radic;</tspan> Fisher&ndash;Rao",
             0.87, 0.72, 1.00)]
    xmax, dy = 15.0, 44

    def X(v):
        return L + v / xmax * (R - L)

    for v in (0, 1, 5, 10, 15):
        x = X(v)
        o.append(f'<line x1="{x}" y1="{T - 14}" x2="{x}" y2="{T + dy * 4 - 18}" '
                 f'stroke="currentColor" stroke-width="1" opacity="0.13"/>')
        o.append(txt(x, T - 22, f"{v}", 10, op=0.55))
    o.append(f'<line x1="{X(1)}" y1="{T - 14}" x2="{X(1)}" y2="{T + dy * 4 - 18}" '
             f'stroke="{ACC}" stroke-width="2"/>')
    o.append(txt(X(1) + 8, T - 40, "1.0 = 実測KLと一致", 10.5, anchor="start", fill=ACC))
    o.append(txt(16, 24, "推定した距離の2乗 &divide; 実測KL（中央値と範囲、36ペア）",
                 11.5, anchor="start", op=0.78))
    for i, (name, med, lo_, hi_) in enumerate(rows):
        y = T + i * dy
        o.append(txt(L - 16, y + 4, name, 11, anchor="end"))
        o.append(f'<line x1="{X(lo_)}" y1="{y}" x2="{X(hi_)}" y2="{y}" '
                 f'stroke="currentColor" stroke-width="2" opacity="0.38"/>')
        for v in (lo_, hi_):
            o.append(f'<line x1="{X(v)}" y1="{y - 5}" x2="{X(v)}" y2="{y + 5}" '
                     f'stroke="currentColor" stroke-width="2" opacity="0.38"/>')
        o.append(f'<circle cx="{X(med)}" cy="{y}" r="5.5" fill="currentColor" '
                 f'stroke="#fff" stroke-width="2"/>')
        if med < 1.5:   # left of the dot, clear of the 1.0 line
            o.append(txt(X(med) - 12, y + 4, f"{med:.2f}", 11, anchor="end", weight="600"))
        else:
            o.append(txt(X(med), y - 13, f"{med:.2f}", 11, weight="600"))
        o.append(txt(X(hi_) + 10, y + 4, f"上端 {hi_:.2f}", 9.5, anchor="start", op=0.6))
    o.append(txt(16, 286, "&alpha;=1 は中央値で 3.5 倍以上に出て、上端は 14 を超える。"
                          "&alpha;=0 は範囲ごと 1.0 の下にとどまる。", 10.5,
                 anchor="start", op=0.62))
    return "".join(o)


# ---------------------------------------------------------------- 4 sphere
def sphere():
    W, H = 700, 334
    o = [head(W, H, "&alpha;=0 の座標は半径1の球面上にあり、2体の差は球面を貫く弦で、"
                    "弧より 2〜4% 短いことを示す図")]
    cx, cy, r = 350, 330, 220

    def P(deg):
        a = math.radians(deg)
        return cx + r * math.cos(a), cy - r * math.sin(a)

    # the surface
    o.append(f'<path d="M{P(150)[0]:.1f},{P(150)[1]:.1f} '
             f'A{r},{r} 0 0 1 {P(30)[0]:.1f},{P(30)[1]:.1f}" fill="none" '
             f'stroke="currentColor" stroke-width="1.6" opacity="0.4"/>')
    o.append(txt(P(150)[0] + 2, P(150)[1] - 12,
                 "&#8214;z&#8214; = 1 の球面（22体すべてが乗る）", 11,
                 anchor="start", op=0.7))
    bx, by = P(130)
    zx, zy = P(50)
    # the arc between the two agents, then the chord
    o.append(f'<path d="M{bx:.1f},{by:.1f} A{r},{r} 0 0 1 {zx:.1f},{zy:.1f}" '
             f'fill="none" stroke="{ACC}" stroke-width="3.5"/>')
    o.append(f'<line x1="{bx:.1f}" y1="{by:.1f}" x2="{zx:.1f}" y2="{zy:.1f}" '
             f'stroke="currentColor" stroke-width="2"/>')
    o.append(txt(cx, P(90)[1] - 14, "弧 = Fisher&ndash;Rao 測地線 2arccos(BC)",
                 11.5, fill=ACC, weight="600"))
    o.append(txt(cx, by + 24, "弦 = 座標の差 &#8214;z &minus; b&#8214;", 11.5, weight="600"))
    o.append(txt(cx, by + 42, "この2本の長さの比が 0.957&ndash;0.981", 10.5, op=0.68))
    for x, y, s_, an, ox in ((bx, by, "基準 h4", "end", -11),
                             (zx, zy, "別のエージェント", "start", 11)):
        o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="currentColor" '
                 f'stroke="#fff" stroke-width="2"/>')
        o.append(txt(x + ox, y + 4, s_, 11.5, anchor=an, weight="600"))
    o.append(box(28, 252, 300, 62,
                 ["座標の差は接ベクトルではない", "弦の和は球面に乗らない"]))
    o.append(box(372, 252, 300, 62,
                 ["それでも接空間に移して足すと", "cos 0.995 &rarr; 0.995（変わらない）"], acc=True))
    return "".join(o)


# ---------------------------------------------------------------- 5 template
def template():
    W, H = 880, 400
    o = [head(W, H, "同じプロンプト本文が chat template で別の前置きに包まれ、"
                    "対数尤度と座標が動いて地図の順位が崩れる流れ")]
    o.append(box(8, 150, 150, 100,
                 ["プロンプト本文", "タスク・観測", "許容行動リスト", "（全員同じ）"]))
    TA, TB = 40, 240
    o.append(f'<rect x="212" y="{TA}" width="300" height="140" rx="3" fill="none" '
             f'stroke="{ACC}" stroke-width="2"/>')
    o.append(txt(362, TA + 20, "smol135-it / qwen05-it のテンプレート", 11.5,
                 weight="600", fill=ACC))
    mono = ["&lt;|im_start|&gt;system",
            "You are a helpful AI assistant",
            "named SmolLM&#8230;&lt;|im_end|&gt;",
            "&lt;|im_start|&gt;user → 本文 →",
            "&lt;|im_start|&gt;assistant"]
    for i, s in enumerate(mono):
        o.append(txt(228, TA + 44 + i * 19, s, 10.5, anchor="start", family=MONO,
                     op=0.8))
    o.append(f'<rect x="212" y="{TB}" width="300" height="100" rx="3" fill="none" '
             f'stroke="currentColor" stroke-width="1.4" stroke-dasharray="5 4"/>')
    o.append(txt(362, TB + 22, "smol135 / pythia410（テンプレートなし）", 11.5,
                 weight="600"))
    o.append(txt(228, TB + 52, "本文がそのまま入る", 10.5, anchor="start", family=MONO,
                 op=0.8))
    o.append(txt(228, TB + 76, "既定の system プロンプトも付かない", 10.5,
                 anchor="start", op=0.7))
    o.append(arrow(162, 180, 208, TA + 110))
    o.append(arrow(162, 220, 208, TB + 30))
    o.append(box(560, TA + 16, 150, 108,
                 ["log &pi; が変わる", "前置きが全トークン", "の確率を動かす"], acc=True))
    o.append(box(560, TB + 6, 150, 88,
                 ["log &pi; は素のまま", "変化ちょうど 0", "（対照）"]))
    o.append(arrow(516, TA + 70, 556, TA + 70, acc=True))
    o.append(arrow(516, TB + 50, 556, TB + 50))
    o.append(box(738, 150, 134, 100,
                 ["地図の順位が崩れる", "&alpha;=0 のペア距離", "Spearman &minus;0.050"],
                 acc=True))
    o.append(arrow(714, TA + 70, 764, 146, acc=True))
    o.append(arrow(714, TB + 50, 764, 254))
    o.append(txt(W / 2, 384,
                 "同じモデルのテンプレート有無の距離は、モデル間の距離の 86%（中央値 0.667 対 0.772）",
                 11, op=0.62))
    return "".join(o)


if __name__ == "__main__":
    write("fig_pipeline.svg", pipeline())
    write("fig_tail.svg", tail())
    write("fig_ratio.svg", ratio())
    write("fig_sphere.svg", sphere())
    write("fig_template.svg", template())
