# -*- coding: utf-8 -*-
"""Rasterise the article's SVG figures for note.com, which needs images.

rsvg-convert is an XML parser, so the named HTML entities the figures carry
(&alpha;, &rarr;, &#8214; ...) have to be resolved first; only the five XML
entities may stay. Output is 2x the CSS width so the text holds up on a phone.

    python3 note/svg2png.py             # every figure listed below
    python3 note/svg2png.py fig_sphere  # one of them
"""
import html, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "viz")
OUT = os.path.join(HERE, "images")
KEEP = {"amp", "lt", "gt", "quot", "apos"}

# published name -> (svg, width in px of the PNG)
FILES = {
    "10_pipeline":  ("fig_pipeline.svg", 1360),
    "20_map_a1":    ("map9_a1.svg",      1320),
    "30_tail":      ("fig_tail.svg",     1360),
    "31_sphere":    ("fig_sphere.svg",   1360),
    "32_ratio":     ("fig_ratio.svg",    1360),
    "33_map_a0":    ("map9_a0.svg",      1320),
    "40_tsne":      ("tsne_6.svg",       1360),
    "41_template":  ("fig_template.svg", 1360),
    "50_eq_step":   ("eq_step.svg",       900),
    "51_eq_hel":    ("eq_hel.svg",        900),
    "52_eq_geo":    ("eq_geo.svg",        560),
    "53_eq_bounds": ("eq_bounds.svg",     980),
    "54_eq_log":    ("eq_log.svg",        980),
}


def resolve(s):
    """Named HTML entities -> characters, so an XML parser accepts the file."""
    def sub(m):
        return m.group(0) if m.group(1) in KEEP else html.unescape(m.group(0))
    return re.sub(r"&([a-zA-Z][a-zA-Z0-9]*);", sub, s)


def one(name):
    svg, w = FILES[name]
    src = os.path.join(SRC, svg)
    txt = resolve(open(src, encoding="utf-8").read())
    dst = os.path.join(OUT, f"{name}.png")
    p = subprocess.run(["rsvg-convert", "-w", str(w), "-b", "white", "-o", dst],
                       input=txt.encode(), capture_output=True)
    if p.returncode or not os.path.exists(dst):
        raise SystemExit(f"{name}: {p.stderr.decode().strip()}")
    print(f"  {name}.png  {os.path.getsize(dst):>8} B  from {svg}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for n in (sys.argv[1:] or FILES):
        one(n)
