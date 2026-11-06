# -*- coding: utf-8 -*-
"""Template vs. plain text: how much of the LLM map is the chat template?

Reads coords_plain/ (written by notmpl.py) against the original coordinates
and writes notmpl.json for the report.
"""
import itertools, json, os
import numpy as np
from scipy.stats import pearsonr, spearmanr
import alpha14, models, agent

HERE = os.path.dirname(os.path.abspath(__file__))
PLAIN = os.path.join(HERE, "coords_plain")
NAMES = [n for n, _ in models.MODELS]
PAIRS = list(itertools.combinations(NAMES, 2))
EPS = 1e-300


def plain():
    out = {}
    for n in NAMES:
        c = json.load(open(os.path.join(PLAIN, f"{n}.json")))
        out[n] = c
    return out


def z(vecs, idx, kind):
    M = len(idx)
    if kind == "α=1 raw":
        v = np.array([vecs[k][idx[k]] for k in range(M)])
        return (v - v.mean()) / np.sqrt(M)
    pol = [agent.normalise(v) for v in vecs]
    if kind == "α=1 norm":
        v = np.array([np.log(pol[k][idx[k]] + EPS) for k in range(M)])
        return (v - v.mean()) / np.sqrt(M)
    return np.concatenate([np.sqrt(p) for p in pol]) / np.sqrt(M)


if __name__ == "__main__":
    P = plain()
    idx = alpha14.chosen_index(NAMES[0])
    orig = {n: alpha14._src(n) for n in NAMES}
    KINDS = ["α=1 raw", "α=1 norm", "α=0 √π"]
    ZO = {k: {n: z(orig[n]["vecs"], orig[n]["idx"], k) for n in NAMES} for k in KINDS}
    ZP = {k: {n: z(P[n]["vecs"], P[n]["idx"], k) for n in NAMES} for k in KINDS}

    per = []
    for n in NAMES:
        po = [agent.normalise(v) for v in orig[n]["vecs"]]
        pp = [agent.normalise(v) for v in P[n]["vecs"]]
        a0 = float(np.sqrt(np.mean([2.0 * (1.0 - float(np.sqrt(x * y).sum()))
                                    for x, y in zip(po, pp)])))
        per.append({"name": n, "had": bool(P[n]["had_template"]),
                    "shift": float(np.linalg.norm(ZO["α=1 norm"][n]
                                                  - ZP["α=1 norm"][n])),
                    "self_a0": a0})
    print(f"{'モデル':<12} テンプレ  座標の移動  自分自身とのα=0距離")
    for r in per:
        print(f"  {r['name']:<12} {'あり' if r['had'] else 'なし'}   "
              f"{r['shift']:.3f}   {r['self_a0']:.3f}")

    pairs = {}
    for k in KINDS:
        a = np.array([np.linalg.norm(ZO[k][i] - ZO[k][j]) for i, j in PAIRS])
        b = np.array([np.linalg.norm(ZP[k][i] - ZP[k][j]) for i, j in PAIRS])
        pairs[k] = {"pearson": float(pearsonr(a, b)[0]),
                    "spearman": float(spearmanr(a, b)[0]),
                    "ratio": float(np.median(b / a))}
        print(f"{k:<10} Pearson {pairs[k]['pearson']:.3f} "
              f"Spearman {pairs[k]['spearman']:.3f} 比 {pairs[k]['ratio']:.3f}")

    sp = pairs["α=0 √π"]["spearman"]
    tmpl = [r for r in per if r["had"]]
    med = float(np.median([r["self_a0"] for r in tmpl])) if tmpl else 0.0
    cross = float(np.median([np.linalg.norm(ZO["α=0 √π"][i] - ZO["α=0 √π"][j])
                             for i, j in PAIRS]))
    if sp >= 0.9:
        v = (f"<b>テンプレートを外しても、6体の並びは変わらない</b>"
             f"（&alpha;=0 のペア距離の Spearman {sp:.3f}）。"
             f"ただし同じ重み・同じハーネスでテンプレートの有無だけが違う2つの方策は "
             f"&alpha;=0 距離で中央 {med:.3f} 離れており、"
             f"これは別モデル同士の中央 {cross:.3f} の "
             f"{med/cross*100:.0f}% に当たる。"
             f"<b>テンプレートはエージェントの一部として実際に効いているが、"
             f"モデル間の違いのほうが大きく、地図の形は保たれる。</b>")
        cls = "ok"
    else:
        v = (f"<b>テンプレートを外すと並びが変わる</b>"
             f"（&alpha;=0 のペア距離の Spearman {sp:.3f}）。"
             f"同じ重みでテンプレートの有無だけが違う2つの方策は "
             f"&alpha;=0 距離で中央 {med:.3f} 離れ、"
             f"別モデル同士の中央 {cross:.3f} の {med/cross*100:.0f}% に当たる。"
             f"<b>1回目の base / instruct の対比は、テンプレートの違いと分離できていない。</b>")
        cls = "warn"
    print("\n" + v.replace("<b>", "").replace("</b>", ""))
    json.dump({"per_model": per, "pairs": pairs, "verdict": v,
               "verdict_class": cls, "self_median": med, "cross_median": cross},
              open(os.path.join(HERE, "notmpl.json"), "w"),
              ensure_ascii=False, indent=1)
