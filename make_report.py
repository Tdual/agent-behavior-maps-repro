# -*- coding: utf-8 -*-
"""Build the HTML report for the reproduction. Screen reading, no PDF."""
import io, os
import report_alpha, report_viz, report_extra

CSS = """
  body { font-family: "Hiragino Sans","Hiragino Kaku Gothic ProN","Noto Sans CJK JP",sans-serif;
         font-size: 10.5pt; line-height: 1.8; color: #333; max-width: 980px;
         margin: 0 auto; padding: 32px 28px 80px; background: #fff; }
  h1 { font-size: 24pt; color: #1a1a2e; margin: 0 0 6px; line-height: 1.35; }
  .subtitle { font-size: 13pt; color: #555; margin: 0 0 4px; }
  .date { font-size: 10.5pt; color: #777; margin: 0 0 28px; }
  h2 { font-size: 15pt; color: #16213e; border-bottom: 2px solid #e8eaf6;
       padding-bottom: 6px; margin: 40px 0 14px; }
  h3 { font-size: 12.5pt; color: #0f3460; margin: 26px 0 8px; }
  h4 { font-size: 11pt; color: #0f3460; margin: 18px 0 6px; }
  p { margin: 0 0 10px; }
  table { width: 100%; border-collapse: collapse; margin: 10px 0 6px; font-size: 9.5pt; }
  th { background: #e8eaf6; padding: 6px 9px; border: 0.5px solid #ccc;
       text-align: left; font-weight: bold; white-space: nowrap; }
  td { padding: 6px 9px; border: 0.5px solid #ccc; vertical-align: top; }
  td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
  .caption { font-size: 9pt; color: #666; margin: 0 0 16px; }
  .highlight { background: #fff9c4; padding: 1px 3px; border-radius: 2px; }
  code { background: #f5f5f5; padding: 1px 5px; border-radius: 3px; font-size: 9.5pt; }
  pre { background: #fafafa; border: 1px solid #e0e0e0; border-radius: 4px;
        padding: 12px 14px; font-size: 9.5pt; overflow-x: auto; line-height: 1.6; }
  ul, ol { margin: 0 0 12px; padding-left: 24px; }
  li { margin-bottom: 4px; }
  .lead { background: #f7f9fc; border-left: 4px solid #16213e; padding: 14px 18px;
          margin: 0 0 24px; }
  .lead p:last-child { margin-bottom: 0; }
  .warn { background: #fff6e5; border-left: 4px solid #e0a000; padding: 12px 16px;
          margin: 14px 0; }
  .warn p:last-child { margin-bottom: 0; }
  .ok { background: #eef7ee; border-left: 4px solid #3b8c3b; padding: 12px 16px;
        margin: 14px 0; }
  .ok p:last-child { margin-bottom: 0; }
  .diagram { margin: 16px 0 20px; padding: 16px; background: #fafafa;
             border: 1px solid #e0e0e0; border-radius: 4px; }
  .row { display: flex; align-items: center; justify-content: center;
         flex-wrap: wrap; gap: 6px; }
  .box { padding: 7px 12px; font-size: 9.5pt; font-weight: bold; white-space: nowrap;
         border-radius: 3px; }
  .arrow { color: #999; font-size: 15pt; }
  .bar { height: 15px; background: #16213e; border-radius: 2px; display: inline-block;
         vertical-align: middle; }
  .bar2 { height: 15px; background: #c0a000; border-radius: 2px; display: inline-block;
          vertical-align: middle; }
  .legend { font-size: 9pt; color: #666; margin-top: 8px; }
  .verdict-ok { color: #1f6b1f; font-weight: bold; }
  .verdict-no { color: #a33; font-weight: bold; }
  .verdict-na { color: #777; }
  .files td:first-child { font-family: ui-monospace, Menlo, monospace; white-space: nowrap; }
"""


def tbl(head, rows, numcols=(), cap=None):
    o = ["<table>", "<tr>"]
    for i, h in enumerate(head):
        o.append(f'<th class="n">{h}</th>' if i in numcols else f"<th>{h}</th>")
    o.append("</tr>")
    for r in rows:
        o.append("<tr>")
        for i, c in enumerate(r):
            o.append(f'<td class="n">{c}</td>' if i in numcols else f"<td>{c}</td>")
        o.append("</tr>")
    o.append("</table>")
    if cap:
        o.append(f'<p class="caption">{cap}</p>')
    return "\n".join(o)


def bars(items, scale, unit="", second=None):
    """items: list of (label, value). Horizontal CSS bars."""
    o = ['<table style="font-size:9.5pt">']
    for lab, v in items:
        w = max(2, v / scale * 420)
        o.append(f'<tr><td style="border:none;white-space:nowrap;padding:3px 10px 3px 0">{lab}</td>'
                 f'<td style="border:none;padding:3px 0"><span class="bar" style="width:{w:.0f}px"></span>'
                 f'<span style="margin-left:8px;color:#555">{v:.3f}{unit}</span></td></tr>')
    o.append("</table>")
    return "\n".join(o)


H = []
A = H.append

A(f'<!DOCTYPE html><html lang="ja"><head><meta charset="utf-8">'
  f'<title>Agent Behavior Maps 再現レポート</title><style>{CSS}</style></head><body>')

A('<h1>Agent Behavior Maps（arXiv:2610.30014）の小規模再現</h1>')
A('<p class="subtitle">ローカル計算のみ・費用ゼロ・2つの設定で測定</p>')
A('<p class="date">2026年11月4日</p>')

A('''<div class="lead">
<p><b>目的。</b> 論文の中心的な主張「エージェントの座標距離が軌跡の KL ダイバージェンスを追い、
座標の近傍は挙動が似る」を、手元の小さな環境と小さなモデルで追試する。</p>
<p><b>結論。</b> <span class="highlight">論文の主要な結果は再現した。</span>
ただし再現したのは1つの LLM のハーネス違いという設定であり、論文が照合に使った
「別々の LLM 同士」では再現できていない。あわせて、小規模な追試では測定の作り方が
結論を左右することが分かった。要点は4つ。<b>挙動の比較には乱数の共有が必要であり、
正規化した座標は論文の定義より KL をよく追い、
&alpha;=0（平方根）埋め込みに替えると埋め込みの誤差がほぼ解消し、
chat template の違いが別モデル間の違いの 86% を占める。</b></p>
</div>''')

# ---------------------------------------------------------------- 1
A('<h2>1. 何を測ったか</h2>')
A('''<p>論文の定義では、エージェントは <b>LLM と ハーネスの組</b> である。
共通の参照軌跡の上で、記録された行動の対数尤度を教師強制で測り、それを座標にする。
座標の差の二乗が、2つのエージェントの方策の KL ダイバージェンスに比例する、という主張である。</p>''')

A('''<div class="diagram"><div class="row">
<span class="box" style="background:#e3f2fd;border:1px solid #90caf9">参照軌跡<br>24本 193ステップ</span>
<span class="arrow">&rarr;</span>
<span class="box" style="background:#e3f2fd;border:1px solid #90caf9">各エージェントが<br>教師強制で採点</span>
<span class="arrow">&rarr;</span>
<span class="box" style="background:#ede7f6;border:1px solid #b39ddb">座標<br>193次元</span>
<span class="arrow">&rarr;</span>
<span class="box" style="background:#fff3e0;border:1px solid #ffb74d">座標距離</span>
</div>
<div class="row" style="margin-top:14px">
<span class="box" style="background:#e8f5e9;border:1px solid #a5d6a7">評価タスク<br>8件 最大8手</span>
<span class="arrow">&rarr;</span>
<span class="box" style="background:#e8f5e9;border:1px solid #a5d6a7">各エージェントが<br>自分の方策で実行</span>
<span class="arrow">&rarr;</span>
<span class="box" style="background:#e8f5e9;border:1px solid #a5d6a7">相互採点<br>36ファイル</span>
<span class="arrow">&rarr;</span>
<span class="box" style="background:#ffebee;border:1px solid #ef9a9a">実測KL<br>閉形式</span>
</div>
<p class="legend">上段が論文の座標、下段が照合用の真値。許容行動リストがあるので方策は有限集合上の分布になり、
1ステップの KL は閉形式の和で、軌跡の KL も密度推定を必要としない。</p>
</div>''')

A('<h3>座標の軸は参照軌跡のステップである</h3>')
A('<p>座標は <b>193次元</b>で、各軸が参照軌跡の1ステップに対応する。'
  '値はそのステップで記録された行動の対数尤度である。</p>')
A('<pre>エージェントの座標 = ( 参照1手目の行動の対数尤度,\n'
  '                      参照2手目の行動の対数尤度,\n'
  '                      ...\n'
  '                      参照193手目の行動の対数尤度 )</pre>')
A('<p>ハーネスの設定（履歴の窓、system プロンプト、温度）は座標の軸ではない。'
  '<b>各エージェントはこの193次元空間の1点であり、ハーネスの設定はその点を指すラベルである。</b>'
  '設定を変えると193個の対数尤度がすべて変わるので、点が動く。その動きが第5節で扱う変位である。</p>')

A('<h3>環境</h3>')
A('''<p>ALFWorld 風の自作の家事環境。受け皿9種、物体10種、目標は put / heat / cool / clean の4種。
決定論的で、各状態に許容行動リストがある。健全性は固定のオラクル計画で確認した
（300シードすべて成功、3〜9手、許容行動は9〜14個）。</p>''')

A('<h3>履歴の窓とは何か</h3>')
A('<p>プロンプトに<b>直前の何手ぶんの（観測・行動）の組を入れるか</b>である。'
  'どの窓でも課題文と最初の観測は入る。そこに直近の履歴を何手足すかが窓である。'
  '実装は <code>history[-window:]</code> を並べるだけで、モデルの重みは変わらない。</p>')
A('''<div class="diagram">
<table style="font-size:9.5pt;table-layout:fixed">
<tr><th style="width:50%">窓 4（プレフィックス 206 トークン）</th>
    <th style="width:50%">窓 0（プレフィックス 140 トークン）</th></tr>
<tr><td><pre style="margin:0;font-size:8.5pt;line-height:1.5">Task: heat the apple and put it on the shelf
Observation: You arrive in the kitchen. On the
  sinkbasin you see nothing.
Observation: You go to the fridge. The fridge is
  closed.
Action: open fridge
Observation: You open the fridge. On the fridge
  you see a apple.
Action: take apple from fridge
Observation: You pick up the apple. You are
  carrying a apple.
Action: go to microwave
Observation: You go to the microwave. The
  microwave is closed. You are carrying a apple.
Action: open microwave</pre></td>
<td><pre style="margin:0;font-size:8.5pt;line-height:1.5">Task: heat the apple and put it on the shelf
Observation: You arrive in the kitchen. On the
  sinkbasin you see nothing.</pre>
<p style="font-size:9pt;color:#666;margin-top:10px">りんごを持っていることも、
電子レンジの前にいることも、プロンプトからは分からない。</p></td></tr>
</table>
<p class="legend">同じ課題の5手目。この後に許容行動12個の列挙と <code>Action:</code> が続く。</p>
</div>''')

A('<h3>エージェント</h3>')
A(tbl(["", "1回目", "2回目"],
      [["構成", "別々の LLM 6体", "1つの LLM・ハーネス9通り"],
       ["内訳", "SmolLM2-135M(-Instruct)、SmolLM2-360M-Instruct、Qwen2.5-0.5B(-Instruct)、pythia-410m",
        "Qwen2.5-0.5B-Instruct 固定。履歴の窓 4/2/1/0、system プロンプト 既定/簡潔/なし、温度 1.0/0.9/1.1/1.25"],
       ["ペア数", "15", "36"],
       ["意図", "論文と同じ「別々の LLM」の設定", "論文が保証の範囲とする「近いペア」を人工的に作る"]]))

A('''<div class="warn"><p><b>温度は意図的な探針である。</b> 温度は方策を変えるが、
トークン単位の対数確率を1つも変えない。したがって論文の <code>raw</code> 座標
（記録された行動の合計対数尤度）は温度の違いを原理的に見られず、
座標距離が厳密に0で実測KLが非ゼロのペアができる。</p></div>''')

# ---------------------------------------------------------------- 2
A('<h2>2. 再現状況</h2>')
A(tbl(["論文の結果", "論文の値", "本再現", "判定"],
      [["座標距離が軌跡KLの順序を追う", "Pearson 0.90 / Spearman 0.95",
        "1回目 0.324 / 0.393<br>2回目 0.941 / <b>0.975</b>", '<span class="verdict-ok">再現</span>'],
       ["軌跡版はステップ版より悪い", "水準 7.5倍 対 74倍",
        "Spearman 0.975 &rarr; 0.949、水準比 4.1倍 &rarr; 7.3倍", '<span class="verdict-ok">再現</span>'],
       ["座標の近傍は挙動が似る", "edit 0.88 / 2-gram 0.96",
        "edit <b>0.897</b> / 2-gram <b>0.950</b>", '<span class="verdict-ok">再現</span>'],
       ["ハーネスの変位の合成", "τ-bench 0.90 / 0.98 / 0.46<br>ALFWorld 0.63 / 0.89 / 0.97",
        "2成分・重なり 0.932 / 0.848 / 0.458<br>2成分・独立 <b>0.998</b> / 0.826 / 0.228"
        "<br>3成分 0.937 / 0.750 / 0.508", '<span class="verdict-ok">再現</span>'],
       ["成功率の予測", "kNN 0.88、4ベンチ平均 0.83 対 0.74",
        "測れない（オラクル経路の完走確率が 1e&minus;05 の水準）",
        '<span class="verdict-na">未検証</span>']],
      cap="2回目（1つの LLM・ハーネス9通り・共有乱数）の値。規模は論文の約 1/50。"
          "合成は第9.5節で3成分と独立成分まで広げた。"))

A('''<div class="warn"><p><b>再現の非対称。</b>
論文が KL との照合に使ったのは<b>8つの別々の LLM</b> である。
同じ構成（別々の LLM 6体）での1回目は Spearman 0.393 にとどまり、
再現したのは1つの LLM のハーネス違いという、論文が照合に使っていない設定だった。
論文の limitations は「別の LLM 同士のペアは保証範囲に入る保証がない」と書いているので
矛盾はしないが、<b>論文自身は別々の LLM で 0.95 を出している。そこは再現できていない。</b>
差が規模（988タスク 対 8タスク）なのかモデル群の選び方なのかは特定できていない。</p>
<p>第9.1節で乱数の共有に直して測り直したが、6体の挙動は飽和していて
（正規化編集距離の中央値 0.938、最大 1.000）、挙動の近さを比べる余地がなかった。
いっぽう<b>座標距離が実測の軌跡KLを追うかでは、別々の LLM でも
&alpha;=0 が最良（0.771 / 0.775）</b>で、2回目と同じ順位が出た。</p></div>''')

# ---------------------------------------------------------------- 3
A('<h2>3. 座標距離と実測KL</h2>')
A(tbl(["座標", "版", "ペア群", "n", "Pearson", "Spearman", "水準比"],
      [["raw", "step", "全体", "36", "0.911", "0.934", "3.53"],
       ["raw", "step", "ハーネスのみ", "15", "0.882", "0.846", "3.87"],
       ["raw", "step", "温度のみ", "6", "&mdash;", "&mdash;", "0.00"],
       ["raw", "traj", "全体", "36", "0.845", "0.909", "6.91"],
       ["<b>norm</b>", "<b>step</b>", "<b>全体</b>", "<b>36</b>", "<b>0.941</b>", "<b>0.975</b>", "<b>4.09</b>"],
       ["norm", "step", "ハーネスのみ", "15", "0.918", "0.900", "3.74"],
       ["norm", "step", "温度のみ", "6", "0.987", "0.886", "15.91"],
       ["norm", "traj", "全体", "36", "0.926", "0.949", "7.33"]],
      numcols=(3, 4, 5, 6),
      cap="中心化あり。raw は論文の定義（合計トークン対数尤度）、norm は許容集合で正規化した"
          "もの（方策そのもの）。水準比は推定値 ÷ 実測KL の中央値。"))

A('<h3>論文にない観察: 正規化した座標のほうが良い</h3>')
A('''<p>測っている対象が正規化された方策の KL なので、同じ対象を指す座標のほうが追える。
論文は <code>raw</code> しか使っていない。</p>''')

A('<h3>raw 座標の盲点</h3>')
A(tbl(["ペア", "raw距離", "norm距離", "実測対称KL"],
      [["h4 / h4-T1.1", "0", "0.303", "0.045"],
       ["h4 / h4-T1.25", "0", "0.667", "0.269"],
       ["h4-T0.9 / h4-T1.25", "0", "1.036", "0.556"]],
      numcols=(1, 2, 3),
      cap="温度だけが違うペア。この6ペアを含めても raw の順位相関が落ちないのは、"
          "たまたまこの6ペアが実測KLでも最小の6組で、距離0が正しく最下位に並ぶため。"))
A('''<div class="warn"><p><b>実務上の意味。</b>
サンプリング設定（温度・top-p）だけが違うエージェントは、論文の座標では同一点に落ちる。
エージェントを比較するとき温度を揃えていないなら、この地図は無言でその差を捨てる。</p></div>''')

A('<h3>相関が出るかどうかを決めるのは、ペアがKLの幅を持つかどうか</h3>')
A("""<p>1回目（別々の LLM 6体）と2回目（1 LLM・ハーネス9通り）で結果が大きく違う。
違いは、ペアの実測KLがどれだけの範囲に散らばっているかである。</p>""")
A(tbl(["", "1回目（別々の LLM 6体）", "2回目（1 LLM・ハーネス9通り）"],
      [["対称軌跡KL 最小", "9.039", "<b>0.045</b>"],
       ["対称軌跡KL 最大 &divide; 最小", "4.5倍", "<b>802倍</b>"],
       ["step版 Spearman", "0.393", "<b>0.975</b>"]], numcols=(1, 2),
      cap="1回目は1ステップあたり 0.90〜4.04 nats に固まっていた"
          "（許容行動の最大エントロピー ln12 = 2.485 に対して飽和側）。2回目は 0.007〜4.03。"))
A("""<p>順位相関は、順位をつけるべき差がペアの間にあって初めて意味を持つ。
1回目は全ペアがほぼ同じだけ離れていたので、測っていたのは順序ではなく誤差だった。
<b>この方法を試すときは、まず対象のペアが近いものから遠いものまで揃っているかを先に見るとよい。</b></p>
<p>局所性の定数 &epsilon;（Assumption 1）はこの向きを説明しない。実測の中央値は1回目 4.21、
2回目 10.97 で、相関が出た2回目のほうが大きい。&epsilon; は許容行動すべてにわたる最大値なので
一度も選ばれない裾の行動に支配されるのに対し、KL は &pi; で重み付けして裾を無視する。
論文は &epsilon; を実測しておらず、閾値も示していない。</p>""")

# ---------------------------------------------------------------- 4
A('<h2>4. 座標の近傍は挙動が似るか</h2>')
A('<p>この項目は、<b>統計量の定義と乱数の扱いで結果が大きく動く</b>。'
  '36ペアの Spearman 順位相関では 0.18〜0.34 にとどまり、'
  '論文と同じ統計量・共有乱数では 0.897 / 0.950 になる。順に示す。</p>')

A('<h3>統計量の定義</h3>')
A('''<p>論文の 0.88 / 0.96 は Table 8 の<b>標準化平均差</b>である。各モデルについて
「残りのモデルへの平均挙動距離 − 近傍5体への平均距離」を全ペアの挙動距離の標準偏差で割り、
モデルで平均し、モデルで対応づけた Wilcoxon 符号順位検定で検定する。
36ペアの Spearman 順位相関は全体の単調性を見る量で、近傍だけを見る論文の主張とは
測っている対象が違う。</p>''')

A('<h3>乱数の扱い</h3>')
A('''<p>各エージェントに独立な乱数列を与えると、<b>同一方策でも行動列が違う</b>。
h4 を種だけ変えて3回走らせると、自分同士で離れる。これが雑音の床である。</p>''')
A(bars([("h4(種1) 対 h4(種2)", 0.438), ("h4(種1) 対 h4(種3)", 0.578),
        ("h4(種2) 対 h4(種3)", 0.562)], 1.0))
A('<p class="caption">同一エージェントの反復どうしの正規化編集距離。初手の方策のエントロピーは平均 0.823 nats。</p>')
A(tbl(["指標", "自己反復（床）", "エージェント間（独立乱数）", "エージェント間（共有乱数）"],
      [["edit 最小〜最大", "0.438〜0.578", "0.500〜0.891", "<b>0.016〜0.859</b>"],
       ["edit 標準偏差", "&mdash;", "0.095", "<b>0.230</b>"],
       ["2-gram 最小〜最大", "0.627〜0.702", "<span class='highlight'>0.587</span>〜0.990", "<b>0.031〜0.930</b>"]],
      cap="独立乱数では、2-gram のエージェント間の最小値 0.587 が自己反復の床 0.627 を下回っている。"
          "別物のペアが、同じものの反復より似ているという値が出ていた。信号が雑音以下である。"))

A('<h3>直し方: 共有乱数</h3>')
A('''<pre>u = f(タスク, 手数)          # エージェントに依存しない一様乱数
行動 = 各エージェントの方策の累積分布で u を逆変換</pre>
<p>方策が同一なら行動列も同一になり、床は0になる。
実測KL 0.045 の h4 / h4-T1.1 の編集距離は、独立乱数の 0.734 から <b>0.016</b> になる。</p>''')

A('<h3>論文と同じ統計量での結果</h3>')
A(tbl(["近さの基準", "挙動の指標", "近傍k", "独立乱数", "共有乱数", "p（共有乱数）"],
      [["norm座標", "edit", "3", "0.438", "<b>0.897</b>", "0.012"],
       ["norm座標", "2-gram", "3", "0.454", "<b>0.950</b>", "0.020"],
       ["raw座標", "edit", "3", "0.604", "0.869", "0.020"],
       ["raw座標", "2-gram", "3", "0.750", "0.939", "0.027"],
       ["実測KL", "edit", "3", "0.526", "0.933", "0.004"],
       ["実測KL", "2-gram", "3", "0.541", "<b>0.984</b>", "0.008"]],
      numcols=(2, 3, 4, 5),
      cap="9体で平均、Wilcoxon 符号順位検定。9体では論文の近傍5体が8体中5体になり退化するので k=3 を示す。"
          "論文（ALFWorld・100モデル・近傍5体）は edit 0.88 / 2-gram 0.96。"))
A('''<div class="ok"><p><b>norm座標の 0.897 / 0.950 は論文の 0.88 / 0.96 とほぼ一致する。</b>
参考の Spearman も 0.181 &rarr; 0.793、0.235 &rarr; 0.786（いずれも p&lt;0.001）に上がった。
一方で座標距離と KL の対応（第3節）は独立乱数でも 0.964 だった。
<b>雑音に弱いのは挙動側の指標だけである。</b></p></div>''')

# ---------------------------------------------------------------- 5
A('<h2>5. ハーネスの変位の合成</h2>')
A('''<p>基点を h4（窓4・既定プロンプト）とし、窓の変更とプロンプトの変更を組み合わせた
4つのハーネスを追加で座標化した。実測の変位と、単独の変更による変位の和を比べる。</p>''')
A('''<p>ここでいう「変更」は論文の harness component にあたり、
<b>座標空間の基底ではなく、ハーネスに加える操作の種類である。</b>
足しているのは操作ではなく、操作が生む193次元の変位ベクトルである。</p>''')
A(tbl(["用語", "指すもの", "次元"],
      [["座標の軸", "参照軌跡の各ステップ", "193"],
       ["エージェント", "座標空間の1点", "&mdash;"],
       ["変位 &Delta;", "2点の差", "193次元ベクトル"],
       ["ハーネスの変更（論文の component）", "「窓を変える」「プロンプトを変える」という操作の種類", "2種類"]],
      numcols=(2,)))
A(tbl(["記号", "中身"],
      [["h4", "窓4・既定の system プロンプト（<b>基点</b>）"],
       ["h2 / h1 / h0", "窓だけを 2 / 1 / 0 に変えたもの"],
       ["n4 / s4", "窓4のまま、system プロンプトを なし / 簡潔版 に変えたもの"],
       ["n2 / s2 / n0 / s0", "<b>窓とプロンプトを同時に</b>変えたもの"]]))

A('''<div class="diagram">
<svg viewBox="0 0 560 210" width="100%" style="max-width:560px;display:block;margin:0 auto">
  <defs><marker id="a" markerWidth="9" markerHeight="9" refX="7" refY="3"
    orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#840791"/></marker>
   <marker id="b" markerWidth="9" markerHeight="9" refX="7" refY="3"
    orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#c0a000"/></marker></defs>
  <line x1="60" y1="170" x2="250" y2="60" stroke="#840791" stroke-width="2" marker-end="url(#a)"/>
  <text x="120" y="105" fill="#840791" font-size="12">&#916;A（窓）</text>
  <line x1="60" y1="170" x2="210" y2="178" stroke="#840791" stroke-width="2" marker-end="url(#a)"/>
  <text x="105" y="198" fill="#840791" font-size="12">&#916;B（プロンプト）</text>
  <line x1="250" y1="60" x2="400" y2="68" stroke="#bbb" stroke-width="1.5" stroke-dasharray="4 3"/>
  <line x1="60" y1="170" x2="400" y2="68" stroke="#c0a000" stroke-width="2.5" marker-end="url(#b)"/>
  <text x="250" y="140" fill="#c0a000" font-size="12">v&#770; = &#916;A + &#916;B（予測）</text>
  <line x1="60" y1="170" x2="360" y2="92" stroke="#16213e" stroke-width="2.5" marker-end="url(#a)"/>
  <text x="330" y="118" fill="#16213e" font-size="12">v（実測）</text>
  <circle cx="60" cy="170" r="4" fill="#333"/><text x="30" y="188" font-size="11">h4</text>
  <text x="408" y="72" font-size="11" fill="#c0a000">和</text>
  <text x="366" y="88" font-size="11" fill="#16213e">A+B</text>
</svg>
<p class="legend">和が実測より長い（ノルム比が1未満）。2つの変更が同じもの（入力の情報量）を削るため、
効果が飽和して重なる。</p></div>''')

A('<h3>論文の定義（Appendix C）</h3>')
A('''<pre>v  = z(A+B) − z(base)        実測の合成変位
v^ = ΔA + ΔB                 単独の変更による変位の和
cos      = cos(v^, v)
ノルム比  = ‖v‖ / ‖v^‖        実測 ÷ 予測
相対誤差  = ‖v^ − v‖ / ‖v‖
座標      = 中心化したステップ版
まとめ    = ケースごとの中央値</pre>
<p>相対誤差の列が二乗でなく生の比であることは、論文が与える恒等式
<code>‖e‖²/‖v‖² = 1 + r⁻² − 2c/r</code> で検算した。τ-bench の c=0.90, r=0.98 を入れると
0.452 になり、報告値 0.46 と一致する（二乗なら 0.205 で合わない）。</p>''')

A(tbl(["組合せ", "cos", "ノルム比", "相対誤差"],
      [["n2 = h2 + n4（窓2・プロンプトなし）", "0.922", "0.890", "0.430"],
       ["s2 = h2 + s4（窓2・簡潔）", "0.933", "0.905", "0.398"],
       ["n0 = h0 + n4（窓0・プロンプトなし）", "0.937", "<span class='highlight'>0.683</span>", "0.634"],
       ["s0 = h0 + s4（窓0・簡潔）", "0.930", "0.802", "0.486"],
       ["<b>4件の中央値（論文の方式）</b>", "<b>0.932</b>", "<b>0.848</b>", "<b>0.458</b>"],
       ["参考: 4件連結", "0.926", "0.779", "0.521"]],
      numcols=(1, 2, 3),
      cap="norm座標・中心化あり。raw でも 0.939 / 0.849 / 0.454、中心化なしでも 0.929 / 0.841 / 0.471。"
          "座標の取り方と中心化に鈍感である。"))

A(tbl(["", "cos", "ノルム比", "相対誤差"],
      [["論文 τ-bench", "0.90", "0.98", "0.46"],
       ["論文 ALFWorld", "0.63", "0.89", "0.97"],
       ["<b>本再現（ALFWorld 風）</b>", "<b>0.932</b>", "<b>0.848</b>", "<b>0.458</b>"]],
      numcols=(1, 2, 3)))

A('''<p><b>相対誤差 0.458 は τ-bench の 0.46 と一致し、cos はそれを上回る。</b>
本再現の環境は ALFWorld 風だが、論文の ALFWorld（cos 0.63・相対誤差 0.97、和がほとんど役に立たない）
ではなく τ-bench 側に落ちる。ただし<b>ノルム比 0.848 は両方より低く、和が実測を 1.18 倍に過大評価している</b>。</p>
<p>過大評価の内訳に解釈できる構造がある。窓を0にしたうえでプロンプトを外す n0 のノルム比が
0.683 で最小、窓を2に留めた n2 は 0.890。<b>窓を縮めて文脈を削った後では、
プロンプトを外して削れるものが残っていない。</b></p>''')
A('''<div class="warn"><p><b>実務上の読み。</b> 変位の向きは足し算で予測できるので
「どの方向に動くか」は言える。一方でノルムは 1.18 倍ずれ、相対誤差 0.46 なので
<b>「どれだけ動くか」は組合せを実際に測らないと分からない</b>。2つの変更が独立でないときは特にそうなる。</p></div>''')

A('<h3>ハーネスの変更ごとの平均座標シフト</h3>')
A('<p>中心化はしない（中心化はこの量をちょうど消すため）。1ステップあたり nats、h4 基準。</p>')
A(tbl(["ハーネスの変更", "シフト"],
      [["h2（窓 4&rarr;2）", "&minus;0.489"], ["h1（窓 4&rarr;1）", "<b>&minus;1.890</b>"],
       ["h0（窓 4&rarr;0）", "&minus;0.513"], ["n4（プロンプトなし）", "&minus;0.008"],
       ["s4（簡潔プロンプト）", "+0.753"]], numcols=(1,),
      cap="論文（τ-bench・12モデルの中央値）は retrieve +9.6 / summarize +3.5 / replan &minus;4.2 / think &minus;6.2。"))
A('''<p>本再現のシフトは1桁以上小さい（最大 1.89 対 9.6）。論文のハーネスの変更は入力そのものを
作り変える（検索結果を足す、要約に置き換える）のに対し、こちらは履歴の窓とプロンプトを削るだけなので、
変化が小さいのは整合する。</p>
<p><b>窓の単調性が崩れている点は注意が必要。</b> h1（&minus;1.890）が h0（&minus;0.513）より大きく動く。
窓1の「直前の1手だけが見える」が窓0の「履歴がまったくない」より中途半端な文脈を与えるため、
という解釈はできるが、<b>この解釈は検証していない。</b></p>''')

# ---------------------------------------------------------------- 6
A('<h2>6. 測定上の注意</h2>')

A('<h3>水準比は論文が主張していない精度である</h3>')
A('<p>論文は距離を「KL の順序の proxy としてのみ使う、較正された大きさの推定ではない」と明言し、'
  'limitations で中央値 7.5倍の過大評価を自己申告している。'
  '本レポートの水準比（3.5〜4.1倍）は参考値である。</p>')

A('<h3>教師強制の採点コスト</h3>')
A('<p>許容行動すべてを一括で採点するとき、全位置の logits を実体化すると'
  '候補11 &times; 系列400トークン &times; 語彙152k &times; 4バイト = 2.67GB になり、'
  '<code>log_softmax</code> と <code>float()</code> で約8GB を1回の呼び出しで確保する。'
  '必要なのは継続部分の6〜9位置だけである。<code>logits_to_keep</code> で切ると'
  '193ステップが <b>121秒</b>で終わり、数値は一致する（最大差 9.5e&minus;07、相対 6.5e&minus;08）。'
  '小さいモデルでも、この一点でメモリとディスクが尽きる。</p>')

# ---------------------------------------------------------------- 7
# ---------------------------------------------------------------- 7
report_alpha.section(A, tbl)

# ---------------------------------------------------------------- 8
report_viz.section(A, tbl)

# ---------------------------------------------------------------- 9
report_extra.section(A, tbl)

# ---------------------------------------------------------------- 10
A('<h2>10. 確認したこと</h2>')
A('''<ul>
<li>環境の健全性: オラクルの計画が300シードすべてで成功</li>
<li>自己KL が全エージェントで厳密に 0（相互採点の整合性）</li>
<li>解析の単体検算: 温度つき softmax の対数比が (v<sub>i</sub>&minus;v<sub>j</sub>)/T と一致、
閉形式KLがモンテカルロ40万標本と 0.0003 以内で一致、距離が定数オフセット 0.3 に対して 0.3 を返し
中心化後に厳密0、累積分布の逆変換が20万標本で分布と 0.005 以内で一致、編集距離の境界条件</li>
<li>バグ修正が数値を変えないこと（キャッシュ済み座標と最大差 9.5e&minus;07、相対 6.5e&minus;08）</li>
<li>ハーネス6通りが実際に異なるスコアを出していること（プレフィックス 140〜206 トークン）</li>
<li>合成の結果が raw / norm × 中心化あり / なしの4通りで一致（座標の取り違えによる偽の結論でないこと）</li>
<li>論文の相対誤差の列が生の比であることを恒等式で検算</li>
<li>&alpha;=0 の局所関係・大域の下限・球面の弦と弧の一致（第7節）</li>
<li>chat template を外す操作が、テンプレートを持たない2体で移動量ちょうど 0 になること（対照）</li>
<li>参照集合を替えても座標の順位が保たれること（3集合・Spearman 0.93&ndash;0.97）</li>
<li>語彙の切り詰めで捨てた裾の確率が測定精度で 0（上位20,000、9ハーネス・193状態）</li>
<li>3成分の合成が2成分と同じ精度で成り立つこと（前向き計算 0回で導出）</li>
<li>打ち切り手数が8タスクすべてでオラクルの手数を上回っていること</li>
<li>費用ゼロ（ローカル MPS のみ、API 不使用）</li>
</ul>''')

A('<h2>11. 確認できなかったこと</h2>')
A('''<ul>
<li><b>論文がやった設定（別々の LLM 同士）では再現できていない。</b>
第9.1節で共有乱数に直しても、6体の挙動は飽和していて比べる余地がない。
差が規模なのかモデル群の選び方なのかは特定できていない</li>
<li><b>成功率の予測は測れていない。</b>第9.6節で原因を特定した。
オラクル経路を完走する確率が 1e&minus;05 の水準で、温度を下げるとさらに悪化する。
<b>このモデルとこの環境の組では原理的に測れない</b>ので、
論文の5つの結果のうち1つは未検証のまま残る</li>
<li>規模が論文の約 1/50（参照193ステップ 対 10,388、評価8タスク 対 988、
2回目は1 LLM 対 100モデル）。相関の信頼区間は非常に広い</li>
<li>ロールアウトは全部打ち切りで終わっている（成功0のため）。論文の軌跡は自然に終了する</li>
<li>自作環境は ALFWorld 風であって本物の ALFWorld ではない。
論文の ALFWorld との食い違いの原因は特定できていない。
本物で測るには環境パッケージとデータの取得が別途要る</li>
<li>&alpha;=0 の切り詰めの検証（第9.4節）は<b>次の1トークン</b>の分布であり、
行動1個ぶんの系列ではない。また対象の分布が鋭く尖っている場合に限る</li>
<li>chat template の寄与（第9.3節）は6体での測定である。
100体の地図のうちどれだけがテンプレートなのかは、本再現からは言えない</li>
<li>参照集合は3つに増えたが（第9.2節）、どれも同じ環境・同じ24本である。
環境を替えたときの安定性は未検証</li>
<li>著者の実装と突き合わせていない（非公開）。ハーネスやトークン化の細部は本再現の解釈である</li>
</ul>''')

A('<h2>12. 成果物</h2>')
A(f'<p><code>~/Workspace/agent-behavior-maps/</code></p>')
A('<div class="files">' + tbl(["ファイル", "中身"],
      [["env.py", "ALFWorld 風の自作環境。決定論的、許容行動リストつき、固定オラクル"],
       ["agent.py", "ハーネス（窓・system プロンプト・温度）と教師強制の採点"],
       ["models.py", "1回目の6モデルの定義"],
       ["pipeline.py", "1回目: refs / coords / roll / cross の4段、各段キャッシュ"],
       ["close.py", "2回目: 9ハーネス、共有乱数のロールアウト、合成用の座標"],
       ["report.py", "1回目の解析"],
       ["close_report.py", "2回目の 座標距離 対 実測KL、ε、群別"],
       ["close_behaviour.py", "論文と同じ統計量での挙動の近傍"],
       ["close_compose.py", "ハーネスの変位の合成と変更ごとのシフト"],
       ["close_alpha.py", "&alpha;=0 埋め込みと参照分布上の厳密KL"],
       ["alpha14.py", "6つの LLM の許容行動すべてのスコア（&alpha;=0 を14体に広げる）"],
       ["viz.py、viz14.py", "古典的MDS による2次元の地図（SVG）"],
       ["tsne.py", "論文と同じ t-SNE。perplexity の掃引と初期値による揺れの測定"],
       ["crn6.py、crn6_an.py", "1回目の6モデルを共有乱数でやり直す（第9.1節）"],
       ["refs2.py、refs2_an.py", "参照集合を2つ追加して &lambda; の感度を測る（第9.2節）"],
       ["notmpl.py、notmpl_an.py", "chat template を外した採点と比較（第9.3節）"],
       ["trunc.py", "次トークン分布の上位 k 切り詰めの影響（第9.4節）"],
       ["compose2.py", "3成分の連鎖と、互いに独立な成分の合成（第9.5節）"],
       ["alpha_compose.py", "&alpha;=0 座標での合成"],
       ["why0.py", "成功 0/8 の原因の切り分け（第9.6節）"],
       ["make_report.py", "このレポートの生成"],
       ["FINDINGS.md", "426行。設定と測定値の記録"],
       ["close/、coords/、rollouts/、cross/", "生の測定値"]]) + '</div>')

A(tbl(["", "計算時間", "費用"],
      [["1回目（6モデル）", "約2時間（バグ修正前の空転を除く）", "0"],
       ["2回目（9ハーネス・独立乱数）", "約50分", "0"],
       ["2回目（共有乱数で作り直し）", "約25分", "0"],
       ["合成（4ハーネスの座標）", "6分", "0"],
       ["&alpha;=0 埋め込みの解析", "数秒（前向き計算 0回）", "0"],
       ["14体ぶんの許容行動スコア取り直し", "8分", "0"],
       ["2次元の地図", "数秒（前向き計算 0回）", "0"],
       ["共有乱数での6モデル再ロールアウト", "24分", "0"],
       ["参照集合2つの作成と9体の採点", "24分", "0"],
       ["chat template なしの採点（6体）", "9分", "0"],
       ["次トークン分布の取り出しと切り詰めの解析", "7分", "0"],
       ["3成分・独立成分の合成", "数秒（前向き計算 0回）", "0"],
       ["成功 0/8 の切り分け", "1分", "0"],
       ["<b>合計</b>", "&mdash;", "<b>0</b>"]],
      cap="すべてローカルの MPS（fp32）。外部 API は使っていない。"))

A('</body></html>')

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report.html")
open(out, "w", encoding="utf-8").write("\n".join(H))
print(out, os.path.getsize(out), "bytes")
