# note 入稿手順 — Agent Behavior Maps の再現

## 前提

| 項目 | 値 |
|---|---|
| 原本 | `note/post.md`（本文 8,213字・画像20枚） |
| インポート用 | `note/post.wxr.ready.xml`（25.8 KB、XML 構文チェック済み） |
| プレビュー | `note/preview.html`（note の記事幅 620px を模したもの） |
| slug | `agent-behavior-maps-repro` |
| アカウント | 個人（note.com/tdual） |
| 表の扱い | note に表ブロックが無いので7枚の PNG にした（`tables.json` → `render_tables.py`） |
| 図・式の扱い | 記事の SVG を `svg2png.py` で 2倍の PNG にした（`rsvg-convert`） |

## 画像

S3 に20枚（本文用）＋見出し画像1枚を置いた。全て HTTP 200 を確認済み。

```
https://note-articles-images-2025.s3.ap-northeast-1.amazonaws.com/agent-behavior-maps-repro/<ファイル名>
```

| ファイル | 中身 |
|---|---|
| `00_thumbnail.png` | 見出し画像 1280x670（WXR には入らない。手動で設定する） |
| `01_claims.png` 〜 `07_success.png` | 表7枚 |
| `10_pipeline.png` | 座標の作り方の図 |
| `20_map_a1.png` / `33_map_a0.png` | 9体の地図（α=1 / α=0） |
| `30_tail.png` / `31_sphere.png` / `32_ratio.png` | 裾の扱い・球面と弦・水準比 |
| `40_tsne.png` / `41_template.png` | 6体の t-SNE 地図・テンプレートの寄与 |
| `50_eq_step.png` 〜 `54_eq_log.png` | 数式5点 |

## 手順

1. note にログインし、プロフィールアイコン → 「自分の記事」
2. 右上の「インポート」→「WordPress (WXR)」
3. `note/post.wxr.ready.xml` を選ぶ（Finder からドラッグできる）
4. 下書きに入るので、次を目視で確認する
   - 見出し（`##` と `###`）が階層どおりか
   - 画像が20枚とも表示されているか
   - 太字・リンクが残っているか
   - 水平線（`---`）で挟んだ「ここから先は論文に書かれていない」の断りが残っているか
5. 見出し画像を設定する。「記事設定」→ 画像を追加 → URL から
   `https://note-articles-images-2025.s3.ap-northeast-1.amazonaws.com/agent-behavior-maps-repro/00_thumbnail.png`
6. ハッシュタグを付ける（下記）
7. プレビューでスマホ幅を確認してから公開

## ハッシュタグ案

```
#AI #LLM #生成AI #機械学習 #論文読み #arXiv #情報幾何 #エージェント #AIエージェント #再現実験
```

## 直すとき

```bash
cd ~/Workspace/agent-behavior-maps

# 図を直した場合
python3 figs.py && python3 eqs.py && python3 note/svg2png.py

# 表を直した場合
python3 ~/.claude/skills/note-article/scripts/render_tables.py note/tables.json note/images/

# 画像を S3 に反映
aws s3 sync note/images/ s3://note-articles-images-2025/agent-behavior-maps-repro/ \
  --profile matrixflow-beta-305119140072 --exclude "*" --include "*.png" \
  --cache-control "public, max-age=31536000"

# 本文を直したら WXR を作り直す
python3 ~/.claude/skills/note-article/scripts/generate_wxr.py note/post.md note/post.wxr.ready.xml \
  --slug agent-behavior-maps-repro --date 2026-11-06 --preview note/preview.html \
  --base-url https://note-articles-images-2025.s3.ap-northeast-1.amazonaws.com/agent-behavior-maps-repro
```

インポート後は note 側に画像のコピーが残る。**S3 を更新しても公開済み記事の画像は変わらない。**
公開後に直すときは note の編集画面で差し替える。
