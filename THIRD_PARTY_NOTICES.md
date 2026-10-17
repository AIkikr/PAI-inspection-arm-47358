# 一次資料・依存ライブラリ

このプロジェクトのアーム形状と課題固有コードは新規に作成したものです。
講義の非公開スライド、演習ノートブック、ロボットメッシュ、学習済み講義モデルは同梱していません。
依存物は実行時にそれぞれの配布元から取得します。以下のコード・モデル・データの利用条件を確認してください。

- MuJoCo 3.3.7 / Apache-2.0
  - https://github.com/google-deepmind/mujoco
  - https://mujoco.readthedocs.io/en/3.3.7/python.html
  - https://mujoco.readthedocs.io/en/3.3.7/XMLreference.html
- Qwen3-1.7B / Apache-2.0（モデルカード記載）
  - https://huggingface.co/Qwen/Qwen3-1.7B
  - https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/LICENSE
  - commit SHAは実行時に各runのplanner_contract.jsonへ保存
- Hugging Face Transformers / Apache-2.0
  - https://github.com/huggingface/transformers
- Hugging Face Accelerate / Apache-2.0
  - https://github.com/huggingface/accelerate
- PyTorch / BSD-style
  - https://github.com/pytorch/pytorch/blob/main/LICENSE
- ImageIO / BSD-2-Clause、imageio-ffmpeg / BSD-2-Clause
  - https://github.com/imageio/imageio
  - https://github.com/imageio/imageio-ffmpeg
  - FFmpeg本体とエンコーダの条件は各配布ビルドに従う。実行用依存であり、このソースZIPには同梱しない。
- Pillow / HPND系（配布元のLICENSEを参照）
  - https://github.com/python-pillow/Pillow/blob/main/LICENSE
- ReportLab
  - https://www.reportlab.com/
  - https://pypi.org/project/reportlab/
- pypdf（生成PDFの読込確認）
  - https://github.com/py-pdf/pypdf
- Noto CJK（画面文字を描画する実行環境内のフォント）
  - https://github.com/notofonts/noto-cjk
  - フォントファイルはソースZIPに同梱しない。
- Colabの利用制限
  - https://research.google.com/colaboratory/faq.html

最新であることを優先して全依存を更新するのではなく、主要パッケージはこの版の組合せに固定しています。
固定したという理由だけで全体の動作が検証済みになるわけではありません。VALIDATION.mdを参照してください。
