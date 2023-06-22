## make_noise.py
ガウシアンノイズをフォルダにかけるプログラム。__
__main__ 実行時には引数に　`processing_ratio`, `mu`, `sigma`　をとることができる。

* 変数
  * processing_ratio　-　画像フォルダのどれくらいの割合に処理を行うかの比率(例　0.25)
  * mu -　ガウシアン関数におけるμ
  * sigma -　ガウシアン関数におけるσ

初期設定では
`processing_ratio　=　0.1`
`mu = 0`
`sigma = 100`
