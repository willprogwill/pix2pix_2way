## make_noise.py
ガウシアンノイズを教師画像データにかけるプログラム。  
__main__ 実行時には引数に　`processing_ratio`, `mu`, `sigma`をとることができる。

* 変数
  * processing_ratio - 画像フォルダのどれくらいの割合に処理を行うかの比率(例　0.2)
  * mu - ガウシアン関数におけるμ
  * sigma - ガウシアン関数におけるσ

初期設定では
`processing_ratio　=　0.1`  
`mu = 0`  
`sigma = 100`  

## animate_Map_pix2pix4noise.py
simulateに対するノイズは事前にノイズ用のデータを作成するべきかも
