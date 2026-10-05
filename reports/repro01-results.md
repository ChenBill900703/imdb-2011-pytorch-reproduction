# repro01 實驗結果

所有預先指定模型均列出；空白表示尚無結果。
CV 為選參分數，不是官方 test 成績；監督式詞向量的 CV 另有 README 所述限制。

| 模型 | 原論文 test % | 本次 test % | 差距（百分點） | CV % | C |
|---|---:|---:|---:|---:|---:|
| BoW (bnc) | 87.80 | 87.376 | -0.424 | 87.572 | 1 |
| BoW (delta) | 88.23 | 88.228 | -0.002 | 88.380 | 0.1 |
| Semantic | 87.30 | 87.732 | 0.432 | 88.116 | 10 |
| Semantic + BoW | 88.28 | 88.684 | 0.404 | 88.872 | 0.1 |
| Full | 87.44 | 87.688 | 0.248 | 88.084 | 0.1 |
| Full + BoW | 88.33 | 88.680 | 0.350 | 88.924 | 0.1 |
| Full + unlabeled | 87.99 | 88.396 | 0.406 | 88.268 | 0.1 |
| Full + unlabeled + BoW | 88.89 | 89.004 | 0.114 | 89.032 | 0.1 |

參考：[原論文 Table 2 的 Our Dataset 欄](https://aclanthology.org/P11-1015.pdf)。

差距絕對值達 1 個百分點者標記為需要診斷；此門檻在本輪官方 test 評估前設定，不是統計顯著性或復現成功的判定。

## 訓練狀態

未達停止門檻的模型不可宣稱已收斂。

- semantic：20 輪；loss 828.740761 → 744.977733；4.07 分鐘；converged=False。
- full：20 輪；loss 828.746870 → 744.922214；4.11 分鐘；converged=False。
- full_unsup：20 輪；loss 796.725398 → 748.003217；13.26 分鐘；converged=False。

## 執行記錄

- repro01-bow：SUCCEEDED，0.25 分鐘。
- repro01-bow-test：SUCCEEDED，0.07 分鐘。
- repro01-delta：SUCCEEDED，0.29 分鐘。
- repro01-delta-test：SUCCEEDED，0.07 分鐘。
- repro01-full：SUCCEEDED，4.21 分鐘。
- repro01-full-combined：SUCCEEDED，0.68 分鐘。
- repro01-full-combined-test：SUCCEEDED，0.07 分鐘。
- repro01-full-vectors：SUCCEEDED，0.14 分鐘。
- repro01-full-vectors-test：SUCCEEDED，0.07 分鐘。
- repro01-full_unsup：SUCCEEDED，13.39 分鐘。
- repro01-full_unsup-combined：SUCCEEDED，0.67 分鐘。
- repro01-full_unsup-combined-test：SUCCEEDED，0.07 分鐘。
- repro01-full_unsup-vectors：SUCCEEDED，0.14 分鐘。
- repro01-full_unsup-vectors-test：SUCCEEDED，0.07 分鐘。
- repro01-semantic：SUCCEEDED，4.17 分鐘。
- repro01-semantic-combined：SUCCEEDED，0.68 分鐘。
- repro01-semantic-combined-test：SUCCEEDED，0.07 分鐘。
- repro01-semantic-vectors：SUCCEEDED，0.14 分鐘。
- repro01-semantic-vectors-test：SUCCEEDED，0.07 分鐘。

## 原論文差距檢查

本輪各列的絕對差距均小於 1 百分點；這不等於原作者實作已被精確重現。

## 改善方向

以下只是單一 seed 的描述性對照，微小差值不代表統計顯著。

| 比較 | 原論文改善（百分點） | 本次改善（百分點） |
|---|---:|---:|
| 加入情感監督（vectors） | 0.14 | -0.044 |
| 加入無標籤資料（vectors） | 0.55 | 0.708 |
| 加入無標籤資料（combined） | 0.56 | 0.324 |
| 完整模型串接相對 BoW | 1.09 | 1.628 |

原作者未公開的設定及本實作的假設見 README。本次只有 seed 42；不能推論多次實驗的平均或變異。
