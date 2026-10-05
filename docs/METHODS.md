# 方法與實作假設

本文件補充首頁的白話流程，區分論文描述與本次為了能執行而補定的設定。

## 模型與目標函數

令 `X[d,w]` 是文件 d 中詞 w 的次數、`R[w]` 是詞向量、`theta[d]` 是文件隱變量。程式中的 R 以詞為列，與論文矩陣的方向互為轉置。

語意機率為：

```text
p(w | d) = softmax(theta[d] @ R.T + word_bias)[w]
semantic_nll = -sum_d sum_w X[d,w] * log p(w | d)
```

情感機率為 `sigmoid(R[w] @ psi + sentiment_bias)`。有標籤文件的星等轉成 `s[d] = (rating - 1) / 9`，以軟標籤二元交叉熵計算每個詞的情感損失。

```text
sentiment_nll = sum_labeled_d sum_w X[d,w] * BCE(s[d], p_positive[w]) / class_count[d]
objective = (semantic_nll + sentiment_weight * sentiment_nll
             + nu * sum(R**2) + lambda * sum(theta**2)) / N
```

`class_count[d]` 是與該篇影評同極性的有標籤文件數；N 是參與語意訓練的文件數。最後除 N 只是整個目標共同縮放，沒有各自平均後改變項目的相對權重。

情感項以每個詞的正負目標質量聚合，數學上等價於逐 token 計算；測試會比對兩者的值與梯度。无標籤文件只參與語意項。Semantic 設定把情感項權重設為 0。

## 怎麼最佳化？

每一個外迴圈有兩步：

1. 固定所有文件向量，更新詞向量、詞偏置與情感參數。完整語料的梯度分批累積，不是每個 batch 各自更新一次。
2. 固定詞參數，以文件區塊求解各篇文件的 MAP 向量。

兩步都使用 PyTorch L-BFGS 與 strong-Wolfe line search。完整 5,000 詞 softmax，沒有負採樣。每個區塊有內部迭代上限，因此不保證每次都求到精確最優解。

## 文件特徵與 SVM

訓練語意模型時用真正詞頻；產生分類特徵時改用詞是否出現的 binary counts：

```text
document_vector = L2_normalize(binary_counts @ R)
bow_vector = L2_normalize(binary_counts)
combined = concatenate(bow_vector, document_vector)
```

BoW 與詞向量區塊各自正規化；串接後沒有再共同正規化。分類時不用 MAP 文件向量。

SVM 使用 LIBLINEAR 的 L2 regularization、squared-hinge loss；`dual="auto"` 依樣本與特徵數選求解方式。C 由固定五折分層 CV 選取。delta-IDF 權重只在各 fold 的訓練部分擬合，避免該權重偷看驗證標籤。

## 哪些是本次補定的？

| 項目 | 本次選擇及影響 |
|---|---|
| 求解器 | L-BFGS；不是作者完整 solver 設定的直接移植 |
| 正則化 | `lambda=nu=1`，不是已確認的作者數值 |
| 正則化符號 | 原文最大化式印為正 norm 加 log likelihood；本版依 regularization 語意，最小化 NLL 加正 L2，屬明列的解讀 |
| 情感權重 | 按式 (11) 使用每類文件數倒數，額外 weight=1；不自行放大情感項 |
| 初始化 | seed 42；詞、情感向量與文件向量使用 0.01 尺度的高斯初始化 |
| 詞表來源 | 所有版本都由 labeled train 的詞頻建立；無標籤資料不改變預設詞表 |
| BoW 範圍 | 預設同樣 5,000 詞；提供 `--bow-scope all` 作另行敏感度分析，但本輪未用 |
| delta 平滑 | add-one 的類別 document-frequency log ratio；確切平滑是否與作者一致仍待核對 |
| 數值精度 | FP32；不用混合精度，避免 line search 額外不穩定 |
| 停止 | 20 外迴圈上限，或相對改善低於 1e-5 連續 3 輪；本輪三模型都碰到上限 |
| SVM | solver/loss、5 folds、C grid 是明列預設，不是作者完整訓練設定的證明 |

## 評估邊界

官方 test 不參與詞表、詞向量、SVM 或 C 選擇。SVM 的 CV 使用已固定詞向量；對 Full 而言，詞向量看過所有訓練標籤，因此該 CV 不是整條流程無偏的驗證分數。若要調詞模型，需另建 development split，重新訓練該階段。

本輪只以官方 test 作最後一次固定模型評估。公開結果包含所有預先指定模型，不以測試成績選 seed。單 seed、有限迭代與未知原作者設定，限制了「完整科學復現」的主張。

## 可追溯性

每個 run 保存命令、套件版本、來源雜湊、參數、模型與輸出。固定來源保存在 `reports/repro01-source/`。公開版本對機器私有路徑做了正規化，數值、模型與逐筆預測沒有改動；細節見 `PUBLICATION.md`。
