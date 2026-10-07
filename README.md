# IMDB 情感分類復現｜Learning Word Vectors for Sentiment Analysis

## 論文資訊

- **Paper**：Learning Word Vectors for Sentiment Analysis
- **Authors**：Andrew L. Maas、Raymond E. Daly、Peter T. Pham、Dan Huang、Andrew Y. Ng、Christopher Potts
- **Venue / Year**：ACL 2011，pp. 142–150
- **Paper URL**：[ACL Anthology](https://aclanthology.org/P11-1015/)／[全文](https://aclanthology.org/P11-1015.pdf)
- **Official Code**：完整原始訓練程式未確認；[作者網站](https://ai.stanford.edu/~amaas/data/sentiment/)提供官方資料與預先處理特徵。本專案是依論文重寫的 PyTorch 實作。

## 論文摘要

原論文研究如何學到適合情感分析的詞向量。只利用詞語共現學習語意，未必能區分情感極性，因此作者結合無監督的文件語意模型與影評評分監督，使詞向量同時保留語意與情感資訊，再用文件特徵進行情感分類。論文也建立較大的 IMDB 影評資料集，評估無標籤資料及詞向量與詞袋特徵互補的效果。以下為繁體中文整理，來源為[原論文摘要與第 3–4 節](https://aclanthology.org/P11-1015.pdf)。

## 研究目的

研究問題是：**能否把語意共現與情感標籤納入同一個詞表示學習目標，讓學到的表示更適合判斷正負評？** 動機在於語意相近或出現在相似語境的詞，不一定具有相同情感極性。作者另探討，在有標籤資料有限時，額外無標籤影評能否改善表示，以及稠密詞向量能否與 BoW 提供互補資訊。來源：[原論文第 1、3、4 節](https://aclanthology.org/P11-1015.pdf)。

## 研究方法

研究任務是：讀一篇英文影評，判斷它是正面還是負面。

原論文先讓每個詞有一個 **50 維的數值表示（詞向量）**。訓練時有兩個目標：

1. **學語意**：利用一篇影評裡出現的詞，學習詞與文件之間的關係。
2. **學情感**：利用影評的 1～10 星評分，讓詞向量也包含情感訊息。

詞向量學好後，將一篇影評中出現的詞向量相加並正規化，得到文件特徵。最後用 **線性 SVM** 判斷正負評。另一組實驗把這個特徵與 BoW（詞是否出現）接在一起。

語意部分以文件隱變量與 log-linear／softmax 模型描述詞出現機率；情感部分以詞向量預測文件評分對應的情感目標。兩者形成聯合目標，交替更新詞參數與文件的 MAP 隱變量。上述是原論文方法；本專案的 L-BFGS、正則化與停止條件等補定設定另列於後方及[方法與假設](docs/METHODS.md)。來源：[原論文第 3 節及第 4.3 節](https://aclanthology.org/P11-1015.pdf)。

## 原論文主要成果

與本專案相關的是 [Table 2 的 Our Dataset 欄](https://aclanthology.org/P11-1015.pdf#page=7)：BoW 為 **87.80%**，Semantic 為 **87.30%**，Full 為 **87.44%**，Full 加額外無標籤資料為 **87.99%**；再與 BoW 串接後為 **88.89%**。這些是原論文的測試準確率。結果顯示此資料集上無標籤資料與 BoW 互補有助分類，但不能把每項方法在其他資料集的效果一概而論。完整八組對照保留於「本專案復現結果」。

## 本專案復現範圍

本專案只對照 Table 2 的 IMDB／Our Dataset 欄，共八組 BoW、delta-IDF、Semantic、Full、額外無標籤資料及 BoW 串接結果；使用官方訓練／測試分割。

**本專案的研究問題**：用現代 PyTorch，能不能把 Maas 等人 2011 年的 IMDB 情感分類結果重新做出來？

這個流程沒有使用 BERT 或外部預訓練詞向量。實驗範圍是原論文的 IMDB 部分；沒有重跑 PL04、Subjectivity、LDA 或 LSA。

## 本專案復現結果

下面全部都是同一份 **25,000 篇官方測試資料**的準確率，不是訓練分數。

| 方法 | 原論文 | 本次 | 差距（百分點） |
|---|---:|---:|---:|
| BoW：只看哪些詞出現過 | 87.80% | 87.376% | −0.424 |
| BoW + delta-IDF：增加情感詞的區別 | 88.23% | 88.228% | −0.002 |
| Semantic：只學語意詞向量 | 87.30% | 87.732% | +0.432 |
| Semantic + BoW | 88.28% | 88.684% | +0.404 |
| Full：同時學語意與情感 | 87.44% | 87.688% | +0.248 |
| Full + BoW | 88.33% | 88.680% | +0.350 |
| Full + 額外無標籤資料 | 87.99% | 88.396% | +0.406 |
| **Full + 額外無標籤資料 + BoW** | **88.89%** | **89.004%** | **+0.114** |

原論文數字來自 [Table 2 的 Our Dataset 欄](https://aclanthology.org/P11-1015.pdf)。差距是「本次減原論文」，單位是百分點；不是相對提升百分比。

八組最大絕對差距是 **0.432 個百分點**。主要模型正確分類 **22,251／25,000 篇**。所有模型都公開，沒有只挑最高分。

## 結果討論與限制

**支持的結論**：本輪主要準確率接近作者；增加無標籤資料與串接 BoW 的改善方向，也與原論文一致。

**必須保留的限制**：

- **情感監督的小幅提升沒有重現。** Semantic 為 87.732%，Full 為 87.688%，本次略降 0.044 百分點；原論文則增加 0.14。
- **三個詞模型都沒有達到嚴格收斂門檻。** 它們在 20 輪預算上限結束，記錄中的 `converged` 都是 `false`。
- **只有一個 seed。** 沒有多次實驗的平均與變異，也不能因多了 0.114 百分點，就宣稱統計上優於作者。
- **部分作者設定未知。** 求解器、正則化、詞表細節與特徵尺度的補充假設均公開；接近的分數不能證明實作逐項一致。

本輪事先以「絕對差距達 1 百分點」作為需要診斷的標記；這不是統計顯著性或復現成功的標準。更完整的差異說明見 [結果解讀](reports/repro01-interpretation.md)。

## 實驗與復現細節

### 1. 使用官方資料，不重新分割測試集

從 [Stanford 作者網站](https://ai.stanford.edu/~amaas/data/sentiment/) 下載 `aclImdb_v1`：

- **25,000 篇有標籤訓練資料**：學詞向量、訓練分類器與選參數。
- **50,000 篇無標籤資料**：只在 Full + unlabeled 版本中，協助學語意。
- **25,000 篇測試資料**：全部模型訓練與選參完成後，才進行最終評估。

直接使用作者提供的 `.feat` 詞頻檔案，減少重新分詞造成的差異。資料壓縮檔與訓練檔案的 SHA256 都有保存。原始影評不隨本儲存庫重新散布，請用下載指令取得。

### 2. 先測程式，再跑真實資料

先完成 10 項測試，檢查目標函數、梯度、文件特徵、續訓、資料隔離與 CUDA 運算。人工小資料只用來測程式，沒有拿它充當實驗結果。

接著在完整 75,000 篇語料上短跑，確認 GPU 能執行、loss 有下降，再開始正式的一輪實驗。

### 3. 固定設定，訓練三種詞模型

| 設定 | 本輪使用值 |
|---|---|
| 詞模型 | Semantic、Full、Full + unlabeled |
| 詞表 | 從有標籤訓練集詞頻排序，略過最常見 50 詞，再取 5,000 詞 |
| 詞向量維度 | 50 |
| 星等目標 | `(rating − 1) / 9`，保留軟評分資訊 |
| 求解方式 | 詞參數與文件向量交替最佳化，L-BFGS |
| softmax | 完整 5,000 詞，沒有用負採樣近似 |
| 外迴圈上限 | 20 輪 |
| 每輪內部迭代上限 | 詞參數 10、文件向量 20 |
| 文件批次大小 | 256 |
| 隨機種子 | 42 |
| 數值精度 | FP32 |

完整設定在 [configs](configs/)，原論文未交代而由本專案補定的項目見 [方法與假設](docs/METHODS.md)。

### 4. 只用訓練資料選 SVM 參數

對每一組特徵做五折交叉驗證，從 `C = 0.01、0.1、1、10、100` 選參數，再用完整訓練資料擬合 SVM。所有候選分數都保留。

這裡的交叉驗證是**分類器選參分數**：詞向量在切分 SVM folds 前已經學好；Full 的詞向量看過這批訓練標籤。因此不能把該 CV 分數當作整條流程的獨立泛化估計。對外比較使用另行隔離的官方測試集。

### 5. 一次評估固定模型，核對原始輸出

所有設定與分類器固定後，評估官方測試集。沒有看完本輪測試成績後再改參數重跑。

額外核對了八組的逐筆預測、正確筆數、準確率、混淆矩陣、測試資料雜湊與執行版本，均通過。查看 [機器可讀核對結果](reports/repro01-audit.json)。

## 環境

| 階段 | 實測耗時 |
|---|---:|
| Semantic 詞模型，20 輪 | 約 4.07 分鐘 |
| Full 詞模型，20 輪 | 約 4.11 分鐘 |
| Full + unlabeled 詞模型，20 輪 | 約 13.26 分鐘 |
| 全部訓練、選參及測試 | **25 分 20 秒** |

環境為 **Windows、RTX 3070 Ti 8GB、Python 3.12.14、PyTorch 2.11.0+cu128**。版本清單見 [requirements-lock.txt](requirements-lock.txt)，起訖時間見 [duration.json](reports/repro01-duration.json)。這個時間不包含安裝環境、下載資料與撰寫程式；其他硬體的速度可能不同。

## 如何重跑

以下以 PowerShell 為例。請先安裝 Python 3.12 與可供 PyTorch 使用的 GPU 驅動。

```powershell
git clone https://github.com/ChenBill900703/imdb-2011-pytorch-reproduction.git
cd imdb-2011-pytorch-reproduction
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pytest -q tests
.\.venv\Scripts\python.exe -m imdb2011 download
```

重新執行整組固定實驗，使用**新名稱**以保留已公開結果：

```powershell
.\scripts\run_experiments.ps1 -RunPrefix repro02 -Evaluate
```

這會執行兩組 BoW 基線、三組詞模型、六組詞模型分類器，最後才評估八組官方測試結果。每個命令的時間上限預設 3,600 秒；失敗會停止並留下記錄。若只想訓練與選參，省略 `-Evaluate`。

只重跑主要模型：

```powershell
.\.venv\Scripts\python.exe -m imdb2011 train --config configs/full_unsup.json --output runs/my-full-unsup
.\.venv\Scripts\python.exe -m imdb2011 fit-svm --features combined --checkpoint runs/my-full-unsup/checkpoint.pt --output runs/my-classifier
.\.venv\Scripts\python.exe -m imdb2011 evaluate --run runs/my-classifier
```

下載資料後，可重新核對本儲存庫保存的第一輪結果：

```powershell
.\.venv\Scripts\python.exe scripts/audit_results.py --prefix repro01
```

`--resume` 可從詞模型 checkpoint 接續；只允許更改裝置與增加外迴圈上限。已公開的 test 結果不應成為反覆調參的目標；需要深入診斷時，應從訓練資料另切開發集。

## Repository 結構

| 路徑 | 內容 |
|---|---|
| [imdb2011/](imdb2011/) | 資料、模型、交替訓練、SVM 與測試程式 |
| [configs/](configs/) | 三種模型及短跑設定 |
| [tests/](tests/) | 程式正確性測試 |
| [runs/](runs/) | 第一輪模型、分類器、逐筆預測、loss 與執行記錄 |
| [reports/repro01-results.md](reports/repro01-results.md) | 八組結果、CV、耗時與改善方向 |
| [reports/repro01-source/](reports/repro01-source/) | 產生第一輪結果的固定程式快照 |
| [docs/METHODS.md](docs/METHODS.md) | 數學目標與實作假設 |
| [PUBLICATION.md](PUBLICATION.md) | 公開版本的整理與路徑正規化說明 |

## 參考資料

Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. 2011. **Learning Word Vectors for Sentiment Analysis**. ACL 2011, pp. 142–150.

- [原論文與 BibTeX](https://aclanthology.org/P11-1015/)
- [官方資料集](https://ai.stanford.edu/~amaas/data/sentiment/)

本專案僅重寫程式與整理本次實驗；原論文、資料集及其權利仍屬各自作者與權利人。
