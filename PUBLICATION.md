# 公開版本說明

本儲存庫公開 2026-10-04 完成的第一輪實驗；公開整理日期為 2026-10-05。

包含完整程式、設定、測試、套件版本、第一輪 checkpoint、SVM 分類器、逐筆預測與記錄。
不包含原始影評資料集、虛擬環境、帳號憑證或工作機私人目錄。

## 哪些內容經過整理？

- README 重新撰寫為公開閱讀版本，另加方法說明；沒有重寫實驗數字。
- `command.json` 的 Python 執行檔改成 `.venv/Scripts/python.exe`，不公開工作機絕對路徑。
- 結果摘要 JSON 的證據路徑改成 `/`，方便跨平台讀取。
- 原始資料的下載指紋放到 `reports/dataset-download.json`；資料本身請向作者網站下載。
- 公開核對腳本支援 `--data` 指定資料目錄。實驗期間的訓練程式快照與其雜湊保留原樣。

`reports/publication-manifest.json` 記錄輸出檔案的來源雜湊、公開版雜湊與轉換原因。
模型、逐筆預測、test metrics、CV 數值與 loss 紀錄均按原始位元組保留。

原始 run-record 的 plan-id 沿用早期診斷模板；它不等於本輪沒有執行正式測試。
完整說明與對應計畫見 `reports/repro01-record-context.md`、`reports/repro01-plan.md`。

`reports/repro01-audit.json` 是實驗完成時保存的本機核對結果。任何人可在下載官方資料並安裝相同依賴後，以 `scripts/audit_results.py --prefix repro01` 再次核對。

GitHub 公開儲存庫本身不是原論文作者對此實作的背書。本專案也沒有新增或更改原資料集的授權。
