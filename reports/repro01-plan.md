# repro01 固定設定實驗計畫

使用者授權：2026-10-04 要求現在開始，越快完成越好。
本輪為既定設定的一個 seed 完整執行，不是無限調參或達標保證。

- 命令：`scripts/run_experiments.ps1 -RunPrefix repro01 -TimeoutSeconds 3600 -Evaluate`。
- 依序執行 BoW、delta、Semantic、Full、Full-unsup，以及詞模型的 vectors / combined SVM。
- 使用現有 configs；seed 42，詞模型最多 20 個外迴圈。未收斂仍如實標註為預算上限。
- C 候選 0.01、0.1、1、10、100；訓練內 5-fold CV；SVM dual=auto。
- `dual=auto` 是本輪開始前的計算效率修改，維持 L2 squared-hinge 目標；10 項測試通過後啟動。
- 三組模型／分類器都完成後才讀官方 test。所有結果都保留，不按 test 成績挑參數或種子。
- 每命令上限 3,600 秒；錯誤或逾時停止。保留每輪 checkpoint、stdout、命令、程式雜湊與設定。
- 不使用付費或外部運算資源。

耗時粗估：Full-unsup 20 × 約 35 秒 ≈ 12 分鐘；兩個 25k 模型合计約 8 分鐘；分類器、載入與評估另留 5–20 分鐘。合計先估 20–45 分鐘，隨 line search、CPU/GPU 負載修正。

本輪未涵蓋：多 seed、獨立 development split 的詞模型超參數搜尋、所有未知作者設定的敏感度分析。這些不能因為本輪跑完就標記為已完成。

使用者追加要求（官方 test 評估之前）：觀察與作者數據是否差太遠。
為每一列計算 test accuracy 與 Table 2 的差距；绝對差距至少 1 百分點列為診斷標記。
這不是統計檢定、不是事後挑選容許誤差，也不授權按 test 成績反覆調參。
若超出，先檢查收斂、詞彙／特徵處理、正則化与情感項尺度，將確定事實與待驗證原因分開。
