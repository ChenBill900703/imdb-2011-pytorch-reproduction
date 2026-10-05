# repro01 執行記錄的計畫關聯補充

本輪執行前已制定 `reports/repro01-plan.md` 並取得使用者「請開始、越快完成越好」授權。
所有 `runs/repro01-*/run-record.yaml` 由既有診斷記錄器產生；其 `experiment_plan_id`
仍沿用模板的 `imdb2011-code-validation`，`simulation.purpose` 也沿用診斷文字。

這是記錄器的描述性 metadata 限制。本文件將全部 `repro01-*` run 關聯到本轮固定設定計畫，
不改寫原始記錄、命令、輸出、程式雜湊、時間或結果。其 `simulation.usage: NONE` 是正確的；
本輪使用真實官方資料，最終評估會讀官方 test。診斷計畫中的「不讀 test」不適用本輪已事先宣告的最終評估。

報告中的每個數值仍須由各模型 `test_metrics.json`、`test_predictions.npz` 支持。
