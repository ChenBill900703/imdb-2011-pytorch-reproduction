"""Summarize every prespecified result, including incomplete or failed runs."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re

TARGETS = [
    ("bow", "BoW (bnc)", 87.80),
    ("delta", "BoW (delta)", 88.23),
    ("semantic-vectors", "Semantic", 87.30),
    ("semantic-combined", "Semantic + BoW", 88.28),
    ("full-vectors", "Full", 87.44),
    ("full-combined", "Full + BoW", 88.33),
    ("full_unsup-vectors", "Full + unlabeled", 87.99),
    ("full_unsup-combined", "Full + unlabeled + BoW", 88.89),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.prefix):
        parser.error("Invalid prefix")
    root = Path(__file__).resolve().parents[1]
    rows = []
    for suffix, name, target in TARGETS:
        run = root / "runs" / f"{args.prefix}-{suffix}" / "classifier"
        selection_path = run / "selection.json"
        test_path = run / "test_metrics.json"
        selection = json.loads(selection_path.read_text()) if selection_path.exists() else None
        test = json.loads(test_path.read_text()) if test_path.exists() else None
        rows.append({"name": name, "paper_accuracy_percent": target,
                     "cv_accuracy_percent": selection["cv_accuracy"] * 100 if selection else None,
                     "test_accuracy_percent": test["accuracy"] * 100 if test else None,
                     "difference_pp": test["accuracy"] * 100 - target if test else None,
                     "needs_gap_review": abs(test["accuracy"] * 100 - target) >= 1.0 if test else None,
                     "C": selection["best_C"] if selection else None,
                     "test_evidence": str(test_path.relative_to(root)) if test else None})
    models = []
    for variant in ("semantic", "full", "full_unsup"):
        log = root / "runs" / f"{args.prefix}-{variant}" / "model" / "training.jsonl"
        if log.exists():
            records = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
            if records:
                models.append({"variant": variant, "iterations": len(records),
                               "first_objective": records[0]["objective"],
                               "last_objective": records[-1]["objective"],
                               "training_seconds": sum(r["seconds"] for r in records),
                               "converged": records[-1]["converged"]})
    execution = []
    for record in sorted((root / "runs").glob(f"{args.prefix}-*/run-record.yaml")):
        content = record.read_text(encoding="utf-8")
        status = re.search(r"^run_status: (\w+)$", content, re.M).group(1)
        start = json.loads(re.search(r"^started_at: (.+)$", content, re.M).group(1))
        end = json.loads(re.search(r"^ended_at: (.+)$", content, re.M).group(1))
        seconds = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() if start and end else None
        execution.append({"run": record.parent.name, "status": status, "seconds": seconds})
    contrasts = []
    for name, before, after in [("加入情感監督（vectors）", 2, 4),
                                 ("加入無標籤資料（vectors）", 4, 6),
                                 ("加入無標籤資料（combined）", 5, 7),
                                 ("完整模型串接相對 BoW", 0, 7)]:
        a, b = rows[before], rows[after]
        expected = b["paper_accuracy_percent"] - a["paper_accuracy_percent"]
        actual = None if a["test_accuracy_percent"] is None or b["test_accuracy_percent"] is None else b["test_accuracy_percent"] - a["test_accuracy_percent"]
        contrasts.append({"name": name, "paper_gain_pp": expected, "observed_gain_pp": actual,
                          "same_direction": actual > 0 if actual is not None else None})
    report = {"prefix": args.prefix, "results": rows, "models": models, "execution": execution, "contrasts": contrasts,
              "complete": all(row["test_accuracy_percent"] is not None for row in rows)}
    directory = root / "reports"
    directory.mkdir(exist_ok=True)
    (directory / f"{args.prefix}-results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [f"# {args.prefix} 實驗結果", "", "所有預先指定模型均列出；空白表示尚無結果。",
             "CV 為選參分數，不是官方 test 成績；監督式詞向量的 CV 另有 README 所述限制。", "",
             "| 模型 | 原論文 test % | 本次 test % | 差距（百分點） | CV % | C |",
             "|---|---:|---:|---:|---:|---:|"]
    def fmt(value):
        return "—" if value is None else f"{value:.3f}"
    for row in rows:
        lines.append(f"| {row['name']} | {row['paper_accuracy_percent']:.2f} | {fmt(row['test_accuracy_percent'])} | {fmt(row['difference_pp'])} | {fmt(row['cv_accuracy_percent'])} | {row['C'] if row['C'] is not None else '—'} |")
    lines.extend(["", "參考：[原論文 Table 2 的 Our Dataset 欄](https://aclanthology.org/P11-1015.pdf)。", "",
                  "差距絕對值達 1 個百分點者標記為需要診斷；此門檻在本輪官方 test 評估前設定，不是統計顯著性或復現成功的判定。", "",
                  "## 訓練狀態", "", "未達停止門檻的模型不可宣稱已收斂。", ""])
    for model in models:
        lines.append(f"- {model['variant']}：{model['iterations']} 輪；loss {model['first_objective']:.6f} → {model['last_objective']:.6f}；{model['training_seconds']/60:.2f} 分鐘；converged={model['converged']}。")
    lines.extend(["", "## 執行記錄", ""])
    for item in execution:
        elapsed = f"{item['seconds']/60:.2f} 分鐘" if item["seconds"] is not None else "進行中"
        lines.append(f"- {item['run']}：{item['status']}，{elapsed}。")
    flagged = [row['name'] for row in rows if row['needs_gap_review']]
    lines.extend(["", "## 原論文差距檢查", ""])
    if flagged:
        lines.append("需要優先診斷（絕對差距至少 1 百分點）：" + "、".join(flagged) + "。")
    elif report["complete"]:
        lines.append("本輪各列的絕對差距均小於 1 百分點；這不等於原作者實作已被精確重現。")
    else:
        lines.append("結果尚未齊全，暫不判定是否貼近原論文。")
    lines.extend(["", "## 改善方向", "", "以下只是單一 seed 的描述性對照，微小差值不代表統計顯著。", "",
                  "| 比較 | 原論文改善（百分點） | 本次改善（百分點） |", "|---|---:|---:|"])
    for contrast in contrasts:
        lines.append(f"| {contrast['name']} | {contrast['paper_gain_pp']:.2f} | {fmt(contrast['observed_gain_pp'])} |")
    lines.extend(["", "原作者未公開的設定及本實作的假設見 README。本次只有 seed 42；不能推論多次實驗的平均或變異。", ""])
    (directory / f"{args.prefix}-results.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
