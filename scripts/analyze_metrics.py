import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class RunMetrics:
    run: str
    model: str
    dataset_csv: str
    device: str
    epochs: int
    best_epoch: int
    train_size: int
    val_size: int
    test_size: int
    test_top1: float
    test_median_km: float
    test_acc_25km: float
    test_acc_50km: float
    test_acc_200km: float
    test_acc_750km: float
    val_top1: float
    best_selection_metric: float
    train_top1_last: Optional[float]
    val_top1_last: Optional[float]

    @property
    def generalization_gap(self) -> Optional[float]:
        if self.train_top1_last is None:
            return None
        return self.train_top1_last - self.val_top1


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_history_last(history_csv: Path) -> Tuple[Optional[float], Optional[float]]:
    if not history_csv.exists():
        return None, None
    last_row: Optional[Dict[str, str]] = None
    with open(history_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            last_row = row
    if not last_row:
        return None, None
    train_top1 = as_float(last_row.get("train_top1"), default=0.0)
    val_top1 = as_float(last_row.get("val_top1"), default=0.0)
    return train_top1, val_top1


def load_run_metrics(metrics_path: Path) -> RunMetrics:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    test = payload.get("test_metrics", {})
    val = payload.get("val_metrics", {})
    split = payload.get("split_sizes", {})

    history_csv = Path(payload.get("history_csv", ""))
    train_last, val_last = load_history_last(history_csv)

    return RunMetrics(
        run=metrics_path.parent.name,
        model=str(payload.get("model_name", "")),
        dataset_csv=str(payload.get("dataset_csv", "")),
        device=str(payload.get("device", "")),
        epochs=as_int(payload.get("epochs")),
        best_epoch=as_int(payload.get("best_epoch")),
        train_size=as_int(split.get("train")),
        val_size=as_int(split.get("val")),
        test_size=as_int(split.get("test")),
        test_top1=as_float(test.get("top1")),
        test_median_km=as_float(test.get("median_km")),
        test_acc_25km=as_float(test.get("acc_25km")),
        test_acc_50km=as_float(test.get("acc_50km")),
        test_acc_200km=as_float(test.get("acc_200km")),
        test_acc_750km=as_float(test.get("acc_750km")),
        val_top1=as_float(val.get("top1")),
        best_selection_metric=as_float(payload.get("best_selection_metric")),
        train_top1_last=train_last,
        val_top1_last=val_last,
    )


def collect_runs(models_dir: Path) -> List[RunMetrics]:
    runs: List[RunMetrics] = []
    for path in sorted(models_dir.glob("*/metrics.json")):
        try:
            runs.append(load_run_metrics(path))
        except Exception as e:
            print(f"Skipping invalid metrics file {path}: {e}")
    return runs


def sort_runs(runs: Iterable[RunMetrics], objective: str) -> List[RunMetrics]:
    if objective == "top1":
        return sorted(
            runs,
            key=lambda r: (-r.test_top1, r.test_median_km, -r.test_acc_50km),
        )
    return sorted(
        runs,
        key=lambda r: (r.test_median_km, -r.test_acc_50km, -r.test_top1),
    )


def format_float(value: Optional[float], places: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{places}f}"


def print_table(runs: List[RunMetrics]) -> None:
    if not runs:
        print("No runs found.")
        return
    header = (
        "rank run                          model             dataset             "
        "top1    med_km  acc50   acc200  gap"
    )
    print(header)
    print("-" * len(header))
    for idx, r in enumerate(runs, start=1):
        dataset_name = Path(r.dataset_csv).name
        gap = r.generalization_gap
        print(
            f"{idx:>4} {r.run:<28} {r.model:<17} {dataset_name:<18} "
            f"{r.test_top1:>6.4f} {r.test_median_km:>7.2f} {r.test_acc_50km:>7.4f} "
            f"{r.test_acc_200km:>7.4f} {format_float(gap, 4):>7}"
        )


def load_dataset_summaries(processed_dir: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for p in sorted(processed_dir.glob("dataset_summary*.json")):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            output_csv = Path(str(payload.get("output_csv", ""))).name
            if output_csv:
                out[output_csv] = payload
        except Exception:
            continue
    return out


def print_dataset_context(runs: List[RunMetrics], summaries: Dict[str, Dict[str, Any]]) -> None:
    datasets = sorted({Path(r.dataset_csv).name for r in runs})
    if not datasets:
        return
    print("\nDataset context:")
    for ds in datasets:
        summary = summaries.get(ds)
        if not summary:
            print(f"- {ds}: no dataset summary found")
            continue
        print(
            f"- {ds}: rows={summary.get('output_rows')} classes={summary.get('num_classes')} "
            f"cell_size_deg={summary.get('cell_size_deg')} min_class_count={summary.get('min_class_count')}"
        )


def print_recommendations(runs: List[RunMetrics], objective: str) -> None:
    if not runs:
        return
    best_by_distance = min(runs, key=lambda r: r.test_median_km)
    best_by_top1 = max(runs, key=lambda r: r.test_top1)
    picked = runs[0]

    print("\nModel pick:")
    if objective == "top1":
        print(
            f"- Objective is top1. Pick `{picked.run}` "
            f"(test_top1={picked.test_top1:.4f}, median_km={picked.test_median_km:.2f})."
        )
    else:
        print(
            f"- Objective is distance. Pick `{picked.run}` "
            f"(median_km={picked.test_median_km:.2f}, acc50={picked.test_acc_50km:.4f}, top1={picked.test_top1:.4f})."
        )
    print(
        f"- Best by distance: `{best_by_distance.run}` ({best_by_distance.test_median_km:.2f} km)."
    )
    print(f"- Best by top1: `{best_by_top1.run}` ({best_by_top1.test_top1:.4f}).")

    print("\nNeed more data signals:")
    best = best_by_distance if objective == "distance" else best_by_top1
    if best.test_acc_50km < 0.45:
        print("- `acc_50km` is below 0.45: likely still data-limited for close-range precision.")
    else:
        print("- `acc_50km` is reasonably strong for current setup.")
    if best.test_median_km > 100:
        print("- Median error > 100 km: expand geographic coverage and/or increase per-cell samples.")
    else:
        print("- Median error <= 100 km: next gains may come more from model/hyperparameter tuning.")
    overfit_runs = [r for r in runs if r.generalization_gap is not None and r.generalization_gap > 0.15]
    if overfit_runs:
        names = ", ".join(r.run for r in overfit_runs)
        print(f"- Overfitting signal (train-val gap > 0.15) in: {names}.")
    else:
        print("- No severe overfitting signal from train/val gap.")


def write_csv_report(path: Path, runs: List[RunMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "run",
        "model",
        "dataset_csv",
        "device",
        "epochs",
        "best_epoch",
        "train_size",
        "val_size",
        "test_size",
        "test_top1",
        "test_median_km",
        "test_acc_25km",
        "test_acc_50km",
        "test_acc_200km",
        "test_acc_750km",
        "val_top1",
        "best_selection_metric",
        "generalization_gap",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, r in enumerate(runs, start=1):
            writer.writerow(
                {
                    "rank": idx,
                    "run": r.run,
                    "model": r.model,
                    "dataset_csv": r.dataset_csv,
                    "device": r.device,
                    "epochs": r.epochs,
                    "best_epoch": r.best_epoch,
                    "train_size": r.train_size,
                    "val_size": r.val_size,
                    "test_size": r.test_size,
                    "test_top1": r.test_top1,
                    "test_median_km": r.test_median_km,
                    "test_acc_25km": r.test_acc_25km,
                    "test_acc_50km": r.test_acc_50km,
                    "test_acc_200km": r.test_acc_200km,
                    "test_acc_750km": r.test_acc_750km,
                    "val_top1": r.val_top1,
                    "best_selection_metric": r.best_selection_metric,
                    "generalization_gap": r.generalization_gap,
                }
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare geobot training run metrics and suggest next steps.")
    ap.add_argument("--models-dir", default="models", help="Directory containing model run subdirectories.")
    ap.add_argument(
        "--processed-dir",
        default="data/processed/mapillary",
        help="Directory containing dataset summary JSON files.",
    )
    ap.add_argument(
        "--objective",
        choices=["distance", "top1"],
        default="distance",
        help="Ranking objective.",
    )
    ap.add_argument(
        "--csv-out",
        default="models/metrics_leaderboard.csv",
        help="Path to write CSV leaderboard.",
    )
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    processed_dir = Path(args.processed_dir)
    csv_out = Path(args.csv_out)

    runs = collect_runs(models_dir)
    ranked = sort_runs(runs, objective=args.objective)
    print_table(ranked)
    summaries = load_dataset_summaries(processed_dir)
    print_dataset_context(ranked, summaries)
    print_recommendations(ranked, objective=args.objective)
    write_csv_report(csv_out, ranked)
    print(f"\nCSV leaderboard written to: {csv_out}")


if __name__ == "__main__":
    main()
