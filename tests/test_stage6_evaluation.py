import csv
import json

from src.evaluation.test_evaluation import save_test_artifacts


def test_test_evaluation_saves_metrics_and_predictions(tmp_path) -> None:
    rows = [{"frame_path": "real.jpg", "label": "0"}, {"frame_path": "fake.jpg", "label": "1"}]
    metrics_path = tmp_path / "metrics.json"
    predictions_path = tmp_path / "predictions.csv"
    metrics = save_test_artifacts(rows, [0, 1], [0.1, 0.9], metrics_path, predictions_path, "best.pt", "test.csv")
    assert metrics["pr_auc"] == 1.0
    assert json.loads(metrics_path.read_text())["confusion_matrix"] == [[1, 0], [0, 1]]
    with predictions_path.open(newline="") as handle:
        predictions = list(csv.DictReader(handle))
    assert predictions[1]["prediction"] == "1"
