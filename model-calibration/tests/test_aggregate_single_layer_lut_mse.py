import csv
import json

from examples.aggregate_single_layer_lut_mse import aggregate_results


def test_aggregate_results_sorts_by_k_and_v(tmp_path):
    (tmp_path / "b.json").write_text(json.dumps({"ncentroid": 16, "vec_len": 4, "mse": 2.0}), encoding="utf-8")
    (tmp_path / "a.json").write_text(json.dumps({"ncentroid": 2, "vec_len": 8, "mse": 1.0}), encoding="utf-8")

    output_csv = tmp_path / "results.csv"
    rows = aggregate_results(str(tmp_path), str(output_csv))

    assert [(row["ncentroid"], row["vec_len"]) for row in rows] == [(2, 8), (16, 4)]
    written = list(csv.DictReader(output_csv.open()))
    assert written[0]["mse"] == "1.0"
