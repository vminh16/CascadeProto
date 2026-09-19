"""SUM-1..3 (05 §3.8c, gate G1): the phase-14 summary reads results correctly and keeps the paper's numbers."""

import json
import os
import statistics

from experiments import phase14, summarize


def write(save, run, which, protocol, miou, dry_run=False):
    path = phase14.result_path(run, str(save), which, protocol)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"miou": miou, "protocol": protocol, "dry_run": dry_run}, f)


def find(name):
    return [r for r in phase14.all_runs() if r.name == name][0]


def row_line(text, row):
    return [line for line in text.splitlines() if line.startswith(f"| {row} |")]


def test_sum1_paper_numbers_are_consistent():
    """Every printed Avg is the rounded mean of S0 and S1; shared cells carry the same numbers."""
    for table in summarize.PAPER.values():
        for s0, s1, avg in table.values():
            assert abs((s0 + s1) / 2 - avg) <= 0.0051
    p = summarize.PAPER
    assert p["T4"]["full"] == p["T5"]["T4"] == p["T2"]["N2K1"] == (88.53, 84.53, 86.53)
    assert set(p["T4"]) == {"baseline", "lma", "gate", "cascade", "full"}
    assert set(p["T5"]) == {f"T{t}" for t in range(1, 7)}


def test_sum2_cells_average_and_difference(tmp_path):
    write(tmp_path, find("full_S0_N2K1"), "best", "fixed100", 0.8800)
    write(tmp_path, find("full_S0_N2K1"), "last", "fixed100", 0.8700)
    write(tmp_path, find("full_S1_N2K1"), "best", "fixed100", 0.8400)
    write(tmp_path, find("lma_S0_N2K1"), "best", "fixed100", 0.8000, dry_run=True)  # ignored
    text = summarize.report(str(tmp_path), "fixed100")
    t4 = text.split("### Table 4")[1].split("###")[0]
    assert row_line(t4, "full") == ["| full | 88.00 | 87.00 | 84.00 | - | 86.00 | 88.53 / 84.53 / 86.53 | -0.53 |"]
    assert row_line(t4, "lma")[0].startswith("| lma | - | - | - | - | - |")
    t5 = text.split("### Table 5")[1].split("###")[0]
    assert row_line(t5, "T4")[0].startswith("| T4 | 88.00 | 87.00 | 84.00 |")  # the same run fills Table 5
    random = summarize.report(str(tmp_path), "random600")
    assert row_line(random.split("### Table 4")[1], "full")[0].startswith("| full | - |")  # protocols separate


def test_sum3_seed_spread(tmp_path):
    values = {0: 0.88, 1: 0.86, 2: 0.87}
    for seed, v in values.items():
        write(tmp_path, find("full_S0_N2K1" + (f"_seed{seed}" if seed else "")), "best", "fixed100", v)
    text = summarize.seed_markdown(str(tmp_path), "fixed100")
    pct = [100 * v for v in values.values()]
    assert f"mean {statistics.mean(pct):.2f}, std {statistics.stdev(pct):.2f} over 3 seeds" in text
    assert "seed 1: 86.00" in text
    assert "needs at least two" in summarize.seed_markdown(str(tmp_path / "empty"), "fixed100")
