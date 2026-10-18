"""Sparse-history checks for the ECM comparison and submission heatmap."""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.image as mpimg

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ecm_report"))

from plot_changes import (  # noqa: E402
    compress_progress_changes,
    cumulative_curves,
    draw_progress_change,
    draw_submission_heatmap,
    heatmap_height_px,
    heatmap_width_px,
    heatmap_xticks,
    history_dates,
    parse_date,
    progress_changes,
    progress_start_date,
    selected_heatmap_dates,
    submission_heatmap,
)


class ReportPlotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.conn.executescript(
            """
            CREATE TABLE progress_history (
                exponent INTEGER, report_date TEXT, t_level INTEGER, curves INTEGER
            );
            CREATE TABLE exponents (exponent INTEGER, factored INTEGER);
            INSERT INTO progress_history VALUES
                (1001, '20261001', 25, 0),
                (2500, '20261001', 30, 10),
                (4000, '20261001', 25, 0),
                (1001, '20261002', 25, 100),
                (1200, '20261002', 25, 10),
                (2500, '20261002', 30, 5),
                (1001, '20261003', 30, 20),
                (2500, '20261003', 30, 25);
            """
        )

    def tearDown(self) -> None:
        self.conn.close()

    def test_level_boundary_and_null_mean_completed_curves(self) -> None:
        self.assertEqual(cumulative_curves(25, 100), 100)
        self.assertEqual(cumulative_curves(25, None), 280)
        self.assertEqual(cumulative_curves(30, 20), 300)
        self.assertEqual(cumulative_curves(30, None), 920)
        with self.assertRaises(ValueError):
            cumulative_curves(30, -1)

    def test_two_date_plot_uses_as_of_sparse_state(self) -> None:
        rows = progress_changes(self.conn, 1000, 2999, "20261001", "20261003")
        self.assertEqual([row[0] for row in rows], [1001, 1200, 2500])
        self.assertIsNone(rows[1][1])
        self.assertGreater(rows[0][2], rows[0][1])
        self.assertGreater(rows[2][2], rows[2][1])
        self.assertEqual(history_dates(self.conn), ["20261001", "20261002", "20261003"])

    def test_heatmap_counts_positive_events_and_excludes_first_seen(self) -> None:
        cells, corrections, first_seen = submission_heatmap(
            self.conn, 1000, 2999, ["20261002", "20261003"], "20261003", 1000
        )
        self.assertEqual(cells.tolist(), [[100.0, 0.0], [200.0, 20.0]])
        self.assertEqual(corrections, 1)
        self.assertEqual(first_seen, 1)

    def test_heatmap_date_window_height_and_fine_exponent_bins(self) -> None:
        dates = [(datetime(2026, 8, 1) + timedelta(days=i)).strftime("%Y%m%d")
                 for i in range(80)]
        self.assertEqual(heatmap_height_px(dates[:30], 800, 60), 400)
        self.assertEqual(heatmap_height_px(dates[:60], 800, 60), 800)
        self.assertEqual(heatmap_height_px(dates, 800, 60), 800)
        self.assertEqual(heatmap_width_px(200, 1000), 1000)
        self.assertEqual(heatmap_width_px(2, 2800), 920)
        with self.assertRaises(ValueError):
            heatmap_width_px(200, 499)
        self.assertEqual(selected_heatmap_dates(dates, None, dates[-1], 60), dates[-60:])
        cells, _, _ = submission_heatmap(
            self.conn, 1000, 2999, history_dates(self.conn), "20261003", 100
        )
        self.assertEqual(cells.shape, (3, 20))
        self.assertEqual(cells.sum(), 320)

    def test_exponent_ticks_and_progress_lookback(self) -> None:
        dates = history_dates(self.conn)
        self.assertEqual(progress_start_date(dates, None, "20261003", None), "20261002")
        self.assertEqual(progress_start_date(dates, None, "20261003", 1), "20261002")
        self.assertEqual(progress_start_date(dates, None, "20261003", 2), "20261001")
        self.assertEqual(progress_start_date(dates, "20261002", "20261003", 2), "20261002")
        self.assertEqual(heatmap_xticks(1, 2000, 100, 1000),
                         ([0, 9, 19], ["1", "1000", "2000"]))
        positions, labels = heatmap_xticks(1, 2000, 100, "auto")
        self.assertEqual(len(positions), 8)
        self.assertEqual((labels[0], labels[-1]), ("1", "2000"))

    def test_change_compression_keeps_first_seen_and_opposing_changes(self) -> None:
        rows = [(1001, 20.0, 21.0), (1200, None, 25.0),
                (2500, 30.0, 28.0), (2700, 29.0, 31.0),
                (3000, None, 40.0)]
        buckets = compress_progress_changes(rows, 2)
        self.assertEqual((buckets[0].exp_rep, buckets[0].old,
                          buckets[0].new, buckets[0].first_y),
                         (1001, 20.0, 21.0, 25.0))
        self.assertEqual((buckets[1].old, buckets[1].new,
                          buckets[1].has_gain, buckets[1].has_loss),
                         (29.5, 29.5, True, True))
        self.assertEqual((buckets[2].old, buckets[2].new, buckets[2].first_y),
                         (None, 40.0, 40.0))
        maximum = compress_progress_changes(rows, 2, "max")
        self.assertEqual((maximum[1].old, maximum[1].new), (30.0, 31.0))
        with self.assertRaises(ValueError):
            compress_progress_changes(rows, 0)

    def test_rendered_charts(self) -> None:
        rows = progress_changes(self.conn, 1000, 2999, "20261001", "20261003")
        dates = history_dates(self.conn)
        cells, _, _ = submission_heatmap(self.conn, 1000, 2999, dates, dates[-1], 1000)
        evidence_dir = Path(__file__).resolve().parents[2] / "data" / "experiments"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=evidence_dir) as temp:
            progress = Path(temp) / "progress.png"
            heatmap = Path(temp) / "heatmap.png"
            draw_progress_change(rows, 1000, 2999, dates[0], dates[-1], progress,
                                 80, compress=2)
            draw_submission_heatmap(cells, dates, 1000, 2999, 1000, "curves", heatmap,
                                    80, max_width_px=800)
            self.assertEqual(progress.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(heatmap.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(mpimg.imread(heatmap).shape[0], 320)
            self.assertEqual(mpimg.imread(heatmap).shape[1], 800)

    def test_dates_require_eight_real_digits(self) -> None:
        self.assertEqual(parse_date("20261018"), "20261018")
        for value in ("2026101", "20261301", "20260230", "2026-10-18"):
            with self.assertRaises(Exception):
                parse_date(value)


if __name__ == "__main__":
    unittest.main()
