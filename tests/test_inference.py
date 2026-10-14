import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from escount.calibration.calibrate import calibrate_project
from escount.calibration.rotation_correction import rotate_image
from escount.cli import main as cli_main
from escount.es_counting import tree_signatures
from escount.features.cache import prepare
from escount.features.extract import AugmentGrid
from escount.inference.engine import OUTPUT_FILES, InferenceEngine, OutputOptions
from escount.inference.plots import analysis_figure
from escount.models.select_rois import select_project
from escount.models.train import train_project

PERIOD, TOP, HEIGHT, WIDTH, ANGLE = 12, 60, 1200, 1400, 0.6
COUNTS = (70, 76, 82, 88)


def stack_image(sheets, seed, angle=ANGLE):
    """Horizontal sheet stack (period 12 px) tilted so that rotate_image(result, angle) straightens it."""
    pad = 200
    image = np.full((HEIGHT + 2 * pad, WIDTH + 2 * pad), 15.0, dtype=np.float32)
    rows = np.arange(pad + TOP, pad + TOP + sheets * PERIOD)
    image[rows] = np.where((rows - pad - TOP) % PERIOD < 8, 170.0, 50.0)[:, None]
    tilted = rotate_image(image, -angle)[pad:pad + HEIGHT, pad:pad + WIDTH]
    tilted = tilted + np.random.default_rng(seed).normal(0, 3, tilted.shape)
    return np.clip(tilted, 0, 255).astype(np.uint8)


class InferenceEndToEndTests(unittest.TestCase):
    """calibrate -> prepare -> train -> select-rois -> infer on a temporary data root."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = cls.root = Path(cls._tmp.name)
        cls.case_dir = root / "Data" / "Syn"
        cls.case_dir.mkdir(parents=True)
        cv2.imwrite(str(cls.case_dir / "background_BW.png"), np.zeros((HEIGHT, WIDTH), dtype=np.uint8))
        for seed, sheets in enumerate(COUNTS):
            cv2.imwrite(str(cls.case_dir / f"{sheets:03d}.png"), stack_image(sheets, seed))
        # unseen image (not used for training); even count because a 'both' counter can give N + 0.5 and
        # np.rint rounds ties to even
        cls.extra = root / "extra" / "080.png"
        cls.extra.parent.mkdir()
        cv2.imwrite(str(cls.extra), stack_image(80, 11))
        cls.profiles, features = root / "projects", root / "features"
        calibrate_project(cls.case_dir, cls.profiles)
        grid = AugmentGrid(widths=(10, 20), pitches=(30,), x_offsets=(0,), projections=("mean", "gaussian"))
        prepare("Syn", root / "Data", cls.profiles, features, workers=1, grid=grid)
        quiet = lambda *args, **kwargs: None
        train_project("Syn", features, cls.profiles, device="cpu", log=quiet)
        select_project("Syn", features, cls.profiles, device="cpu")
        cls.before = tree_signatures(root / "Data")
        cls.engine = InferenceEngine.load("Syn", profiles_root=cls.profiles)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_cli_folder_writes_every_output_and_predicts_counts(self):
        out = self.root / "out_folder"
        status = cli_main(["infer", "--project", "Syn", "--input", str(self.case_dir), "--output", str(out),
                           "--profiles-root", str(self.profiles)])
        self.assertEqual(status, 0)
        table = pd.read_csv(out / "predictions.csv")
        self.assertEqual(list(table.columns), ["image", "GT", "N_pred", "hit"])
        self.assertEqual(len(table), len(COUNTS))
        self.assertEqual(list(table["N_pred"]), list(table["GT"]))
        self.assertTrue(table["hit"].all())
        for sheets in COUNTS:
            image_dir = out / f"{sheets:03d}"
            self.assertEqual(sorted(p.name for p in image_dir.iterdir()), sorted(OUTPUT_FILES.values()))
            result = json.loads((image_dir / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["N_pred"], sheets)
            self.assertEqual(set(result["timings_ms"]),
                             {"read", "rotate", "extent", "features", "model", "aggregate", "total"})
            self.assertEqual(result["source_sha256_before"], result["source_sha256_after"])
            self.assertTrue(result["source_unchanged"])
            self.assertGreaterEqual(len(result["rois"]), 1)
            signals = pd.read_csv(image_dir / "signals.csv")
            self.assertEqual(len(signals.columns), 1 + len(result["rois"]))
        self.assertEqual(tree_signatures(self.root / "Data"), self.before)

    def test_single_file_input_and_unseen_image(self):
        out = self.root / "out_single"
        status = cli_main(["infer", "--project", "Syn", "--input", str(self.extra), "--output", str(out),
                           "--profiles-root", str(self.profiles), "--no-plots"])
        self.assertEqual(status, 0)
        self.assertEqual([p.name for p in out.iterdir() if p.is_dir()], ["080"])
        self.assertFalse((out / "080" / "analysis.png").exists())
        self.assertEqual(pd.read_csv(out / "predictions.csv")["N_pred"].tolist(), [80])
        # array input gives the same count
        result = self.engine.infer(cv2.imread(str(self.extra), cv2.IMREAD_UNCHANGED), name="080.png")
        self.assertEqual(result["N_pred"], 80)
        self.assertTrue(result["hit"])

    def test_each_toggle_removes_only_its_output(self):
        image = self.case_dir / "076.png"
        for option, name in OUTPUT_FILES.items():
            out = self.root / f"toggle_{option}"
            self.engine.run(image, out, OutputOptions(**{option: False}))
            present = {p.name for p in (out / "076").iterdir()}
            self.assertNotIn(name, present, option)
            self.assertEqual(present, set(OUTPUT_FILES.values()) - {name}, option)
            self.assertTrue((out / "predictions.csv").is_file())
        self.assertEqual(tree_signatures(self.root / "Data"), self.before)

    def test_output_inside_input_folder_is_refused(self):
        with self.assertRaises(ValueError):
            self.engine.run(self.case_dir, self.case_dir / "out")
        self.assertEqual(tree_signatures(self.root / "Data"), self.before)


class OptionAndPlotTests(unittest.TestCase):
    def test_dpi_minimum_64(self):
        with self.assertRaises(ValueError):
            OutputOptions(dpi=63)
        self.assertEqual(OutputOptions(dpi=64).dpi, 64)
        with self.assertRaises(SystemExit):
            cli_main(["infer", "--project", "Syn", "--input", ".", "--dpi", "63"])

    def test_analysis_grid_at_most_four_columns_with_titles_and_marks(self):
        rows = np.arange(100, 400)
        signal = np.sin(2 * np.pi * rows / 12.0)
        panels = [{"rows": rows, "signal": signal, "high_idx": [3, 15], "low_idx": [9, 21], "prediction": 25.0 + i,
                   "label": f"mean w10 x{i}"} for i in range(6)]
        figure = analysis_figure(panels, "title", dpi=64)
        visible = [ax for ax in figure.axes if ax.get_visible()]
        self.assertEqual(len(visible), 6)
        for index, ax in enumerate(visible, start=1):
            self.assertLessEqual(ax.get_subplotspec().get_gridspec().ncols, 4)
            self.assertTrue(ax.get_title().startswith(f"N{index}_pred"))
            self.assertEqual(len(ax.lines), 3)                         # signal + highs + lows
        self.assertEqual(figure.axes[0].get_subplotspec().get_gridspec().nrows, 2)


if __name__ == "__main__":
    unittest.main()
