"""Trained synthetic projects for the server / UI API tests (calibrate -> prepare -> train -> select)."""
from pathlib import Path

import cv2
import numpy as np

from escount.calibration.calibrate import calibrate_project
from escount.features.cache import prepare
from escount.features.extract import AugmentGrid
from escount.models.select_rois import select_general, select_project
from escount.models.train import train_general, train_project
from test_inference import HEIGHT, WIDTH, stack_image

CASES = {"SynA": (70, 76, 82, 88), "SynB": (72, 78, 84, 90)}
GRID = AugmentGrid(widths=(10, 20), pitches=(30,), x_offsets=(0,), projections=("mean", "gaussian"))


def build(root, general=False):
    """Write Data/<case>/ images + mask under `root` and train every case (and General if asked).

    Returns (data_root, profiles_root, features_root, general_config path).
    """
    root = Path(root)
    data, profiles, features = root / "Data", root / "projects", root / "features"
    quiet = lambda *args, **kwargs: None
    cases = CASES if general else dict(list(CASES.items())[:1])
    for offset, (case, counts) in enumerate(cases.items()):
        case_dir = data / case
        case_dir.mkdir(parents=True)
        cv2.imwrite(str(case_dir / "background_BW.png"), np.zeros((HEIGHT, WIDTH), dtype=np.uint8))
        for seed, sheets in enumerate(counts):
            cv2.imwrite(str(case_dir / f"{sheets:03d}.png"), stack_image(sheets, 10 * offset + seed))
        calibrate_project(case_dir, profiles)
        prepare(case, data, profiles, features, workers=1, grid=GRID)
        train_project(case, features, profiles, device="cpu", log=quiet)
        select_project(case, features, profiles, device="cpu")
    config = root / "general.json"
    config.write_text('{"exclude_from_general": []}\n', encoding="utf-8")
    if general:
        train_general(features, profiles, config, device="cpu", log=quiet)
        select_general(features, profiles, device="cpu")
    return data, profiles, features, config
