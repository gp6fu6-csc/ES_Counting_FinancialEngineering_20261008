"""Synthetic feature tables with the full column set of escount.features.extract (test helper)."""
import numpy as np
import pandas as pd

from escount.features.extract import count_columns


def synthetic_table(cases=("A",), images_per_case=8, rows_per_image=24, seed=0, learnable=True):
    """Rows with a learnable delta = GT - period_count (snr regime) and biased count columns.

    Per image: snr is 8 (delta +3) or 2 (delta -2), alternating; every count column = GT + an image
    offset of +-2, so the single-indicator baseline misses every image while a model can learn delta.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for case in cases:
        for index in range(images_per_case):
            gt = int(rng.integers(40, 160))
            high = index % 2 == 0
            delta = 3 if high else -2
            offset = 2 if rng.random() < 0.5 else -2
            for slot in range(rows_per_image):
                w = (10, 20)[slot % 2]
                projection = ("mean", "median", "gaussian")[slot % 3]
                x = 100 + 30 * (slot // 2)
                row = {"case": case, "image": f"{gt:03d}_{index}.png", "image_id": f"{case}/{gt:03d}_{index}.png",
                       "gt": gt, "gt_group": 1, "w": w, "pitch": 30, "x_offset": 0, "x": x, "y0": 100, "y1": 900,
                       "projection": projection, "proj_id": ("mean", "median", "gaussian").index(projection),
                       "sigma": 0.25 * w if projection == "gaussian" else np.nan,
                       "slot_key": f"{projection}|{w}|{x}", "pos_key": f"{projection}|{w}|{x / 1000:.2f}",
                       "L": 800, "x_norm": x / 1000.0, "period": 800.0 / (gt - delta),
                       "period_count": gt - delta, "snr": (8.0 if high else 2.0) if learnable else rng.random(),
                       "std": 20.0 + rng.random(), "period_stability": 0.01 * rng.random()}
                for column in count_columns():
                    row[column] = float(gt + offset)
                rows.append(row)
    return pd.DataFrame(rows)
