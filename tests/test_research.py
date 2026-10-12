import hashlib
import json
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from PIL import Image

from escount.research.daemon import (
    TSV_COLUMNS, Researcher, TrialBudget, adopt, keep_or_discard, load_study, read_tsv,
)
from escount.research.evaluate import rank_key
from escount.models.train import load_general_config
from escount.research.report import CASE_FILES, COMPARISON_FILES, build_report, default_cases
from feature_tables import synthetic_table

SEEDS = {"P1": 0, "P2": 1}


def synthetic_fn(projects, grid):
    """Stand-in for the real feature computation: deterministic synthetic tables per project."""
    table = pd.concat([synthetic_table(cases=(p,), images_per_case=4, rows_per_image=12, seed=SEEDS[p])
                       for p in projects], ignore_index=True)
    images = table.drop_duplicates("image_id")
    expected = [{"image_id": r.image_id, "case": r.case, "image": r.image, "gt": r.gt} for r in images.itertuples()]
    return table, expected


def researcher(root, cases):
    return Researcher(["P1", "P2"], cases, storage=f"sqlite:///{(root / 'study.db').as_posix()}",
                      research_root=root / "research", candidates_root=root / "candidates",
                      profiles_root=root / "profiles", config_path=root / "missing_general.json", device="cpu",
                      budget=TrialBudget(max_rows_per_image=50, n_estimators=20, svr_max_rows=500),
                      table_fn=synthetic_fn, log=lambda *_: None)


def tree_hashes(folder):
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(folder).rglob("*")) if p.is_file()}


class ImageCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images, self.links = [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "img":
            self.images.append(attrs.get("src", ""))
        for key in ("src", "href"):
            if key in attrs and not attrs[key].startswith("data:image/png;base64,"):
                self.links.append((tag, key, attrs[key]))


class KeepDiscardTests(unittest.TestCase):
    def test_rank_key_decides_keep(self):
        best = SimpleNamespace(hit_rate=0.5, roi_count=3.0, latency_ms=1.0)
        self.assertEqual(keep_or_discard(best, None), "keep")
        self.assertEqual(keep_or_discard(SimpleNamespace(hit_rate=0.6, roi_count=9, latency_ms=9), best), "keep")
        self.assertEqual(keep_or_discard(SimpleNamespace(hit_rate=0.4, roi_count=1, latency_ms=0), best), "discard")
        # equal hit rate: fewer ROIs wins even with a higher latency
        self.assertEqual(keep_or_discard(SimpleNamespace(hit_rate=0.5, roi_count=2, latency_ms=9), best), "keep")
        self.assertEqual(keep_or_discard(SimpleNamespace(hit_rate=0.5, roi_count=4, latency_ms=0), best), "discard")
        # equal hit rate and ROIs: lower latency wins; an exact tie is discarded
        self.assertEqual(keep_or_discard(SimpleNamespace(hit_rate=0.5, roi_count=3, latency_ms=0.5), best), "keep")
        self.assertEqual(keep_or_discard(SimpleNamespace(hit_rate=0.5, roi_count=3, latency_ms=1.0), best), "discard")


class DaemonTests(unittest.TestCase):
    def test_resume_keeps_trial_numbering_contiguous(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            self.assertEqual(researcher(root, ["P1"]).run(trials=3), 3)
            self.assertEqual(researcher(root, ["P1"]).run(trials=2), 2)       # reopened study
            study = load_study("P1", f"sqlite:///{(root / 'study.db').as_posix()}")
            self.assertEqual([t.number for t in study.trials], [0, 1, 2, 3, 4])
            tsv_path = root / "research" / "P1" / "results.tsv"
            self.assertEqual(tsv_path.read_text(encoding="utf-8").splitlines()[0], "\t".join(TSV_COLUMNS))
            tsv = read_tsv(tsv_path)
            self.assertEqual(tsv["trial"].tolist(), [0, 1, 2, 3, 4])
            self.assertLessEqual(set(tsv["status"]), {"keep", "discard"})
            # keep rows are exactly the strict rank-key improvements, in trial order
            best = None
            for _, row in tsv.iterrows():
                hits, total = map(int, row["hits/total"].split("/"))
                result = SimpleNamespace(hit_rate=hits / total, roi_count=float(row["roi_count"]),
                                         latency_ms=float(row["latency_ms"]))
                expected = "keep" if best is None or rank_key(result) < rank_key(best) else "discard"
                self.assertEqual(row["status"], expected)
                best = result if expected == "keep" else best
            candidate = json.loads((root / "candidates" / "P1" / "best_params.json").read_text(encoding="utf-8"))
            self.assertEqual(candidate["case"], "P1")
            self.assertIn("grid", candidate["extraction"])
            self.assertGreater(candidate["geometry"]["D"] * 3, 2 * candidate["geometry"]["w"])

    def test_production_profile_unchanged_until_adopt(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            profile = root / "profiles" / "P1"
            profile.mkdir(parents=True)
            (profile / "project.json").write_text('{"case": "P1"}', encoding="utf-8")
            before = tree_hashes(profile)
            researcher(root, ["P1"]).run(trials=2)
            self.assertEqual(tree_hashes(profile), before)
            target = adopt("P1", root / "candidates", root / "profiles")
            self.assertNotEqual(tree_hashes(profile), before)
            self.assertEqual((target / "best_params.json").read_bytes(),
                             (root / "candidates" / "P1" / "best_params.json").read_bytes())


class ReportTests(unittest.TestCase):
    def test_report_has_every_file_and_is_self_contained(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            researcher(root, ["P1", "general"]).run(trials=3)
            cases = ["P1", "P2", "general"]            # P2 has no trials: placeholders, files still exist
            out = build_report(root / "report", cases, root / "research", root / "candidates",
                               f"sqlite:///{(root / 'study.db').as_posix()}", root / "profiles",
                               root / "missing_general.json", log=lambda *_: None)
            for case in cases:
                for name in CASE_FILES:
                    self.assertTrue((out / case / name).is_file(), f"{case}/{name}")
                self.assertEqual((out / case / "results.tsv").read_text(encoding="utf-8").splitlines()[0],
                                 "\t".join(TSV_COLUMNS))
            for name in COMPARISON_FILES:
                self.assertTrue((out / "comparison" / name).is_file(), name)
            self.assertFalse((out / "comparison" / "ring_inclusion.png").exists())   # no general_no_ring case
            self.assertTrue((out / "summary.md").is_file())
            pngs = sorted(out.rglob("*.png"))
            self.assertEqual(len(pngs), 3 * len(cases) + 4)
            for path in pngs:
                with Image.open(path) as image:
                    self.assertEqual(image.size, (3200, 1800), path)
                    self.assertAlmostEqual(image.info["dpi"][0], 200, delta=0.5)
            per_image = pd.read_csv(out / "general" / "per_image.csv")
            self.assertEqual(list(per_image.columns), ["project", "image", "GT", "prediction", "hit", "scheme", "fold"])
            self.assertEqual(set(per_image["scheme"]), {"LOIO", "LOPO"})
            parser = ImageCollector()
            parser.feed((out / "report.html").read_text(encoding="utf-8"))
            self.assertEqual(len(parser.images), len(pngs))
            self.assertTrue(all(src.startswith("data:image/png;base64,") for src in parser.images))
            self.assertEqual(parser.links, [])
            latest = root / "report" / "latest"
            self.assertEqual(tree_hashes(latest), tree_hashes(out))


class ExclusionTests(unittest.TestCase):
    def test_excluded_project_is_not_researched_or_reported(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            config = root / "general.json"
            config.write_text(json.dumps({"excluded_projects": ["Ring-type"], "exclude_from_general": ["Ring-type"],
                                          "excluded_reason": {"Ring-type": "LOIO 3/6"}}), encoding="utf-8")
            daemon = Researcher(["P1", "P2", "Ring-type"], None, storage=f"sqlite:///{(root / 'study.db').as_posix()}",
                                research_root=root / "research", candidates_root=root / "candidates",
                                profiles_root=root / "profiles", config_path=config, device="cpu",
                                table_fn=synthetic_fn, log=lambda *_: None)
            self.assertEqual(daemon.projects, ["P1", "P2"])
            self.assertEqual(daemon.cases, ["P1", "P2", "general"])
            self.assertEqual(daemon.members["general"], ["P1", "P2"])
            with self.assertRaises(ValueError):
                Researcher(["P1", "Ring-type"], ["Ring-type"], config_path=config, table_fn=synthetic_fn)
            self.assertEqual(default_cases(load_general_config(config)),
                             ["20260904-thinkness02_postition1", "20260904-thinkness02_postition2", "T-type", "general"])
            out = build_report(root / "report", None, root / "research", root / "candidates",
                               f"sqlite:///{(root / 'study.db').as_posix()}", root / "profiles", config,
                               log=lambda *_: None)
            self.assertEqual(sorted(p.name for p in out.iterdir() if p.is_dir()),
                             ["20260904-thinkness02_postition1", "20260904-thinkness02_postition2", "T-type",
                              "comparison", "general"])
            self.assertFalse((out / "comparison" / "ring_inclusion.png").exists())
            summary = (out / "summary.md").read_text(encoding="utf-8")
            self.assertIn("Ring-type 已依使用者決定排除", summary)
            self.assertIn("LOIO 3/6", summary)
            self.assertNotIn("general_no_ring |", summary)

    def test_resume_with_different_projects_is_refused(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            researcher(root, ["general"]).run(trials=1)          # general = P1 + P2
            with self.assertRaises(ValueError):
                Researcher(["P1"], ["general"], storage=f"sqlite:///{(root / 'study.db').as_posix()}",
                           research_root=root / "research", candidates_root=root / "candidates",
                           config_path=root / "missing_general.json", table_fn=synthetic_fn)


if __name__ == "__main__":
    unittest.main()
