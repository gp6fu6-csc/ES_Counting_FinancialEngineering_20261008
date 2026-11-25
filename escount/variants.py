"""ROI-strip variants ("方案") of a project and the saved training runs of each variant.

A variant only owns the counting strips (strip_params: N, w, pitch, side_margin, x_offset). The mask,
the rotation ROIs and the angle stay shared with the base project, so recalibrating the base is
picked up by every variant (sync_variant re-copies them before each CLI step).

Case ids (used wherever a case is a folder name under profiles_root / cache_root / results_root):

    <base>              the project itself (default strips in <base>/project.json), unchanged behaviour
    <base>@<variant>    a variant: its own profile folder, model/, feature cache and results folder
    <base>@<variant>~<run>   one saved training run of that variant (read-only model snapshot)

Folder layout (profiles_root = outputs/projects):

    <base>@<variant>/project.json, background_BW_rotated.png, model/     the latest training
    <base>@<variant>/runs/<run>/run.json, model/                         every `run-variants` training kept
"""
import csv
import re
import shutil
from datetime import datetime
from pathlib import Path

from escount.calibration.calibrate import PROFILE_NAME, ROTATED_MASK_NAME, ProjectProfile
from escount.es_counting import read_json, write_json
from escount.height import MODEL_NAME as HEIGHT_MODEL_NAME
from escount.signals.roi_bank import check_geometry

VARIANT_SEP = "@"
RUN_SEP = "~"
RUNS_DIR = "runs"
RUN_FILE = "run.json"
STRIP_KEYS = ("n", "w", "pitch", "side_margin", "x_offset")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$")
RUN_SCHEMA_VERSION = 1


def split_case(case):
    """'base@variant~run' -> (base, variant or None, run or None)."""
    head, _, run = str(case).partition(RUN_SEP)
    base, _, variant = head.partition(VARIANT_SEP)
    return base, variant or None, run or None


def base_case(case):
    return split_case(case)[0]


def profile_case(case):
    """The case id of the profile / feature cache folder: 'base' or 'base@variant' (run suffix dropped)."""
    base, variant, _ = split_case(case)
    return f"{base}{VARIANT_SEP}{variant}" if variant else base


def variant_case(base, variant):
    return f"{base}{VARIANT_SEP}{variant}" if variant else base


def check_name(name):
    if not NAME_RE.match(name or ""):
        raise ValueError(f"invalid variant name {name!r}: use 1-40 letters, digits, '_' or '-' (first char a letter/digit)")
    return name


def normalize_strips(strips):
    """Validated strip_params dict (int values, the agreed ROI geometry rules)."""
    missing = [key for key in STRIP_KEYS if key not in strips]
    if missing:
        raise ValueError(f"strip parameters missing: {missing}")
    values = {key: int(strips[key]) for key in STRIP_KEYS}
    if not 1 <= values["n"] <= 64:
        raise ValueError(f"strip count n must be in [1, 64], got {values['n']}")
    if values["x_offset"] < 0:
        raise ValueError(f"x_offset must be >= 0, got {values['x_offset']}")
    check_geometry(values["w"], values["pitch"], values["side_margin"])
    return values


def _copy_if_different(source, target):
    source, target = Path(source), Path(target)
    if source.is_file() and (not target.is_file() or target.read_bytes() != source.read_bytes()):
        shutil.copy2(source, target)


def sync_variant(case, profiles_root="outputs/projects", strips=None):
    """Create / refresh the profile folder of a variant from its base project. Returns the folder.

    Everything calibrated (rotation ROIs, angle, mask, valid region) is copied from <base>/project.json;
    the variant keeps its own strip_params and selected_rois. `strips` sets / replaces strip_params
    (a new variant without `strips` starts from the base's strips).
    """
    base, variant, _ = split_case(case)
    if variant is None:
        raise ValueError(f"{case!r} is not a variant case id (expected <project>{VARIANT_SEP}<variant>)")
    check_name(variant)
    root = Path(profiles_root)
    shared = root / base / PROFILE_NAME
    if not shared.is_file():
        raise FileNotFoundError(f"{shared} missing; run `escount calibrate --project {base}` first")
    folder = root / profile_case(case)
    own_path = folder / PROFILE_NAME
    own = ProjectProfile.load(own_path) if own_path.is_file() else None
    profile = ProjectProfile.load(shared)
    profile.case = profile_case(case)
    profile.selected_rois = list(own.selected_rois) if own else []
    if strips is not None:
        profile.strip_params = normalize_strips(strips)
    elif own is not None:
        profile.strip_params = dict(own.strip_params)
    folder.mkdir(parents=True, exist_ok=True)
    if not own_path.is_file() or own_path.read_text(encoding="utf-8") != profile.to_json():
        profile.save(own_path)
    _copy_if_different(root / base / ROTATED_MASK_NAME, folder / ROTATED_MASK_NAME)
    _copy_if_different(root / base / HEIGHT_MODEL_NAME, folder / HEIGHT_MODEL_NAME)
    return folder


def variant_names(base, profiles_root="outputs/projects"):
    root = Path(profiles_root)
    if not root.is_dir():
        return []
    return sorted(folder.name.split(VARIANT_SEP, 1)[1] for folder in root.glob(f"{base}{VARIANT_SEP}*")
                  if (folder / PROFILE_NAME).is_file())


def variant_strips(case, profiles_root="outputs/projects"):
    """strip_params of 'base' or 'base@variant' (None when the profile does not exist)."""
    path = Path(profiles_root) / profile_case(case) / PROFILE_NAME
    return dict(ProjectProfile.load(path).strip_params) if path.is_file() else None


# ---------------------------------------------------------------- saved training runs
def runs_root(case, profiles_root="outputs/projects"):
    return Path(profiles_root) / profile_case(case) / RUNS_DIR


def run_model_dir(case, run, profiles_root="outputs/projects"):
    return runs_root(case, profiles_root) / run / "model"


def new_run_id(case, profiles_root="outputs/projects", now=None):
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    run, number = stamp, 1
    while (runs_root(case, profiles_root) / run).exists():
        number += 1
        run = f"{stamp}-{number}"
    return run


def read_run(case, run, profiles_root="outputs/projects"):
    path = runs_root(case, profiles_root) / run / RUN_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found")
    return read_json(path)


def summarize_manifest(manifest):
    """The numbers a reader needs to compare runs (full detail stays in model/manifest.json)."""
    deployed = (manifest.get("selection") or {}).get("deployed") or {}
    ensemble = manifest.get("all_rois_ensemble") or {}
    recommended = ensemble.get("recommended")
    evaluation = (ensemble.get("evaluation") or {}).get(recommended) or {}
    return {"method": manifest.get("method"), "model_type": manifest.get("model_type"),
            "trained_rows": manifest.get("trained_rows"), "images": manifest.get("images"),
            "minimal": {"roi_count": len(manifest.get("selected_rois") or []), "hits": deployed.get("hits"),
                        "total": deployed.get("total"), "estimate": deployed.get("estimate")},
            "all_rois": {"roi_count": len(manifest.get("all_rois") or []), "recommended": recommended,
                         "hits": evaluation.get("hits"), "total": evaluation.get("total"),
                         "mae": evaluation.get("mae")}}


def run_label(record):
    """One-line text for the results-page dropdown."""
    model = record.get("model") or {}
    minimal, everything = model.get("minimal") or {}, model.get("all_rois") or {}

    def hits(part):
        return f"{part['hits']}/{part['total']}" if part.get("hits") is not None else "—"

    status = "" if record.get("status") == "ok" else f" [{record.get('status')}]"
    strips = record.get("strip_params") or {}
    return (f"{record['run']} · N={strips.get('n')} w={strips.get('w')} D={strips.get('pitch')} · "
            f"最少 ROI {hits(minimal)} · 全部 ROI {hits(everything)}{status}")


def snapshot_run(case, run, record, profiles_root="outputs/projects"):
    """Keep the freshly trained model/ of `case` as runs/<run>/model and write runs/<run>/run.json."""
    folder = runs_root(case, profiles_root) / run
    model = Path(profiles_root) / profile_case(case) / "model"
    if record.get("status") == "ok" and (model / "manifest.json").is_file():
        shutil.copytree(model, folder / "model", dirs_exist_ok=True)
    write_json(folder / RUN_FILE, {"schema_version": RUN_SCHEMA_VERSION, **record})
    return folder


SUMMARY_COLUMNS = (
    "project", "variant", "run", "created", "status", "note",
    "n", "w", "pitch", "side_margin", "x_offset",
    "device", "max_rois", "max_rows_per_image",
    "model_type", "method", "images",
    "minimal_rois", "minimal_hits", "minimal_total",
    "all_rois", "all_recommended", "all_hits", "all_total", "all_mae",
    "research_track", "research_trials", "research_status", "research_best_trial", "research_best_hit_rate",
    "research_best_w", "research_best_D", "research_best_model",
)


def summary_rows(projects, profiles_root="outputs/projects"):
    """One row per saved training run of every variant of `projects` (oldest first), for side-by-side comparison."""
    rows = []
    for base in projects:
        for name in variant_names(base, profiles_root):
            case = variant_case(base, name)
            for item in sorted(list_runs(case, profiles_root), key=lambda x: x["run"]):
                record = read_run(case, item["run"], profiles_root)
                strips, options = record.get("strip_params") or {}, record.get("options") or {}
                model = record.get("model") or {}
                minimal, everything = model.get("minimal") or {}, model.get("all_rois") or {}
                research = record.get("research") or {}
                best = research.get("best_params") or {}
                rows.append({
                    "project": base, "variant": name, "run": item["run"], "created": record.get("created"),
                    "status": record.get("status"), "note": record.get("note"),
                    "n": strips.get("n"), "w": strips.get("w"), "pitch": strips.get("pitch"),
                    "side_margin": strips.get("side_margin"), "x_offset": strips.get("x_offset"),
                    "device": options.get("device"), "max_rois": options.get("max_rois"),
                    "max_rows_per_image": options.get("max_rows_per_image"),
                    "model_type": model.get("model_type"), "method": model.get("method"), "images": model.get("images"),
                    "minimal_rois": minimal.get("roi_count"), "minimal_hits": minimal.get("hits"),
                    "minimal_total": minimal.get("total"),
                    "all_rois": everything.get("roi_count"), "all_recommended": everything.get("recommended"),
                    "all_hits": everything.get("hits"), "all_total": everything.get("total"),
                    "all_mae": everything.get("mae"),
                    "research_track": research.get("track") if research.get("trials") else None,
                    "research_trials": research.get("trials"), "research_status": research.get("status"),
                    "research_best_trial": best.get("trial"),
                    "research_best_hit_rate": (best.get("result") or {}).get("hit_rate"),
                    "research_best_w": (best.get("geometry") or {}).get("w"),
                    "research_best_D": (best.get("geometry") or {}).get("D"),
                    "research_best_model": (best.get("model") or {}).get("kind"),
                })
    return rows


def write_variants_summary(path, projects, profiles_root="outputs/projects"):
    """variants_summary.csv (UTF-8 with BOM so Excel shows the Chinese notes). Returns the row count."""
    rows = summary_rows(projects, profiles_root)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def list_runs(case, profiles_root="outputs/projects"):
    """Saved runs of a variant, newest first: [{run, label, status, created, has_model}]."""
    root = runs_root(case, profiles_root)
    items = []
    if root.is_dir():
        for folder in sorted((p for p in root.iterdir() if (p / RUN_FILE).is_file()), reverse=True):
            record = read_json(folder / RUN_FILE)
            items.append({"run": folder.name, "label": run_label({**record, "run": folder.name}),
                          "status": record.get("status"), "created": record.get("created"),
                          "has_model": (folder / "model" / "manifest.json").is_file()})
    return items
