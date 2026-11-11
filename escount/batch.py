"""escount batch: template geometry -> per-project training -> General -> AutoResearch (4 tracks) -> report.

Only chains existing functions (calibrate_project, prepare, train_project, select_project, train_all_rois,
train_general, select_general, Researcher, run_*_research, build_report); nothing in them is changed.

Every output goes to an isolated workspace (default outputs/case_study/), so same-named projects that were
trained earlier under outputs/projects/ are never touched:

  <output>/projects/<case>/        project.json (+ model/), General models
  <output>/features/<case>/        feature cache
  <output>/research/<track>/...    results.tsv per track (default, mae, exact_hit, ensemble)
  <output>/candidates/<track>/...  best_params.json per track
  <output>/mask_overlays/<case>.png reference image with the valid region tinted red (check the mask!)
  <output>/report/latest/          cross-project report (default track) + <output>/report_<track>/
  <output>/summary.md|csv|json     per-project stage status and a cross-track comparison table

A project is a sub-folder of a --data-root holding >= 3 labelled images (file name = sheet count, e.g. 190.png).
testX.png, test.txt, sub-folders (e.g. test/) and folders whose name starts with "test" are never used.
"""
import csv
import json
import shutil
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

DEFAULT_DATA_ROOT = "Data/Case_Study"
DEFAULT_OUTPUT = "outputs/case_study"
MIN_IMAGES = 3
OVERLAY_WIDTH = 1224
TRACKS = ("mae", "exact_hit", "ensemble")          # extra AutoResearch tracks (the "default" track is separate)
STAGES = ("template", "prepare", "train", "select_rois", "all_rois")


# ---------------------------------------------------------------- discovery
def discover(data_roots, only=None):
    """-> (projects, skipped). project = {name, root, path, images}; skipped = [(name, reason)]."""
    from escount.features.cache import labelled_images
    from escount.es_counting import list_images

    projects, skipped, seen = [], [], {}
    for root in data_roots:
        root = Path(root)
        if not root.is_dir():
            raise SystemExit(f"escount batch: data root not found: {root}")
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            if only and folder.name not in only:
                continue
            if folder.name.lower().startswith("test"):
                skipped.append((folder.name, "test data set (folder name starts with 'test')"))
                continue
            if not list_images(folder):
                skipped.append((folder.name, "no png images"))
                continue
            if folder.name in seen:
                raise SystemExit(f"escount batch: project name {folder.name!r} exists in both {seen[folder.name]} and "
                                 f"{root}; outputs are keyed by name, rename one of them")
            seen[folder.name] = root
            images = labelled_images(folder)
            if len(images) < MIN_IMAGES:
                skipped.append((folder.name, f"only {len(images)} labelled image(s), need >= {MIN_IMAGES}"))
                continue
            projects.append({"name": folder.name, "root": root, "path": folder, "images": len(images)})
    missing = sorted(set(only or []) - {p["name"] for p in projects} - {n for n, _ in skipped})
    for name in missing:
        skipped.append((name, "not found under the data roots"))
    return projects, skipped


# ---------------------------------------------------------------- template
def load_template(template, template_data_root, template_profiles_root):
    """-> (ProjectProfile, mask path). `template` = case id (under the template roots) or a project folder path."""
    from escount.calibration.calibrate import PROFILE_NAME, ProjectProfile
    from escount.es_counting import MASK_NAME

    folder = Path(template)
    if not folder.is_dir():
        folder = Path(template_data_root) / template
    if not folder.is_dir():
        raise SystemExit(f"escount batch: template project folder not found: {template} "
                         f"(also tried {Path(template_data_root) / template})")
    profile_path = Path(template_profiles_root) / folder.name / PROFILE_NAME
    if not profile_path.is_file():
        raise SystemExit(f"escount batch: template project.json not found: {profile_path} "
                         f"(calibrate the template first: escount calibrate --project {folder.name})")
    mask_path = folder / MASK_NAME
    if not mask_path.is_file():
        raise SystemExit(f"escount batch: template mask not found: {mask_path}")
    profile = ProjectProfile.load(profile_path)
    if profile.quality_status != "ok":
        raise SystemExit(f"escount batch: template quality_status is {profile.quality_status!r}")
    return profile, mask_path


def backup_project_json(path):
    """project.json -> project_YYYYMMDD_N.json (first unused N of today) in the same folder. Returns the new path."""
    path = Path(path)
    if not path.is_file():
        return None
    stamp = datetime.now().strftime("%Y%m%d")
    n = 1
    while (path.parent / f"project_{stamp}_{n}.json").exists():
        n += 1
    target = path.parent / f"project_{stamp}_{n}.json"
    path.rename(target)
    return target


def apply_template(project, template, template_mask, profiles_root, overlay_dir, log=print):
    """Copy the template mask (only if the project has none), ROIs and strips; angle is re-estimated here."""
    from escount.calibration.calibrate import (
        PROFILE_NAME, CalibrationConfig, CalibrationQualityError, ProjectProfile, calibrate_project,
    )
    from escount.es_counting import MASK_NAME, list_images, pick_reference, read_gray

    folder = project["path"]
    shape = read_gray(list_images(folder)[0]).shape
    size = template.image_size
    if (size["height"], size["width"]) != tuple(shape):
        raise ValueError(f"image size {shape[1]}x{shape[0]} differs from the template {size['width']}x{size['height']}")
    if not (folder / MASK_NAME).is_file():
        shutil.copy2(template_mask, folder / MASK_NAME)
        log(f"  copied template mask -> {folder / MASK_NAME}")
    profile_path = Path(profiles_root) / project["name"] / PROFILE_NAME
    backup = backup_project_json(profile_path)
    if backup:
        log(f"  previous project.json backed up as {backup.name}")
    config = CalibrationConfig(roi_side=max(roi.side for roi in template.rois))
    try:
        profile = calibrate_project(folder, profiles_root, rois=template.rois, config=config)
    except CalibrationQualityError as error:
        raise ValueError(str(error)) from error
    profile.strip_params = dict(template.strip_params)
    profile.save(profile_path)
    write_overlay(folder, pick_reference(list_images(folder)), Path(overlay_dir) / f"{project['name']}.png")
    log(f"  angle_deg={profile.angle_deg:+.4f} after_residual={profile.after_residual_deg:.4f} "
        f"quality={profile.quality_status} strips={profile.strip_params}")
    return profile


def write_overlay(folder, reference, path):
    """Reference image with the valid region (black in background_BW.png) tinted transparent red."""
    from escount.calibration.rotation_correction import normalize_mask, write_image
    from escount.es_counting import read_gray, read_mask

    image = read_gray(reference)
    valid = normalize_mask(read_mask(folder)[1])
    colour = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    red = np.zeros_like(colour)
    red[:, :] = (0, 0, 255)
    colour[valid] = cv2.addWeighted(colour, 0.6, red, 0.4, 0)[valid]
    scale = min(1.0, OVERLAY_WIDTH / colour.shape[1])
    if scale < 1.0:
        colour = cv2.resize(colour, (OVERLAY_WIDTH, round(colour.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_image(path, colour)


# ---------------------------------------------------------------- stages
def run_stage(record, stage, function, log):
    """Run one stage, store ok/failed + seconds in record; returns True on success."""
    start = time.perf_counter()
    try:
        function()
        record[stage] = {"status": "ok", "seconds": round(time.perf_counter() - start, 1)}
        return True
    except Exception as error:                                   # a failed project must not stop the batch
        log(f"  [{stage}] FAILED: {error}")
        record[stage] = {"status": "failed", "error": f"{type(error).__name__}: {error}",
                         "trace": traceback.format_exc(limit=3), "seconds": round(time.perf_counter() - start, 1)}
        return False


def train_projects(projects, template, template_mask, args, paths, results, log=print):
    from escount.calibration.calibrate import PROFILE_NAME
    from escount.features.cache import prepare
    from escount.models.all_rois import train_all_rois
    from escount.models.select_rois import select_project
    from escount.models.train import train_project

    for index, project in enumerate(projects, 1):
        name, root = project["name"], str(project["root"])
        record = results[name]
        log(f"\n=== [{index}/{len(projects)}] {name} ({project['images']} labelled images) ===")
        if args.keep_existing and (paths["projects"] / name / PROFILE_NAME).is_file():
            record["template"] = {"status": "kept existing project.json", "seconds": 0}
            ok = True
        else:
            ok = run_stage(record, "template", lambda: apply_template(project, template, template_mask,
                                                                      paths["projects"], paths["overlays"], log), log)
        steps = (("prepare", lambda: prepare(name, root, paths["projects"], paths["features"], args.workers)),
                 ("train", lambda: train_project(name, paths["features"], paths["projects"], args.device,
                                                 args.max_rows_per_image)),
                 ("select_rois", lambda: select_project(name, paths["features"], paths["projects"], args.device,
                                                        args.max_rois)),
                 ("all_rois", lambda: train_all_rois(name, paths["projects"], root, args.device)))
        for stage, function in steps:
            if not ok:
                record[stage] = {"status": "skipped (earlier stage failed)"}
                continue
            log(f"  -> {stage}")
            ok = run_stage(record, stage, function, log)
        record["ok"] = ok


def train_general_models(names, args, paths, log=print):
    from escount.models.select_rois import select_general
    from escount.models.train import train_general

    info = {}
    if len(names) < 2:
        return {"status": "skipped (needs >= 2 trained projects)"}
    log("\n=== General model (all projects together) ===")
    if not run_stage(info, "train", lambda: train_general(paths["features"], paths["projects"], args.config,
                                                          args.device, args.max_rows_per_image, cases=names), log):
        return info
    run_stage(info, "select_rois", lambda: select_general(paths["features"], paths["projects"], args.device,
                                                          args.max_rois, args.config), log)
    return info


def run_research(groups, args, paths, log=print):
    """Four tracks. `groups` = [(data_root, [case, ...])]. General variants only when there is a single root."""
    from escount.research.daemon import Researcher

    out = {}
    single = len(groups) == 1
    storage = lambda track: f"sqlite:///{(paths['research'] / f'study_{track}.db').as_posix()}"
    paths["research"].mkdir(parents=True, exist_ok=True)
    all_names = [name for _, names in groups for name in names]

    def default_track():
        for root, names in groups:
            researcher = Researcher(names, None if single else names, storage("default"),
                                    paths["research"] / "default", paths["candidates"] / "default", str(root),
                                    paths["projects"], args.config, args.device, args.workers)
            researcher.run(trials=args.trials)

    def extra_track(track):
        from scripts.run_research_ensemble import run_ensemble_research
        from scripts.run_research_exact_hit import run_exact_research
        from scripts.run_research_mae import run_mae_research

        function = {"mae": run_mae_research, "exact_hit": run_exact_research, "ensemble": run_ensemble_research}[track]
        for root, names in groups:
            function(cases=names, trials=args.trials, storage=storage(track), device=args.device,
                     data_root=str(root), profiles_root=str(paths["projects"]), output_root=str(paths["output"]))

    log("\n=== AutoResearch: default track ===")
    run_stage(out, "default", default_track, log)
    for track in TRACKS:
        log(f"\n=== AutoResearch: {track} track ===")
        run_stage(out, track, lambda track=track: extra_track(track), log)
    out["cases"] = all_names
    return out


# ---------------------------------------------------------------- report / summary
def build_reports(names, paths, args, log=print):
    from escount.research.report import build_report

    info = {}
    default_cases = list(names) + (["general"] if (paths["candidates"] / "default" / "general").is_dir() else [])
    log("\n=== Report (default track, cross-project) ===")
    run_stage(info, "default", lambda: build_report(
        paths["output"] / "report", default_cases, paths["research"] / "default", paths["candidates"] / "default",
        f"sqlite:///{(paths['research'] / 'study_default.db').as_posix()}", paths["projects"], args.config, log), log)
    for track in TRACKS:
        run_stage(info, track, lambda track=track: build_report(
            paths["output"] / f"report_{track}", list(names), paths["research"] / track, paths["candidates"] / track,
            f"sqlite:///{(paths['research'] / f'study_{track}.db').as_posix()}", paths["projects"], args.config, log), log)
    return info


def best_of(path):
    """-> {hits, total, hit_rate, mae} of a best_params.json, or {} when the track has no keep."""
    if not Path(path).is_file():
        return {}
    result = json.loads(Path(path).read_text(encoding="utf-8")).get("result") or {}
    return {key: result.get(key) for key in ("hits", "total", "hit_rate", "mae", "roi_count")}


def write_summary(results, general, names, paths, template_name, skipped):
    """summary.json / summary.csv / summary.md: stage status + best result of every track per project."""
    rows = []
    for name in names + (["general"] if (paths["candidates"] / "default" / "general").is_dir() else []):
        row = {"project": name}
        record = results.get(name, {})
        row["pipeline"] = ("ok" if record.get("ok") else "failed") if name in results else "-"
        for track, folder in (("default", "default"), ("mae", "mae"), ("exact_hit", "exact_hit"),
                              ("ensemble", "ensemble")):
            best = best_of(paths["candidates"] / folder / name / "best_params.json")
            if best:
                rate = best.get("hit_rate")
                row[track] = f"{best.get('hits')}/{best.get('total')}" + (f" ({rate:.0%})" if rate is not None else "") \
                    + (f" MAE {best['mae']:.2f}" if best.get("mae") is not None else "")
            else:
                row[track] = "-"
        rows.append(row)
    summary = {"created": datetime.now().isoformat(timespec="seconds"), "template": template_name,
               "projects": results, "general": general, "skipped": [{"name": n, "reason": r} for n, r in skipped],
               "comparison": rows}
    (paths["output"] / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    columns = ["project", "pipeline", "default", "mae", "exact_hit", "ensemble"]
    with open(paths["output"] / "summary.csv", "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = [f"# escount batch summary ({summary['created']})", "", f"Template: `{template_name}`", "",
             "Best leave-one-image-out exact hits per track (hits/total; mae track also lists MAE).", "",
             "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(str(row[c]) for c in columns) + " |" for row in rows]
    failed = [(n, s, r.get("error")) for n, rec in results.items() for s, r in rec.items()
              if isinstance(r, dict) and r.get("status") == "failed"]
    if failed:
        lines += ["", "## Failed stages", ""] + [f"- {n} / {s}: {e}" for n, s, e in failed]
    if skipped:
        lines += ["", "## Skipped folders", ""] + [f"- {n}: {r}" for n, r in skipped]
    (paths["output"] / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


# ---------------------------------------------------------------- driver
def run_batch(args, log=print):
    projects, skipped = discover(args.data_root, args.projects)
    template, template_mask = load_template(args.template, args.template_data_root, args.template_profiles_root)
    output = Path(args.output)
    paths = {"output": output, "projects": output / "projects", "features": output / "features",
             "research": output / "research", "candidates": output / "candidates",
             "overlays": output / "mask_overlays"}
    log(f"template: {args.template} (ROIs {len(template.rois)}, strips {template.strip_params})")
    log(f"outputs : {output.resolve()}")
    for project in projects:
        has_mask = (project["path"] / "background_BW.png").is_file()
        has_profile = (paths["projects"] / project["name"] / "project.json").is_file()
        log(f"  project {project['name']}: {project['images']} labelled images, "
            f"mask {'own' if has_mask else 'from template'}, "
            f"project.json {'exists -> ' + ('kept' if args.keep_existing else 'backed up, then replaced') if has_profile else 'new'}")
    for name, reason in skipped:
        log(f"  skipped {name}: {reason}")
    if args.dry_run:
        log("dry run: nothing was written")
        return 0
    if not projects:
        raise SystemExit("escount batch: no usable project found")
    output.mkdir(parents=True, exist_ok=True)

    results = {project["name"]: {} for project in projects}
    train_projects(projects, template, template_mask, args, paths, results, log)
    good = [p for p in projects if results[p["name"]].get("ok")]
    names = [p["name"] for p in good]
    general = train_general_models(names, args, paths, log) if names and not args.no_general else {}
    if names and not args.skip_research:
        groups = []
        for root in dict.fromkeys(p["root"] for p in good):
            groups.append((root, [p["name"] for p in good if p["root"] == root]))
        general["research"] = run_research(groups, args, paths, log)
        general["reports"] = build_reports(names, paths, args, log)
    summary = write_summary(results, general, names, paths, args.template, skipped)
    failed = len(projects) - len(good)
    log(f"\nbatch finished: {len(good)} ok, {failed} failed, {len(skipped)} skipped -> {(output / 'summary.md').resolve()}")
    return 1 if failed or any(isinstance(v, dict) and v.get("status") == "failed" for v in general.values()) else 0
