"""`escount variant` and `escount run-variants`: ROI-strip variants and multi-dataset x multi-variant training.

run-variants trains every (project, variant) pair one after the other with the same CLI steps the web
training page uses (prepare -> train -> select-rois -> all-rois -> infer), optionally runs AutoResearch
once per project first, keeps the trained model and writes runs/<run>/run.json (all options, the
AutoResearch settings and result, accuracy) into the variant's profile folder. See escount.variants.
"""
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from escount.variants import (
    VARIANT_SEP, check_name, list_runs, new_run_id, normalize_strips, snapshot_run, split_case, summarize_manifest,
    sync_variant, variant_case, variant_names, variant_strips, write_variants_summary,
)

TRACK_DIRS = {"default": "", "mae": "mae", "exact": "exact_hit", "ensemble": "ensemble"}   # candidates/<dir>/<case>
DEFAULT_STRIPS = {"n": 10, "w": 30, "pitch": 50, "side_margin": 50, "x_offset": 0}      # = models.all_rois.DEFAULT_STRIPS


def _base_name(project):
    path = Path(project)
    return path.name if path.is_dir() else split_case(project)[0]


def _project_json(profiles_root, base):
    return Path(profiles_root) / base / "project.json"


# ---------------------------------------------------------------- escount variant create | list
def cmd_variant(args):
    from escount.calibration.calibrate import ProjectProfile

    base = _base_name(args.project)
    project_json = _project_json(args.profiles_root, base)
    if not project_json.is_file():
        raise SystemExit(f"escount variant: {project_json} missing; run `escount calibrate --project {base}` first")
    if args.action == "list":
        names = variant_names(base, args.profiles_root)
        print(f"{base}: {len(names)} variant(s)" + (f" ({', '.join(names)})" if names else ""))
        for name in names:
            case = variant_case(base, name)
            model = (Path(args.profiles_root) / case / "model" / "manifest.json").is_file()
            print(f"  {name}: strips {variant_strips(case, args.profiles_root)}, "
                  f"{len(list_runs(case, args.profiles_root))} saved run(s), latest model {'yes' if model else 'no'}")
        return 0
    try:
        name = check_name(args.name)
        case = variant_case(base, name)
        current = variant_strips(case, args.profiles_root)
        if current is None and args.copy_from:
            source = variant_strips(args.copy_from if VARIANT_SEP in args.copy_from else base, args.profiles_root)
            if source is None:
                raise ValueError(f"--copy-from {args.copy_from!r} has no project.json")
            current = source
        strips = dict(current or ProjectProfile.load(project_json).strip_params or DEFAULT_STRIPS)
        for key in ("n", "w", "pitch", "side_margin", "x_offset"):
            value = getattr(args, key)
            if value is not None:
                strips[key] = value
        folder = sync_variant(case, args.profiles_root, normalize_strips(strips))
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"escount variant: {error}") from error
    print(f"variant {case}: strips {variant_strips(case, args.profiles_root)} -> {folder.resolve()}")
    print("train it with: escount run-variants --projects", base, "--variants", name)
    return 0


# ---------------------------------------------------------------- escount run-variants
def _run_step(runner, argv, log=print):
    """Run one CLI step in-process. Returns (return code, seconds); a SystemExit message is printed, not raised."""
    log("$ escount " + " ".join(argv))
    start = time.perf_counter()
    try:
        code = runner(argv)
    except SystemExit as error:
        code = error.code
        if isinstance(code, str):
            print(code, file=sys.stderr)
            code = 1
    except Exception:                                   # a failed step must not stop the other pairs
        traceback.print_exc()
        code = 1
    return (code or 0), time.perf_counter() - start


def _research_once(base, args, runner):
    """AutoResearch for one project (independent of the variant strips: it searches its own w / D geometry)."""
    argv = ["research", "--track", args.track, "--trials", str(args.trials), "--cases", base, "--device", args.device,
            "--data-root", args.data_root, "--profiles-root", args.profiles_root, "--config", args.config,
            "--storage", args.storage, "--research-root", args.research_root,
            "--candidates-root", args.candidates_root]
    if args.workers:
        argv += ["--workers", str(args.workers)]
    code, seconds = _run_step(runner, argv)
    best_path = Path(args.candidates_root) / TRACK_DIRS[args.track] / base / "best_params.json"
    best = json.loads(best_path.read_text(encoding="utf-8")) if code == 0 and best_path.is_file() else None
    return {"track": args.track, "trials": args.trials, "device": args.device, "storage": args.storage,
            "research_root": args.research_root, "candidates_root": args.candidates_root,
            "status": "ok" if code == 0 else "failed", "seconds": round(seconds, 1),
            "best_params_path": str(best_path), "best_params": best}


def _report_paths(args):
    """research_root / candidates_root / storage / study suffix of the chosen track (as written by `escount research`)."""
    folder = TRACK_DIRS[args.track]
    if not folder:
        return args.research_root, args.candidates_root, args.storage, ""
    return f"research/{folder}", f"candidates/{folder}", f"sqlite:///research/study_{folder}.db", f"_{folder}"


def _build_report(bases, args, runner):
    """After all trainings: analyse each base project with its own model (the report's prediction figure reads
    <results-root>/<project>/predictions.csv), run `escount report` and write variants_summary.csv.
    Returns the list of failed parts (empty = everything worked)."""
    failed = []
    for base in bases:
        if (Path(args.profiles_root) / base / "model" / "manifest.json").is_file():
            code, _ = _run_step(runner, ["infer", "--project", base, "--input", str(Path(args.data_root) / base),
                                         "--output", str(Path(args.results_root) / base),
                                         "--profiles-root", args.profiles_root, "--config", args.config])
            if code != 0:
                failed.append(f"infer {base}")
        else:
            print(f"[report] {base} has no base-project model: its prediction figure stays empty "
                  f"(the variants' own results are in {args.results_root}/{base}@<variant>~<run>)")
    research_root, candidates_root, storage, suffix = _report_paths(args)
    out = args.report_out + (suffix if suffix else "")
    code, _ = _run_step(runner, ["report", "--cases", *bases, "--out", out, "--results-root", args.results_root,
                                 "--research-root", research_root, "--candidates-root", candidates_root,
                                 "--storage", storage, "--study-suffix", suffix,
                                 "--profiles-root", args.profiles_root, "--config", args.config])
    if code != 0:
        failed.append("report")
    try:
        rows = write_variants_summary(Path(out) / "variants_summary.csv", bases, args.profiles_root)
        print(f"[report] variants_summary.csv: {rows} run(s) -> {(Path(out) / 'variants_summary.csv').resolve()}")
    except Exception:
        traceback.print_exc()
        failed.append("variants_summary")
    print(f"[report] report directory: {Path(out).resolve()}"
          + (f"; FAILED parts: {failed}" if failed else ""))
    return failed


def _resolve_strips(base, variant, args):
    """strip_params for base@variant: its own, else copied from another project that has this variant name."""
    case = variant_case(base, variant)
    if variant_strips(case, args.profiles_root) is not None:
        return None, None
    root = Path(args.profiles_root)
    donors = sorted(folder.name for folder in root.glob(f"*{VARIANT_SEP}{variant}")
                    if (folder / "project.json").is_file()) if root.is_dir() else []
    if not donors:
        raise SystemExit(f"escount run-variants: variant {variant!r} does not exist for any project; create it in the "
                         f"calibration page or with `escount variant create --project {base} --name {variant}`")
    return variant_strips(donors[0], args.profiles_root), donors[0]


def run_variants(args, runner):
    from escount.calibration.calibrate import ProjectProfile
    from escount.models.store import read_manifest

    bases = [_base_name(project) for project in args.projects]
    for base in bases:
        if not (Path(args.data_root) / base).is_dir():
            raise SystemExit(f"escount run-variants: project {base!r} not found under {args.data_root}")
        if not _project_json(args.profiles_root, base).is_file():
            raise SystemExit(f"escount run-variants: {base} is not calibrated; run `escount calibrate --project {base}`")
    for variant in args.variants:
        try:
            check_name(variant)
        except ValueError as error:
            raise SystemExit(f"escount run-variants: {error}") from error
    plan = [(base, variant, *_resolve_strips(base, variant, args)) for base in bases for variant in args.variants]
    print(f"run-variants: {len(plan)} training(s), one after the other: "
          + ", ".join(variant_case(base, variant) for base, variant, *_ in plan))

    options = {"device": args.device, "max_rois": args.max_rois, "max_rows_per_image": args.max_rows_per_image,
               "workers": args.workers, "infer": not args.no_infer, "config": args.config}
    roots = ["--data-root", args.data_root, "--profiles-root", args.profiles_root]
    cache = ["--cache-root", args.cache_root]
    research_by_base = {}
    summary = []
    for base, variant, copied, donor in plan:
        if base not in research_by_base:
            research_by_base[base] = _research_once(base, args, runner) if args.trials > 0 else \
                {"track": args.track, "trials": 0, "status": "not run"}
        case = variant_case(base, variant)
        sync_variant(case, args.profiles_root, strips=copied)
        if donor:
            print(f"[{case}] strips copied from {donor}: {variant_strips(case, args.profiles_root)}")
        run = new_run_id(case, args.profiles_root)
        results_dir = Path(args.results_root) / f"{case}~{run}"
        started = datetime.now()
        print(f"\n=== [{case}] run {run} ===")
        steps = [("prepare", ["prepare", "--project", case, *roots, *cache, "--config", args.config]
                  + (["--workers", str(args.workers)] if args.workers else [])),
                 ("train", ["train", "--project", case, "--device", args.device, "--max-rows-per-image",
                            str(args.max_rows_per_image), *roots, *cache, "--config", args.config]),
                 ("select-rois", ["select-rois", "--project", case, "--device", args.device, "--max-rois",
                                  str(args.max_rois), *roots, *cache, "--config", args.config]),
                 ("all-rois", ["all-rois", "--project", case, "--device", args.device, *roots])]
        if not args.no_infer:
            steps.append(("infer", ["infer", "--project", case, "--input", str(Path(args.data_root) / base),
                                    "--output", str(results_dir), "--profiles-root", args.profiles_root,
                                    "--config", args.config]))
        log, status = [], "ok"
        for name, argv in steps:
            code, seconds = _run_step(runner, argv)
            log.append({"step": name, "status": "ok" if code == 0 else "failed", "seconds": round(seconds, 1)})
            if code != 0:
                status = "failed"
                print(f"[{case}] step {name} failed (code {code}); the remaining steps of this pair are skipped")
                break
        model_dir = Path(args.profiles_root) / case / "model"
        manifest = read_manifest(model_dir) if status == "ok" and (model_dir / "manifest.json").is_file() else None
        profile = ProjectProfile.load(Path(args.profiles_root) / case / "project.json")
        record = {"run": run, "case": base, "variant": variant, "profile_case": case, "status": status,
                  "created": started.isoformat(timespec="seconds"),
                  "finished": datetime.now().isoformat(timespec="seconds"), "note": args.note,
                  "strip_params": profile.strip_params,
                  "calibration": {"angle_deg": profile.angle_deg, "reference_image": profile.reference_image,
                                  "quality_status": profile.quality_status},
                  "options": options, "steps": log, "research": research_by_base[base],
                  "model": summarize_manifest(manifest) if manifest else None,
                  "results_dir": str(results_dir) if status == "ok" and not args.no_infer else None}
        folder = snapshot_run(case, run, record, args.profiles_root)
        print(f"[{case}] run {run}: {status}; saved {folder.resolve()}")
        summary.append((case, run, status))
    print("\nrun-variants summary:")
    for case, run, status in summary:
        print(f"  {case}~{run}: {status}")
    problems = [f"research {base}" for base, item in research_by_base.items() if item.get("status") == "failed"]
    if args.report:
        problems += _build_report(bases, args, runner)
    if problems:
        print(f"run-variants: parts that failed (training of the other pairs was not affected): {problems}")
    return 0 if all(status == "ok" for _, _, status in summary) and not problems else 3


def cmd_run_variants(args):
    from escount.cli import main

    return run_variants(args, main)
