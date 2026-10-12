"""`escount` command line entry point.

  escount calibrate --project <id|path>   -> square-ROI rotation calibration, writes project.json
  escount prepare --project <id> | --all  -> feature cache outputs/features/<case>/features.parquet
  escount evaluate --baseline [--project] -> legacy V3 LOO exact hits (evaluator regression check)
  escount train --project <id> | --general [--device cuda|cpu] -> model/ + manifest.json
  escount select-rois --project <id> | --general -> minimal ROIs (project.json, manifest)
  escount benchmark --project <id>        -> single-thread CPU latency of the selected ROIs
The migrated V3 pipelines stay reachable unchanged:
  escount legacy-count  <args>   -> escount.es_counting (calibrate / analyze / run-all / predict)
  escount legacy-rotate <args>   -> escount.calibration.rotation_correction
"""
import argparse
import sys
import time
from pathlib import Path

from escount import __version__

LEGACY = {
    "legacy-count": "escount.es_counting",
    "legacy-rotate": "escount.calibration.rotation_correction",
}


def resolve_project(project, data_root):
    """A case id is a folder under data_root; an existing directory path (e.g. Data/T-type) also works."""
    path = Path(project)
    if path.is_dir():
        return path.resolve()
    candidate = Path(data_root) / project
    if candidate.is_dir():
        return candidate.resolve()
    raise SystemExit(f"escount: project {project!r} not found (neither a directory nor a case under {data_root})")


def cmd_calibrate(args):
    from escount.calibration.calibrate import CalibrationConfig, CalibrationQualityError, calibrate_project

    case_dir = resolve_project(args.project, args.data_root)
    config = CalibrationConfig(roi_side=args.roi_side)
    try:
        profile = calibrate_project(case_dir, args.profiles_root, reference=args.reference, config=config)
        status = 0
    except CalibrationQualityError as error:
        from escount.calibration.calibrate import ProjectProfile

        profile = ProjectProfile.load(Path(args.profiles_root) / case_dir.name / "project.json")
        print(f"escount calibrate: {error}", file=sys.stderr)
        status = 2
    except ValueError as error:
        raise SystemExit(f"escount calibrate: {error}") from error
    print(f"case={profile.case} reference={profile.reference_image} angle_deg={profile.angle_deg:+.4f} "
          f"roi_angles_deg={[round(a, 4) for a in profile.roi_angles_deg]} "
          f"before_residual_deg={profile.before_residual_deg:.4f} after_residual_deg={profile.after_residual_deg:.4f} "
          f"quality_status={profile.quality_status}")
    print(f"wrote {(Path(args.profiles_root) / case_dir.name / 'project.json').resolve()}")
    return status


def _case_list(args):
    from escount.features.cache import case_ids

    if getattr(args, "all", False) or not args.project:
        return case_ids(args.data_root)
    return [resolve_project(args.project, args.data_root).name]


def cmd_prepare(args):
    from escount.features.cache import prepare

    for case in _case_list(args):
        start = time.perf_counter()
        path, rebuilt = prepare(case, args.data_root, args.profiles_root, args.cache_root, args.workers, force=args.force)
        import pyarrow.parquet as pq

        rows = pq.ParquetFile(path).metadata.num_rows
        print(f"{case}: {'built' if rebuilt else 'fresh (reused)'} {rows} rows -> {path.resolve()} "
              f"({time.perf_counter() - start:.1f} s)")
    return 0


def cmd_evaluate(args):
    from escount.research.baseline import as_hits, legacy_baseline, legacy_report_hits

    if not args.baseline:
        raise SystemExit("escount evaluate: only --baseline is available (legacy V3 regression check)")
    for case in _case_list(args):
        final, results = legacy_baseline(Path(args.data_root) / case)
        line = (f"{case}: legacy re-run final={final} hits {as_hits(results[final])} "
                f"(kd {as_hits(results['kd'])}, macd {as_hits(results['macd'])})")
        report_dir = Path(args.legacy_root) / case
        if (report_dir / "calibration_report.json").is_file():
            method, hits, total = legacy_report_hits(report_dir)
            line += f"; calibration_report.json {method} {hits}/{total}"
        print(line)
    return 0


def cmd_train(args):
    from escount.models.train import train_general, train_project

    if args.general:
        _, comparison = train_general(args.cache_root, args.profiles_root, args.config, args.device,
                                      args.max_rows_per_image)
        print(f"ring comparison -> {(Path(args.profiles_root) / 'general' / 'ring_comparison.json').resolve()}")
    else:
        train_project(resolve_project(args.project, args.data_root).name, args.cache_root, args.profiles_root,
                      args.device, args.max_rows_per_image)
    return 0


def _print_selection(name, result):
    print(f"{name}: {result['roi_count']} ROIs, LOIO exact hits {result['hits']}/{result['total']} "
          f"(full set {result['full_set']['hits']}/{result['full_set']['total']} with "
          f"{result['full_set']['roi_count']:g} ROIs/image); "
          f"{[r['key'] for r in result['selected_rois']]}")


def cmd_select_rois(args):
    from escount.models.select_rois import select_general, select_project

    if args.general:
        for name, result in select_general(args.cache_root, args.profiles_root, args.device, args.max_rois).items():
            _print_selection(name, result)
    else:
        case = resolve_project(args.project, args.data_root).name
        _print_selection(case, select_project(case, args.cache_root, args.profiles_root, args.device, args.max_rois))
    return 0


def cmd_benchmark(args):
    from escount.inference.benchmark import benchmark

    report = benchmark(resolve_project(args.project, args.data_root).name, args.data_root, args.profiles_root)
    for item in report["per_image"]:
        stages = " ".join(f"{k}={v:.1f}" for k, v in item["timings_ms"].items())
        print(f"{item['image']}: pred={item['prediction']} gt={item['gt']} total={item['total_ms']:.1f} ms ({stages})")
    print(f"{report['case']}: {report['roi_count']} ROIs, single-thread CPU median {report['median_ms']:.1f} ms, "
          f"max {report['max_ms']:.1f} ms (limit {report['limit_ms']:.0f} ms, within={report['within_limit']})")
    return 0 if report["within_limit"] else 3


def _common(command, cache=True):
    command.add_argument("--data-root", default="Data", help="root folder of case ids (default: Data)")
    command.add_argument("--profiles-root", default="outputs/projects", help="project profiles (default: outputs/projects)")
    if cache:
        command.add_argument("--cache-root", default="outputs/features", help="feature cache (default: outputs/features)")


def _target(command, general=False):
    group = command.add_mutually_exclusive_group(required=True)
    group.add_argument("--project", help="case id (Data folder name) or project folder path")
    if general:
        group.add_argument("--general", action="store_true", help="General models (general, general_no_ring)")
    else:
        group.add_argument("--all", action="store_true", help="every case under --data-root")


def build_parser():
    parser = argparse.ArgumentParser(prog="escount", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"escount {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="command")
    for name, module in LEGACY.items():
        sub.add_parser(name, help=f"legacy V3 CLI ({module}); remaining args are passed through", add_help=False)
    calibrate = sub.add_parser("calibrate", help="square-ROI rotation calibration -> project.json")
    calibrate.add_argument("--project", required=True, help="case id (Data folder name) or project folder path")
    calibrate.add_argument("--data-root", default="Data", help="root folder of case ids (default: Data)")
    calibrate.add_argument("--profiles-root", default="outputs/projects", help="output root (default: outputs/projects)")
    calibrate.add_argument("--roi-side", type=int, default=384, help="square ROI side in px, >= 300 (default: 384)")
    calibrate.add_argument("--reference", help="reference image file name (default: largest file-name number)")
    calibrate.set_defaults(func=cmd_calibrate)

    prepare = sub.add_parser("prepare", help="feature cache (multiprocess) -> outputs/features/<case>/features.parquet")
    _target(prepare)
    _common(prepare)
    prepare.add_argument("--workers", type=int, default=None, help="worker processes (default: os.cpu_count())")
    prepare.add_argument("--force", action="store_true", help="rebuild even if the cache is fresh")
    prepare.set_defaults(func=cmd_prepare)

    evaluate = sub.add_parser("evaluate", help="fixed exact-hit evaluator; --baseline re-runs the legacy V3 LOO")
    evaluate.add_argument("--project", help="case id (default: every case)")
    evaluate.add_argument("--baseline", action="store_true", help="legacy V3 baseline (w=15, D=30, side margin 100)")
    evaluate.add_argument("--legacy-root", default="outputs/es_counting", help="legacy outputs for comparison")
    _common(evaluate, cache=False)
    evaluate.set_defaults(func=cmd_evaluate)

    train = sub.add_parser("train", help="Project / General training (Ridge, SVR, XGBoost cuda, LightGBM)")
    _target(train, general=True)
    _common(train)
    train.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="XGBoost device (cuda falls back to cpu)")
    train.add_argument("--max-rows-per-image", type=int, default=1000, help="training rows per image (default 1000)")
    train.add_argument("--config", default="configs/general.json", help="General config (exclude_from_general)")
    train.set_defaults(func=cmd_train)

    select = sub.add_parser("select-rois", help="greedy minimal-ROI selection (Grouped-LOO exact hit)")
    _target(select, general=True)
    _common(select)
    select.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    select.add_argument("--max-rois", type=int, default=25)
    select.set_defaults(func=cmd_select_rois)

    bench = sub.add_parser("benchmark", help="single-thread CPU latency of the selected ROIs (< 1 s per image)")
    bench.add_argument("--project", required=True, help="case id or project folder path")
    _common(bench, cache=False)
    bench.set_defaults(func=cmd_benchmark)
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in LEGACY:
        import importlib

        module = importlib.import_module(LEGACY[argv[0]])
        saved = sys.argv
        sys.argv = [f"escount {argv[0]}", *argv[1:]]  # legacy mains parse sys.argv
        try:
            return module.main()
        finally:
            sys.argv = saved
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
