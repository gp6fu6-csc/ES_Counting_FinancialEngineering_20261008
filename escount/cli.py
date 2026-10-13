"""`escount` command line entry point.

  escount calibrate --project <id|path>   -> square-ROI rotation calibration, writes project.json
  escount prepare --project <id> | --all  -> feature cache outputs/features/<case>/features.parquet
  escount evaluate --baseline [--project] -> legacy V3 LOO exact hits (evaluator regression check)
  escount train --project <id> | --general [--device cuda|cpu] -> model/ + manifest.json
  escount select-rois --project <id> | --general -> minimal ROIs (project.json, manifest)
  escount benchmark --project <id>        -> single-thread CPU latency of the selected ROIs
  escount infer --project <id> --input <file|folder> [--model project|general] [--no-plots] ...
                                          -> per-image outputs + predictions.csv
  escount research --trials N | --hours H [--cases ...] [--storage URL]
                                          -> Optuna autoresearch, research/<case>/results.tsv, candidates/
  escount report [--out outputs/research_report] -> presentation report (<timestamp>/ and latest/)
  escount adopt <case>                    -> archive candidates/<case>/ in outputs/projects/<case>/research/
                                             (record only: inference keeps using train + select-rois model/)
  escount serve [--host 127.0.0.1] [--port 8000] [--api-key K] [--root Data]
                                          -> FastAPI inference server (LabVIEW), OpenAPI at /docs
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


def _active_case_list(args):
    """_case_list without the excluded_projects of --config (an explicitly named excluded case is an error)."""
    from escount.models.train import active_projects, check_active, load_general_config

    config = load_general_config(args.config)
    cases = _case_list(args)
    if getattr(args, "all", False) or not args.project:
        skipped = [case for case in cases if case not in active_projects(cases, config)]
        if skipped:
            print(f"skipping excluded_projects {skipped} ({args.config})")
        return active_projects(cases, config)
    try:
        check_active(cases[0], config)
    except ValueError as error:
        raise SystemExit(f"escount {args.command}: {error} ({args.config})") from error
    return cases


def cmd_prepare(args):
    from escount.features.cache import prepare

    for case in _active_case_list(args):
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
        train_project(_active_case_list(args)[0], args.cache_root, args.profiles_root,
                      args.device, args.max_rows_per_image)
    return 0


def _print_selection(name, result):
    nested = result["nested_loio"]
    print(f"{name}: {result['roi_count']} ROIs, LOIO exact hits {result['hits']}/{result['total']} "
          f"(subset chosen on all images, optimistic; selection-inside-fold estimate "
          f"{nested['hits']}/{nested['total']}) "
          f"(full set {result['full_set']['hits']}/{result['full_set']['total']} with "
          f"{result['full_set']['roi_count']:g} ROIs/image); "
          f"{[r['key'] for r in result['selected_rois']]}")


def cmd_select_rois(args):
    from escount.models.select_rois import select_general, select_project

    if args.general:
        for name, result in select_general(args.cache_root, args.profiles_root, args.device, args.max_rois,
                                           args.config).items():
            _print_selection(name, result)
    else:
        case = resolve_project(args.project, args.data_root).name
        _print_selection(case, select_project(case, args.cache_root, args.profiles_root, args.device, args.max_rois))
    return 0


def cmd_benchmark(args):
    from escount.inference.benchmark import benchmark

    try:
        report = benchmark(resolve_project(args.project, args.data_root).name, args.data_root, args.profiles_root)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"escount benchmark: {error}") from error
    for item in report["per_image"]:
        stages = " ".join(f"{k}={v:.1f}" for k, v in item["timings_ms"].items())
        print(f"{item['image']}: pred={item['prediction']} gt={item['gt']} total={item['total_ms']:.1f} ms ({stages})")
    print(f"{report['case']}: {report['roi_count']} ROIs, single-thread CPU median {report['median_ms']:.1f} ms, "
          f"max {report['max_ms']:.1f} ms (limit {report['limit_ms']:.0f} ms, within={report['within_limit']})")
    return 0 if report["within_limit"] else 3


def cmd_infer(args):
    import cv2
    from threadpoolctl import threadpool_limits

    from escount.inference.engine import InferenceEngine, OutputOptions

    try:
        options = OutputOptions(corrected=not args.no_corrected, analysis_plot=not args.no_plots,
                                signals_csv=not args.no_signals, result_json=not args.no_json, dpi=args.dpi)
    except ValueError as error:
        raise SystemExit(f"escount infer: {error}") from error
    case = Path(args.project).name if Path(args.project).is_dir() else args.project
    output = Path(args.output) if args.output else Path("outputs") / "infer" / case
    previous = cv2.getNumThreads()
    cv2.setNumThreads(1)                      # single-process, single-thread CPU (as in `escount benchmark`)
    try:
        with threadpool_limits(limits=1):
            engine = InferenceEngine.load(case, args.model, args.profiles_root, args.config)
            results = engine.run(args.input, output, options)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"escount infer: {error}") from error
    finally:
        cv2.setNumThreads(previous)
    for result in results:
        stages = " ".join(f"{k}={v:.1f}" for k, v in result["timings_ms"].items() if k != "total")
        print(f"{result['image']}: N_pred={result['N_pred']} GT={result['gt']} hit={result['hit']} "
              f"total={result['timings_ms']['total']:.1f} ms ({stages}) outputs={result['outputs']}")
    labelled = [r for r in results if r["gt"] is not None]
    print(f"{case} ({args.model} model {engine.model_case}, {len(engine.runtime.manifest['selected_rois'])} ROIs): "
          f"{len(results)} images, exact hits {sum(bool(r['hit']) for r in labelled)}/{len(labelled)}; "
          f"wrote {(output / 'predictions.csv').resolve()}")
    return 0


def cmd_research(args):
    from escount.features.cache import case_ids
    from escount.research.daemon import Researcher

    if args.trials is not None and args.trials < 1 or args.hours is not None and args.hours <= 0:
        raise SystemExit("escount research: --trials must be >= 1 and --hours > 0")
    try:
        researcher = Researcher(case_ids(args.data_root), args.cases, args.storage, args.research_root,
                                args.candidates_root, args.data_root, args.profiles_root, args.config, args.device,
                                args.workers)
    except ValueError as error:
        raise SystemExit(f"escount research: {error}") from error
    start = time.perf_counter()
    done = researcher.run(trials=args.trials, hours=args.hours)
    print(f"research: {done} trials over {researcher.cases} in {time.perf_counter() - start:.1f} s; "
          f"results {Path(args.research_root).resolve()}, candidates {Path(args.candidates_root).resolve()}")
    return 0


def cmd_report(args):
    from escount.research.report import build_report

    build_report(args.out, args.cases, args.research_root, args.candidates_root, args.storage,
                 args.profiles_root, args.config)
    return 0


def cmd_adopt(args):
    from escount.research.daemon import adopt

    try:
        target = adopt(args.case, args.candidates_root, args.profiles_root)
    except FileNotFoundError as error:
        raise SystemExit(f"escount adopt: {error}") from error
    print(f"copied candidates/{args.case} -> {target.resolve()} (record only: inference still uses "
          f"outputs/projects/{args.case}/model/ from escount train + select-rois)")
    return 0


def cmd_serve(args):
    from escount.server.app import ServerConfig, check_bind, create_app

    try:
        check_bind(args.host, args.api_key)
    except ValueError as error:
        raise SystemExit(f"escount serve: {error}") from error
    import uvicorn

    from escount.inference.benchmark import single_thread

    single_thread()                           # inference: single-process, single-thread CPU
    config = ServerConfig(profiles_root=args.profiles_root, allowed_roots=tuple(args.root or ["Data"]),
                          api_key=args.api_key, general_config=args.config)
    uvicorn.run(create_app(config), host=args.host, port=args.port, log_level="info")
    return 0


DEFAULT_STORAGE = "sqlite:///research/study.db"   # = escount.research.daemon.DEFAULT_STORAGE (lazy import)


def _research_paths(command):
    command.add_argument("--storage", default=DEFAULT_STORAGE, help=f"Optuna storage (default: {DEFAULT_STORAGE})")
    command.add_argument("--research-root", default="research", help="results.tsv root (default: research)")
    command.add_argument("--candidates-root", default="candidates", help="candidate root (default: candidates)")
    command.add_argument("--profiles-root", default="outputs/projects", help="project profiles (default: outputs/projects)")
    command.add_argument("--config", default="configs/general.json",
                         help="General config (excluded_projects, exclude_from_general)")


def _common(command, cache=True):
    command.add_argument("--data-root", default="Data", help="root folder of case ids (default: Data)")
    command.add_argument("--profiles-root", default="outputs/projects", help="project profiles (default: outputs/projects)")
    if cache:
        command.add_argument("--cache-root", default="outputs/features", help="feature cache (default: outputs/features)")


def _target(command, general=False):
    group = command.add_mutually_exclusive_group(required=True)
    group.add_argument("--project", help="case id (Data folder name) or project folder path")
    if general:
        group.add_argument("--general", action="store_true",
                           help="General model(s): general (+ general_no_ring while Ring-type is active)")
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
    prepare.add_argument("--config", default="configs/general.json", help="General config (excluded_projects)")
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
    train.add_argument("--config", default="configs/general.json",
                       help="General config (excluded_projects, exclude_from_general)")
    train.set_defaults(func=cmd_train)

    select = sub.add_parser("select-rois", help="greedy minimal-ROI selection (Grouped-LOO exact hit)")
    _target(select, general=True)
    _common(select)
    select.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    select.add_argument("--max-rois", type=int, default=25)
    select.add_argument("--config", default="configs/general.json", help="General config (excluded_projects)")
    select.set_defaults(func=cmd_select_rois)

    bench = sub.add_parser("benchmark", help="single-thread CPU latency of the selected ROIs (< 1 s per image)")
    bench.add_argument("--project", required=True, help="case id or project folder path")
    _common(bench, cache=False)
    bench.set_defaults(func=cmd_benchmark)

    infer = sub.add_parser("infer", help="InferenceEngine on an image file or folder (single-process CPU)")
    infer.add_argument("--project", required=True, help="case id (profile under --profiles-root) or profile folder")
    infer.add_argument("--input", required=True, help="image file or folder of PNG images")
    infer.add_argument("--output", help="output folder (default: outputs/infer/<case>)")
    infer.add_argument("--model", choices=("project", "general"), default="project",
                       help="project model or the production General model (default: project)")
    infer.add_argument("--no-plots", action="store_true", help="skip analysis.png")
    infer.add_argument("--no-corrected", action="store_true", help="skip corrected.png")
    infer.add_argument("--no-signals", action="store_true", help="skip signals.csv")
    infer.add_argument("--no-json", action="store_true", help="skip result.json")
    infer.add_argument("--dpi", type=int, default=100, help="analysis.png dpi, >= 64 (default: 100)")
    infer.add_argument("--config", default="configs/general.json", help="General config (production General model)")
    infer.add_argument("--profiles-root", default="outputs/projects", help="project profiles (default: outputs/projects)")
    infer.set_defaults(func=cmd_infer)

    research = sub.add_parser("research", help="Optuna TPE autoresearch over every case (resumable, SQLite)")
    budget = research.add_mutually_exclusive_group(required=True)
    budget.add_argument("--trials", type=int, help="new trials per case (round-robin)")
    budget.add_argument("--hours", type=float, help="total wall-clock hours (round-robin)")
    research.add_argument("--cases", nargs="+", help="case ids (default: every project not in excluded_projects + general)")
    research.add_argument("--data-root", default="Data", help="root folder of case ids (default: Data)")
    research.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="XGBoost device (cuda falls back to cpu)")
    research.add_argument("--workers", type=int, default=None, help="feature worker processes (default: os.cpu_count())")
    _research_paths(research)
    research.set_defaults(func=cmd_research)

    report = sub.add_parser("report", help="presentation report -> outputs/research_report/<timestamp>/ + latest/")
    report.add_argument("--out", default="outputs/research_report", help="report root (default: outputs/research_report)")
    report.add_argument("--cases", nargs="+", help="case ids (default: projects not in excluded_projects + general)")
    _research_paths(report)
    report.set_defaults(func=cmd_report)

    adopt = sub.add_parser("adopt", help="copy candidates/<case>/ to outputs/projects/<case>/research/ (record only; "
                                         "inference still uses the train + select-rois model)")
    adopt.add_argument("case", help="case id (project or general)")
    adopt.add_argument("--candidates-root", default="candidates", help="candidate root (default: candidates)")
    adopt.add_argument("--profiles-root", default="outputs/projects", help="project profiles (default: outputs/projects)")
    adopt.set_defaults(func=cmd_adopt)

    serve = sub.add_parser("serve", help="FastAPI inference server + web UI (default 127.0.0.1:8000)")
    serve.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1; others need --api-key)")
    serve.add_argument("--port", type=int, default=8000, help="port (default 8000)")
    serve.add_argument("--api-key", help="require this value in the X-API-Key header (all endpoints but /health)")
    serve.add_argument("--root", action="append", help="folder /infer/path may read from (repeatable; default Data)")
    serve.add_argument("--profiles-root", default="outputs/projects", help="project profiles (default: outputs/projects)")
    serve.add_argument("--config", default="configs/general.json", help="General config (production General model)")
    serve.set_defaults(func=cmd_serve)
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
