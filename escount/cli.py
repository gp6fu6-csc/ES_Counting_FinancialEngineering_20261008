"""`escount` command line entry point.

  escount calibrate --project <id|path>   -> square-ROI rotation calibration, writes project.json
The migrated V3 pipelines stay reachable unchanged:
  escount legacy-count  <args>   -> escount.es_counting (calibrate / analyze / run-all / predict)
  escount legacy-rotate <args>   -> escount.calibration.rotation_correction
"""
import argparse
import sys
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
