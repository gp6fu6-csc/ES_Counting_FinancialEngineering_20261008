"""`escount` command line entry point.

New subcommands (calibrate, prepare, train, infer, research, ...) are added by later tasks.
The migrated V3 pipelines stay reachable unchanged:
  escount legacy-count  <args>   -> escount.es_counting (calibrate / analyze / run-all / predict)
  escount legacy-rotate <args>   -> escount.calibration.rotation_correction
"""
import argparse
import sys

from escount import __version__

LEGACY = {
    "legacy-count": "escount.es_counting",
    "legacy-rotate": "escount.calibration.rotation_correction",
}


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
    parser = argparse.ArgumentParser(prog="escount", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"escount {__version__}")
    parser.add_argument("command", nargs="?", choices=sorted(LEGACY), help="subcommand")
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
