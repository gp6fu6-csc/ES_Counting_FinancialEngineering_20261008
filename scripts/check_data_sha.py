"""Fail if any file under Data/ differs from the recorded SHA-256 baseline.

Baseline: outputs/task1/data_sha256_before.txt, one line per file: `HASH  Data\\<case>\\<file>`.
Usage (repo root):  uv run python scripts/check_data_sha.py [--baseline PATH] [--data PATH]
"""
import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read_baseline(path):
    entries = {}
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        digest, name = line.strip().split(None, 1)
        entries[name.strip().replace("\\", "/")] = digest.upper()
    return entries


def current_hashes(data_dir, root):
    return {str(p.relative_to(root)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest().upper()
            for p in sorted(Path(data_dir).rglob("*")) if p.is_file()}


def compare(baseline, current):
    """Return sorted difference lines (missing, extra, changed); empty when identical."""
    lines = [f"missing  {name}" for name in sorted(set(baseline) - set(current))]
    lines += [f"extra    {name}" for name in sorted(set(current) - set(baseline))]
    lines += [f"changed  {name}" for name in sorted(set(baseline) & set(current)) if baseline[name] != current[name]]
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path, default=ROOT / "outputs" / "task1" / "data_sha256_before.txt")
    parser.add_argument("--data", type=Path, default=ROOT / "Data")
    args = parser.parse_args(argv)
    baseline = read_baseline(args.baseline)
    differences = compare(baseline, current_hashes(args.data, args.data.resolve().parent))
    if differences:
        print("\n".join(differences))
        print(f"Data SHA check FAILED: {len(differences)} difference(s)")
        return 1
    print(f"Data SHA unchanged: {len(baseline)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
