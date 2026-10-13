"""Fail if any installed dependency has a non-permissive licence.

Allowed: MIT (incl. MIT-CMU), BSD, Apache-2.0, PSF, ISC, LGPL (dynamic linking only).
Rejected: GPL, AGPL, SSPL, commercial / proprietary / non-commercial terms, unknown licences.
Explicit exceptions (ACCEPTED_EXCEPTIONS, documented in README.md): tqdm (MPL-2.0 AND MIT).

Usage (repo root):  uv run python scripts/check_licenses.py
Checks every package installed in the current environment (pip-licenses, --from=mixed) except
this project itself, and lists uv.lock packages that are not installed on this platform.
"""
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJECT = "escount"

ALLOWED = re.compile(
    r"\bMIT\b|\bBSD\b|BSD-\d|\bApache\b|\bPSF\b|Python Software Foundation|\bISC\b|\bLGPL|Lesser General Public",
    re.IGNORECASE)
FORBIDDEN = re.compile(
    r"(?<![L])\bA?GPL|(?<!Lesser )(?<!Library )General Public License|\bSSPL|Server Side Public|"
    r"Commercial|Proprietary|Business Source|\bBUSL|Elastic License|Commons Clause|\bCC-BY-NC|UNKNOWN",
    re.IGNORECASE)
NON_COMMERCIAL = re.compile(r"Non-?Commercial", re.IGNORECASE)
# Packages outside the allow-list that are accepted explicitly (listed in README.md, licence table).
ACCEPTED_EXCEPTIONS = {
    "tqdm": ("MPL-2.0 AND MIT", "required by optuna; used unmodified. MPL-2.0 is file-level copyleft only "
                                "(no fees, no usage limits)"),
}


def classify(license_text):
    """Return (ok, reason) for a pip-licenses licence string. In an SPDX 'A AND B' expression every
    part must pass; in 'A OR B' one passing part is enough."""
    text = (license_text or "").strip()
    if not text:
        return False, "empty licence metadata"
    if FORBIDDEN.search(text) or NON_COMMERCIAL.search(text):
        return False, "forbidden or unknown licence"
    parts = [part.strip(" ()") for part in re.split(r"\s+AND\s+", text)]
    if all(ALLOWED.search(part) for part in parts):
        return True, "permissive"
    return False, "licence not in the allow-list"


def check_package(name, license_text):
    """classify(), then the explicit ACCEPTED_EXCEPTIONS (only for the exact licence string recorded)."""
    ok, reason = classify(license_text)
    exception = ACCEPTED_EXCEPTIONS.get(name.lower())
    if not ok and exception and (license_text or "").strip() == exception[0]:
        return True, f"accepted exception: {exception[1]}"
    return ok, reason


def installed_licenses():
    output = subprocess.run(
        [sys.executable, "-m", "piplicenses", "--from=mixed", "--format=json", "--with-system",
         f"--ignore-packages={PROJECT}"],
        check=True, capture_output=True, text=True, encoding="utf-8").stdout
    return json.loads(output)


def lock_only_packages(installed_names):
    lock = ROOT / "uv.lock"
    if not lock.is_file():
        return []
    locked = {p["name"] for p in tomllib.loads(lock.read_text(encoding="utf-8")).get("package", [])}
    normalized = {n.lower().replace("_", "-") for n in installed_names}
    return sorted(n for n in locked if n.lower().replace("_", "-") not in normalized and n != PROJECT)


def main():
    packages = installed_licenses()
    failures = []
    for item in sorted(packages, key=lambda p: p["Name"].lower()):
        name, version, text = item["Name"], item["Version"], item["License"]
        ok, reason = check_package(name, text)
        print(f"{'OK  ' if ok else 'FAIL'} {name}=={version}: {text} ({reason})")
        if not ok:
            failures.append(name)
    skipped = lock_only_packages([p["Name"] for p in packages])
    if skipped:
        print(f"note: in uv.lock but not installed on this platform (not checked): {', '.join(skipped)}")
    if failures:
        print(f"licence check FAILED: {', '.join(failures)}")
        return 1
    print(f"licence check passed: {len(packages)} packages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
