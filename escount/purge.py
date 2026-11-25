"""Delete what `escount` generated for one data set (case): models, feature cache, infer results, run records.

Shared by `escount purge` (CLI) and POST /training/{case}/purge (server). Never touches the image folder (Data/).
Default keeps the hand-made calibration (project.json, its backups, background_BW*.png, height_model.json);
`all_files=True` removes it too. AutoResearch records (research/, candidates/, Optuna studies) need `research=True`.
"""
import re
import shutil
from pathlib import Path

CASE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@~-]*$")      # same alphabet as the server's case ids
KEEP = ("project.json", "project_*.json", "background_BW*.png", "height_model.json")
TRACK_DIRS = ("default", "mae", "exact_hit", "ensemble")
STUDY_SUFFIXES = ("", "_mae", "_exact_hit", "_ensemble")


def _size(path):
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _matches(name, case, with_variants):
    """Exact case, its saved runs (<case>~<run>) and, for a base project, optionally every variant (<case>@...)."""
    return name == case or name.startswith(case + "~") or (with_variants and name.startswith(case + "@"))


def _children(root, case, with_variants, subdirs=()):
    """Entries of `root` (and `root/<subdir>`) that belong to the case; anything resolving outside `root` is skipped."""
    root = Path(root)
    if not root.is_dir():
        return []
    anchor = root.resolve()
    found = []
    for folder in (root, *(root / sub for sub in subdirs)):
        if not folder.is_dir():
            continue
        for path in folder.iterdir():
            resolved = path.resolve()
            if _matches(path.name, case, with_variants) and resolved != anchor and resolved.is_relative_to(anchor):
                found.append(path)
    return found


def _profile_items(folder, all_files, research):
    """Whole folder with `all_files`; otherwise every entry except the calibration files (and research/ without `research`)."""
    if all_files:
        return [folder]
    keep = lambda path: any(path.match(pattern) for pattern in KEEP) or (path.name == "research" and not research)
    return [path for path in folder.iterdir() if not keep(path)]


def plan_purge(case, *, profiles_root="outputs/projects", cache_root="outputs/features", results_root="outputs/infer",
               research_root="research", candidates_root="candidates", storage=None,
               with_variants=False, research=False, all_files=False):
    """-> [{"path", "kind", "bytes"}] of everything `purge_case` would remove (nothing is deleted)."""
    if not CASE_RE.match(case or ""):
        raise ValueError(f"invalid case id {case!r}")
    items = []
    for path in _children(profiles_root, case, with_variants):
        if path.is_dir():
            items += [{"path": item, "kind": "project"} for item in _profile_items(path, all_files, research)]
    for kind, root in (("features", cache_root), ("results", results_root)):
        items += [{"path": path, "kind": kind} for path in _children(root, case, with_variants)]
    if research:
        for kind, root in (("research", research_root), ("candidates", candidates_root)):
            items += [{"path": path, "kind": kind} for path in _children(root, case, with_variants, TRACK_DIRS)]
        from escount.research.daemon import DEFAULT_STORAGE, load_study

        storage = storage or DEFAULT_STORAGE
        names = [case + suffix for suffix in STUDY_SUFFIXES]
        if with_variants:                       # variants are research cases of their own (<case>@<variant>[_<track>])
            names += [name for name in _study_names(storage) if name.startswith(case + "@")]
        for name in dict.fromkeys(names):
            if load_study(name, storage) is not None:
                items.append({"path": f"optuna study {name}", "kind": "study", "name": name, "storage": storage})
    for item in items:
        item["bytes"] = _size(item["path"]) if isinstance(item["path"], Path) and item["path"].exists() else 0
    return items


def _study_names(storage):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        return [summary.study_name for summary in optuna.get_all_study_summaries(storage)]
    except Exception:                           # missing / unreadable db: nothing to delete
        return []


def purge_case(case, dry_run=True, **options):
    """Delete (or with dry_run only list) the case's generated data. Returns the plan; paths as strings."""
    items = plan_purge(case, **options)
    if not dry_run:
        for item in items:
            if item["kind"] == "study":
                import optuna

                optuna.delete_study(study_name=item["name"], storage=item["storage"])
            elif item["path"].is_dir():
                shutil.rmtree(item["path"])
            else:
                item["path"].unlink(missing_ok=True)
    return [{**item, "path": str(item["path"])} for item in items]
