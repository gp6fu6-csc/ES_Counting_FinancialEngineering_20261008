"""Presentation-ready research report (Task 9): `escount report`.

Writes <out>/<YYYYmmdd_HHMMSS>/ and replaces <out>/latest/ with a copy of it:
  <case>/          for each case (4 projects, general, general_no_ring):
                   best_params.json, results.tsv, progress.png, param_importance.png,
                   param_sensitivity.png, per_image.csv
  comparison/      project_vs_general.csv + .png, ring_inclusion.png, roi_count_vs_hit.png,
                   param_importance_heatmap.png, latency.png
  summary.md, report.html (self-contained: every figure as data:image/png;base64, tables inline)
Every listed file is always written; when a study has too little data an explanatory placeholder
figure / header-only table is produced instead. All figures: 16:9, 200 dpi, Microsoft JhengHei.
"""
import base64
import html
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from escount.models.train import GENERAL_CASES, RING_CASE, load_general_config, production_general
from escount.research import figures
from escount.research.daemon import (
    BEST_NAME, DEFAULT_STORAGE, PER_IMAGE_COLUMNS, PER_IMAGE_NAME, TSV_COLUMNS, TSV_NAME, adopted_hash, load_study,
    read_tsv,
)
from escount.research.evaluate import wilson_interval

PROJECT_CASES = ("20260904-thinkness02_postition1", "20260904-thinkness02_postition2", "Ring-type", "T-type")
DEFAULT_CASES = PROJECT_CASES + GENERAL_CASES
CASE_FILES = ("best_params.json", "results.tsv", "progress.png", "param_importance.png", "param_sensitivity.png",
              "per_image.csv")
COMPARISON_FILES = ("project_vs_general.csv", "project_vs_general.png", "ring_inclusion.png", "roi_count_vs_hit.png",
                    "param_importance_heatmap.png", "latency.png")
CORE_PARAMS = ("w", "D", "model", "roi_cap")
SERIES = (("project", "LOIO"), ("general", "LOIO"), ("general", "LOPO"), ("general_no_ring", "LOIO"),
          ("general_no_ring", "LOPO"))
COLOURS = ("#1f77b4", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#9467bd", "#8c564b")


def short(case):
    """Axis label: long dated project folder names -> their last '_' part (e.g. postition1)."""
    return case.rsplit("_", 1)[-1] if len(case) > 18 and case not in GENERAL_CASES else case


# ---------------------------------------------------------------- data access
def case_data(case, research_root, candidates_root, storage, profiles_root):
    best_path = Path(candidates_root) / case / BEST_NAME
    best = json.loads(best_path.read_text(encoding="utf-8")) if best_path.is_file() else None
    per_image_path = Path(candidates_root) / case / PER_IMAGE_NAME
    per_image = pd.read_csv(per_image_path) if per_image_path.is_file() else pd.DataFrame(columns=PER_IMAGE_COLUMNS)
    tsv = read_tsv(Path(research_root) / case / TSV_NAME)
    tsv["hit_rate"] = pd.to_numeric(tsv["hit_rate"], errors="coerce")
    tsv["roi_count"] = pd.to_numeric(tsv["roi_count"], errors="coerce")
    tsv["latency_ms"] = pd.to_numeric(tsv["latency_ms"], errors="coerce")
    adopted = adopted_hash(case, profiles_root)
    return {"case": case, "best": best, "per_image": per_image, "tsv": tsv, "study": load_study(case, storage),
            "adopted": best is not None and adopted == best["config_hash"], "adopted_hash": adopted}


def completed_trials(study):
    if study is None:
        return []
    import optuna

    return [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]


def importances(study):
    """(fANOVA importances dict, None) or (None, reason) when they cannot be estimated."""
    trials = completed_trials(study)
    if study is None:
        return None, "尚無 Optuna study (先執行 escount research)"
    if len(trials) < 2:
        return None, f"完成的 trial 只有 {len(trials)} 個；fANOVA 至少需要 2 個"
    if np.ptp([t.value for t in trials]) == 0:
        return None, f"{len(trials)} 個 trial 的命中率完全相同 ({trials[0].value:.3f})，目標值無變化，無法估計重要度"
    from optuna.importance import FanovaImportanceEvaluator, get_param_importances

    try:
        result = get_param_importances(study, evaluator=FanovaImportanceEvaluator(seed=0))
    except Exception as error:   # optuna raises for degenerate search spaces
        return None, f"fANOVA 無法計算：{error}"
    result = {k: float(v) for k, v in result.items() if np.isfinite(v)}
    if not result:
        return None, "所有 trial 沒有共同的參數，無法估計重要度"
    return result, None


def hit_stats(per_image, project, scheme):
    rows = per_image[(per_image["project"] == project) & (per_image["scheme"] == scheme)]
    if rows.empty:
        return None
    hits, total = int(rows["hit"].astype(bool).sum()), len(rows)
    low, high = wilson_interval(hits, total)
    return {"hits": hits, "total": total, "hit_rate": hits / total, "wilson_low": low, "wilson_high": high}


def ring_decision(data):
    """Same rule as training: exclude Ring-type only if it lowers the other projects' LOIO + LOPO hits."""
    general, no_ring = data.get("general"), data.get("general_no_ring")
    if not general or not no_ring or general["per_image"].empty or no_ring["per_image"].empty:
        return {"available": False, "text": "general / general_no_ring 尚無研究結果，無法判斷 Ring-type 是否納入"}
    rows, totals = [], {"with": 0, "without": 0}
    others = sorted(set(general["per_image"]["project"]) - {RING_CASE})
    for scheme in ("LOIO", "LOPO"):
        for project in others:
            with_ring = hit_stats(general["per_image"], project, scheme)
            without = hit_stats(no_ring["per_image"], project, scheme)
            if with_ring is None or without is None:
                continue
            rows.append({"scheme": scheme, "project": project, "with_ring": with_ring, "without_ring": without})
            totals["with"] += with_ring["hits"]
            totals["without"] += without["hits"]
    hurts = totals["with"] < totals["without"]
    text = (f"其他專案命中數 (LOIO+LOPO)：含 Ring {totals['with']}、不含 Ring {totals['without']} → "
            + ("Ring-type 拉低其他專案，建議 exclude_from_general = [\"Ring-type\"]" if hurts
               else "Ring-type 未拉低其他專案，建議保留於 General (exclude_from_general = [])"))
    return {"available": True, "rows": rows, "with_ring_hits": totals["with"], "without_ring_hits": totals["without"],
            "ring_hurts_others": hurts, "recommended_exclude_from_general": [RING_CASE] if hurts else [], "text": text}


# ---------------------------------------------------------------- per-case figures
def plot_progress(path, case, tsv):
    rows = tsv.dropna(subset=["hit_rate"])
    if rows.empty:
        return figures.placeholder(path, f"{case}：研究進度", "尚無完成的 trial (先執行 escount research)")
    fig, ax = figures.new_figure()
    kept = rows["status"] == "keep"
    ax.scatter(rows["trial"], rows["hit_rate"], s=60, color="#9e9e9e", label="discard", zorder=2)
    ax.scatter(rows.loc[kept, "trial"], rows.loc[kept, "hit_rate"], s=180, color="#d62728", marker="*",
               label="keep (新最佳)", zorder=3)
    best_so_far = rows["hit_rate"].cummax()
    ax.step(rows["trial"], best_so_far, where="post", color="#1f77b4", linewidth=2.5, label="best-so-far")
    for _, row in rows[kept].iterrows():
        ax.annotate(f"{row['hits/total']}  ROI={row['roi_count']:g}", (row["trial"], row["hit_rate"]),
                    textcoords="offset points", xytext=(8, 10), fontsize=13)
    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("trial")
    ax.set_ylabel("exact hit rate (LOIO)")
    ax.set_ylim(-0.03, 1.08)
    ax.set_title(f"{case}：命中率 vs trial（keep 標記、best-so-far 線）")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    return figures.save(fig, path)


def plot_importance(path, case, values, reason):
    if values is None:
        return figures.placeholder(path, f"{case}：參數重要度 (fANOVA)", reason)
    fig, ax = figures.new_figure()
    items = sorted(values.items(), key=lambda kv: kv[1])
    ax.barh([k for k, _ in items], [v for _, v in items], color="#1f77b4")
    for index, (_, value) in enumerate(items):
        ax.text(value, index, f" {value:.3f}", va="center", fontsize=13)
    ax.set_xlabel("importance (Optuna FanovaImportanceEvaluator, normalised)")
    ax.set_title(f"{case}：參數重要度 (fANOVA)")
    ax.grid(axis="x", alpha=0.3)
    return figures.save(fig, path)


def plot_sensitivity(path, case, study, values):
    trials = completed_trials(study)
    if not trials:
        return figures.placeholder(path, f"{case}：參數敏感度", "尚無完成的 trial")
    names = [k for k, _ in sorted(values.items(), key=lambda kv: -kv[1])][:4] if values else \
        [p for p in CORE_PARAMS if all(p in t.params for t in trials)]
    names = names or sorted(set.intersection(*(set(t.params) for t in trials)))[:4]
    fig, axes = figures.new_figure(2, 2)
    for ax, name in zip(axes.flat, names):
        xs = [t.params.get(name) for t in trials]
        ys = [t.value for t in trials]
        kept = [t.user_attrs.get("status") == "keep" for t in trials]
        if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in xs):
            positions = xs
        else:
            labels = sorted({str(x) for x in xs})
            positions = [labels.index(str(x)) for x in xs]
            ax.set_xticks(range(len(labels)), labels)
        ax.scatter(positions, ys, s=70, c=["#d62728" if k else "#1f77b4" for k in kept], alpha=0.85)
        ax.set_title(name)
        ax.set_ylabel("hit rate")
        ax.set_ylim(-0.03, 1.08)
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(names):]:
        ax.set_axis_off()
    fig.suptitle(f"{case}：命中率 vs 主要參數（紅 = keep）", fontsize=20)
    return figures.save(fig, path)


# ---------------------------------------------------------------- comparison figures
def comparison_rows(data, projects):
    rows = []
    for project in projects:
        sources = {"project": data.get(project), "general": data.get("general"),
                   "general_no_ring": data.get("general_no_ring")}
        for model, scheme in SERIES:
            source = sources[model]
            stats = hit_stats(source["per_image"], project, scheme) if source else None
            if stats is None:
                continue
            best = source["best"] or {}
            rows.append({"project": project, "model": model, "scheme": scheme, **stats,
                         "trial": best.get("trial"), "config_hash": best.get("config_hash")})
    return pd.DataFrame(rows, columns=["project", "model", "scheme", "hits", "total", "hit_rate", "wilson_low",
                                       "wilson_high", "trial", "config_hash"])


def plot_project_vs_general(path, table, projects):
    if table.empty:
        return figures.placeholder(path, "Project vs General", "尚無任何研究結果")
    fig, ax = figures.new_figure()
    width = 0.8 / len(SERIES)
    for index, (model, scheme) in enumerate(SERIES):
        part = table[(table["model"] == model) & (table["scheme"] == scheme)].set_index("project")
        xs, ys, low, high, labels = [], [], [], [], []
        for position, project in enumerate(projects):
            if project in part.index:
                row = part.loc[project]
                xs.append(position + (index - (len(SERIES) - 1) / 2) * width)
                ys.append(row["hit_rate"])
                low.append(row["hit_rate"] - row["wilson_low"])
                high.append(row["wilson_high"] - row["hit_rate"])
                labels.append(f"{int(row['hits'])}/{int(row['total'])}")
        if xs:
            ax.bar(xs, ys, width, yerr=[low, high], capsize=4, color=COLOURS[index], label=f"{model} {scheme}")
            for x, y, top, label in zip(xs, ys, high, labels):
                ax.text(x, y + top + 0.02, label, ha="center", va="bottom", fontsize=11, rotation=90)
    ax.set_xticks(range(len(projects)), [short(p) for p in projects])
    ax.set_ylabel("exact hit rate (誤差線 = Wilson 95% CI)")
    ax.set_ylim(0, 1.35)
    ax.set_title("Project model vs General vs General(no Ring)：LOIO / LOPO 命中率")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=3, loc="upper right")
    return figures.save(fig, path)


def plot_ring_inclusion(path, decision):
    if not decision["available"] or not decision["rows"]:
        return figures.placeholder(path, "Ring-type 納入 General 的影響", decision["text"])
    fig, axes = figures.new_figure(1, 2)
    for ax, scheme in zip(axes, ("LOIO", "LOPO")):
        rows = [r for r in decision["rows"] if r["scheme"] == scheme]
        names = [short(r["project"]) for r in rows]
        positions = np.arange(len(rows))
        for offset, key, label, colour in ((-0.2, "with_ring", "general (含 Ring)", "#ff7f0e"),
                                           (0.2, "without_ring", "general_no_ring (不含 Ring)", "#2ca02c")):
            values = [r[key]["hit_rate"] for r in rows]
            ax.bar(positions + offset, values, 0.4, color=colour, label=label)
            for x, r in zip(positions + offset, rows):
                ax.text(x, r[key]["hit_rate"] + 0.02, f"{r[key]['hits']}/{r[key]['total']}", ha="center", fontsize=12)
        ax.set_xticks(positions, names)
        ax.set_ylim(0, 1.35)
        ax.set_title(f"{scheme}：其他專案的命中率")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper center", ncol=2, fontsize=13)
    fig.suptitle(decision["text"], fontsize=17)
    return figures.save(fig, path)


def plot_roi_vs_hit(path, data):
    fig, ax = figures.new_figure()
    plotted = False
    for index, (case, item) in enumerate(data.items()):
        rows = item["tsv"].dropna(subset=["hit_rate", "roi_count"])
        if rows.empty:
            continue
        plotted = True
        colour = COLOURS[index % len(COLOURS)]
        ax.scatter(rows["roi_count"], rows["hit_rate"], s=60, color=colour, alpha=0.6, label=short(case))
        if item["best"]:
            result = item["best"]["result"]
            ax.scatter([result["roi_count"]], [result["hit_rate"]], s=400, marker="*", color=colour,
                       edgecolor="black", zorder=3)
    if not plotted:
        return figures.placeholder(path, "ROI 數量 vs 命中率", "尚無完成的 trial")
    ax.set_xlabel("ROI count per image")
    ax.set_ylabel("exact hit rate (LOIO)")
    ax.set_ylim(-0.03, 1.08)
    ax.set_title("ROI 數量 vs 命中率（所有 trial；★ = 各案例最佳）")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    return figures.save(fig, path)


def plot_importance_heatmap(path, cases, importance):
    params = sorted({p for values in importance.values() if values for p in values})
    if not params:
        return figures.placeholder(path, "參數重要度熱圖 (param × case)", "所有案例皆無法估計 fANOVA 重要度（trial 太少或命中率無變化）")
    matrix = np.full((len(params), len(cases)), np.nan)
    for column, case in enumerate(cases):
        for row, param in enumerate(params):
            if importance.get(case) and param in importance[case]:
                matrix[row, column] = importance[case][param]
    fig, ax = figures.new_figure()
    image = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap="viridis", vmin=0, vmax=max(np.nanmax(matrix), 1e-9))
    for row in range(len(params)):
        for column in range(len(cases)):
            value = matrix[row, column]
            ax.text(column, row, "—" if np.isnan(value) else f"{value:.2f}", ha="center", va="center", fontsize=11,
                    color="white" if not np.isnan(value) and value < 0.5 * np.nanmax(matrix) else "black")
    ax.set_xticks(range(len(cases)), [short(c) for c in cases])
    ax.set_yticks(range(len(params)), params)
    fig.colorbar(image, ax=ax, label="fANOVA importance")
    ax.set_title("參數重要度熱圖 (param × case；— = 無法估計 / 不適用)")
    return figures.save(fig, path)


def plot_latency(path, data, profiles_root):
    fig, axes = figures.new_figure(1, 2)
    cases = [c for c, item in data.items() if item["best"]]
    values = [data[c]["best"]["result"]["latency_ms"] for c in cases]
    if cases:
        axes[0].bar(range(len(cases)), values, color="#1f77b4")
        for x, value in enumerate(values):
            axes[0].text(x, value, f"{value:.2f}", ha="center", va="bottom", fontsize=12)
        axes[0].set_xticks(range(len(cases)), [short(c) for c in cases], rotation=15, ha="right")
    else:
        axes[0].text(0.5, 0.5, "尚無研究結果", ha="center", transform=axes[0].transAxes)
    axes[0].set_ylabel("ms / image")
    axes[0].set_title("研究最佳設定：特徵 → 預測 延遲\n(評估器，不含讀圖/旋轉)")
    bench = {}
    for case in data:
        path_json = Path(profiles_root) / case / "benchmark.json"
        if path_json.is_file():
            report = json.loads(path_json.read_text(encoding="utf-8"))
            bench[case] = (report.get("median_ms"), report.get("max_ms"))
    if bench:
        names = list(bench)
        positions = np.arange(len(names))
        axes[1].bar(positions - 0.2, [bench[n][0] for n in names], 0.4, label="median", color="#2ca02c")
        axes[1].bar(positions + 0.2, [bench[n][1] for n in names], 0.4, label="max", color="#d62728")
        axes[1].axhline(1000, color="black", linestyle="--", label="1 s 上限")
        axes[1].set_xticks(positions, [short(n) for n in names], rotation=15, ha="right")
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "無 benchmark.json (escount benchmark)", ha="center", transform=axes[1].transAxes)
    axes[1].set_ylabel("ms / image")
    axes[1].set_title("正式設定端到端延遲\n(escount benchmark，單執行緒 CPU)")
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
    return figures.save(fig, path)


# ---------------------------------------------------------------- tables / text
def pct(value):
    return "—" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{100 * value:.1f}%"


def case_summary(case, item):
    best = item["best"]
    trials = int(item["tsv"]["hit_rate"].notna().sum())
    if not best:
        return {"案例": case, "最佳 trial": "—", "命中 hits/total": "—", "命中率": "—", "Wilson 95% CI": "—",
                "ROI 數": "—", "延遲 ms/影像": "—", "trials": trials, "已採用": "否", "config_hash": "—"}
    result = best["result"]
    low, high = result["wilson95"]
    return {"案例": case, "最佳 trial": best["trial"], "命中 hits/total": f"{result['hits']}/{result['total']}",
            "命中率": pct(result["hit_rate"]), "Wilson 95% CI": f"{pct(low)} – {pct(high)}",
            "ROI 數": f"{result['roi_count']:g}", "延遲 ms/影像": f"{result['latency_ms']:.3f}", "trials": trials,
            "已採用": "是" if item["adopted"] else "否", "config_hash": best["config_hash"]}


def markdown_table(frame):
    if frame.empty:
        return "(無資料)\n"
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *body]) + "\n"


def lopo_table(data):
    rows = []
    for case in GENERAL_CASES:
        best = (data.get(case) or {}).get("best")
        if best and best.get("lopo"):
            lopo = best["lopo"]
            rows.append({"案例": case, "LOPO hits/total": f"{lopo['hits']}/{lopo['total']}",
                         "命中率": pct(lopo["hit_rate"]),
                         "Wilson 95% CI": f"{pct(lopo['wilson95'][0])} – {pct(lopo['wilson95'][1])}",
                         "各專案": ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in lopo["per_case"].items())})
    return pd.DataFrame(rows)


def write_summary(path, created, summary, comparison, lopo, decision, config):
    lines = ["# ESCounting V4 研究報告 (autoresearch)", "",
             f"產生時間：{created}", "",
             "- 唯一指標：exact hit rate，round(預測) == GT（無 MAE、無 ±1 容許）。",
             "- 排序：(-hit_rate, roi_count, latency_ms)；驗證：LOIO（General 另記 LOPO），評估器固定不改。",
             "- ROI 規則：每列距 background_BW 有效區域左右邊界 ≥ 50 px；D > 2w/3（重疊 < w/3）；垂直方向不平滑。",
             "- 延遲欄位 = 評估器中「特徵 → 預測」每張影像毫秒；端到端延遲見 comparison/latency.png。", "",
             "## 各案例最佳結果", "", markdown_table(summary),
             "## General 的 LOPO（跨專案）", "", markdown_table(lopo),
             "## Ring-type 決策", "", decision["text"], "",
             f"- configs/general.json exclude_from_general = {json.dumps(config['exclude_from_general'], ensure_ascii=False)}；"
             f"正式 General 模型 = `{production_general(config)}`", "",
             "## Project vs General（Wilson 95% CI）", "", markdown_table(comparison.round(4)),
             "## 檔案", "",
             "- 每個案例資料夾：" + "、".join(CASE_FILES),
             "- comparison/：" + "、".join(COMPARISON_FILES),
             "- report.html：所有圖表與表格內嵌的單一檔案（可離線開啟）", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _img(path, alt):
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(alt)}"><figcaption>{html.escape(alt)}</figcaption></figure>'


def _table(frame):
    if frame.empty:
        return "<p>(無資料)</p>"
    return frame.to_html(index=False, escape=True, border=0, na_rep="")


def write_html(path, out_dir, created, cases, summary, comparison, lopo, decision):
    style = ("body{font-family:'Microsoft JhengHei',sans-serif;margin:24px;background:#fafafa;color:#222}"
             "table{border-collapse:collapse;margin:8px 0 16px;font-size:14px}"
             "th,td{border:1px solid #ccc;padding:4px 8px;text-align:left}th{background:#e8eef6}"
             "img{max-width:100%;border:1px solid #ddd;background:#fff}pre{background:#fff;border:1px solid #ddd;"
             "padding:8px;overflow:auto;max-height:420px}section{margin-bottom:40px}")
    parts = ["<!DOCTYPE html>", '<html lang="zh-Hant"><head><meta charset="utf-8">',
             "<title>ESCounting V4 研究報告</title>", f"<style>{style}</style></head><body>",
             "<h1>ESCounting V4 研究報告 (autoresearch)</h1>", f"<p>產生時間：{html.escape(created)}</p>",
             "<p>唯一指標 exact hit rate (round(預測) == GT)；排序 (-hit_rate, roi_count, latency_ms)；"
             "LOIO（General 另記 LOPO）；ROI 每列距有效區域邊界 ≥ 50 px、D &gt; 2w/3、垂直方向不平滑。</p>",
             "<section><h2>各案例最佳結果</h2>", _table(summary), "<h2>General 的 LOPO（跨專案）</h2>", _table(lopo),
             f"<h2>Ring-type 決策</h2><p>{html.escape(decision['text'])}</p></section>",
             "<section><h2>比較 (comparison)</h2>", _table(comparison.round(4))]
    for name in COMPARISON_FILES:
        if name.endswith(".png"):
            parts.append(_img(out_dir / "comparison" / name, f"comparison/{name}"))
    parts.append("</section>")
    for case in cases:
        case_dir = out_dir / case
        parts.append(f"<section><h2>{html.escape(case)}</h2>")
        for name in ("progress.png", "param_importance.png", "param_sensitivity.png"):
            parts.append(_img(case_dir / name, f"{case}/{name}"))
        parts.append("<h3>best_params.json</h3><pre>"
                     + html.escape((case_dir / "best_params.json").read_text(encoding="utf-8")) + "</pre>")
        parts.append("<h3>results.tsv</h3>" + _table(pd.read_csv(case_dir / "results.tsv", sep="\t", dtype=str)))
        parts.append("<h3>per_image.csv</h3>" + _table(pd.read_csv(case_dir / "per_image.csv")))
        parts.append("</section>")
    parts.append("</body></html>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


# ---------------------------------------------------------------- driver
def _out_dir(out_root, timestamp):
    out_dir = Path(out_root) / timestamp
    suffix = 2
    while out_dir.exists():
        out_dir = Path(out_root) / f"{timestamp}_{suffix}"
        suffix += 1
    return out_dir


def build_report(out_root="outputs/research_report", cases=DEFAULT_CASES, research_root="research",
                 candidates_root="candidates", storage=DEFAULT_STORAGE, profiles_root="outputs/projects",
                 config_path="configs/general.json", log=print):
    """Write the timestamped report directory and refresh <out_root>/latest/. Returns the timestamp dir."""
    figures.setup_style()
    created = datetime.now()
    out_dir = _out_dir(out_root, created.strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True)
    cases = list(cases)
    data = {case: case_data(case, research_root, candidates_root, storage, profiles_root) for case in cases}
    importance = {}
    for case, item in data.items():
        case_dir = out_dir / case
        case_dir.mkdir()
        best = item["best"] or {"case": case, "status": "尚無 keep 的 trial（先執行 escount research）"}
        (case_dir / "best_params.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
        tsv_source = Path(research_root) / case / TSV_NAME
        if tsv_source.is_file():
            shutil.copy2(tsv_source, case_dir / "results.tsv")
        else:
            (case_dir / "results.tsv").write_text("\t".join(TSV_COLUMNS) + "\n", encoding="utf-8")
        item["per_image"].to_csv(case_dir / "per_image.csv", index=False, encoding="utf-8")
        values, reason = importances(item["study"])
        importance[case] = values
        plot_progress(case_dir / "progress.png", case, item["tsv"])
        plot_importance(case_dir / "param_importance.png", case, values, reason)
        plot_sensitivity(case_dir / "param_sensitivity.png", case, item["study"], values)
        log(f"[report] {case}: {len(item['tsv'])} trials, best "
            f"{(item['best'] or {}).get('result', {}).get('hits', '-')}/{(item['best'] or {}).get('result', {}).get('total', '-')}")
    comparison_dir = out_dir / "comparison"
    comparison_dir.mkdir()
    projects = [c for c in cases if c not in GENERAL_CASES]
    for case in GENERAL_CASES:
        if case in data:
            projects += [p for p in data[case]["per_image"]["project"].unique() if p not in projects]
    comparison = comparison_rows(data, projects)
    comparison.to_csv(comparison_dir / "project_vs_general.csv", index=False, encoding="utf-8")
    plot_project_vs_general(comparison_dir / "project_vs_general.png", comparison, projects)
    decision = ring_decision(data)
    plot_ring_inclusion(comparison_dir / "ring_inclusion.png", decision)
    plot_roi_vs_hit(comparison_dir / "roi_count_vs_hit.png", data)
    plot_importance_heatmap(comparison_dir / "param_importance_heatmap.png", cases, importance)
    plot_latency(comparison_dir / "latency.png", data, profiles_root)
    summary = pd.DataFrame([case_summary(case, data[case]) for case in cases])
    lopo = lopo_table(data)
    config = load_general_config(config_path)
    stamp = created.isoformat(timespec="seconds")
    write_summary(out_dir / "summary.md", stamp, summary, comparison, lopo, decision, config)
    write_html(out_dir / "report.html", out_dir, stamp, cases, summary, comparison, lopo, decision)
    latest = Path(out_root) / "latest"
    if latest.exists():
        shutil.rmtree(latest)
    shutil.copytree(out_dir, latest)
    log(f"[report] wrote {out_dir.resolve()} and {latest.resolve()}")
    return out_dir
