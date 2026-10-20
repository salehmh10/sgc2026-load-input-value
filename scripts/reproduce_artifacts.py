"""Regenerate scientific figures and tables from publication aggregates.

Defaults to the immutable historical summaries. An explicit --data-dir accepts
aggregates exported from a newly completed full experiment. This renderer does
not train models or independently recompute bootstrap intervals.
Run from any working directory; output paths are repository-relative by default.
"""
from __future__ import annotations

import argparse
import csv
import math
import shutil
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import warnings
from decimal import Decimal, ROUND_HALF_EVEN

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
from matplotlib.patches import Patch, Rectangle
from matplotlib import font_manager
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "reference"
REFERENCE = DATA
NEW_EXECUTION = False
OUTPUT = ROOT / "results" / "reproduced"
OUT, TABLES, VALIDATION = OUTPUT / "figures", OUTPUT / "tables", OUTPUT / "validation"
FAMILIES = ["Ridge-ARX", "XGBoost", "TiDE"]
FOLDS = ["warm_late_summer", "cold"]
ARMS = ["FULL", "NO_DERIVED", "NO_TEMP", "NO_HUMIDITY", "NO_CALENDAR", "NO_WEATHER", "FULL_PROXY"]
PRIMARY = ["NO_WEATHER", "NO_TEMP"]
COLORS = {"Ridge-ARX": "#0072B2", "XGBoost": "#D55E00", "TiDE": "#009E73"}
ARM_LABELS = {"FULL": r"\full", "NO_DERIVED": "No derived", "NO_TEMP": "No temperature", "NO_HUMIDITY": "No humidity", "NO_CALENDAR": "No calendar", "NO_WEATHER": "No weather", "FULL_PROXY": r"\proxy"}
SEEDS, SEED_MARKERS = ["42", "314", "2718"], ["o", "s", "^"]
DRAWN = []


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name: str) -> list[dict]:
    with (DATA / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def style() -> None:
    try:
        font_manager.findfont("Times New Roman", fallback_to_default=False)
        family = "Times New Roman"
    except ValueError:
        family = "STIXGeneral"
        warnings.warn("Times New Roman unavailable; using STIXGeneral.")
    plt.rcParams.update({"font.family": family, "font.size": 8.5, "axes.titlesize": 9,
                        "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                        "legend.fontsize": 8, "axes.linewidth": 0.55, "lines.linewidth": 0.8,
                        "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "stix",
                        "savefig.facecolor": "white", "axes.unicode_minus": True})


def clean(ax, grid=True) -> None:
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("0.55")
    ax.tick_params(axis="both", length=2.5, width=0.5, color="0.45")
    ax.tick_params(axis="y", length=0)
    if grid:
        ax.grid(axis="x", color="0.90", linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)


def save(fig, stem: str) -> None:
    fig.savefig(OUT / (stem + ".pdf"), metadata={"Creator": "scripts/reproduce_artifacts.py", "CreationDate": None, "ModDate": None})
    fig.savefig(OUT / (stem + ".png"), dpi=400, metadata={"Software": "scripts/reproduce_artifacts.py"})
    plt.close(fig)


def primary_value(contrasts: list[dict], seeds: list[dict]) -> None:
    lookup = {(r["fold"], r["family"], r["arm"]): r for r in contrasts if r["partition"] == "Evaluation"}
    seed_lookup = {(r["fold"], r["family"], r["arm"], r["seed"]): r for r in seeds}
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.70), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.165, right=0.986, bottom=0.235, top=0.79, wspace=0.13)
    rows = [(arm, family) for arm in PRIMARY for family in FAMILIES]
    labels = [family for arm, family in rows]
    for ax, fold, title in zip(axes, FOLDS, ["(a) Warm evaluation", "(b) Cold evaluation"]):
        ax.axhspan(-0.5, 2.5, facecolor="#F1F4F6", zorder=-1)
        for i, (arm, family) in enumerate(rows):
            r = lookup[(fold, family, arm)]
            delta, low, high = [float(r[k]) for k in ["delta_percentage_points", "block3_CI_low", "block3_CI_high"]]
            center_y = i - (0.09 if family != "Ridge-ARX" else 0)
            if NEW_EXECUTION:
                # A bootstrap interval need not contain its point estimate.
                ax.hlines(center_y, low, high, color=COLORS[family], linewidth=1.05, zorder=4)
                ax.vlines([low, high], center_y-.05, center_y+.05, color=COLORS[family], linewidth=.7, zorder=4)
                ax.plot(delta, center_y, "D", ms=3.7, color=COLORS[family], zorder=4)
            else:
                ax.errorbar(delta, center_y, xerr=[[delta-low], [high-delta]], fmt="D", ms=3.7,
                            color=COLORS[family], ecolor=COLORS[family], capsize=2.1, elinewidth=1.05, capthick=0.7, zorder=4)
            DRAWN.append({"figure": "i15_primary_value", "fold": fold, "family": family, "arm": arm,
                          "kind": r["seed"], "x": delta, "y": center_y, "CI_low": low, "CI_high": high})
            if family != "Ridge-ARX":
                for seed, marker in zip(SEEDS, SEED_MARKERS):
                    value = float(seed_lookup[(fold, family, arm, seed)]["delta_percentage_points"])
                    ax.scatter(value, i+0.18, marker=marker, s=15, facecolor="white", edgecolor=COLORS[family], linewidth=0.75, zorder=5)
                    DRAWN.append({"figure": "i15_primary_value", "fold": fold, "family": family, "arm": arm,
                                  "kind": seed, "x": value, "y": i+0.18})
        ax.axvline(0, color="0.35", lw=0.8, ls=(0, (3, 2)), zorder=2)
        ax.set_title(title, loc="left", pad=4, fontweight="bold")
        ax.set_xlim(-3.15, 2.25)
        ax.set_xticks([-3, -2, -1, 0, 1, 2])
        ax.set_ylim(5.55, -0.55)
        ax.set_yticks(range(6), labels)
        clean(ax)
    if NEW_EXECUTION:
        values = [0.] + [float(row[field]) for row in contrasts if row["partition"] == "Evaluation"
                         and row["arm"] in PRIMARY for field in ("delta_percentage_points", "block3_CI_low", "block3_CI_high")]
        values += [float(row["delta_percentage_points"]) for row in seeds]
        low, high = min(values), max(values)
        pad = max((high-low)*.1, .1)
        axes[0].set_xlim(low-pad, high+pad)
        axes[0].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(7))
    axes[0].text(-0.245, 0.755, "No weather", transform=axes[0].transAxes, ha="center", va="center", rotation=90, fontsize=8.5)
    axes[0].text(-0.245, 0.265, "No temperature", transform=axes[0].transAxes, ha="center", va="center", rotation=90, fontsize=8.5)
    handles = [Line2D([0], [0], color="0.25", marker="D", ms=3.7, lw=1, label="Estimate + 95% block CI")]
    handles.extend(Line2D([0], [0], color="0.25", marker=marker, markerfacecolor="white", ls="none", ms=3.6, label="Seed " + seed)
                   for seed, marker in zip(SEEDS, SEED_MARKERS))
    fig.legend(handles=handles, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.55, 1.015), columnspacing=1.6, handlelength=1.5)
    fig.text(0.56, 0.067, "Δ mean daily WAPE (pp), condition − FULL; positive = worse", ha="center", fontsize=8.5)
    save(fig, "i15_primary_value")


def mean_peak(contrasts: list[dict]) -> None:
    lookup = {(r["fold"], r["family"], r["arm"]): r for r in contrasts if r["partition"] == "Evaluation"}
    fig, axes = plt.subplots(2, 1, figsize=(3.48, 3.05), sharex=True)
    fig.subplots_adjust(left=0.21, right=0.975, bottom=0.17, top=0.77, hspace=0.40)
    titles = ["(a) Warm · 10 complete days", "(b) Cold · 25 complete days"]
    if NEW_EXECUTION:
        titles = [f"({letter}) {label} · {lookup[(fold, 'Ridge-ARX', 'NO_WEATHER')]['CompleteDays']} complete days"
                  for letter, label, fold in zip("ab", ["Warm", "Cold"], FOLDS)]
    for ax, fold, title in zip(axes, FOLDS, titles):
        ax.axhline(0, color="0.45", lw=0.6, ls=(0, (3, 2)))
        ax.axvline(0, color="0.45", lw=0.6, ls=(0, (3, 2)))
        ax.plot(0, 0, "+", color="black", ms=5, mew=0.8, zorder=4)
        for family in FAMILIES:
            for arm, marker in zip(PRIMARY, ["o", "^"]):
                r = lookup[(fold, family, arm)]
                x, y = float(r["delta_percentage_points"]), float(r["delta_PeakMagnitudeMAE_kW"])
                ax.scatter(x, y, marker=marker, s=23, facecolor=COLORS[family], edgecolor="white", linewidth=0.4, zorder=4)
                DRAWN.append({"figure": "i15_mean_peak_tradeoff", "fold": fold, "family": family, "arm": arm, "kind": r["seed"], "x": x, "y": y})
        ax.set_title(title, loc="left", pad=2, fontsize=8.5, fontweight="bold")
        ax.set_xlim(-2.65, 0.65)
        ax.set_xticks([-2.5, -2, -1.5, -1, -0.5, 0, 0.5])
        if fold == "warm_late_summer":
            ax.set_ylim(-400, 760)
            ax.set_yticks([-250, 0, 250, 500])
        else:
            ax.set_ylim(-2050, 700)
            ax.set_yticks([-2000, -1000, 0])
        clean(ax, grid=False)
        if NEW_EXECUTION:
            yvalues = [0.] + [float(lookup[(fold, family, arm)]["delta_PeakMagnitudeMAE_kW"])
                             for family in FAMILIES for arm in PRIMARY]
            low, high = min(yvalues), max(yvalues)
            pad = max((high-low)*.15, 1.)
            ax.set_ylim(low-pad, high+pad)
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    if NEW_EXECUTION:
        values = [0.] + [float(row["delta_percentage_points"]) for row in contrasts
                         if row["partition"] == "Evaluation" and row["arm"] in PRIMARY]
        low, high = min(values), max(values)
        pad = max((high-low)*.15, .1)
        axes[0].set_xlim(low-pad, high+pad)
        axes[0].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
    family_handles = [Line2D([0], [0], marker="o", color=COLORS[f], ls="none", ms=4, label=f) for f in FAMILIES]
    condition_handles = [Line2D([0], [0], marker=m, color="0.25", ls="none", ms=4, label=l)
                         for m, l in [("o", "No weather"), ("^", "No temperature")]]
    fig.legend(handles=family_handles, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.55, 1.005), columnspacing=1, handletextpad=0.4)
    fig.legend(handles=condition_handles, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.55, 0.93), columnspacing=1.3, handletextpad=0.4)
    fig.text(0.025, 0.50, "Δ peak magnitude MAE (kW)", rotation=90, ha="center", va="center", fontsize=8.5)
    fig.text(0.59, 0.035, "Δ mean daily WAPE (pp)", ha="center", fontsize=8.5)
    save(fig, "i15_mean_peak_tradeoff")


def tables(accuracy: list[dict], contrasts: list[dict]) -> None:
    scores = {(r["fold"], r["family"], r["arm"]): r for r in accuracy if r["partition"] == "Evaluation"}
    lines = [r"\begin{table*}[t]", r"\caption{Evaluation mean daily WAPE (\%). Ridge-ARX is deterministic; XGBoost and TiDE use the mean prediction of three seeds. Each window contains 28 days and 626 (warm) or 669 (cold) observed targets. Persistence references use the same reconstructed history and observed targets.}", r"\label{tab:results}", r"\centering", r"\begin{tabular}{@{}lrrrcrrr@{}}", r"\toprule", r"& \multicolumn{3}{c}{Warm} && \multicolumn{3}{c}{Cold}\\", r"\cmidrule(lr){2-4}\cmidrule(l){6-8}", r"Condition & Ridge-ARX & XGBoost & TiDE && Ridge-ARX & XGBoost & TiDE\\", r"\midrule"]
    if NEW_EXECUTION:
        counts = "; ".join(f"{label}: {scores[(fold, 'Ridge-ARX', 'FULL')]['Days']} days, "
                           f"{scores[(fold, 'Ridge-ARX', 'FULL')]['ObservedHours']} observed targets"
                           for label, fold in zip(["warm", "cold"], FOLDS))
        lines[1] = (r"\caption{Evaluation mean daily WAPE (\%). Ridge-ARX is deterministic; "
                    "XGBoost and TiDE use the mean prediction of three seeds. "
                    + counts + ". Persistence references use the same reconstructed history and observed targets.}")
    for arm in ARMS:
        cells = [[str(Decimal(scores[(fold, family, arm)]["MeanDailyWAPE"]).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_EVEN)) for family in FAMILIES] for fold in FOLDS]
        lines.append(ARM_LABELS[arm] + " & " + " & ".join(cells[0]) + " && " + " & ".join(cells[1]) + r"\\")
    lines.append(r"\midrule")
    references = read("i15_simple_references.csv")
    for reference in references:
        label, warm, cold = [reference[k] for k in ["reference", "warm_mean_daily_WAPE_percent", "cold_mean_daily_WAPE_percent"]]
        if NEW_EXECUTION:
            warm, cold = [str(Decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN))
                          for value in (warm, cold)]
        lines.append(label + r" & \multicolumn{3}{c}{" + warm + r"} && \multicolumn{3}{c}{" + cold + r"}\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (TABLES / "i15_accuracy.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    differences = {(r["fold"], r["family"], r["arm"], r["partition"]): r for r in contrasts}
    lines = [r"\begin{table}[t]", r"\caption{Changes in mean daily WAPE relative to \full\ (pp). W: no weather; T: no temperature. Val: validation; Eval: evaluation. Positive differences favor \full; $+0.00$ denotes a rounded positive value.}", r"\label{tab:contrasts}", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule", r"&& \multicolumn{2}{c}{Warm} & \multicolumn{2}{c}{Cold}\\", r"\cmidrule(lr){3-4}\cmidrule(l){5-6}", r"Model & Removed & Val & Eval & Val & Eval\\", r"\midrule"]
    for family in FAMILIES:
        for j, arm in enumerate(PRIMARY):
            cells = [f"${Decimal(differences[(fold, family, arm, part)]['delta_percentage_points']).quantize(Decimal('0.01'), rounding=ROUND_HALF_EVEN):+.2f}$"
                     for fold in FOLDS for part in ["Validation", "Evaluation"]]
            lines.append((family if j == 0 else "") + " & " + ("W" if arm == "NO_WEATHER" else "T") + " & " + " & ".join(cells) + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (TABLES / "i15_validation.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_points(contrasts: list[dict], seeds: list[dict]) -> None:
    lookup = {(r["fold"], r["family"], r["arm"], r["seed"]): r for r in contrasts + seeds if r["partition"] == "Evaluation"}
    for point in DRAWN:
        r = lookup[(point["fold"], point["family"], point["arm"], point["kind"])]
        assert point["x"] == float(r["delta_percentage_points"])
        if point["figure"] == "i15_mean_peak_tradeoff":
            assert point["y"] == float(r["delta_PeakMagnitudeMAE_kW"])
        if "CI_low" in point:
            assert point["CI_low"] == float(r["block3_CI_low"])
            assert point["CI_high"] == float(r["block3_CI_high"])
    assert len(DRAWN) == 48



def study_windows() -> None:
    rows = read("study_windows.csv")
    lookup = {(r["fold"], r["partition"]): r for r in rows}
    colors = {"train": "#CDD2D6", "validation": "#7FAFC8", "evaluation": "#D67F43"}
    fig = plt.figure(figsize=(7.16, 2.2))
    ax = fig.add_axes([0.085, 0.30, 0.70, 0.53])
    summaries = fig.add_axes([0.815, 0.30, 0.18, 0.53])
    for fold, y in zip(FOLDS, [1, 0]):
        for part in ["train", "validation", "evaluation"]:
            r = lookup[(fold, part)]
            start = mdates.date2num(datetime.fromisoformat(r["start"]))
            end = mdates.date2num(datetime.fromisoformat(r["end_exclusive"]))
            ax.barh(y, end-start, left=start, height=0.32, color=colors[part], edgecolor="none", zorder=2)
            if part == "train":
                ax.text((start+end)/2, y, f"Train · {r['days']} days", ha="center", va="center", fontsize=8.2)
            elif part == "validation":
                ax.text((start+end)/2, y+0.28, "Val", ha="center", va="center", fontsize=8)
            else:
                ax.text((start+end)/2, y+0.28, "Eval", ha="center", va="center", fontsize=8)
        r = lookup[(fold, "evaluation")]
        start = datetime.fromisoformat(r["start"])
        last = datetime.fromisoformat(r["end_exclusive"]) - timedelta(days=1)
        text = (f"{start:%b} {start.day}–{last.day}, {start.year}\n"
                f"Median T: {float(r['temperature_median_C']):g} °C\n"
                f"Range: {float(r['temperature_min_C']):g} to {float(r['temperature_max_C']):g} °C")
        summaries.text(0, y, text, va="center", ha="left", fontsize=8.2, linespacing=1.22)

    warm = lookup[("warm_late_summer", "evaluation")]
    x0, x1 = [mdates.date2num(datetime.fromisoformat(warm[k])) for k in ["start", "end_exclusive"]]
    ax.add_patch(Rectangle((x0, -0.16), x1-x0, 0.32, facecolor="none", edgecolor="#A15421", hatch="////", linewidth=0.5, zorder=3))
    for x in [x0, x1]:
        ax.plot([x, x], [0.17, 0.82], linestyle=(0, (2, 2)), linewidth=0.65, color="#A15421", zorder=1)
    ax.text((x0+x1)/2, 0.50, "Overlap", ha="center", va="center", fontsize=8, color="#884115",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})
    ax.set_ylim(-0.40, 1.45)
    summaries.set_ylim(ax.get_ylim())
    summaries.set_axis_off()
    ax.set_yticks([1, 0], ["Warm", "Cold"])
    ax.set_xlim(datetime(2024, 7, 22), datetime(2026, 1, 10))
    if NEW_EXECUTION:
        start = min(datetime.fromisoformat(row["start"]) for row in rows)
        end = max(datetime.fromisoformat(row["end_exclusive"]) for row in rows)
        pad = max((end-start) * .03, timedelta(days=1))
        ax.set_xlim(start-pad, end+pad)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[2, 5, 8, 11]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    clean(ax, grid=False)
    handles = [Patch(facecolor=colors[k], edgecolor="none", label=k.title()) for k in colors]
    fig.legend(handles=handles, ncol=3, frameon=False, loc="upper left", bbox_to_anchor=(0.08, 1.0), handlelength=1.5, columnspacing=2.0)
    fig.text(0.815, 0.92, "Evaluation weather", fontsize=9, weight="bold")
    window_days = {row["days"] for row in rows if row["partition"] != "train"}
    caption = (f"Each validation/evaluation window: {next(iter(window_days))} days. " if len(window_days) == 1
               else "Validation/evaluation lengths are recorded in study_windows.csv. ")
    fig.text(0.085, 0.035, caption + "Hatched dates: warm evaluation reused in cold training.", fontsize=8)
    save(fig, "study_windows")


def verify_reference() -> dict:
    """Check admitted files and scientific identities without private inputs."""
    expected = json.loads((DATA / "SHA256SUMS.json").read_text(encoding="utf-8"))
    if NEW_EXECUTION:
        manifest = json.loads((DATA / "publication-export.json").read_text(encoding="utf-8"))
        if manifest.get("origin") != "new_full_experiment" or manifest.get("historical_reference") is not False:
            raise ValueError("--data-dir requires aggregates from a new completed full experiment.")
        names = {"i15_accuracy.csv", "i15_contrasts.csv", "i15_primary_seed_points.csv",
                 "i15_simple_references.csv", "study_windows.csv"}
        if set(expected) != names or set(manifest["exports"]) != names:
            raise ValueError("New-execution export does not contain the five publication tables.")
        for name in names:
            if manifest["exports"][name]["sha256"] != expected[name]:
                raise ValueError("Export provenance and checksums disagree.")
    for name, digest in expected.items():
        if sha256(DATA / name) != digest:
            raise ValueError(f"Reference checksum mismatch: {name}")
    accuracy, contrasts, seeds = [read(name) for name in
        ["i15_accuracy.csv", "i15_contrasts.csv", "i15_primary_seed_points.csv"]]
    if [len(accuracy), len(contrasts), len(seeds)] != [84, 72, 24]:
        raise ValueError("Unexpected publication row counts")
    index = {(r["fold"], r["family"], r["partition"], r["arm"], r["seed"]): r for r in accuracy}
    if len(index) != 84:
        raise ValueError("Duplicate accuracy records")
    def close(a, b, label):
        if not math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"Inconsistent {label}: {a} versus {b}")
    for r in contrasts + seeds:
        close(r["delta_percentage_points"], float(r["other_WAPE"]) - float(r["FULL_WAPE"]), "condition-minus-FULL contrast")
        if r["comparison"] != r["arm"] + "-FULL":
            raise ValueError("Unexpected contrast direction")
        if sum(int(r[k]) for k in ["days_FULL_better", "days_FULL_worse", "days_tied"]) != int(r["days"]):
            raise ValueError("Daily contrast counts do not sum to support")
        for prefix in ["iid", "block3"]:
            low, high = float(r[prefix + "_CI_low"]), float(r[prefix + "_CI_high"])
            if not (math.isfinite(low) and math.isfinite(high) and low <= high):
                raise ValueError("Invalid stored confidence bounds")
    for r in contrasts:
        prefix = (r["fold"], r["family"], r["partition"])
        full = index[prefix + ("FULL", r["seed"])]
        other = index[prefix + (r["arm"], r["seed"])]
        close(r["FULL_WAPE"], full["MeanDailyWAPE"], "FULL score")
        close(r["other_WAPE"], other["MeanDailyWAPE"], "condition score")
        for delta, score in [("delta_PeakMagnitudeMAE_kW", "PeakMagnitudeMAE"), ("delta_PeakTimingMAE_hours", "PeakTimingMAE")]:
            close(r[delta], float(other[score]) - float(full[score]), delta)
    windows = read("study_windows.csv")
    if len(windows) != 6:
        raise ValueError("Expected six chronological partitions")
    for r in windows:
        duration = datetime.fromisoformat(r["end_exclusive"]) - datetime.fromisoformat(r["start"])
        if duration.days != int(r["days"]) or duration.total_seconds() / 3600 != int(r["grid_hours"]):
            raise ValueError("Window dates and support disagree")
        if r["partition"] != "train":
            score = index[(r["fold"], "Ridge-ARX", r["partition"].title(), "FULL", "deterministic")]
            for field, metric in [("observed_targets", "ObservedHours"), ("complete_days", "CompleteDays"), ("days", "Days")]:
                close(r[field], score[metric], "window " + field)
    return {"reference_files": len(expected), "accuracy_rows": 84, "contrast_rows": 72,
            "individual_seed_rows": 24, "chronological_partitions": 6,
            "stored_interval_bounds_checked": 384}


def verify_rendered() -> dict:
    """Independently inspect tables and plot records against fixed references."""
    from decimal import Decimal, ROUND_HALF_EVEN
    expected_points = json.loads((VALIDATION / "plotted_points.json").read_text(encoding="utf-8"))
    lookup = {(r["fold"], r["family"], r["arm"], r["seed"]): r for r in
              read("i15_contrasts.csv") + read("i15_primary_seed_points.csv") if r["partition"] == "Evaluation"}
    coordinates = 0
    if len(expected_points) != 48:
        raise ValueError("Expected 48 plotted points")
    seen = set()
    for p in expected_points:
        identity = (p["figure"], p["fold"], p["family"], p["arm"], p["kind"])
        if identity in seen:
            raise ValueError("Repeated plotted point")
        seen.add(identity)
        row = lookup[(p["fold"], p["family"], p["arm"], p["kind"])]
        if p["x"] != float(row["delta_percentage_points"]):
            raise ValueError("Plot contrast mismatch")
        coordinates += 1
        if p["figure"] == "i15_mean_peak_tradeoff":
            if p["y"] != float(row["delta_PeakMagnitudeMAE_kW"]):
                raise ValueError("Peak coordinate mismatch")
            coordinates += 1
        if "CI_low" in p:
            if [p["CI_low"], p["CI_high"]] != [float(row["block3_CI_low"]), float(row["block3_CI_high"])]:
                raise ValueError("Plotted interval mismatch")
            coordinates += 2
    cells_checked = 0
    scores = {(r["fold"], r["family"], r["arm"]): r for r in read("i15_accuracy.csv") if r["partition"] == "Evaluation"}
    labels = {label: arm for arm, label in ARM_LABELS.items()}
    for line in (TABLES / "i15_accuracy.tex").read_text(encoding="utf-8").splitlines():
        cells = [c.replace(r"\\", "").strip() for c in line.split("&")]
        if cells[0] not in labels:
            continue
        actual = [cell for cell in cells[1:] if cell]
        expected = [str(Decimal(scores[(fold, family, labels[cells[0]])]["MeanDailyWAPE"]).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)) for fold in FOLDS for family in FAMILIES]
        if actual != expected:
            raise ValueError("Accuracy table cell mismatch")
        cells_checked += len(expected)
    differences = {(r["fold"], r["family"], r["partition"], r["arm"]): r for r in read("i15_contrasts.csv")}
    family = None
    for line in (TABLES / "i15_validation.tex").read_text(encoding="utf-8").splitlines():
        cells = [c.replace(r"\\", "").strip() for c in line.split("&")]
        if len(cells) != 6 or cells[1] not in ["W", "T"]:
            continue
        family = cells[0] or family
        arm = {"W": "NO_WEATHER", "T": "NO_TEMP"}[cells[1]]
        expected = []
        for fold in FOLDS:
            for partition in ["Validation", "Evaluation"]:
                value = Decimal(differences[(fold, family, partition, arm)]["delta_percentage_points"]).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)
                expected.append(f"${value:+.2f}$")
        if cells[2:] != expected:
            raise ValueError("Contrast table cell or sign mismatch")
        cells_checked += len(expected)
    for reference in read("i15_simple_references.csv"):
        values = [reference[k] for k in ("warm_mean_daily_WAPE_percent", "cold_mean_daily_WAPE_percent")]
        if NEW_EXECUTION:
            values = [str(Decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN))
                      for value in values]
        expected = reference["reference"] + r" & \multicolumn{3}{c}{" + values[0] + r"} && \multicolumn{3}{c}{" + values[1] + r"}\\"
        if expected not in (TABLES / "i15_accuracy.tex").read_text(encoding="utf-8"):
            raise ValueError("Persistence reference table mismatch")
    fixed_conditions = REFERENCE / "expected-tables" / "conditions.tex"
    if (TABLES / "conditions.tex").read_bytes() != fixed_conditions.read_bytes():
        raise ValueError("Fixed conditions table mismatch")
    for name in ([] if NEW_EXECUTION else ["i15_accuracy.tex", "i15_validation.tex", "conditions.tex"]):
        if (TABLES / name).read_text(encoding="utf-8") != (DATA / "expected-tables" / name).read_text(encoding="utf-8"):
            raise ValueError("Historical table content mismatch: " + name)
    if cells_checked != 66 or coordinates != 84:
        raise ValueError("Incomplete numeric output verification")
    return {"plotted_points": 48, "coordinates_and_ci_bounds": 84,
            "rounded_table_cells": 66,
            ("execution_persistence_cells" if NEW_EXECUTION else "inherited_persistence_cells"): 6,
            "fixed_conditions_table": "unchanged",
            "historical_tables": "not compared: new execution" if NEW_EXECUTION else "match"}


def main(argv=None) -> None:
    global OUT, TABLES, VALIDATION, DATA, NEW_EXECUTION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Write figures/, tables/, validation/ below this directory instead of results/reproduced/.")
    parser.add_argument("--data-dir", type=Path, help="Use aggregates exported from a completed new full run; requires an explicit fresh --output-dir.")
    parser.add_argument("--check-only", action="store_true", help="Verify references and existing rendered outputs without regenerating them.")
    args = parser.parse_args(argv)
    DATA = args.data_dir.resolve() if args.data_dir else REFERENCE
    NEW_EXECUTION = args.data_dir is not None
    if NEW_EXECUTION and args.output_dir is None:
        parser.error("--data-dir requires an explicit --output-dir.")
    base = args.output_dir.resolve() if args.output_dir else OUTPUT
    for protected in (DATA, REFERENCE):
        if base == protected or base in protected.parents or protected in base.parents:
            raise ValueError("Rendering output must not overlap aggregate inputs or historical references.")
    if NEW_EXECUTION:
        if not args.check_only and base.exists() and any(base.iterdir()):
            raise FileExistsError("New-execution rendering requires a new or empty output directory.")
    OUT, TABLES, VALIDATION = base / "figures", base / "tables", base / "validation"
    reference_report = verify_reference()
    if not args.check_only:
        for folder in [OUT, TABLES, VALIDATION]:
            folder.mkdir(parents=True, exist_ok=True)
        DRAWN.clear()
        accuracy, contrasts, seeds = [read(n) for n in ["i15_accuracy.csv", "i15_contrasts.csv", "i15_primary_seed_points.csv"]]
        style()
        study_windows()
        primary_value(contrasts, seeds)
        mean_peak(contrasts)
        tables(accuracy, contrasts)
        shutil.copyfile(REFERENCE / "expected-tables/conditions.tex", TABLES / "conditions.tex")
        validate_points(contrasts, seeds)
        (VALIDATION / "plotted_points.json").write_text(json.dumps(DRAWN, indent=2) + "\n", encoding="utf-8")
    numeric_report = verify_rendered()
    outputs = sorted([*OUT.glob("*.pdf"), *OUT.glob("*.png"), *TABLES.glob("*.tex")])
    report = {"status": "PASS", "references": reference_report, "numeric_fidelity": numeric_report,
              "outputs": {p.parent.name + "/" + p.name: sha256(p) for p in outputs},
              "origin": "new_full_experiment" if NEW_EXECUTION else "historical_reference",
              "scope": ("Rendering aggregates from a newly completed full experiment; no reference scores are substituted."
                        if NEW_EXECUTION else "Regeneration from fixed publication summaries; no training, private data access, or independent bootstrap recomputation."),
              "rounding": "Tables round only at final presentation; plotted values and CI bounds retain source precision.",
              "image_equivalence": "Scientific coordinates are checked; image bytes may differ with fonts and dependency versions."}
    (VALIDATION / "artifact-validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
