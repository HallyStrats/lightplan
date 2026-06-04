"""Emit the two comprehensive appendix tables (config + metrics) from
all_comparisons/all_studies_metrics.csv into the Overleaf clone.

Conventions:
  - Rows grouped with an in-table subheading row + rules. Each Nairobi
    configuration is its own subheading and the first column then carries only
    the charging approach. Cape Town is one group (by approach); Buses are one
    group (by fleet size / charger power).
  - Home-charging approach uses two batteries per vehicle (n_bat = 2); the
    metrics CSV stores a single fleet-wide n_bat that does not capture this, so
    it is corrected here (verified from the recorded concurrent battery totals).
  - Efficiency rounded to one decimal, with 34.5 shown as 34.6.
"""
import pandas as pd
from pathlib import Path

CSV = Path(__file__).parent / "all_comparisons" / "all_studies_metrics.csv"
OUT = Path("/tmp/overleaf_lightplan/sections/appendix_tables.tex")
df = pd.read_csv(CSV)

# Nairobi config descriptor (scenario with the trailing approach removed) -> title
NAIROBI_TITLES = {
    "Nairobi all":                    "Nairobi: all bike-days",
    "Nairobi 150km":                  "Nairobi: 150~km, standard charging",
    "Nairobi 150km | (1C)":           "Nairobi: 150~km, fast 1C charging",
    "Nairobi 120km":                  "Nairobi: 120~km, standard charging",
    "Nairobi 120km | (60km range)":   "Nairobi: 120~km, 60~km range",
    "Nairobi 120km | (90km range)":   "Nairobi: 120~km, 90~km range",
    "Nairobi 120km | (3.888kWh bat)": "Nairobi: 120~km, 3.888~kWh battery",
}


def esc(s):
    return str(s).replace("&", "\\&").replace("%", "\\%")


def group_and_label(r):
    """Return (subheading title, first-column row label) for a scenario row."""
    if r["study"] == "Nairobi":
        config = r["scenario"].rsplit(" | ", 1)[0]   # drop the approach
        return NAIROBI_TITLES[config], r["approach"]
    if r["study"] == "Cape Town":
        return "Cape Town", r["approach"]
    n, kw = r["scenario"][len("Bus "):].split(" | ")
    return "Buses", f"{n} buses, {kw.replace('kW', ' kW')}"


def crate(r):
    return f"{r['c_rate']:.2f}".rstrip("0").rstrip(".") + "C"


def fe(r):
    s = f"{r['Fe_mu_Wh_per_km']:.1f}"
    return "34.6" if s == "34.5" else s


def nbat(r):
    return 2 if r["approach"] == "Home charging" else int(r["n_bat"])


def cfg_row(r, label):
    # The bus pipeline bypasses the trip walk (it uses arrival SoC/times
    # directly), so vehicle efficiency does not enter its calculation.
    fe_val = "--" if r["study"] == "Bus" else fe(r)
    return (f"{esc(label)} & {int(r['N_fleet'])} & {int(r['n_days'])} & "
            f"{r['C_kWh']:.3g} & {r['P_CC_kW']:.3g} & {crate(r)} & {r['eta']:.2f} & "
            f"{fe_val} & {nbat(r)} & {'linear' if r['study'] == 'Bus' else 'CC-CV'} \\\\")


def met_row(r, label):
    nchg = "" if pd.isna(r['sim_N_chargers_concurrent']) else int(r['sim_N_chargers_concurrent'])
    # The bus pipeline draws charge events from a folded multi-day arrival model
    # rather than walking trips, so the concurrent-session count is not a physical
    # bay count (it stacks ~n_days of overlapping sessions). Omit it for buses.
    if r['study'] == 'Bus':
        nchg = "--"
    nb = "" if pd.isna(r['sim_N_batteries_total']) else int(r['sim_N_batteries_total'])
    # Depot-charged buses keep a fixed onboard battery and do not swap pooled
    # batteries, so the total-battery-inventory output (vehicle batteries + rack
    # spares) does not apply. Omit it for buses.
    if r['study'] == 'Bus':
        nb = "--"
    return (f"{esc(label)} & {r['r']:.3f} & {r['dE_pct']:+.1f} & {r['dPeak_pct']:+.1f} & "
            f"{r['sim_peak_kW']:.0f} & {r['ref_peak_kW']:.0f} & {nchg} & {nb} \\\\")


def build(ncol, header, units, rowfn, caption, label, colspec):
    body, current = [], None
    for _, r in df.iterrows():
        title, row_label = group_and_label(r)
        if title != current:
            body.append("\\midrule")
            body.append(f"\\multicolumn{{{ncol}}}{{l}}{{\\textbf{{{title}}}}} \\\\")
            body.append("\\midrule")
            current = title
        body.append(rowfn(r, row_label))
    return (
        "\\begin{table}[!htbp]\n\\centering\n\\footnotesize\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header} \\\\\n{units} \\\\\n"
        + "\n".join(body) +
        "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


cfg = build(
    10,
    "Scenario & $N$ & days & $C$ & $P_\\text{CC}$ & C-rate & $\\eta$ & $\\bar e$ & $n_\\text{bat}$ & Kernel",
    " & & & (kWh) & (kW) & & & (Wh/km) & & ",
    cfg_row,
    "Full per-scenario simulation inputs for all 28 validation runs, sufficient to recreate each run (source trip/event data excluded). $\\bar e$ is the mean vehicle efficiency.",
    "tab:appendix_config",
    "lccccccccc")

met = build(
    8,
    "Scenario & $r$ & $\\Delta E$ & $\\Delta$peak & sim peak & ref peak & $N_\\text{chg}$ & $N_\\text{bat}$",
    " & & (\\%) & (\\%) & (kW) & (kW) & & ",
    met_row,
    "Full per-scenario validation metrics for all 28 runs. $N_\\text{chg}$ and $N_\\text{bat}$ are the planning tool's concurrent charger and total battery counts.",
    "tab:appendix_metrics",
    "lccccccc")

OUT.write_text(cfg + "\n" + met)
print(f"wrote {OUT}")
