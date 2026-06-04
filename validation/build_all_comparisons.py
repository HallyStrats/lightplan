"""Master comparison-plot script. Produces every cross-study chart into
all_comparisons/. Conventions per request:
  - No headline title text (captions will live in the document).
  - Quanta (kWh, batteries, chargers) → bar charts.
  - Pearson r and kW (peak power) → dot charts.
  - Bus reference uses median + 30-min rolling smooth (source pipeline's
    own reduction); other refs use the mean-day curve.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
NAI = ROOT / "Nairobi_Simulation"
CPT = ROOT / "Cape_Town_Simulation"
BUS = ROOT / "Bus_Simulation"
OUT = ROOT / "all_comparisons"
OUT.mkdir(exist_ok=True)
OVERLEAF = Path("/tmp/overleaf_lightplan/figures")

E_BAR_BODA = 34.55   # Wh/km (mean for the base configs; the 60/90 km-range variants differ, see note)
ETA        = 1.0     # boda/Cape Town validation runs used eta=1.0, so grid-side phys = battery-side


# ----- data loaders -----
def boda_ref_mean(csv, col):
    ref = pd.read_csv(csv, index_col=0, parse_dates=True)
    m = ref.index.hour * 60 + ref.index.minute
    return ref.groupby(m)[col].mean()


def bus_ref_smoothed(csv):
    ref = pd.read_csv(csv, parse_dates=["time_str"])
    m = ref["time_str"].dt.hour * 60 + ref["time_str"].dt.minute
    med = ref.groupby(m)["total_grid_load"].median()
    full = pd.Series(index=range(1440), dtype=float)
    full.update(med)
    full = full.interpolate().fillna(0)
    return full.rolling(window=30, center=True, min_periods=1).mean()


def bus_ref_mean(csv):
    ref = pd.read_csv(csv, parse_dates=["time_str"])
    m = ref["time_str"].dt.hour * 60 + ref["time_str"].dt.minute
    return ref.groupby(m)["total_grid_load"].mean()


def sim_load(csv, scenario=None):
    df = pd.read_csv(csv)
    if scenario is not None:
        df = df[df["scenario"] == scenario].sort_values("t_hour")
        x = df["t_hour"].values * 60
        y = df["L_total_mean_kW"].values
    else:
        x = df["t_hours"].values * 60
        y = df["L_mean_kW"].values
    return x, y


# ----- collect scenarios -----
SCENARIOS = []

# Nairobi (3 filters × {App 1, App 2})
NAI_ROWS = [
    ("Nairobi all",         NAI / "comparison" / "trips_all_matched"   / "RESULTS_trips_load.csv",
                             NAI / "reference_outputs" / "all"   / "all_charging_load_minute.csv",   26, 8648, 119, False),
    ("Nairobi 150km",       NAI / "comparison" / "trips_150km_matched" / "RESULTS_trips_150km_load.csv",
                             NAI / "reference_outputs" / "150km" / "150km_standard_charging_load_minute.csv", 26, 7195, 119, True),
    ("Nairobi 150km\n(1C)", NAI / "comparison" / "trips_150km_fast"    / "RESULTS_trips_150km_load.csv",
                             NAI / "reference_outputs" / "150km" / "150km_fast_charging_load_minute.csv", 26, 7195, 119, True),
    ("Nairobi 120km",          NAI / "comparison" / "trips_120km_matched"   / "RESULTS_trips_120km_load.csv",
                                NAI / "reference_outputs" / "120km" / "120km_standard_charging_load_minute.csv", 26, 5526, 119, True),
    ("Nairobi 120km\n(60km range)", NAI / "comparison" / "trips_120km_60km_range" / "RESULTS_trips_120km_load.csv",
                                     NAI / "reference_outputs" / "120km" / "120km_60km_range_charging_load_minute.csv", 26, 5526, 119, True),
    ("Nairobi 120km\n(90km range)", NAI / "comparison" / "trips_120km_90km_range" / "RESULTS_trips_120km_load.csv",
                                     NAI / "reference_outputs" / "120km" / "120km_90km_range_charging_load_minute.csv", 26, 5526, 119, True),
    ("Nairobi 120km\n(3.888kWh bat)", NAI / "comparison" / "trips_120km_bigbat" / "RESULTS_trips_120km_load.csv",
                                       NAI / "reference_outputs" / "120km" / "120km_90km_range_same_efficiency_charging_load_minute.csv", 26, 5526, 119, True),
]
NAI_APPS_BASE = [("Battery swapping", "grid_load_app_1", "Swapping"),
                 ("Hybrid (swap + home)", "grid_load_app_2", "Both")]
NAI_APP3 = ("Home charging", "grid_load_app_3", "Home charging")

for row_lbl, sim_csv, ref_csv, n_days, fleet_km, n_fleet, include_app3 in NAI_ROWS:
    phys = fleet_km * E_BAR_BODA / ETA / 1000
    apps = NAI_APPS_BASE + ([NAI_APP3] if include_app3 else [])
    summary_csv = sim_csv.with_name(sim_csv.name.replace("_load.csv", "_summary.csv"))
    for scen, ref_col, app in apps:
        x, y = sim_load(sim_csv, scen)
        ref_curve = boda_ref_mean(ref_csv, ref_col)
        n = min(len(ref_curve), len(y))
        r = np.corrcoef(ref_curve.values[:n], y[:n])[0, 1]
        E_sim = y.mean() * 24
        E_ref = pd.read_csv(ref_csv, index_col=0, parse_dates=True)[ref_col].sum() / 60 / n_days
        SCENARIOS.append({
            "label": f"{row_lbl}\n{app}", "x": x, "y": y,
            "ref_x": ref_curve.index.values, "ref_y": ref_curve.values,
            "r": r, "sim_E": E_sim, "ref_E": E_ref, "phys_E": phys,
            "sim_pk": y.max(), "ref_pk": ref_curve.max(),
            "n_fleet": n_fleet, "n_days": n_days,
            "summary_csv": summary_csv,
            "scenario": scen, "approach": app,
        })

# Cape Town (App 1, App 2)
CPT_REF = CPT / "reference_outputs" / "actual_data_run" / "charging_load_per_minute.csv"
CPT_SIM = CPT / "comparison" / "trips_all_operating" / "RESULTS_trips_load.csv"
for scen, ref_col, app in [("Battery swapping", "grid_load_app_1", "Swapping"),
                            ("Hybrid (swap + home)", "grid_load_app_2", "Both")]:
    x, y = sim_load(CPT_SIM, scen)
    ref_curve = boda_ref_mean(CPT_REF, ref_col)
    n = min(len(ref_curve), len(y))
    r = np.corrcoef(ref_curve.values[:n], y[:n])[0, 1]
    E_sim = y.mean() * 24
    E_ref = pd.read_csv(CPT_REF, index_col=0, parse_dates=True)[ref_col].sum() / 60 / 14
    SCENARIOS.append({
        "label": f"Cape Town\n{app}", "x": x, "y": y,
        "ref_x": ref_curve.index.values, "ref_y": ref_curve.values,
        "r": r, "sim_E": E_sim, "ref_E": E_ref,
        "phys_E": 9487 * E_BAR_BODA / ETA / 1000,
        "sim_pk": y.max(), "ref_pk": ref_curve.max(),
        "n_fleet": 125, "n_days": 14,
        "summary_csv": CPT / "comparison" / "trips_all_operating" / "RESULTS_trips_summary.csv",
        "scenario": scen, "approach": app,
    })

# Bus (30, 40, 50) × {30kW, 60kW}
for kw_lbl, kw_tag in [("30kW", "30kW"), ("60kW", "60kW")]:
    for n_bus in [30, 40, 50]:
        sim_name = "load.csv" if kw_lbl == "30kW" else "60kW_load.csv"
        sim_csv = BUS / "comparison" / f"RESULTS_SYNTHETIC_charge_events_{n_bus}bus_{sim_name}"
        ref_csv = BUS / "reference_outputs" / "synthetic" / f"{n_bus}_buses_{kw_tag}_unmanaged_20250101_to_20250131_simulation.csv"
        x, y = sim_load(sim_csv)
        ref_curve = bus_ref_smoothed(ref_csv)
        ref_mean = bus_ref_mean(ref_csv)
        n = min(len(ref_curve), len(y))
        r = np.corrcoef(ref_curve.values[:n], y[:n])[0, 1]
        evt = pd.read_csv(BUS / "inputs" / f"SYNTHETIC_charge_events_{n_bus}bus.csv")
        phys = (1 - evt["s"] / 100).sum() * 230 / 31 / 0.9   # bus validation runs used eta=0.9
        SCENARIOS.append({
            "label": f"Bus {n_bus}\n{kw_lbl}", "x": x, "y": y,
            "ref_x": ref_curve.index.values, "ref_y": ref_curve.values,
            "r": r, "sim_E": y.mean() * 24, "ref_E": ref_mean.mean() * 24, "phys_E": phys,
            "sim_pk": y.max(), "ref_pk": ref_curve.max(),
            "n_fleet": n_bus, "n_days": 31,
            "summary_csv": BUS / "comparison" / f"RESULTS_SYNTHETIC_charge_events_{n_bus}bus_{'60kW_summary.csv' if kw_lbl == '60kW' else 'summary.csv'}",
            "scenario": f"Bus {kw_lbl} unmanaged", "approach": kw_lbl,
        })


# ----- 1. mean-day curves — one PNG per study -----
def study_of_label(lbl):
    if lbl.startswith("Nairobi"): return "Nairobi"
    if lbl.startswith("Cape Town"): return "Cape Town"
    if lbl.startswith("Bus"): return "Bus"
    return "?"

per_study = {"Nairobi": [], "Cape Town": [], "Bus": []}
for sc in SCENARIOS:
    per_study[study_of_label(sc["label"])].append(sc)

for study, scens in per_study.items():
    n = len(scens)
    if n == 0: continue
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3 * nrows), sharex=True,
                              squeeze=False)
    for i, sc in enumerate(scens):
        ax = axes.flat[i]
        N = sc["n_fleet"]
        ax.plot(sc["ref_x"], sc["ref_y"] / N, "k--", lw=1.2, label="Source ref")
        ax.plot(sc["x"],     sc["y"]     / N, "C0-", lw=1.2, label="Planning tool")
        ax.set_title(sc["label"] + f"  (N={N})", fontsize=9)
        ax.set_xlim(0, 1440); ax.set_xticks([0, 360, 720, 1080, 1440]); ax.set_xticklabels(["00", "06", "12", "18", "24"])
        ax.grid(alpha=0.3)
        ax2 = ax.twinx()
        lo, hi = ax.get_ylim()
        ax2.set_ylim(lo * N, hi * N)
        ax2.tick_params(axis="y", labelsize=7)
        is_rightmost = (i % ncols == ncols - 1) or (i == n - 1)
        if is_rightmost:
            ax2.set_ylabel("Fleet grid load (kW)", fontsize=8)
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")
    for j in range(n, nrows * ncols):
        axes.flat[j].axis("off")
    for ax in axes[:, 0]: ax.set_ylabel("Per-vehicle grid load (kW/veh)")
    for ax in axes[-1, :]: ax.set_xlabel("Hour of day")
    fig.tight_layout()
    fname = f"curves_{study.lower().replace(' ', '_')}.png"
    fig.savefig(OUT / fname, dpi=130)
    if OVERLEAF.exists():
        if study == "Nairobi":
            fig.savefig(OVERLEAF / "figA1_nairobi_appendix.png", dpi=130, bbox_inches="tight")
        elif study == "Bus":
            fig.savefig(OVERLEAF / "figA2_bus_appendix.png", dpi=130, bbox_inches="tight")
    print(f"wrote {fname}")

df = pd.DataFrame(SCENARIOS)
# group rows by study
def study_of(lbl):
    if lbl.startswith("Nairobi"): return "Nairobi"
    if lbl.startswith("Cape Town"): return "Cape Town"
    if lbl.startswith("Bus"): return "Bus"
    return "?"
df["study"] = df["label"].apply(study_of)
studies = ["Nairobi", "Cape Town", "Bus"]
groups  = {s: df[df["study"] == s].reset_index(drop=True) for s in studies}
widths  = [len(groups[s]) for s in studies]


def plot_three_panel(filename, plot_fn, ylabel, sharey=False, **kwargs):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5),
                             gridspec_kw={"width_ratios": widths},
                             sharey=sharey)
    for ax, s in zip(axes, studies):
        g = groups[s]
        xs = np.arange(len(g))
        plot_fn(ax, g, xs, **kwargs)
        ax.set_xticks(xs)
        ax.set_xticklabels(g["label"], fontsize=8)
        ax.set_title(s, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=130)
    print(f"wrote {filename}")


# ----- 2. daily energy (bars; sim / ref / ground truth) -----
def _energy(ax, g, xs, **_):
    w = 0.38
    ax.bar(xs - w / 2, g["sim_E"], w, label="Planning tool",   color="C0")
    ax.bar(xs + w / 2, g["ref_E"], w, label="Source ref", color="0.4")
    top = max(g["sim_E"].max(), g["ref_E"].max())
    for i, (s, r) in enumerate(zip(g["sim_E"], g["ref_E"])):
        ax.text(i - w / 2, s + top * 0.01, f"{s:.0f}", ha="center", fontsize=7)
        ax.text(i + w / 2, r + top * 0.01, f"{r:.0f}", ha="center", fontsize=7)
plot_three_panel("all_studies_energy.png", _energy, "Daily energy (kWh/day)")

# ----- 3. peak power (dots) -----
def _peak(ax, g, xs, **_):
    ax.scatter(xs, g["sim_pk"], s=70, color="C0", label="Planning tool", zorder=3)
    ax.scatter(xs, g["ref_pk"], s=70, color="0.4", marker="D", label="Source ref", zorder=3)
    for i in xs:
        ax.plot([i, i], [g["sim_pk"].iloc[i], g["ref_pk"].iloc[i]], "0.7", lw=0.8, zorder=1)
plot_three_panel("all_studies_peak.png", _peak, "Peak grid load (kW)")

# ----- 4. Pearson r (dots) -----
r_lo = min(0.95, df["r"].min() - 0.04)
def _r(ax, g, xs, **_):
    ax.scatter(xs, g["r"], s=80, color="C3", zorder=3)
    for i, v in enumerate(g["r"]):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=7)
    ax.axhline(1.0, color="0.5", lw=0.8, ls="--")
    ax.set_ylim(r_lo, 1.02)
plot_three_panel("all_studies_pearson.png", _r, "Pearson r (mean-day curve)", sharey=True)

# ----- 5. Cape Town resources (bars) -----
rt = pd.read_csv(ROOT / "comparison_resources_table.csv")
rt = rt[rt["Study"] == "Cape Town"].reset_index(drop=True)
rt["App"] = rt["App"].replace({"App 1": "Swapping", "App 2": "Both", "App 3": "Home charging"})
xs = np.arange(len(rt)); w = 0.35
fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
ax = axes[0]
ax.bar(xs - w/2, rt["Sim chargers"],         w, label="Planning tool",  color="C0")
ax.bar(xs + w/2, rt["Ref chargers (peak)"],  w, label="Source ref", color="0.4")
ax.set_ylabel("Chargers (peak concurrent)")
ax.set_xticks(xs); ax.set_xticklabels(rt["App"]); ax.grid(axis="y", alpha=0.3); ax.legend()
for i, (s, r) in enumerate(zip(rt["Sim chargers"], rt["Ref chargers (peak)"])):
    ax.text(i - w/2, s + 1, str(s), ha="center", fontsize=9)
    ax.text(i + w/2, r + 1, str(r), ha="center", fontsize=9)
ax = axes[1]
ax.bar(xs - w/2, rt["Sim batteries (total)"],          w, label="Planning tool",  color="C1")
ax.bar(xs + w/2, rt["Ref batteries (peak on-charge)"], w, label="Source ref", color="0.4")
ax.set_ylabel("Batteries"); ax.set_xticks(xs); ax.set_xticklabels(rt["App"]); ax.grid(axis="y", alpha=0.3); ax.legend()
for i, (s, r) in enumerate(zip(rt["Sim batteries (total)"], rt["Ref batteries (peak on-charge)"])):
    ax.text(i - w/2, s + 2, str(s), ha="center", fontsize=9)
    ax.text(i + w/2, r + 2, str(r), ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "capetown_resources.png", dpi=130)
print("wrote capetown_resources.png")

# ----- 6. Paper figures: Cape Town curves-only + Bus fast-charge comparison -----
by_label = {sc["label"]: sc for sc in SCENARIOS}


def _save_paper(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    if OVERLEAF.exists():
        fig.savefig(OVERLEAF / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}")


# Cape Town: curves only (bar plot moved to a table in the manuscript)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, (lbl, ttl) in zip(axes, [("Cape Town\nSwapping", "(a) Swap-only"),
                                 ("Cape Town\nBoth", "(b) Swap + home")]):
    sc = by_label[lbl]; N = sc["n_fleet"]
    ax.plot(sc["ref_x"], sc["ref_y"] / N, "k--", lw=1.2, label="Source ref")
    ax.plot(sc["x"], sc["y"] / N, "C0-", lw=1.2, label="Planning tool")
    ax.set_xlim(0, 1440); ax.set_xticks([0, 360, 720, 1080, 1440])
    ax.set_xticklabels(["00", "06", "12", "18", "24"])
    ax.grid(alpha=0.3); ax.set_xlabel("Hour of day")
    ax.set_title(ttl, loc="left", fontsize=10)
    ax2 = ax.twinx(); lo, hi = ax.get_ylim(); ax2.set_ylim(lo * N, hi * N)
    ax2.tick_params(axis="y", labelsize=7); ax2.set_ylabel("Fleet grid load (kW)", fontsize=8)
    if ax is axes[0]:
        ax.set_ylabel("Per-vehicle grid load (kW/veh)")
        ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); _save_paper(fig, "fig6_capetown.png")

# Bus fast-charge: 30 buses, 30 kW vs 60 kW, shared fleet-kW y axis
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, (lbl, ttl) in zip(axes, [("Bus 30\n30kW", "(a) 30 kW charger"),
                                 ("Bus 30\n60kW", "(b) 60 kW charger")]):
    sc = by_label[lbl]
    ax.plot(sc["ref_x"], sc["ref_y"], "k--", lw=1.2, label="Source ref")
    ax.plot(sc["x"], sc["y"], "C0-", lw=1.2, label="Planning tool")
    ax.set_xlim(0, 1440); ax.set_xticks([0, 360, 720, 1080, 1440])
    ax.set_xticklabels(["00", "06", "12", "18", "24"])
    ax.grid(alpha=0.3); ax.set_xlabel("Hour of day")
    ax.set_title(ttl, loc="left", fontsize=10)
axes[0].set_ylabel("Fleet grid load (kW)")
axes[0].legend(fontsize=8, loc="upper left")
fig.tight_layout(); _save_paper(fig, "fig_bus_fastcharge.png")

# Nairobi 150 km: three approaches (main-results figure)
fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
for ax, (lbl, ttl) in zip(axes, [("Nairobi 150km\nSwapping", "(a) Swap-only"),
                                 ("Nairobi 150km\nBoth", "(b) Swap + home"),
                                 ("Nairobi 150km\nHome charging", "(c) Home only")]):
    sc = by_label[lbl]; N = sc["n_fleet"]
    ax.plot(sc["ref_x"], sc["ref_y"] / N, "k--", lw=1.2, label="Source ref")
    ax.plot(sc["x"], sc["y"] / N, "C0-", lw=1.2, label="Planning tool")
    ax.set_xlim(0, 1440); ax.set_xticks([0, 360, 720, 1080, 1440])
    ax.set_xticklabels(["00", "06", "12", "18", "24"])
    ax.grid(alpha=0.3); ax.set_xlabel("Hour of day")
    ax.set_title(ttl, loc="left", fontsize=10)
    ax2 = ax.twinx(); lo, hi = ax.get_ylim(); ax2.set_ylim(lo * N, hi * N)
    ax2.tick_params(axis="y", labelsize=7)
    if ax is axes[-1]:
        ax2.set_ylabel("Fleet grid load (kW)", fontsize=8)
    if ax is axes[0]:
        ax.set_ylabel("Per-vehicle grid load (kW/veh)")
        ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); _save_paper(fig, "fig5_nairobi.png")

# Save summary CSV with config columns pulled from each scenario's summary file
def cfg(sc, key):
    try:
        d = dict(pd.read_csv(sc["summary_csv"]).values)
        return d.get(key, "")
    except Exception:
        return ""

def res(sc, key):
    try:
        d = dict(pd.read_csv(sc["summary_csv"]).values)
        for prefix in ("results.quick.", f"results.{sc['scenario']}.", "results.Battery swapping."):
            k = prefix + key
            if k in d: return d[k]
        return ""
    except Exception:
        return ""

rows = []
for sc in SCENARIOS:
    study = "Nairobi" if sc["label"].startswith("Nairobi") else ("Cape Town" if sc["label"].startswith("Cape Town") else "Bus")
    rows.append({
        "study": study,
        "scenario": sc["label"].replace("\n", " | "),
        "approach": sc["approach"],
        "N_fleet": sc["n_fleet"],
        "n_days": sc["n_days"],
        "kernel_source":   cfg(sc, "inputs.kernel_source"),
        "C_kWh":           cfg(sc, "battery.capacity_kWh"),
        "P_CC_kW":         cfg(sc, "charger.P_CC_kW"),
        "c_rate":          cfg(sc, "charger.c_rate"),
        "cc_fraction":     cfg(sc, "charger.cc_fraction"),
        "cv_to_cc_ratio":  cfg(sc, "charger.cv_to_cc_ratio"),
        "eta":             cfg(sc, "charger.efficiency"),
        "s_full":          cfg(sc, "battery.s_full"),
        "Fe_mu_Wh_per_km": cfg(sc, "Fe.mu_Wh_per_km"),
        "Fs_mu_fraction":  cfg(sc, "Fs.mu_fraction"),
        "n_bat":           cfg(sc, "battery.batteries_per_vehicle"),
        "n_events":        cfg(sc, "n_events"),
        "runtime_s_total": cfg(sc, "runtime_seconds"),
        "runtime_s_scenario": res(sc, "runtime_seconds"),
        "r":         round(sc["r"], 4),
        "sim_E_kWh": round(sc["sim_E"], 1),
        "ref_E_kWh": round(sc["ref_E"], 1),
        "phys_E_kWh": round(sc["phys_E"], 1),
        "dE_pct":    round((sc["sim_E"]/sc["ref_E"] - 1) * 100, 2),
        "sim_peak_kW": round(sc["sim_pk"], 1),
        "ref_peak_kW": round(sc["ref_pk"], 1),
        "dPeak_pct": round((sc["sim_pk"]/sc["ref_pk"] - 1) * 100, 2),
        "sim_N_chargers_concurrent": res(sc, "N_chargers_concurrent"),
        "sim_N_batteries_total":     res(sc, "N_batteries_total"),
        "sim_swap_events":           res(sc, "swap_events"),
        "sim_home_events":           res(sc, "home_events"),
    })
df_out = pd.DataFrame(rows)
df_out.to_csv(OUT / "all_studies_metrics.csv", index=False)
print(f"wrote all_studies_metrics.csv ({len(df_out)} rows, {len(df_out.columns)} cols)")
print(f"\nAll outputs in: {OUT}")
