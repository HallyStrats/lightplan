"""Streamlit UI for the lightplan pipeline. Run: `streamlit run app.py`.

Single app with two view modes:
  • Results dashboard    — run once, see overlaid scenarios and metrics.
  • Step-by-step view    — walk through each pipeline stage with equations
                           and visualisations for a single scenario.
"""

import time

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import gaussian_kde

from pipeline import (
    Config, DistParams, PhysicalParams,
    cc_cv_kernel, compute_auto_N, rescale_trips, trip_walk,
    charge_start_intensity, grid_load, resample_kernel, resources,
    diagnostic_indices, fleet_demand_profile, min_feasible_N, run_pipeline,
    run_pipeline_from_events, synth_trips,
)

st.set_page_config(page_title="lightplan", layout="wide")
st.title("lightplan — fleet electrification planner")

SCENARIOS = {
    "Battery swapping": {"daytime_swap": True, "overnight_charge": False, "carryover": True},
    "Home charging": {"daytime_swap": False, "overnight_charge": True, "carryover": False},
    "Hybrid (swap + home)": {"daytime_swap": True, "overnight_charge": True, "carryover": False},
}
SCENARIO_COLOURS = {
    "Battery swapping": "tab:blue",
    "Home charging": "tab:orange",
    "Hybrid (swap + home)": "tab:green",
}

st.caption("Set inputs on tabs 1–4, then run on tab 5. After the run you can switch between the results dashboard and a step-by-step walkthrough.")

# ---------- Help text ----------

TRIP_CSV_HELP = (
    "CSV with one row per trip and these exact column names:\n\n"
    "• **bike_id** — integer ID of the vehicle (e.g. 0, 1, 2 …)\n"
    "• **day** — integer day index (0, 1, 2 …)\n"
    "• **t_start** — trip start time as hours of the day in [0, 24), e.g. 8.25 = 08:15\n"
    "• **dt** — trip duration in hours, e.g. 0.5 = 30 minutes\n"
    "• **d** — trip distance in kilometres\n\n"
    "Example:\n"
    "```\nbike_id,day,t_start,dt,d\n0,0,8.10,0.30,5.2\n0,0,17.40,0.40,6.0\n```"
)
EFF_CSV_HELP = (
    "Optional. CSV with a **single column** of observed energy-efficiency values "
    "in Wh/km (one number per row). If provided, these samples replace the Normal(μ, σ)."
)
SOC_CSV_HELP = (
    "Optional. CSV with a **single column** of observed battery state-of-charge "
    "values at the moment a battery was swapped — each as a **percent** in [0, 100]. "
    "If provided, these samples replace the Normal(μ, σ)."
)

# ---------- Session state ----------

ss = st.session_state
ss.setdefault("trips", None)
ss.setdefault("results", None)
ss.setdefault("input_name", None)
ss.setdefault("user_kernel_raw", None)
ss.setdefault("e_csv_name", None)
ss.setdefault("s_csv_name", None)
ss.setdefault("kernel_csv_name", None)
ss.setdefault("ev_csv_name", None)


def fit_kernel_params(t_min, P_kW, C: float, s_full: float, n_bat: int = 1) -> dict | None:
    """Best-fit (P_CC, cc_fraction, cv_to_cc_ratio) for the parametric CC-CV
    model given an uploaded **battery-side** power profile representing a
    full charge from SoC 0 → s_full. η is a separate user-set property of
    the charger — it doesn't enter this fit."""
    t = np.asarray(t_min, dtype=float)
    P = np.asarray(P_kW, dtype=float)
    if len(P) < 2 or P.max() <= 0:
        return None
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    E_batt = float(trapz(P, t / 60.0))
    if E_batt <= 0:
        return None

    top = max(1, int(0.10 * len(P)))
    P_CC0 = float(np.median(np.sort(P)[-top:]))
    thresh = 0.95 * P_CC0
    i = 0
    while i < len(P) and P[i] >= thresh:
        i += 1
    t_cc_h0 = float(t[max(0, i - 1)] / 60.0)
    cc_frac0 = float(np.clip((P_CC0 * t_cc_h0) / E_batt, 0.30, 0.98))

    horizon_h = float(t.max()) / 60.0 * 1.05
    n_steps = 1024
    dt_h = horizon_h / n_steps
    taus_min = np.arange(n_steps) * dt_h * 60.0

    def predict_on_t(params):
        P_CC, cc_frac, cv_ratio = params
        phys = PhysicalParams(C=C, s_full=s_full, n_bat=int(n_bat),
                              P_CC=float(P_CC), s_cc_cv=float(cc_frac),
                              tau_CV=0.4, charger_efficiency=1.0,
                              cv_to_cc_ratio=float(cv_ratio))
        k = cc_cv_kernel(0.0, phys, dt_h, n_steps)
        return np.interp(t, taus_min, k)

    def loss(params):
        return float(np.sum((predict_on_t(params) - P) ** 2))

    cv_ratio0 = 1.5
    try:
        best = None
        seeds = [
            (P_CC0, cc_frac0, 1.5),
            (P_CC0, cc_frac0, 0.5),
            (P_CC0, cc_frac0, 1.0),
            (P_CC0, cc_frac0, 2.5),
            (P_CC0, 0.70, 1.0),
        ]
        for x0 in seeds:
            res = minimize(
                loss, x0=list(x0),
                bounds=[(0.05, 100.0), (0.30, 0.98), (0.1, 5.0)],
                method="L-BFGS-B",
                options={"maxiter": 400, "ftol": 1e-10},
            )
            if best is None or res.fun < best.fun:
                best = res
        P_CC, cc_frac, cv_ratio = (
            float(best.x[0]), float(best.x[1]), float(best.x[2]),
        )
    except Exception:
        P_CC, cc_frac, cv_ratio = P_CC0, cc_frac0, cv_ratio0
    t_cc_h_fit = (cc_frac * s_full * C) / max(P_CC, 1e-9)
    return {"P_CC": P_CC, "cc_fraction": cc_frac, "eta": 1.0,
            "cv_to_cc_ratio": cv_ratio,
            "P_grid": P_CC, "E_grid": s_full * C,
            "t_cc_min": t_cc_h_fit * 60.0,
            "E_batt": s_full * C, "E_data": E_batt}


def _apply_kernel_fit():
    """Checkbox on_change callback: when toggled ON, copy fitted parametric
    params into widget state and disable the uploaded-kernel pathway.
    η is *not* set from the fit — the uploaded CSV is battery-side, so it
    can't determine η; the user keeps that slider as a separate property."""
    fit = ss.get("_kernel_fit")
    if ss.get("use_fit_chk") and fit is not None:
        C_now = float(ss.get("C") or 3.24)
        ss["c_rate_w"] = float(np.clip(fit["P_CC"] / max(C_now, 1e-9), 0.05, 5.0))
        ss["cc_fraction_w"] = int(round(fit["cc_fraction"] * 100))
        ss["cv_ratio_w"] = float(np.clip(fit.get("cv_to_cc_ratio", 1.5), 0.1, 4.0))
        ss["user_kernel_raw"] = None
        ss["kernel_csv_name"] = None


def _results_name(suffix: str) -> str:
    base = ss.get("input_name") or "run"
    return f"RESULTS_{base}_{suffix}.csv"


def build_summary_df(
    extra: dict | None = None,
    results: dict | None = None,
    runtime_s: float | None = None,
) -> pd.DataFrame:
    rows = [
        ("battery.capacity_kWh", ss.get("C")),
        ("battery.batteries_per_vehicle", ss.get("n_bat")),
        ("battery.s_full", ss.get("s_full")),
        ("charger.P_CC_kW", ss.get("P_CC")),
        ("charger.c_rate", ss.get("c_rate")),
        ("charger.cc_fraction", ss.get("s_cc_cv")),
        ("charger.efficiency", ss.get("charger_efficiency")),
        ("charger.cv_to_cc_ratio", ss.get("cv_to_cc_ratio")),
        ("Fe.mu_Wh_per_km", ss.get("e_mu")),
        ("Fe.sigma_Wh_per_km", ss.get("e_sig")),
        ("Fe.n_samples", len(ss["e_samples"]) if ss.get("e_samples") is not None else 0),
        ("Fs.mu_fraction", ss.get("s_mu")),
        ("Fs.sigma_fraction", ss.get("s_sig")),
        ("Fs.n_samples", len(ss["s_samples"]) if ss.get("s_samples") is not None else 0),
        ("inputs.trips_file", ss.get("input_name")),
        ("inputs.Fe_samples_file", ss.get("e_csv_name")),
        ("inputs.Fs_samples_file", ss.get("s_csv_name")),
        ("inputs.kernel_file", ss.get("kernel_csv_name")),
        ("inputs.events_file", ss.get("ev_csv_name")),
        ("inputs.kernel_source",
         "uploaded_csv" if ss.get("user_kernel_raw") is not None else "parametric_cc_cv"),
    ]
    if extra:
        rows.extend(extra.items())
    if runtime_s is not None:
        rows.append(("runtime_seconds", round(runtime_s, 4)))
    if results:
        for name, out in results.items():
            tag = f"results.{name}"
            res = out.get("resources", {}) or {}
            idx = out.get("indices", {}) or {}
            L = out.get("L_total")
            rows.extend([
                (f"{tag}.N_vehicles", res.get("N_bikes")),
                (f"{tag}.N_chargers_power_based", res.get("N_chargers")),
                (f"{tag}.N_chargers_concurrent", res.get("N_chargers_concurrent")),
                (f"{tag}.N_batteries_total_concurrent",
                 res.get("N_batteries_total_concurrent")),
                (f"{tag}.N_batteries_total", res.get("N_batteries_total")),
                (f"{tag}.runtime_seconds",
                 round(float(out.get("runtime_seconds", 0.0)), 4)
                 if out.get("runtime_seconds") is not None else None),
                (f"{tag}.peak_mean_load_kW",
                 float(L.max()) if L is not None and len(L) else None),
                (f"{tag}.mean_load_kW",
                 float(L.mean()) if L is not None and len(L) else None),
                (f"{tag}.peak_worst_day_load_kW",
                 float(np.max(out.get("L_total_max", [0])))
                 if out.get("L_total_max") is not None else None),
                (f"{tag}.mean_worst_day_load_kW",
                 float(np.mean(out.get("L_total_max", [0])))
                 if out.get("L_total_max") is not None else None),
                (f"{tag}.swap_events", int(len(out.get("swap_t", [])))),
                (f"{tag}.home_events", int(len(out.get("home_t", [])))),
                (f"{tag}.soc_clamp_events", int(out.get("soc_clamp_events", 0))),
                (f"{tag}.chi", idx.get("chi")),
                (f"{tag}.psi", idx.get("psi")),
                (f"{tag}.rho_R", idx.get("rho_R")),
            ])
    return pd.DataFrame(rows, columns=["parameter", "value"])


# ---------- Tabs (shared by both modes) ----------

tab1, tab2, tab3, tab4, tab5, tab_quick = st.tabs([
    "1 · Mobility patterns",
    "2 · Energy use",
    "3 · Swap state-of-charge",
    "4 · Charger profile",
    "5 · Run / step through",
    "⚡ Quick: events → grid load",
])


# ===== Tab 1: Mobility patterns =====
with tab1:
    st.subheader("Trip data")
    st.write(
        "Choose how to provide trip data. Whichever option you pick, the plots below "
        "update live so you can see the mobility pattern before running anything."
    )
    src = st.radio(
        "How do you want to provide trip data?",
        ["Generate synthetic data", "Upload my own CSV"],
        horizontal=True, help=TRIP_CSV_HELP, key="src",
    )
    if src == "Upload my own CSV":
        up = st.file_uploader("Upload trips CSV", type=["csv"], help=TRIP_CSV_HELP, key="trips_csv")
        if up is not None:
            try:
                ss.trips = pd.read_csv(up)
                ss["input_name"] = up.name.rsplit(".", 1)[0]
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                ss.trips = None
        else:
            ss.trips = None
            ss["input_name"] = None
    else:
        c1, c2 = st.columns(2)
        with c1:
            n_bikes_syn = st.slider("Number of vehicles", 5, 500, 50)
            n_days_syn = st.slider("Number of days of data", 1, 30, 7)
        with c2:
            tpd = st.slider("Average trips per vehicle per day", 1.0, 12.0, 4.0)
            km_per_trip = st.slider("Average distance per trip (km)", 0.5, 30.0, 5.0)
        syn_seed = st.number_input("Random seed for synthetic data", 0, 9999, 42, step=1)
        ss.trips = synth_trips(n_bikes_syn, n_days_syn, tpd, km_per_trip, seed=int(syn_seed))
        ss["input_name"] = "synthetic"

    trips = ss.trips
    required = {"bike_id", "day", "t_start", "dt", "d"}
    if trips is None:
        st.info("Waiting for a CSV upload. Hover the ⓘ on the uploader to see the required format.")
    elif missing := (required - set(trips.columns)):
        st.error(f"Trip data missing columns: {missing}. Required: {sorted(required)}")
    else:
        st.success(f"{len(trips)} trips · {trips['bike_id'].nunique()} vehicles · "
                   f"{trips['day'].nunique()} days · {trips['d'].sum():.0f} total km")
        col_a, col_b = st.columns(2)
        with col_a:
            figA, axA = plt.subplots(figsize=(5, 3))
            axA.hist(trips["t_start"] % 24, bins=48, color="steelblue", alpha=0.85)
            axA.set_xlim(0, 24); axA.set_xlabel("Hour of day")
            axA.set_ylabel("Number of trips"); axA.set_title("When trips start")
            st.pyplot(figA)
        with col_b:
            figB, axB = plt.subplots(figsize=(5, 3))
            axB.hist(trips["d"], bins=30, color="darkorange", alpha=0.85)
            axB.set_xlabel("Trip distance (km)"); axB.set_ylabel("Number of trips")
            axB.set_title("How far each trip is")
            st.pyplot(figB)
        col_c, col_d = st.columns(2)
        with col_c:
            figC, axC = plt.subplots(figsize=(5, 3))
            daily = trips.groupby(["bike_id", "day"])["d"].sum().to_numpy()
            axC.hist(daily, bins=30, color="seagreen", alpha=0.85)
            axC.set_xlabel("Distance per vehicle per day (km)")
            axC.set_ylabel("Number of vehicle-days")
            axC.set_title("Daily riding load per vehicle")
            st.pyplot(figC)
        with col_d:
            figD, axD = plt.subplots(figsize=(5, 3))
            n_days_view = max(trips["day"].nunique(), 1)
            hours_demand = np.zeros(24)
            for h, d in zip(np.floor(trips["t_start"] % 24).astype(int), trips["d"]):
                hours_demand[h % 24] += d
            axD.bar(np.arange(24), hours_demand / n_days_view, color="purple", alpha=0.85)
            axD.set_xlim(-0.5, 23.5); axD.set_xlabel("Hour of day")
            axD.set_ylabel("Average km ridden (fleet)")
            axD.set_title("Fleet-wide riding demand by hour")
            st.pyplot(figD)
        with st.expander("Raw trip table (first 20 rows)"):
            st.dataframe(trips.head(20))


# ===== Tab 2: Energy use =====
with tab2:
    st.subheader("Energy use per kilometre")
    st.write("How many watt-hours of battery each vehicle consumes per kilometre ridden.")
    c1, c2 = st.columns(2)
    with c2:
        e_csv = st.file_uploader("Optional: upload observed Wh/km samples",
                                 type=["csv"], key="ecsv", help=EFF_CSV_HELP)
        e_samples = pd.read_csv(e_csv).iloc[:, 0].to_numpy() if e_csv else None
        ss["e_csv_name"] = e_csv.name if e_csv else None
        if e_samples is not None and len(e_samples):
            st.success(
                f"✅ Using **{len(e_samples)} uploaded samples** as the ground truth. "
                f"Empirical mean = {np.mean(e_samples):.2f} Wh/km, "
                f"std = {np.std(e_samples):.2f}. Sliders are disabled."
            )
        else:
            e_samples = None
    use_samples_e = e_samples is not None
    with c1:
        e_mu = st.slider("Average energy efficiency (Wh/km)", 5.0, 60.0, 34.56,
                         disabled=use_samples_e)
        e_sig = st.slider("Spread of energy efficiency (± Wh/km)", 0.0, 20.0, 0.0,
                          disabled=use_samples_e)
    ss["e_mu"], ss["e_sig"], ss["e_samples"] = e_mu, e_sig, e_samples

    figE, axE = plt.subplots(figsize=(9, 3.2))
    if use_samples_e:
        x_min = max(0, float(np.min(e_samples)) * 0.5)
        x_max = float(np.max(e_samples)) * 1.2
        x = np.linspace(x_min, x_max, 400)
        axE.hist(e_samples, bins=30, density=True, alpha=0.55,
                 color="steelblue", label=f"Uploaded samples (n={len(e_samples)})")
        if len(e_samples) > 1 and np.std(e_samples) > 0:
            kde = gaussian_kde(e_samples)
            axE.plot(x, kde(x), color="navy", lw=2.2, label="KDE fit to samples")
        axE.axvline(float(np.mean(e_samples)), ls=":", color="navy", alpha=0.8,
                    label=f"Empirical mean = {np.mean(e_samples):.1f}")
        axE.set_title("Energy use — empirical distribution from uploaded CSV (used as ground truth)")
    else:
        x_min, x_max = max(0, e_mu - 4 * e_sig), e_mu + 4 * e_sig
        x = np.linspace(x_min, max(x_max, e_mu + 1), 400)
        if e_sig > 0:
            pdf = np.exp(-0.5 * ((x - e_mu) / e_sig) ** 2) / (e_sig * np.sqrt(2 * np.pi))
            axE.plot(x, pdf, color="black", lw=2, label=f"Normal(μ={e_mu:.1f}, σ={e_sig:.1f})")
        axE.axvline(e_mu, ls=":", color="gray", alpha=0.7, label="Mean")
        axE.set_title("Distribution of how much energy each km costs")
    axE.set_xlabel("Energy efficiency (Wh/km)")
    axE.set_ylabel("Probability density")
    axE.legend(); st.pyplot(figE)
    st.caption("Lower values = more efficient vehicles.")


# ===== Tab 3: Swap state-of-charge (in percent) =====
with tab3:
    st.subheader("Battery state of charge when swapped")
    st.write("How depleted the battery typically is at the moment a rider swaps it.")
    c1, c2 = st.columns(2)
    with c2:
        s_csv = st.file_uploader("Optional: upload observed swap-SoC samples (percent)",
                                 type=["csv"], key="scsv", help=SOC_CSV_HELP)
        if s_csv is not None:
            raw = pd.read_csv(s_csv).iloc[:, 0].to_numpy()
            s_samples = raw / 100.0 if np.nanmax(raw) > 1.5 else raw
        else:
            s_samples = None
        ss["s_csv_name"] = s_csv.name if s_csv else None
        if s_samples is not None and len(s_samples):
            st.success(
                f"✅ Using **{len(s_samples)} uploaded samples** as the ground truth. "
                f"Empirical mean = {np.mean(s_samples)*100:.1f}%, "
                f"std = {np.std(s_samples)*100:.1f}%. Sliders are disabled."
            )
        else:
            s_samples = None
    use_samples_s = s_samples is not None
    with c1:
        s_mu_pct = st.slider("Average state of charge at swap (%)", 0, 100, 20,
                             disabled=use_samples_s)
        s_sig_pct = st.slider("Spread of state of charge at swap (± %)", 0, 40, 0,
                              disabled=use_samples_s)
    # internal: 0-1 scale
    s_mu, s_sig = s_mu_pct / 100.0, s_sig_pct / 100.0
    ss["s_mu"], ss["s_sig"], ss["s_samples"] = s_mu, s_sig, s_samples

    figS, axS = plt.subplots(figsize=(9, 3.2))
    x = np.linspace(0, 100, 400)
    if use_samples_s:
        pct = s_samples * 100.0
        axS.hist(pct, bins=30, range=(0, 100), density=True, alpha=0.55,
                 color="darkorange", label=f"Uploaded samples (n={len(s_samples)})")
        if len(pct) > 1 and np.std(pct) > 0:
            kde = gaussian_kde(pct)
            axS.plot(x, kde(x), color="saddlebrown", lw=2.2, label="KDE fit to samples")
        axS.axvline(float(np.mean(pct)), ls=":", color="saddlebrown", alpha=0.8,
                    label=f"Empirical mean = {np.mean(pct):.1f}%")
        axS.set_title("Swap SoC — empirical distribution from uploaded CSV (used as ground truth)")
    else:
        if s_sig_pct > 0:
            pdf = np.exp(-0.5 * ((x - s_mu_pct) / s_sig_pct) ** 2) / (s_sig_pct * np.sqrt(2 * np.pi))
            axS.plot(x, pdf, color="black", lw=2,
                     label=f"Normal(μ={s_mu_pct}%, σ={s_sig_pct}%)")
        axS.axvline(s_mu_pct, ls=":", color="gray", alpha=0.7, label="Mean")
        axS.set_title("How depleted batteries are when riders choose to swap")
    axS.set_xlim(0, 100)
    axS.set_xlabel("State of charge when swapped (%)")
    axS.set_ylabel("Probability density")
    axS.legend(); st.pyplot(figS)
    st.caption("Lower values = riders run batteries closer to empty.")


# ===== Tab 4: Charger profile =====
with tab4:
    st.subheader("Battery and charger specs")
    st.write("These set the shape of a single charging session.")
    st.info("💡 Enter **full capacity**, not usable — the operating window is already accounted for.")
    # ---- Basic settings ----
    c1, c2, c3 = st.columns(3)
    with c1:
        C = st.number_input(
            "Battery capacity (kWh)", 0.1, 250.0, 3.24, step=0.1,
            help="Full capacity — operating window is already accounted for.",
        )
    with c2:
        c_rate = st.number_input(
            "Charger C-rate", 0.001, 5.0, 0.5, step=0.001, format="%.3f",
            key="c_rate_w",
            help=(
                "Charger output power as a multiple of battery capacity. "
                "`P_CC = C-rate × battery capacity`. E.g. 0.5 C on a 3.24 kWh "
                "battery → 1.62 kW charger output."
            ),
        )
    with c3:
        charger_eta = st.slider(
            "Charger efficiency", 0.5, 1.0, 0.95, step=0.01, key="charger_eta_w",
            help="Wall-plug-to-battery efficiency. Grid energy = battery energy / η.",
        )
    P_CC = float(C) * float(c_rate)
    st.caption(f"⚡ Computed charger output power: **{P_CC:.2f} kW** "
               f"(battery-side). Grid plateau = {P_CC/max(charger_eta,1e-9):.2f} kW.")

    # ---- Advanced settings ----
    with st.expander("⚙️ Advanced settings"):
        a1, a2 = st.columns(2)
        with a1:
            s_full_pct = st.slider("Full-charge cutoff (%)", 50, 100, 100)
        with a2:
            cc_fraction_pct = st.slider(
                "CC-phase energy fraction (%)", 50, 95, 85, key="cc_fraction_w",
                help="Fraction of total charge energy delivered during the "
                     "constant-current phase (HallyStrats default = 80%).",
            )
            cv_to_cc_ratio = st.slider(
                "CV/CC time ratio", 0.1, 4.0, 1.5, step=0.05, key="cv_ratio_w",
                help=("CV-phase duration as a multiple of CC-phase duration. "
                      "Default 1.5 = HallyStrats convention. Set lower for a "
                      "steeper CV tail."),
            )
        st.caption("`Batteries per vehicle` is now set per scenario on tab 5.")
    s_full, s_cc_cv = s_full_pct / 100.0, cc_fraction_pct / 100.0
    # n_bat is per-scenario now (set on tab 5); keep a global default for
    # previews and any non-scenario-specific consumers.
    n_bat = 1
    ss.update({"C": C, "n_bat": int(n_bat), "s_full": s_full,
               "P_CC": P_CC, "s_cc_cv": s_cc_cv, "tau_CV": 0.4,
               "charger_efficiency": float(charger_eta),
               "cv_to_cc_ratio": float(cv_to_cc_ratio),
               "c_rate": float(c_rate)})

    phys_prev = PhysicalParams(C=C, s_full=s_full, n_bat=int(n_bat),
                               P_CC=P_CC, s_cc_cv=s_cc_cv, tau_CV=0.4,
                               charger_efficiency=float(charger_eta),
                               cv_to_cc_ratio=float(cv_to_cc_ratio))
    n_steps_prev, dt_prev = 600, 0.01
    starts = [(10, "tab:red"), (30, "tab:orange"),
              (50, "tab:green"), (70, "tab:blue")]
    P_grid = P_CC / max(charger_eta, 1e-9)
    taus = np.arange(n_steps_prev) * dt_prev
    figK, (axG, axB) = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    axGsoc = axG.twinx(); axBsoc = axB.twinx()
    for s_start_pct, colour in starts:
        s0 = s_start_pct / 100.0
        # Kernel is battery-side; grid-side = battery-side / η.
        k_batt = cc_cv_kernel(s0, phys_prev, dt_prev, n_steps_prev)
        k_grid = k_batt / max(charger_eta, 1e-9)
        soc_pct = (s0 + np.cumsum(k_batt) * dt_prev / max(C, 1e-9)) * 100.0
        soc_pct = np.minimum(soc_pct, phys_prev.s_full * 100.0)
        axG.plot(taus, k_grid, color=colour, lw=2, label=f"start {s_start_pct}%")
        axB.plot(taus, k_batt, color=colour, lw=2, label=f"start {s_start_pct}%")
        axGsoc.plot(taus, soc_pct, color=colour, lw=1.0, ls=":", alpha=0.65)
        axBsoc.plot(taus, soc_pct, color=colour, lw=1.0, ls=":", alpha=0.65)
    axG.axhline(P_grid, ls=":", color="gray", alpha=0.7,
                label=f"plateau {P_grid:.2f} kW")
    axB.axhline(P_CC, ls=":", color="gray", alpha=0.7,
                label=f"plateau {P_CC:.2f} kW")
    axG.set_title(f"Grid-side (drawn from grid, η = {charger_eta:.2f})")
    axB.set_title("Battery-side (into battery)")
    for ax in (axG, axB):
        ax.set_xlabel("Time since plugged in (hours)")
        ax.legend(fontsize=8, loc="upper right")
    axG.set_ylabel("Charging power (kW)")
    for ax_soc in (axGsoc, axBsoc):
        ax_soc.set_ylim(0, 105)
        ax_soc.set_ylabel("SoC (%)", color="gray")
        ax_soc.tick_params(axis="y", colors="gray")
    figK.tight_layout()
    st.pyplot(figK)
    st.caption(
        "Solid = charging power (kW, left axis). Dotted = battery SoC (%, "
        "right axis). Lower charger efficiency raises the left plot while "
        "leaving the right one unchanged."
    )

    # Per-arrival-SoC energy table
    st.markdown("**Energy per session by arrival SoC** (single battery)")
    soc_rows_pct = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    e_rows = []
    for s_arr_pct in soc_rows_pct:
        s_arr = s_arr_pct / 100.0
        if s_arr >= phys_prev.s_full:
            continue
        E_batt = (phys_prev.s_full - s_arr) * phys_prev.C  # kWh into battery
        E_grid = E_batt / max(phys_prev.charger_efficiency, 1e-9)
        # Session duration: CC delivers cc_fraction of battery energy at P_CC,
        # then CV phase lasts cv_to_cc_ratio × CC duration.
        t_cc_h = (phys_prev.s_cc_cv * E_batt) / max(P_CC, 1e-9)
        t_total_min = t_cc_h * (1.0 + phys_prev.cv_to_cc_ratio) * 60.0
        e_rows.append({
            "Arrival SoC (%)": s_arr_pct,
            "Energy to battery (kWh)": round(E_batt, 3),
            "Energy from grid (kWh)": round(E_grid, 3),
            "Session duration (min)": round(t_total_min, 1),
        })
    e_table = pd.DataFrame(e_rows)
    # Highlight the 20% → 100% and 0% → 100% rows
    st.dataframe(e_table, hide_index=True, use_container_width=True)
    e_0_100 = float((phys_prev.s_full - 0.0) * phys_prev.C)
    e_20_100 = float((phys_prev.s_full - 0.20) * phys_prev.C)
    st.caption(
        f"Full charge **0% → {phys_prev.s_full*100:.0f}%**: "
        f"{e_0_100:.2f} kWh to battery / "
        f"{e_0_100/max(phys_prev.charger_efficiency,1e-9):.2f} kWh from grid · "
        f"Partial charge **20% → {phys_prev.s_full*100:.0f}%**: "
        f"{e_20_100:.2f} kWh to battery / "
        f"{e_20_100/max(phys_prev.charger_efficiency,1e-9):.2f} kWh from grid."
    )

    st.markdown("---")
    st.markdown("**Optional: upload your own measured charging kernel**")
    k_csv = st.file_uploader(
        "Charging kernel CSV (columns: t_min, P_kW)",
        type=["csv"], key="kernel_csv",
        help=(
            "CSV with two columns: **t_min** (minutes since plug-in) and "
            "**P_kW** (**battery-side** charging power — what flows INTO "
            "the battery, no charger losses included). Represents a single "
            "canonical full charge from SoC 0 to `s_full`. For sessions "
            "starting at higher SoC, the simulator uses the *tail* of the "
            "curve so energy delivered to the battery = `(s_full − s) × C`. "
            "Charger efficiency η is applied separately at the grid step. "
            "Overrides the parametric CC-CV kernel above.\n\n"
            "Minute-by-minute example rows:\n"
            "```\nt_min,P_kW\n0,1.62\n1,1.62\n2,1.62\n3,1.62\n4,1.62\n"
            "5,1.62\n6,1.62\n7,1.62\n8,1.62\n9,1.62\n10,1.62\n11,1.55\n"
            "12,1.47\n13,1.39\n14,1.32\n15,1.25\n20,0.97\n25,0.75\n"
            "30,0.58\n40,0.34\n50,0.20\n60,0.12\n75,0.05\n90,0.00\n```"
        ),
    )
    if k_csv is not None:
        try:
            kdf = pd.read_csv(k_csv)
            if not {"t_min", "P_kW"}.issubset(kdf.columns):
                st.error("CSV must contain columns named exactly `t_min` and `P_kW`.")
                ss["user_kernel_raw"] = None
                ss["kernel_csv_name"] = None
            else:
                t_h_arr = kdf["t_min"].to_numpy(dtype=float) / 60.0
                ss["user_kernel_raw"] = (t_h_arr,
                                          kdf["P_kW"].to_numpy(dtype=float))
                ss["kernel_csv_name"] = k_csv.name
                fit = fit_kernel_params(kdf["t_min"], kdf["P_kW"], C, s_full,
                                        n_bat=int(n_bat))
                ss["_kernel_fit"] = fit

                if fit is not None:
                    st.info(
                        f"📐 **Best-fit parametric params** for this curve "
                        f"(battery-side, η excluded): "
                        f"P_CC = **{fit['P_CC']:.2f} kW**, "
                        f"CC-fraction = **{fit['cc_fraction']*100:.1f}%**, "
                        f"CV/CC ratio = **{fit.get('cv_to_cc_ratio', 1.5):.2f}** "
                        f"(total battery energy {fit['E_batt']:.2f} kWh, "
                        f"CC duration {fit['t_cc_min']:.1f} min, "
                        f"uploaded data ∫P dt = {fit['E_data']:.2f} kWh). "
                        "Charger efficiency stays as you set it above."
                    )
                    st.checkbox(
                        "Adjust parameters to fit uploaded CSV "
                        "(switch to parametric model with fitted values)",
                        value=False, key="use_fit_chk",
                        on_change=_apply_kernel_fit,
                        help=(
                            "Tick this to copy the fitted parameters into the "
                            "sliders above and use the parametric CC-CV "
                            "kernel instead of the raw uploaded curve. "
                            "You can still tweak the sliders afterwards."
                        ),
                    )

                st.success(
                    f"Loaded {len(kdf)} samples · duration "
                    f"{kdf['t_min'].max():.1f} min · peak {kdf['P_kW'].max():.2f} kW. "
                    "Parametric kernel above is now ignored."
                )

                horizon_min = max(float(kdf["t_min"].max()) * 1.05, 60.0)
                dt_h_cmp = (horizon_min / 60.0) / 600
                taus_min = np.arange(600) * dt_h_cmp * 60.0
                # Parametric kernel is battery-side; grid-side = battery / η.
                k_param_cur_batt = cc_cv_kernel(0.0, phys_prev, dt_h_cmp, 600)
                k_param_fit_batt = None
                if fit is not None:
                    phys_fit = PhysicalParams(
                        C=C, s_full=s_full, n_bat=int(n_bat),
                        P_CC=fit["P_CC"], s_cc_cv=fit["cc_fraction"],
                        tau_CV=0.4, charger_efficiency=1.0,
                        cv_to_cc_ratio=fit.get("cv_to_cc_ratio", 1.5),
                    )
                    k_param_fit_batt = cc_cv_kernel(0.0, phys_fit, dt_h_cmp, 600)

                # Uploaded CSV is battery-side per the new convention.
                up_batt = kdf["P_kW"].to_numpy()
                cur_batt = k_param_cur_batt
                fit_batt = k_param_fit_batt
                # Grid-side derivations
                up_grid = up_batt / max(charger_eta, 1e-9)
                cur_grid = cur_batt / max(charger_eta, 1e-9)
                fit_grid = (fit_batt / max(charger_eta, 1e-9)) if fit_batt is not None else None

                figU, (axGu, axBu) = plt.subplots(1, 2, figsize=(11, 3.8),
                                                  sharey=True)
                axGu.plot(kdf["t_min"], up_grid, color="tab:red",
                          lw=2.2, label="Uploaded / η")
                axGu.plot(taus_min, cur_grid, color="tab:blue",
                          lw=1.6, ls="--", label="Param (current) / η")
                if fit_grid is not None:
                    axGu.plot(taus_min, fit_grid, color="tab:green",
                              lw=1.8, ls="-.", label="Param (best-fit) / η")
                axBu.plot(kdf["t_min"], up_batt, color="tab:red",
                          lw=2.2, label="Uploaded")
                axBu.plot(taus_min, cur_batt, color="tab:blue",
                          lw=1.6, ls="--", label="Param (current)")
                if fit_batt is not None:
                    axBu.plot(taus_min, fit_batt, color="tab:green",
                              lw=1.8, ls="-.", label="Param (best-fit)")

                axGsoc = axGu.twinx(); axBsoc = axBu.twinx()
                t_min_arr = kdf["t_min"].to_numpy()
                dt_min_up = np.diff(t_min_arr, prepend=t_min_arr[0])
                soc_up = np.cumsum(up_batt * (dt_min_up / 60.0)) / max(C, 1e-9) * 100.0
                soc_up = np.minimum(soc_up, s_full * 100.0)
                soc_cur = np.cumsum(cur_batt) * (dt_h_cmp) / max(C, 1e-9) * 100.0
                soc_cur = np.minimum(soc_cur, s_full * 100.0)
                axGsoc.plot(t_min_arr, soc_up, color="tab:red", lw=1.0,
                            ls=":", alpha=0.6)
                axGsoc.plot(taus_min, soc_cur, color="tab:blue", lw=1.0,
                            ls=":", alpha=0.6)
                axBsoc.plot(t_min_arr, soc_up, color="tab:red", lw=1.0,
                            ls=":", alpha=0.6)
                axBsoc.plot(taus_min, soc_cur, color="tab:blue", lw=1.0,
                            ls=":", alpha=0.6)
                if fit_batt is not None:
                    soc_fit = np.cumsum(fit_batt) * dt_h_cmp / max(C, 1e-9) * 100.0
                    soc_fit = np.minimum(soc_fit, s_full * 100.0)
                    axGsoc.plot(taus_min, soc_fit, color="tab:green", lw=1.0,
                                ls=":", alpha=0.6)
                    axBsoc.plot(taus_min, soc_fit, color="tab:green", lw=1.0,
                                ls=":", alpha=0.6)
                for ax_soc in (axGsoc, axBsoc):
                    ax_soc.set_ylim(0, 105)
                    ax_soc.set_ylabel("SoC (%)", color="gray")
                    ax_soc.tick_params(axis="y", colors="gray")

                axGu.set_title("Grid-side")
                axBu.set_title(f"Battery-side (η = {charger_eta:.2f})")
                for ax in (axGu, axBu):
                    ax.set_xlabel("Time since plugged in (minutes)")
                    ax.set_xlim(0, horizon_min)
                    ax.legend(fontsize=7, loc="upper right")
                axGu.set_ylabel("Charging power (kW)")
                figU.suptitle("Uploaded vs parametric (single battery, full charge from 0%)",
                              fontsize=10)
                figU.tight_layout()
                st.pyplot(figU)
                st.caption("Solid = power (left axis). Dotted = battery SoC (%, right axis).")
        except Exception as e:
            st.error(f"Could not read kernel CSV: {e}")
            ss["user_kernel_raw"] = None
            ss["kernel_csv_name"] = None
    else:
        ss["user_kernel_raw"] = None
        ss["kernel_csv_name"] = None

    st.markdown("---")
    st.caption(
        "The parametric kernel follows the model from "
        "[HallyStrats/li-ion_charging_profile]"
        "(https://github.com/HallyStrats/li-ion_charging_profile): "
        "CC phase delivers `CC-phase energy fraction` of the target energy at "
        "`P_CC`, then CV phase decays exponentially with λ chosen by binary "
        "search so the total grid energy equals `(s_full − s) × C / η`."
    )


# ===== Tab 5: Run / step through =====
with tab5:
    st.subheader("Fleet, scenarios, then run")

    st.markdown("**Fleet sizing**")
    n_data_vehicles = (ss.trips["bike_id"].nunique()
                       if ss.trips is not None else None)
    match_N = st.checkbox(
        "Match number of vehicles to input data",
        value=True, disabled=(n_data_vehicles is None),
        help=(
            f"Set N equal to the number of unique `bike_id`s in the trip data "
            f"({n_data_vehicles if n_data_vehicles is not None else '—'}). "
            "Overrides the manual number and the auto-size option."
        ),
    )
    auto_N = st.checkbox(
        "Auto-size the fleet from peak demand", value=False, disabled=match_N,
        help="If on, the smallest fleet size that can serve peak hourly demand is picked. "
             "If off (default), specify the number of vehicles directly.",
    )
    if match_N and n_data_vehicles is not None:
        N_val = int(n_data_vehicles)
        m_max, util = 60.0, 0.8
        st.caption(f"Fleet size locked to **{N_val}** (= unique vehicles in trip data).")
    elif auto_N:
        N_val = None
        c1, c2 = st.columns(2)
        m_max = c1.slider("Max distance per vehicle per day (km)", 10.0, 200.0, 60.0)
        util = c2.slider("Target utilisation (0–1)", 0.1, 1.0, 0.8)
    else:
        N_val = st.number_input("Number of vehicles in the fleet", 1, 5000, 50, step=1)
        m_max, util = 60.0, 0.8

    st.markdown("**Charging scenarios**")
    st.caption("Pick one or more strategies. Multiple selections are overlaid in the dashboard view. "
               "Choose `Batteries per vehicle` per scenario — the number of parallel charging "
               "sessions that fire at each charge-start event.")
    chosen: list[str] = []
    scen_nbat: dict[str, int] = {}
    c1, c2, c3 = st.columns(3)
    for col, name, default, nbat_default in [
        (c1, "Battery swapping", True, 1),
        (c2, "Home charging", True, 2),
        (c3, "Hybrid (swap + home)", True, 1),
    ]:
        with col:
            picked = st.checkbox(name, value=default, key=f"scen_chk_{name}")
            scen_nbat[name] = st.selectbox(
                "Batteries per vehicle",
                [1, 2], index=[1, 2].index(nbat_default),
                key=f"scen_nbat_{name}",
                disabled=not picked,
            )
            if picked:
                chosen.append(name)

    c1, c2 = st.columns(2)
    n_bins = c1.select_slider("Load curve resolution (bins per 24 h)",
                              [96, 288, 720, 1440], value=1440)
    seed = c2.number_input("Simulation random seed", 0, 9999, 42, step=1)

    st.markdown("---")
    can_run = (ss.trips is not None) and (len(chosen) > 0)
    if ss.trips is None:
        st.warning("No trip data — set this up on tab 1 before running.")
    if not chosen:
        st.warning("Pick at least one charging scenario.")

    # ---- Feasibility gate ----
    feasibility_block = None
    if can_run:
        Fe_mean_tmp = (float(np.mean(ss["e_samples"]))
                       if ss["e_samples"] is not None and len(ss["e_samples"])
                       else ss["e_mu"])
        for name in chosen:
            triggers = SCENARIOS[name]
            cfg_tmp = Config(N=N_val, m_max=m_max, utilisation=util,
                             n_bins=int(n_bins), seed=int(seed), **triggers)
            phys_tmp = PhysicalParams(
                C=ss["C"], s_full=ss["s_full"],
                n_bat=int(scen_nbat.get(name, 1)),
                P_CC=ss["P_CC"], s_cc_cv=ss["s_cc_cv"], tau_CV=ss["tau_CV"],
                charger_efficiency=ss.get("charger_efficiency", 1.0),
                cv_to_cc_ratio=ss.get("cv_to_cc_ratio", 1.5),
            )
            feas = min_feasible_N(ss.trips, Fe_mean_tmp, phys_tmp, cfg_tmp)
            if feas["min_N"] is None:
                continue  # swap available — energy not the constraint
            if feas["single_trip_infeasible"]:
                feasibility_block = (
                    f"❌ **Infeasible — {name}.** "
                    f"The longest single trip is {feas['max_trip_km']:.1f} km, "
                    f"which exceeds one full charge "
                    f"({feas['range_km']:.1f} km at {Fe_mean_tmp:.0f} Wh/km, "
                    f"{phys_tmp.n_bat}× {ss['C']:.2f} kWh). "
                    "Home-only charging cannot serve this trip regardless of "
                    "fleet size. Enable battery swapping, increase battery "
                    "capacity, or split long trips."
                )
                break
            current_N = N_val if N_val is not None else compute_auto_N(ss.trips, m_max, util)
            if current_N < feas["min_N"]:
                feasibility_block = (
                    f"❌ **Infeasible — {name}.** "
                    f"Fleet rides {feas['total_km_per_day']:.0f} km/day; with "
                    f"home-only charging and a per-vehicle range of "
                    f"{feas['range_km']:.1f} km, the **minimum fleet size is "
                    f"{feas['min_N']}** (you set {current_N}). "
                    "Raise the fleet size, enable daytime swapping, or increase "
                    "battery capacity / efficiency."
                )
                break

    if feasibility_block:
        st.error(feasibility_block)

    run_btn = st.button("▶ Run pipeline", type="primary", use_container_width=True,
                        disabled=(not can_run) or (feasibility_block is not None))

    if run_btn and can_run and not feasibility_block:
        Fe = DistParams(mu=ss["e_mu"], sigma=ss["e_sig"], samples=ss["e_samples"])
        Fs = DistParams(mu=ss["s_mu"], sigma=ss["s_sig"], samples=ss["s_samples"])
        results = {}
        uk_raw = ss.get("user_kernel_raw")
        user_kernel = (resample_kernel(uk_raw[0], uk_raw[1], n_bins=int(n_bins))
                       if uk_raw is not None else None)
        _t_total = time.perf_counter()
        with st.spinner("Running pipeline…"):
            for name in chosen:
                triggers = SCENARIOS[name]
                cfg = Config(N=N_val, m_max=m_max, utilisation=util,
                             n_bins=int(n_bins), seed=int(seed), **triggers)
                n_bat_scen = int(scen_nbat.get(name, 1))
                phys = PhysicalParams(
                    C=ss["C"], s_full=ss["s_full"], n_bat=n_bat_scen,
                    P_CC=ss["P_CC"], s_cc_cv=ss["s_cc_cv"], tau_CV=ss["tau_CV"],
                    charger_efficiency=ss.get("charger_efficiency", 1.0),
                    cv_to_cc_ratio=ss.get("cv_to_cc_ratio", 1.5),
                )
                _t_scen = time.perf_counter()
                out = run_pipeline(ss.trips, Fe, Fs, phys, cfg, user_kernel=user_kernel)
                out["runtime_seconds"] = time.perf_counter() - _t_scen
                if out.get("soc_clamp_events", 0) > 0:
                    st.warning(
                        f"⚠️ **{name}**: {out['soc_clamp_events']} trip(s) drew "
                        f"more energy than a full battery could supply (upper-tail "
                        f"`Fe` draws) — SoC was clamped to 0 for those. Results "
                        f"still produced; raise battery capacity / lower mean "
                        f"Wh/km if the rate is high."
                    )
                out["_cfg"] = cfg  # stash for walkthrough
                out["_phys"] = phys
                out["_Fe"] = Fe
                out["_Fs"] = Fs
                results[name] = out
        ss["runtime_s"] = time.perf_counter() - _t_total
        ss.results = results


# ===== Quick tab: events → grid load (bypass trip walk) =====
with tab_quick:
    st.subheader("Skip the trip walk — go straight from charge-start events to grid load")
    st.write(
        "Use this when you already have empirical charge-start times and SoC-at-"
        "plug-in values, and just want the kernel/load/charger-count answers. "
        "Upload a CSV with **two columns**:"
    )
    st.markdown(
        "- **t** — charge-start time in hours. Either hour-of-day in `[0, 24)` "
        "(treated as a single day) or absolute hours across multiple days "
        "(e.g. `25.3` = 01:18 on day 1).\n"
        "- **s** — battery SoC at the moment of plug-in. Percent (0–100) "
        "or fraction (0–1) — auto-detected."
    )
    ev_csv = st.file_uploader("Charge-start events CSV", type=["csv"], key="ev_csv")
    ev_df = None
    ev_basename = None
    if ev_csv is None:
        ss["ev_csv_name"] = None
    if ev_csv is not None:
        try:
            ev_df = pd.read_csv(ev_csv)
            ev_basename = ev_csv.name.rsplit(".", 1)[0]
            ss["ev_csv_name"] = ev_csv.name
            if not {"t", "s"}.issubset(ev_df.columns):
                st.error("CSV must contain columns named exactly `t` and `s`.")
                ev_df = None
            else:
                s_vals = ev_df["s"].to_numpy(dtype=float)
                s_pct = s_vals * 100.0 if np.nanmax(s_vals) <= 1.5 else s_vals
                st.success(f"{len(ev_df)} events loaded. "
                           f"t range: {ev_df['t'].min():.2f}–{ev_df['t'].max():.2f} h · "
                           f"s mean: {np.mean(s_pct):.1f}%")
                pc1, pc2 = st.columns(2)
                with pc1:
                    figT, axT = plt.subplots(figsize=(5, 2.8))
                    axT.hist(np.mod(ev_df["t"].to_numpy(), 24.0), bins=48,
                             range=(0, 24), color="tab:purple", alpha=0.85)
                    axT.set_xlim(0, 24)
                    axT.set_xlabel("Hour of day"); axT.set_ylabel("Events")
                    axT.set_title("Charge-start time distribution")
                    st.pyplot(figT)
                with pc2:
                    figS, axS_q = plt.subplots(figsize=(5, 2.8))
                    axS_q.hist(s_pct, bins=30, range=(0, 100),
                               color="darkorange", alpha=0.85)
                    axS_q.set_xlim(0, 100)
                    axS_q.set_xlabel("SoC at charge start (%)")
                    axS_q.set_ylabel("Events")
                    axS_q.set_title("Arrival SoC distribution")
                    st.pyplot(figS)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    c1, c2, c3 = st.columns(3)
    with c1:
        q_N = st.number_input("Fleet size N", 1, 5000, 50, step=1, key="q_N")
    with c2:
        q_days = st.number_input("Number of days the events cover", 1, 90, 1,
                                 step=1, key="q_days")
    with c3:
        q_nbins = st.number_input("Time bins (resolution)", 96, 4320, 1440,
                                  step=96, key="q_nbins")

    st.caption("Charger/battery specs are read from **tab 4**. Set them before running.")

    if st.button("▶ Run quick pipeline", type="primary",
                 use_container_width=True, disabled=ev_df is None):
        phys_q = PhysicalParams(C=ss["C"], s_full=ss["s_full"], n_bat=ss["n_bat"],
                                P_CC=ss["P_CC"], s_cc_cv=ss["s_cc_cv"], tau_CV=ss["tau_CV"], charger_efficiency=ss.get("charger_efficiency", 1.0), cv_to_cc_ratio=ss.get("cv_to_cc_ratio", 1.5))
        cfg_q = Config(N=int(q_N), n_bins=int(q_nbins))
        uk_raw_q = ss.get("user_kernel_raw")
        user_kernel_q = (resample_kernel(uk_raw_q[0], uk_raw_q[1], n_bins=int(q_nbins))
                         if uk_raw_q is not None else None)
        _t0 = time.perf_counter()
        with st.spinner("Computing…"):
            out = run_pipeline_from_events(ev_df, phys_q, cfg_q,
                                           n_days=int(q_days),
                                           user_kernel=user_kernel_q)
        q_runtime = time.perf_counter() - _t0

        res = out["resources"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Peak load (kW)", f"{out['L_total'].max():.2f}")
        m2.metric("Mean load (kW)", f"{out['L_total'].mean():.2f}")
        m3.metric(
            "Chargers required",
            res.get("N_chargers_concurrent", res["N_chargers"]),
            help=f"Power-based estimate: {res['N_chargers']}. "
                 "Concurrent-session count shown here is more accurate when "
                 "CV-tail overlap matters.",
        )
        m4.metric("Peak/avg ratio ρ_R", f"{out['indices']['rho_R']:.2f}")

        fig, ax = plt.subplots(figsize=(9, 3.0))
        ax.plot(out["t_axis"], out["L_total"], color="black", lw=1.6, label="mean")
        ax.fill_between(out["t_axis"], 0, out["L_total"], color="tab:blue", alpha=0.25)
        if out.get("L_total_max") is not None and float(np.max(out["L_total_max"])) > 0:
            ax.plot(out["t_axis"], out["L_total_max"], color="black",
                    lw=1.2, ls=":", label="worst day")
            ax.legend(fontsize=8)
        ax.set_xlabel("Hour of day"); ax.set_ylabel("Grid-side fleet load (kW)")
        ax.set_title("Grid load L(t)"); ax.set_xlim(0, 24)
        st.pyplot(fig)

        figL, axL = plt.subplots(figsize=(9, 2.6))
        axL.bar(np.arange(24), out["Lambda"], color="tab:purple", alpha=0.85)
        axL.set_xlabel("Hour of day")
        axL.set_ylabel("Charge starts / hour (fleet)")
        axL.set_title("Charge-start intensity Λ(t)")
        st.pyplot(figL)

        load_df = pd.DataFrame({
            "t_hours": out["t_axis"],
            "L_mean_kW": out["L_total"],
            "L_max_kW": out.get("L_total_max",
                                np.zeros_like(out["L_total"])),
        })
        lam_df = pd.DataFrame({"hour": np.arange(24), "Lambda_per_h": out["Lambda"]})
        prev_name = ss.get("input_name")
        ss["input_name"] = ev_basename or prev_name
        summary_q_df = build_summary_df(
            extra={
                "mode": "quick_events",
                "N_fleet": int(q_N),
                "n_days": int(q_days),
                "n_bins": int(q_nbins),
                "n_events": len(ev_df),
            },
            results={"quick": out},
            runtime_s=q_runtime,
        )
        d1, d2, d3 = st.columns(3)
        d1.download_button("Grid load CSV", load_df.to_csv(index=False),
                           _results_name("load"), "text/csv", use_container_width=True)
        d2.download_button("Λ(t) CSV", lam_df.to_csv(index=False),
                           _results_name("lambda"), "text/csv", use_container_width=True)
        d3.download_button("Summary CSV", summary_q_df.to_csv(index=False),
                           _results_name("summary"), "text/csv", use_container_width=True)
        ss["input_name"] = prev_name


# ============================================================
# RESULTS DASHBOARD (mode 1)
# ============================================================

def render_dashboard(results):
    st.divider(); st.header("Results")

    summary_rows = []
    for name, out in results.items():
        dt_h = 24.0 / len(out["L_total"])
        daily_kwh = float(out["L_total"].sum() * dt_h)
        avg_kw = daily_kwh / 24.0
        summary_rows.append({
            "Scenario": name,
            "Fleet size": out["N"],
            "Daily fleet energy (kWh)": round(daily_kwh, 1),
            "Average grid load (kW)": round(avg_kw, 2),
            "Peak grid load (kW)": round(float(out["L_total"].max()), 2),
            "Chargers (power)": out["resources"]["N_chargers"],
            "Chargers (concurrent)":
                out["resources"].get("N_chargers_concurrent",
                                     out["resources"]["N_chargers"]),
            "Total batteries": out["resources"].get(
                "N_batteries_total_concurrent",
                out["resources"]["N_batteries_total"]),
            "Alignment χ": round(out["indices"]["chi"], 3),
            "Range adequacy ψ": ("∞" if not np.isfinite(out["indices"]["psi"])
                                 else round(out["indices"]["psi"], 2)),
            "Peak/avg ρ_R": round(out["indices"]["rho_R"], 2),
        })
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)
    st.caption("χ — charging-vs-riding alignment (low good). "
               "ψ — range adequacy (≥1.5 good). "
               "ρ_R — peak-to-average load (near 1 good).")

    fig1, ax1 = plt.subplots(figsize=(9, 3.2))
    any_max = False
    for name, out in results.items():
        col = SCENARIO_COLOURS[name]
        ax1.plot(out["t_axis"], out["L_total"], lw=2, color=col,
                 label=f"{name} (mean)")
        L_max = out.get("L_total_max")
        if L_max is not None and float(np.max(L_max)) > 0:
            ax1.plot(out["t_axis"], L_max, lw=1.2, ls=":", color=col,
                     label=f"{name} (worst day)")
            any_max = True
    ax1.set_xlabel("Hour of day"); ax1.set_ylabel("Grid load (kW, fleet-wide)")
    ax1.set_title("Grid load through the day"); ax1.set_xlim(0, 24)
    ax1.legend(fontsize=8, ncol=2 if any_max else 1)
    st.pyplot(fig1)
    if any_max:
        st.caption("Solid = mean fleet load (average across all data days). "
                   "Dotted = worst-day envelope (per-minute-of-day max across all data days). "
                   "Size infrastructure on the dotted line.")

    event_scenarios = {n: o for n, o in results.items() if len(o["swap_t"]) + len(o["home_t"])}
    fig2, ax2 = plt.subplots(figsize=(9, 3))
    if event_scenarios:
        hours = np.arange(24); n_scen = len(event_scenarios)
        width = 0.8 / max(n_scen, 1)
        for i, (name, out) in enumerate(event_scenarios.items()):
            offset = (i - (n_scen - 1) / 2) * width
            ax2.bar(hours + offset, out["Lambda"], width=width,
                    color=SCENARIO_COLOURS[name], label=name, alpha=0.85)
        ax2.legend()
    ax2.set_xlabel("Hour of day")
    ax2.set_ylabel("Charge-start events per hour (fleet-wide)")
    ax2.set_title("When charging sessions begin during the day")
    ax2.set_xlim(-0.5, 23.5); st.pyplot(fig2)
    st.caption("A *charge-start event* is either a daytime swap or an end-of-day home plug-in.")

    col_a, col_b = st.columns(2)
    with col_a:
        fig3, ax3 = plt.subplots(figsize=(5, 3))
        if event_scenarios:
            for name, out in event_scenarios.items():
                s_all = np.concatenate([out["swap_s"], out["home_s"]]) * 100.0
                ax3.hist(s_all, bins=30, alpha=0.5,
                         color=SCENARIO_COLOURS[name], label=name)
            ax3.legend(fontsize=8)
        ax3.set_xlabel("State of charge at plug-in (%)")
        ax3.set_ylabel("Number of events")
        ax3.set_title("How depleted batteries are when charging starts")
        st.pyplot(fig3)
    with col_b:
        fig4, ax4 = plt.subplots(figsize=(5, 3))
        if event_scenarios:
            for name, out in event_scenarios.items():
                t_all = np.concatenate([out["swap_t"], out["home_t"]])
                ax4.hist(np.mod(t_all, 24.0), bins=48, alpha=0.5,
                         color=SCENARIO_COLOURS[name], label=name)
            ax4.legend(fontsize=8)
        ax4.set_xlabel("Hour of day"); ax4.set_ylabel("Number of events")
        ax4.set_title("When charging sessions begin"); ax4.set_xlim(0, 24)
        st.pyplot(fig4)

    st.subheader("Downloads")
    event_frames, load_frames = [], []
    for name, out in results.items():
        event_frames.append(pd.DataFrame({
            "scenario": name,
            "type": ["swap"] * len(out["swap_t"]) + ["home"] * len(out["home_t"]),
            "t": np.concatenate([out["swap_t"], out["home_t"]])
                 if len(out["home_t"]) else out["swap_t"],
            "soc_percent": np.concatenate([out["swap_s"], out["home_s"]]) * 100.0
                           if len(out["home_s"]) else out["swap_s"] * 100.0,
        }))
        load_frames.append(pd.DataFrame({
            "scenario": name, "t_hour": out["t_axis"],
            "L_total_mean_kW": out["L_total"],
            "L_swap_mean_kW": out["L_swap"], "L_home_mean_kW": out["L_home"],
            "L_total_max_kW": out.get("L_total_max",
                                       np.zeros_like(out["L_total"])),
            "L_swap_max_kW": out.get("L_swap_max",
                                      np.zeros_like(out["L_total"])),
            "L_home_max_kW": out.get("L_home_max",
                                      np.zeros_like(out["L_total"])),
        }))
    events = pd.concat(event_frames, ignore_index=True)
    loads = pd.concat(load_frames, ignore_index=True)
    summary_df = build_summary_df(
        extra={
            "mode": "full_pipeline",
            "scenarios": ", ".join(results.keys()),
            "N_fleet": next(iter(results.values()))["N"],
            "N_data": next(iter(results.values()))["N_data"],
            "n_days": next(iter(results.values()))["n_days"],
        },
        results=results,
        runtime_s=ss.get("runtime_s"),
    )
    d1, d2, d3 = st.columns(3)
    d1.download_button("Events CSV", events.to_csv(index=False),
                       _results_name("events"), "text/csv", use_container_width=True)
    d2.download_button("Load curve CSV", loads.to_csv(index=False),
                       _results_name("load"), "text/csv", use_container_width=True)
    d3.download_button("Summary CSV", summary_df.to_csv(index=False),
                       _results_name("summary"), "text/csv", use_container_width=True)


# ============================================================
# WALKTHROUGH (mode 2)
# ============================================================

def render_walkthrough(results):
    names = list(results.keys())
    if len(names) > 1:
        name = st.radio("Scenario to walk through", names,
                        horizontal=True, key="walk_scen")
    else:
        name = names[0]
    out = results[name]
    cfg, phys, Fe, Fs = out["_cfg"], out["_phys"], out["_Fe"], out["_Fs"]
    trips_used = out["trips_used"]
    swap_t, swap_s = out["swap_t"], out["swap_s"]
    home_t, home_s = out["home_t"], out["home_s"]
    Lambda = out["Lambda"]
    t_axis, L_swap, L_home, L_total = (out["t_axis"], out["L_swap"],
                                       out["L_home"], out["L_total"])
    res, idx, N = out["resources"], out["indices"], out["N"]
    N_data, n_days_eff = out["N_data"], out["n_days"]

    st.divider()
    st.header(f"Walkthrough — {name}")

    STEPS = [
        "1 · Inputs", "2 · Fleet size", "3 · Trip walk",
        "4 · Charge-start intensity Λ(t)", "5 · CC-CV kernel",
        "6 · Grid load L(t)", "7 · Resources", "8 · Diagnostic indices",
        "9 · Summary",
    ]
    step = st.radio("Step", STEPS, horizontal=True, label_visibility="collapsed")
    st.divider()

    if step == STEPS[0]:
        st.markdown(
            "The pipeline takes a **trip log**, two **distributions** "
            "(energy use `F_e` in Wh/km and swap state-of-charge `F_s` as a "
            "percent), and a few **physical parameters** (battery capacity, "
            "charger power, etc.)."
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Trip log (first 10 rows)**")
            st.dataframe(trips_used.head(10), use_container_width=True)
            st.caption(f"{len(trips_used):,} trips · {N_data} vehicles · {n_days_eff} day(s).")
            m_profile = fleet_demand_profile(trips_used, n_hours=24)
            fig, ax = plt.subplots(figsize=(5.5, 2.2))
            ax.bar(np.arange(24), m_profile, color="tab:gray")
            ax.set_xlabel("Hour of day"); ax.set_ylabel("km / vehicle-day")
            ax.set_title("Hourly per-vehicle demand m(t)")
            st.pyplot(fig)
        with c2:
            st.markdown("**Energy-use distribution F_e (Wh/km)**")
            x = np.linspace(max(0, Fe.mu - 4*Fe.sigma), Fe.mu + 4*Fe.sigma, 200)
            y = np.exp(-0.5 * ((x - Fe.mu) / max(Fe.sigma, 1e-6))**2)
            y /= y.sum() * (x[1] - x[0])
            fig, ax = plt.subplots(figsize=(5.5, 2.0))
            ax.plot(x, y, color="tab:blue"); ax.fill_between(x, y, alpha=0.2, color="tab:blue")
            ax.set_xlabel("Wh / km"); ax.set_ylabel("density")
            st.pyplot(fig)
            st.markdown("**Swap-SoC distribution F_s (%)**")
            x = np.linspace(0, 100, 200)
            mu_p, sig_p = Fs.mu * 100, max(Fs.sigma * 100, 1e-6)
            y = np.exp(-0.5 * ((x - mu_p) / sig_p)**2); y /= y.sum() * (x[1] - x[0])
            fig, ax = plt.subplots(figsize=(5.5, 2.0))
            ax.plot(x, y, color="tab:orange"); ax.fill_between(x, y, alpha=0.2, color="tab:orange")
            ax.set_xlabel("SoC at swap (%)"); ax.set_ylabel("density")
            st.pyplot(fig)

    elif step == STEPS[1]:
        st.markdown(
            f"Fleet size used: **{N}**. Auto mode picks the smallest fleet "
            "that can serve peak hourly demand:"
        )
        st.latex(r"M_\text{peak} = \max_h \sum_{\text{trips in hour } h} d \,/\, n_\text{days}")
        st.latex(r"N_\text{auto} = \left\lceil \frac{M_\text{peak}}{m_\text{max} \cdot \text{utilisation}} \right\rceil")
        hours = np.floor(trips_used["t_start"].to_numpy()).astype(int) % 24
        dem = np.zeros(24)
        for h, d in zip(hours, trips_used["d"].to_numpy()):
            dem[h] += d
        dem /= n_days_eff
        fig, ax = plt.subplots(figsize=(7, 2.4))
        ax.bar(np.arange(24), dem, color="tab:gray")
        ax.axhline(dem.max(), ls="--", color="tab:red",
                   label=f"M_peak = {dem.max():.1f} km/h")
        ax.set_xlabel("Hour of day"); ax.set_ylabel("km / hour (fleet, daily avg)")
        ax.legend(); st.pyplot(fig)

    elif step == STEPS[2]:
        st.markdown(
            "We walk each vehicle's trips in time order. After each trip we subtract "
            "energy used from the state of charge. When SoC drops below the rider's "
            "(stochastic) target, we record a **swap event** at the trip's end time "
            "and reset SoC to full. If overnight charging is on, the end-of-day SoC "
            "is also recorded as a **home plug-in event**."
        )
        st.latex(r"\Delta\text{SoC} = \frac{e \cdot d_\text{trip}}{n_\text{bat} \cdot C \cdot 1000}")
        st.markdown(
            f"Triggers: **daytime_swap = {cfg.daytime_swap}**, "
            f"**overnight_charge = {cfg.overnight_charge}**, "
            f"**carryover = {cfg.carryover}**."
        )
        rng = np.random.default_rng(int(cfg.seed))
        cap_wh = phys.n_bat * phys.C * 1000.0
        sample_bikes = sorted(trips_used["bike_id"].unique())[:3]
        fig, ax = plt.subplots(figsize=(8, 3.2))
        swap_drawn = home_drawn = False
        for b in sample_bikes:
            bt = trips_used[trips_used["bike_id"] == b].sort_values(["day", "t_start"])
            ts, socs, sw_t, sw_s, hm_t, hm_s = [], [], [], [], [], []
            def _walk_step(soc, target, e, d, t_start, t_end):
                soc_after = soc - (e * d) / cap_wh
                if cfg.daytime_swap and soc_after < max(target, 0.0):
                    sw_t.append(t_start); sw_s.append(soc * 100)
                    soc = phys.s_full; target = Fs.draw(rng)
                    soc_after = soc - (e * d) / cap_wh
                soc_after = max(soc_after, 0.0)
                ts.append(t_end); socs.append(soc_after * 100)
                return soc_after, target
            if cfg.carryover:
                soc = phys.s_full; target = Fs.draw(rng)
                for _, trip in bt.iterrows():
                    t_start = trip["day"] * 24.0 + trip["t_start"]
                    t_end = t_start + trip["dt"]
                    soc, target = _walk_step(soc, target, Fe.draw(rng),
                                             trip["d"], t_start, t_end)
            else:
                for day, bd in bt.groupby("day"):
                    soc = phys.s_full; target = Fs.draw(rng)
                    last_t = None
                    for _, trip in bd.iterrows():
                        t_start = day * 24.0 + trip["t_start"]
                        t_end = t_start + trip["dt"]
                        soc, target = _walk_step(soc, target, Fe.draw(rng),
                                                 trip["d"], t_start, t_end)
                        last_t = t_end
                    if cfg.overnight_charge and last_t is not None:
                        hm_t.append(last_t); hm_s.append(max(soc, 0) * 100)
            ax.plot(ts, socs, marker="o", ms=3, label=f"vehicle {b}")
            if sw_t:
                ax.scatter(sw_t, sw_s, marker="x", s=80, color="red", zorder=5,
                           label="swap" if not swap_drawn else None)
                swap_drawn = True
            if hm_t:
                ax.scatter(hm_t, hm_s, marker="^", s=70, color="green",
                           edgecolors="black", zorder=5,
                           label="home plug-in" if not home_drawn else None)
                home_drawn = True
        ax.set_xlabel("time (h)"); ax.set_ylabel("State of charge (%)")
        ax.set_ylim(0, 105)
        ax.axhline(Fs.mu * 100, ls="--", color="tab:orange",
                   label=f"swap target ≈ {Fs.mu*100:.0f}%")
        ax.legend(loc="lower left", fontsize=8)
        ax.set_title("SoC for 3 example vehicles — red × = swap, green ▲ = home plug-in")
        st.pyplot(fig)
        c1, c2 = st.columns(2)
        c1.metric("Swap events", len(swap_t))
        c2.metric("Home plug-in events", len(home_t))

    elif step == STEPS[3]:
        st.markdown(
            "A **charge-start event** is anything that begins a charging "
            "session — a daytime swap or an end-of-day home plug-in. Bin them "
            "into hours and scale to the planned fleet."
        )
        st.latex(
            r"\Lambda(t) = \frac{N}{N_\text{data} \cdot n_\text{days}} \cdot "
            r"\frac{\text{count of events in hour } t}{\Delta t}"
        )
        fig, ax = plt.subplots(figsize=(8, 2.6))
        ax.bar(np.arange(24), Lambda, color="tab:blue")
        ax.set_xlabel("Hour of day"); ax.set_ylabel("charge starts / hour (fleet)")
        st.pyplot(fig)
        st.caption(
            f"Swap events: {len(swap_t)}; home plug-ins: {len(home_t)}. "
            f"Peak hour ≈ {int(np.argmax(Lambda)):02d}:00 with {Lambda.max():.2f} starts/h."
        )

    elif step == STEPS[4]:
        st.markdown(
            "Each charging session draws power according to a fixed shape: a "
            "**constant-current (CC)** plateau, then an **exponential CV decay** "
            "as SoC approaches full. The kernel below is for a **single battery on "
            "its own charger** — a vehicle with $n_\\text{bat}$ batteries fires "
            "$n_\\text{bat}$ such sessions in parallel at each charge-start event."
        )
        st.latex(r"\tau_1(s) = \max\!\left(0, \frac{(s_\text{cc-cv} - s) \cdot C}{P_\text{CC}}\right)")
        st.latex(
            r"K(\tau; s) = \begin{cases}"
            r"P_\text{CC} & 0 \le \tau < \tau_1 \\"
            r"P_\text{CC} \cdot e^{-(\tau - \tau_1)/\tau_\text{CV}} & \tau \ge \tau_1"
            r"\end{cases}"
        )
        dt_h = 1/60.0; n_steps = 60 * 4
        fig, ax = plt.subplots(figsize=(8, 2.8))
        for s_pct in [10, 30, 50, 70, 90]:
            k = cc_cv_kernel(s_pct / 100.0, phys, dt_h, n_steps)
            ax.plot(np.arange(n_steps)*dt_h*60, k, label=f"start SoC = {s_pct}%")
        ax.set_xlabel("minutes since plug-in"); ax.set_ylabel("kW (grid-side, per session)")
        ax.legend(fontsize=8); st.pyplot(fig)

    elif step == STEPS[5]:
        st.markdown(
            f"Place a kernel at every event and sum on a {cfg.n_bins}-bin time "
            "axis (wrapping over 24 h). Each event contributes "
            f"**$n_\\text{{bat}} = {phys.n_bat}$** parallel kernels (one per "
            "battery on its own charger), then scaled by "
            "$N/(N_\\text{data} \\cdot n_\\text{days})$."
        )
        st.latex(r"L(t) = L_\text{swap}(t) + L_\text{home}(t)")
        fig, ax = plt.subplots(figsize=(8, 3.0))
        ax.plot(t_axis, L_swap, label="L_swap (mean)", color="tab:blue")
        ax.plot(t_axis, L_home, label="L_home (mean)", color="tab:orange")
        ax.plot(t_axis, L_total, label="L_total (mean)", color="black", lw=1.8)
        L_total_max = out.get("L_total_max")
        if L_total_max is not None and float(np.max(L_total_max)) > 0:
            ax.plot(t_axis, L_total_max, label="L_total (worst day)",
                    color="black", lw=1.2, ls=":")
        ax.set_xlabel("Hour of day"); ax.set_ylabel("Grid-side fleet load (kW)")
        ax.legend(); st.pyplot(fig)
        c1, c2, c3 = st.columns(3)
        c1.metric("Peak load (kW)", f"{L_total.max():.2f}")
        c2.metric("Average load (kW)", f"{L_total.mean():.2f}")
        c3.metric("Daily energy (kWh)", f"{L_total.mean()*24:.1f}")

    elif step == STEPS[6]:
        st.latex(r"N_\text{chg} = \left\lceil \max_t L(t) \,/\, P_\text{CC} \right\rceil")
        st.latex(r"N_\text{bat,total} = N \cdot n_\text{bat} + N_\text{chg}")
        N_chg_conc = res.get("N_chargers_concurrent", res["N_chargers"])
        N_bat_conc = res.get("N_batteries_total_concurrent",
                             res["N_batteries_total"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Vehicles", res["N_bikes"])
        c2.metric("Chargers (concurrent)", N_chg_conc,
                  help=f"Slot-occupancy count. Power-capacity count = {res['N_chargers']}.")
        c3.metric("Batteries (total)", N_bat_conc,
                  help=f"N · n_bat + concurrent chargers. Power-based total = {res['N_batteries_total']}.")
        st.caption(
            f"Power-based: ⌈{L_total.max():.2f} kW ÷ "
            f"{phys.P_CC/max(phys.charger_efficiency,1e-9):.2f} kW/charger⌉ "
            f"= **{res['N_chargers']}**.  Slot-occupancy (peak concurrent sessions): "
            f"**{N_chg_conc}**. The concurrent count is the physically realistic "
            "one when chargers in their CV taper run below peak power."
        )

    elif step == STEPS[7]:
        c1, c2, c3 = st.columns(3)
        c1.metric("χ (coincidence)", f"{idx['chi']:.3f}")
        c2.metric("ψ (range adequacy)",
                  f"{idx['psi']:.2f}" if np.isfinite(idx['psi']) else "∞")
        c3.metric("ρ_R (peak/avg)", f"{idx['rho_R']:.2f}")
        st.latex(r"\chi = \frac{\langle m_\text{fleet}, \Lambda \rangle}{\|m_\text{fleet}\| \cdot \|\Lambda\|}")
        st.caption("Low (≈ 0–0.3) is good — vehicles charge outside riding peaks.")
        st.latex(r"\psi = \frac{n_\text{bat} \cdot C \cdot (s_\text{full} - \bar s_F) \cdot 1000 / \bar e}{\bar D}")
        st.caption("≥ 1.5 is comfortable; < 1 = a typical day can't be done on one charge.")
        st.latex(r"\rho_R = \frac{\max_t L(t)}{\overline{L(t)}}")
        st.caption("Near 1 is good; ≥ 4 means short, sharp peaks.")
        m_profile = fleet_demand_profile(trips_used, n_hours=24)
        fig, ax = plt.subplots(figsize=(8, 2.8))
        ax2 = ax.twinx()
        ax.bar(np.arange(24), m_profile, color="tab:gray", alpha=0.4)
        ax2.plot(np.arange(24), Lambda, color="tab:blue", marker="o")
        ax.set_xlabel("Hour of day"); ax.set_ylabel("km/vehicle-day", color="gray")
        ax2.set_ylabel("charge starts/h", color="tab:blue")
        ax.set_title(f"Ride demand vs. charging — χ = {idx['chi']:.3f}")
        st.pyplot(fig)

    else:  # Summary
        st.markdown(f"**Scenario:** {name}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Vehicles", res["N_bikes"])
        c2.metric("Chargers",
                  res.get("N_chargers_concurrent", res["N_chargers"]))
        c3.metric("Batteries",
                  res.get("N_batteries_total_concurrent",
                          res["N_batteries_total"]))
        c4.metric("Peak kW", f"{L_total.max():.2f}")
        c1, c2, c3 = st.columns(3)
        c1.metric("χ", f"{idx['chi']:.3f}")
        c2.metric("ψ", f"{idx['psi']:.2f}" if np.isfinite(idx['psi']) else "∞")
        c3.metric("ρ_R", f"{idx['rho_R']:.2f}")
        fig, ax = plt.subplots(figsize=(8, 3.0))
        ax.plot(t_axis, L_total, color="black", lw=1.8)
        ax.fill_between(t_axis, L_total, alpha=0.2, color="black")
        ax.set_xlabel("Hour of day"); ax.set_ylabel("Grid-side fleet load (kW)")
        ax.set_title("Final grid load curve L(t)")
        st.pyplot(fig)


# ---------- Dispatch ----------

if ss.results:
    st.divider()
    view = st.radio(
        "View results as",
        ["Results dashboard", "Step-by-step walkthrough"],
        horizontal=True, key="view_mode",
        help=(
            "**Dashboard** overlays the final answers for all chosen scenarios.\n\n"
            "**Walkthrough** breaks the pipeline into 9 explained stages for a "
            "single scenario."
        ),
    )
    if view == "Results dashboard":
        render_dashboard(ss.results)
    else:
        render_walkthrough(ss.results)
