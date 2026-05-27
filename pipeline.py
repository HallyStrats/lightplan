"""Lightweight planning pipeline — implementation of math_pipeline.md."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


class InfeasibleTripError(RuntimeError):
    """Raised when no swap policy can keep SoC ≥ 0 for a given trip — typically
    a single trip whose energy demand exceeds a full battery."""


@dataclass
class PhysicalParams:
    C: float = 3.24
    s_full: float = 0.99
    n_bat: int = 1
    P_CC: float = 1.62
    s_cc_cv: float = 0.80  # reinterpreted as CC-phase energy fraction
    tau_CV: float = 0.4    # unused (kept for backward compat); λ now solved
    charger_efficiency: float = 1.0
    cv_to_cc_ratio: float = 1.5  # default = HallyStrats @ cc_fraction=0.80


@dataclass
class DistParams:
    mu: float
    sigma: float
    samples: Optional[np.ndarray] = None

    def draw(self, rng: np.random.Generator) -> float:
        if self.samples is not None and len(self.samples) > 0:
            return float(rng.choice(self.samples))
        return float(rng.normal(self.mu, self.sigma))


@dataclass
class Config:
    N: Optional[int] = None
    daytime_swap: bool = True
    overnight_charge: bool = False
    carryover: bool = True
    m_max: float = 60.0
    utilisation: float = 0.8
    n_bins: int = 1440
    seed: int = 0


def compute_auto_N(trips: pd.DataFrame, m_max: float, utilisation: float) -> int:
    n_days = max(trips["day"].nunique(), 1)
    hour = np.floor(trips["t_start"].to_numpy()).astype(int) % 24
    demand_per_hour = np.zeros(24)
    for h, d in zip(hour, trips["d"].to_numpy()):
        demand_per_hour[h] += d
    M_peak = demand_per_hour.max() / n_days
    return max(int(np.ceil(M_peak / (m_max * utilisation))), 1)


def rescale_trips(trips: pd.DataFrame, N: int, N_data: int) -> pd.DataFrame:
    if N == N_data:
        return trips
    out = trips.copy()
    out["d"] = out["d"] * (N_data / N)
    return out


def trip_walk(trips, Fe, Fs, phys, cfg, initial_soc=None,
              return_end_soc=False, seed_offset=0):
    """`initial_soc`: optional dict {bike_id: starting_soc}. Bikes not in
    the dict (or when the arg is None) default to `phys.s_full`. Used by
    the burn-in / steady-state initialisation in `run_pipeline`.

    `return_end_soc`: if True, also returns a dict of each bike's SoC after
    its last trip (before any end-of-sim settle swap). Used to seed a second
    pass with steady-state initial conditions.

    `seed_offset`: added to `cfg.seed` to decouple the warm-up pass from the
    actual pass (so the RNG sequence isn't identical between them)."""
    rng = np.random.default_rng(cfg.seed + int(seed_offset))
    swap_t, swap_s, home_t, home_s = [], [], [], []
    end_soc: dict = {}
    init = initial_soc if initial_soc is not None else {}
    cap_wh = phys.n_bat * phys.C * 1000.0
    trips_sorted = trips.sort_values(["bike_id", "day", "t_start"])

    clamp_count = [0]

    def _step(soc, target, e, d, t_start):
        soc_after = soc - (e * d) / cap_wh
        # Look-ahead swap: fire if the next trip would dip below the rider's
        # target SoC, OR below zero outright (covers cases where target was
        # drawn low/negative). Recorded swap SoC is the pre-trip SoC.
        threshold = max(target, 0.0)
        if cfg.daytime_swap and soc_after < threshold:
            swap_t.append(t_start); swap_s.append(soc)
            soc = phys.s_full
            target = Fs.draw(rng)
            soc_after = soc - (e * d) / cap_wh
        if soc_after < 0:
            # A single trip exceeded a full battery — happens occasionally when
            # the Fe draw lands in the upper tail. Clamp at zero and count it
            # so the UI can surface the rate.
            clamp_count[0] += 1
            soc_after = 0.0
        return soc_after, target

    if cfg.carryover:
        for bike_id, bike_trips in trips_sorted.groupby("bike_id", sort=False):
            soc = float(init.get(bike_id, phys.s_full))
            target = Fs.draw(rng)
            last_t_end = None
            for _, trip in bike_trips.iterrows():
                t_start = trip["day"] * 24.0 + trip["t_start"]
                soc, target = _step(soc, target, Fe.draw(rng), trip["d"], t_start)
                last_t_end = t_start + float(trip["dt"])
            end_soc[bike_id] = soc
            # End-of-simulation settle swap: every bike started at s_full, so
            # it must finish at s_full too. Record the residual partial charge
            # as one final swap so total grid energy = total trip energy.
            if cfg.daytime_swap and last_t_end is not None and soc < phys.s_full:
                swap_t.append(last_t_end)
                swap_s.append(soc)
    else:
        for (bike_id, day), bd_trips in trips_sorted.groupby(
            ["bike_id", "day"], sort=False
        ):
            soc = phys.s_full
            target = Fs.draw(rng)
            last_t_end = None
            day_offset = float(day) * 24.0
            for _, trip in bd_trips.iterrows():
                t_start = day_offset + float(trip["t_start"])
                t_end = t_start + float(trip["dt"])
                soc, target = _step(soc, target, Fe.draw(rng), trip["d"], t_start)
                last_t_end = t_end
            if cfg.overnight_charge and last_t_end is not None:
                home_t.append(last_t_end); home_s.append(soc)

    result = (
        np.asarray(swap_t),
        np.asarray(swap_s),
        np.asarray(home_t),
        np.asarray(home_s),
        int(clamp_count[0]),
    )
    if return_end_soc:
        return (*result, end_soc)
    return result


def cc_cv_kernel(s: float, phys: PhysicalParams, dt_h: float, n_steps: int) -> np.ndarray:
    """CC-CV **battery-side** power profile for a SINGLE battery on one charger.

    The kernel describes what flows INTO the battery — no charger efficiency
    here. `grid_load` divides the accumulated fleet load by η once at the end
    to convert to wall-plug power.

    Implementation follows the HallyStrats/li-ion_charging_profile model:
    CC phase delivers a fraction `s_cc_cv` (= CC_FRACTION) of the target
    battery energy at constant power `P_CC`, then CV phase decays
    exponentially with decay constant λ found by binary search so total
    battery-side energy matches (s_full − s) · C to within 0.005 kWh.
    """
    out = np.zeros(n_steps)
    if s >= phys.s_full:
        return out

    cc_fraction = float(np.clip(phys.s_cc_cv, 0.05, 0.99))
    energy_batt = (phys.s_full - s) * phys.C  # kWh INTO battery
    if energy_batt <= 0 or phys.P_CC <= 0:
        return out

    cc_energy = energy_batt * cc_fraction
    cc_time_h = cc_energy / phys.P_CC
    if cc_time_h <= 0:
        return out

    # CV-to-CC time ratio: free parameter (default 1.5 = HallyStrats @ cc=0.80)
    cc_cv_ratio = float(getattr(phys, "cv_to_cc_ratio", 1.5))
    cv_time_h = max(cc_cv_ratio, 0.05) * cc_time_h
    total_time_h = cc_time_h + cv_time_h
    taus = np.arange(n_steps) * dt_h

    # Binary-search for λ so total battery-side energy matches target
    lam_lo, lam_hi = 0.1, 20.0
    lam = (lam_lo + lam_hi) / 2.0
    target = energy_batt
    tol = 0.005
    for _ in range(100):
        in_cv = (taus >= cc_time_h) & (taus <= total_time_h)
        k = np.where(taus < cc_time_h, phys.P_CC, 0.0)
        k[in_cv] = phys.P_CC * np.exp(-lam * (taus[in_cv] - cc_time_h))
        total = float(np.sum(k) * dt_h)
        diff = total - target
        if abs(diff) <= tol:
            break
        if diff < 0:
            lam_hi = lam
        else:
            lam_lo = lam
        lam = (lam_lo + lam_hi) / 2.0

    in_cv = (taus >= cc_time_h) & (taus <= total_time_h)
    out[taus < cc_time_h] = phys.P_CC
    out[in_cv] = phys.P_CC * np.exp(-lam * (taus[in_cv] - cc_time_h))
    return out


def charge_start_intensity(swap_t, home_t, N, N_data, n_days, n_hours=24):
    """Charge-start-event intensity Λ(t): swaps + home plug-ins per fleet-hour."""
    all_t = np.concatenate([np.asarray(swap_t), np.asarray(home_t)])
    if len(all_t) == 0:
        return np.zeros(n_hours)
    hours = np.floor(np.mod(all_t, 24.0)).astype(int) % n_hours
    counts = np.bincount(hours, minlength=n_hours).astype(float)
    return (N / (N_data * n_days)) * counts


def resample_kernel(t_h, P_kW, n_bins=1440, horizon_h=24.0):
    """Resample a user-provided (time, power) charging curve onto an n_bins
    grid covering [0, horizon_h] hours. Beyond the supplied time range the
    power is treated as zero (charging finished). Returns a length-n_bins
    array of kW values."""
    t_h = np.asarray(t_h, dtype=float)
    P_kW = np.asarray(P_kW, dtype=float)
    order = np.argsort(t_h)
    t_h, P_kW = t_h[order], P_kW[order]
    grid = np.arange(n_bins) * (horizon_h / n_bins)
    k = np.interp(grid, t_h, P_kW, left=P_kW[0] if len(P_kW) else 0.0, right=0.0)
    k[grid > t_h.max()] = 0.0
    return k


def _partial_user_kernel(uk, s, phys, dt_h):
    """Treat the uploaded kernel as a canonical full-charge **battery-side**
    profile (SoC 0 → s_full delivering s_full · C kWh into the battery).
    For a session starting at SoC s, return the tail portion of the kernel
    whose cumulative battery energy equals (s_full − s) · C. The head of
    the curve (already delivered before arrival) is skipped."""
    uk = np.asarray(uk, dtype=float)
    n = len(uk)
    cum = np.cumsum(uk) * dt_h
    E_full = float(cum[-1]) if n else 0.0
    if E_full <= 0 or s >= phys.s_full:
        return np.zeros(n)
    frac = max(0.0, min(s / max(phys.s_full, 1e-9), 1.0))
    skip_energy = E_full * frac
    idx_start = int(np.searchsorted(cum, skip_energy))
    if idx_start >= n:
        return np.zeros(n)
    out = np.zeros(n)
    tail = uk[idx_start:]
    out[: len(tail)] = tail
    return out


def grid_load(swap_t, swap_s, home_t, home_s, N, N_data, n_days, phys,
              n_bins=1440, user_kernel=None):
    """Build the fleet-wide grid-side load curve L(t) over a 24-h horizon.

    All kernels (parametric and uploaded) are **battery-side** kW per single
    session. The function accumulates per-event kernels into a battery-side
    fleet load and then divides by `phys.charger_efficiency` once at the end
    so the returned arrays are wall-plug power.

    If `user_kernel` is provided (1-D numpy array, length n_bins, battery-side
    kW per single-battery session for a canonical full charge from SoC 0 →
    s_full), it overrides the parametric CC-CV kernel. At each event the
    tail starting from cumulative energy = s · C is used so total battery
    energy delivered = (s_full − s) · C, matching the parametric kernel's
    per-SoC scaling. n_bat multiplication still applies (one parallel
    session per on-vehicle battery)."""
    dt_h = 24.0 / n_bins
    t_axis = np.arange(n_bins) * dt_h
    scale = N / (N_data * n_days)

    def accumulate(t_events, s_events):
        L = np.zeros(n_bins)
        if len(t_events) == 0:
            return L
        start_bins = (np.mod(t_events, 24.0) / dt_h).astype(int) % n_bins
        for sb, s in zip(start_bins, s_events):
            # n_bat parallel batteries → n_bat parallel charging sessions
            if user_kernel is not None:
                k = _partial_user_kernel(user_kernel, float(s), phys, dt_h) * phys.n_bat
            else:
                k = cc_cv_kernel(float(s), phys, dt_h, n_bins) * phys.n_bat
            end = sb + n_bins
            if end <= n_bins:
                L[sb:end] += k
            else:
                first = n_bins - sb
                L[sb:] += k[:first]
                L[: end - n_bins] += k[first:]
        return L * scale

    # Internal accumulate gives battery-side fleet load. Divide by η once at
    # the end to convert into the grid-side (wall-plug) load reported to the
    # rest of the pipeline. Energy from grid = energy to battery / η.
    eta = max(getattr(phys, "charger_efficiency", 1.0), 1e-9)
    L_swap = accumulate(swap_t, swap_s) / eta
    L_home = accumulate(home_t, home_s) / eta
    return t_axis, L_swap, L_home, L_swap + L_home


def resources(L, N, phys):
    # L is grid-side; each charger draws P_grid = P_CC / η from the grid at peak.
    eta = max(getattr(phys, "charger_efficiency", 1.0), 1e-9)
    P_grid = phys.P_CC / eta
    N_chg = int(np.ceil(L.max() / P_grid)) if L.max() > 0 else 0
    return {"N_bikes": N, "N_chargers": N_chg, "N_batteries_total": N * phys.n_bat + N_chg}


def daily_max_load(swap_t, swap_s, home_t, home_s, N, N_data, n_days, phys,
                   n_bins=1440, user_kernel=None):
    """Per-hour-of-day **maximum** grid-side load envelope across all data days.

    The standard `grid_load` output is a per-day mean (events from every day
    are folded into one cyclic axis and divided by n_days). This function
    instead builds a (n_days × n_bins) matrix where each row is one day's
    fleet load, then takes max over rows to give the *worst-day* envelope
    at each minute-of-day.

    Returns (L_swap_max, L_home_max, L_total_max), grid-side."""
    dt_h = 24.0 / n_bins
    eta = max(getattr(phys, "charger_efficiency", 1.0), 1e-9)
    scale = N / max(N_data, 1)  # per fleet, no /n_days

    def _matrix(t_events, s_events):
        M = np.zeros((n_days + 1, n_bins))
        for t_ev, s_ev in zip(t_events, s_events):
            if user_kernel is not None:
                k = _partial_user_kernel(user_kernel, float(s_ev), phys, dt_h) * phys.n_bat
            else:
                k = cc_cv_kernel(float(s_ev), phys, dt_h, n_bins) * phys.n_bat
            day_idx = int(float(t_ev) / 24.0)
            if day_idx < 0 or day_idx >= n_days:
                continue
            sb = int((float(t_ev) % 24.0) / dt_h) % n_bins
            end = sb + n_bins
            if end <= n_bins:
                M[day_idx, sb:end] += k
            else:
                first = n_bins - sb
                M[day_idx, sb:] += k[:first]
                if day_idx + 1 < n_days + 1:
                    M[day_idx + 1, :end - n_bins] += k[first:]
        return M[:n_days] * (scale / eta)

    M_swap = _matrix(swap_t, swap_s)
    M_home = _matrix(home_t, home_s)
    M_total = M_swap + M_home
    if n_days <= 0 or M_total.size == 0:
        empty = np.zeros(n_bins)
        return empty, empty, empty
    return M_swap.max(axis=0), M_home.max(axis=0), M_total.max(axis=0)


def concurrent_session_peak(swap_t, swap_s, home_t, home_s, N, N_data, n_days,
                            phys, n_bins=1440, user_kernel=None,
                            power_threshold_frac=0.01):
    """Peak number of **simultaneous charging sessions** (physical charger
    slots in use) across the full **unfolded multi-day simulation**.

    Builds an `(n_days + 1) × n_bins` occupancy array on the absolute
    timeline (one extra day reserved for sessions whose tails spill past
    the last data day). For each event, the active region of its kernel
    (where instantaneous battery-side power exceeds
    `power_threshold_frac` of its own peak) is added at the event's
    absolute start bin and counted `n_bat` times (parallel on-vehicle
    batteries). The global max across the unfolded timeline is returned.

    Sizing infrastructure on the unfolded peak captures the worst day's
    spike, which the cyclic per-day-average view would smooth out."""
    dt_h = 24.0 / n_bins
    scale = N / max(N_data, 1)  # unfolded: no /n_days averaging
    total_bins = (max(n_days, 1) + 1) * n_bins
    occupancy = np.zeros(total_bins)

    pairs = list(zip(list(swap_t), list(swap_s))) + list(zip(list(home_t), list(home_s)))
    if not pairs:
        return 0.0

    for t_ev, s_ev in pairs:
        if user_kernel is not None:
            k = _partial_user_kernel(user_kernel, float(s_ev), phys, dt_h)
        else:
            k = cc_cv_kernel(float(s_ev), phys, dt_h, n_bins)
        peak_k = float(k.max())
        if peak_k <= 0:
            continue
        active = (k > peak_k * power_threshold_frac).astype(float) * phys.n_bat
        sb = int(float(t_ev) / dt_h)
        if sb < 0 or sb >= total_bins:
            continue
        end = min(sb + n_bins, total_bins)
        occupancy[sb:end] += active[: end - sb]

    occupancy *= scale
    return float(occupancy.max())


def fleet_demand_profile(trips, n_hours=24):
    n_days = max(trips["day"].nunique(), 1)
    hour = np.floor(trips["t_start"].to_numpy()).astype(int) % n_hours
    m = np.zeros(n_hours)
    for h, d in zip(hour, trips["d"].to_numpy()):
        m[h] += d
    return m / n_days


def diagnostic_indices(trips, Lambda, L, Fe_mean, Fs_mean, phys):
    m_fleet = fleet_demand_profile(trips, n_hours=len(Lambda))
    num = float(np.dot(m_fleet, Lambda))
    den = float(np.linalg.norm(m_fleet) * np.linalg.norm(Lambda))
    chi = num / den if den > 0 else 0.0

    n_days = max(trips["day"].nunique(), 1)
    n_bikes_data = max(trips["bike_id"].nunique(), 1)
    D_bar = trips["d"].sum() / (n_bikes_data * n_days)
    range_km = phys.n_bat * phys.C * (phys.s_full - Fs_mean) * 1000.0 / max(Fe_mean, 1e-6)
    psi = range_km / D_bar if D_bar > 0 else float("inf")
    rho = float(L.max() / L.mean()) if L.mean() > 0 else 0.0
    return {"chi": chi, "psi": psi, "rho_R": rho}


def run_pipeline_from_events(events_df, phys, cfg, n_days=1, user_kernel=None):
    """Skip the trip walk: take user-supplied charge-start events (t, s) directly
    and run Λ → L → resources. Useful when the planner already has empirical
    arrival times and pre-charge SoC distributions and wants to bypass the
    upstream mobility model."""
    t = np.asarray(events_df["t"].to_numpy(), dtype=float)
    s = np.asarray(events_df["s"].to_numpy(), dtype=float)
    if np.nanmax(s) > 1.5:
        s = s / 100.0
    N = int(cfg.N) if cfg.N else max(1, int(np.ceil(len(t) / max(n_days, 1))))
    N_data = N
    home_t = np.array([]); home_s = np.array([])
    Lambda = charge_start_intensity(t, home_t, N, N_data, n_days)
    t_axis, L_swap, L_home, L_total = grid_load(
        t, s, home_t, home_s, N, N_data, n_days, phys, cfg.n_bins,
        user_kernel=user_kernel,
    )
    res = resources(L_total, N, phys)
    peak_conc = concurrent_session_peak(t, s, home_t, home_s, N, N_data,
                                        n_days, phys, cfg.n_bins,
                                        user_kernel=user_kernel)
    res["N_chargers_concurrent"] = int(np.ceil(peak_conc))
    res["N_batteries_concurrent"] = int(np.ceil(peak_conc))
    res["N_batteries_total_concurrent"] = N * phys.n_bat + int(np.ceil(peak_conc))
    L_swap_max, L_home_max, L_total_max = daily_max_load(
        t, s, home_t, home_s, N, N_data, n_days, phys, cfg.n_bins,
        user_kernel=user_kernel,
    )
    rho = float(L_total.max() / L_total.mean()) if L_total.mean() > 0 else 0.0
    return {
        "N": N, "N_data": N_data, "n_days": n_days, "trips_used": None,
        "swap_t": t, "swap_s": s, "home_t": home_t, "home_s": home_s,
        "Lambda": Lambda, "t_axis": t_axis,
        "L_swap": L_swap, "L_home": L_home, "L_total": L_total,
        "L_swap_max": L_swap_max, "L_home_max": L_home_max,
        "L_total_max": L_total_max,
        "resources": res, "indices": {"chi": None, "psi": None, "rho_R": rho},
        "soc_clamp_events": 0,
    }


def min_feasible_N(trips_df, Fe_mean, phys, cfg):
    """Minimum fleet size such that every trip can be energy-served.

    Returns a dict with:
      - 'min_N': required fleet size (int), or None if energy is not the binding
        constraint (any scenario with daytime_swap=True).
      - 'range_km': effective per-charge range with mean efficiency.
      - 'max_trip_km': longest single trip in the (unrescaled) data.
      - 'total_km_per_day': fleet-wide km/day (rescaling-invariant).
      - 'single_trip_infeasible': True if even one trip exceeds one full charge.
    """
    cap_kwh = phys.n_bat * phys.C * phys.s_full
    range_km = cap_kwh * 1000.0 / max(Fe_mean, 1e-6)
    n_days = max(trips_df["day"].nunique(), 1)
    total_km_per_day = float(trips_df["d"].sum()) / n_days
    max_trip = float(trips_df["d"].max())
    N_data = max(trips_df["bike_id"].nunique(), 1)

    if cfg.daytime_swap:
        # Daytime swaps available — energy per bike-day is unbounded.
        return {
            "min_N": None, "range_km": range_km, "max_trip_km": max_trip,
            "total_km_per_day": total_km_per_day,
            "single_trip_infeasible": max_trip > range_km,
        }

    # Home-only: each (bike, day) must fit within one charge.
    avg_floor = int(np.ceil(total_km_per_day / range_km))
    single_floor = int(np.ceil(N_data * max_trip / range_km))
    min_N = max(avg_floor, single_floor)
    return {
        "min_N": min_N, "range_km": range_km, "max_trip_km": max_trip,
        "total_km_per_day": total_km_per_day,
        "single_trip_infeasible": max_trip > range_km,
    }


def synth_trips(n_bikes, n_days, trips_per_day_mean, mean_km_per_trip, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_bikes):
        for d in range(n_days):
            k = max(1, int(rng.poisson(trips_per_day_mean)))
            am = rng.uniform(size=k) < 0.5
            t_starts = np.where(am, rng.normal(8.0, 1.5, k), rng.normal(17.0, 2.0, k))
            t_starts = np.clip(np.sort(t_starts), 0.0, 23.5)
            dists = np.clip(rng.normal(mean_km_per_trip, mean_km_per_trip * 0.4, k), 0.5, None)
            dts = dists / 20.0
            for ts, dt, dd in zip(t_starts, dts, dists):
                rows.append({"bike_id": b, "day": d, "t_start": float(ts),
                             "dt": float(dt), "d": float(dd)})
    return pd.DataFrame(rows)


def run_pipeline(trips, Fe, Fs, phys, cfg, user_kernel=None):
    N_data = max(trips["bike_id"].nunique(), 1)
    n_days = max(trips["day"].nunique(), 1)
    N = compute_auto_N(trips, cfg.m_max, cfg.utilisation) if cfg.N is None else int(cfg.N)
    trips_used = rescale_trips(trips, N, N_data)

    swap_t, swap_s, home_t, home_s, clamp_n = trip_walk(
        trips_used, Fe, Fs, phys, cfg,
    )
    Lambda = charge_start_intensity(swap_t, home_t, N, N_data, n_days)
    t_axis, L_swap, L_home, L_total = grid_load(
        swap_t, swap_s, home_t, home_s, N, N_data, n_days, phys, cfg.n_bins,
        user_kernel=user_kernel,
    )
    res = resources(L_total, N, phys)
    peak_conc = concurrent_session_peak(swap_t, swap_s, home_t, home_s,
                                        N, N_data, n_days, phys, cfg.n_bins,
                                        user_kernel=user_kernel)
    res["N_chargers_concurrent"] = int(np.ceil(peak_conc))
    res["N_batteries_concurrent"] = int(np.ceil(peak_conc))
    res["N_batteries_total_concurrent"] = N * phys.n_bat + int(np.ceil(peak_conc))
    L_swap_max, L_home_max, L_total_max = daily_max_load(
        swap_t, swap_s, home_t, home_s, N, N_data, n_days, phys, cfg.n_bins,
        user_kernel=user_kernel,
    )
    Fe_mean = float(np.mean(Fe.samples)) if Fe.samples is not None and len(Fe.samples) else Fe.mu
    Fs_mean = float(np.mean(Fs.samples)) if Fs.samples is not None and len(Fs.samples) else Fs.mu
    idx = diagnostic_indices(trips_used, Lambda, L_total, Fe_mean, Fs_mean, phys)

    return {
        "N": N, "N_data": N_data, "n_days": n_days, "trips_used": trips_used,
        "swap_t": swap_t, "swap_s": swap_s, "home_t": home_t, "home_s": home_s,
        "Lambda": Lambda, "t_axis": t_axis,
        "L_swap": L_swap, "L_home": L_home, "L_total": L_total,
        "L_swap_max": L_swap_max, "L_home_max": L_home_max,
        "L_total_max": L_total_max,
        "resources": res, "indices": idx,
        "soc_clamp_events": clamp_n,
    }
