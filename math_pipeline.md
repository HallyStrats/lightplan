# `lightplan` — mathematical pipeline

The model takes three input distributions plus a small set of physical parameters and produces a fleet-wide grid load curve, swap-event times, SoC-at-arrival distribution, and resource counts. The core operation is a deterministic walk through each bike's trip sequence, with stochastic draws from the input distributions at each step.

---

## 1. Inputs

### Data-derived inputs (three distributions plus one density)

| Symbol | Type | Meaning |
|---|---|---|
| $m_b(t)$ | per-bike trip records | sequence of $(t_\text{start}, \Delta t, d)$ triples for each bike-day |
| $F_e$ | distribution over $\mathbb{R}_+$ | energy efficiency in Wh/km |
| $F_s$ | distribution over $[0, 1]$ | SoC at the moment a battery is swapped |
| $h(t)$ | density on $[0, 24)$ | last-trip-end-time density per bike-day |

$F_e$ and $F_s$ are typically parameterised as $\mathcal{N}(\mu, \sigma^2)$ but can be supplied as empirical samples.

### Physical parameters

| Symbol | Value (example) | Meaning |
|---|---:|---|
| $C$ | 3.24 kWh | nominal battery capacity |
| $s_\text{full}$ | 0.99 | upper SoC cutoff |
| $n_\text{bat}$ | 1 or 2 | number of batteries per bike (effective capacity $= n_\text{bat} \cdot C$) |
| $P_\text{CC}$ | 1.62 kW | constant-current charger power |
| $s_\text{cc-cv}$ | 0.80 | SoC at which charging transitions CC $\to$ CV |
| $\tau_\text{CV}$ | 0.4 h | exponential CV-phase time constant |

### Configuration

| Knob | Choices |
|---|---|
| $N$ | integer (fixed) or `'auto'` (computed from peak demand) |
| `daytime_swap` | bool — bikes swap mid-day when SoC crosses target |
| `overnight_charge` | bool — bikes plug in at end of last trip |
| `carryover` | bool — battery state persists across days |
| $m_\text{max}$, `utilisation` | scalars — only used when $N = \text{'auto'}$ |

---

## 2. Fleet size

**Fixed mode.** $N$ is supplied as an integer.

**Auto mode.** Compute peak fleet hourly demand from the trip records:

$$M_\text{peak} = \max_h \sum_{\text{trips in hour } h} d_{\text{trip}} / n_\text{days}$$

Then:

$$N = \left\lceil \frac{M_\text{peak}}{m_\text{max} \cdot \text{utilisation}} \right\rceil$$

If $N \ne N_\text{data}$, rescale every trip distance by the demand-preserving factor:

$$d'_\text{trip} = d_\text{trip} \cdot \frac{N_\text{data}}{N}$$

Each simulated bike now does the proportional share of the same total demand.

---

## 3. The trip walk

The walk is the only non-trivial operation in the pipeline. It produces a sequence of $(t_b, s_b)$ swap events from each bike's trips. There are two modes depending on the `carryover` trigger.

### 3a. Daily-reset mode (`carryover = False`)

For each `(bike, day)`:

```
soc ← s_full
target ← max(0.05, sample from F_s)
for each trip in chronological order:
    e ← max(15.0, sample from F_e)              # Wh/km
    Δsoc = (e · d_trip) / (n_bat · C · 1000)
    soc_after = soc − Δsoc

    if daytime_swap and soc_after < target:
        record swap event at trip end-time t_b with SoC s_b = soc_after
        soc ← s_full
        target ← max(0.05, sample from F_s)
    else:
        soc ← soc_after

if overnight_charge:
    record home plug-in event at last-trip-end-time with SoC s_b = soc
```

### 3b. Carryover mode (`carryover = True`)

Walk the entire multi-day trip sequence for each bike continuously, without resetting `soc` between days. No end-of-day home plug-in is recorded.

### Output of the walk

After processing all bikes, two arrays:

$$\text{swap\_events} = \{(t_b, s_b)\}_{b=1}^{N_\text{swap}}, \quad \text{home\_events} = \{(t_c, s_c)\}_{c=1}^{N_\text{home}}$$

Each event is a charging session that will draw power from the grid starting at $t$ with initial SoC $s$.

---

## 4. From events to swap intensity $\Lambda(t)$

Bin the swap events into hourly buckets on a $n_\text{bins}/24$-hour grid, scale to the predicted fleet:

$$\Lambda(t) = \frac{N}{N_\text{data} \cdot n_\text{days}} \cdot \frac{\text{count of swap events in bin containing } t}{\Delta t}$$

units of swaps/hour fleet-wide.

This $\Lambda(t)$ is the *charge-start-event intensity* — equivalently the **swap-time distribution scaled to fleet size**.

---

## 5. CC-CV charger kernel

The per-session power profile as a function of time since charge start, parameterised by the starting SoC $s$:

$$\tau_1(s) = \max\!\left(0, \frac{(s_\text{cc-cv} - s) \cdot n_\text{bat} \cdot C}{P_\text{CC}}\right) \quad \text{(end of CC phase)}$$

$$\tau_2 = -\tau_\text{CV} \log\!\left(1 - \frac{(s_\text{full} - s_\text{cc-cv}) \cdot n_\text{bat} \cdot C}{P_\text{CC} \cdot \tau_\text{CV}}\right) \quad \text{(CV-phase duration)}$$

$$K(\tau; s) = \begin{cases}
P_\text{CC} & 0 \le \tau < \tau_1(s) \\
P_\text{CC} \cdot e^{-(\tau - \tau_1)/\tau_\text{CV}} & \tau_1(s) \le \tau < \tau_1 + \tau_2 \\
0 & \tau \ge \tau_1 + \tau_2
\end{cases}$$

A swap that arrives at low SoC has a longer CC plateau (more energy delivered); a swap that arrives at high SoC starts in CV almost immediately (less energy).

Total energy delivered per session:

$$E_\text{session}(s) = \int_0^\infty K(\tau; s)\, d\tau = (s_\text{full} - s) \cdot n_\text{bat} \cdot C \quad \text{(kWh)}$$

---

## 6. Grid load

For each charging event $(t_b, s_b)$, place its kernel $K(\cdot; s_b)$ on the time axis starting at $t_b$, wrapping periodically over 24 hours. Sum across all events and scale to fleet:

$$L_\text{swap}(t) = \frac{N}{N_\text{data} \cdot n_\text{days}} \sum_{b \in \text{swap\_events}} K\big((t - t_b) \bmod 24;\, s_b\big)$$

$$L_\text{home}(t) = \frac{N}{N_\text{data} \cdot n_\text{days}} \sum_{c \in \text{home\_events}} K\big((t - t_c) \bmod 24;\, s_c\big)$$

$$\boxed{L(t) = L_\text{swap}(t) + L_\text{home}(t)}$$

Computed as a sum of placed kernels on a $n_\text{bins}$-bin grid (typically $n_\text{bins} = 1440$, 1-minute resolution).

---

## 7. Resources

**Chargers** — sized to cover instantaneous peak load:

$$N_\text{chg} = \lceil \max_t L(t) / P_\text{CC} \rceil$$

**Batteries** — on-bike plus on-charger:

$$N_\text{bat,total} = N \cdot n_\text{bat} + N_\text{chg}$$

**Bikes** — equal to $N$ (fixed input or auto-computed in step 2).

---

## 8. Diagnostic indices

Three dimensionless indices computed from the outputs.

**Coincidence index** — how aligned charging events are with riding peaks:

$$\chi = \frac{\langle m_\text{fleet}, \Lambda \rangle}{\|m_\text{fleet}\| \cdot \|\Lambda\|} \in [0, 1]$$

**Range adequacy** — how the typical effective range compares to daily distance:

$$\psi = \frac{n_\text{bat} \cdot C \cdot (s_\text{full} - \bar s_F) \cdot 1000 / \bar e}{\bar D}$$

where $\bar e$, $\bar s_F$ are the means of $F_e$ and $F_s$, and $\bar D$ is mean daily km per bike.

**Resource inflation** — peak-to-average load ratio:

$$\rho_R = \frac{\max_t L(t)}{\overline{L(t)}}$$

---

## 9. End-to-end summary

```
Inputs ─────────┐
  trips         │
  F_e, F_s      │
  h(t)          │
  C, charger    │
  triggers      │
  N or 'auto'   │
                ▼
          (auto N → rescale trips)
                │
                ▼
          Trip walk: per bike-day or per bike (carryover)
            for each trip:
              sample e ~ F_e, sample target ~ F_s
              decrement SoC; check threshold
              record swap event (t_b, s_b) or end-of-day plug-in
                │
                ▼
          Λ(t) = histogram of swap times, scaled by N
                │
                ▼
          L_swap(t) = Σ K(τ; s_b) shifted to each t_b
          L_home(t) = Σ K(τ; s_c) shifted to each t_c
                │
                ▼
          L(t) = L_swap + L_home
                │
                ▼
          N_chg = ⌈max L / P_CC⌉
          N_bat = N · n_bat + N_chg
          χ, ψ, ρ_R
```

---

## 10. Scenario configurations

| Scenario | `daytime_swap` | `overnight_charge` | `carryover` | $n_\text{bat}$ |
|---|:-:|:-:|:-:|:-:|
| Swap-only with continuous operation | ✓ | ✗ | ✓ | 1 |
| Swap + home (daily reset) | ✓ | ✓ | ✗ | 1 |
| Home only with sufficient range | ✗ | ✓ | ✗ | 2 |

Other combinations are valid (e.g. home-only with single battery for short-range fleets) but the three above correspond to the canonical paper scenarios.

---

## 11. Three operating regimes

When sweeping $N$ on a fixed-demand problem, the model exhibits three qualitative regimes that emerge from the physics without parameter changes.

| Regime | Fleet size | Per-bike km/day | Behaviour | Peak time |
|---|---|---|---|---|
| Demand-rich | $N \gg N_\text{auto}$ | $\ll$ range | bikes underutilised, no daytime swaps | late evening |
| Balanced | $N \approx N_\text{auto} / \text{utilisation}$ | $\approx$ range | ~1 swap/bike/day | early evening |
| Demand-stressed | $N \le N_\text{auto}$ | $>$ range | every bike swaps mid-day, sometimes twice | afternoon |
