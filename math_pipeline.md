# `lightplan` — mathematical pipeline

The model takes three input distributions plus a small set of physical parameters and produces a fleet-wide grid load curve, swap/home charging events, an SoC-at-arrival sample, and resource counts. The core operation is a deterministic walk through each bike's trip sequence, with stochastic draws from the input distributions at each step.

All charging kernels in this pipeline are **battery-side** (power flowing *into* the battery). Charger efficiency $\eta$ is applied exactly once, when the fleet-wide battery-side load is converted to a grid-side (wall-plug) load in §6.

---

## 1. Inputs

### Data-derived inputs

| Symbol | Type | Meaning |
|---|---|---|
| $m_b(t)$ | per-bike trip records | sequence of $(t_\text{start}, \Delta t, d)$ triples for each bike-day |
| $F_e$ | distribution over $\mathbb{R}_+$ | energy efficiency in Wh/km |
| $F_s$ | distribution over $[0, 1]$ | rider target SoC threshold for triggering a swap |

$F_e$ and $F_s$ are typically parameterised as $\mathcal{N}(\mu, \sigma^2)$ but can be supplied as empirical samples.

### Physical parameters

| Symbol | Field | Default | Meaning |
|---|---|---:|---|
| $C$ | `C` | 3.24 kWh | nominal battery capacity |
| $s_\text{full}$ | `s_full` | 0.99 | upper SoC cutoff |
| $n_\text{bat}$ | `n_bat` | 1 or 2 | batteries per bike (parallel sessions per event) |
| $P_\text{CC}$ | `P_CC` | 1.62 kW | constant-current battery-side charger power |
| $f_\text{CC}$ | `s_cc_cv` | 0.80 | CC-phase **energy fraction** (HallyStrats `CC_FRACTION`) |
| $r_\text{CV/CC}$ | `cv_to_cc_ratio` | 1.5 | CV-to-CC time ratio |
| $\eta$ | `charger_efficiency` | 1.0 | wall-plug → battery efficiency |

Notes:

- `s_cc_cv` is misnamed for historical reasons — it is the **energy fraction** delivered in CC, not an SoC threshold.
- `tau_CV` exists on the dataclass for backward compatibility but is **unused**; the CV-phase decay constant is solved numerically (§5).

### Configuration

| Knob | Choices |
|---|---|
| $N$ | integer (fixed) or `'auto'` (computed from peak demand) |
| `daytime_swap` | bool — record a swap when the next trip would push SoC below the target |
| `overnight_charge` | bool — record a home plug-in at the end of the last trip (daily-reset mode only) |
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

The walk produces two event arrays: $(t_b, s_b)$ swap events and $(t_c, s_c)$ home plug-in events. Time stamps are stored as **absolute hours** on the multi-day timeline (so day $d$, local time $t$ becomes $24 d + t$); the $\bmod\, 24$ fold to a daily axis only happens later in §4 and §6.

### Per-trip update (shared between modes)

For each trip:

```
e ← sample from F_e          (Wh/km, drawn fresh each trip)
Δsoc = (e · d) / (n_bat · C · 1000)
soc_after = soc − Δsoc

# Look-ahead swap: if the upcoming trip would dip below threshold,
# swap NOW (recorded at the trip's start time and at the PRE-trip SoC).
threshold = max(target, 0)
if daytime_swap and soc_after < threshold:
    record swap event (t_start, soc)         # pre-trip SoC, trip start time
    soc ← s_full
    target ← sample from F_s
    soc_after = soc − Δsoc                   # re-apply with full battery

# SoC clamp + counter: occasional Fe upper-tail draws can still exceed
# a full battery on a single trip. Clamp to 0 and count the event.
if soc_after < 0:
    clamp_count += 1
    soc_after = 0

soc ← soc_after
```

Key points that differ from a naive implementation:

- Swap is **fired before** the trip when the next trip would breach the threshold, not after. The recorded $(t_b, s_b)$ uses the **trip-start time** $t_\text{start}$ and the **pre-trip SoC**.
- $\Delta\text{soc}$ uses the **on-vehicle** capacity $n_\text{bat} \cdot C$ (so a 2-battery bike has half the per-km SoC drop of a 1-battery bike).
- A `clamp_count` is incremented whenever a single trip would otherwise drive SoC below zero; the counter is surfaced as `soc_clamp_events`.

### 3a. Carryover mode (`carryover = True`)

For each bike, walk the entire multi-day trip sequence continuously. SoC and `target` persist across days. No home plug-ins are recorded.

**End-of-simulation settle swap.** After the last trip, if `daytime_swap` is on and the final SoC is below $s_\text{full}$, record one final swap event $(t_\text{last\_end}, \text{soc})$. Every bike begins at $s_\text{full}$ and must end at $s_\text{full}$, so this closes the energy balance: total grid energy delivered = total trip energy demanded.

### 3b. Daily-reset mode (`carryover = False`)

For each `(bike, day)`:

- `soc ← s_full`, `target ← sample from F_s`.
- Walk the day's trips with the per-trip update above.
- If `overnight_charge` is on, record a home plug-in $(t_\text{last\_end}, \text{soc})$ at the end of the last trip.

### Output of the walk

Two event arrays plus a clamp counter:

$$\text{swap\_events} = \{(t_b, s_b)\}, \quad \text{home\_events} = \{(t_c, s_c)\}, \quad \text{clamp\_count} \in \mathbb{N}.$$

---

## 4. Charge-start event intensity $\Lambda(t)$

All charge-start events (swaps **and** home plug-ins) are folded mod-24 into hourly bins and scaled to the predicted fleet:

$$\Lambda(t) = \frac{N}{N_\text{data} \cdot n_\text{days}} \cdot \frac{\big|\{e \in \text{swap} \cup \text{home} : (t_e \bmod 24) \in \text{bin}(t)\}\big|}{\Delta t}$$

units of charge-starts/hour fleet-wide.

---

## 5. Charging kernel $K(\tau; s)$

The per-session **battery-side** power profile as a function of time since charge start, parameterised by the starting SoC $s$.

### Parametric CC-CV kernel

Let the target battery-side energy be

$$E(s) = (s_\text{full} - s) \cdot C \quad \text{(kWh into battery)}.$$

The CC-phase energy fraction is $f_\text{CC}$ (`s_cc_cv` field; default 0.80), so

$$\tau_\text{CC}(s) = \frac{f_\text{CC} \cdot E(s)}{P_\text{CC}}, \qquad \tau_\text{CV}(s) = r_\text{CV/CC} \cdot \tau_\text{CC}(s), \qquad \tau_\text{end}(s) = \tau_\text{CC} + \tau_\text{CV}.$$

The CV decay constant $\lambda$ is **binary-searched** so the total kernel energy matches $E(s)$ to within 0.005 kWh:

$$K(\tau; s) = \begin{cases}
P_\text{CC} & 0 \le \tau < \tau_\text{CC}(s) \\
P_\text{CC} \cdot e^{-\lambda(s) \cdot (\tau - \tau_\text{CC}(s))} & \tau_\text{CC}(s) \le \tau \le \tau_\text{end}(s) \\
0 & \tau > \tau_\text{end}(s)
\end{cases}$$

By construction $\int_0^\infty K(\tau; s)\, d\tau = E(s)$ exactly (battery-side).

This follows the [HallyStrats/li-ion_charging_profile](https://github.com/HallyStrats/li-ion_charging_profile) parameterisation: $f_\text{CC}$ is the CC-phase energy fraction (their `CC_FRACTION`); $\lambda$ replaces a fixed time constant by solving the energy-conservation constraint for each starting SoC.

### Uploaded kernel (override)

If the user supplies a kernel $K_\text{user}(\tau)$ (battery-side kW, resampled onto the $n_\text{bins}$ grid), it is treated as the **canonical full-charge curve** for a session $s = 0 \to s_\text{full}$ delivering $s_\text{full} \cdot C$ kWh into the battery. For an event arriving at SoC $s$, the **tail** of the curve is used:

1. Compute the cumulative battery energy: $\Phi(\tau) = \int_0^\tau K_\text{user}(\tau')\, d\tau'$.
2. Skip the head whose energy has already been delivered: $\Phi(\tau_\text{skip}) = s \cdot C$.
3. Return $K_\text{user}(\tau_\text{skip} + \tau)$ as the per-event kernel.

This gives the same total battery energy $E(s) = (s_\text{full} - s) \cdot C$ as the parametric kernel, with the user's measured shape.

In both cases the kernel is multiplied by $n_\text{bat}$ before being placed on the load timeline (parallel sessions, one per on-vehicle battery).

---

## 6. Grid load

### Cyclic per-day-mean load $L(t)$

For each event, place its (single-battery) kernel $K(\cdot; s)$ on the 24-hour daily axis starting at $t \bmod 24$, wrapping periodically. Multiply by $n_\text{bat}$, accumulate across all events, scale to fleet, and convert battery-side → grid-side by dividing by $\eta$:

$$L_\text{swap}(t) = \frac{1}{\eta} \cdot \frac{N}{N_\text{data} \cdot n_\text{days}} \sum_{b \in \text{swap}} n_\text{bat} \cdot K\big((t - t_b) \bmod 24;\, s_b\big)$$

$$L_\text{home}(t) = \frac{1}{\eta} \cdot \frac{N}{N_\text{data} \cdot n_\text{days}} \sum_{c \in \text{home}} n_\text{bat} \cdot K\big((t - t_c) \bmod 24;\, s_c\big)$$

$$\boxed{L(t) = L_\text{swap}(t) + L_\text{home}(t)} \quad \text{(grid-side kW, per-day-mean envelope)}$$

Computed on an $n_\text{bins}$-bin grid (default $n_\text{bins} = 1440$, 1-minute resolution).

### Worst-day load envelope $L_\max(t)$

The cyclic $L(t)$ above averages each minute-of-day across all data days, hiding spikes. The pipeline also reports a **worst-day** envelope: build the unfolded $(n_\text{days} \times n_\text{bins})$ load matrix and take the max across days:

$$L_\max(t) = \max_{d \in \{0,\dots,n_\text{days}-1\}} L^{(d)}(t).$$

The fleet scale used here is $N / N_\text{data}$ (no $/n_\text{days}$, because each row is a single day, not a mean).

---

## 7. Resource sizing

Each charger draws $P_\text{grid} = P_\text{CC} / \eta$ from the grid at peak.

### Power-based charger count (cyclic per-day-mean)

$$N_\text{chg,power} = \lceil \max_t L(t) / P_\text{grid} \rceil.$$

### Concurrent-session peak (preferred)

The headline charger count is sized on the **number of simultaneous physical charging slots in use** over the **unfolded multi-day** timeline. Build an $(n_\text{days} + 1) \times n_\text{bins}$ occupancy array. For each event, mark the bins where its kernel exceeds a small fraction of its own peak power (default 1%) as occupied, count $n_\text{bat}$ parallel sessions, place at the event's absolute start bin, scale by $N / N_\text{data}$, then take the max over the full timeline:

$$N_\text{chg,conc} = \lceil \max_{t \in [0, n_\text{days}]} O(t) \rceil.$$

This captures the worst-day concurrency spike that the cyclic per-day-mean view smooths out, and avoids over-sizing by power when many sessions overlap with small per-session draw.

### Battery count

$$N_\text{bat,total,conc} = N \cdot n_\text{bat} + N_\text{chg,conc}.$$

### Bikes

$$N_\text{bikes} = N.$$

---

## 8. Feasibility check (`min_feasible_N`)

For home-only scenarios (`daytime_swap = False`), an energy-feasibility check returns the minimum fleet size such that every (bike, day) trip sequence fits within one charge. With $\bar e$ the mean efficiency and effective range $R = n_\text{bat} \cdot C \cdot s_\text{full} \cdot 1000 / \bar e$:

$$N_\text{floor,avg} = \lceil \text{total km/day} / R \rceil, \qquad N_\text{floor,single} = \lceil N_\text{data} \cdot \max_\text{trip} d / R \rceil,$$

$$N_\text{min} = \max(N_\text{floor,avg}, N_\text{floor,single}).$$

A single trip with $d > R$ is flagged as infeasible — no fleet size can serve it without a daytime swap. With `daytime_swap = True`, there is no energy floor (a bike can swap mid-day arbitrarily many times).

---

## 9. Diagnostic indices

**Coincidence index** — how aligned charge-start events are with riding peaks:

$$\chi = \frac{\langle m_\text{fleet}, \Lambda \rangle}{\|m_\text{fleet}\| \cdot \|\Lambda\|} \in [0, 1].$$

**Range adequacy** — typical effective range over mean daily distance per bike:

$$\psi = \frac{n_\text{bat} \cdot C \cdot (s_\text{full} - \bar s_F) \cdot 1000 / \bar e}{\bar D}.$$

**Resource inflation** — peak-to-mean load ratio of the cyclic load:

$$\rho_R = \frac{\max_t L(t)}{\overline{L(t)}}.$$

---

## 10. Direct-events bypass (`run_pipeline_from_events`)

If the planner already has empirical charge-start events $(t, s)$ — measured arrival times and pre-charge SoCs — the trip walk can be skipped entirely. The events are treated as swap events ($t$ in hours since simulation start, $s$ in $[0, 1]$ or auto-detected as percent if max > 1.5), fed straight into $\Lambda$, $L$, $L_\max$, and the concurrent-session-peak sizing. $\chi$ and $\psi$ are not computed in this mode.

---

## 11. End-to-end summary

```
Inputs ─────────┐
  trips         │
  F_e, F_s      │
  C, charger,   │
  η, n_bat      │
  triggers      │
  N or 'auto'   │
                ▼
          (auto N → rescale trips)
                │
                ▼
          Trip walk (carryover or daily-reset)
            for each trip:
              sample e, decrement SoC
              look-ahead swap → record (t_start, pre-trip SoC)
              clamp + count any SoC < 0
            end-of-sim settle swap (carryover only)
                │
                ▼
          Λ(t) = histogram of swap + home events, scaled by N
                │
                ▼
          Build per-session battery-side kernel K(τ; s)
            parametric CC-CV with binary-searched λ, OR
            uploaded kernel with cumulative-energy tail-skip
                │
                ▼
          Cyclic L(t) = (1/η) · Σ n_bat · K placed at (t_b mod 24)
          Worst-day L_max(t) over unfolded n_days × n_bins
                │
                ▼
          N_chg,conc = concurrent_session_peak (unfolded)
          N_chg,power = ⌈max L / (P_CC/η)⌉
          N_bat,total = N · n_bat + N_chg,conc
          χ, ψ, ρ_R
```

---

## 12. Scenario configurations

| Scenario | `daytime_swap` | `overnight_charge` | `carryover` | $n_\text{bat}$ |
|---|:-:|:-:|:-:|:-:|
| Swap-only with continuous operation | ✓ | ✗ | ✓ | 1 |
| Swap + home (daily reset) | ✓ | ✓ | ✗ | 1 |
| Home only with sufficient range | ✗ | ✓ | ✗ | 2 |

Other combinations are valid but the three above correspond to the canonical paper scenarios.

---

## 13. Three operating regimes

When sweeping $N$ on a fixed-demand problem, the model exhibits three qualitative regimes that emerge from the physics without parameter changes.

| Regime | Fleet size | Per-bike km/day | Behaviour | Peak time |
|---|---|---|---|---|
| Demand-rich | $N \gg N_\text{auto}$ | $\ll$ range | bikes underutilised, no daytime swaps | late evening |
| Balanced | $N \approx N_\text{auto} / \text{utilisation}$ | $\approx$ range | ~1 swap/bike/day | early evening |
| Demand-stressed | $N \le N_\text{auto}$ | $>$ range | every bike swaps mid-day, sometimes twice | afternoon |
