"""Generate SYNTHETIC_charge_events_{N}bus.csv for the lightplan, mirroring
the bus_charging_simulator.py event-draw distributions over the same 31-day
window. Produces (t, s) tuples — hour-of-day, arrival SoC%.

Per bus_charging_simulator.py:
  num_buses_day = round(N * 0.9)  per day, from skewnorm(a=4.18, loc=7.95h, scale=3.24h)
  num_buses_eve = round(N * 0.9)  per day, from skewnorm(a=3.03, loc=17.35h, scale=2.98h)
  SoC day ~ N(54.38, 17.75), clipped [1, 97]
  SoC eve ~ N(55.35, 16.07), clipped [1, 97]
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import skewnorm, norm

p = argparse.ArgumentParser()
p.add_argument("--n_buses", type=int, required=True)
p.add_argument("--n_days", type=int, default=31)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--out", type=Path, default=None)
args = p.parse_args()

rng = np.random.default_rng(args.seed)
# scipy uses its own RNG; seed it for reproducibility
np.random.seed(args.seed)

events = []
for day in range(args.n_days):
    n_day = int(round(args.n_buses * 0.9))
    n_eve = int(round(args.n_buses * 0.9))

    t_day = skewnorm.rvs(a=4.18, loc=7.95, scale=3.24, size=n_day)
    t_day = np.clip(t_day, 7.0, 15.99)
    s_day = norm.rvs(loc=54.38, scale=17.75, size=n_day)
    s_day = np.clip(s_day, 1.0, 97.0)

    t_eve = skewnorm.rvs(a=3.03, loc=17.35, scale=2.98, size=n_eve)
    t_eve = np.clip(t_eve, 16.0, 23.99)
    s_eve = norm.rvs(loc=55.35, scale=16.07, size=n_eve)
    s_eve = np.clip(s_eve, 1.0, 97.0)

    t = np.concatenate([t_day, t_eve])
    s = np.concatenate([s_day, s_eve])
    events.append(pd.DataFrame({"t": t, "s": s}))

df = pd.concat(events, ignore_index=True)
out = args.out or Path(f"SYNTHETIC_charge_events_{args.n_buses}bus.csv")
df.to_csv(out, index=False)
print(f"wrote {out}: {len(df)} events ({args.n_buses} buses × {args.n_days} days × 1.8/day)")
