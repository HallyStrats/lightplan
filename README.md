# lightplan web app

Streamlit UI for the `math_pipeline.md` model: trip walk → swap intensity Λ(t) →
grid load L(t) → resource counts + diagnostic indices (χ, ψ, ρ_R).

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

At the top, switch between **Results dashboard** (one-click answers, multiple
scenarios overlaid) and **Step-by-step walkthrough** (single scenario broken
into 9 explained stages with equations and visualisations).

## Trip CSV format

| column   | meaning                              |
|----------|--------------------------------------|
| bike_id  | integer bike identifier              |
| day      | integer day index (0, 1, 2, …)       |
| t_start  | trip start time, hours in [0, 24)    |
| dt       | trip duration, hours                 |
| d        | trip distance, km                    |

`F_e` and `F_s` default to Normal (set μ, σ via sliders). Optionally upload a
1-column CSV of empirical samples to override either distribution.

If you don't have data, switch the sidebar to **Synthetic** and explore
parameter space directly.
