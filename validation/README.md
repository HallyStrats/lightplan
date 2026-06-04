# Validation bundle

Reproducibility artifact for the validation in the paper *"A generalisable tool
for grid-load and charging-infrastructure planning across motorcycle and bus
fleets, demonstrated in sub-Saharan Africa."*

It contains, for all three case studies (Nairobi boda-boda, Cape Town last-mile
delivery, South African bus depots), the source-pipeline reference outputs, the
planning-tool outputs, the analysis scripts, the consolidated metrics that
populate the appendix tables and the validation figures, and (for the bus case)
the synthetic arrival-event inputs and their generator. The motorcycle per-trip
telemetry inputs are withheld for rider privacy (see Notes); they are not needed
to regenerate any table or figure.

## Reproduce from a single command

From this folder:

```bash
pip install pandas numpy matplotlib
python build_all_comparisons.py
```

This reads the bundled tool outputs and source-pipeline reference outputs,
recomputes the 28-scenario validation metrics (Pearson `r`, daily-energy `ΔE`,
peak `Δpeak`, charger/battery counts), writes `all_comparisons/all_studies_metrics.csv`,
and renders the validation comparison figures into `all_comparisons/`. The
regenerated metrics CSV reproduces the committed `all_comparisons/all_studies_metrics.csv`
exactly.

To regenerate the appendix LaTeX tables from that metrics CSV:

```bash
python make_appendix_tables.py
```

## Layout

```
all_comparisons/all_studies_metrics.csv   # the 28-scenario metrics behind Tables A.1–A.2
build_all_comparisons.py                  # recompute metrics + figures (single command)
make_appendix_tables.py                   # metrics CSV -> appendix LaTeX tables
comparison_resources_table.csv            # Cape Town charger/battery counts (resources figure)

<Study>_Simulation/
  inputs/             # bus only: synthetic arrival events + generator (motorcycle telemetry withheld)
  comparison/         # planning-tool outputs (RESULTS_*_load.csv, _events.csv, _summary.csv)
  reference_outputs/  # source-pipeline reference load curves the tool is validated against
```
Studies: `Nairobi_Simulation`, `Cape_Town_Simulation`, `Bus_Simulation`.

## What each scenario maps to

- **Nairobi (20 panels):** 7 configurations × 2–3 charging approaches. Tool
  outputs in `Nairobi_Simulation/comparison/trips_*/RESULTS_*`; source curves in
  `Nairobi_Simulation/reference_outputs/{all,150km,120km}/*_charging_load_minute.csv`.
- **Cape Town (2 panels):** Swap-only and Both. Tool output in
  `Cape_Town_Simulation/comparison/trips_all_operating/`; source curve in
  `Cape_Town_Simulation/reference_outputs/actual_data_run/charging_load_per_minute.csv`.
- **Bus (6 panels):** 3 fleet sizes × 2 charger powers. Tool outputs in
  `Bus_Simulation/comparison/RESULTS_SYNTHETIC_*`; source curves in
  `Bus_Simulation/reference_outputs/synthetic/*_unmanaged_*.csv`. Synthetic
  arrival-event inputs and their generator are in `Bus_Simulation/inputs/`.

## Notes

- **Privacy:** the motorcycle per-trip telemetry inputs (Nairobi and Cape Town)
  are **not** included — neither the raw GPS tracks nor any reduced per-trip
  records — so no rider movement data is published. The validation here
  regenerates entirely from the bundled pipeline outputs and source reference
  curves, which need no input data; only the bus *synthetic* arrival events and
  their generator are included.
- **Nairobi window:** the validation used the full tracking extract of 119
  vehicles over 26 active recording days (2023-11-13 to 2023-12-21, two active
  blocks separated by a dormant gap). The source study reported a baseline
  subset of this record; the planning tool is validated against the source
  pipeline re-run on the complete extract.
- **Requirements:** Python 3.10+, pandas, numpy, matplotlib.
