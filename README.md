# COCOON-Sionna

COCOON-Sionna is an outdoor distributed-MIMO simulation and AP-placement
pipeline built on top of Sionna RT. The project:

- building an OSM-derived outdoor scene for KU Leuven Gent Campus Rabot
- simulating pedestrian mobility and pairwise CSI for `AP-AP`, `AP-UE`, and `UE-UE`
- generating wall-mounted candidate AP positions at `1.5 m`
- generating one rooftop central-AP proxy from the building nearest the area center
- comparing `central_massive_mimo`, `distributed_fixed`, and any enabled movable optimizations over the same CSI-derived objective
- exporting reusable simulation data first and generating figures in postprocessing

At a high level, each scenario does the following:

1. Build or load a prebuilt outdoor scene and its walk graph.
2. Generate wall-mounted candidate AP positions over building boundaries at `1.5 m`.
3. Generate UE trajectories over the walkable space.
4. Compute path-based wireless channels with Sionna RT.
5. Compare `central_massive_mimo`, `distributed_fixed`, and the enabled movable strategies from `distributed_movable`, `distributed_movable_optimization_2`, and `distributed_movable_optimization_3`.
6. Export CSI, optional coverage-map data, trajectories, per-mode summaries, and placement schedules.
7. Rebuild figures, animations, and ESR analyses from the stored outputs via postprocessing.

## System Model

We use the following notation for the downlink distributed-MIMO model.

- `\mathcal{A} = \{1,\dots,M\}`: set of APs.
- `\mathcal{U} = \{1,\dots,K\}`: set of scheduled UEs.
- `N_m`: number of transmit antenna ports at AP `m`.
- `N_{\mathrm{t}} = \sum_{m \in \mathcal{A}} N_m`: total number of distributed transmit ports.
- `N_{r,k}`: number of receive antenna ports at UE `k`.

For UE `k`, define the per-AP downlink channel block

```math
\mathbf{H}_{k,m} \in \mathbb{C}^{N_{r,k} \times N_m},
\qquad m \in \mathcal{A}.
```

The full distributed channel seen by UE `k` is the horizontal concatenation

```math
\mathbf{H}_k
=
\left[\mathbf{H}_{k,1}\ \mathbf{H}_{k,2}\ \cdots\ \mathbf{H}_{k,M}\right]
\in \mathbb{C}^{N_{r,k} \times N_{\mathrm{t}}}.
```

Ignoring channel-estimation error for now, UE `k` applies a receive combiner
`\mathbf{u}_k \in \mathbb{C}^{N_{r,k}}` and obtains the effective downlink channel

```math
\mathbf{h}_k^{\mathsf{H}} = \mathbf{u}_k^{\mathsf{H}} \mathbf{H}_k
\in \mathbb{C}^{1 \times N_{\mathrm{t}}}.
```

Stacking all scheduled users gives

```math
\mathbf{H}_{\mathrm{eff}}
=
\begin{bmatrix}
\mathbf{h}_1^{\mathsf{H}} \\
\mathbf{h}_2^{\mathsf{H}} \\
\vdots \\
\mathbf{h}_K^{\mathsf{H}}
\end{bmatrix}
\in \mathbb{C}^{K \times N_{\mathrm{t}}}.
```

The transmitted signal is

```math
\mathbf{x}
=
\sum_{i \in \mathcal{U}} \mathbf{f}_i \sqrt{p_i}\, s_i
=
\mathbf{F}\mathbf{P}^{1/2}\mathbf{s},
```

where:

- `\mathbf{f}_i \in \mathbb{C}^{N_{\mathrm{t}}}` is the precoding vector for UE `i`,
- `p_i` is the downlink power allocated to UE `i`,
- `s_i` is the unit-power symbol for UE `i`,
- `\mathbf{F} = [\mathbf{f}_1,\dots,\mathbf{f}_K]`.

For zero forcing, the project uses the effective distributed channel and forms a
least-squares pseudo-inverse precoder, followed by column normalization:

```math
\mathbf{F}_{\mathrm{ZF}}
=
\mathbf{H}_{\mathrm{eff}}^{\dagger}.
```

The received signal at UE `k` is

```math
y_k
=
\mathbf{h}_k^{\mathsf{H}}\mathbf{f}_k \sqrt{p_k}\, s_k
+
\sum_{i \in \mathcal{U},\, i \neq k}
\mathbf{h}_k^{\mathsf{H}}\mathbf{f}_i \sqrt{p_i}\, s_i
+
n_k,
```

with `n_k \sim \mathcal{CN}(0,\sigma^2)`, where

```math
\sigma^2 = k_{\mathrm{B}} T B.
```

The downlink SINR of UE `k` is therefore

```math
\mathrm{SINR}^{\mathrm{DL}}_k
=
\frac{
    p_k \left|\mathbf{h}_k^{\mathsf{H}}\mathbf{f}_k\right|^2
}{
    \sum_{i \in \mathcal{U},\, i \neq k}
    p_i \left|\mathbf{h}_k^{\mathsf{H}}\mathbf{f}_i\right|^2
    + \sigma^2
}.
```

The implementation keeps the interference term explicitly and exports
`desired_power_w`, `interference_power_w`, `noise_power_w`, linear SINR, and
SINR in dB for every evaluated UE snapshot. It does not rely on the
interference-free simplification.

The total AP power budget is the sum of all AP transmit powers and is split
equally across active user streams.

## Placement Model

Each run compares three placement strategies over the same deployment model.

The Placement Model does **not** optimize a true massive-MIMO single
co-located array. The `central_massive_mimo` mode is a rooftop proxy with the
same total antenna-element and transmit-power budget as the distributed
deployments.

- `num_fixed_aps`: APs that remain active and stationary for the full scenario
- `num_movable_aps`: APs selected from the candidate AP positions
- baseline = initial AP constellation = fixed APs plus the initial movable AP placement
- `N` always refers to `num_movable_aps`

### Candidate AP Positions

Candidate AP positions are the potential wall-mounted positions available to
the movable APs. They are generated along building walls at `1.5 m` height and
are not part of the fixed AP set.

Candidate generation is controlled by:

- `candidate_wall_height_m`
- `candidate_wall_spacing_m`
- `candidate_corner_clearance_m`
- `candidate_wall_offset_m`
- `candidate_min_spacing_m`

For scenes that do not expose wall geometry directly, the same candidate AP
positions can be supplied explicitly. In the current three-mode comparison,
`placement.num_fixed_aps` must remain `0`: the distributed baseline is the
initial sampled distributed AP constellation, and the central AP uses a
separate rooftop candidate pool derived from building metadata.

### Placement Config Example

```yaml
placement:
  num_fixed_aps: 0
  num_movable_aps: 4
  enable_optimization_1: true
  enable_optimization_2: true
  enable_optimization_3: true
  window_interval_s: 10.0
  historical_csi_decay_rate_per_s: 0.0693
  candidate_wall_height_m: 1.5
  candidate_wall_spacing_m: 8.0
  candidate_corner_clearance_m: 2.0
  candidate_wall_offset_m: 0.5
  candidate_min_spacing_m: 3.0
  random_seed: 7
  heuristic_k_nearest: 8
  optimization_2_distance_threshold_m: 25.0
  exact_max_iterations: 50000
```

### Baseline

The baseline is the initial distributed AP constellation:

- sample `N` distributed APs once from the wall-mounted candidate AP positions
- keep that sampled constellation fixed for the full run

That defines the `distributed_fixed` mode and serves as the comparison
reference.

### Strategies To Compare

Each scenario compares these deployment modes:

- `central_massive_mimo`: select the building whose rooftop representative
  point is nearest the area center, place one rooftop proxy AP there, mount it
  `1.5 m` above the roof, and orient it toward the area center
- `distributed_fixed`: sample `N` distributed APs once from the wall-mounted
  candidate AP positions and keep that constellation fixed for the full run
- `distributed_movable`: keep the same distributed AP budget, but relocate the
  APs per window using a local-CSI heuristic with exponentially decayed
  historical CSI over the wall-mounted candidate AP positions
- `distributed_movable_optimization_2`: keep the same distributed AP budget,
  but relocate the APs per window using a nearest-user `UE-UE` proxy-CSI rule:
  each candidate site binds to the closest user snapshot within the configured
  distance threshold, and the resulting proxy CSI is turned into a proxy ESR
- `distributed_movable_optimization_3`: keep the same distributed AP budget,
  but relocate the APs per window using the same nearest-user distance-threshold
  proxy as optimization 2 and rank candidates by proxy average power
  `|CSI|^2` instead of proxy ESR

Set `placement.enable_optimization_1`, `placement.enable_optimization_2`, and
`placement.enable_optimization_3` to enable or disable the three movable
heuristics independently. `central_massive_mimo` and `distributed_fixed` remain
available in all runs.

### Placement Scoring

Placement evaluation mixes deployed CSI and proxy measurements:

1. Deployed `AP-UE` CSI provides the trajectory SINR samples used for outage
   and percentile-based reporting after a subset has been selected and moved.
2. `UE-UE` CSI provides the peer-aware weighting used as a tie-break term and
   also provides the nearest-user proxy CSI used by optimization 2.
3. `UE-UE` proxy power `|CSI|^2` provides the nearest-user power metric used
   by optimization 3.
4. `AP-AP` CSI is exported for analysis, but the placement score does not
   depend on a full radio map.

`distributed_movable` is optimization 1. It uses the configured `K` nearest UE
snapshots around each candidate AP position, aggregates local CSI over time
with exponential decay, and ranks candidates by their local weighted `P10`
SINR. Samples at the current decision time keep weight `1`, while older CSI is
down-weighted by `exp(-\lambda \Delta t)`.

`distributed_movable_optimization_2` is optimization 2. The candidate-position
`AP-UE` CSI is unknown before relocation, so the controller does not ray-trace
every dormant candidate. Instead, for each candidate site and snapshot, it
binds the site to the closest user within
`optimization_2_distance_threshold_m`, uses that user's measured `UE-UE` proxy
CSI to all receivers as a surrogate candidate-to-UE channel, and ranks
candidates by the resulting proxy ESR `\sum \log_2(1+\mathrm{SINR})`.

`distributed_movable_optimization_3` is optimization 3. It uses the same
candidate-to-nearest-user proxy association as optimization 2, but discards the
complex proxy CSI and ranks candidates only by the proxy average received power
`|CSI|^2`.

### Evaluation Flow Per Scenario

With ray tracing enabled, a full scenario run proceeds as follows:

1. Build the scene once with `build-scene`, then load the prebuilt scene and mobility graph.
2. Generate distributed candidate AP positions along the walls at `1.5 m`.
3. Generate one rooftop central-AP proxy candidate from the building nearest the area center.
4. Generate UE trajectories.
5. Compute the CSI used for scoring:
   `UE-UE`, deployed-`AP-UE`, and exported `AP-AP`.
6. Build the distributed baseline as the initial sampled AP constellation.
7. Compare the deployment modes:
   `distributed_fixed` stays static, `distributed_movable` applies the
   historical `K`-nearest local-CSI heuristic, `distributed_movable_optimization_2`
   applies the distance-threshold nearest-user `UE-UE` proxy-ESR heuristic,
   `distributed_movable_optimization_3` applies the same nearest-user
   proxy-power heuristic, and
   `central_massive_mimo` evaluates the fixed rooftop proxy over the full
   trajectory.
8. Export per-mode placements, schedules, optional coverage-map data, SINR
   comparisons, and summary metrics.
9. Generate figures and animations separately with `postprocess`.

When `solver.enable_ray_tracing: false`, the pipeline skips CSI-driven
placement scoring but still exports trajectories, AP layouts, schedules, and
summary data. When `coverage.enabled: false`, the pipeline still runs the
placement comparison but skips coverage-map computation and exports.

## Layout

- `src/cocoon_sionna/`: pipeline code
- `scenarios/`: runnable scenario configs
- `data/`: checked-in boundary, AP site, and demo mobility inputs
- `tests/`: unit and smoke tests

## Environment

The repo is set up for Python `3.12` and the existing Windows virtualenv in
`.venv/`. The project metadata is in `pyproject.toml`.

Install into the repo venv:

```powershell
.venv\Scripts\python.exe -m pip install -e .[dev]
```

## Run

Run the Rabot scenario:

```powershell
.venv\Scripts\python.exe -m cocoon_sionna.cli run scenarios/rabot.yaml
```

Enable more verbose progress logging:

```powershell
.venv\Scripts\python.exe -m cocoon_sionna.cli --log-level DEBUG run scenarios/rabot.yaml
```

During interactive runs, the CLI now shows:

- a scenario-level progress bar
- snapshot progress bars for `UE-UE` and `AP-UE` CSI extraction
- explicit backend selection output showing whether the run is using `GPU` or `CPU`

Build only the Rabot outdoor scene assets:

```powershell
.venv\Scripts\python.exe -m cocoon_sionna.cli build-scene scenarios/rabot.yaml
```

`build-scene` now forces an in-place rebuild of the OSM scene assets and clears
stale mesh files from `generated/.../meshes`, so deleting the whole generated
folder first should no longer be necessary.

For OSM-backed scenes, you can define the build area either with
`scene.boundary_path` pointing to a GeoJSON polygon or with
`scene.boundary_bbox: [west, south, east, north]` copied directly from the
OpenStreetMap export dialog. After changing either one, rebuild the Rabot scene
assets before running the scenario again:

```powershell
.venv\Scripts\python.exe -m cocoon_sionna.cli build-scene scenarios/rabot.yaml
```

Run the full Rabot placement-strategy comparison:

```powershell
.venv\Scripts\python.exe -m cocoon_sionna.cli run scenarios/rabot.yaml
```

Rebuild all figures and analyses from the stored outputs:

```powershell
.venv\Scripts\python.exe -m cocoon_sionna.cli postprocess scenarios/rabot.yaml
```

For faster placement-comparison runs with less disk I/O, you can disable CSI
artifact writes in the scenario YAML:

```yaml
outputs:
  write_csi_exports: false
```

## Outputs

Each `run` writes to its configured output directory and produces reusable data:

- `candidate_ap_positions.csv`
- `central_ap_rooftop_candidates.csv`
- `central_massive_mimo_ap.csv`
- `central_massive_mimo_schedule.csv`
- `distributed_fixed_aps.csv`
- `distributed_fixed_schedule.csv`
- `distributed_movable_aps.csv`
- `distributed_movable_schedule.csv`
- `distributed_movable_optimization_2_aps.csv`
- `distributed_movable_optimization_2_schedule.csv`
- `distributed_movable_optimization_3_aps.csv`
- `distributed_movable_optimization_3_schedule.csv`
- `strategy_comparison.csv`
- `summary.json`
- `infra_csi_snapshots.npz`
- `peer_csi_snapshots.npz`
- `trajectory.csv`
- `scene_metadata.json`
- `walk_graph.json`
- `user_sinr_summary.csv`
- `user_sinr_timeseries.csv`
- `user_sinr_snapshots.npz`
- `run.log`

When `outputs.write_csi_exports: false`, the pipeline skips
`peer_csi_snapshots.npz` and `infra_csi_snapshots.npz` and also avoids the full
CFR/CIR/tau export path.

When `solver.enable_ray_tracing: true` and `coverage.enabled: true`, the output
directory also includes:

- `coverage_map.npz`
- `fixed_coverage_map.npz`

`candidate_ap_positions.csv` stores the wall-generated distributed candidate AP
positions.
`central_ap_rooftop_candidates.csv` stores the rooftop candidates considered by
`central_massive_mimo`.
`central_massive_mimo_ap.csv`, `distributed_fixed_aps.csv`, and
`distributed_movable_aps.csv`, and
`distributed_movable_optimization_2_aps.csv`, and
`distributed_movable_optimization_3_aps.csv` store the selected deployment for
each mode.
`central_massive_mimo_schedule.csv`, `distributed_fixed_schedule.csv`, and
`distributed_movable_schedule.csv`, and
`distributed_movable_optimization_2_schedule.csv`, and
`distributed_movable_optimization_3_schedule.csv` store the per-window AP
schedules. The central and fixed distributed schedules remain static; the
movable distributed schedules can change per relocation window.
`strategy_comparison.csv` reports the per-mode score, outage, and percentile
metrics.
`summary.json` records the same comparison metrics together with the selected
compute backend, Mitsuba variant, the central-AP antenna/power budget, copied
scene-context paths, and the postprocess animation speed.
`infra_csi_snapshots.npz` contains `AP-AP` and `AP-UE` CSI exports for the
evaluated placements, including explicit desired/interference/noise power terms
and linear SINR for `AP-UE`.
`peer_csi_snapshots.npz` contains `UE-UE` CSI exports and the peer-derived
weighting used in the placement score.
`user_sinr_timeseries.csv` stores per-snapshot per-user SINR rows for each
strategy, including an explicit `snapshot_index` column for postprocessing.
`user_sinr_snapshots.npz` stores the same SINR data in array form with
`snapshot_index`, `times_s`, `ue_ids`, `strategy_names`, and per-strategy
`sinr_linear`, `sinr_db`, `desired_power_w`, `interference_power_w`, and
`noise_power_w` matrices.

The `postprocess` command recreates visual outputs such as:

- `scene_layout.png`
- `scene_animation.mp4` or `scene_animation.gif`
- `scene_animation_with_central_massive_mimo.mp4` or `.gif`
- `trajectory_colormap.png`
- `coverage_map.png`
- `fixed_coverage_map.png`
- `user_sinr_cdf.png`
- ESR time-series/CDF figures and the full analysis bundle under `postprocessing/`

Each static postprocessed figure also gets a same-name `.tex` PGFPlots file
that reads CSV exports directly and reuses the shared placement-model color
definitions across the plots.

## Postprocessing

For manuscript work, the fastest option is to run the bundled helper script:

```bash
python scripts/run_all_postprocessing.py scenarios/rabot.yaml
```

This resolves the scenario output directory automatically and runs the full
postprocessing bundle in one pass:

- scene-layout, animation, trajectory, and coverage visualizations
- strategy summary tables
- SINR snapshot analysis tables and figures
- ESR time-series, ESR CDF, and time-conditioned ESR CDF figures
- AP relocation schedule analysis
- a combined manuscript summary and manifest

You can also point it directly at an output directory:

```bash
python scripts/run_all_postprocessing.py outputs/rabot
```

The lower-level analysis scripts remain available when you only want one part
of the toolkit:

- `python scripts/analyze_strategy_performance.py outputs/rabot`
- `python scripts/analyze_sinr_snapshots.py outputs/rabot`
- `python scripts/analyze_ap_schedule.py outputs/rabot`
- `python scripts/build_manuscript_report.py outputs/rabot`

For OSM-built scenes, the generated Sionna assets are emitted as:

- `scene.xml`
- `meshes/ground.ply`
- `meshes/*-wall.ply`
- `meshes/*-roof.ply`
- `build-scene.log`

## Notes

- Default RF assumptions are `3.5 GHz`, `100 MHz`, and `256` CFR bins.
- The runtime always probes the CUDA Mitsuba backend first. If the GPU probe fails,
  it falls back automatically to the LLVM CPU backend and logs the reason.
- Candidate AP positions are wall-based by default and use
  `candidate_wall_height_m: 1.5`.
- The central AP is rooftop-mounted `1.5 m` above the roof and is selected from
  one fixed building: the rooftop representative point nearest the area center.
- The distributed baseline is the initial sampled AP constellation and remains
  fixed for the full run.
- `distributed_movable` reuses the same distributed AP budget but can relocate
  the APs per optimization window using exponentially decayed historical CSI
  over the configured `K` nearest UE snapshots.
- `distributed_movable_optimization_2` reuses the same distributed AP budget
  but relocates the APs from a nearest-user `UE-UE` proxy-CSI model and ranks
  the candidate subset by proxy ESR.
- `distributed_movable_optimization_3` reuses the same distributed AP budget
  but relocates the APs from the same nearest-user distance-threshold proxy
  model using proxy average power `|CSI|^2`.
- The central AP is normalized to the same total antenna-element budget and the
  same total transmit-power budget as the distributed deployments.
- The Placement Model does not optimize a true single co-located massive-MIMO
  array; `central_massive_mimo` is a normalized rooftop proxy.
- v1 is outdoor-only and ignores vegetation, weather, traffic, and indoor areas.
- The Rabot boundary and AP site files are seed inputs and can be tightened
  later without changing code.
