# COCOON-Sionna

COCOON-Sionna is an outdoor distributed-MIMO simulation and AP-placement
pipeline built on top of Sionna RT. The project:

- building an OSM-derived outdoor scene for KU Leuven Gent Campus Rabot
- simulating pedestrian mobility and pairwise CSI for `AP-AP`, `AP-UE`, and `UE-UE`
- generating wall-mounted candidate AP positions at `1.5 m`
- splitting each deployment into fixed APs plus movable APs
- comparing three placement strategies over the same CSI-derived objective

At a high level, each scenario does the following:

1. Build or load an outdoor scene and its walk graph.
2. Generate wall-mounted candidate AP positions over building boundaries at `1.5 m`.
3. Generate UE trajectories over the walkable space.
4. Compute path-based wireless channels with Sionna RT.
5. Compare the `random_baseline`, `local_csi_p10`, and `capped_exact_search` placement strategies.
6. Export CSI, optional coverage maps, trajectories, per-strategy summaries, and placement schedules.

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

Under perfect CSI and ideal ZF, the multi-user interference terms vanish,
` \mathbf{h}_k^{\mathsf{H}}\mathbf{f}_i = 0` for `i \neq k`, so

```math
\mathrm{SINR}^{\mathrm{DL}}_k
=
\frac{
    p_k \left|\mathbf{h}_k^{\mathsf{H}}\mathbf{f}_k\right|^2
}{
    \sigma^2
}.
```

In the current implementation, the total AP power budget is the sum of all AP
transmit powers and is split equally across active user streams.

## Placement Model

Each run compares three placement strategies over the same deployment model.

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
positions can be supplied explicitly. The fixed APs remain active in every CSI
solve, but they are excluded from the movable candidate pool.

### Placement Config Example

```yaml
placement:
  num_fixed_aps: 2
  num_movable_aps: 4
  enable_capped_exact_search: true
  window_interval_s: 10.0
  candidate_wall_height_m: 1.5
  candidate_wall_spacing_m: 8.0
  candidate_corner_clearance_m: 2.0
  candidate_wall_offset_m: 0.5
  candidate_min_spacing_m: 3.0
  random_seed: 7
  heuristic_k_nearest: 8
  exact_max_iterations: 50000
```

### Baseline

The baseline is the initial AP constellation:

- all fixed APs
- `N` movable APs sampled once from the candidate AP positions

That initial movable placement defines the `random_baseline` strategy and stays
unchanged across all relocation windows. It is the reference deployment used in
the strategy comparison.

### Strategies To Compare

Each scenario compares these three strategies:

- `random_baseline`: place `N` movable APs randomly on candidate AP positions
  once, then keep that constellation fixed for the full run
- `local_csi_p10`: for each candidate AP position, gather the `K` nearest UE
  snapshots, score the candidate with local CSI-derived `P10` SINR, then select
  the movable AP positions per relocation window with this non-exhaustive
  heuristic
- `capped_exact_search`: evaluate movable-AP combinations over the full
  candidate AP pool for each relocation window, stop at `exact_max_iterations`,
  and return the best combination found so far together with an explicit capped
  status when the full search is not completed

Set `enable_capped_exact_search: false` to compare only `random_baseline` and
`local_csi_p10`.

### Placement Scoring

Placement evaluation uses CSI only:

1. `AP-UE` CSI provides the trajectory SINR samples used for outage and
   percentile-based scoring.
2. `UE-UE` CSI provides the peer-aware weighting used as a tie-break term.
3. `AP-AP` CSI is exported for analysis, but the placement score does not
   depend on a full radio map.

The `local_csi_p10` heuristic uses the `K` nearest UE snapshots around each
candidate AP position and ranks candidates by their local `P10` SINR. This
targets a 90%-user SINR floor rather than peak local performance. The
comparison summary still reports the full trajectory-level score for each
strategy.

### Evaluation Flow Per Scenario

With ray tracing enabled, a full scenario run proceeds as follows:

1. Build or load the scene and mobility graph.
2. Generate candidate AP positions along the walls at `1.5 m`.
3. Generate UE trajectories.
4. Compute the CSI used for scoring:
   `UE-UE`, `AP-UE`, and exported `AP-AP`.
5. Build the baseline as the initial AP constellation:
   fixed APs plus the sampled `random_baseline` movable APs.
6. Compare the three strategies over the relocation windows:
   `random_baseline` stays fixed, while `local_csi_p10` and
   `capped_exact_search` choose movable AP positions per window.
7. Export per-strategy placements, schedules, optional coverage maps, SINR
   comparisons, and summary metrics.

When `solver.enable_ray_tracing: false`, the pipeline skips CSI-driven
placement scoring but still exports trajectories, AP layouts, schedules, and
scene visualizations. When `coverage.enabled: false`, the pipeline still runs
the placement comparison but skips coverage-map computation and exports.

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

For faster placement-comparison runs with less disk I/O, you can disable CSI
artifact writes in the scenario YAML:

```yaml
outputs:
  write_csi_exports: false
```

## Outputs

Each scenario writes to its configured output directory and produces:

- `fixed_aps.csv`
- `candidate_ap_positions.csv`
- `random_baseline_movable_aps.csv`
- `random_baseline_schedule.csv`
- `local_csi_p10_movable_aps.csv`
- `local_csi_p10_schedule.csv`
- `capped_exact_search_movable_aps.csv`
- `capped_exact_search_schedule.csv`
- `strategy_comparison.csv`
- `summary.json`
- `infra_csi_snapshots.npz`
- `peer_csi_snapshots.npz`
- `trajectory.csv`
- `scene_render.png`
- `scene_camera.mp4`
- `scene_layout.png`
- `scene_animation.mp4` or `scene_animation.gif`
- `trajectory_colormap.png`
- `user_sinr_cdf.png`
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
- `coverage_map.png`
- `fixed_coverage_map.npz`
- `fixed_coverage_map.png`

`candidate_ap_positions.csv` stores the wall-generated candidate AP positions
available to the movable AP pool.
`random_baseline_movable_aps.csv`, `local_csi_p10_movable_aps.csv`, and
`capped_exact_search_movable_aps.csv` store the movable AP placements chosen by
each strategy.
`random_baseline_schedule.csv`, `local_csi_p10_schedule.csv`, and
`capped_exact_search_schedule.csv` store the per-window movable AP schedules.
`strategy_comparison.csv` reports the per-strategy score, outage, percentile,
and capped/exact status for the exhaustive search.
`summary.json` records the same comparison metrics together with the selected
compute backend and Mitsuba variant.
`infra_csi_snapshots.npz` contains `AP-AP` and `AP-UE` CSI exports for the
evaluated placements.
`peer_csi_snapshots.npz` contains `UE-UE` CSI exports and the peer-derived
weighting used in the placement score.
`scene_render.png` is a rendered view of the loaded Sionna scene using
`scene.render(...)`, with the selected APs and the active UE snapshot shown as
radio devices inside the 3D environment.
`scene_camera.mp4` is a rendered 3D camera video of the loaded Sionna scene,
using a fixed oblique camera while the UE devices move over time. This output
requires `ffmpeg`.
`scene_layout.png` shows a top-down view of the loaded scene, including the walk
graph, building footprints when available, fixed APs, candidate AP positions,
selected placements, and UE trajectories.
`scene_animation.mp4` animates the moving UEs over the loaded scene together
with the AP placements. If `ffmpeg` is unavailable, the pipeline writes
`scene_animation.gif` instead. Playback speed is controlled via
`outputs.scene_animation_speedup` in the scenario YAML, so you can export the
animation faster than real time, e.g. `10.0` for `10x`.
`trajectory_colormap.png` shows the full UE trajectories in a single PNG, with a
colormap over time so the motion direction can be followed visually. Initial UE
seeding is spread across the walk graph, and route selection biases toward
underexplored walkable edges to improve overall area coverage.
`user_sinr_cdf.png` compares the CDF of instantaneous distributed-MIMO zero-forcing
SINR samples across all user snapshots for the compared placement strategies.
`user_sinr_timeseries.csv` stores per-snapshot per-user SINR rows for each
strategy, including an explicit `snapshot_index` column for postprocessing.
`user_sinr_snapshots.npz` stores the same SINR data in array form with
`snapshot_index`, `times_s`, `ue_ids`, `strategy_names`, and one
`<strategy>_sinr_db` matrix per strategy.

## Postprocessing

For manuscript work, the fastest option is to run the bundled helper script:

```bash
python scripts/run_all_postprocessing.py scenarios/rabot.yaml
```

This resolves the scenario output directory automatically and runs the full
postprocessing bundle in one pass:

- strategy summary tables
- SINR snapshot analysis tables and figures
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
- Fixed APs are always active and never part of the movable candidate pool.
- The baseline is the initial AP constellation: fixed APs plus the initial
  `random_baseline` movable AP placement.
- `random_baseline` stays static across all relocation windows, while
  `local_csi_p10` and `capped_exact_search` are evaluated per window.
- `capped_exact_search` reports whether the full candidate-combination search
  completed or stopped at `exact_max_iterations`.
- v1 is outdoor-only and ignores vegetation, weather, traffic, and indoor areas.
- The Rabot boundary and AP site files are seed inputs and can be tightened
  later without changing code.
