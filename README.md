# COCOON-Sionna

COCOON-Sionna is an outdoor distributed-MIMO simulation and AP-placement
pipeline built on top of Sionna RT. The project:

- validating placement logic on the bundled `etoile` scene
- building an OSM-derived outdoor scene for KU Leuven Gent Campus Rabot
- simulating pedestrian mobility and pairwise CSI for `AP-AP`, `AP-UE`, and `UE-UE`
- anchoring fixed APs to building walls and selecting a spread-out baseline constellation
- optimizing a second mobile AP constellation on wall anchors using UE-UE and UE-AP CSI

At a high level, each scenario does the following:

1. Build or load an outdoor scene and its walk graph.
2. Generate UE trajectories over the walkable space.
3. Compute path-based wireless channels with Sionna RT.
4. Evaluate fixed and mobile wall-mounted AP constellations using CSI-derived trajectory SINR.
5. Reposition the mobile AP constellation on a configured relocation schedule.
6. Export CSI, optional coverage maps, trajectories, SINR comparison plots, and AP movement schedules.

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

## Optimization Model

The optimization implemented in this repository is a placement and relocation
problem over a finite set of AP candidate sites.

- A fixed AP constellation is always defined first.
- For OSM scenes, candidate AP sites are generated along building walls and the
  fixed baseline is chosen as a far-apart subset of those wall anchors.
- For non-OSM scenes, baseline APs come from `candidate_sites_path`.
- A second, mobile AP constellation can then be optimized over the same
  candidate pool.

Two deployment modes are supported:

- `optimization.enable_optimization: true`: the mobile AP constellation is
  optimized and may relocate over time.
- `optimization.enable_optimization: false`: the mobile AP constellation is
  disabled and the fixed AP constellation is reused unchanged.

In all cases, the fixed AP constellation remains the reference deployment used
for comparison.

You can now split the deployment into:

- `num_fixed_aps`: APs that stay at their seed positions for the whole run
- `num_mobile_aps`: APs that are optimized over the candidate AP pool

If `num_mobile_aps` is omitted, the code falls back to the legacy
`num_selected_aps` field for backward compatibility.

### Candidate Sites

Candidate APs are the possible positions that the movable APs are allowed to
occupy during optimization. They are not the same thing as the always-fixed APs.

For OSM-derived outdoor scenes, the code extracts building footprints, creates
wall-mounted anchor points, and enforces spacing/clearance constraints via:

- `wall_candidate_spacing_m`
- `wall_corner_clearance_m`
- `wall_mount_height_m`
- `wall_mount_offset_m`
- `candidate_min_spacing_m`

For non-OSM scenes, the pipeline loads the candidate AP list from
`candidate_sites_path`. It can also augment that list with a bounded number of
trajectory-derived candidate sites near UE motion.

### Fixed Baseline

The fixed baseline is the non-relocated AP deployment:

- In OSM scenes, the code selects a spread-out wall-mounted set using farthest
  spacing.
- In non-OSM scenes, the code takes the configured fixed and movable seed APs
  from `candidate_sites_path`.

The baseline deployment is the union of:

- the always-fixed APs
- the movable APs at their initial seed positions

This baseline is used to compute:

- fixed `AP-UE` CSI
- fixed `AP-AP` CSI
- an optional fixed coverage map
- fixed instantaneous UE SINR samples

These outputs form the reference against which the mobile or optimized
deployment is compared.

### Mobile Relocation

When optimization is enabled, the code creates a second AP constellation with
the configured `num_mobile_aps` count, while keeping the `num_fixed_aps` APs
active in place.

- Time is partitioned into relocation windows using
  `optimization.relocation_interval_s`.
- In each window, the optimizer chooses a subset of candidate wall anchors.
- The chosen anchor positions are then assigned back to the original mobile AP
  IDs using minimum-distance matching, so AP identities stay stable even when
  positions move.
- The always-fixed APs are included in the AP-UE and AP-AP CSI solves, but they
  are excluded from the movable candidate pool.

The resulting AP schedule is exported to `mobile_ap_schedule.csv`.

### Objective Function

Each candidate subset is scored from CSI-derived terms only:

1. Instantaneous trajectory outage over all UE snapshots from `AP-UE` CSI.
2. A trajectory percentile term from the same `AP-UE` CSI.
3. A peer-aware tie-break term derived from `UE-UE` link quality.

`AP-AP` CSI is still exported for downstream analysis, but the placement score no
longer depends on a full radio map.

For a candidate subset, the code computes:

- `trajectory_outage`: fraction of UE snapshots with SINR below
  `sinr_threshold_db`
- `traj_p10`: 10th percentile SINR over UE trajectory samples

The peer-aware term uses `UE-UE` link powers to produce `need_weights`, so weak
peer connectivity increases the tie-break emphasis on the corresponding
trajectory samples.

The final score is:

```math
\mathrm{score}
=
- w_{\mathrm{out}} \,\mathrm{outage}
+ w_{p10}\, p_{10}
+ w_{\mathrm{peer}} \,\mathrm{peer\_tiebreak},
```

with weights:

- `outage_weight`
- `percentile_weight`
- `peer_tiebreak_weight`

Higher score is better.

### Search Strategy

The optimizer does not solve a continuous placement problem. It performs a
discrete search over the configured candidate IDs:

1. Greedy selection: add one candidate at a time, each time choosing the site
   that gives the best score improvement.
2. One-swap refinement: attempt single replacements of selected sites with
   unselected candidates and accept any improving swap.

This is implemented in `src/cocoon_sionna/optimization.py`.

### Evaluation Flow Per Scenario

With ray tracing enabled, a full scenario run proceeds as follows:

1. Build or load the scene and mobility graph.
2. Generate UE trajectories.
3. Compute `UE-UE` CSI and derive peer need weights.
4. Compute the fixed AP baseline:
   `AP-UE`, `AP-AP`, and optional coverage.
5. If optimization is enabled, optimize the mobile AP constellation over the
   relocation windows using CSI-derived scoring; otherwise reuse the fixed AP constellation.
6. Export fixed/mobile CSI, optional coverage maps, SINR comparisons, AP schedules, and
   summary metrics.

When `solver.enable_ray_tracing: false`, the pipeline skips CSI, coverage, and
optimization scoring, but it still exports trajectories, AP layouts, schedules,
and scene visualizations. When `coverage.enabled: false`, the pipeline still
runs CSI and optimization but skips coverage-map computation and exports.

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

Build and run the bundled validation scenario:

```powershell
.venv\Scripts\python.exe -m cocoon_sionna.cli run scenarios/etoile_demo.yaml
```

Enable more verbose progress logging:

```powershell
.venv\Scripts\python.exe -m cocoon_sionna.cli --log-level DEBUG run scenarios/etoile_demo.yaml
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

Run the full Rabot optimization pipeline:

```powershell
.venv\Scripts\python.exe -m cocoon_sionna.cli run scenarios/rabot.yaml
```

For faster optimization runs with less disk I/O, you can disable CSI artifact
writes and cache storage in the scenario YAML:

```yaml
outputs:
  write_csi_exports: false
  enable_csi_cache: false
```

## Outputs

Each scenario writes to its configured output directory and produces:

- `recommended_aps.csv`
- `fixed_aps.csv`
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
- `mobile_ap_schedule.csv`
- `run.log`

When `outputs.write_csi_exports: false`, the pipeline skips
`peer_csi_snapshots.npz` and `infra_csi_snapshots.npz` and also avoids the full
CFR/CIR/tau export path. When `outputs.enable_csi_cache: false`, the run does
not read or write `.csi_cache`.

When `solver.enable_ray_tracing: true` and `coverage.enabled: true`, the output
directory also includes:

- `coverage_map.npz`
- `coverage_map.png`
- `fixed_coverage_map.npz`
- `fixed_coverage_map.png`

`infra_csi_snapshots.npz` contains fixed/mobile `AP-AP` and fixed/mobile `AP-UE` CSI exports.
`peer_csi_snapshots.npz` contains `UE-UE` CSI exports and the peer-derived
weighting used to bias candidate placement evaluation.
`scene_render.png` is a rendered view of the loaded Sionna scene using
`scene.render(...)`, with the selected APs and the active UE snapshot shown as
radio devices inside the 3D environment.
`scene_camera.mp4` is a rendered 3D camera video of the loaded Sionna scene,
using a fixed oblique camera while the UE devices move over time. This output
requires `ffmpeg`.
`scene_layout.png` shows a top-down view of the loaded scene, including the walk
graph, building footprints when available, candidate APs, selected APs, and UE
trajectories.
`scene_animation.mp4` animates the moving UEs over the loaded scene together
with the AP placements. If `ffmpeg` is unavailable, the pipeline writes
`scene_animation.gif` instead. Playback speed is controlled via
`outputs.scene_animation_speedup` in the scenario YAML, so you can export the
animation faster than real time, e.g. `10.0` for `10x`.
`mobile_ap_schedule.csv` stores the mobile AP positions per relocation window.
You can turn that schedule into a dedicated animation with:

```powershell
.venv\Scripts\python.exe scripts\visualize_mobile_ap_schedule.py scenarios/rabot.yaml
```

This writes `mobile_ap_schedule_animation.mp4` to the scenario output
directory, or `mobile_ap_schedule_animation.gif` if `ffmpeg` is unavailable.
The script reuses the scene background, overlays the mobile AP movement, and
adds UE motion when `trajectory.csv` is present.
`trajectory_colormap.png` shows the full UE trajectories in a single PNG, with a
colormap over time so the motion direction can be followed visually. Initial UE
seeding is spread across the walk graph, and route selection biases toward
underexplored walkable edges to improve overall area coverage.
`user_sinr_cdf.png` compares the CDF of instantaneous distributed-MIMO zero-forcing
SINR samples across all user snapshots for a fixed static AP baseline versus the
optimized AP placement.
`summary.json` now also records the selected compute backend and Mitsuba variant.

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
- The baseline deployment is built from `num_fixed_aps` always-fixed APs plus
  `num_mobile_aps` movable seed APs. If `num_mobile_aps` is omitted, the code
  falls back to `num_selected_aps`.
- v1 is outdoor-only and ignores vegetation, weather, traffic, and indoor areas.
- The Rabot boundary and AP site files are seed inputs and can be tightened
  later without changing code.
