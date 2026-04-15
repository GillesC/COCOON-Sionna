import argparse
import json
from pathlib import Path

from cocoon_sionna.postprocess import resolve_output_dir_argument, run_manuscript_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full manuscript-oriented postprocessing bundle for a scenario YAML or output directory"
    )
    parser.add_argument("target", help="Scenario YAML path or scenario output directory")
    parser.add_argument("--analysis-dir", type=Path, default=None, help="Directory where the postprocessing files should be written")
    parser.add_argument("--threshold-min-db", type=float, default=-10.0)
    parser.add_argument("--threshold-max-db", type=float, default=20.0)
    parser.add_argument("--threshold-step-db", type=float, default=1.0)
    parser.add_argument("--outage-threshold-db", type=float, default=0.0)
    args = parser.parse_args()

    output_dir = resolve_output_dir_argument(args.target)
    artifacts = run_manuscript_report(
        output_dir=output_dir,
        analysis_dir=args.analysis_dir,
        threshold_min_db=args.threshold_min_db,
        threshold_max_db=args.threshold_max_db,
        threshold_step_db=args.threshold_step_db,
        outage_threshold_db=args.outage_threshold_db,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                **{key: str(value) for key, value in artifacts.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
