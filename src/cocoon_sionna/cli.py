"""Command line entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_scenario_config
from .logging_utils import configure_logging
from .pipeline import build_scene_only, run_scenario


def _default_log_path(command: str, scenario_path: str, config) -> Path | None:
    if command == "run":
        return config.outputs.output_dir / "run.log"
    if command == "build-scene" and config.scene.scene_output_dir is not None:
        return config.scene.scene_output_dir / "build-scene.log"
    return Path(scenario_path).with_suffix(".log")


def main() -> None:
    parser = argparse.ArgumentParser(prog="cocoon-sionna")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Runtime log verbosity",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a full scenario")
    run_parser.add_argument("scenario", help="Path to a scenario YAML file")

    build_parser = subparsers.add_parser("build-scene", help="Build the OSM-derived scene assets only")
    build_parser.add_argument("scenario", help="Path to a scenario YAML file")

    args = parser.parse_args()
    config = load_scenario_config(args.scenario)
    configure_logging(args.log_level, _default_log_path(args.command, args.scenario, config))
    if args.command == "run":
        print(json.dumps(run_scenario(config), indent=2))
    elif args.command == "build-scene":
        artifacts = build_scene_only(config)
        print(
            json.dumps(
                {
                    "scene_xml_path": str(artifacts.scene_xml_path),
                    "metadata_path": str(artifacts.metadata_path) if artifacts.metadata_path else None,
                    "walk_graph_path": str(artifacts.walk_graph_path) if artifacts.walk_graph_path else None,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
