"""Scene building for OSM-derived outdoor environments."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
import shutil

from shapely.geometry import Polygon

from .config import SceneConfig
from .geo import LocalFrame, load_geojson_polygon, osm_export_bbox_polygon
from .mesh import build_ground_mesh, build_roof_mesh, build_wall_mesh, write_ascii_ply
from .osm import OverpassClient, extract_buildings, extract_walk_graph

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SceneArtifacts:
    scene_xml_path: Path
    metadata_path: Path | None
    walk_graph_path: Path | None


def _scene_xml(
    scene_cfg: SceneConfig,
    wall_files: list[tuple[str, str]],
    roof_files: list[tuple[str, str]],
) -> str:
    material_ids = {
        "ground": "mat_ground",
        "wall": "mat_wall",
        "roof": "mat_roof",
    }
    lines = [
        '<scene version="2.1.0">',
        "",
        "<!-- Materials -->",
        "",
        f'\t<bsdf type="itu-radio-material" id="{material_ids["ground"]}">',
        f'\t\t<string name="type" value="{scene_cfg.material_ground}"/>',
        '\t\t<float name="thickness" value="0.1"/>',
        "\t</bsdf>",
        f'\t<bsdf type="itu-radio-material" id="{material_ids["wall"]}">',
        f'\t\t<string name="type" value="{scene_cfg.material_wall}"/>',
        '\t\t<float name="thickness" value="0.1"/>',
        "\t</bsdf>",
        f'\t<bsdf type="itu-radio-material" id="{material_ids["roof"]}">',
        f'\t\t<string name="type" value="{scene_cfg.material_roof}"/>',
        '\t\t<float name="thickness" value="0.1"/>',
        "\t</bsdf>",
        "",
        "<!-- Shapes -->",
        "",
        '\t<shape type="ply" id="mesh-ground">',
        '\t\t<string name="filename" value="meshes/ground.ply"/>',
        '\t\t<boolean name="face_normals" value="true"/>',
        f'\t\t<ref id="{material_ids["ground"]}" name="bsdf"/>',
        "\t</shape>",
    ]
    for mesh_id, filename in wall_files:
        lines.extend(
            [
                f'\t<shape type="ply" id="mesh-{mesh_id}-wall">',
                f'\t\t<string name="filename" value="meshes/{filename}"/>',
                '\t\t<boolean name="face_normals" value="true"/>',
                f'\t\t<ref id="{material_ids["wall"]}" name="bsdf"/>',
                "\t</shape>",
            ]
        )
    for mesh_id, filename in roof_files:
        lines.extend(
            [
                f'\t<shape type="ply" id="mesh-{mesh_id}-roof">',
                f'\t\t<string name="filename" value="meshes/{filename}"/>',
                '\t\t<boolean name="face_normals" value="true"/>',
                f'\t\t<ref id="{material_ids["roof"]}" name="bsdf"/>',
                "\t</shape>",
            ]
        )
    lines.append("</scene>")
    return "\n".join(lines) + "\n"


class OSMSceneBuilder:
    def __init__(self, scene_cfg: SceneConfig) -> None:
        self.scene_cfg = scene_cfg

    def build(self) -> SceneArtifacts:
        if self.scene_cfg.boundary_path is None and self.scene_cfg.boundary_bbox is None:
            raise ValueError("OSM scene build requires scene.boundary_path or scene.boundary_bbox")
        if self.scene_cfg.scene_output_dir is None:
            raise ValueError("OSM scene build requires scene.scene_output_dir")

        output_dir = self.scene_cfg.scene_output_dir
        mesh_dir = output_dir / "meshes"
        output_dir.mkdir(parents=True, exist_ok=True)
        if mesh_dir.exists():
            shutil.rmtree(mesh_dir)
        mesh_dir.mkdir(parents=True, exist_ok=True)
        if self.scene_cfg.boundary_bbox is not None:
            boundary_source = {
                "west": self.scene_cfg.boundary_bbox[0],
                "south": self.scene_cfg.boundary_bbox[1],
                "east": self.scene_cfg.boundary_bbox[2],
                "north": self.scene_cfg.boundary_bbox[3],
            }
            boundary_lonlat = osm_export_bbox_polygon(*self.scene_cfg.boundary_bbox)
        else:
            assert self.scene_cfg.boundary_path is not None
            boundary_source = str(self.scene_cfg.boundary_path)
            boundary_lonlat = load_geojson_polygon(self.scene_cfg.boundary_path)
        logger.info("Building OSM scene from %s into %s", boundary_source, output_dir)

        centroid = boundary_lonlat.centroid
        frame = LocalFrame.from_lonlat(centroid.x, centroid.y)
        boundary_local = Polygon(
            [frame.lonlat_to_local_xy(x, y) for x, y in boundary_lonlat.exterior.coords]
        )

        parsed = OverpassClient(self.scene_cfg.overpass_url).fetch(boundary_lonlat)
        buildings = extract_buildings(
            parsed,
            boundary_lonlat=boundary_lonlat,
            frame=frame,
            default_height_m=self.scene_cfg.default_building_height_m,
            meters_per_level=self.scene_cfg.building_height_per_level_m,
        )
        if not buildings:
            raise RuntimeError("No buildings were extracted from the configured OSM boundary")
        logger.info("Extracted %d buildings from OSM", len(buildings))

        write_ascii_ply(mesh_dir / "ground.ply", build_ground_mesh(boundary_local))

        wall_files: list[tuple[str, str]] = []
        roof_files: list[tuple[str, str]] = []
        for building in buildings:
            wall_name = f"{building.name}-wall.ply"
            roof_name = f"{building.name}-roof.ply"
            write_ascii_ply(mesh_dir / wall_name, build_wall_mesh(building.polygon_local, building.height_m))
            write_ascii_ply(mesh_dir / roof_name, build_roof_mesh(building.polygon_local, building.height_m))
            wall_files.append((building.name, wall_name))
            roof_files.append((building.name, roof_name))

        graph = extract_walk_graph(
            parsed=parsed,
            boundary_local=boundary_local,
            frame=frame,
            building_polygons_local=[building.polygon_local for building in buildings],
            entry_buffer_m=5.0,
        )
        logger.info(
            "Derived walk graph with %d nodes and %d edges",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
        walk_graph_path = output_dir / "walk_graph.json"
        with walk_graph_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "nodes": [
                        {
                            "id": int(node_id),
                            "x": float(attrs["x"]),
                            "y": float(attrs["y"]),
                            "entry_candidate": bool(attrs.get("entry_candidate", False)),
                        }
                        for node_id, attrs in graph.nodes(data=True)
                    ],
                    "edges": [
                        {"u": int(u), "v": int(v), "length": float(attrs["length"])}
                        for u, v, attrs in graph.edges(data=True)
                    ],
                },
                handle,
                indent=2,
            )

        scene_xml_path = output_dir / "scene.xml"
        scene_xml_path.write_text(_scene_xml(self.scene_cfg, wall_files, roof_files), encoding="ascii")

        metadata_path = output_dir / "scene_metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "origin_lon": frame.origin_lon,
                    "origin_lat": frame.origin_lat,
                    "epsg": frame.epsg,
                    "boundary_local": [list(coord) for coord in boundary_local.exterior.coords],
                    "walk_graph_path": str(walk_graph_path),
                    "buildings": [
                        {
                            "name": building.name,
                            "height_m": building.height_m,
                            "polygon_local": [list(coord) for coord in building.polygon_local.exterior.coords],
                        }
                        for building in buildings
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Scene assets written: %s", scene_xml_path)
        return SceneArtifacts(
            scene_xml_path=scene_xml_path,
            metadata_path=metadata_path,
            walk_graph_path=walk_graph_path,
        )
