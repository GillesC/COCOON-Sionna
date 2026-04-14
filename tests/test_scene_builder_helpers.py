from shapely.geometry import Polygon

from cocoon_sionna.geo import osm_export_bbox_polygon
from cocoon_sionna.mesh import build_ground_mesh, build_roof_mesh, build_wall_mesh


def test_mesh_builders_emit_faces():
    polygon = Polygon([(0, 0), (4, 0), (4, 3), (0, 3), (0, 0)])
    ground = build_ground_mesh(polygon)
    roof = build_roof_mesh(polygon, 5.0)
    wall = build_wall_mesh(polygon, 5.0)

    assert len(ground.faces) > 0
    assert len(roof.faces) > 0
    assert len(wall.faces) > 0


def test_osm_export_bbox_polygon_matches_expected_order():
    polygon = osm_export_bbox_polygon(3.706070, 51.058969, 3.712513, 51.060592)
    assert list(polygon.exterior.coords) == [
        (3.706070, 51.058969),
        (3.712513, 51.058969),
        (3.712513, 51.060592),
        (3.706070, 51.060592),
        (3.706070, 51.058969),
    ]
