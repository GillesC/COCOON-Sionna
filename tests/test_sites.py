from cocoon_sionna.sites import generate_wall_candidate_sites, select_farthest_sites


def test_generate_wall_candidate_sites_places_facade_points():
    metadata = {
        "buildings": [
            {
                "name": "block_a",
                "polygon_local": [[0.0, 0.0], [20.0, 0.0], [20.0, 10.0], [0.0, 10.0], [0.0, 0.0]],
            }
        ]
    }

    sites = generate_wall_candidate_sites(
        metadata=metadata,
        spacing_m=8.0,
        mount_height_m=7.0,
        corner_clearance_m=2.0,
        mount_offset_m=0.25,
        min_spacing_m=4.0,
    )

    assert sites
    assert all(site.mount_type == "facade" for site in sites)
    assert all(site.source == "wall_metadata" for site in sites)
    assert all(abs(site.z_m - 7.0) < 1e-9 for site in sites)


def test_select_farthest_sites_prefers_spread():
    metadata = {
        "buildings": [
            {
                "name": "block_a",
                "polygon_local": [[0.0, 0.0], [30.0, 0.0], [30.0, 10.0], [0.0, 10.0], [0.0, 0.0]],
            }
        ]
    }
    sites = generate_wall_candidate_sites(
        metadata=metadata,
        spacing_m=6.0,
        mount_height_m=7.0,
        corner_clearance_m=1.0,
        mount_offset_m=0.25,
        min_spacing_m=2.0,
    )

    selected = select_farthest_sites(sites, 3)

    assert len(selected) == 3
    assert len({site.site_id for site in selected}) == 3


def test_generate_wall_candidate_sites_skips_boundary_facing_candidates():
    metadata = {
        "boundary_local": [[0.0, 0.0], [40.0, 0.0], [40.0, 30.0], [0.0, 30.0], [0.0, 0.0]],
        "buildings": [
            {
                "name": "edge_block",
                "polygon_local": [[10.0, 24.0], [20.0, 24.0], [20.0, 29.0], [10.0, 29.0], [10.0, 24.0]],
            }
        ],
    }
    unfiltered_sites = generate_wall_candidate_sites(
        metadata={"buildings": metadata["buildings"]},
        spacing_m=6.0,
        mount_height_m=7.0,
        corner_clearance_m=1.0,
        mount_offset_m=0.25,
        min_spacing_m=2.0,
    )
    filtered_sites = generate_wall_candidate_sites(
        metadata=metadata,
        spacing_m=6.0,
        mount_height_m=7.0,
        corner_clearance_m=1.0,
        mount_offset_m=0.25,
        min_spacing_m=2.0,
    )

    assert any(site.y_m > 28.5 and 45.0 <= site.yaw_deg <= 135.0 for site in unfiltered_sites)
    assert filtered_sites
    assert not any(site.y_m > 28.5 and 45.0 <= site.yaw_deg <= 135.0 for site in filtered_sites)
