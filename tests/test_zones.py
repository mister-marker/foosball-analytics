"""
tests/test_zones.py

Unit tests for ZoneMapper.

Run with:
    pytest tests/test_zones.py -v
"""

import pytest
from core.zones import ZoneMapper


# ---------------------------------------------------------------------------
# Fallback (hardcoded) mode
# ---------------------------------------------------------------------------

class TestZoneMapperFallback:
    """Tests for ZoneMapper when no config is supplied."""

    def setup_method(self) -> None:
        self.mapper = ZoneMapper()  # fallback zones

    def test_left_edge_is_goalkeeper_left(self) -> None:
        assert self.mapper.get_zone(rel_x=0.0, rel_y=0.5) == "goalkeeper_left"

    def test_right_edge_is_goalkeeper_right(self) -> None:
        assert self.mapper.get_zone(rel_x=1.0, rel_y=0.5) == "goalkeeper_right"

    def test_centre_is_midfield(self) -> None:
        assert self.mapper.get_zone(rel_x=0.5, rel_y=0.5) == "midfield"

    def test_clamps_slightly_out_of_range(self) -> None:
        """Slightly negative or > 1.0 values should be clamped, not error."""
        zone = self.mapper.get_zone(rel_x=-0.01, rel_y=0.5)
        assert zone == "goalkeeper_left"


# ---------------------------------------------------------------------------
# Config-driven mode
# ---------------------------------------------------------------------------

class TestZoneMapperConfig:
    """Tests for ZoneMapper when a config dict is supplied."""

    def setup_method(self) -> None:
        config = {
            "left_half": {"x_min": 0.0, "x_max": 0.5},
            "right_half": {"x_min": 0.5, "x_max": 1.0},
        }
        self.mapper = ZoneMapper(zones_config=config)

    def test_left_half_correct(self) -> None:
        assert self.mapper.get_zone(rel_x=0.25, rel_y=0.5) == "left_half"

    def test_right_half_correct(self) -> None:
        assert self.mapper.get_zone(rel_x=0.75, rel_y=0.5) == "right_half"

    def test_missing_boundary_raises_value_error(self) -> None:
        bad_config = {"zone_a": {"x_min": 0.0}}  # x_max is missing
        with pytest.raises(ValueError, match="x_max"):
            ZoneMapper(zones_config=bad_config)


# ---------------------------------------------------------------------------
# Normalise helper
# ---------------------------------------------------------------------------

class TestNormalise:
    def test_centre_pixel(self) -> None:
        rel_x, rel_y = ZoneMapper.normalise(400, 250, 800, 500)
        assert rel_x == pytest.approx(0.5)
        assert rel_y == pytest.approx(0.5)

    def test_top_left_pixel(self) -> None:
        rel_x, rel_y = ZoneMapper.normalise(0, 0, 800, 500)
        assert rel_x == pytest.approx(0.0)
        assert rel_y == pytest.approx(0.0)

    def test_zero_frame_size_raises(self) -> None:
        with pytest.raises(ValueError):
            ZoneMapper.normalise(0, 0, 0, 500)
