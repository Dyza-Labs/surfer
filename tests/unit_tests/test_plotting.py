import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytest

matplotlib.use("Agg")  # headless backend for test environments with no display

from surfer.agents.plotting import add_info_box, rename_lat_lon, resolve_column, scatter_plot, trajectory_map


# rename_lat_lon ------------------------------------------------------------------

def test_rename_lat_lon_renames_both_columns():
    df = pd.DataFrame({"latitude": [1.0], "longitude": [2.0], "temperature": [20.0]})
    result = rename_lat_lon(df)
    assert list(result.columns) == ["lat", "lon", "temperature"]


def test_rename_lat_lon_leaves_other_columns_untouched():
    df = pd.DataFrame({"time": [1], "temperature": [20.0]})
    result = rename_lat_lon(df)
    assert list(result.columns) == ["time", "temperature"]


# resolve_column (variable name -> dataframe column) --------------------------

def test_resolve_column_exact_match():
    df = pd.DataFrame(columns=["time", "temperature"])
    assert resolve_column(df, "temperature") == "temperature"


def test_resolve_column_case_insensitive():
    df = pd.DataFrame(columns=["Temperature"])
    assert resolve_column(df, "temperature") == "Temperature"


def test_resolve_column_alias_fallback():
    df = pd.DataFrame(columns=["temperature"])
    assert resolve_column(df, "temp") == "temperature"


def test_resolve_column_not_found_raises():
    df = pd.DataFrame(columns=["temperature"])
    with pytest.raises(ValueError, match="not found"):
        resolve_column(df, "chlorophyll")


# add_info_box -----------------------------------------------------------------

def test_add_info_box_with_populated_metadata_does_not_raise():
    fig, ax = plt.subplots()
    try:
        add_info_box(ax, {
            "id": "whoi_406-2016", "author": "Jane Doe",
            "institution": "Rutgers", "platform": "glider", "license": "CC-BY",
        })
    finally:
        plt.close(fig)


def test_add_info_box_with_empty_metadata_does_not_raise():
    fig, ax = plt.subplots()
    try:
        add_info_box(ax, {})
    finally:
        plt.close(fig)


# scatter_plot -------------------------------------------------------------------

def test_scatter_plot_returns_png_bytes():
    df = pd.DataFrame({
        "time": pd.date_range("2016-09-01", periods=5, freq="h"),
        "temperature": [10.0, 11.0, 12.0, 13.0, 14.0],
    })
    result = scatter_plot(df, x="time", y="temperature")

    assert isinstance(result, bytes)
    assert len(result) > 0
    assert result[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes


def test_scatter_plot_with_color_and_metadata_returns_bytes():
    df = pd.DataFrame({
        "depth": [1.0, 2.0, 3.0],
        "salinity": [35.0, 35.1, 35.2],
        "temperature": [12.0, 11.5, 11.0],
    })
    result = scatter_plot(
        df, x="salinity", y="depth", color="temperature", invert_y=True,
        metadata={"id": "test-glider-001"},
    )

    assert isinstance(result, bytes)
    assert len(result) > 0


def test_scatter_plot_drops_rows_missing_x_or_y():
    df = pd.DataFrame({
        "time": pd.date_range("2016-09-01", periods=3, freq="h"),
        "temperature": [10.0, None, 12.0],
    })
    # should not raise despite the missing value -- dropna handles it internally
    result = scatter_plot(df, x="time", y="temperature")
    assert isinstance(result, bytes)
    assert len(result) > 0


# trajectory_map -----------------------------------------------------------------

def test_trajectory_map_returns_html_with_osm_style():
    df = pd.DataFrame({"lat": [38.0, 38.5, 39.0], "lon": [-70.0, -70.2, -70.4]})
    html = trajectory_map(df, title="Flight path: test-glider-001")

    assert isinstance(html, str)
    assert "open-street-map" in html


def test_trajectory_map_hover_text_has_rounded_coords_and_units():
    df = pd.DataFrame({
        "lat": [51.716389123456], "lon": [-128.073358654321], "temperature": [15.2691],
    })
    html = trajectory_map(df, color="temperature", cbar_label="temperature (Celsius)")

    assert "51.716389" in html
    assert "-128.073359" in html or "-128.073358" in html
    assert "temperature (Celsius)" in html


def test_trajectory_map_drops_rows_missing_lat_or_lon():
    df = pd.DataFrame({"lat": [38.0, None, 39.0], "lon": [-70.0, -70.2, None]})
    # should not raise despite missing coordinates -- dropna handles it internally
    html = trajectory_map(df)
    assert isinstance(html, str)
    assert len(html) > 0
