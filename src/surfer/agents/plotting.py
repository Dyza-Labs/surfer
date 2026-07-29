"""Shared plotting functions for the erddap and thredds agents.

Operates on plain pandas DataFrames and metadata dicts -- no erddapy/netCDF assumptions.
Callers do the platform-specific work of producing a DataFrame first.
"""

import io
import math
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless backend -- no display in a server/container environment
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

_VAR_ALIASES = {
    "temp": "temperature",
    "salt": "salinity",
    "chl": "chlorophyll",
    "lat": "lat", "latitude": "lat",
    "lon": "lon", "longitude": "lon",
    "oxygen": "dissolved_oxygen",
    "cond": "conductivity",
    "turb": "turbidity",
}


def rename_lat_lon(df: pd.DataFrame) -> pd.DataFrame:
    """Aliases latitude/longitude columns to the short lat/lon names the rest of
    plotting.py expects. Shared by both agents' DataFrame-prep steps."""
    return df.rename(columns={"latitude": "lat", "longitude": "lon"})


def resolve_column(df: pd.DataFrame, name: str) -> str:
    """Match a variable name to a dataframe column: exact, then case-insensitive, then a
    known alias (e.g. 'temp' -> 'temperature'). A fallback for minor drift, not fuzzy search."""
    if name in df.columns:
        return name
    col_map = {c.lower(): c for c in df.columns}
    lowered = name.lower()
    if lowered in col_map:
        return col_map[lowered]
    alias = _VAR_ALIASES.get(lowered)
    if alias and alias in col_map:
        return col_map[alias]
    raise ValueError(f"Variable '{name}' not found. Available columns: {sorted(df.columns)}")


def add_info_box(ax, metadata: dict) -> None:
    """Adds a dataset metadata box (id, author, institution, platform, license) to a matplotlib axis."""
    info = (
        f"Dataset ID:  {metadata.get('id', 'N/A')}\n"
        f"Author(s):      {str(metadata.get('author', 'N/A'))[:35]}\n"
        f"Institution: {str(metadata.get('institution', 'N/A'))[:35]}\n"
        f"Platform:    {metadata.get('platform', 'N/A')}\n"
        f"License:     {metadata.get('license', 'N/A')}\n"
    )
    ax.text(
        1.13, 1.0, info,
        transform=ax.transAxes,
        fontsize=6,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8, edgecolor="gray"),
    )


def scatter_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    step_size: int = 1,
    invert_y: bool = False,
    title: Optional[str] = None,
    cbar_label: Optional[str] = None,
    metadata: Optional[dict] = None,
    filetype: str = "png",
) -> bytes:
    """Generic 2D scatter, optionally colored by a third variable -- used for timeseries,
    profile, and custom x-vs-y plots alike. Drops rows missing x/y, thins by step_size,
    and returns the rendered image as in-memory bytes."""
    df = df.dropna(subset=[x, y]).iloc[::step_size]
    if x == "time" or pd.api.types.is_datetime64_any_dtype(df[x]):
        df = df.copy()
        df[x] = pd.to_datetime(df[x])

    fig, ax = plt.subplots(figsize=(17, 5))
    if color:
        cs = ax.scatter(df[x], df[y], c=df[color], s=15, marker="o", edgecolor="none", cmap="viridis")
        cbar = fig.colorbar(cs, orientation="vertical", extend="both")
        cbar.ax.set_ylabel(cbar_label or color)
    else:
        ax.scatter(df[x], df[y], s=15, marker="o", edgecolor="none", cmap="viridis")

    if invert_y:
        ax.invert_yaxis()
    ax.set_xlim(df[x].min(), df[x].max())
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if title:
        ax.set_title(title, fontsize=11)

    if x == "time" or pd.api.types.is_datetime64_any_dtype(df[x]):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n%d-%b"))

    if metadata:
        add_info_box(ax, metadata)

    buf = io.BytesIO()
    plt.savefig(buf, format=filetype, dpi=300)
    plt.close(fig)
    return buf.getvalue()


def trajectory_map(
    df: pd.DataFrame,
    lat: str = "lat",
    lon: str = "lon",
    color: Optional[str] = None,
    cbar_label: Optional[str] = None,
    title: str = "Flight Path",
) -> str:
    """Plotly trajectory map over an OpenStreetMap basemap, optionally colored by a third
    variable. Returns a self-contained interactive HTML string."""
    df = df.dropna(subset=[lat, lon])

    marker: dict[str, object] = dict(size=6)
    if color:
        marker.update(color=df[color], colorscale="Plasma", showscale=True, colorbar=dict(title=cbar_label or color))

    hover_parts = [
        "(" + df[lat].round(6).astype(str) + ", " + df[lon].round(6).astype(str) + ")"
    ]
    if "time" in df.columns:
        hover_parts.append(pd.to_datetime(df["time"]).astype(str))
    if color:
        hover_parts.append(f"{cbar_label or color}: " + df[color].round(2).astype(str))
    hover_text = hover_parts[0]
    for part in hover_parts[1:]:
        hover_text = hover_text + "<br>" + part

    fig = go.Figure()
    fig.add_trace(
        go.Scattermapbox(
            lat=df[lat],
            lon=df[lon],
            mode="markers",
            marker=marker,
            text=hover_text,
            hoverinfo="text",
        )
    )
    fig.update_layout(
        title=title,
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=df[lat].mean(), lon=df[lon].mean()),
            zoom=5,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=700,
    )
    return fig.to_html(include_plotlyjs="cdn")

def multi_trajectory_map(
    tracks: dict[str, pd.DataFrame],
    lat: str = "lat",
    lon: str = "lon",
    title: str = "Datasets in region",
) -> str:
    """Overlay many datasets' positions on one basemap -- one trace per dataset, each with
    its own color and legend entry. Same HTML contract as trajectory_map."""

    palette = px.colors.qualitative.Plotly

    fig = go.Figure()
    all_lats, all_lons = [], []

    for i, (dataset_id, df) in enumerate(tracks.items()):
        df = df.dropna(subset=[lat, lon])
        if df.empty:
            continue
        all_lats.append(df[lat].astype(float))
        all_lons.append(df[lon].astype(float))
        fig.add_trace(
            go.Scattermapbox(
                lat=df[lat],
                lon=df[lon],
                mode="markers",
                marker=dict(size=5, color=palette[i % len(palette)]),
                name=dataset_id,           # legend label; color auto-assigned per trace
                subplot="mapbox",          # pin every trace to the ONE shared map (layout.mapbox)
                hovertemplate=f"{dataset_id}<br>(%{{lat:.4f}}, %{{lon:.4f}})<extra></extra>",
            )
        )

    if not all_lats:
        raise ValueError("No plottable positions in any dataset.")

    combined_lat = pd.concat(all_lats)
    combined_lon = pd.concat(all_lons)

    # Auto-fit zoom to the combined extent: a fixed zoom + mean-center crops datasets
    # that are far apart, leaving only one visible per view (looks like separate maps).
    span = max(
        combined_lat.max() - combined_lat.min(),
        (combined_lon.max() - combined_lon.min()) * 0.7,  # rough lon->lat aspect
        0.01,
    )
    zoom = min(max(math.log2(360 / span) - 1.0, 1), 12)

    fig.update_layout(
        title=title,
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=combined_lat.mean(), lon=combined_lon.mean()),
            zoom=zoom,
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=700,
    )
    return fig.to_html(include_plotlyjs="cdn")