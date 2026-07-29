"""
Retired: superseded by surfer.agents.plotting + the plot_timeseries/plot_profile/plot_map/
plot_custom tools in erddap_dataset_tools.py. Kept here instead of deleted outright, not
imported by anything active. The CLI-only entry points (getTimes/getCoords/
buildPlotfromTimes/buildPlotfromCoords) that took interactive input() were never ported --
the new tools use erddap_dataset_tools._apply_constraints instead.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
import webbrowser
import os
from surfer.agents.erddap_agent import ERDDAPConstraints
from erddapy import ERDDAP
from typing import Optional


"""
This is the graphing ENGINE. Here is a list of all of the functions and the things to-be-added:

1) ScatterPlot
    -Iputs ERDDAP object (ERDDAP), stepsize (int), and filetype (str)
    -Graphs time vs depth with temperature as color gradient.
    -Comes with infobox automatically containing metadata such as author, institution, project name

2) Salinity Plot
    -Iputs ERDDAP object (ERDDAP), stepsize (int), and filetype (str)
    -Graphs time and depth while salinity as color gradient.
    -Comes with infobox automatically containing metadata such as author, institution, project name

3) Coordinate Plot
    -Inputs ERDDAP object (ERDDAP), stepsize (int)                         
    -Returns an HTML graph file using plotly that shows the coordinates of a glider's trajectory

4) inititate
    -makes ERDDAP object a dataframe. embedded function so would never call on our own

5) initiateString
    -Honestly pointless. Ignore

6) addInfoBox / getDataLabels
    -embedded function built to add an info box to each slide from the meta-data.

7) getTimes + getCoords
    -given an ERDDAP object, allows user to input time or coordinate bounds which are then
    updated in the object constraints

8) buildPlotfromTimes / buildPlotfromCoords
    -graphing functions with coordinatePlot that take in database ID and calls getTimes to graph.
    -alternatively, you can just call coordinatePlot(getTimes(get_erddap('databseid)))

"""


def scatterPlot(f, step_size: int = 1, filetype: str = "png"):
    from surfer.agents.tools.erddap_dataset_tools import fix_labels, get_erddap

    facts = getDataLabels(f)
    df = initiate(f)
    df = fix_labels(df)
    # need to rename --> parsing error as the library had 'temperature' when physical column name of dataset was 'temperature (Celsius)
    # minor changes like these can completely skew the graph result. tread with caution!

    df = df.dropna(subset=["temperature"])  # drop null rows of temperature csv
    df = df.iloc[::step_size]

    df["time"] = pd.to_datetime(
        df["time"]
    )  # converts numerical time to actual readable time
    fig, ax = plt.subplots(figsize=(17, 5))  # sets size
    # kw = keywords, cs = color scatter
   #kw = dict(s=15, c=df["temperature"], marker="o", edgecolor="none", cmap="viridis")
    cs = ax.scatter(
        df["time"], 
        df["depth"], 
        s=15,
        c=df["temperature"],
        marker = "o",
        edgecolor="none",
        cmap = "viridis"
    )
    # first entry x, second y, **kw adds previous to cs

    ax.invert_yaxis()
    ax.set_xlim(df["time"].min(), df["time"].max())
    ax.set_ylabel("Depth (m)")

    ax.set_title(
        f"Ocean Temperature Profile\nProject: {facts.get('project', 'NA')}", fontsize=11
    )

    cbar = fig.colorbar(cs, orientation="vertical", extend="both")
    cbar.ax.set_ylabel("Temperature ($^\circ$C)")

    xfmt = mdates.DateFormatter("%H:%M\n%d-%b")
    ax.xaxis.set_major_formatter(xfmt)

    addInfoBox(ax, facts)
    plt.savefig(
        f"temperatureplot.{filetype}", format=f"{filetype}", dpi=300
    )  ## change for your file
    plt.show()
    plt.close()


def salinityPlot(f, step_size: int = 1, filetype: str = "png"):
    from surfer.agents.tools.erddap_dataset_tools import fix_labels, get_erddap
    # rename columns sans the labels for better formatting
    facts = getDataLabels(f)
    f = initiate(f)

    f = f.rename(
        columns={
            "time (UTC)": "time",
            "depth (m)": "depth",
            "temperature (Celsius)": "temperature",
            "salinity (1)": "salinity",
        }
    )
    f = f.dropna(subset=["salinity"])
    f = f.iloc[::step_size]
    f["time"] = pd.to_datetime(f["time"])
    # sets plot size
    fig, ax = plt.subplots(figsize=(17, 5))

    # dict(size of dots, color = data type, marker = marker shape, edgecolor = edgecolor color, cmap = color of the heat map distribution)
    #kw = dict(s=15, c=f["salinity"], marker="o", edgecolor="blue", cmap="viridis")
    # builds the scatter plot with x-type, y-type
    cs = ax.scatter(
        f["time"], 
        f["depth"], 
        s=15,
        c=f["salinity"],
        marker="o",
        edgecolor="none",
        cmap="viridis"
    )
    ax.invert_yaxis()
    ax.set_xlim(f["time"].min(), f["time"].max())
    ax.set_ylabel("Depth (m)")

    cbar = fig.colorbar(cs, orientation="vertical", extend="both")
    cbar.ax.set_ylabel("Salinity (psu)")

    xfmt = mdates.DateFormatter("%H:%M\n%d-%b")
    ax.xaxis.set_major_formatter(xfmt)
    addInfoBox(ax, facts)

    plt.savefig(f"salinityplot.{filetype}", format=f"{filetype}", dpi=300)
    plt.show()
    plt.close()


def coordinatePlot(f, step_size: int = 1):  # graph coords to temperature on a map            ##allow users to choose file type
    from surfer.agents.tools.erddap_dataset_tools import fix_labels, get_erddap

    f = initiate(f)
    df = fix_labels(f)

    df = df.dropna(subset=["temperature"])
    df["time"] = pd.to_datetime(df["time"])

    df = df.iloc[::step_size]

    formatted_time = df["time"].dt.strftime("%-H:%M<br>%-d-%b")

    hover_text = (
        "Time: "
        + formatted_time
        + "<br>"
        + "Temp:"
        + df["temperature"].round(2).astype(str)
        + "°C"
        + "       Depth:"
        + df["depth"].round(2).astype(str)
        + "(m)"
    )

    fig = go.Figure()  ##creates blank Plotly figure

    fig.add_trace(
        go.Scattermapbox(  # adds layer of data to figure
            lat=df["lat"],
            lon=df["lon"],
            mode="markers",  # dots-- not lines connecting them
            marker=dict(  # styles each dot
                size=6,
                color=df["temperature"],
                colorscale="Plasma",
                showscale=True,
                colorbar=dict(title="Temperature(°C)"),
            ),
            # round(2) --> rounds to 2 decimal
            text=hover_text,  # what you see when hovering over point
            hoverinfo="text+lat+lon",
        )
    )

    # zoom = np.interp(max_range, [0.1,0.5,1,5,10,40],[12,9,8,6,5,3])
    # loads the background, makes center, zoom

    # add depth as variable
    fig.update_layout(
        title="Flight Path (colored by Temperature)",
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=df["lat"].mean(), lon=df["lon"].mean()),
            zoom=5,
        ),
        # size of file outside of map
        margin=dict(l=0, r=0, t=40, b=0),
        height=700,
    )
    fig.write_html("glidertrajectory.html")
    webbrowser.open("file://" + os.path.abspath("glidertrajectory.html"))
    # fig.write_image('glidertrajectory.png',scale=3)
    fig.show()

_VAR_ALIASES = {  ##some placeholders to then check in the fix labels column
    "temp": "temperature",
    "salt": "salinity",
    "chl": "chlorophyll",
    "lat": "lat", "latitude": "lat",
    "lon": "lon", "longitude": "lon",
    "oxygen": "dissolved_oxygen",
    "cond": "conductivity",
    "turb": "turbidity",
}


# def resolve_column(df, var_name: str) -> str:
#     """
#     Matches a variable name (already resolved via resolve_dataset_variables_tool, or a lat/lon alias) against
#     a dataframe clenaed by fix_labels. Case-insensitive exact match.
#     """
#     col_map = {c.lower(): c for c in df.columns}
#     key = {"latitude": "lat", "longitude": "lon"}.get(var_name.lower(), var_name.lower())

#     if key in col_map:
#         return col_map[key]
#     raise ValueError(f"Variable '{var_name}' not found. Avaailable columns: {sorted(df.columns)}")



def initiate(g):  # initializes database
    return g.to_pandas().reset_index()


def initiateString(
    dataset_id, erddap_object
):  # optional helper function to assign glider databaseID to object
    erddap_object.dataset_id = dataset_id
    erddap_object.protocol = "tabledap"

    erddap_object.variables = [
        "time",
        "latitude",
        "longitude",
        "depth",
        "temperature",
        "salinity",
    ]
    return erddap_object.to_pandas().reset_index()


def addInfoBox(ax, attrs):

    info = (  # pulling from labels dictionary
        f"Dataset ID:  {attrs.get('id', 'N/A')}\n"
        f"Author(s):      {attrs.get('author', 'N/A')[:35]}\n"
        f"Institution: {attrs.get('institution', 'N/A')[:35]}\n"
        f"Platform:    {attrs.get('platform', 'N/A')}\n"
        f"License:     {attrs.get('license', 'N/A')}\n"
    )

    ax.text(
        1.13,
        1.0,
        info,
        transform=ax.transAxes,
        fontsize=6,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="lightyellow",
            alpha=0.8,
            edgecolor="gray",
        ),
    )


def getDataLabels(obj):
    current_id = obj.dataset_id
    info_url = obj.get_info_url(dataset_id=current_id, response="csv")
    info_df = pd.read_csv(info_url)

    attributes = info_df[info_df["Variable Name"] == "NC_GLOBAL"]

    def get_attr_value(attr_name, default="N/A"):
        match = attributes[
            attributes["Attribute Name"] == attr_name
        ]  # the specific value in attributes that we want
        return match["Value"].values[0] if not match.empty else default

    labels = {
        "id": current_id,
        "author": get_attr_value("contributor_name"),
        "institution": get_attr_value("institution"),
        "platform": get_attr_value("platform"),
        "project": get_attr_value("project"),
    }
    return labels


def getTimes(e: ERDDAP, time_bounds: Optional[ERDDAPConstraints] = None) -> ERDDAP:

    if e.constraints is None:
        raise ValueError(f"ERDDAP object for dataset {e.dataset_id} has no constraints")

    db_start = str(e.constraints["time>="]).strip("(),' ")
    db_end = str(e.constraints["time<="]).strip("(),' ")

    if time_bounds is not None:
        start = time_bounds.min_time or db_start
        end = time_bounds.max_time or db_end
    else:
        #CLI path for bounds
        print(f"\nDataset runs from {db_start} to {db_end}")

        start = input("Start date (YYYY-MM-DD) or press Enter for full range: ").strip()
        end = input("End date (YYYY-MM-DD) or press Enter for full range: ").strip()

        start = f"{start}T00:00:00Z" if start else db_start
        end = f"{end}T23:59:59Z" if end else db_end
    
    #add just in case
    if "T" not in start:
        start = f"{start}T00:00:00Z"
    if "T" not in end:
        end = f"{end}T23:59:59Z"
 
    if pd.to_datetime(start) > pd.to_datetime(end):
        print("Start is after end — using full dataset range.")
        return e
    if pd.to_datetime(start) < pd.to_datetime(db_start):
        print(f"Start clamped to dataset begin: {db_start}")
        start = db_start
    if pd.to_datetime(end) > pd.to_datetime(db_end):
        print(f"End clamped to dataset end: {db_end}")
        end = db_end
 
    e.constraints["time>="] = start
    e.constraints["time<="] = end
    return e
 



def getCoords(e: ERDDAP, coord_bounds: Optional[ERDDAPConstraints] = None) -> ERDDAP:
    """Apply coordinate bounds to an ERDDAP object.
 
    If coord_bounds is provided (agent path), applies it directly.
    If not (CLI path), prompts the user interactively via input().
    """

    if e.constraints is None:
        raise ValueError(f"ERDDAP object for dataset {e.dataset_id} has no constraints")
    
    db_lat_min = float(str(e.constraints["latitude>="]).strip("(),' "))
    db_lat_max = float(str(e.constraints["latitude<="]).strip("(),' "))
    db_lon_min = float(str(e.constraints["longitude>="]).strip("(),' "))
    db_lon_max = float(str(e.constraints["longitude<="]).strip("(),' "))
 
    if coord_bounds is not None:
        lat_min = coord_bounds.min_lat if coord_bounds.min_lat is not None else db_lat_min
        lat_max = coord_bounds.max_lat if coord_bounds.max_lat is not None else db_lat_max
        lon_min = coord_bounds.min_lon if coord_bounds.min_lon is not None else db_lon_min
        lon_max = coord_bounds.max_lon if coord_bounds.max_lon is not None else db_lon_max
    else:
        # CLI path — safe to call input() here
        print(f"\nGlider latitude goes from {db_lat_min}° to {db_lat_max}°")
        print(f"Glider longitude goes from {db_lon_min}° to {db_lon_max}°")
        lat_min = float(input("Starting latitude or Enter for full range: ").strip() or db_lat_min)
        lat_max = float(input("Ending latitude or Enter for full range: ").strip() or db_lat_max)
        lon_min = float(input("Starting longitude or Enter for full range: ").strip() or db_lon_min)
        lon_max = float(input("Ending longitude or Enter for full range: ").strip() or db_lon_max)
 
    # Swap if inverted
    if lat_min > lat_max:
        lat_min, lat_max = lat_max, lat_min
    if lon_min > lon_max:
        lon_min, lon_max = lon_max, lon_min
 
    # Clamp to dataset bounds
    e.constraints["latitude>="] = max(lat_min, db_lat_min)
    e.constraints["latitude<="] = min(lat_max, db_lat_max)
    e.constraints["longitude>="] = max(lon_min, db_lon_min)
    e.constraints["longitude<="] = min(lon_max, db_lon_max)
    return e


def buildPlotfromTimes(dbid: str, server: str, time_bounds: Optional[ERDDAPConstraints] = None):  ##databaseid -- builds plot given time bounds and id
    from surfer.agents.tools.erddap_dataset_tools import get_erddap

    e = get_erddap(
        dataset_id=dbid,
        server=server,
        variables=["depth", "latitude", "longitude", "salinity",
                   "temperature", "time"],
    )
    coordinatePlot(getTimes(e,time_bounds=time_bounds))


def buildPlotfromCoords(dbid: str, server: str, coord_bounds: Optional[ERDDAPConstraints] = None):
    from surfer.agents.tools.erddap_dataset_tools import get_erddap

    e = get_erddap(
        dataset_id=dbid,
        server=server,
        variables=["depth", "latitude", "longitude", "salinity", "temperature", "time"],
    )
    coordinatePlot(getCoords(e,coord_bounds=coord_bounds))
