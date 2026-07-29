import operator
from typing import Optional, Literal, Annotated, Any

from langchain.agents import AgentState
from langchain.messages import ToolMessage
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel, Field, field_validator
from siphon.catalog import TDSCatalog


thredds_system_prompt = (
    "You help users browse, retrieve, and visualize environmental data from THREDDS servers. "
    "Be precise and transparent about what data was used; never fabricate dataset contents or variable names. "
    "Ask the user when genuinely unsure rather than guessing.\n"
    "- Given a new server URL, call register_server_tool first; call no other tool until it succeeds.\n"
    "- For a broad or exploratory question (e.g. 'what can you do with this server?'), call browse_catalog_tool "
    "ONCE at the current position and answer from that single result -- report the sub-catalog names and what "
    "they likely hold, then stop; a live fetch per hop is wasted on information the user didn't ask for. Only "
    "descend deeper when the user names a specific dataset, instrument, or sub-catalog.\n"
    "- THREDDS has no server-side search or categorize endpoints. Discovery is tree navigation: call "
    "browse_catalog_tool to walk down through sub-catalogs, or find_datasets_tool to search dataset *names* "
    "by substring across the whole subtree -- this is name matching only, not a metadata search. If the user "
    "asks for datasets by institution, standard_name, or keyword, say plainly that isn't possible here.\n"
    "- find_datasets_tool and browse_catalog_tool results give you two separate pieces of a dataset "
    "reference: the sub-catalog's catalog_url and the dataset's own filename. Every dataset_id-taking tool "
    "(describe_dataset_tool, plot_timeseries, plot_profile, plot_map, plot_custom, get_dataset_download_tool) "
    "takes catalog_url as dataset_id and the filename as a separate dataset_name argument -- pass both "
    "whenever the catalog holds more than one dataset (common for catalogs with many files, e.g. a "
    "'Gridded' folder). Never invent or hand-construct a dataset's OPeNDAP/download URL yourself.\n"
    "- Use set_active_catalog_tool to jump directly back to a known catalog URL (e.g. one already seen in a "
    "prior browse_catalog_tool or find_datasets_tool result) without walking the tree again.\n"
    "- Before plot_timeseries, plot_profile, plot_map, plot_custom, or get_dataset_download_tool, call "
    "describe_dataset_tool -- it gives the dataset's real coverage and maps a plain-language name (e.g. "
    "'temperature') to its exact variable name. If several variables plausibly match, ask the user.\n"
    "- Plot tool choice: plot_timeseries (vs. time), plot_profile (vs. depth), plot_map (trajectory colored by "
    "a variable), plot_custom (any other variable-vs-variable pair, e.g. temperature-salinity). All render "
    "locally -- there is no server-rendered image option here.\n"
    "- This agent supports trajectory/profile glider-style datasets only (cdm_data_type == 'TrajectoryProfile'). "
    "Gridded model output is not yet supported; describe_dataset_tool will say so plainly if a dataset turns "
    "out to be a Grid dataset."
)


# Schemas
# -------
class THREDDSServer(BaseModel):
    """Identifies a THREDDS server's top-level catalog. Stored in state, not passed per tool call."""
    url: str = Field(description="URL of a THREDDS catalog (catalog.xml or catalog.html)")

    @field_validator("url", mode="before")
    @classmethod
    def sanitize_url(cls, v: str) -> str:
        """Normalize any THREDDS catalog URL to its catalog.xml form."""
        v = v.split("?")[0].split("#")[0].rstrip("/")  # Strip query string and fragment
        if v.endswith("catalog.html"):
            return v[: -len("catalog.html")] + "catalog.xml"
        if not v.endswith((".xml", ".html")):
            return v + "/catalog.xml"
        return v


class THREDDSAgentState(AgentState):
    """State for the THREDDS agent. Persists across turns."""
    servers: Annotated[dict[str, dict], operator.or_]
    active_server_url: Optional[str]
    active_catalog_url: Optional[str]
    artifacts: Annotated[list[dict], operator.add]


class THREDDSConstraints(BaseModel):
    """Spatial and temporal constraints for subsetting THREDDS datasets. Unlike ERDDAP's
    equivalent, these are applied client-side via boolean masking on an already-open
    xr.Dataset, not baked into a server-side query string."""
    min_time: Optional[str] = Field(
        default=None,
        description="Start time in ISO 8601 format e.g. '2016-07-01T00:00:00Z'. Omit to use dataset start."
    )
    max_time: Optional[str] = Field(
        default=None,
        description="End time in ISO 8601 format e.g. '2017-02-01T00:00:00Z'. Omit to use dataset end."
    )
    min_lat: Optional[float] = Field(default=None, description="Minimum latitude in decimal degrees e.g. 38.0")
    max_lat: Optional[float] = Field(default=None, description="Maximum latitude in decimal degrees e.g. 41.0")
    min_lon: Optional[float] = Field(default=None, description="Minimum longitude in decimal degrees e.g. -72.0")
    max_lon: Optional[float] = Field(default=None, description="Maximum longitude in decimal degrees e.g. -69.0")


class THREDDSDataset(BaseModel):
    dataset_id: str = Field(
        description=(
            "Catalog URL of the sub-catalog listing the dataset, as returned by "
            "browse_catalog_tool or find_datasets_tool's catalog_url field."
        )
    )
    dataset_name: Optional[str] = Field(
        default=None,
        description=(
            "Exact dataset filename within that catalog (find_datasets_tool's name field), "
            "e.g. '20130111T000000_20130514T000000_challenger_ru29.nc'. Required whenever "
            "the catalog lists more than one dataset; omit only if dataset_id's catalog "
            "lists exactly one."
        ),
    )
    variables: Optional[list[str]] = Field(
        default=None,
        description=(
            "Exact variable names to load. "
            "Call describe_dataset_tool first to get exact names. "
            "Omit to load all variables."
        )
    )
    constraints: Optional[THREDDSConstraints] = Field(
        default=None,
        description="Spatial and temporal constraints to subset the data"
    )


class THREDDSPlotBase(BaseModel):
    """Shared fields for the plot tools in thredds_dataset_tools.py. Doesn't extend
    THREDDSDataset: plots derive their needed columns from x/y/color, not a raw variable list."""
    dataset_id: str = Field(
        description=(
            "Catalog URL of the sub-catalog listing the dataset, as returned by "
            "browse_catalog_tool or find_datasets_tool's catalog_url field."
        )
    )
    dataset_name: Optional[str] = Field(
        default=None,
        description=(
            "Exact dataset filename within that catalog (find_datasets_tool's name field), "
            "e.g. '20130111T000000_20130514T000000_challenger_ru29.nc'. Required whenever "
            "the catalog lists more than one dataset; omit only if dataset_id's catalog "
            "lists exactly one."
        ),
    )
    constraints: Optional[THREDDSConstraints] = Field(
        default=None,
        description="Spatial and temporal constraints to subset the data"
    )
    step_size: int = Field(
        default=1,
        description="Plot every Nth data point. Use 1 for all points. Increase to thin dense datasets.",
    )


class THREDDSTimeseriesPlot(THREDDSPlotBase):
    """A variable plotted against time."""
    y_var: str = Field(description="Variable name to plot against time, e.g. 'temperature'.")
    color_var: Optional[str] = Field(default=None, description="Optional variable to color points by, e.g. 'depth'.")
    file_type: Literal["png", "pdf"] = Field(default="png", description="Output file format")


class THREDDSProfilePlot(THREDDSPlotBase):
    """A variable plotted against depth."""
    x_var: str = Field(description="Variable name to plot against depth, e.g. 'temperature'.")
    color_var: Optional[str] = Field(default=None, description="Optional variable to color points by, e.g. 'time'.")
    file_type: Literal["png", "pdf"] = Field(default="png", description="Output file format")


class THREDDSMapPlot(THREDDSPlotBase):
    """A trajectory map of the dataset's latitude/longitude."""
    color_var: Optional[str] = Field(
        default=None, description="Optional variable to color the trajectory by, e.g. 'temperature'."
    )


class THREDDSCustomPlot(THREDDSPlotBase):
    """
    Scatter plot of any two variables against each other -- for requests that don't fit
    plot_timeseries (vs. time), plot_profile (vs. depth), or plot_map (lat/lon), e.g. a
    temperature-salinity diagram. If user input is akin to plot "variable" versus "other
    variable", assume "variable" is x_var, "other variable" is y_var.
    """
    x_var: str = Field(description="Variable name to plot on the x-axis, e.g. 'salinity'.")
    y_var: str = Field(default="time", description="Variable name to plot on the y-axis. Defaults to 'time' if none stated.")
    color_var: Optional[str] = Field(default=None, description="Optional variable to color points by, e.g. 'temperature'.")
    file_type: Literal["png", "pdf"] = Field(default="png", description="Output file format")


# State helpers
# -------------
def get_server(runtime: ToolRuntime, url: str) -> THREDDSServer:
    """Look up a registered server by URL. Raises if not registered."""
    url = THREDDSServer.sanitize_url(url)
    servers = runtime.state.get("servers", {})
    server_dict = servers.get(url)
    if not server_dict:
        raise ValueError(f"Server {url} not registered. Call register_server_tool first.")
    return THREDDSServer(**server_dict)


def get_active_server(runtime: ToolRuntime) -> THREDDSServer:
    """Get the currently active server. Raises if none set."""
    url = runtime.state.get("active_server_url")
    if not url:
        raise ValueError("No active server set. Call set_active_server_tool first.")
    return get_server(runtime, url)


def get_active_catalog_url(runtime: ToolRuntime) -> str:
    """Get the currently active catalog URL (the agent's current position in the tree). Raises if none set."""
    url = runtime.state.get("active_catalog_url")
    if not url:
        raise ValueError("No active catalog. Call register_server_tool first.")
    return url


def check_thredds_status(url: str) -> dict[str, Any]:
    """Check if a THREDDS catalog is reachable by fetching and parsing its catalog.xml.
    Returns {"reachable": bool, "catalog_name": str | None, "error": str | None}.
    catalog_name substitutes for ERDDAP's /version string as the human-readable
    confirmation -- THREDDS has no server-software-version concept the way ERDDAP does."""
    try:
        cat = TDSCatalog(url)
        return {"reachable": True, "catalog_name": cat.catalog_name, "error": None}
    except Exception as err:
        return {"reachable": False, "catalog_name": None, "error": str(err)}


# State tools
# -----------
@tool
def check_server_status_tool(url: str) -> str:
    """Check whether a THREDDS catalog is currently reachable. Use before
    browsing or requesting data if the server's availability is uncertain."""
    url = THREDDSServer.sanitize_url(url)
    result = check_thredds_status(url)
    if result["reachable"]:
        return f"The catalog at {url} is reachable: {result['catalog_name']}"
    else:
        return f"The catalog at {url} is not reachable. Error: {result['error']}"


@tool
def register_server_tool(runtime: ToolRuntime, url: str) -> Command:
    """Register a THREDDS server and check its status.
    Always call this first when given a new server URL before any other tool."""
    server = THREDDSServer(url=url)
    result = check_thredds_status(server.url)
    status_detail = (
        f"Catalog: {result['catalog_name']}."
        if result["reachable"]
        else f"Error: {result['error']}"
    )
    return Command(
        update={
            "servers": {server.url: server.model_dump()},
            "active_server_url": server.url,
            "active_catalog_url": server.url,
            "messages": [
                ToolMessage(
                    content=(
                        f"Server {server.url} registered and set as active. "
                        f"Reachable: {result['reachable']}. "
                        f"{status_detail}"
                    ),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
def set_active_server_tool(runtime: ToolRuntime, url: str) -> Command:
    """Set the active server to a previously registered URL, resetting the current
    catalog position back to that server's root. Use when switching between multiple servers."""
    server = get_server(runtime, url)
    return Command(
        update={
            "active_server_url": server.url,
            "active_catalog_url": server.url,
            "messages": [
                ToolMessage(
                    content=f"Active server switched to {server.url}.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
def set_active_catalog_tool(runtime: ToolRuntime, url: str) -> str | Command:
    """Jump directly to a known catalog URL (e.g. one already seen in a prior
    browse_catalog_tool or find_datasets_tool result) without walking the tree again."""
    get_active_server(runtime)  # Raises if no server registered yet
    try:
        TDSCatalog(url)
    except Exception as err:
        return (
            f"Could not fetch catalog at '{url}': {err}. Use a catalog_url exactly as "
            f"returned by browse_catalog_tool or find_datasets_tool rather than constructing "
            f"one -- URL path segments don't always match the display names shown."
        )
    return Command(
        update={
            "active_catalog_url": url,
            "messages": [
                ToolMessage(
                    content=f"Active catalog switched to {url}.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )
