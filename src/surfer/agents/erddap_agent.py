import re
import operator
from typing import Optional, Literal, Annotated, Any
import requests

from langchain.agents import AgentState
from langchain.messages import ToolMessage
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel, Field, field_validator


erddap_system_prompt = (
    "You help users search, retrieve, and visualize environmental data from ERDDAP servers. "
    "Be precise and transparent about what data was used; never fabricate dataset contents or variable names. "
    "Ask the user when genuinely unsure rather than guessing.\n"
    "- Given a new server URL, call register_server_tool first; call no other tool until it succeeds.\n"
    "- Before plot_timeseries, plot_profile, plot_map, plot_custom, or get_dataset_download_tool, call "
    "describe_dataset_tool -- it gives the dataset's real coverage and maps a plain-language name (e.g. "
    "'temperature') to its exact variable name. If several variables plausibly match, ask the user.\n"
    "- Plot tool choice: plot_timeseries (vs. time), plot_profile (vs. depth), plot_map (trajectory colored by "
    "a variable), plot_custom (any other variable-vs-variable pair, e.g. temperature-salinity). plot_region_map "
    "instead maps ALL datasets on the active server with data in a lat/lon region on one combined map -- use it "
    "for 'what data exists here' questions rather than a single known dataset; it requires a full bounding box.\n"
    "- Before search_server_tool, call resolve_categorize_values_tool for any institution/ioos_category/"
    "standard_name/cdm_data_type filter -- these need exact raw values, which are often underscored/lowercased "
    "and won't match human-readable names shown elsewhere (e.g. get_server_summary_tool's institution list). "
    "If several values plausibly match, ask the user.\n"
    "- For a plain-language quantity (e.g. 'temperature', 'salinity'), resolve it to a CF standard_name first: "
    "resolve_categorize_values_tool(categorize_by='standard_name', query=<term>) (QC/status-flag values are "
    "pre-filtered out). Then: none found -> fall back to variable_name. One result -> search immediately, no "
    "confirmation. Multiple unit/convention variants of one quantity (e.g. sea_water_salinity vs. "
    "sea_water_practical_salinity) -> search the more common variant (prefer 'practical_'), don't ask. Multiple "
    "results differing in physical meaning (e.g. chlorophyll concentration vs. fluorescence) -> ask the user.\n"
    "- Plot tools render their output to the user automatically. Never offer links to view a plot, "
    "never ask whether the user wants to see it, and never describe visual details you have not observed."
)


# Schemas
# -------
class ERDDAPServer(BaseModel):
    """Identifies an ERDDAP server. Stored in state, not passed per tool call."""
    url: str = Field(description="URL of an ERDDAP server")
    protocol: Literal["tabledap", "griddap",] = Field(
        default="tabledap", description="Standard by which to request data"
    ) # We hardly use the wms protocol but should remember it exists
    response: Literal[
    "asc","croissant","csv","csv0","csvp",
    "das","dataTable","dds","dods",
    "esriCsv","fgdc","geoJson",
    "htmlTable","iso19115","iso19115_2",
    "iso19115_3_2016","iso19139_2007","itx",
    "json","jsonlCSV","jsonlCSV1","jsonlKVP","mat",
    "nc","ncCF","ncCFHeader","ncCFMA",
    "ncCFMAHeader","ncHeader","nccsv","nccsvMetadata",
    "ncoJson","odvTxt","parquet","parquetWMeta",
    "tsv","tsv0","tsvp","wav","xhtml",
] = Field(
        default="csv", description="Output format in which to receive data"
    )

    @field_validator("url", mode="before")
    @classmethod
    def sanitize_url(cls, v: str) -> str:
        """Normalize any ERDDAP URL to its base form, https://{base}/erddap/"""
        v = v.split("?")[0].split("#")[0]
        match = re.search(r"(https?://[^/]+/erddap)", v, re.IGNORECASE)
        if match:
            return match.group(1).rstrip("/")
        return v.rstrip("/").removesuffix("/index.html")


class ERDDAPAgentState(AgentState):
    """State for the ERDDAP agent. Persists across turns."""
    servers: Annotated[dict[str, dict], operator.or_]
    active_server_url: Optional[str]
    artifacts: Annotated[list[dict], operator.add]


class ERDDAPConstraints(BaseModel):
    """Spatial and temporal constraints for subsetting ERDDAP datasets.
    Maps to the kwargs dict in erddapy. Supports ISO 8601 format and relative time.
    Example dict: 
    constraints = {
        'time>=': '2016-09-02T00:00:00Z',
        'time<=': '2016-09-20T00:00:00Z',
        'latitude>=': 38.0,
        'latitude<=': 41.0,
        'longitude>=': -72.0,
        'longitude<=': -69.0
    }"""
    min_time: Optional[str] = Field(
        default=None,
        description="Start time in ISO 8601 format e.g. '2016-07-01T00:00:00Z' or relative e.g. 'now-7days'." \
        "Omit to use dataset start."
    )
    max_time: Optional[str] = Field(
        default=None,
        description="End time in ISO 8601 format e.g. '2017-02-01T00:00:00Z' or relative e.g. 'now'." \
        "Omit to use dataset end."
    )
    min_lat: Optional[float] = Field(default=None, description="Minimum latitude in decimal degrees e.g. 38.0")
    max_lat: Optional[float] = Field(default=None, description="Maximum latitude in decimal degrees e.g. 41.0")
    min_lon: Optional[float] = Field(default=None, description="Minimum longitude in decimal degrees e.g. -72.0")
    max_lon: Optional[float] = Field(default=None, description="Maximum longitude in decimal degrees e.g. -69.0")

    def to_kwargs(self) -> dict[str, Any]:
        """Convert to the erddapy constraints dict format."""
        c = {}
        if self.min_time is not None:
            c["min_time"] = self.min_time
        if self.max_time is not None:
            c["max_time"] = self.max_time
        if self.min_lat is not None:
            c["min_lat"] = self.min_lat
        if self.max_lat is not None:
            c["max_lat"] = self.max_lat
        if self.min_lon is not None:
            c["min_lon"] = self.min_lon
        if self.max_lon is not None:
            c["max_lon"] = self.max_lon
        return c


class ERDDAPSearch(BaseModel):
    """Parameters for searching ERDDAP datasets. Maps to get_search_url() in erddapy.
    Validated against the active server's categorize endpoints inside search_server_tool."""
    search_for: str = Field(
        default="all", description=(
            "Google-like keyword search of dataset metadata. "
            "Use quotes for phrases e.g. '\"sea water temperature\"'. "
            "Prefix with - to exclude e.g. 'glider -delayed'. "
            "Omit to return all datasets."
        )
    )
    cdm_data_type: Optional[str] = Field(
        default=None,
        description="Data type filter e.g. 'TimeSeries', 'Trajectory', 'Grid'",
    )
    institution: Optional[str] = Field(
        default=None,
        description="Institution name filter e.g. 'IOOS', 'NOAA', 'Rutgers University'",
    )
    ioos_category: Optional[str] = Field(
        default=None,
        description="IOOS category filter e.g. 'Temperature', 'Salinity', 'Currents'",
    )
    standard_name: Optional[str] = Field(
        default=None,
        description=(
            "CF standard name filter e.g. 'sea_water_temperature'. Preferred for a plain-language quantity -- "
            "resolve the exact value with resolve_categorize_values_tool first."
        ),
    )
    variable_name: Optional[str] = Field(
        default=None,
        description="Raw variable name filter, matches only an exact column name. Fallback if standard_name finds nothing.",
    )
    constraints: Optional[ERDDAPConstraints] = Field(
        default=None,
        description="Spatial and temporal constraints to subset the data"
    )
    items_per_page: int = Field(default=1_000_000, description="Max number of results to return")


class ERDDAPDataset(BaseModel):
    dataset_id: str = Field(description="ERDDAP dataset ID e.g. 'ru29-20150623T1046'")
    variables: Optional[list[str]] = Field(
        default=None,
        description=(
            "Exact variable names to download. "
            "Call describe_dataset_tool first to get exact names. "
            "Omit to download all variables."
        )
    )
    constraints: Optional[ERDDAPConstraints] = Field(
        default=None,
        description="Spatial and temporal constraints to subset the data"
    )
    distinct: bool = Field(
        default=False,
        description="If true, only unique values will be downloaded"
    )


class ERDDAPPlotBase(BaseModel):
    """Shared fields for the plot tools in erddap_dataset_tools.py. Doesn't extend
    ERDDAPDataset: plots derive their needed columns from x/y/color, not a raw variable list."""
    dataset_id: str = Field(description="ERDDAP dataset ID e.g. 'ru29-20150623T1046'")
    constraints: Optional[ERDDAPConstraints] = Field(
        default=None,
        description="Spatial and temporal constraints to subset the data"
    )
    engine: Literal["local", "erddap"] = Field(
        default="local",
        description=(
            "'local' downloads the data and renders the plot here (matplotlib/plotly), giving full control "
            "over styling. 'erddap' skips the download and returns a URL to ERDDAP's own server-rendered "
            "image for the same variables/constraints -- faster, no local file."
        ),
    )
    step_size: int = Field(
        default=1,
        description="Plot every Nth data point. Use 1 for all points. Increase to thin dense datasets.",
    )


class ERDDAPTimeseriesPlot(ERDDAPPlotBase):
    """A variable plotted against time."""
    y_var: str = Field(description="Variable name to plot against time, e.g. 'temperature'.")
    color_var: Optional[str] = Field(default=None, description="Optional variable to color points by, e.g. 'depth'.")
    file_type: Literal["png", "pdf"] = Field(default="png", description="Output file format")


class ERDDAPProfilePlot(ERDDAPPlotBase):
    """A variable plotted against depth."""
    x_var: str = Field(description="Variable name to plot against depth, e.g. 'temperature'.")
    color_var: Optional[str] = Field(default=None, description="Optional variable to color points by, e.g. 'time'.")
    file_type: Literal["png", "pdf"] = Field(default="png", description="Output file format")


class ERDDAPMapPlot(ERDDAPPlotBase):
    """A trajectory map of the dataset's latitude/longitude."""
    color_var: Optional[str] = Field(
        default=None, description="Optional variable to color the trajectory by, e.g. 'temperature'."
    )
    file_type: Literal["png", "pdf"] = Field(
        default="png",
        description=(
            "Output format for engine='erddap' (ERDDAP's own server-rendered map). Ignored for "
            "engine='local', which always writes an interactive HTML file."
        ),
    )


class ERDDAPCustomPlot(ERDDAPPlotBase):
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

class ERDDAPRegionMapPlot(BaseModel):
    """Map every dataset on the active server that has data inside a lat/lon region
    (optionally time-bounded) -- one combined trajectory map, one color per dataset."""
    constraints: ERDDAPConstraints = Field(
        description="Region to search. min/max lat and lon are required; time bounds optional."
    )
    search_for: str = Field(
        default="all",
        description="Optional keyword filter applied alongside the region, e.g. 'glider'.",
    )
    max_datasets: int = Field(
        default=10,
        description="Cap on datasets drawn. Each one costs a data download; keep modest.",
    )
    step_size: int = Field(
        default=25,
        description="Plot every Nth point per dataset. Composite maps need heavy thinning.",
    )

    @field_validator("constraints")
    @classmethod
    def require_bbox(cls, v: "ERDDAPConstraints") -> "ERDDAPConstraints":
        if None in (v.min_lat, v.max_lat, v.min_lon, v.max_lon):
            raise ValueError("A full bounding box (min/max lat and lon) is required for a region map.")
        return v

# State helpers
# -------------
def get_server(runtime: ToolRuntime, url: str) -> ERDDAPServer:
    """Look up a registered server by URL. Raises if not registered."""
    url = ERDDAPServer.sanitize_url(url)
    servers = runtime.state.get("servers", {})
    server_dict = servers.get(url)
    if not server_dict:
        raise ValueError(f"Server {url} not registered. Call register_server_tool first.")
    return ERDDAPServer(**server_dict)


def get_active_server(runtime: ToolRuntime) -> ERDDAPServer:
    """Get the currently active server. Raises if none set."""
    url = runtime.state.get("active_server_url")
    if not url:
        raise ValueError("No active server set. Call set_active_server_tool first.")
    return get_server(runtime, url)


def check_server_status(url: str) -> dict[str, Any]:
    """Check if an ERDDAP server is online using its /version endpoint.
    Returns {"reachable": bool, "version": str | None, "error": str | None}."""
    try:
        r = requests.get(f"{url}/version", timeout=10)
        r.raise_for_status()
        return {"reachable": True, "version": r.text.strip(), "error": None}
    except Exception as err:
        return {"reachable": False, "version": None, "error": str(err)}


# State tools
# -----------
@tool
def check_server_status_tool(url: str) -> str:
    """Check whether an ERDDAP server is currently reachable. Use before
    issuing a search or data request if the server's availability is uncertain."""
    url = ERDDAPServer.sanitize_url(url)
    result = check_server_status(url)
    if result["reachable"]:
        return f"The server at {url} is reachable and running {result["version"]}"
    else:
        return f"The server at {url} is not reachable. Error: {result["error"]}"


@tool
def register_server_tool(runtime: ToolRuntime, url: str) -> Command:
    """Register an ERDDAP server and check its status.
    Always call this first when given a new server URL before any other tool."""
    server = ERDDAPServer(url=url)
    result = check_server_status(server.url)
    status_detail = (
        f"Version: {result['version']}."
        if result["reachable"]
        else f"Error: {result['error']}"
    )
    return Command(
        update={
            "servers": {server.url: server.model_dump()},
            "active_server_url": server.url,
            "messages": [
                ToolMessage(
                    content=(
                        f"Server {server.url} registered and set as active. "
                        f"Reachable: {result['reachable']}. "
                        f"{status_detail}"
                    ),
                    tool_call_id=runtime.tool_call_id
                    )
            ],
        }
    )


@tool
def set_active_server_tool(runtime: ToolRuntime, url: str) -> Command:
    """Set the active server to a previously registered URL.
    Use when switching between multiple servers."""
    server = get_server(runtime, url)
    return Command(
        update={
            "active_server_url": server.url,
            "messages": [
                ToolMessage(
                    content=f"Active server switched to {server.url}.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


# Shared helpers for erddap_server_tools.py / erddap_dataset_tools.py
# -------------------------------------------------------------------
def describe_erddap_error(err: Exception) -> str:
    """Extract ERDDAP's descriptive error body, falling back to str(err). Handles both
    urllib.error.HTTPError (.read()) and erddapy's requests.exceptions.HTTPError."""
    try:
        return getattr(err, "read")().decode("utf-8", errors="replace").strip()
    except Exception:
        return str(err)


def is_qc_variable(name: str) -> bool:
    """True if a dataset variable name looks like a QC/quality-control flag column."""
    lowered = name.lower()
    return "qc" in lowered or "qartod" in lowered or lowered.endswith("flag")


def is_qc_keyword(name: str) -> bool:
    """True if a server-wide keyword looks QC-related. Narrower than is_qc_variable:
    keyword text is free-form (e.g. 'Red Flag Warning'), so the '..._flag' suffix check
    would false-positive here."""
    lowered = name.lower()
    return "qc" in lowered or "qartod" in lowered
