import re
import time
from typing import Any, Optional

import pandas as pd
import requests
from erddapy import ERDDAP
from langchain.tools import tool, ToolRuntime

from surfer.agents.erddap_agent import (
    ERDDAPConstraints, ERDDAPSearch, get_active_server, describe_erddap_error,
    is_qc_keyword, is_qc_variable,
)

_categorize_values_cache: dict[tuple[str, str], tuple[float, list[str]]] = {}
_CATEGORIZE_CACHE_TTL_SECONDS = 60
_CATEGORIZE_CACHE_MAX_SIZE = 256


def clean_erddap_title(title: str) -> str:
    """Cleans a raw ERDDAP title like 'rutgers_university' into 'Rutgers University'."""
    if title is None:
        return ""
    cleaned_title = str(title).strip("_ ").replace("_", " ")
    cleaned_title = cleaned_title.title()
    return cleaned_title


def get_categorize_values(server_url: str, categorize_by: str) -> list[str] | str:
    """Fetch the raw values ERDDAP accepts for a categorize field (e.g. 'institution',
    'ioos_category', 'standard_name', 'cdm_data_type'), or an error string if invalid.

    Cached for 60s per (server_url, categorize_by): resolve_categorize_values_tool and
    search_server_tool's validation otherwise both re-fetch the same list seconds apart --
    same pattern as erddap_dataset_tools.py's _fetch_dataset_variables."""
    key = (server_url, categorize_by)
    cached = _categorize_values_cache.get(key)
    if cached is not None and time.monotonic() - cached[0] < _CATEGORIZE_CACHE_TTL_SECONDS:
        return cached[1]

    e = ERDDAP(server=server_url, protocol="tabledap", response="csv")
    url = e.get_categorize_url(categorize_by=categorize_by, response="csv")
    try:
        df = pd.read_csv(url)
    except Exception as err:
        return f"Could not retrieve {categorize_by} categories from {server_url}: {describe_erddap_error(err)}"

    values = df["Category"].dropna().tolist()
    if len(_categorize_values_cache) >= _CATEGORIZE_CACHE_MAX_SIZE:
        _categorize_values_cache.clear()  # short-TTL design
    _categorize_values_cache[key] = (time.monotonic(), values)
    return values


def filter_categorize_values(values: list[str], query: Optional[str] = None) -> list[str]:
    """Narrow raw categorize values to those containing `query` (case-insensitive)."""
    if not query:
        return values
    q = query.lower()
    return [v for v in values if q in v.lower()]


def get_server_summary(url: str) -> dict[str, Any]:
    """Get a server's dataset count, protocols, institutions, and keywords. Hits the
    /info/index.csv and /categorize endpoints, so heavier than most tool calls."""
    e = ERDDAP(server = url, protocol="tabledap", response="csv")
    df = pd.read_csv(e.get_info_url(response="csv"))
    df = df[df["Dataset ID"] != "allDatasets"]

    protocols = [p for p in ("griddap", "tabledap", "wms") if df[p].notna().any()]

    index_html = requests.get(f"{url}/index.html", timeout=10).text
    match = re.search(r"View a List of All ([\d,]+) Datasets", index_html)
    if match is None:
        raise ValueError(f"Could not find dataset count on {url}/index.html")
    count = int(match.group(1).replace(",", ""))

    raw_institutions = get_categorize_values(url, "institution")
    if isinstance(raw_institutions, str):
        raise ValueError(raw_institutions)
    unique_institutions = sorted({clean_erddap_title(inst) for inst in raw_institutions if inst.strip()})

    raw_keywords = get_categorize_values(url, "keywords")
    if isinstance(raw_keywords, str):
        raise ValueError(raw_keywords)
    unique_keywords = sorted(
        {clean_erddap_title(kw) for kw in raw_keywords if kw.strip() and not is_qc_keyword(kw)}
    )

    return {
        "dataset_count": count,
        "protocols": protocols,
        "institutions": unique_institutions,
        "keywords": unique_keywords,
    }


@tool
def get_server_summary_tool(runtime: ToolRuntime) -> str:
    """Summarize an ERDDAP server with its dataset count, protocols, institutions, and keywords.
    Use when a user is looking for datasets with specific themes or institutions or just wants a
    high-level overview of a server."""
    server = get_active_server(runtime)
    try:
        result = get_server_summary(server.url)
    except (requests.RequestException, ValueError) as err:  # expected failures only
        return f"Could not summarize {server.url}: {describe_erddap_error(err)}"
    return f"Summary of the server at {server.url}:\n\
    Unique datasets: {result["dataset_count"]}\n\
    Available protocols: {result["protocols"]}\n\
    Institutions: {result["institutions"]}.\n\
    Dataset keywords: {result["keywords"][:50]}.\n"  # cap keyword list -- some servers have hundreds


def validate_categorize_value(server_url: str, categorize_by: str, value: str) -> Optional[str]:
    """None if `value` is valid under `categorize_by` on the server, else an error message
    listing available options."""
    available = get_categorize_values(server_url, categorize_by)
    if isinstance(available, str):
        return available
    if value not in available:
        return f"'{value}' not found in {categorize_by} on {server_url}. Available: {available[:20]}"
    return None


@tool
def resolve_categorize_values_tool(runtime: ToolRuntime, categorize_by: str, query: Optional[str] = None) -> str:
    """
    Look up the exact values ERDDAP accepts for a categorize field (e.g. 'institution',
    'ioos_category', 'standard_name', 'cdm_data_type') on the active server. Pass `query`
    to narrow the list on fields with many values. For standard_name, QC/status-flag
    values are already excluded.
    """
    server = get_active_server(runtime)
    available = get_categorize_values(server.url, categorize_by)
    if isinstance(available, str):
        return available

    available = filter_categorize_values(available, query=query)
    if categorize_by == "standard_name":
        available = [v for v in available if not is_qc_variable(v)]
    if not available:
        return f"No matching {categorize_by} values found on {server.url}."

    lines = [f"Available {categorize_by} values on {server.url}:"]
    lines.extend(f"- `{v}`" for v in sorted(available))
    return "\n".join(lines)


def search_server(
    server: str,
    query: str,
    variables: Optional[list[str]] = None,
    constraints: Optional[dict[str, Any]] = None,
    protocol: str = "tabledap",
    response: str = "csv",
    **kwargs: Any
) -> pd.DataFrame | str:
    """Searches an ERDDAP server for datasets. `query` supports -<keyword> to exclude.
    `protocol` must match the registered server -- get_search_url() falls back to it when
    **kwargs has none, so a mismatch here silently searches the wrong protocol's datasets.
    Returns a DataFrame of results, or an error string."""
    variables = variables or []
    constraints = constraints or {}
    e = ERDDAP(server=server, protocol=protocol, response=response)

    if len(variables) == 1:
        kwargs["variableName"] = variables[0]
    elif variables:
        # Multiple variables fall back to a concatenated query
        query = " ".join([query] + variables).strip()
    search_kwargs = {
        "search_for": query,
        **constraints,
        **kwargs
    }
    try:
        e_url = e.get_search_url(**search_kwargs)
        df = pd.read_csv(e_url)
        return df
    except Exception as err:
        return f"Search failed on {server}: {describe_erddap_error(err)}"


@tool(args_schema=ERDDAPSearch)
def search_server_tool(
    runtime: ToolRuntime,
    search_for: str = "all",
    cdm_data_type: Optional[str] = None,
    institution: Optional[str] = None,
    ioos_category: Optional[str] = None,
    standard_name: Optional[str] = None,
    variable_name: Optional[str] = None,
    constraints: Optional[ERDDAPConstraints] = None,
    items_per_page: int = 1_000_000,
) -> str:
    """Search the active ERDDAP server for datasets matching specified criteria."""
    try:
        active_server = get_active_server(runtime)
        for categorize_by, value in (
            ("institution", institution),
            ("ioos_category", ioos_category),
            ("standard_name", standard_name),
            ("cdm_data_type", cdm_data_type),
        ):
            if value is None:
                continue
            error = validate_categorize_value(active_server.url, categorize_by, value)
            if error:
                return error

        kwargs = {}
        if cdm_data_type:
            kwargs["cdm_data_type"] = cdm_data_type
        if institution:
            kwargs["institution"] = institution
        if ioos_category:
            kwargs["ioos_category"] = ioos_category
        if standard_name:
            kwargs["standard_name"] = standard_name

        dict_constraints = constraints.to_kwargs() if constraints else {}
        result = search_server(
            server=active_server.url,
            query=search_for,
            variables=[variable_name] if variable_name else [],
            constraints=dict_constraints,
            protocol=active_server.protocol,
            response=active_server.response,
            items_per_page=items_per_page,
            **kwargs
        )
        if isinstance(result, str):
            return result

        if result.empty or "Dataset ID" not in result.columns:
            return f"No datasets found on {active_server.url} matching the search criteria."

        unique_datasets = result[["Dataset ID", "Title"]].drop_duplicates().to_dict(orient="records")

        output = f"Found {len(unique_datasets)} dataset(s) on {active_server.url}:\n"
        for ds in unique_datasets[:30]:
            output += f"- `{ds['Dataset ID']}`: {ds['Title']}\n"

        if len(unique_datasets) > 30:
            output += f"... and {len(unique_datasets) - 30} more datasets."

        return output
    except Exception as err:
        return f"An unexpected error occurred during search execution: {describe_erddap_error(err)}"
