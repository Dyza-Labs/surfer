import time
from dataclasses import dataclass
from typing import Optional

from langchain.messages import ToolMessage
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from siphon.catalog import TDSCatalog

from surfer.agents.thredds_agent import get_active_catalog_url, get_active_server


def browse_catalog(catalog_url: str, descend_into: Optional[str] = None) -> dict | str:
    """Fetch the catalog at catalog_url. If descend_into names a sub-catalog (case-insensitive
    substring match against its title), follow it and describe that child instead. Returns
    {"catalog_url": str, "catalog_name": str, "sub_catalogs": [str, ...], "datasets": [str, ...]},
    or an error string with a "did you mean" hint if descend_into doesn't match anything, or a
    clean error if catalog_url itself can't be fetched (e.g. stale/hand-built, rather than one
    just returned by this same function or find_datasets_tool)."""
    try:
        cat = TDSCatalog(catalog_url)
    except Exception as err:
        return (
            f"Could not fetch catalog at '{catalog_url}': {err}. Use the catalog_url from a "
            f"previous browse_catalog_tool or find_datasets_tool result rather than constructing "
            f"one -- URL path segments don't always match the display names shown."
        )

    if descend_into is not None:
        query = descend_into.lower()
        matches = [name for name in cat.catalog_refs if query in name.lower()]
        if not matches:
            available = list(cat.catalog_refs.keys())
            return f"No sub-catalog matching '{descend_into}' at {catalog_url}. Available: {available}"
        if len(matches) > 1 and descend_into not in cat.catalog_refs:
            return f"'{descend_into}' is ambiguous at {catalog_url}. Did you mean one of: {matches}?"
        name = descend_into if descend_into in cat.catalog_refs else matches[0]
        try:
            cat = cat.catalog_refs[name].follow()
        except Exception as err:
            return f"Could not follow sub-catalog '{name}' from {catalog_url}: {err}"

    return {
        "catalog_url": cat.catalog_url,
        "catalog_name": cat.catalog_name,
        "sub_catalogs": list(cat.catalog_refs.keys()),
        "datasets": list(cat.datasets.keys()),
    }


@tool
def browse_catalog_tool(runtime: ToolRuntime, descend_into: Optional[str] = None) -> str | Command:
    """List the sub-catalogs and datasets at the current position in the catalog tree.
    Pass descend_into with a sub-catalog name to walk one level deeper -- this updates the
    active catalog position. Omit it to just list the current level again."""
    catalog_url = get_active_catalog_url(runtime)
    result = browse_catalog(catalog_url, descend_into=descend_into)
    if isinstance(result, str):
        return result

    summary = (
        f"Catalog '{result['catalog_name']}' at {result['catalog_url']}:\n"
        f"Sub-catalogs: {result['sub_catalogs']}\n"
        f"Datasets: {result['datasets']}"
    )
    if descend_into is None:
        return summary

    return Command(
        update={
            "active_catalog_url": result["catalog_url"],
            "messages": [ToolMessage(content=summary, tool_call_id=runtime.tool_call_id)],
        }
    )


# Names/paths only in v1; a metadata field (e.g. standard_name, institution) could be
# added later, but that would need to open() each leaf, which this index deliberately avoids.
@dataclass
class CatalogIndexEntry:
    name: str
    catalog_url: str
    parent_path: list[str]


_catalog_index_cache: dict[str, tuple[float, list[CatalogIndexEntry]]] = {}
_CATALOG_INDEX_TTL_SECONDS = 300


def build_catalog_index(root_catalog_url: str, max_depth: int = 8) -> list[CatalogIndexEntry]:
    """Walk the full catalog_refs tree from root_catalog_url, recording every leaf dataset
    as a CatalogIndexEntry. Cached per root URL; a fresh crawl only happens once per TTL
    window. Names/paths only: never open()s a dataset, so cost is one catalog.xml fetch per
    sub-catalog, not per leaf."""
    cached = _catalog_index_cache.get(root_catalog_url)
    if cached is not None and time.monotonic() - cached[0] < _CATALOG_INDEX_TTL_SECONDS:
        return cached[1]

    entries: list[CatalogIndexEntry] = []
    skipped = 0
    root = TDSCatalog(root_catalog_url)
    queue: list[tuple[TDSCatalog, list[str]]] = [(root, [])]

    while queue:
        cat, path = queue.pop(0)
        for name in cat.datasets:
            entries.append(CatalogIndexEntry(name=name, catalog_url=cat.catalog_url, parent_path=path))
        if len(path) >= max_depth:
            continue
        for ref_name, ref in cat.catalog_refs.items():
            try:
                child = ref.follow()
            except Exception:
                skipped += 1  # a stale/broken catalogRef shouldn't sink the whole walk
                continue
            queue.append((child, path + [ref_name]))

    if skipped:
        print(f"build_catalog_index: skipped {skipped} unreachable sub-catalog(s) under {root_catalog_url}")
    _catalog_index_cache[root_catalog_url] = (time.monotonic(), entries)
    return entries


def search_catalog_index(index: list[CatalogIndexEntry], query: str) -> list[CatalogIndexEntry]:
    """Case-insensitive substring match against entry.name. Pure filter, no I/O."""
    q = query.lower()
    return [entry for entry in index if q in entry.name.lower()]


@tool
def find_datasets_tool(runtime: ToolRuntime, query: str) -> str:
    """Search dataset *names* by substring across the whole subtree of the active server's
    root catalog. This is name matching only -- it does not inspect any dataset's actual
    CF metadata (standard_name, institution, etc.), since there's no server-side search to
    query. Builds a local index of the tree on first use per server and reuses it briefly,
    so results reflect a recent, not necessarily live, snapshot of the tree."""
    server = get_active_server(runtime)
    index = build_catalog_index(server.url)
    matches = search_catalog_index(index, query)
    if not matches:
        return f"No datasets matching '{query}' found under {server.url}."

    lines = [f"Found {len(matches)} dataset(s) matching '{query}' under {server.url}:"]
    for entry in matches[:30]:
        path = " > ".join(entry.parent_path) if entry.parent_path else "(root)"
        lines.append(f"- `{entry.name}` at {entry.catalog_url} (path: {path})")
    if len(matches) > 30:
        lines.append(f"... and {len(matches) - 30} more.")
    return "\n".join(lines)


@tool
def get_catalog_summary_tool(runtime: ToolRuntime) -> str:
    """Summarize the current catalog level only: its name, sub-catalog count, dataset count,
    and service names declared here. Does NOT roll up institutions/keywords across the whole
    server the way ERDDAP's get_server_summary_tool does -- that would require an unbounded
    recursive walk plus per-leaf metadata resolution, which is out of scope."""
    catalog_url = get_active_catalog_url(runtime)
    try:
        cat = TDSCatalog(catalog_url)
    except Exception as err:
        return f"Could not fetch catalog at '{catalog_url}': {err}"
    services = sorted({s.name for s in cat.services}) if hasattr(cat, "services") else []
    return (
        f"Catalog '{cat.catalog_name}' at {cat.catalog_url}:\n"
        f"Sub-catalogs: {len(cat.catalog_refs)}\n"
        f"Datasets at this level: {len(cat.datasets)}\n"
        f"Services declared here: {services}"
    )
