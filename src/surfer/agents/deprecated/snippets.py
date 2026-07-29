"""Code removed from the active codebase, kept here instead of deleted outright."""

import time
import urllib.parse

import pandas as pd
from erddapy import ERDDAP

from surfer.agents.tools.erddap_dataset_tools import get_erddap, fix_labels


# Removed from erddap_server_tools.py. Reimplements erddapy's get_search_url()
def search_gliders(
    query: str = "all",
    search_by: str = "institution",  # Options: 'institution', 'author', 'title', 'all'
    variable: str | None = None,  # e.g., 'salinity', 'temperature', 'pressure'
    lon_bounds: tuple | None = None,
    lat_bounds: tuple | None = None,
    time_bounds: tuple | None = None,
    cdm_data_type: str = "trajectoryprofile",
    server: str = "https://gliders.ioos.us/erddap",
) -> list:
    """Searchs the glider server for datasets matching a query string, filtering by name and a given
    title (ex. institution, author)."""

    VARIABLE_MAP = {
        "salinity": "sea_water_practical_salinity",
        "salt": "sea_water_practical_salinity",
        "temperature": "sea_water_temperature",
        "temp": "sea_water_temperature",
        "pressure": "sea_water_pressure",
        "depth": "depth",
        "oxygen": "mole_concentration_of_dissolved_molecular_oxygen_in_sea_water",
        "chlorophyll": "concentration_of_chlorophyll_in_sea_water",
        "density": "sea_water_density",
    }

    base_url = f"{server.rstrip('/')}/search/advanced.csv"

    # Unpack a list query into separate elements, or wrap a single string into a list
    if isinstance(query, list):
        queries_to_run = query
    else:
        queries_to_run = [query]

    all_dataset_ids = set()  # Using a set automatically prevents duplicate entries
    combined_results = []

    # Loop through each individual query item independently
    for single_query in queries_to_run:
        params = {"response": "csv", "protocol": "tabledap"}

        is_all = False
        if isinstance(single_query, str) and single_query.lower() == "all":
            is_all = True

        # Construct single-target parameters
        if not is_all:
            if search_by == "institution" and len(str(single_query)) < 15:
                params["searchFor"] = single_query
            elif search_by == "institution":
                params["institution"] = single_query
            else:
                params["searchFor"] = single_query

        # Handle variable mapping
        if variable:
            if isinstance(variable, list):
                # If variable is a list, we handle the first entry or map sequentially
                mapped_vars = [VARIABLE_MAP.get(v.lower().strip(), v) for v in variable]
                params["standard_name"] = mapped_vars[0] or "" # Advanced search requires single standard_name strings
            else:
                clean_var = variable.lower().strip()
                params["standard_name"] = VARIABLE_MAP.get(clean_var, variable)

        if lon_bounds:
            params["minLon"], params["maxLon"] = lon_bounds
        if lat_bounds:
            params["minLat"], params["maxLat"] = lat_bounds
        if time_bounds:
            params["minTime"], params["maxTime"] = time_bounds
        if cdm_data_type:
            params["cdm_data_type"] = cdm_data_type.lower().strip()

        try:
            encoded_payload = urllib.parse.urlencode(params)
            full_url = f"{base_url}?{encoded_payload}"
            results = pd.read_csv(full_url)

            if not results.empty and "Dataset ID" in results.columns:
                combined_results.append(results)
                all_dataset_ids.update(results["Dataset ID"].tolist())

        except Exception:
            continue

    # If all searches completely failed to yield entries
    if not combined_results:
        print(
            "\n[Notice] Search completed: 0 combined datasets found matching criteria."
        )
        return []

    final_df = pd.concat(combined_results).drop_duplicates(subset=["Dataset ID"])

    if search_by == "title" and query.lower() != "all":
        # Apply client-side regex check for multi-title options if string validation is needed
        title_regex = "|".join(queries_to_run) if isinstance(query, list) else query
        final_df = final_df[
            final_df["Title"].str.contains(title_regex, case=False, na=False)
        ]

    print(
        f"\nSuccessfully isolated {len(final_df)} unique datasets across all search parameters:"
    )
    display_cols = [
        c for c in ["Title", "Institution", "Email"] if c in final_df.columns
    ]
    print(final_df[display_cols].to_string(index=False))

    return list(all_dataset_ids)


# Removed from erddap_dataset_tools.py. Superseded by get_erddap().
def get_erddap_object(
    dataset_id: str, datasets: dict, server: str = "https://gliders.ioos.us/erddap"
) -> ERDDAP:
    """Experimental and broken function. Keep here."""

    if datasets[dataset_id] is not None:
        return datasets[dataset_id] # KeyError potential

    e = ERDDAP(server=server, protocol="tabledap", response="nc")
    e.dataset_id = dataset_id
    datasets[dataset_id] = e
    return e


# Removed from erddap_dataset_tools.py. Previously used for exploration and testing.
def profile_gliders(gliders: list, server: str, start_idx: int = 0, end_idx: int | None = None):
    """
    Profiles a subset of glider datasets by verifying columns, dropping nulls,
    and calculating spatial/temporal boundaries.
    Loops through list of dataset IDS and print summary of each one.
    """
    if end_idx is None:
        end_idx = len(gliders)

    for i in range(start_idx, end_idx):
        dataset_id = gliders[i]
        print(f"\nDataset Id: {dataset_id}")
        print("-" * 60)

        try:
            di = get_erddap(dataset_id, server=server)
            di = fix_labels(di)

            temp_cols = [c for c in di.columns if "temperature" in c.lower()]
            sal_cols = [c for c in di.columns if "salinity" in c.lower()]
            actual_targets = temp_cols + sal_cols

            di_clean = di.dropna(subset=actual_targets, how="all")
            print(f"Original rows: {len(di)} | Rows with valid data: {len(di_clean)}")

            if not di_clean.empty:
                if "time" in di_clean.columns:
                    print(
                        f"Time Range:  {di_clean['time'].min()} --> {di_clean['time'].max()}"
                    )

                lat_col = next(
                    (c for c in di_clean.columns if "lat" in c.lower()), None
                )
                lon_col = next(
                    (c for c in di_clean.columns if "lon" in c.lower()), None
                )

                if lat_col and lon_col:
                    print(
                        f"Lat Range:   {di_clean[lat_col].min():.4f} --> {di_clean[lat_col].max():.4f}"
                    )
                    print(
                        f"Lon Range:   {di_clean[lon_col].min():.4f} --> {di_clean[lon_col].max():.4f}"
                    )
            else:
                print(
                    "No non-null data found for specified target columns in database."
                )

        except Exception as err:
            print(f"Error profiling dataset {dataset_id}: {err}")

        print("-" * 60)
        time.sleep(0.5)


# Removed from erddap_dataset_tools.py. Superseded by resolve_dataset_variables_tool /
# get_dataset_variables, plus letting the real ERDDAP request surface its own error.
def validate_dataset_variables(
    server: str,
    dataset_id: str,
    variables: list[str] | None = None,
    protocol: str = "tabledap",
    response: str = "csv",
) -> str | None:
    """
    Check whether `dataset_id` exists on the server and, if `variables` is given,
    that every name in it is valid for that dataset.

    Returns:
        None if valid, or an error message listing available options if not.
    """
    e = ERDDAP(server=server, protocol=protocol, response=response)
    try:
        info_df = pd.read_csv(e.get_info_url(dataset_id=dataset_id, response="csv"))
    except Exception:
        return f"Dataset '{dataset_id}' not found on {server}."

    if variables:
        available = info_df[info_df["Row Type"] == "variable"]["Variable Name"].tolist()
        invalid = [v for v in variables if v not in available]
        if invalid:
            return f"Variables {invalid} not found on '{dataset_id}'. Available: {available}"
    return None