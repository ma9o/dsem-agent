"""Aggregation registry for DSEM time-series aggregations using Polars."""

from typing import Callable

import polars as pl

# Type alias for aggregator functions
# Takes column name, returns Polars expression
Aggregator = Callable[[str], pl.Expr]


def agg_mean(col: str) -> pl.Expr:
    """Mean aggregation."""
    return pl.col(col).mean()


def agg_p90(col: str) -> pl.Expr:
    """90th percentile."""
    return pl.col(col).quantile(0.9)


def agg_entropy(col: str) -> pl.Expr:
    """Shannon entropy (normalized counts)."""
    # For continuous data, this computes entropy of binned values
    # For categorical, computes entropy directly
    return (
        pl.col(col)
        .value_counts()
        .struct.field("count")
        .cast(pl.Float64)
        .map_batches(
            lambda s: pl.Series([-((p := s / s.sum()) * (p + 1e-10).log()).sum()])
        )
        .first()
    )


def agg_range(col: str) -> pl.Expr:
    """Range (max - min)."""
    return pl.col(col).max() - pl.col(col).min()


def agg_cv(col: str) -> pl.Expr:
    """Coefficient of variation (std/mean)."""
    return pl.col(col).std() / (pl.col(col).mean() + 1e-10)


def agg_iqr(col: str) -> pl.Expr:
    """Interquartile range (p75 - p25)."""
    return pl.col(col).quantile(0.75) - pl.col(col).quantile(0.25)


# Main aggregation registry
AGGREGATION_REGISTRY: dict[str, Aggregator] = {
    # --- Standard statistics ---
    "mean": agg_mean,
    "sum": lambda c: pl.col(c).sum(),
    "min": lambda c: pl.col(c).min(),
    "max": lambda c: pl.col(c).max(),
    "std": lambda c: pl.col(c).std(),
    "var": lambda c: pl.col(c).var(),
    "last": lambda c: pl.col(c).last(),
    "first": lambda c: pl.col(c).first(),
    "count": lambda c: pl.col(c).count(),
    # --- Distributional ---
    "median": lambda c: pl.col(c).median(),
    "p10": lambda c: pl.col(c).quantile(0.1),
    "p25": lambda c: pl.col(c).quantile(0.25),
    "p75": lambda c: pl.col(c).quantile(0.75),
    "p90": agg_p90,
    "p99": lambda c: pl.col(c).quantile(0.99),
    "skew": lambda c: pl.col(c).skew(),
    "kurtosis": lambda c: pl.col(c).kurtosis(),
    "iqr": agg_iqr,
    # --- Spread/variability ---
    "range": agg_range,
    "cv": agg_cv,
    # --- Domain-specific ---
    "entropy": agg_entropy,
    "instability": lambda c: pl.col(c).diff().abs().mean(),  # Mean absolute change
    "trend": lambda c: pl.col(c).diff().mean(),  # Average direction of change
    "n_unique": lambda c: pl.col(c).n_unique(),
}


def get_aggregator(name: str) -> Aggregator:
    """Get an aggregator function by name.

    Args:
        name: Aggregation function name (must be in AGGREGATION_REGISTRY)

    Returns:
        Polars expression factory function

    Raises:
        ValueError: If aggregation name is not in registry
    """
    if name not in AGGREGATION_REGISTRY:
        available = ", ".join(sorted(AGGREGATION_REGISTRY.keys()))
        raise ValueError(f"Unknown aggregation '{name}'. Available: {available}")
    return AGGREGATION_REGISTRY[name]


def list_aggregations() -> list[str]:
    """List all available aggregation names."""
    return sorted(AGGREGATION_REGISTRY.keys())


def apply_aggregation(df: pl.DataFrame, col: str, agg_name: str, group_by: list[str] | None = None) -> pl.DataFrame:
    """Apply an aggregation to a DataFrame column.

    Args:
        df: Input DataFrame
        col: Column to aggregate
        agg_name: Name of aggregation from registry
        group_by: Optional columns to group by before aggregating

    Returns:
        Aggregated DataFrame
    """
    agg_fn = get_aggregator(agg_name)
    expr = agg_fn(col).alias(f"{col}_{agg_name}")

    if group_by:
        return df.group_by(group_by).agg(expr)
    return df.select(expr)


def _truncate_to_granularity(ts: pl.Expr, granularity: str) -> pl.Expr:
    """Truncate a datetime expression to the specified granularity.

    Args:
        ts: Polars datetime expression
        granularity: One of 'hourly', 'daily', 'weekly', 'monthly', 'yearly'

    Returns:
        Truncated datetime expression
    """
    truncate_map = {
        "hourly": "1h",
        "daily": "1d",
        "weekly": "1w",
        "monthly": "1mo",
        "yearly": "1y",
    }
    if granularity not in truncate_map:
        raise ValueError(f"Unknown granularity '{granularity}'. Must be one of: {list(truncate_map.keys())}")
    return ts.dt.truncate(truncate_map[granularity])


def _coerce_value_to_numeric(value) -> float | None:
    """Coerce a value to numeric, returning None if not possible."""
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def aggregate_worker_measurements(
    dataframes: list[pl.DataFrame],
    schema: dict,
) -> dict[str, pl.DataFrame]:
    """Aggregate worker measurements into time-series DataFrames by causal_granularity.

    Takes raw worker outputs (dimension, value, timestamp) and produces
    time-series DataFrames ready for causal modeling:
    1. Concatenates all worker DataFrames
    2. Parses timestamps
    3. Groups dimensions by their causal_granularity
    4. For each granularity, buckets timestamps and applies aggregation
    5. Returns one DataFrame per granularity with dimensions as columns

    Args:
        dataframes: List of DataFrames from workers, each with columns
                   (dimension, value, timestamp)
        schema: DSEM schema dict containing dimension definitions with
               causal_granularity and aggregation functions

    Returns:
        Dict mapping granularity -> DataFrame. Each DataFrame has 'time_bucket'
        as first column, then one column per observed dimension at that granularity.
        Time-invariant dimensions (causal_granularity=None) are in key 'time_invariant'
        as a single-row DataFrame.
    """
    if not dataframes:
        return {}

    # Concatenate all worker dataframes
    combined = pl.concat(dataframes, how="vertical")

    if combined.is_empty():
        return {}

    # Build dimension metadata from schema
    dim_info = {}
    for dim in schema.get("dimensions", []):
        name = dim.get("name")
        if not name:
            continue
        # Only process observed dimensions (latent have no measurements)
        if dim.get("observability") == "latent":
            continue
        dim_info[name] = {
            "causal_granularity": dim.get("causal_granularity"),
            "aggregation": dim.get("aggregation", "mean"),
        }

    # Parse timestamps - try multiple formats
    combined = combined.with_columns(
        pl.col("timestamp")
        .str.to_datetime(strict=False)
        .alias("parsed_ts")
    )

    # Coerce values to numeric
    combined = combined.with_columns(
        pl.col("value")
        .map_elements(_coerce_value_to_numeric, return_dtype=pl.Float64)
        .alias("numeric_value")
    )

    # Group dimensions by granularity
    dims_by_granularity: dict[str | None, list[str]] = {}
    for dim_name, info in dim_info.items():
        gran = info["causal_granularity"]
        if gran not in dims_by_granularity:
            dims_by_granularity[gran] = []
        dims_by_granularity[gran].append(dim_name)

    results: dict[str, pl.DataFrame] = {}

    for granularity, dim_names in dims_by_granularity.items():
        if granularity is None:
            # Time-invariant dimensions - aggregate all values into single row
            time_invariant_cols = {}
            for dim_name in dim_names:
                dim_data = combined.filter(pl.col("dimension") == dim_name)
                if dim_data.is_empty():
                    continue

                agg_name = dim_info[dim_name]["aggregation"]
                try:
                    agg_fn = get_aggregator(agg_name)
                except ValueError:
                    agg_fn = get_aggregator("mean")

                agg_value = dim_data.select(agg_fn("numeric_value")).item()
                time_invariant_cols[dim_name] = [agg_value]

            if time_invariant_cols:
                results["time_invariant"] = pl.DataFrame(time_invariant_cols)
            continue

        # Time-varying dimensions at this granularity
        # Filter to rows with valid timestamps
        gran_data = combined.filter(
            pl.col("parsed_ts").is_not_null() &
            pl.col("dimension").is_in(dim_names)
        )

        if gran_data.is_empty():
            continue

        # Process each dimension and collect for joining
        dim_dfs = []
        for dim_name in dim_names:
            dim_data = gran_data.filter(pl.col("dimension") == dim_name)
            if dim_data.is_empty():
                continue

            # Bucket timestamps to this granularity
            dim_data = dim_data.with_columns(
                _truncate_to_granularity(pl.col("parsed_ts"), granularity).alias("time_bucket")
            )

            # Get aggregation function
            agg_name = dim_info[dim_name]["aggregation"]
            try:
                agg_fn = get_aggregator(agg_name)
            except ValueError:
                agg_fn = get_aggregator("mean")

            # Group by time bucket and aggregate
            aggregated = (
                dim_data
                .group_by("time_bucket")
                .agg(agg_fn("numeric_value").alias(dim_name))
                .sort("time_bucket")
            )

            dim_dfs.append(aggregated)

        if not dim_dfs:
            continue

        # Join all dimensions at this granularity on time_bucket
        result = dim_dfs[0]
        for df in dim_dfs[1:]:
            result = result.join(df, on="time_bucket", how="full", coalesce=True)

        # Sort by time
        result = result.sort("time_bucket")
        results[granularity] = result

    return results
