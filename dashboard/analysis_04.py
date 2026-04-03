import math
import numpy as np
import pandas as pd


def _to_float(value):
    try:
        return float(value)
    except Exception:
        return None


def rank_species_by_traits(
    species_table: pd.DataFrame,
    numeric_cols: list[str],
    norm_meta: dict,
    trait_inputs: list[dict],
    top_k: int = 10,
):
    """
    Match species using numerical trait inputs only (Euclidean in normalized space).
    """
    candidates = species_table.copy()
    numeric_traits = []

    for item in trait_inputs:
        feature = (item.get("feature") or "").strip()
        raw_value = item.get("value")
        if feature == "" or raw_value in [None, ""]:
            continue
        if feature not in candidates.columns or feature not in numeric_cols:
            continue

        val = _to_float(raw_value)
        if val is not None:
            numeric_traits.append((feature, val))

    if len(candidates) == 0:
        return []

    means = norm_meta.get("means", {})
    stds = norm_meta.get("stds", {})

    results = []
    for species_name, row in candidates.iterrows():
        dist_terms = []
        used = 0
        for feat, user_val in numeric_traits:
            rv = row.get(feat)
            if pd.isna(rv):
                continue
            std = stds.get(feat)
            mean = means.get(feat)
            if std is None or mean is None or pd.isna(std) or float(std) == 0:
                continue
            z_user = (float(user_val) - float(mean)) / float(std)
            z_row = (float(rv) - float(mean)) / float(std)
            dist_terms.append((z_row - z_user) ** 2)
            used += 1

        if len(numeric_traits) > 0 and used == 0:
            # user provided numeric traits but this row has none usable
            continue

        dist = math.sqrt(sum(dist_terms)) if dist_terms else 0.0
        results.append((species_name, dist, used))

    results.sort(key=lambda x: x[1])
    return results[:top_k]

