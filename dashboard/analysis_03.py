import numpy as np
import pandas as pd
try:
    from Bio import Phylo  # type: ignore
except Exception:  # pragma: no cover
    Phylo = None


RADAR_FEATURES_DEFAULT = ["tarsus", "wing_len", "tail", "mass", "hwi", "beak_depth"]

# Row-stats columns shown in Section 3 UI are filtered client-side: see static/js/sectio3.js SECTION3_ROW_TRAIT_KEYS.


def build_species_table(df: pd.DataFrame, species_col: str = "species_birdtree") -> pd.DataFrame:
    """Return a de-duplicated table keyed by species name."""
    if species_col not in df.columns:
        raise ValueError(f"Missing required species column: {species_col}")
    table = df.dropna(subset=[species_col]).drop_duplicates(subset=[species_col]).set_index(species_col, drop=False)
    return table


def get_numeric_columns(df: pd.DataFrame, exclude: set[str] | None = None) -> list[str]:
    exclude = exclude or set()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num_cols if c not in exclude]


def precompute_normalized_numeric(species_table: pd.DataFrame, numeric_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """Z-score normalize numeric columns once, for fast Euclidean distances."""
    X = species_table[numeric_cols].astype(float)
    means = X.mean(axis=0, skipna=True)
    stds = X.std(axis=0, skipna=True).replace(0, np.nan)
    X_norm = (X - means) / stds
    meta = {"means": means.to_dict(), "stds": stds.to_dict(), "numeric_cols": numeric_cols}
    return X_norm, meta


def euclidean_distance_normed(norm_df: pd.DataFrame, species_a: str, species_b: str) -> tuple[float | None, int]:
    """Euclidean distance on normalized numeric features. Returns (distance, n_features_used)."""
    if species_a not in norm_df.index or species_b not in norm_df.index:
        return None, 0
    a = norm_df.loc[species_a]
    b = norm_df.loc[species_b]
    mask = np.isfinite(a.values) & np.isfinite(b.values)
    if mask.sum() == 0:
        return None, 0
    diff = a.values[mask] - b.values[mask]
    return float(np.sqrt(np.sum(diff * diff))), int(mask.sum())


def load_tree(tree_path: str):
    if Phylo is None:
        raise ImportError("Biopython is not installed (missing 'Bio'). Install 'biopython' to enable phylogeny distance.")
    trees = Phylo.parse(tree_path, "newick")
    tree = next(trees)
    terminals = [term.name for term in tree.get_terminals() if term.name]
    return tree, set(terminals)


def phylo_distance(tree, terminal_set: set[str], species_a: str, species_b: str) -> float | None:
    if not tree or species_a not in terminal_set or species_b not in terminal_set:
        return None
    try:
        return float(tree.distance(species_a, species_b))
    except Exception:
        return None


def row_to_jsonable_dict(row: pd.Series) -> dict:
    out = {}
    for k, v in row.items():
        if pd.isna(v):
            out[k] = None
        elif isinstance(v, (np.integer, np.int64)):
            out[k] = int(v)
        elif isinstance(v, (np.floating, np.float64)):
            out[k] = float(v)
        else:
            out[k] = v
    return out

