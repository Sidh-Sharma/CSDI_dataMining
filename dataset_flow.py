# dataset_flow.py
"""
Utilities to load the Kaggle fluid flow CSV and keep only rows where flow_type == 'laminar'.
"""

from pathlib import Path
import pandas as pd
from typing import Union

DEFAULT_CSV = Path("data") / "fluid_kaggle" / "flow.csv"


def read_laminar_flow(csv_path: Union[str, Path] = DEFAULT_CSV) -> pd.DataFrame:
    """
    Read flow CSV and return only rows with flow_type == 'laminar' (case-insensitive).

    - csv_path: path to data/fluid_kaggle/flow.csv by default
    - Returns: pandas DataFrame filtered to laminar rows, index reset.
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p.resolve()}")

    df = pd.read_csv(p)
    # Ensure flow_type column exists
    if "flow_type" not in df.columns:
        raise KeyError("Expected column 'flow_type' not found in CSV")

    # Normalize and filter
    # Cast to string to avoid issues with NaN, then strip and lower
    flow_vals = df["flow_type"].astype(str).str.strip().str.lower()
    mask = flow_vals == "laminar"
    df_laminar = df.loc[mask].reset_index(drop=True)
    df_laminar.drop(columns=["flow_type","sample_id"], inplace=True)
    return df_laminar


if __name__ == "__main__":
    # quick sanity check when run as a script
    df = read_laminar_flow()
    print(f"Loaded {len(df)} laminar rows from {DEFAULT_CSV}")