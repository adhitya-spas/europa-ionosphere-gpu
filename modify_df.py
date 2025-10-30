import pandas as pd

# All angles used in this file should be interpreted as true incidence angle (from vertical, 0 = nadir, 90 = horizontal)

def save_mission_df(df: pd.DataFrame, path: str = "new_mission_df.pkl") -> None:
    """
    Save the mission DataFrame to disk in pickle format.
    This preserves the Python lists in the 'Altitude' and 'Ne' columns exactly.
    """
    df.to_pickle(path)
    print(f"Saved DataFrame ({len(df)} rows) to {path!r}")

def load_mission_df(path: str = "new_mission_df.pkl") -> pd.DataFrame:
    """
    Load the mission DataFrame back from disk.
    Returns a DataFrame with exactly the same structure you saved.
    """
    df = pd.read_pickle(path)
    print(f"Loaded DataFrame ({len(df)} rows) from {path!r}")
    return df

# # modify_df.py
# import pandas as pd
# import numpy as np
# import json

# # Columns that contain array-like data
# _LIST_COLS = ("Altitude", "Ne")

# def _to_jsonable(x):
#     """Recursively convert NumPy arrays/scalars to plain Python for JSON."""
#     if isinstance(x, np.ndarray):
#         return x.tolist()
#     if isinstance(x, np.generic):
#         return x.item()
#     if isinstance(x, (list, tuple)):
#         return [_to_jsonable(i) for i in x]
#     try:
#         if pd.isna(x):
#             return None
#     except Exception:
#         pass
#     return x

# def save_mission_df(df: pd.DataFrame, path: str = "new_mission_df.csv") -> None:
#     """
#     Save the mission DataFrame to CSV.
#     List/ndarray columns (Altitude, Ne) are JSON-encoded so they round-trip.
#     """
#     df_copy = df.copy()
#     for col in _LIST_COLS:
#         if col in df_copy.columns:
#             df_copy[col] = df_copy[col].map(lambda v: json.dumps(_to_jsonable(v)))
#     df_copy.to_csv(path, index=False)
#     print(f"Saved DataFrame ({len(df_copy)} rows) to {path!r}")

# def _loads_safe(s):
#     if s is None:
#         return None
#     if isinstance(s, float) and np.isnan(s):
#         return None
#     if isinstance(s, str) and (s == "" or s.lower() == "nan"):
#         return None
#     return json.loads(s)

# def load_mission_df(path: str = "new_mission_df.csv", as_numpy: bool = True) -> pd.DataFrame:
#     """
#     Load the mission DataFrame from CSV.
#     List/ndarray columns are reconstructed from JSON strings.
#     If as_numpy=True, Altitude/Ne are returned as numpy arrays; else as Python lists.
#     """
#     df = pd.read_csv(path, dtype=str)  # keep JSON strings intact
#     for col in _LIST_COLS:
#         if col in df.columns:
#             parsed = df[col].map(_loads_safe)
#             if as_numpy:
#                 # Convert lists/None -> np.ndarray (dtype=float) or None
#                 def to_np(a):
#                     if a is None:
#                         return None
#                     arr = np.array(a, dtype=float)  # adjust dtype if needed
#                     return arr
#                 df[col] = parsed.map(to_np)
#             else:
#                 df[col] = parsed
#     print(f"Loaded DataFrame ({len(df)} rows) from {path!r}")
#     return df