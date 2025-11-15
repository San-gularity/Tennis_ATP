"""Create ML_final.csv: optimized implementation for matches 2005-2024.

This rewritten script focuses on performance for large data:
- load raw CSVs once and build a fast lookup by (date,winner,loser,match_num)
- parse dates vectorized with pandas to obtain years
- perform the Elo two-pass with simple Python loops over NumPy-backed columns
- stream matching raw rows to output CSV to avoid building huge intermediate DataFrames

Output: `Data/ML_final.csv` containing the original raw columns plus
`winner_elo` and `loser_elo` for matches whose year is in [2005,2024].
"""
from __future__ import annotations

import os
import sys
import pandas as pd

# ensure package import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from eloboi import find_match_files, load_matches, EloEngine
from datetime import datetime
import csv
import math


def build_ml_dataset(output_csv: str = "Data/ML_final.csv") -> str:
    # Prefer SingleMatches folder if available
    files = find_match_files(root="Data/SingleMatches")
    if not files:
        # fallback to searching entire repo
        files = find_match_files(root=".")

    # load raw CSVs with all columns (pandas for vectorized date parsing)
    raw_dfs = []
    for f in files:
        try:
            raw_dfs.append(pd.read_csv(f, low_memory=False))
        except Exception:
            continue
    if not raw_dfs:
        raise RuntimeError("No match files found to process")

    matches_raw = pd.concat(raw_dfs, ignore_index=True)

    # normalize tourney_date in raw to datetime (YYYYMMDD or flexible)
    if "tourney_date" in matches_raw.columns:
        matches_raw["_tourney_date_dt"] = pd.to_datetime(matches_raw["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
    else:
        matches_raw["_tourney_date_dt"] = pd.NaT

    # build a fast lookup map: key -> list of raw row dicts
    # key is (date_iso, winner_name, loser_name, match_num)
    raw_lookup = {}
    for idx, r in matches_raw.iterrows():
        date_dt = r.get("_tourney_date_dt")
        if pd.isna(date_dt):
            date_key = None
        else:
            date_key = date_dt.date().isoformat()
        w = r.get("winner_name")
        l = r.get("loser_name")
        mnum = r.get("match_num") if "match_num" in r else None
        key = (date_key, str(w) if not pd.isna(w) else None, str(l) if not pd.isna(l) else None, int(mnum) if not pd.isna(mnum) else None)
        raw_lookup.setdefault(key, []).append(r)

    # load standardized matches (for consistent fields and ordering)
    matches = load_matches(files)
    if matches is None or matches.empty:
        raise RuntimeError("No standardized matches available for processing")

    # ensure chronological order and lightweight arrays for iteration
    matches = matches.sort_values(["tourney_date", "match_num"]).reset_index(drop=True)
    winner_arr = matches["winner_name"].astype(str).to_list()
    loser_arr = matches["loser_name"].astype(str).to_list()
    level_arr = matches.get("tourney_level").tolist()
    date_arr = matches["tourney_date"].tolist()
    matchnum_arr = matches.get("match_num").fillna(0).astype(int).tolist()

    engine = EloEngine()

    # First pass: count matches per player (vectorized-ish)
    for a, b in zip(winner_arr, loser_arr):
        if a == "nan" or b == "nan":
            continue
        engine.update_matches_count(a, b)

    # Prepare output CSV writer (stream rows)
    out_dir = os.path.dirname(output_csv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # We'll write raw columns + winner_elo, loser_elo. If raw had no columns, use standardized columns
    raw_columns = list(matches_raw.columns) if len(matches_raw.columns) > 0 else list(matches.columns)
    # remove internal helper column if present
    if "_tourney_date_dt" in raw_columns:
        raw_columns = [c for c in raw_columns if c != "_tourney_date_dt"]

    out_columns = raw_columns + ["winner_elo", "loser_elo"]

    # open CSV and stream rows
    written = 0
    with open(output_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=out_columns)
        writer.writeheader()

        # Second pass: iterate matches, get pre-match elo, write matching raw rows
        for a, b, level, mdate, mnum in zip(winner_arr, loser_arr, level_arr, date_arr, matchnum_arr):
            if a == "nan" or b == "nan":
                # update engine with result and skip
                engine.update_elo(str(a), str(b), str(a), level, mdate if not pd.isna(mdate) else pd.Timestamp("1900-01-01"), int(mnum))
                continue

            engine._ensure_player(a)
            engine._ensure_player(b)

            rA = engine._get_last_rating(a) if engine._get_last_rating(a) is not None else 1500.0
            rB = engine._get_last_rating(b) if engine._get_last_rating(b) is not None else 1500.0

            # compute year quickly
            year = None
            try:
                year = int(pd.to_datetime(mdate).year)
            except Exception:
                year = None

            if year is not None and 2005 <= year <= 2024:
                # key to lookup raw rows
                date_key = pd.to_datetime(mdate).date().isoformat() if not pd.isna(mdate) else None
                key = (date_key, a, b, int(mnum) if not math.isnan(mnum) else None)
                candidates = raw_lookup.get(key)
                if not candidates:
                    # try looser match without match_num
                    key2 = (date_key, a, b, None)
                    candidates = raw_lookup.get(key2, [])

                if candidates:
                    for raw_row in candidates:
                        out = {c: (raw_row.get(c) if c in raw_row else "") for c in raw_columns}
                        # ensure tourney_date is ISO string
                        if "tourney_date" in out and out["tourney_date"] is not None:
                            try:
                                out["tourney_date"] = pd.to_datetime(out["tourney_date"]).date().isoformat()
                            except Exception:
                                pass
                        out["winner_elo"] = float(rA)
                        out["loser_elo"] = float(rB)
                        writer.writerow(out)
                        written += 1
                else:
                    # fallback: write a minimal row from standardized match
                    out = {c: "" for c in raw_columns}
                    out["tourney_date"] = pd.to_datetime(mdate).date().isoformat() if not pd.isna(mdate) else ""
                    out["winner_name"] = a
                    out["loser_name"] = b
                    out["match_num"] = int(mnum)
                    out["winner_elo"] = float(rA)
                    out["loser_elo"] = float(rB)
                    writer.writerow(out)
                    written += 1

            # update engine after recording pre-match elos
            engine.update_elo(a, b, a, level, mdate if not pd.isna(mdate) else pd.Timestamp("1900-01-01"), int(mnum))

    if written == 0:
        print("No matches from 2005-2024 found; ML_final will be empty")
    else:
        print(f"Wrote {written} rows to {os.path.abspath(output_csv)}")

    return output_csv


if __name__ == "__main__":
    try:
        path = build_ml_dataset("Data/ML_final.csv")
        print(f"Wrote ML dataset to: {os.path.abspath(path)}")
    except Exception as e:
        print(f"Failed to build ML dataset: {e}")
        raise
