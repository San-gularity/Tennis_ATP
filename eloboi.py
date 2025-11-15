"""
Python translation of the original R `eloboi` script.

Provides:
- reading ATP match CSVs matching `atp_matches_YYYY.csv`
- compute_elo(): builds per-player Elo time series
- summary_players(), between_dates(), plotting helpers

Notes/assumptions:
- Uses pandas for tabular data and matplotlib for plotting
- Expects CSVs to contain columns: winner_name, loser_name, tourney_level, tourney_date, match_num
- Dates are parsed from YYYYMMDD integers/strings
"""

from __future__ import annotations

import glob
import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# --- Data loading ---------------------------------------------------------
def find_match_files(root: str = ".") -> List[str]:
    # find files named like atp_matches_YYYY.csv (no extra underscore)
    pattern = re.compile(r"atp_matches_[^_]*\.csv$")
    files = []
    for path in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True):
        if pattern.search(os.path.basename(path)):
            files.append(path)
    return sorted(files)


def load_matches(files: Optional[List[str]] = None) -> pd.DataFrame:
    if files is None:
        files = find_match_files()
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, low_memory=False))
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame()
    matches_raw = pd.concat(dfs, ignore_index=True)
    cols = ["winner_name", "loser_name", "tourney_level", "tourney_date", "match_num"]
    matches = matches_raw.loc[:, [c for c in cols if c in matches_raw.columns]].copy()
    # parse dates stored like 20190101
    matches["tourney_date"] = pd.to_datetime(matches["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
    matches["match_num"] = pd.to_numeric(matches.get("match_num", 0), errors="coerce").fillna(0).astype(int)
    matches = matches.sort_values(["tourney_date", "match_num"]).reset_index(drop=True)
    return matches

def load_clay_matches_1990_2024(files: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load matches, keep only CLAY surface and dates between 1990-01-01 and 2024-12-31.
    Returns a DataFrame with the same columns that EloEngine expects.
    """
    if files is None:
        files = find_match_files()

    dfs: List[pd.DataFrame] = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, low_memory=False))
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    matches_raw = pd.concat(dfs, ignore_index=True)

    # require a surface column
    if "surface" not in matches_raw.columns:
        raise RuntimeError("No 'surface' column found in match CSVs")

    # keep only clay matches (case-insensitive)
    clay = matches_raw[matches_raw["surface"].str.upper() == "CLAY"].copy()

    # parse tourney_date
    clay["tourney_date"] = pd.to_datetime(
        clay["tourney_date"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    )

    # filter date range 2000–2024
    start = pd.Timestamp("1990-01-01")
    end   = pd.Timestamp("2024-12-31")
    clay = clay[(clay["tourney_date"] >= start) & (clay["tourney_date"] <= end)]

    # keep only the columns EloEngine actually uses
    cols = ["winner_name", "loser_name", "tourney_level", "tourney_date", "match_num"]
    clay = clay.loc[:, [c for c in cols if c in clay.columns]].copy()

    # clean match_num and sort
    clay["match_num"] = pd.to_numeric(clay.get("match_num", 0), errors="coerce").fillna(0).astype(int)
    clay = clay.sort_values(["tourney_date", "match_num"]).reset_index(drop=True)

    return clay

def export_grass_elo_1990_2024(out_csv: str = "Data/elo_grass_1990_2024.csv") -> str:
    """
    Compute Elo ratings using only matches on GRASS surface between 1990 and 2024,
    and export a wide CSV: 'date' + one column per player.
    """
    # find all match files like atp_matches_YYYY.csv
    files = find_match_files(root=".")
    if not files:
        files = find_match_files(root="Data/SingleMatches")

    if not files:
        print("No match CSV files found.")
        return ""

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, low_memory=False))
        except Exception as e:
            print(f"Skipping {f}: {e}")

    if not dfs:
        print("No data loaded from CSVs.")
        return ""

    matches_raw = pd.concat(dfs, ignore_index=True)

    # make sure needed columns exist
    for c in ["winner_name", "loser_name", "tourney_date"]:
        if c not in matches_raw.columns:
            raise RuntimeError(f"Required column '{c}' not found in match data")

    if "surface" not in matches_raw.columns:
        raise RuntimeError("No 'surface' column found in match data")

    # 1) keep only GRASS matches
    grass = matches_raw[matches_raw["surface"].str.upper() == "GRASS"].copy()

    # 2) parse date and filter to 1990–2024
    grass["tourney_date"] = pd.to_datetime(
        grass["tourney_date"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    )

    start = pd.Timestamp("1990-01-01")
    end   = pd.Timestamp("2024-12-31")
    grass = grass[(grass["tourney_date"] >= start) & (grass["tourney_date"] <= end)]

    if grass.empty:
        print("No GRASS matches between 1990 and 2024.")
        return ""

    # 3) keep only columns EloEngine needs
    cols = ["winner_name", "loser_name", "tourney_level", "tourney_date", "match_num"]
    present = [c for c in cols if c in grass.columns]
    grass = grass.loc[:, present].copy()

    # 4) sort by date and match_num
    if "match_num" in grass.columns:
        grass["match_num"] = pd.to_numeric(
            grass["match_num"], errors="coerce"
        ).fillna(0).astype(int)
        grass = grass.sort_values(["tourney_date", "match_num"])
    else:
        grass = grass.sort_values(["tourney_date"])

    grass = grass.reset_index(drop=True)

    # 5) compute Elo and export wide CSV (date + one column per player)
    engine = EloEngine()
    engine.compute_from_matches(grass)

    path = export_all_players_timeseries_csv(engine, out_csv=out_csv)
    print(f"Saved GRASS Elo (1990–2024) to: {os.path.abspath(path)}")
    return path



# --- Elo machinery -------------------------------------------------------
FIRST_DATE = pd.Timestamp("1900-01-01")


class EloEngine:
    def __init__(self):
        # per-player list of records (each record is dict with ranking, date, num)
        self.players: Dict[str, List[Dict]] = {}
        self.matches_count: Dict[str, int] = {}

    def _get_last_rating(self, player: str) -> float:
        recs = self.players.get(player)
        if not recs:
            return 1500.0
        return float(recs[-1]["ranking"])

    def _ensure_player(self, player: str):
        if player not in self.players:
            self.players[player] = [{"ranking": 1500.0, "date": FIRST_DATE, "num": 0}]

    def update_matches_count(self, player_a: str, player_b: str):
        self.matches_count[player_a] = self.matches_count.get(player_a, 0) + 1
        self.matches_count[player_b] = self.matches_count.get(player_b, 0) + 1

    def update_elo(self, player_a: str, player_b: str, winner: str, level: str, match_date: pd.Timestamp, match_num: int):
        self._ensure_player(player_a)
        self._ensure_player(player_b)

        rA = self._get_last_rating(player_a)
        rB = self._get_last_rating(player_b)

        eA = 1.0 / (1.0 + 10 ** ((rB - rA) / 400.0))
        eB = 1.0 / (1.0 + 10 ** ((rA - rB) / 400.0))

        if winner == player_a:
            sA, sB = 1.0, 0.0
        else:
            sA, sB = 0.0, 1.0

        kA = 250.0 / ((self.matches_count.get(player_a, 0) + 5) ** 0.4)
        kB = 250.0 / ((self.matches_count.get(player_b, 0) + 5) ** 0.4)
        k = 1.1 if str(level) == "G" else 1.0

        rA_new = rA + (k * kA) * (sA - eA)
        rB_new = rB + (k * kB) * (sB - eB)

        self.players[player_a].append({"ranking": float(rA_new), "date": match_date, "num": int(match_num)})
        self.players[player_b].append({"ranking": float(rB_new), "date": match_date, "num": int(match_num)})

    def compute_from_matches(self, matches: pd.DataFrame):
        # first pass: update match counts
        for _, row in matches.iterrows():
            a = row.get("winner_name")
            b = row.get("loser_name")
            if pd.isna(a) or pd.isna(b):
                continue
            self.update_matches_count(str(a), str(b))

        # second pass: compute elos in order
        for _, row in matches.iterrows():
            a = row.get("winner_name")
            b = row.get("loser_name")
            if pd.isna(a) or pd.isna(b):
                continue
            a = str(a)
            b = str(b)
            level = row.get("tourney_level")
            match_date = row.get("tourney_date")
            match_num = int(row.get("match_num", 0))
            # winner is the 'a' (winner_name)
            self.update_elo(a, b, a, level, match_date if not pd.isna(match_date) else FIRST_DATE, match_num)

    # --- reporting utilities
    def summary_players(self) -> pd.DataFrame:
        rows = []
        for name, recs in self.players.items():
            rankings = [r["ranking"] for r in recs]
            if not rankings:
                continue
            rows.append({"name": name, "ranking": max(rankings), "meanr": float(np.mean(rankings)), "medianr": float(np.median(rankings))})
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.sort_values("ranking", ascending=False).reset_index(drop=True)

    def between_dates(self, date1: pd.Timestamp, date2: pd.Timestamp) -> pd.DataFrame:
        rows = []
        for name, recs in self.players.items():
            # filter records between dates
            filtered = [r for r in recs if (r["date"] >= date1 and r["date"] <= date2)]
            if not filtered:
                continue
            rankings = [r["ranking"] for r in filtered]
            rows.append({"name": name, "ranking": max(rankings), "meanr": float(np.mean(rankings)), "medianr": float(np.median(rankings))})
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.sort_values("ranking", ascending=False).reset_index(drop=True)


# --- helpers -------------------------------------------------------------
def get_year(year: int) -> pd.Timestamp:
    return pd.Timestamp(year, 1, 1)


def get_year_month(year: int, month: int) -> pd.Timestamp:
    return pd.Timestamp(year, month, 1)


def greater_equal_year(player_records: List[Dict], year: int) -> pd.DataFrame:
    if not player_records:
        return pd.DataFrame()
    df = pd.DataFrame(player_records)
    return df[df["date"] >= get_year(year)].reset_index(drop=True)


# --- plotting (optional) ------------------------------------------------
def plot_players(players_dict: Dict[str, List[Dict]], names: List[str], start_year: int = 2007):
    import matplotlib.pyplot as plt

    plt.figure()
    for name in names:
        recs = players_dict.get(name)
        if not recs:
            continue
        df = pd.DataFrame(recs)
        df = df[df["date"] >= get_year(start_year)]
        if df.empty:
            continue
        plt.plot(df["date"], df["ranking"], label=name)
    plt.xlabel("Date")
    plt.ylabel("Points")
    plt.title("Elo")
    plt.legend()
    plt.show()


# --- CLI / smoke test and export helpers ---------------------------------

def export_player_timeseries_csv(engine: EloEngine, player: str, out_path: str):
    """Export a single player's Elo time series to CSV."""
    recs = engine.players.get(player)
    if not recs:
        raise ValueError(f"Player not found: {player}")
    df = pd.DataFrame(recs)
    # ensure date column is datetime
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(out_path, index=False)


def export_top_players_timeseries_csv(engine: EloEngine, top_n: int = 5, out_csv: str = "Data/elo_topN.csv") -> str:
    """Create a combined CSV containing Elo time series for the top_n players.

    The CSV will have a 'date' column and one column per player containing the Elo rating
    on each date. Missing values are forward-filled so the series can be plotted directly.
    """
    summary = engine.summary_players()
    if summary.empty:
        raise RuntimeError("No players to export")
    top_players = list(summary.head(top_n)["name"])

    # build per-player DataFrame and merge on date with an outer join
    merged = None
    for name in top_players:
        recs = engine.players.get(name, [])
        if not recs:
            continue
        df = pd.DataFrame(recs)[["date", "ranking"]].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.sort_values("date").drop_duplicates(subset=["date"])  # keep last rating for a date
        df = df.rename(columns={"ranking": name})
        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on="date", how="outer")

    if merged is None or merged.empty:
        raise RuntimeError("No data built for top players")

    merged = merged.sort_values("date").reset_index(drop=True)
    # forward-fill ratings so each date has the most recent rating for each player
    merged = merged.ffill()

    # ensure output directory exists
    out_dir = os.path.dirname(out_csv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    merged.to_csv(out_csv, index=False)
    return out_csv


def export_top_n_players_timeseries_csv(engine: EloEngine, top_n: int, out_csv: str) -> str:
    """Export combined timeseries for top_n players by peak Elo."""
    summary = engine.summary_players()
    if summary.empty:
        raise RuntimeError("No players to export")
    top_players = list(summary.head(top_n)["name"])

    merged = None
    for name in top_players:
        recs = engine.players.get(name, [])
        if not recs:
            continue
        df = pd.DataFrame(recs)[["date", "ranking"]].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.sort_values("date").drop_duplicates(subset=["date"])  # keep last rating for a date
        df = df.rename(columns={"ranking": name})
        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on="date", how="outer")

    if merged is None or merged.empty:
        raise RuntimeError("No data built for top players")

    merged = merged.sort_values("date").reset_index(drop=True)
    merged = merged.ffill()

    out_dir = os.path.dirname(out_csv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    merged.to_csv(out_csv, index=False)
    return out_csv


def export_player_from_start_to_present(engine: EloEngine, player_name: str, out_csv: str) -> str:
    """Export a single player's Elo time series starting from their first non-baseline rating to the present.

    Output columns: date, <player_name>
    """
    # Build combined dataframe for all players
    # Reuse export_latest_n_players_timeseries_csv's merging strategy by creating a combined DF of all players
    merged = None
    for name, recs in engine.players.items():
        if not recs:
            continue
        df = pd.DataFrame(recs)[["date", "ranking"]].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.sort_values("date").drop_duplicates(subset=["date"])  # keep last rating for a date
        df = df.rename(columns={"ranking": name})
        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on="date", how="outer")

    if merged is None or merged.empty:
        raise RuntimeError("No data built for players")

    merged = merged.sort_values("date").reset_index(drop=True)
    merged = merged.ffill()

    if player_name not in merged.columns:
        raise ValueError(f"Player '{player_name}' not found in merged timeseries")

    ser = merged[["date", player_name]].copy()
    mask = ser[player_name].notna() & (ser[player_name] != 1500)
    if not mask.any():
        # write full series (will be baseline only)
        ser.to_csv(out_csv, index=False)
        return out_csv

    first_idx = mask.idxmax()
    out_df = ser.iloc[first_idx:].reset_index(drop=True)

    out_dir = os.path.dirname(out_csv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    out_df.to_csv(out_csv, index=False)
    return out_csv


def export_players_from_reference_start(engine: EloEngine, reference_player: str, out_csv: str) -> str:
    """Export combined timeseries for all players whose first non-baseline date is on/after the reference player's start.

    The CSV will have a 'date' column and one column per selected player. Missing values are forward-filled.
    """
    # First build merged timeseries for all players
    merged = None
    first_dates = {}
    for name, recs in engine.players.items():
        if not recs:
            continue
        df = pd.DataFrame(recs)[["date", "ranking"]].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.sort_values("date").drop_duplicates(subset=["date"])  # keep last rating for a date
        df = df.rename(columns={"ranking": name})
        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on="date", how="outer")

        # detect first real date for this player
        mask = df[name].notna() & (df[name] != 1500)
        if mask.any():
            first_dates[name] = df.loc[mask, 'date'].iloc[0]
        else:
            first_dates[name] = None

    if merged is None or merged.empty:
        raise RuntimeError("No data built for players")

    merged = merged.sort_values("date").reset_index(drop=True)
    merged = merged.ffill()

    if reference_player not in first_dates or first_dates.get(reference_player) is None:
        raise ValueError(f"Reference player '{reference_player}' has no detectable start date")

    ref_start = first_dates[reference_player]

    # select players whose start is on/after ref_start
    selected = [p for p, d in first_dates.items() if d is not None and d >= ref_start]
    if not selected:
        raise RuntimeError("No players found who started on/after reference player's start")

    out_df = merged[['date'] + selected].copy()

    out_dir = os.path.dirname(out_csv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    out_df.to_csv(out_csv, index=False)
    return out_csv


def export_all_players_timeseries_csv(engine: EloEngine, out_csv: str = "Data/elo_all_players.csv") -> str:
    """Create a combined CSV with Elo time series for every player.

    The CSV will have a 'date' column and one column per player. Missing values are forward-filled.
    For large numbers of players this may produce a wide CSV; use with care.
    """
    # build per-player DataFrame and merge on date with an outer join
    merged = None
    for name, recs in engine.players.items():
        if not recs:
            continue
        df = pd.DataFrame(recs)[["date", "ranking"]].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.sort_values("date").drop_duplicates(subset=["date"])  # keep last rating for a date
        df = df.rename(columns={"ranking": name})
        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on="date", how="outer")

    if merged is None or merged.empty:
        raise RuntimeError("No data built for players")

    merged = merged.sort_values("date").reset_index(drop=True)
    merged = merged.ffill()

    out_dir = os.path.dirname(out_csv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    merged.to_csv(out_csv, index=False)
    return out_csv


def export_latest_n_players_timeseries_csv(engine: EloEngine, n: int, out_csv: str = "Data/elo_latestN.csv") -> str:
    """Export combined timeseries for the latest n players by their most recent rating.

    This selects the players with the highest most-recent recorded rating (or most recent entry if NaNs),
    then builds the merged timeseries similarly to other export functions.
    """
    # build a DataFrame of latest ratings per player
    rows = []
    for name, recs in engine.players.items():
        if not recs:
            continue
        last = recs[-1]
        rows.append({"name": name, "last_date": pd.to_datetime(last["date"]), "last_rating": float(last["ranking"])})
    if not rows:
        raise RuntimeError("No player records available")
    latest_df = pd.DataFrame(rows)
    # sort by last_date descending then by last_rating descending to get 'latest' players
    latest_df = latest_df.sort_values(["last_date", "last_rating"], ascending=[False, False]).reset_index(drop=True)
    selected = list(latest_df.head(n)["name"])

    # merge selected players' timeseries
    merged = None
    for name in selected:
        recs = engine.players.get(name, [])
        if not recs:
            continue
        df = pd.DataFrame(recs)[["date", "ranking"]].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.sort_values("date").drop_duplicates(subset=["date"])  # keep last rating for a date
        df = df.rename(columns={"ranking": name})
        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on="date", how="outer")

    if merged is None or merged.empty:
        raise RuntimeError("No data built for selected players")

    merged = merged.sort_values("date").reset_index(drop=True)
    merged = merged.ffill()

    out_dir = os.path.dirname(out_csv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    merged.to_csv(out_csv, index=False)
    return out_csv


def main():
    files = find_match_files(root=".")
    # prefer Data/SingleMatches directory if present
    if not files:
        files = find_match_files(root="Data/SingleMatches")
    matches = load_matches(files)
    if matches.empty:
        print("No match files found or no data loaded.")
        return
    engine = EloEngine()
    engine.compute_from_matches(matches)
    summary = engine.summary_players()
    print("Top 10 players by peak Elo (calculated):")
    print(summary.head(10).to_string(index=False))

    # Export top-5 players time series to CSV for plotting
    try:
        out_csv = "Data/elo_top5.csv"
        path = export_top_players_timeseries_csv(engine, top_n=5, out_csv=out_csv)
        print(f"Saved top-5 Elo time series to: {os.path.abspath(path)}")
    except Exception as e:
        print(f"Failed to export top players CSV: {e}")
        
        # ------------------------------------------------------------------
    # Extra export: Elo ratings for CLAY matches only (1990–2024)
    # ------------------------------------------------------------------
    try:
        clay_matches = load_clay_matches_1990_2024(files)
        if clay_matches.empty:
            print("No clay matches found between 2000 and 2024 – skipping clay Elo export.")
        else:
            clay_engine = EloEngine()
            clay_engine.compute_from_matches(clay_matches)

            clay_out_csv = "Data/elo_clay_2000_2024.csv"
            export_all_players_timeseries_csv(clay_engine, out_csv=clay_out_csv)

            print(f"Saved clay Elo (2000–2024) to: {os.path.abspath(clay_out_csv)}")
    except Exception as e:
        print(f"Failed to export clay Elo CSV: {e}")


    # # Export Roger Federer's time series from start to present
    # try:
    #     out_csv = "Data/elo_federer_from_start.csv"
    #     path = export_player_from_start_to_present(engine, player_name="Roger Federer", out_csv=out_csv)
    #     print(f"Saved Roger Federer's Elo time series from start to present to: {os.path.abspath(path)}")
    # except Exception as e:
    #     print(f"Failed to export Roger Federer's time series CSV: {e}")

    # Export all players who started on/after Roger Federer's start
    try:
        out_csv = "Data/elo_from_federer_start.csv"
        path = export_players_from_reference_start(engine, reference_player="Roger Federer", out_csv=out_csv)
        print(f"Saved players starting from Federer's career start to: {os.path.abspath(path)}")
    except Exception as e:
        print(f"Failed to export players-from-Federer-start CSV: {e}")
    # Export top-400 players time series to CSV for plotting
    try:
        out_csv = "Data/elo_top400.csv"
        path = export_top_n_players_timeseries_csv(engine, top_n=400, out_csv=out_csv)
        print(f"Saved top-400 Elo time series to: {os.path.abspath(path)}")
    except Exception as e:
        print(f"Failed to export top players CSV: {e}")

    # Export latest 400 players' time series to CSV
    try:
        out_csv = "Data/elo_latest400.csv"
        path = export_latest_n_players_timeseries_csv(engine, n=400, out_csv=out_csv)
        print(f"Saved latest 400 players' Elo time series to: {os.path.abspath(path)}")
    except Exception as e:
        print(f"Failed to export latest players CSV: {e}")

    # Export all players' time series to a combined CSV
    # try:
    #     out_csv = "Data/elo_all_players.csv"
    #     path = export_all_players_timeseries_csv(engine, out_csv=out_csv)
    #     print(f"Saved all players' Elo time series to: {os.path.abspath(path)}")
    # except Exception as e:
    #     print(f"Failed to export all players CSV: {e}")


if __name__ == "__main__":
    main()