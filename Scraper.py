import re
import sys
import time
import pathlib
import pandas as pd

URL = "https://tennisabstract.com/reports/atp_elo_ratings.html"
OUT = pathlib.Path("atp_elo_ratings.csv") 

def clean_cols(df):
    cols = [str(c) for c in df.columns]
    return [re.sub(r"\s+", "_", c.strip().lower()) for c in cols]

def fetch_table(url: str, retries: int = 2, delay: float = 1.5) -> pd.DataFrame:
    last_err = None
    for _ in range(retries + 1):
        try:
            tables = pd.read_html(url)   
            for t in tables:
                t.columns = clean_cols(t)
                if any("player" in c for c in t.columns):
                    return t
            raise RuntimeError("No table with a 'player' column was found.")
        except Exception as e:
            last_err = e
            time.sleep(delay)
    raise last_err

def main():
    df = fetch_table(URL)
    want = ["player","age","elo","helo","helo_rank","celo","celo_rank",
            "gelo","gelo_rank","peak_elo","peak_month","atp_rank","log_diff"]
    keep = [c for c in want if c in df.columns]
    df = df[keep].copy()

    # Save CSV
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df):,} rows to: {OUT.resolve()}")
    print(df.head().to_string(index=False))

if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print("Missing parser dependency. Install one of:", file=sys.stderr)
        print("  pip install lxml    # faster", file=sys.stderr)
        print("  # or", file=sys.stderr)
        print("  pip install html5lib", file=sys.stderr)
        raise
