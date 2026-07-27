import pandas as pd
import analyse
from in_out import write_records
import ingest
import argparse

CSV_PATH = "assets/expenses_FY26.csv"
OUTPUT_PATH = "assets/records.json"
BANK = "NAB"


def run_ingest():
    df = pd.read_csv(CSV_PATH)
    records = ingest.run_pipeline(df, BANK)
    write_records(records, OUTPUT_PATH)


def main():
    # Optional CLI param to ingest csv file
    # Must be run at least once on first launch
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", action="store_true", help="Ingest csv file")
    args = parser.parse_args()

    if args.ingest:
        run_ingest()

    analyse.launch()


if __name__ == "__main__":
    main()
