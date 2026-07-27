import pandas as pd
import analyse
from in_out import write_records
import ingest
import argparse

CSV_PATH = "assets/expenses_FY26.csv"
OUTPUT_PATH = "assets/records.json"


def run_ingest():
    df = pd.read_csv(CSV_PATH)
    records = ingest.run_pipeline(df)
    write_records(records, OUTPUT_PATH)


def main():
    # Optional 'ingest' param to run ingest pipeline which processes the csv file
    # and normalises transactions to a list of Record objects. Must be run on first launch
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ingest", action="store_true", help="Run ingest pipeline before analysis"
    )
    args = parser.parse_args()

    if args.ingest:
        run_ingest()

    analyse.launch()


if __name__ == "__main__":
    main()
