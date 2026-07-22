import pandas as pd
from in_out import write_records
from process import run_pipeline as process_df


CSV_PATH = "assets/expenses_FY26.csv"
OUTPUT_PATH = "assets/records.json"


def main():
    # Read in csv as dataframe
    df = pd.read_csv(CSV_PATH)

    records = process_df(df)

    write_records(records, OUTPUT_PATH)

    print("Finished.")


if __name__ == "__main__":
    main()
