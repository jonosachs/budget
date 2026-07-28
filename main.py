"""Launcher for the dashboard.

    python main.py

Data is uploaded through the app itself and lives only in the browser session,
so there is no ingest step here and nothing to prepare first. For local work,
set LOCAL_RECORDS in .env to seed each session from a records file.
"""

import analyse


def main():
    analyse.launch()


if __name__ == "__main__":
    main()
