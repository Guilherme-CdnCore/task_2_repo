from src.space_etl.etl import etl_pipeline
from src.space_etl.utils import read_json_files
import logging

logging.basicConfig(level=logging.INFO)


def main():
    raw_data ={
        "launches": read_json_files("data/raw/launches.json"),
        "rockets": read_json_files("data/raw/rockets.json"),
        "launchpads": read_json_files("data/raw/launchpads.json"),
        "payloads": read_json_files("data/raw/payloads.json"),
        "cores": read_json_files("data/raw/cores.json")
    }

    output_dir = "data"
    logging.info("Starting ETL process...")

    etl_pipeline(raw_data, output_dir)
    logging.info("ETL process completed successfully.")

if __name__ == "__main__":
    main()
