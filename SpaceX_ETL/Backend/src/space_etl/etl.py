import logging
from typing import Dict, Any
from typing import Tuple
import os
import json
from .anomalies import monthly_success_rates, detect_anomalies
from datetime import datetime
from dateutil.parser import parse  # Already there
import pytz 


from .flatteners import (
    flatten_launch,
    flatten_rocket,
    flatten_launchpad,
    flatten_payload,
    flatten_core,
    flatten_launch_payload_bridge,
    flatten_launch_core_bridge,
)

from .exceptions_file import (
    DataExtractionError,
    DataTransformationError,
    DataLoadError,
)

from .utils import (
    read_json_files,
    write_json_file,
    write_csv_file
)
from .storyteller import write_stories

# Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



def extract_data(source_path: str):
    try:
        data = read_json_files(source_path)
        if data is None:
            raise DataExtractionError(f"Could not extract data from {source_path}")
        return data
    
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise DataExtractionError(e)


def transform_data(raw_data: Dict[str, Any]) -> Dict[str, Any]:

    flat_launches = []
    try:
        
        for launch in raw_data.get("launches", []):
            flat = flatten_launch(launch)
            if flat and flat.get("id"):  # Keep if has at least an ID
                flat_launches.append(flat)
            else:
                logger.warning(f"Skipping launch due to flattening error: {launch.get('id', 'unknown')}")


        flat_rockets = [flatten_rocket(r) for r in raw_data.get("rockets", [])]
        flat_launchpads = [flatten_launchpad(lp) for lp in raw_data.get("launchpads", [])]
        flat_payloads = [flatten_payload(p) for p in raw_data.get("payloads", [])]
        flat_cores = [flatten_core(c) for c in raw_data.get("cores", [])]

        # Bridge tables
        launch_payload_bridge = []
        launch_core_bridge = []
        for launch in raw_data.get("launches", []):
            launch_payload_bridge.extend(flatten_launch_payload_bridge(launch))
            launch_core_bridge.extend(flatten_launch_core_bridge(launch))

        return {
            "launches": flat_launches,
            "rockets": flat_rockets,
            "launchpads": flat_launchpads,
            "payloads": flat_payloads,
            "cores": flat_cores,
            "launch_payload_bridge": launch_payload_bridge,
            "launch_core_bridge": launch_core_bridge
        }
     

    except Exception as e:
        logger.error(f"Transformation failed: {e}")
        raise DataTransformationError(e)



def load_all_data(flat_data, output_dir: str):
    try:
        for key, dataset in flat_data.items():

            validator = validate_launch if key == "launches" else lambda x: (True, "")
            clean, rejects = apply_quality_gates(dataset, validator)

            #write clean data
            os.makedirs(f"{output_dir}/clean", exist_ok=True)
            output_file = f"{output_dir}/clean/{key}.csv"
            fieldnames = clean[0].keys() if clean else []
            write_csv_file(output_file, clean, fieldnames)

            logger.info(f"Loaded {len(clean)} clean records to {output_file}")


            #write rejects
            if rejects:
                os.makedirs(f"{output_dir}/out/rejects", exist_ok=True)
                reject_file = f"{output_dir}/out/rejects/{key}.csv"
                reject_fieldnames = rejects[0].keys() if rejects else []
                write_csv_file(reject_file, rejects, reject_fieldnames)

                logger.info(f"Loaded {len(rejects)} rejected records to {reject_file}")

            logger.info(f"Loaded {len(dataset)} records into {output_file}")

    except Exception as e:
        logger.error(f"Load failed: {e}")
        raise DataLoadError(e)



def validate_launch(row: Dict[str, Any]) -> Tuple[bool, str]:

    if not row.get("date_lisbon"):
        reason = "Missing or invalid date_lisbon"
        logger.debug(f"Rejecting {row.get('id')} because {reason}")
        return False, reason

    try:
        launch_date = parse(row["date_lisbon"])
        if launch_date > datetime.now(pytz.UTC):
            reason = "Future date_lisbon"
            logger.debug(f"Rejecting {row.get('id')} because {reason}")
            return False, reason
    except Exception:
        reason = "Invalid date_lisbon format"
        logger.debug(f"Rejecting {row.get('id')} because {reason}")
        return False, reason

    if row.get("rocket") is None:
        reason = "Missing rocket"
        logger.debug(f"Rejecting {row.get('id')} because {reason}")
        return False, reason
    
    if row.get("success") not in [True, False, None]:
        reason = "Invalid success value"
        logger.debug(f"Rejecting {row.get('id')} because {reason}")
        return False, reason
    
    if row.get("id") is None:
        reason = "Missing id"
        logger.debug(f"Rejecting {row.get('name')} because {reason}")
        return False, reason
    
    if row.get("launchpad") is None:
        reason = "Missing launchpad"
        logger.debug(f"Rejecting {row.get('id')} because {reason}")
        return False, reason
    
    if row.get("name") is None:
        reason = "Missing name"
        logger.debug(f"Rejecting {row.get('id')} because {reason}")
        return False, reason
    
    if row.get("flight_number") is None:
        reason = "Missing flight_number"
        logger.debug(f"Rejecting {row.get('id')} because {reason}")
        return False, reason

    return True, ""




def apply_quality_gates(dataset: list, validator) -> Tuple[list, list]:
    clean, rejects = [], []
    for row in dataset:
        if row is None:
            logging.warning("Skipping None row in dataset")
            continue
        
        valid, reason = validator(row)
        if valid:
            clean.append(row)
        else:
            reject_row = row.copy()
            reject_row["reject_reason"] = reason
            rejects.append(reject_row)
            
            logging.warning(f"[REJECTED] Dataset ID: {row.get('id', 'unknown')} | Name: {row.get('name', 'unknown')} | Reason: {reason}")

    return clean, rejects



def write_quality_report(report: dict, output_dir: str):

    os.makedirs(f"{output_dir}/out", exist_ok=True)
    report["assignment_signature"] = "SPX-ETL-2025-IdrA"
    logger.info(f"Report content: {report}")

    if not report:
        logger.warning(f"Warning: Report is empty")

    with open(f"{output_dir}/out/quality_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def summarize_quality(flat_data: dict, rejects: dict, anomalies: list) -> dict:
    summary = {}
    for key in flat_data:
        summary[key] = {
            "clean_count": len(flat_data[key]),
            "reject_count": len(rejects.get(key, [])),
            "reject_reasons": list(set(r.get("reject_reason", "") for r in rejects.get(key, [])))
        }
    summary["anomalies"] = anomalies
    return summary



def etl_pipeline(raw_data: Dict[str, Any], output_dir: str):
   
    flat_data = transform_data(raw_data)

    all_clean = {}
    all_rejects = {}
    anomalies = []
    for key, dataset in flat_data.items():
        validator = validate_launch if key == "launches" else lambda x: (True, "")
        clean, rejects = apply_quality_gates(dataset, validator)
        all_clean[key] = clean
        all_rejects[key] = rejects
        if rejects:
            anomalies.append(f"{len(rejects)} rejects in {key}")

    load_all_data(flat_data, output_dir)


    #run anomaly detect on launches
    launch_rates = monthly_success_rates(all_clean.get("launches", []))
    anomalies = detect_anomalies(launch_rates)

   # Generate quality report
    report = summarize_quality(all_clean, all_rejects, anomalies)
    write_quality_report(report, output_dir)

    #  mission stories part for  non tech people 
    try:
        story_path = write_stories(all_clean.get("launches", []), output_dir)
        logger.info(f"Wrote mission stories to {story_path}")
    except Exception as e:
        logger.warning(f"Could not write mission stories: {e}")

   