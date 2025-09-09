from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime
import hashlib
import json
import csv
import re
import logging
import os
import pytz
from typing import Optional
from dateutil.parser import parse

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  


#------------------#
#      TIME        #
#------------------#
EUROPE_LISBON = pytz.timezone("Europe/Lisbon")
UTC = pytz.UTC


def parse_datetime_utc_to_lisbon(dt_str):
    if not dt_str:
        raise ValueError("Empty datetime string")
    try:
        dt_utc = parse(dt_str)
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=pytz.UTC)
        dt_lisbon = dt_utc.astimezone(pytz.timezone("Europe/Lisbon"))
        return dt_lisbon.isoformat()
    except Exception as e:
        logger.error(f"Failed to parse datetime '{dt_str}': {e}")
        raise ValueError(f"Invalid datetime format: {dt_str}")


def find_invalid_datetimes(datetime_strings: list[str]) -> list[str]:
    invalid = []
    for dt_str in datetime_strings:
        parsed = parse_datetime_utc_to_lisbon(dt_str)
        if parsed is None:
            logger.warning(f"Failed to parse datetime string: {repr(dt_str)}")
            invalid.append(dt_str)
    return invalid



#------------------#
#       DICT       #
#------------------#

def safe_get(obj: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    
    cur = obj
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur 
        

#------------------#
#       HASH       #
#------------------#


def compute_content_hash(obj: Any) -> str:

    s = json.dumps(obj, sort_keys = True, separators = (",", ":"), ensure_ascii = False)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


#------------------#
#       I/O        #
#------------------#

def read_json_files(path: str) ->Any:

    try:
        with open(path, "r", encoding = "utf-8") as f:
            return json.load(f)
        
    except Exception as e:
        logger.error(f"Error reading JSON file {path}: {e}")
        return None
    
def write_json_file(path: str, data: Any) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok = True)
        with open(path, "w", encoding = "utf-8") as f:
            json.dump(data, f, indent = 2, ensure_ascii = False)
    except Exception as e:
        logger.error(f"Error writing JSON file {path}: {e}")


def write_csv_file(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok = True)
    with open(path, "w", newline="", encoding = "utf-8") as f:
        writer = csv.DictWriter(f, fieldnames = fieldnames, extrasaction='ignore') 
        writer.writeheader()

        for row in rows:
            writer.writerow(row)



  