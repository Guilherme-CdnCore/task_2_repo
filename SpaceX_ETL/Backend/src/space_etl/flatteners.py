from __future__ import annotations
from typing import Any, Dict, List 
from .utils import (
    parse_datetime_utc_to_lisbon, 
    safe_get, 
    compute_content_hash
)
import logging
import hashlib
import json



logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  


#------------------#
#    FACT TABLE    #
#------------------#


def flatten_launch(launch: Dict[str, Any]) -> Dict[str, Any]:
    flat = {}
    try:
        # Basic fields
        flat["id"] = launch.get("id")
        flat["flight_number"] = launch.get("flight_number")
        flat["name"] = launch.get("name")

        # Handle date parsing
        date_utc = launch.get("date_utc")
        logger.debug(f"[flatten_launch] Parsing date_utc: {date_utc} for launch id: {flat['id']}")
        
        # Try to parse the date, but don't fail the entire launch if it fails
        try:
            flat["date_lisbon"] = parse_datetime_utc_to_lisbon(date_utc)
            if flat["date_lisbon"] is None and date_utc:
                # Log warning but continue processing
                logger.warning(f"[flatten_launch] Could not parse date_utc '{date_utc}' for launch id {flat['id']}")
        except Exception as e:
            logger.error(f"[flatten_launch] Exception parsing date_utc '{date_utc}' for launch id {flat['id']}: {e}")
            flat["date_lisbon"] = None
        
        flat["date_local"] = launch.get("date_local")
        flat["success"] = launch.get("success")
        flat["details"] = launch.get("details")

        # Rocket and cores
        flat["rocket"] = launch.get("rocket")
        flat["launchpad"] = launch.get("launchpad")

        cores = launch.get("cores", [])
        for i, core in enumerate(cores):
            flat[f'core_{i}_id'] = core.get('core')
            flat[f'core_{i}_flight'] = core.get('flight')
            flat[f'core_{i}_reused'] = core.get('reused')
            flat[f'core_{i}_landing_attempt'] = core.get('landing_attempt')
            flat[f'core_{i}_landing_success'] = core.get('landing_success')
            flat[f'core_{i}_landing_type'] = core.get('landing_type')

        # Payloads
        payloads = launch.get("payloads", [])
        for i, payload in enumerate(payloads):
            flat[f'payload_{i}_id'] = payload

        # Failures
        failures = launch.get("failures", [])
        for i, failure in enumerate(failures):
            flat[f'failure_{i}_time'] = failure.get('time')
            flat[f'failure_{i}_altitude'] = failure.get('altitude')
            flat[f'failure_{i}_reason'] = failure.get('reason')

        # Fairings
        fairings = launch.get("fairings")
        if fairings:
            flat["fairings_reused"] = fairings.get("reused")
            flat["fairings_recovery_attempt"] = fairings.get("recovery_attempt")
            flat["fairings_recovered"] = fairings.get("recovered")

        # Links
        links = launch.get("links", {})
        patch = links.get("patch", {})
        flat["patch_small"] = patch.get("small")
        flat["patch_large"] = patch.get("large")
        flat["webcast"] = links.get("webcast")
        flat["article"] = links.get("article")
        flat["wikipedia"] = links.get("wikipedia")

        # Generate hash
        hash_source = {
            "id": flat.get("id"),
            "flight_number": flat.get("flight_number"),
            "name": flat.get("name"),
            "date_lisbon": flat.get("date_lisbon"),
        }

        hash_str = json.dumps(hash_source, sort_keys=True)
        flat["static_hash"] = hashlib.md5(hash_str.encode('utf-8')).hexdigest()

        return flat

    except Exception as e:
        logger.error(f"[flatten_launch] Critical error flattening launch: {launch.get('id')} | {e}")
        
        return {
            "id": launch.get("id"),
            "flight_number": launch.get("flight_number"),
            "name": launch.get("name"),
            "date_lisbon": None,
            "date_local": launch.get("date_local"),
            "success": launch.get("success"),
            "rocket": launch.get("rocket"),
            "launchpad": launch.get("launchpad"),
            "details": launch.get("details"),
            "static_hash": None
        }


#------------------#                                                       
# DIMENSION TABLE  #
#------------------#


def flatten_rocket(raw: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    try:
        logger.debug(f"Flattening rocket with id=%s, name=%s", raw.get("id"), raw.get("name"))

        row["id"] = raw.get("id")
        row["name"] = raw.get("name")
        row["company"] = raw.get("company")
        row["country"] = raw.get("country")
        row["active"] = raw.get("active")
        row["first_flight"] = raw.get("first_flight")
        row["cost_per_launch"] = raw.get("cost_per_launch")
        row["success_rate_pct"] = raw.get("success_rate_pct")

        #Physical
        height = raw.get("height", {})
        diameter = raw.get("diameter", {})
        mass = raw.get("mass", {})

        row["height_meters"] = height.get("meters")
        row["diameter_meters"] = diameter.get("meters")
        row["mass_kg"] = mass.get("kg")
        row["stages"] = raw.get("stages")
        row["boosters"] = raw.get("boosters")

        #Engines
        engines = raw.get("engines", {})
        isp = engines.get("isp", {})
        thrust_vacuum = engines.get("thrust_vacuum", {})

        row["engines_number"] = engines.get("number")
        row["engines_type"] = engines.get("type")
        row["engines_version"] = engines.get("version")
        row["engines_isp_vacuum"] = isp.get("vacuum")
        row["engines_thrust_vacuum_kN"] = thrust_vacuum.get("kN")

        #First stage
        first_stage = raw.get("first_stage", {})
        thrust_vacuum_fs = first_stage.get("thrust_vacuum", {})

        row["first_stage_engines"] = first_stage.get("engines")
        row["first_stage_reusable"] = first_stage.get("reusable")
        row["first_stage_thrust_vacuum_kN"] = thrust_vacuum_fs.get("kN")


        #Second stage
        second_stage = raw.get("second_stage", {})
        thrust_ss = second_stage.get("thrust", {})

        row["second_stage_engines"] = second_stage.get("engines")
        row["second_stage_reusable"] = second_stage.get("reusable")
        row["second_stage_thrust_kN"] = thrust_ss.get("kN")


        #Payload weights 

        for payload in raw.get("payload_weights", []):
            pid = payload.get("id")
            kg = payload.get("kg")
            if pid and kg is not None:
                row[f"{pid}_kg"] = kg
            else:
                logger.warning("Invalid payload weight entry in rocket id=%s: %s", row["id"], payload)

        #links
        row["wikipedia"] = raw.get("wikipedia")
        images = raw.get("flickr_images", [])
        row["image"] = images[0] if images else None 
        if not images:
            logger.info("No images found for rocket id=%s", row["id"])

        logger.debug("Rocket flattening complete for id=%s", row["id"])

        return row
    
    except Exception as e:
        logger.error(f"Error flattening rocket id=%s: {e}", raw.get("id"))
        return {}



def flatten_launchpad(raw: Dict[str, Any]) -> Dict[str, Any]:
    
    row: Dict[str, Any] = {}
    row["id"] = raw.get("id")
    row["name"] = raw.get("name")
    row["full_name"] = raw.get("full_name")
    row["locality"] = raw.get("locality")
    row["region"] = raw.get("region")
    row["latitude"] = raw.get("latitude")
    row["longitude"] = raw.get("longitude")
    row["status"] = raw.get("status")
    row["launch_attempts"] = raw.get("launch_attempts")
    row["launch_successes"] = raw.get("launch_successes")

    
    row["timezone"] = raw.get("timezone")         # e.g. "America/
    row["details"] = raw.get("details")           # free-text description
    rockets = raw.get("rockets",  []) 
    row["rockets_count"] = len(rockets)           # keep flat; bridge table 
    launches = raw.get("launches") 
    row["launches_count"] = len(launches)

    row["static_hash"] = compute_content_hash(raw)
    return row



def flatten_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    
    row: Dict[str, Any] = {}

    row["id"] = raw.get("id")
    row["name"] = raw.get("name")   
    row["type"] = raw.get("type")
    row["reused"] = raw.get("reused")
    row["launch_id"] = raw.get("launch")
    row["mass_kg"] = raw.get("mass_kg")
    row["mass_lbs"] = raw.get("mass_lbs")
    row["orbit"] = raw.get("orbit")
    row["customers"] = ", ".join(raw.get("customers", []))  
    row["nationalities"] = ", ".join(raw.get("nationalities", []))
    row["manufacturer"] = raw.get("manufacturer")
    row["wikipedia"] = raw.get("wikipedia")

    return row



def flatten_core(raw: Dict[str, Any]) -> Dict[str, Any]:
    
    row: Dict[str, Any] = {}

    row["id"] = raw.get("id")
    row["serial"] = raw.get("serial")
    row["block"] = raw.get("block")
    row["status"] = raw.get("status")
    row["reuse_count"] = raw.get("reuse_count")
    row["rtls_landings"] = raw.get("rtls_landings")
    row["asds_landings"] = raw.get("asds_landings")
    row["water_landings"] = raw.get("water_landings")
    row["launches_count"] = len(raw.get("launches", []))
    row["missions"] = ", ".join(raw.get("missions", []))

    return row


#------------------#
#   BRIDGE TABLE   #
#------------------#


def flatten_launch_payload_bridge(launch: Dict[str, Any]) -> List[Dict[str, Any]]:

    bridges_rows= []
    for payload_id in launch.get("payloads", []):
        bridges_rows.append({
            "launch_id": launch.get("id"),
            "payload_id": payload_id
        })

    return bridges_rows


def flatten_launch_core_bridge(launch: Dict[str, Any]) -> List[Dict[str, Any]]:
    bridge_rows = []
    for i, core in enumerate(launch.get("cores", [])):
        bridge_rows.append({
            "launch_id": launch.get("id"),  
            "core_id": core.get("core"),
            "core_flight": core.get("flight")
        })

    return bridge_rows
        


