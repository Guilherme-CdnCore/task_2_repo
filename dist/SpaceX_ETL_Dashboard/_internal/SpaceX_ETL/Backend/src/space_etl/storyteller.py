import os
import random
from datetime import datetime
from typing import List, Dict


def _pick(template_list: List[str]) -> str:
    return random.choice(template_list)


def _fmt_date(iso_string: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
        return dt.strftime("%b %d, %Y")
    except Exception:
        return iso_string or "an unknown date"


def generate_story(launch: Dict) -> str:
    name = launch.get("name") or launch.get("mission_name") or "Unnamed Mission"
    rocket = launch.get("rocket_name") or launch.get("rocket") or "mystery rocket"
    pad = launch.get("launchpad_name") or launch.get("launchpad") or "an undisclosed pad"
    date = _fmt_date(launch.get("date_lisbon") or launch.get("date_utc") or "")
    success = launch.get("success")
    payloads = launch.get("payloads") or []
    payload_phrase = _pick([
        "a treasure of satellites",
        "an experimental gizmo",
        "a secret payload (shh)",
        "a constellation segment",
        "a cosmic care package",
    ]) if not payloads else f"{len(payloads)} payload(s)"

    opener = _pick([
        f"On {date}, {name} rolled out with quiet confidence.",
        f"{date}: The world watched as {name} readied for flight.",
        f"A crisp {date} morning set the stage for {name}.",
    ])

    ascent = _pick([
        f"Engines thundered and {rocket} leapt from {pad} as if eager to touch the stars.",
        f"Flames bloomed under {rocket}, painting {pad} with sunrise hues.",
        f"With a steady rumble, {rocket} climbed, carving a bright arc over {pad}.",
    ])

    if success is True:
        outcome = _pick([
            f"Minutes later, fairings opened and {payload_phrase} sailed toward destiny.",
            f"Orbit achieved. Cheers erupted as telemetry sang a flawless tune.",
            f"All systems nominal. Another page of spacefaring routine—quietly extraordinary.",
        ])
    elif success is False:
        outcome = _pick([
            "An anomaly cut the flight short, but the data will fly again.",
            "A harsh day for rocketry; lessons logged, resolve reinforced.",
            "Not today—but every setback sharpens tomorrow's launch.",
        ])
    else:
        outcome = _pick([
            "Telemetry fades into the horizon; the verdict remains a mystery.",
            "Outcome pending. Space keeps its secrets—for now.",
            "Records are quiet on what followed. The sky remembers.",
        ])

    closer = _pick([
        "Somewhere, a team smiles at a graph only they can love.",
        "Another mission, another step—space is patient.",
        "In the control room, coffee cools; the dream does not.",
    ])

    return f"{opener} {ascent} {outcome} {closer}"


def write_stories(launches: List[Dict], output_dir: str) -> str:
    os.makedirs(f"{output_dir}/out", exist_ok=True)
    path = f"{output_dir}/out/mission_stories.txt"
    lines = []
    for launch in launches[:50]:  # cap to keep it readable
        story = generate_story(launch)
        title = launch.get("name") or launch.get("mission_name") or launch.get("id") or "Mission"
        lines.append(f"== {title} ==\n{story}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) if lines else "No launches to tell stories about yet.\n")
    return path


