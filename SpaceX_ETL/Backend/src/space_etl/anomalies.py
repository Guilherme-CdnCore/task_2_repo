from typing import List, Dict
import statistics

def monthly_success_rates(launches: List[Dict]) -> Dict[str, float]:
    rates = {}
    for launch in launches:
        month = launch["date_lisbon"][:7]  # YYYY-MM
        rates.setdefault(month, {"success": 0, "total": 0})
        if launch.get("success") is True:
            rates[month]["success"] += 1
        rates[month]["total"] += 1
    
    return {m: v["success"] / v["total"] for m, v in rates.items() if v["total"] > 0}

def detect_anomalies(rates: Dict[str, float]) -> List[str]:
    values = list(rates.values())
    if len(values) < 24:
        return []
    trailing_mean = statistics.mean(values[-24:])
    trailing_stdev = statistics.stdev(values[-24:])
    anomalies = [month for month, rate in rates.items() if abs(rate - trailing_mean) > 3 * trailing_stdev]
    
    return anomalies