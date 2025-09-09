# SpaceX ETL Data Contract

## Overview
This document defines the data contract for the SpaceX ETL pipeline, including schema, types, nullability, semantic rules, and quality expectations.

---

## Table Schemas

### fact_launches
|     Column    |   Type    | Nullable |           Description            |  Allowed Values / Range   |
|---------------|-----------|----------|----------------------------------|---------------------------|
| id            | string    | No       | Launch ID                        |                           |
| flight_number | int       | No       | Flight number                    | >= 1                      |
| name          | string    | No       | Launch name                      |                           |
| date_lisbon   | datetime  | No       | Launch date (Europe/Lisbon)      | Past/Present only         |
| success       | bool      | Yes      | Launch success                   | True / False / Null       |
| rocket        | string    | No       | Rocket ID                        |                           |
| launchpad     | string    | No       | Launchpad ID                     |                           |
| ...           | ...       | ...      | ...                              | ...                       |

---

### dim_rocket
| Column |  Type  | Nullable |  Description  |  Allowed Values / Range |
|--------|--------|----------|---------------|-------------------------|
| id     | string | No       | Rocket ID     |                         |
| name   | string | No       | Rocket name   |                         |
| ...    | ...    | ...      | ...           | ...                     |

---

### dim_launchpad, dim_payload, dim_core
Define columns similarly for each dimension table.

---

### bridge_launch_payload, bridge_launch_core
Define columns for bridge tables.

---

## Semantic Rules

- **Unknown vs Failure**  
  - `success = None` → unknown outcome  
  - `success = False` → launch failed  

- **Payload mass**  
  - Must be non-negative  
  - Null if unknown  

- **No future dates**  
  - All `date_lisbon` must be <= current date  

---

## Timezone Policy
All datetime fields are converted and stored in **Europe/Lisbon** timezone.

---

## SLAs

- **Completeness**  
  All required fields must be present unless marked nullable.  

- **Referential Integrity**  
  All foreign keys must reference valid IDs in dimension tables.  

- **Quality Gates**  
  Schema/type checks, rule checks, anomaly detection.  


