# Data

The "Piraeus Vice" dataset was provided by the Pattern Recognition course and is not included in this repository.

Put the CSV file in this folder. Files here are ignored by git.

Expected columns:

| Column | Description |
|---|---|
| `incident_id` | crime id |
| `split` | `TRAIN`, `VAL` or `TEST` |
| `killer_id` | killer 1–8 (empty for `TEST`) |
| `hour_float`, `latitude`, `longitude`, `victim_age`, `temp_c`, `humidity`, `dist_precinct_km`, `pop_density` | continuous features |
| `weapon_code`, `scene_type`, `weather`, `vic_gender` | categorical features |
