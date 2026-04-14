# Data Dictionary (Core + Planned Enrichment)

## Core target
- `accident_severity`: target label, valid values `{1,2,3}`.

## Core collision features
- `number_of_vehicles`
- `number_of_casualties`
- `date`
- `time` (if available)
- `lsoa_of_accident_location` or location coordinates

## Core weather features
- `cloud_cover`
- `sunshine`
- `global_radiation`
- `max_temp`
- `mean_temp`
- `min_temp`
- `precipitation`
- `pressure`
- `snow` (derived from `snow_depth` if needed)

## Derived temporal features
- `hour`
- `day_of_week`
- `month`
- `is_weekend`
- `season`
- `hour_peak`
- `is_bank_holiday`

## Derived interaction features
- `precipitation_peak_interaction`
- `low_visibility_proxy`

## Spatial key
- `spatial_key`: LSOA-based or grid-based fallback key used for join and aggregation.

## Planned enrichment columns (placeholder)
- `road_class`
- `maxspeed`
- `junction_density`
- `no2`
- `pm25`
- `imd_decile`

