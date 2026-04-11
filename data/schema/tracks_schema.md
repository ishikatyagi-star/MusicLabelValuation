# Tracks Schema (tracks.csv)

| Column | Type | Description |
|--------|------|-------------|
| track_id | string | Unique identifier |
| track_title | string | Track Title |
| artist_name | string | Primary artist |
| release_date | date | Release Date (YYYY-MM-DD) |
| genre | string | Genre categorization |
| duration_sec | int | Length in seconds |
| ownership_share_pct | float | 0.0 to 100.0, the % of masters/publishing owned by this catalog |
| catalog_weight_hint | float | Relative weight of importance to overall revenue |
| is_focus_track | bool | Whether label pushed marketing heavily on this track |
| is_explicit | bool | Explicit content flag |
| language | string | ISO language code |
| territory_origin | string | ISO country code where it was signed |
