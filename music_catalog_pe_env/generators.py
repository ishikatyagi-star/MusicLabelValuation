import json
import os
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def generate_easy_catalog(output_dir: str):
    """
    Generates the 'EASY' catalog:
    - 50 tracks
    - stable evergreen
    - 100% ownership
    - diversified platforms & territories
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(42)
    
    # 1. catalog.json
    catalog = {
        "catalog_id": "easy_catalog_01",
        "catalog_name": "Sunrise Evergreen Collection",
        "acquisition_date_assumption": "2026-06-01",
        "analyst_currency": "USD",
        "ownership_share_pct": 100.0,
        "total_tracks": 50,
        "primary_genres": ["Pop", "Rock", "Indie"],
        "notes": "Stable performer. Mostly organic streaming."
    }
    with open(os.path.join(output_dir, "catalog.json"), "w") as f:
        json.dump(catalog, f, indent=2)

    # 2. tracks.csv
    tracks = []
    base_date = datetime(2018, 1, 1)
    for i in range(50):
        rel_date = base_date + timedelta(days=int(rng.randint(0, 1800)))
        tracks.append({
            "track_id": f"TRK_E_{i+1:03d}",
            "track_title": f"Evergreen Song {i+1}",
            "artist_name": f"Stable Artist {rng.randint(1, 10)}",
            "release_date": rel_date.strftime("%Y-%m-%d"),
            "genre": rng.choice(catalog["primary_genres"]),
            "duration_sec": int(rng.normal(210, 30)),
            "ownership_share_pct": 100.0,
            "catalog_weight_hint": float(rng.uniform(0.5, 3.0)), # Not too concentrated
            "is_focus_track": i < 5,
            "is_explicit": bool(rng.choice([True, False], p=[0.2, 0.8])),
            "language": "en",
            "territory_origin": "US"
        })
    df_tracks = pd.DataFrame(tracks)
    # Normalize weights so sum is 100
    df_tracks['catalog_weight_hint'] = (df_tracks['catalog_weight_hint'] / df_tracks['catalog_weight_hint'].sum()) * 100
    df_tracks.to_csv(os.path.join(output_dir, "tracks.csv"), index=False)

    # 3. monthly_revenue.csv
    revenue_data = []
    months = pd.date_range("2023-01-01", "2025-06-01", freq="MS").strftime("%Y-%m").tolist()
    total_months = len(months)
    
    platforms = ['spotify', 'youtube', 'apple', 'publishing', 'neighboring_rights', 'sync', 'other']
    plat_weights = [0.35, 0.15, 0.20, 0.15, 0.05, 0.05, 0.05]
    
    total_ttm = 0.0
    
    for month_idx, month in enumerate(months):
        month_trend = 1.0 + (month_idx * 0.002) # Slight growth over time
        for _, row in df_tracks.iterrows():
            weight = row['catalog_weight_hint'] / 100.0
            
            # Base monthly revenue per track: ~ 45,000 global total / 50 tracks ~ 900
            # weight controls the distribution
            base_track_rev = 45000.0 * weight * month_trend
            
            # Add some slight noise
            noise = rng.uniform(0.9, 1.1)
            actual_rev = base_track_rev * noise
            
            rev_row = {"month": month, "track_id": row['track_id']}
            track_total = 0.0
            for p, pw in zip(platforms, plat_weights):
                p_rev = actual_rev * pw * rng.uniform(0.95, 1.05)
                # Ensure occasional sync lumpiness
                if p == 'sync':
                    p_rev = actual_rev * pw * (3.0 if rng.random() > 0.9 else 0.2)
                
                rev_row[f"{p}_revenue"] = round(p_rev, 2)
                rev_row[f"{p}_streams"] = int(p_rev / 0.003) if p in ['spotify', 'apple', 'youtube'] else 0
                track_total += round(p_rev, 2)
                
            rev_row["total_revenue"] = track_total
            revenue_data.append(rev_row)
            
            if month_idx >= total_months - 12:
                total_ttm += track_total
                
    df_rev = pd.DataFrame(revenue_data)
    # Order columns properly
    cols = ['month', 'track_id', 'spotify_streams', 'spotify_revenue', 'youtube_streams', 'youtube_revenue', 
            'apple_streams', 'apple_revenue', 'publishing_revenue', 'neighboring_rights_revenue', 
            'sync_revenue', 'other_revenue', 'total_revenue']
    # If missing stream columns add them
    for c in cols:
        if c not in df_rev.columns:
            df_rev[c] = 0
            
    df_rev[cols].to_csv(os.path.join(output_dir, "monthly_revenue.csv"), index=False)

    # 4. platform_mix.csv
    plat_mix = []
    for p, pw in zip(platforms, plat_weights):
        plat_mix.append({
            "platform": p,
            "revenue_share_pct": pw * 100,
            "stream_share_pct": pw * 100 if p in ['spotify', 'youtube', 'apple'] else 0,
            "trailing_12m_revenue": round(total_ttm * pw, 2),
            "trailing_12m_units": int(total_ttm * pw / 0.003) if p in ['spotify', 'apple', 'youtube'] else 0
        })
    pd.DataFrame(plat_mix).to_csv(os.path.join(output_dir, "platform_mix.csv"), index=False)

    # 5. territory_mix.csv
    terr_mix = [
        {"territory": "US", "revenue_share_pct": 40.0, "stream_share_pct": 45.0},
        {"territory": "UK", "revenue_share_pct": 15.0, "stream_share_pct": 12.0},
        {"territory": "DE", "revenue_share_pct": 10.0, "stream_share_pct": 8.0},
        {"territory": "ROW", "revenue_share_pct": 35.0, "stream_share_pct": 35.0},
    ]
    pd.DataFrame(terr_mix).to_csv(os.path.join(output_dir, "territory_mix.csv"), index=False)

    # 6. ground_truth.json
    ground_truth = {
        "task_id": "easy_stable_evergreen",
        "difficulty": "easy",
        "true_normalized_ttm_revenue": round(total_ttm, 2),
        "growth_rate_assumption": 0.02,
        "stability_score": 0.95,
        "top_track_concentration": round(df_tracks.sort_values('catalog_weight_hint', ascending=False).head(3)['catalog_weight_hint'].sum() / 100, 2),
        "top_platform_concentration": 0.35, # spotify
        "territory_concentration": 0.40,
        "anomaly_flags": [],
        "normalized_multiple": 8.5,
        "true_valuation_base": round(total_ttm * 8.5, 2),
        "true_valuation_low": round(total_ttm * 8.5 * 0.85, 2),
        "true_valuation_high": round(total_ttm * 8.5 * 1.15, 2),
        "correct_recommendation": "acquire",
        "must_detect_risks": [],
        "grader_tolerances": {"revenue_pct": 0.10, "valuation_pct": 0.15}
    }
    with open(os.path.join(output_dir, "ground_truth.json"), "w") as f:
        json.dump(ground_truth, f, indent=2)


def generate_medium_catalog(output_dir: str):
    """
    Generates the 'MEDIUM' catalog:
    - 30 tracks
    - Top 3 drive 70% revenue
    - YouTube 55% dependent
    - 85% ownership
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(123)
    
    catalog = {
        "catalog_id": "medium_catalog_01",
        "catalog_name": "Urban Beats & Flow",
        "acquisition_date_assumption": "2026-06-01",
        "analyst_currency": "USD",
        "ownership_share_pct": 85.0,
        "total_tracks": 30,
        "primary_genres": ["Hip-Hop", "R&B"],
        "notes": "Strong recent hits. Checking concentration risk."
    }
    with open(os.path.join(output_dir, "catalog.json"), "w") as f:
        json.dump(catalog, f, indent=2)

    tracks = []
    base_date = datetime(2021, 1, 1)
    
    # Generate weights heavily skewed
    weights = np.array([30.0, 25.0, 15.0] + [rng.uniform(0.1, 2.0) for _ in range(27)])
    weights = (weights / weights.sum()) * 100

    for i in range(30):
        rel_date = base_date + timedelta(days=int(rng.randint(0, 1000)))
        tracks.append({
            "track_id": f"TRK_M_{i+1:03d}",
            "track_title": f"Hit Trap {i+1}" if i < 3 else f"Filler Track {i+1}",
            "artist_name": "Lil Flow",
            "release_date": rel_date.strftime("%Y-%m-%d"),
            "genre": rng.choice(catalog["primary_genres"]),
            "duration_sec": int(rng.normal(180, 20)),
            "ownership_share_pct": 85.0,
            "catalog_weight_hint": float(weights[i]),
            "is_focus_track": i < 3,
            "is_explicit": True,
            "language": "en",
            "territory_origin": "US"
        })
    df_tracks = pd.DataFrame(tracks)
    df_tracks.to_csv(os.path.join(output_dir, "tracks.csv"), index=False)

    revenue_data = []
    months = pd.date_range("2024-06-01", "2026-05-01", freq="MS").strftime("%Y-%m").tolist()
    total_months = len(months)
    
    platforms = ['spotify', 'youtube', 'apple', 'publishing', 'neighboring_rights', 'sync', 'other']
    plat_weights = [0.20, 0.55, 0.10, 0.10, 0.02, 0.0, 0.03]
    
    total_ttm = 0.0
    
    for month_idx, month in enumerate(months):
        # Peak momentum in mid 2025, cooling off
        trend_factor = 1.0 + np.sin(month_idx * np.pi / total_months) * 0.5
        for _, row in df_tracks.iterrows():
            weight = row['catalog_weight_hint'] / 100.0
            
            # Base approx $55k global total per month
            base_track_rev = 55000.0 * weight * trend_factor
            actual_rev = base_track_rev * rng.uniform(0.85, 1.15)
            
            rev_row = {"month": month, "track_id": row['track_id']}
            track_total = 0.0
            for p, pw in zip(platforms, plat_weights):
                p_rev = actual_rev * pw * rng.uniform(0.9, 1.1)
                p_rev *= (85.0 / 100.0) # Apply 85% ownership share mathematically
                rev_row[f"{p}_revenue"] = round(p_rev, 2)
                rev_row[f"{p}_streams"] = int(p_rev / 0.002) if p in ['spotify', 'apple', 'youtube'] else 0
                track_total += round(p_rev, 2)
                
            rev_row["total_revenue"] = track_total
            revenue_data.append(rev_row)
            
            if month_idx >= total_months - 12:
                total_ttm += track_total

    df_rev = pd.DataFrame(revenue_data)
    cols = ['month', 'track_id', 'spotify_streams', 'spotify_revenue', 'youtube_streams', 'youtube_revenue', 
            'apple_streams', 'apple_revenue', 'publishing_revenue', 'neighboring_rights_revenue', 
            'sync_revenue', 'other_revenue', 'total_revenue']
    for c in cols:
        if c not in df_rev.columns:
            df_rev[c] = 0
    df_rev[cols].to_csv(os.path.join(output_dir, "monthly_revenue.csv"), index=False)

    plat_mix = []
    for p, pw in zip(platforms, plat_weights):
        plat_mix.append({
            "platform": p,
            "revenue_share_pct": pw * 100,
            "stream_share_pct": pw * 100 if p in ['spotify', 'youtube', 'apple'] else 0,
            "trailing_12m_revenue": round(total_ttm * pw, 2),
            "trailing_12m_units": int(total_ttm * pw / 0.002) if p in ['spotify', 'apple', 'youtube'] else 0
        })
    pd.DataFrame(plat_mix).to_csv(os.path.join(output_dir, "platform_mix.csv"), index=False)

    terr_mix = [
        {"territory": "US", "revenue_share_pct": 60.0, "stream_share_pct": 65.0},
        {"territory": "UK", "revenue_share_pct": 5.0, "stream_share_pct": 5.0},
        {"territory": "BR", "revenue_share_pct": 10.0, "stream_share_pct": 15.0},
        {"territory": "ROW", "revenue_share_pct": 25.0, "stream_share_pct": 15.0},
    ]
    pd.DataFrame(terr_mix).to_csv(os.path.join(output_dir, "territory_mix.csv"), index=False)

    ground_truth = {
        "task_id": "medium_concentration_risk",
        "difficulty": "medium",
        "true_normalized_ttm_revenue": round(total_ttm, 2),
        "growth_rate_assumption": -0.05,
        "stability_score": 0.60,
        "top_track_concentration": 0.70,
        "top_platform_concentration": 0.55,
        "territory_concentration": 0.60,
        "anomaly_flags": ["HIGH_TRACK_CONCENTRATION", "HIGH_PLATFORM_CONCENTRATION"],
        "normalized_multiple": 5.5,
        "true_valuation_base": round(total_ttm * 5.5, 2),
        "true_valuation_low": round(total_ttm * 5.5 * 0.85, 2),
        "true_valuation_high": round(total_ttm * 5.5 * 1.15, 2),
        "correct_recommendation": "acquire_at_discount",
        "must_detect_risks": ["top_track_concentration", "youtube_dependency"],
        "grader_tolerances": {"revenue_pct": 0.10, "valuation_pct": 0.15}
    }
    with open(os.path.join(output_dir, "ground_truth.json"), "w") as f:
        json.dump(ground_truth, f, indent=2)


def generate_hard_catalog(output_dir: str):
    """
    Generates the 'HARD' catalog:
    - 40 tracks
    - Viral spike in last 3 months
    - 65% ownership, missing data
    - High outlier distortion
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(999)
    
    catalog = {
        "catalog_id": "hard_catalog_01",
        "catalog_name": "Global Viral Assets",
        "acquisition_date_assumption": "2026-06-01",
        "analyst_currency": "USD",
        "ownership_share_pct": 65.0,
        "total_tracks": 40,
        "primary_genres": ["Electronic", "World"],
        "notes": "Recent TikTok trend. Need to normalize earnings carefully."
    }
    with open(os.path.join(output_dir, "catalog.json"), "w") as f:
        json.dump(catalog, f, indent=2)

    tracks = []
    base_date = datetime(2019, 1, 1)
    
    weights = np.array([20.0, 10.0] + [rng.uniform(0.5, 3.0) for _ in range(38)])
    weights = (weights / weights.sum()) * 100

    for i in range(40):
        rel_date = base_date + timedelta(days=int(rng.randint(0, 2000)))
        tracks.append({
            "track_id": f"TRK_H_{i+1:03d}",
            "track_title": f"Viral Dance {i+1}" if i == 0 else f"Global Sound {i+1}",
            "artist_name": "DJ Trend",
            "release_date": rel_date.strftime("%Y-%m-%d"),
            "genre": rng.choice(catalog["primary_genres"]),
            "duration_sec": int(rng.normal(160, 15)),
            "ownership_share_pct": 65.0 if i > 0 else 50.0, # Target track is 50% only
            "catalog_weight_hint": float(weights[i]),
            "is_focus_track": i == 0,
            "is_explicit": False,
            "language": "es" if rng.random() > 0.5 else "en",
            "territory_origin": "BR"
        })
    df_tracks = pd.DataFrame(tracks)
    df_tracks.to_csv(os.path.join(output_dir, "tracks.csv"), index=False)

    revenue_data = []
    months = pd.date_range("2023-06-01", "2026-05-01", freq="MS").strftime("%Y-%m").tolist()
    total_months = len(months)
    
    platforms = ['spotify', 'youtube', 'apple', 'publishing', 'neighboring_rights', 'sync', 'other']
    plat_weights = [0.30, 0.30, 0.15, 0.10, 0.05, 0.02, 0.08] # Base weights
    
    total_ttm_unadjusted = 0.0
    total_ttm_normalized = 0.0
    
    for month_idx, month in enumerate(months):
        is_spike = month_idx >= total_months - 3 # Last 3 months spike
        
        for idx, row in df_tracks.iterrows():
            weight = row['catalog_weight_hint'] / 100.0
            own_share = row['ownership_share_pct'] / 100.0
            
            # Base $35k / month
            base_track_rev = 35000.0 * weight
            
            # If it's the target track and spiking
            if idx == 0 and is_spike:
                actual_rev = base_track_rev * rng.uniform(4.0, 6.0) # 5x spike
            else:
                actual_rev = base_track_rev * rng.uniform(0.8, 1.2)
                
            rev_row = {"month": month, "track_id": row['track_id']}
            track_total = 0.0
            
            for p, pw in zip(platforms, plat_weights):
                spike_multi = 2.0 if (is_spike and idx == 0 and p in ['spotify', 'youtube', 'other']) else 1.0 # TikTok shows in other/shorts
                p_rev = actual_rev * pw * spike_multi * rng.uniform(0.9, 1.1)
                
                # Apply ownership
                p_rev *= own_share
                
                # Inject missing data
                if month_idx < 10 and p == 'publishing':
                    p_rev = np.nan # Missing publishing early on
                    
                rev_row[f"{p}_revenue"] = round(p_rev, 2) if not np.isnan(p_rev) else ""
                rev_row[f"{p}_streams"] = int(p_rev / 0.002) if p in ['spotify', 'apple', 'youtube'] else 0
                track_total += round(p_rev, 2) if not np.isnan(p_rev) else 0.0
                
            rev_row["total_revenue"] = round(track_total, 2)
            revenue_data.append(rev_row)
            
            if month_idx >= total_months - 12:
                total_ttm_unadjusted += track_total
                if is_spike and idx == 0:
                    # Normalized doesn't include the 5x spike, cap it
                    total_ttm_normalized += (base_track_rev * own_share * 1.0)
                else:
                    total_ttm_normalized += track_total

    df_rev = pd.DataFrame(revenue_data)
    cols = ['month', 'track_id', 'spotify_streams', 'spotify_revenue', 'youtube_streams', 'youtube_revenue', 
            'apple_streams', 'apple_revenue', 'publishing_revenue', 'neighboring_rights_revenue', 
            'sync_revenue', 'other_revenue', 'total_revenue']
    df_rev[cols].to_csv(os.path.join(output_dir, "monthly_revenue.csv"), index=False)

    # Messy platform mix capturing the unadjusted TTM
    plat_mix = []
    for p, pw in zip(platforms, plat_weights):
        # Introduce a distorter reflecting the recent spike
        spike_mod = 1.6 if p in ['spotify', 'youtube', 'other'] else 0.8
        plat_mix.append({
            "platform": p,
            "revenue_share_pct": round(pw * spike_mod * 100, 1),
            "stream_share_pct": round(pw * spike_mod * 100, 1) if p in ['spotify', 'youtube', 'apple'] else 0,
            "trailing_12m_revenue": round(total_ttm_unadjusted * pw * spike_mod / 1.3, 2), # approx
            "trailing_12m_units": int(total_ttm_unadjusted * pw * spike_mod / 0.002) if p in ['spotify', 'apple', 'youtube'] else 0
        })
    pd.DataFrame(plat_mix).to_csv(os.path.join(output_dir, "platform_mix.csv"), index=False)

    terr_mix = [
        {"territory": "BR", "revenue_share_pct": 45.0, "stream_share_pct": 50.0},
        {"territory": "US", "revenue_share_pct": 20.0, "stream_share_pct": 15.0},
        {"territory": "MX", "revenue_share_pct": 15.0, "stream_share_pct": 20.0},
        {"territory": "ROW", "revenue_share_pct": 20.0, "stream_share_pct": 15.0},
    ]
    pd.DataFrame(terr_mix).to_csv(os.path.join(output_dir, "territory_mix.csv"), index=False)

    ground_truth = {
        "task_id": "hard_viral_spike_noisy",
        "difficulty": "hard",
        "unadjusted_ttm_revenue": round(total_ttm_unadjusted, 2),
        "true_normalized_ttm_revenue": round(total_ttm_normalized, 2), # Crucial required valuation metric
        "growth_rate_assumption": 0.0, # Normalizing out the spike
        "stability_score": 0.30,
        "top_track_concentration": 0.45,
        "top_platform_concentration": 0.40,
        "territory_concentration": 0.45,
        "anomaly_flags": ["VIRAL_SPIKE_LAST_3M", "PARTIAL_RIGHTS_FOCUS_TRACK", "MISSING_DATA"],
        "normalized_multiple": 4.5,
        "true_valuation_base": round(total_ttm_normalized * 4.5, 2),
        "true_valuation_low": round(total_ttm_normalized * 4.5 * 0.85, 2),
        "true_valuation_high": round(total_ttm_normalized * 4.5 * 1.15, 2),
        "correct_recommendation": "pass",
        "must_detect_risks": ["viral_spike", "incomplete_data", "rights_complexity"],
        "grader_tolerances": {"revenue_pct": 0.15, "valuation_pct": 0.20}
    }
    with open(os.path.join(output_dir, "ground_truth.json"), "w") as f:
        json.dump(ground_truth, f, indent=2)

def generate_all():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "seed_catalogs")
    print(f"Generating synthetic data in {data_dir}...")
    generate_easy_catalog(os.path.join(data_dir, "easy_catalog_01"))
    generate_medium_catalog(os.path.join(data_dir, "medium_catalog_01"))
    generate_hard_catalog(os.path.join(data_dir, "hard_catalog_01"))
    print("Done generating synthetic data.")

if __name__ == "__main__":
    generate_all()
