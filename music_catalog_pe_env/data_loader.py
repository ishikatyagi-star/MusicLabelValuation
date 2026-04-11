import json
import os
from typing import Any, Dict

import pandas as pd


class CatalogDataLoader:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.catalog = {}
        self.tracks = pd.DataFrame()
        self.monthly_revenue = pd.DataFrame()
        self.platform_mix = pd.DataFrame()
        self.territory_mix = pd.DataFrame()
        self.ground_truth = {}
        
    def load(self, catalog_dir: str) -> None:
        """Loads data from a specified catalog directory."""
        full_dir = os.path.join(self.base_dir, catalog_dir)
        
        with open(os.path.join(full_dir, "catalog.json"), "r") as f:
            self.catalog = json.load(f)
            
        with open(os.path.join(full_dir, "ground_truth.json"), "r") as f:
            self.ground_truth = json.load(f)
            
        self.tracks = pd.read_csv(os.path.join(full_dir, "tracks.csv"))
        self.monthly_revenue = pd.read_csv(os.path.join(full_dir, "monthly_revenue.csv"))
        self.platform_mix = pd.read_csv(os.path.join(full_dir, "platform_mix.csv"))
        self.territory_mix = pd.read_csv(os.path.join(full_dir, "territory_mix.csv"))
        
    def query_catalog_summary(self) -> Dict[str, Any]:
        return self.catalog
        
    def query_top_tracks(self, limit: int = 5) -> Dict[str, Any]:
        if "catalog_weight_hint" not in self.tracks.columns:
            return {"error": "Missing weights"}
        top = self.tracks.sort_values(by="catalog_weight_hint", ascending=False).head(limit)
        return top.to_dict(orient="records")
        
    def query_platform_mix(self) -> Dict[str, Any]:
        return self.platform_mix.to_dict(orient="records")
        
    def query_territory_mix(self) -> Dict[str, Any]:
        return self.territory_mix.to_dict(orient="records")
        
    def query_revenue_trend(self, limit_months: int = 12) -> Dict[str, Any]:
        # Group by month and sum totals
        trend = self.monthly_revenue.groupby("month")["total_revenue"].sum().reset_index()
        trend = trend.sort_values("month", ascending=False).head(limit_months)
        return trend.to_dict(orient="records")
        
    def query_anomalies(self) -> Dict[str, Any]:
        # In reality, this would run anomaly detection, here we mock it by returning summary stats
        monthly_totals = self.monthly_revenue.groupby("month")["total_revenue"].sum()
        mean = monthly_totals.mean()
        std = monthly_totals.std()
        max_val = monthly_totals.max()
        z_max = (max_val - mean) / std if std > 0 else 0
        return {
            "mean_monthly_revenue": round(mean, 2),
            "std_dev": round(std, 2),
            "max_z_score": round(z_max, 2),
            "volatility_warning": bool(z_max > 2.0)
        }
