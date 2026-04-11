from .models import TaskConfig

TASKS = {
    "easy_stable_evergreen": TaskConfig(
        task_id="easy_stable_evergreen",
        difficulty="easy",
        catalog_dir="data/seed_catalogs/easy_catalog_01",
        max_steps=25,
        description="Evaluate a stable, diversified evergreen music pop catalog. Identify normalized TTM and standard multiple."
    ),
    "medium_concentration_risk": TaskConfig(
        task_id="medium_concentration_risk",
        difficulty="medium",
        catalog_dir="data/seed_catalogs/medium_catalog_01",
        max_steps=30,
        description="Evaluate a hip-hop catalog. Beware of concentration risks and platform dependency when estimating the multiple."
    ),
    "hard_viral_spike_noisy": TaskConfig(
        task_id="hard_viral_spike_noisy",
        difficulty="hard",
        catalog_dir="data/seed_catalogs/hard_catalog_01",
        max_steps=40,
        description="Evaluate a complex catalog with viral spikes, noisy historical data, and partial rights. Normalize carefully."
    ),
}

def get_task(task_id: str) -> TaskConfig:
    return TASKS.get(task_id)
