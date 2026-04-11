from openenv.core.env_client import EnvClient
from .models import CatalogAction, CatalogObservation, CatalogState

class MusicCatalogPEEnv(EnvClient[CatalogAction, CatalogObservation, CatalogState]):
    """
    EnvClient subclass mapped to the Music Catalog environment.

    Uses the models defined in models.py and automatically maps Pydantic validation over network calls.
    """
    DOCKER_IMAGE = "music-catalog-pe-env"
