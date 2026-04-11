from openenv.core.env_server.http_server import create_app
from music_catalog_pe_env.models import CatalogAction, CatalogObservation
from music_catalog_pe_env.env import MusicCatalogPEEnvironment

app = create_app(
    env=MusicCatalogPEEnvironment, 
    action_cls=CatalogAction, 
    observation_cls=CatalogObservation,
    env_name="music_catalog_pe_env"
)

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
