from openenv.core.env_server.http_server import create_app
from music_catalog_pe_env.models import CatalogAction, CatalogObservation
from music_catalog_pe_env.env import MusicCatalogPEEnvironment

import gradio as gr
from .ui import create_ui

app = create_app(
    env=MusicCatalogPEEnvironment, 
    action_cls=CatalogAction, 
    observation_cls=CatalogObservation,
    env_name="music_catalog_pe_env"
)

# Mount the interactive dashboard exactly at root
demo = create_ui()
app = gr.mount_gradio_app(app, demo, path="/")

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
