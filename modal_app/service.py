"""Modal ASGI deployment for the Triage API and future static demo."""

from __future__ import annotations

import modal

from modal_app.train import APP_NAME, ARTIFACTS_PATH, VOLUME_NAME

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
service_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi>=0.115,<1.0",
        "onnxruntime>=1.19,<2.0",
        "transformers>=4.45,<5.0",
    )
    .add_local_python_source("app")
    .add_local_dir("app/static", remote_path="/root/app/static")
)


@app.function(image=service_image, volumes={ARTIFACTS_PATH: model_volume}, timeout=60 * 10)
@modal.asgi_app()
def fastapi_app():
    from app.main import app as api

    return api
