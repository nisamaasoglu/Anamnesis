"""Entry point: prepare the data, then serve the API and console."""

from __future__ import annotations

import uvicorn

from anamnesis import config
from anamnesis.api import app, ensure_ready

if __name__ == "__main__":
    counts = ensure_ready()
    print(f"{config.HOSPITAL_NAME} (fictional) - synthetic records: {counts}")
    print(f"Console: http://{config.HOST}:{config.PORT}")
    uvicorn.run(app, host=config.HOST, port=config.PORT)
