"""
Daggr Studio — Space entrypoint.

The app itself is composed in `daggrstudio/web/studio.py` (custom frontend at `/`, JSON API at
`/api/*`, the daggr canvas at `/canvas/`, the Gradio Builder at `/builder/`). This file only
starts the server, because `gradio.Server.launch()` is the supported way to bring up a Server
app (it wires Gradio's queue/routing on top of FastAPI).

Local:  python app.py          → http://localhost:7860
Space:  sdk: docker, app_port 7860
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from daggrstudio.web.studio import app  # noqa: E402  (import composes the whole app)

__all__ = ["app"]


def main() -> None:
    port = int(os.environ.get("PORT") or os.environ.get("GRADIO_SERVER_PORT") or 7860)
    # launch() blocks; it is Gradio's documented entrypoint for Server apps and registers
    # the queue + /gradio_api routes that @app.api() endpoints rely on.
    try:
        app.launch(server_name="0.0.0.0", server_port=port, show_error=True,
                   quiet=False, _frontend=False)
    except TypeError:
        # older/newer Server signature: fall back to plain uvicorn so the Space still boots
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
