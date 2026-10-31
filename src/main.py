"""Process entry point (KN-FAPI-01): `app` for uvicorn, `handler` for Lambda behind Mangum.

Configuration is read and checked here, at import, so a bad deploy stops at the start and not at the first request
(KN-FAPI-03). In `postgres` mode the connected role must not bypass Row-Level Security (KN-FAPI-04).

__author__ = "Van Vo"
"""

from __future__ import annotations

from mangum import Mangum

from src.app_factory import create_app
from src.core.config import load_knowledge_settings
from src.core.logging_config import configure_logging

configure_logging()
_settings = load_knowledge_settings()
app = create_app(_settings)
app.state.container.verify_database()

# RFC 9457 bodies use application/problem+json; listed so API Gateway gets them as text, not base64.
handler = Mangum(app, lifespan="off", text_mime_types=["text/", "application/json", "application/problem+json"])
