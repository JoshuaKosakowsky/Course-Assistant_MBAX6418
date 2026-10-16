"""App configuration: paths and class-service settings.

Reads the local .env (gitignored) for secrets; everything here is safe to
import anywhere. Never log or print API key values.

Port 9000 is reserved EXCLUSIVELY for Hermes agentic operations — the
assignment forbids ordinary application traffic on it; use
``validate_no_reserved_port`` before making any class-service call.
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

# Data root: course files, library records, parsed artifacts. Gitignored.
DATA_ROOT = Path(os.getenv("DATA_ROOT", str(REPO_ROOT / "data")))

# Class services (dobolyi.com, ports 9001+; used from Phase 3 on).
VISION_BASE_URL = os.getenv("BASE_URL_VISION", "")
VISION_API_KEY = os.getenv("CLASS_API_KEY", "")
VISION_MODEL = os.getenv("VISION_MODEL", "")

# Text embeddings (class service; OpenAI-compatible /v1/embeddings).
# Endpoint + model verified in M0: port 9002, 2048-dim output.
TEXT_EMBED_BASE_URL = os.getenv("BASE_URL_TEXT_EMBED", "http://dobolyi.com:9002/v1")
TEXT_EMBED_MODEL = os.getenv("TEXT_EMBED_MODEL", "nvidia/Nemotron-3-Embed-1B-BF16")
TEXT_EMBED_API_KEY = os.getenv("CLASS_API_KEY", "")

APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
APP_PORT = int(os.getenv("APP_PORT", "7860"))

RESERVED_PORT = 9000  # Hermes agentic operations only


def validate_no_reserved_port(*base_urls: str) -> None:
    """Raise ValueError if a class-service URL points at port 9000.

    Port 9000 is reserved exclusively for Hermes agentic operations;
    application requests must use ports 9001 and above (assignment rule).
    """
    for url in base_urls:
        match = re.search(r":(\d+)", url or "")
        if match and int(match.group(1)) == RESERVED_PORT:
            raise ValueError(
                f"class-service URL {url!r} targets the reserved Hermes port "
                f"{RESERVED_PORT}; application calls must use ports 9001+"
            )
