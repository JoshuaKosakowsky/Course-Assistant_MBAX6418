"""App configuration: paths and class-service settings.

Reads the local .env (gitignored) for secrets; everything here is safe to
import anywhere. Never log or print API key values.
"""

import os
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

APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
APP_PORT = int(os.getenv("APP_PORT", "7860"))
