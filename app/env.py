"""App configuration shared by the visual index (env + paths).

Kept as a separate module (rather than adding to app/config.py) so this
branch merges cleanly with the Phase 1 branch, which introduces
app/config.py. Fold this into app/config.py once that branch lands.

Reads the gitignored .env. Never log or print API key values.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

# Data root: course files, parsed artifacts, indexes. Gitignored.
DATA_ROOT = Path(os.getenv("DATA_ROOT", str(REPO_ROOT / "data")))

# Class services (dobolyi.com, ports 9001+; see .env.example).
VISUAL_EMBED_BASE_URL = os.getenv("BASE_URL_VISUAL_EMBED", "")
VISUAL_EMBED_API_KEY = os.getenv("CLASS_API_KEY", "")
VISUAL_EMBED_MODEL = os.getenv("VISUAL_EMBED_MODEL", "")
TEXT_EMBED_BASE_URL = os.getenv("BASE_URL_TEXT_EMBED", "") or VISUAL_EMBED_BASE_URL
TEXT_EMBED_MODEL = os.getenv("TEXT_EMBED_MODEL", "")
