"""CLI: index page/slide images into the visual store (issue #3).

Usage:
    python -m app.index_visual <image-directory> [--doc-id NAME]
                               [--data-root PATH] [--no-page-from-filename]

Requires .env settings for the class visual-embedding endpoint
(BASE_URL_VISUAL_EMBED, CLASS_API_KEY, VISUAL_EMBED_MODEL) — see
.env.example. Logs counts on completion.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .env import DATA_ROOT
from .visual_index import VisualIndex

logger = logging.getLogger(__name__)

DESCRIPTION = (__doc__ or "").splitlines()[0] or "Index page/slide images into the visual store."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("image_dir", type=Path, help="directory of page/slide images to index")
    parser.add_argument("--doc-id", default=None, help="document id (default: directory name)")
    parser.add_argument("--data-root", default=None, help="override DATA_ROOT for the index location")
    parser.add_argument(
        "--no-page-from-filename",
        action="store_true",
        help="number pages sequentially instead of parsing the filename",
    )
    args = parser.parse_args(argv)

    if not args.image_dir.is_dir():
        print(f"error: {args.image_dir} is not a directory", file=sys.stderr)
        return 2

    root = Path(args.data_root) / "indexes" / "visual" if args.data_root else None
    try:
        index = VisualIndex(root=root)
        count = index.index_images_in_directory(
            args.image_dir, doc_id=args.doc_id, pages_from_filename=not args.no_page_from_filename
        )
    except RuntimeError as exc:  # e.g. .env not configured for the class endpoint
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Indexed {count} images; total in visual index: {index.count()}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(main())
