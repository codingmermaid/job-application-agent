"""Vercel entry point. Vercel's Python runtime serves the ASGI app named `app` from this file.

vercel.json sends every path here and bundles src/ and data/ with the function.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jobagent.web.app import create_app  # noqa: E402

app = create_app()
