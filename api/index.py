"""Vercel entry point. Vercel's Python runtime serves the ASGI app named `app` from this file.

vercel.json sends every path here as /api/index?__path=/<original path>, and bundles src/ and
data/ with the function. Depending on the runtime, the request may arrive with the original path or
with the function's own path, so the wrapper below restores the original path either way.
"""
import sys
from pathlib import Path
from urllib.parse import parse_qsl, urlencode

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jobagent.web.app import create_app  # noqa: E402

PATH_PARAM = "__path"


def restore_path(asgi_app):
    async def wrapped(scope, receive, send):
        if scope["type"] == "http":
            query = parse_qsl(scope.get("query_string", b"").decode("latin-1"), keep_blank_values=True)
            original = [v for k, v in query if k == PATH_PARAM]
            if original:
                path = "/" + original[-1].lstrip("/")
                rest = urlencode([(k, v) for k, v in query if k != PATH_PARAM])
                scope = {**scope, "path": path, "raw_path": path.encode(), "query_string": rest.encode("latin-1")}
            elif scope.get("path", "").startswith("/api/index"):
                scope = {**scope, "path": "/", "raw_path": b"/"}
        await asgi_app(scope, receive, send)

    return wrapped


app = restore_path(create_app())
