"""
PrefixProxy — serve a sub-app that uses ABSOLUTE root-relative URLs under a path prefix.

Why this exists: the daggr canvas frontend requests `/assets/...`, `/theme.css`,
`/daggr-assets/...`, `/file/...`, `/api/...` and `/ws/...` as root-absolute paths, so it cannot
be mounted at `/canvas` without help: the browser would ask *this* Space for `/assets/...` and
get our own app's 404. This proxy rewrites those references in the responses it forwards, and
strips the prefix from the requests it passes on.

Rewrite rules (applied only inside quoted/parenthesised URL positions, never twice):

    "/assets/…"        -> "/canvas/assets/…"
    "/daggr-assets/…"  -> "/canvas/daggr-assets/…"
    "/file/…"          -> "/canvas/file/…"
    "/api/…"           -> "/canvas/api/…"
    "/ws/…"            -> "/canvas/ws/…"
    "/theme.css"       -> "/canvas/theme.css"

...only for content types that can contain URLs (html, javascript, css), only when the body
arrives with a content-length (so nothing is buffered blindly), and only up to `max_body`
bytes — anything else is streamed through byte-identical.

**What this cannot fix:** JavaScript that builds a URL at runtime from `location.origin`
(e.g. ``new WebSocket(`${location.origin}/ws/${id}`)``). If a frontend does that, no amount of
response rewriting helps and the sub-app genuinely cannot live at a sub-path.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

REWRITABLE_PREFIXES = ("/daggr-assets/", "/assets/", "/theme.css", "/file/", "/api/", "/ws/")

#: A URL is only rewritten when the character before it starts a URL token: a quote, a
#: backtick, an opening paren, `=`, `:` or `>`. Whitespace is deliberately NOT included, so
#: prose like "see /api/docs" is left alone. Known limit: a quoted string that merely
#: *mentions* "/api/" is treated as a URL - text rewriting cannot tell the two apart.
_ALLOWED_BEFORE = {'"', "'", "`", "(", "=", ":", ">"}

REWRITABLE_CONTENT = (
    "text/html", "text/css", "text/javascript", "application/javascript", "application/x-javascript",
)

DEFAULT_MAX_BODY = 8 * 1024 * 1024


class PrefixProxy:
    """ASGI app that serves ``app`` under ``prefix``, rewriting absolute URLs on the way out."""

    def __init__(
        self,
        app: Any,
        prefix: str = "/canvas",
        *,
        rewrite_html: bool = True,
        rewrite_js: bool = True,
        inject: str = "",
        base_href: bool = True,
        passthrough_other: bool = False,
        max_body: int = DEFAULT_MAX_BODY,
    ) -> None:
        self.app = app
        self.prefix = "/" + prefix.strip("/")
        self.rewrite_html = rewrite_html
        self.rewrite_js = rewrite_js
        self.inject = inject
        self.base_href = base_href
        self.passthrough_other = passthrough_other
        self.max_body = max_body
        self._stats = {"requests": 0, "rewritten": 0, "forwarded": 0, "errors": 0}

    # ── diagnostics ───────────────────────────────────────────────────────────

    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def _rewrite_lang(self, content_type: str) -> bool:
        if "html" in content_type:
            return self.rewrite_html
        if "css" in content_type or "javascript" in content_type:
            return self.rewrite_js
        return False

    # ── text surgery ──────────────────────────────────────────────────────────

    def rewrite_text(self, text: str) -> str:
        """Prefix every root-absolute URL reference. Idempotent (never double-prefixes)."""
        for target in REWRITABLE_PREFIXES:
            if target not in text:
                continue
            out: list[str] = []
            index = 0
            while True:
                found = text.find(target, index)
                if found == -1:
                    out.append(text[index:])
                    break
                before = text[found - 1] if found else ""
                already = text[max(0, found - len(self.prefix)):found] == self.prefix
                if before in _ALLOWED_BEFORE and not already:
                    out.append(text[index:found])
                    out.append(self.prefix + target)
                else:
                    out.append(text[index:found + len(target)])
                index = found + len(target)
            text = "".join(out)
        return text

    def rewrite_html_document(self, text: str) -> str:
        text = self.rewrite_text(text)
        if self.base_href and "<head>" in text and f'<base href="{self.prefix}/">' not in text:
            text = text.replace("<head>", f'<head><base href="{self.prefix}/">', 1)
        if self.inject:
            marker = "</body>"
            cut = text.rfind(marker)
            text = (text[:cut] + self.inject + text[cut:]) if cut >= 0 else text + self.inject
        return text

    # ── ASGI ──────────────────────────────────────────────────────────────────

    async def __call__(self, scope: dict[str, Any], receive: Callable, send: Callable) -> None:
        kind = scope.get("type")
        if kind not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        self._stats["requests"] += 1
        path = scope.get("path", "")
        in_prefix = path == self.prefix or path.startswith(self.prefix + "/")
        if not in_prefix:
            if self.passthrough_other:
                await self.app(scope, receive, send)
                return
            self._stats["forwarded"] += 1
            await self._plain(send, kind)
            return

        stripped = path[len(self.prefix):] or "/"
        inner = dict(scope)
        inner["path"] = stripped
        if scope.get("raw_path"):
            inner["raw_path"] = stripped.encode()
        inner["root_path"] = (scope.get("root_path") or "") + self.prefix

        if kind == "websocket":
            await self.app(inner, receive, send)
            return

        await self.app(inner, receive, self._filtering_send(send))

    async def _plain(self, send: Callable, kind: str) -> None:
        if kind == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        body = json.dumps({"error": "not found",
                           "hint": f"this path is served under {self.prefix}/"}).encode()
        await send({"type": "http.response.start", "status": 404, "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ]})
        await send({"type": "http.response.body", "body": body})

    def _filtering_send(self, send: Callable) -> Callable:
        buffer: dict[str, Any] = {"chunks": [], "size": 0, "start": None, "skip": False}

        async def wrapped(message: dict[str, Any]) -> None:
            mtype = message.get("type")

            if mtype == "http.response.start":
                headers = {k.decode().lower(): v.decode() for k, v in message.get("headers", [])}
                content_type = headers.get("content-type", "")
                has_length = "content-length" in headers
                buffer["start"] = message
                buffer["ctype"] = content_type
                # Stream anything we cannot safely buffer: no length, wrong type, or huge.
                try:
                    length = int(headers.get("content-length", "0"))
                except ValueError:
                    length = 0
                buffer["skip"] = (
                    not has_length
                    or not self._rewrite_lang(content_type)
                    or length > self.max_body
                )
                if buffer["skip"]:
                    self._stats["forwarded"] += 1
                    await send(message)
                return

            if mtype == "http.response.body":
                if buffer["skip"]:
                    await send(message)
                    return
                chunk = message.get("body", b"")
                buffer["chunks"].append(chunk)
                buffer["size"] += len(chunk)
                if buffer["size"] > self.max_body:  # grew past the cap mid-flight: bail out safely
                    await self._flush_originally(send, buffer)
                    buffer["skip"] = True
                    return
                if message.get("more_body"):
                    return
                await self._flush_rewritten(send, buffer)
                return

            await send(message)

        return wrapped

    async def _flush_originally(self, send: Callable, buffer: dict[str, Any]) -> None:
        await send(buffer["start"])
        for chunk in buffer["chunks"]:
            await send({"type": "http.response.body", "body": chunk, "more_body": False})

    async def _flush_rewritten(self, send: Callable, buffer: dict[str, Any]) -> None:
        raw = b"".join(buffer["chunks"])
        ctype = buffer.get("ctype", "")
        try:
            text = raw.decode("utf-8")
            original = text
            text = (self.rewrite_html_document(text) if "html" in ctype
                    else self.rewrite_text(text))
            if text == original:
                self._stats["forwarded"] += 1
                body = raw
            else:
                self._stats["rewritten"] += 1
                body = text.encode("utf-8")
        except Exception:
            # A proxy must never be able to break the app it fronts.
            self._stats["errors"] += 1
            body = raw

        start = dict(buffer["start"])
        headers = [(k, v) for k, v in start.get("headers", [])
                   if k.decode().lower() not in ("content-length",)]
        headers.append((b"content-length", str(len(body)).encode()))
        start["headers"] = headers
        await send(start)
        await send({"type": "http.response.body", "body": body})


def strip_prefix(path: str, prefix: str) -> str:
    """Small helper used by tests and diagnostics."""
    prefix = "/" + prefix.strip("/")
    if path == prefix:
        return "/"
    return path[len(prefix):] if path.startswith(prefix + "/") else path


__all__ = ["PrefixProxy", "REWRITABLE_PREFIXES", "strip_prefix"]