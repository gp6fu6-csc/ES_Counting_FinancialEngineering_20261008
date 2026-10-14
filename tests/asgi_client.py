"""Minimal in-process ASGI client for the server tests.

FastAPI's TestClient needs httpx, whose hard dependency certifi is MPL-2.0 (outside the project's
licence allow-list), so the tests call the ASGI app directly instead.
"""
import asyncio
import json as jsonlib
from urllib.parse import urlencode

BOUNDARY = "escount-test-boundary"


class Response:
    def __init__(self, status_code, headers, content):
        self.status_code, self.headers, self.content = status_code, headers, content

    @property
    def text(self):
        return self.content.decode("utf-8")

    def json(self):
        return jsonlib.loads(self.content)


def multipart(files):
    """files = {field: (filename, bytes, content type)} -> (body, content-type header)."""
    chunks = []
    for field, (filename, content, content_type) in files.items():
        chunks.append(f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
                      f"Content-Type: {content_type}\r\n\r\n".encode("utf-8") + content + b"\r\n")
    return b"".join(chunks) + f"--{BOUNDARY}--\r\n".encode("utf-8"), f"multipart/form-data; boundary={BOUNDARY}"


def request(app, method, path, *, json=None, content=None, files=None, headers=None, params=None,
            content_type="application/octet-stream"):
    headers = {key.lower(): value for key, value in (headers or {}).items()}
    body = b""
    if json is not None:
        body, headers["content-type"] = jsonlib.dumps(json).encode("utf-8"), "application/json"
    elif files is not None:
        body, headers["content-type"] = multipart(files)
    elif content is not None:
        body = content
        headers.setdefault("content-type", content_type)
    headers["content-length"] = str(len(body))
    headers.setdefault("host", "127.0.0.1:8000")
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method.upper(),
             "scheme": "http", "path": path, "raw_path": path.encode("utf-8"), "root_path": "",
             "query_string": urlencode(params or {}).encode("ascii"),
             "headers": [(k.encode("latin-1"), v.encode("latin-1")) for k, v in headers.items()],
             "client": ("127.0.0.1", 50000), "server": ("testserver", 80)}

    async def run():
        messages, done, sent = [], asyncio.Event(), False

        async def receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            await done.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            messages.append(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                done.set()

        await app(scope, receive, send)
        return messages

    messages = asyncio.run(run())
    start = next(m for m in messages if m["type"] == "http.response.start")
    content = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    return Response(start["status"], {k.decode("latin-1"): v.decode("latin-1") for k, v in start["headers"]}, content)


class Client:
    """client.get(path, ...) / client.post(path, ...) with default headers."""

    def __init__(self, app, headers=None):
        self.app, self.headers = app, dict(headers or {})

    def get(self, path, **kwargs):
        return request(self.app, "GET", path, headers={**self.headers, **kwargs.pop("headers", {})}, **kwargs)

    def post(self, path, **kwargs):
        return request(self.app, "POST", path, headers={**self.headers, **kwargs.pop("headers", {})}, **kwargs)
