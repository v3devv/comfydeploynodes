"""Loads `custom_routes.py` against small fakes. Standard library only.

No ComfyUI, aiohttp, torch or network is needed: `server`, `execution`,
`folder_paths`, `model_management`, `torch`, `psutil`, `aiohttp` and
`aiofiles` are replaced in `sys.modules` before the module loads. The real
`globals.py` loads beside it (it needs PIL and pydantic, both present).

What leaves the routes for Deploy is captured, never sent:

    posts        every body given to `async_request_with_retry`
    ws           every `send(event, data)` to the ComfyDeploy websocket
    native       every event ComfyUI's own `send_sync` and `send_json` saw
"""

import asyncio
import contextlib
import importlib.util
import io
import json
import logging
import os
import shutil
import sys
import tempfile
import types
import unittest
import warnings
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROUTES_PATH = os.path.join(ROOT, "custom_routes.py")

from lora_harness import REF_SOURCE, SIG_MARKER, claims_for, make_ref, safetensors_bytes  # noqa: E402

REF = make_ref(claims_for(safetensors_bytes()))
BEARER = "tok-" + "5ecret" * 4
PRESIGNED_PUT = (
    "https://acct.r2.cloudflarestorage.com/outputs/run/a.png"
    "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAEXAMPLE%2F20261110&X-Amz-Signature="
    + "0badc0de" * 8
)
SAS_URL = "https://acct.blob.example/c/a.safetensors?sv=2024&sig=" + "c0ffee" * 6 + "&token=" + "7e57" * 6
SECRETS = (
    REF,
    REF.split(".")[1],
    REF_SOURCE,
    SIG_MARKER,
    BEARER,
    "0badc0de" * 8,
    "AKIAEXAMPLE",
    "c0ffee" * 6,
    "7e57" * 6,
)
# The run's own Deploy credential. It rides in a header, never in a body.
RUN_CREDENTIAL = "run" + "-credential"


class _Routes:
    def get(self, _path):
        return lambda fn: fn

    def post(self, _path):
        return lambda fn: fn


class _ClientError(Exception):
    pass


def _fake_aiohttp():
    aiohttp = types.ModuleType("aiohttp")
    web = types.ModuleType("aiohttp.web")

    class Response:
        def __init__(self, status=200, reason=None, text=None, body=None, headers=None):
            self.status, self.reason, self.text, self.body = status, reason, text, body

    class JsonResponse:
        def __init__(self, data, status=200):
            self.data, self.status = data, status

    web.Response = Response
    web.json_response = lambda data, status=200, **_kw: JsonResponse(data, status)
    web.StreamResponse = object
    web.WebSocketResponse = object

    class WSMsgType:
        TEXT, BINARY, ERROR = 1, 2, 3

    aiohttp.web = web
    aiohttp.ClientError = _ClientError
    aiohttp.ClientSession = mock.MagicMock(name="ClientSession")
    aiohttp.ClientTimeout = lambda **kw: kw
    aiohttp.TCPConnector = mock.MagicMock(name="TCPConnector")
    aiohttp.WSMsgType = WSMsgType
    return aiohttp, web


def _fake_server():
    server = types.ModuleType("server")

    class PromptServer:
        instance = None

        def send_sync(self, event, data, sid=None):
            self.native.append((event, json.loads(json.dumps(data, default=str)), data))

        async def send_json(self, event, data, sid=None):
            self.native.append((event, json.loads(json.dumps(data, default=str)), data))

    instance = PromptServer()
    instance.native = []
    instance.routes = _Routes()
    instance.app = types.SimpleNamespace(on_startup=[])
    instance.number = 0
    instance.prompt_queue = mock.MagicMock(name="prompt_queue")
    instance.trigger_on_prompt = lambda data: data
    PromptServer.instance = instance
    server.PromptServer = PromptServer
    return server


def load_routes(output_dir):
    """A fresh custom_routes module, its fake prompt server and fake `web`."""
    aiohttp, web = _fake_aiohttp()
    server = _fake_server()
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.annotated_filepath = lambda name: (name, None)
    folder_paths.get_directory_by_type = lambda _kind: output_dir
    model_management = types.ModuleType("model_management")
    model_management.get_torch_device = lambda: types.SimpleNamespace(type="cpu")
    fakes = {
        "aiohttp": aiohttp,
        "aiohttp.web": web,
        "aiofiles": types.ModuleType("aiofiles"),
        "server": server,
        "execution": types.ModuleType("execution"),
        "folder_paths": folder_paths,
        "model_management": model_management,
        "torch": types.ModuleType("torch"),
        "psutil": types.ModuleType("psutil"),
    }
    names = list(fakes) + ["globals", "monet_redact"]
    saved = {k: sys.modules.get(k) for k in names}
    sys.modules.update(fakes)
    for k in ("globals", "monet_redact"):
        sys.modules.pop(k, None)
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CD_", "USE_LOGFIRE"))}
    spec = importlib.util.spec_from_file_location("custom_routes_under_test", ROUTES_PATH)
    module = importlib.util.module_from_spec(spec)
    try:
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch("atexit.register"), \
                mock.patch("logging.basicConfig"), \
                contextlib.redirect_stdout(io.StringIO()), \
                warnings.catch_warnings():
            # Upstream returns from a finally block (Python 3.14 warns at compile)
            # and globals.py configures pydantic the v1 way (v2 warns).
            warnings.simplefilter("ignore", SyntaxWarning)
            warnings.simplefilter("ignore", DeprecationWarning)
            spec.loader.exec_module(module)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return module, server.PromptServer.instance, web


class RoutesTestCase(unittest.TestCase):
    """custom_routes with Deploy's endpoints captured."""

    prompt_id = "run-0001"

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="routes-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.routes, self.server, self.web = load_routes(self.tmp)
        self.posts = []

        async def fake_request(method, url, disable_timeout=False, token=None, **kwargs):
            self.posts.append((method, url, kwargs.get("json")))
            return types.SimpleNamespace(json=self._json_answer)

        self.routes.async_request_with_retry = fake_request
        self.ws = []

        async def fake_send(event, data, sid=None):
            self.ws.append((event, json.loads(json.dumps(data, default=str))))

        self.routes.send = fake_send
        self.routes.prompt_metadata.clear()
        self.routes.comfy_message_queues.clear()
        self.routes.prompt_metadata[self.prompt_id] = self.routes.SimplePrompt(
            status_endpoint="https://deploy.example/api/update-run",
            file_upload_endpoint="https://deploy.example/api/file-upload",
            workflow_api={"4": {"class_type": "ComfyUIDeployExternalLora", "inputs": {}}},
            token=RUN_CREDENTIAL,
        )

    async def _json_answer(self):
        return {"url": PRESIGNED_PUT}

    def run_async(self, coro):
        async def wrapped():
            result = await coro
            # Let fire and forget tasks (the ws event post) finish too.
            for _ in range(5):
                await asyncio.sleep(0)
            return result

        return asyncio.run(wrapped())

    def assert_no_secret(self, value, where):
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        for secret in SECRETS:
            self.assertNotIn(secret, text, f"a credential leaked into {where}")

    def captured_logs(self):
        return self.assertLogs("comfy-deploy", level=logging.DEBUG)
