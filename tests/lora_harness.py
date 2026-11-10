"""Shared fakes for the External Lora tests. Standard library only.

No ComfyUI install, no network and no Modal volume are needed: `folder_paths`
and `requests` are replaced by small fakes before the node module loads, and
three temporary folders stand in for the runner's loras search order:

    volume  the LoRA volume at /comfyui/models/loras (read only on a runner)
    public  the core volume at /public_models/loras
    cache   the node's run cache at /tmp/monet-loras, registered last

The module's import time `os.makedirs(RUN_CACHE_ROOT)` is intercepted, so the
tests never touch /tmp/monet-loras on the machine that runs them.
"""

import base64
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import shutil
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE_PATH = os.path.join(ROOT, "comfy-nodes", "external_lora.py")
FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "lora_volume")

# The query string of a presigned link. It must never reach a log or an error.
SIG_MARKER = "X-Amz-Signature=" + "deadbeef" + "cafe"
URL = "https://bucket.r2.example/loras/abc.safetensors?" + SIG_MARKER
REF_SOURCE = (
    "https://acct.r2.cloudflarestorage.com/monet-private/ws/loras/p/v.safetensors"
    "?X-Amz-Expires=2700&X-Amz-Signature=" + "feedface" * 8
)

WS = "cmwstest0000000000000001"
PRESET = "cmprtest0000000000000001"
VERSION = "cmlvtest0000000000000001"
DEPLOYMENT = "0b6c3f6e-5a1d-4c2e-9f3a-7d8e9c0b1a2f"
CLAIM_KEYS = ("p", "h", "n", "d", "e", "u")


def load_fixture(name):
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as f:
        return json.load(f)


def safetensors_bytes(payload_size=64 * 1024, fill=b"\0"):
    header = json.dumps(
        {"w": {"dtype": "F32", "shape": [payload_size // 4], "data_offsets": [0, payload_size]}}
    ).encode()
    return struct.pack("<Q", len(header)) + header + fill * payload_size


def b64url(raw):
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def canonical_payload(claims):
    return json.dumps({k: claims[k] for k in CLAIM_KEYS}, separators=(",", ":"), ensure_ascii=True)


def make_ref(claims, signature=None):
    """A reference with the given claims. The runner holds no key, so any
    canonical 43 character signature segment passes the node's shape check."""
    payload = canonical_payload(claims) if isinstance(claims, dict) else claims
    sig = signature or b64url(hashlib.sha256(payload.encode("ascii")).digest())
    return "mlr1." + b64url(payload.encode("ascii")) + "." + sig


def rel_for(sha256, version=VERSION):
    return f"monet/{WS}/{PRESET}/{version}-{sha256[:12]}.safetensors"


def claims_for(data, rel=None, source=REF_SOURCE):
    sha = hashlib.sha256(data).hexdigest()
    return {
        "p": rel or rel_for(sha),
        "h": sha,
        "n": len(data),
        "d": DEPLOYMENT,
        "e": 1794319200,
        "u": source,
    }


class FakeResponse:
    def __init__(self, status, body, headers=None, fail_after=None):
        self.status_code = status
        self._body = body
        self.headers = dict(headers or {})
        self._fail_after = fail_after
        self.closed = False

    @property
    def content(self):
        return self._body

    def iter_content(self, chunk_size=1):
        sent = 0
        for i in range(0, len(self._body), chunk_size):
            if self._fail_after is not None and sent >= self._fail_after:
                raise ConnectionError("connection reset by peer")
            chunk = self._body[i : i + chunk_size]
            sent += len(chunk)
            yield chunk

    def close(self):
        self.closed = True


def ok_response(body):
    return FakeResponse(200, body, {"Content-Length": str(len(body))})


class LoraTestCase(unittest.TestCase):
    """Loads a fresh copy of the node per test against fake folders."""

    def setUp(self):
        base = tempfile.mkdtemp(prefix="extlora-")
        self.addCleanup(shutil.rmtree, base, True)
        self.base = os.path.realpath(base)
        self.volume_dir = os.path.join(self.base, "volume")
        self.public_dir = os.path.join(self.base, "public")
        self.cache_dir = os.path.join(self.base, "cache")
        self.outside_dir = os.path.join(self.base, "outside")
        for d in (self.volume_dir, self.public_dir, self.cache_dir, self.outside_dir):
            os.makedirs(d)

        self.loras_folders = [self.volume_dir, self.public_dir]
        self.registered = []
        folder_paths = types.ModuleType("folder_paths")
        folder_paths.folder_names_and_paths = {"loras": (self.loras_folders, set())}

        def add_model_folder_path(name, path, is_default=False):
            self.registered.append((name, path, is_default))
            paths = folder_paths.folder_names_and_paths[name][0]
            if path not in paths:
                paths.insert(0, path) if is_default else paths.append(path)

        def get_filename_list(name):
            found = set()
            for folder in folder_paths.folder_names_and_paths[name][0]:
                for dirpath, _dirs, files in os.walk(folder):
                    for f in files:
                        if f.endswith(".safetensors"):
                            rel = os.path.relpath(os.path.join(dirpath, f), folder)
                            found.add(rel.replace(os.sep, "/"))
            return sorted(found)

        def get_full_path(name, filename):
            # The same lookup ComfyUI's LoraLoader uses: first folder wins.
            filename = os.path.relpath(os.path.join("/", filename), "/")
            for folder in folder_paths.folder_names_and_paths[name][0]:
                full = os.path.join(folder, filename)
                if os.path.isfile(full):
                    return full
            return None

        folder_paths.add_model_folder_path = add_model_folder_path
        folder_paths.get_filename_list = get_filename_list
        folder_paths.get_full_path = get_full_path
        self.folder_paths = folder_paths

        self.responses = []
        self.calls = []
        fake_requests = types.ModuleType("requests")

        def get(url, **kwargs):
            self.calls.append((url, kwargs))
            if not self.responses:
                raise AssertionError("the node made a request the test did not expect")
            answer = self.responses.pop(0)
            if isinstance(answer, BaseException):
                raise answer
            return answer

        fake_requests.get = get

        patched = {"folder_paths": folder_paths, "requests": fake_requests}
        saved = {k: sys.modules.get(k) for k in patched}
        sys.modules.update(patched)

        def restore():
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

        self.addCleanup(restore)

        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("MONET_LORA_ALLOW_URL", None)

        self.import_makedirs = mock.MagicMock()
        spec = importlib.util.spec_from_file_location("external_lora_under_test", NODE_PATH)
        module = importlib.util.module_from_spec(spec)
        with mock.patch("os.makedirs", self.import_makedirs):
            spec.loader.exec_module(module)
        self.module = module
        self.folders_after_import = list(self.loras_folders)

        # Point the node at the fake folders. The registered cache root is
        # swapped for the temp cache in the same (last) position.
        self.loras_folders[:] = [
            self.cache_dir if p == "/tmp/monet-loras" else p for p in self.loras_folders
        ]
        module.RUN_CACHE_ROOT = self.cache_dir
        module.VOLUME_LORA_ROOT = self.volume_dir
        self.node = module.ComfyUIDeployExternalLora()
        self.secrets = [SIG_MARKER, REF_SOURCE]
        self.logs = ""

    # helpers

    def run_node(self, value, default=None, save_name=None, bearer=None):
        if isinstance(value, str) and value.startswith("mlr1.") and len(value) > 8:
            self.secrets.append(value)
            self.secrets.append(value.split(".")[1])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            try:
                result = self.node.run(
                    "input_lora",
                    default_lora_name=default,
                    lora_save_name=save_name,
                    lora_url=value,
                    bearer_token=bearer,
                )
                error = None
            except Exception as exc:  # noqa: BLE001, the test inspects it
                result, error = None, exc
        logs = out.getvalue()
        self.logs += logs
        self.assert_no_leak(logs, "logs")
        if error is not None:
            self.assert_no_leak(str(error), "error")
        return result, error, logs

    def assert_no_leak(self, text, where):
        for secret in self.secrets:
            if secret:
                self.assertNotIn(secret, text, f"a credential leaked into the {where}")

    def put(self, folder, rel, data):
        path = os.path.join(folder, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def read(self, folder, rel):
        with open(os.path.join(folder, rel), "rb") as f:
            return f.read()

    def files_under(self, folder):
        found = []
        for dirpath, _dirs, files in os.walk(folder):
            for f in files:
                found.append(os.path.relpath(os.path.join(dirpath, f), folder).replace(os.sep, "/"))
        return sorted(found)

    def assert_resolves_to(self, name, folder, rel=None):
        resolved = self.folder_paths.get_full_path("loras", name)
        self.assertIsNotNone(resolved, f"{name} does not resolve in the loras search order")
        self.assertEqual(
            os.path.realpath(resolved), os.path.realpath(os.path.join(folder, rel or name))
        )
