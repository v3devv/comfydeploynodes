"""The External Lora node must never save an error page as a LoRA.

Run from the fork root, with nothing but the standard library:

    python3 -m unittest discover -s tests -v

No ComfyUI install and no network are needed: `folder_paths` and `requests`
are replaced by small fakes before the node module loads.

The bug this guards: the node wrote `response.content` to the loras folder
without looking at the status. An expired presigned link answers 403 with an
XML body, that body was saved under the LoRA's name, and the run then failed
later with a safetensors parse error that pointed nowhere near the cause.
"""

import contextlib
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE_PATH = os.path.join(ROOT, "comfy-nodes", "external_lora.py")

# The query string of a presigned link. It must never reach a log or an error.
SIG_MARKER = "X-Amz-Signature=" + "deadbeef" + "cafe"
URL = "https://bucket.r2.example/loras/abc.safetensors?" + SIG_MARKER


def safetensors_bytes(payload_size=64 * 1024):
    header = json.dumps(
        {"w": {"dtype": "F32", "shape": [payload_size // 4], "data_offsets": [0, payload_size]}}
    ).encode()
    return struct.pack("<Q", len(header)) + header + b"\0" * payload_size


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

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class NodeTestCase(unittest.TestCase):
    def setUp(self):
        self.lora_dir = tempfile.mkdtemp(prefix="extlora-")
        self.addCleanup(shutil.rmtree, self.lora_dir, True)

        folder_paths = types.ModuleType("folder_paths")
        folder_paths.folder_names_and_paths = {"loras": ([self.lora_dir], set())}
        folder_paths.get_filename_list = lambda name: sorted(
            f for f in os.listdir(self.lora_dir) if f.endswith(".safetensors")
        )
        self.response = None
        self.calls = []

        fake_requests = types.ModuleType("requests")

        def get(url, **kwargs):
            self.calls.append((url, kwargs))
            return self.response

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

        spec = importlib.util.spec_from_file_location("external_lora_under_test", NODE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.node = module.ComfyUIDeployExternalLora()

    def run_node(self, name="trained.safetensors"):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            try:
                result = self.node.run("input_lora", lora_save_name=name, lora_url=URL)
                error = None
            except Exception as exc:  # noqa: BLE001, the test inspects it
                result, error = None, exc
        return result, error, out.getvalue()

    def assert_folder_empty(self):
        self.assertEqual(os.listdir(self.lora_dir), [], "a partial or bad file was left in the loras folder")

    def assert_refused(self, error, logs, *must_mention):
        self.assertIsNotNone(error, "the node accepted a response that is not a LoRA")
        message = str(error)
        for fragment in must_mention:
            self.assertIn(fragment, message)
        self.assertNotIn(SIG_MARKER, message, "the presigned URL leaked into the error")
        self.assertNotIn(SIG_MARKER, logs, "the presigned URL leaked into the logs")
        self.assert_folder_empty()


class RefusesBadResponses(NodeTestCase):
    def test_expired_presigned_link_403(self):
        body = b'<?xml version="1.0"?><Error><Code>AccessDenied</Code><Message>Request has expired</Message></Error>'
        self.response = FakeResponse(403, body, {"Content-Type": "application/xml", "Content-Length": str(len(body))})
        _, error, logs = self.run_node()
        self.assert_refused(error, logs, "HTTP 403")

    def test_not_found_404(self):
        self.response = FakeResponse(404, b"Not Found", {"Content-Length": "9"})
        _, error, logs = self.run_node()
        self.assert_refused(error, logs, "HTTP 404")

    def test_server_error_500(self):
        self.response = FakeResponse(500, b"oops")
        _, error, logs = self.run_node()
        self.assert_refused(error, logs, "HTTP 500")

    def test_200_with_a_tiny_body_is_not_a_lora(self):
        body = b"<html>please sign in</html>"
        self.response = FakeResponse(200, body, {"Content-Length": str(len(body))})
        _, error, logs = self.run_node()
        self.assert_refused(error, logs, "bytes")

    def test_200_without_a_safetensors_header(self):
        body = b"<html>" + b"x" * 8192 + b"</html>"
        self.response = FakeResponse(200, body, {"Content-Length": str(len(body))})
        _, error, logs = self.run_node()
        self.assert_refused(error, logs, "safetensors")

    def test_body_shorter_than_content_length(self):
        body = safetensors_bytes()
        self.response = FakeResponse(200, body, {"Content-Length": str(len(body) + 1000)})
        _, error, logs = self.run_node()
        self.assert_refused(error, logs, "ended early")

    def test_connection_dies_mid_download(self):
        body = safetensors_bytes(1 << 20)
        self.response = FakeResponse(200, body, {"Content-Length": str(len(body))}, fail_after=65536)
        _, error, logs = self.run_node()
        self.assert_refused(error, logs)

    def test_a_failed_download_never_shadows_a_later_good_one(self):
        # The old code left the error page under the final name, and the next
        # run then took the early "already present" branch and reused it.
        self.response = FakeResponse(403, b"<Error>expired</Error>")
        self.run_node()
        good = safetensors_bytes()
        self.response = FakeResponse(200, good, {"Content-Length": str(len(good))})
        result, error, _ = self.run_node()
        self.assertIsNone(error)
        self.assertEqual(result, ("trained.safetensors",))
        with open(os.path.join(self.lora_dir, "trained.safetensors"), "rb") as f:
            self.assertEqual(f.read(), good)


class AcceptsGoodResponses(NodeTestCase):
    def test_good_download_lands_whole_under_the_final_name(self):
        good = safetensors_bytes(1 << 20)
        self.response = FakeResponse(200, good, {"Content-Length": str(len(good))})
        result, error, logs = self.run_node()
        self.assertIsNone(error)
        self.assertEqual(result, ("trained.safetensors",))
        self.assertEqual(os.listdir(self.lora_dir), ["trained.safetensors"], "the temp file was not renamed away")
        with open(os.path.join(self.lora_dir, "trained.safetensors"), "rb") as f:
            self.assertEqual(f.read(), good)
        self.assertNotIn(SIG_MARKER, logs, "the presigned URL leaked into the logs")
        _, kwargs = self.calls[0]
        self.assertTrue(kwargs.get("stream"), "the body should stream to disk, not sit in memory")
        self.assertIsNotNone(kwargs.get("timeout"), "a stalled link must not hang the run forever")

    def test_good_download_without_content_length(self):
        good = safetensors_bytes()
        self.response = FakeResponse(200, good)
        result, error, _ = self.run_node()
        self.assertIsNone(error)
        self.assertEqual(result, ("trained.safetensors",))

    def test_existing_lora_is_reused_without_a_request(self):
        with open(os.path.join(self.lora_dir, "trained.safetensors"), "wb") as f:
            f.write(safetensors_bytes())
        result, error, _ = self.run_node()
        self.assertIsNone(error)
        self.assertEqual(result, ("trained.safetensors",))
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
