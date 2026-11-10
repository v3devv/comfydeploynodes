"""The External Lora node in URL mode must never save an error page as a LoRA.

Run from the fork root, with nothing but the standard library:

    python3 -m unittest discover -s tests -v

No ComfyUI install and no network are needed: `folder_paths` and `requests`
are replaced by small fakes before the node module loads (tests/lora_harness.py).

The bug this guards (NF1): the node wrote `response.content` to the loras
folder without looking at the status. An expired presigned link answers 403
with an XML body, that body was saved under the LoRA's name, and the run then
failed later with a safetensors parse error that pointed nowhere near the cause.

URL mode (contract C-NODE step 3): a plain http(s) value is accepted only while
MONET_LORA_ALLOW_URL is unset or exactly "1". It downloads into the run cache
as url/{uuid hex}.safetensors, never into the read only LoRA volume and never
under lora_save_name.
"""

import os
import re
import unittest

from lora_harness import (
    SIG_MARKER,
    URL,
    FakeResponse,
    LoraTestCase,
    ok_response,
    safetensors_bytes,
)

URL_NAME = re.compile(r"url/[0-9a-f]{32}\.safetensors")


class UrlModeCase(LoraTestCase):
    def run_url(self, **kw):
        return self.run_node(URL, **kw)

    def assert_nothing_kept(self):
        self.assertEqual(self.files_under(self.cache_dir), [], "a partial or bad file was left in the cache")
        self.assertEqual(self.files_under(self.volume_dir), [], "the volume is read only")

    def assert_refused(self, error, *must_mention):
        self.assertIsNotNone(error, "the node accepted a response that is not a LoRA")
        message = str(error)
        for fragment in must_mention:
            self.assertIn(fragment, message)
        self.assertNotIn(SIG_MARKER, message, "the presigned URL leaked into the error")
        self.assertIsNone(error.__cause__)
        self.assert_nothing_kept()


class RefusesBadResponses(UrlModeCase):
    def test_expired_presigned_link_403(self):
        body = b'<?xml version="1.0"?><Error><Code>AccessDenied</Code><Message>Request has expired</Message></Error>'
        self.responses = [FakeResponse(403, body, {"Content-Type": "application/xml", "Content-Length": str(len(body))})]
        _, error, _ = self.run_url()
        self.assert_refused(error, "HTTP 403")

    def test_not_found_404(self):
        self.responses = [FakeResponse(404, b"Not Found", {"Content-Length": "9"})]
        _, error, _ = self.run_url()
        self.assert_refused(error, "HTTP 404")

    def test_server_error_500(self):
        self.responses = [FakeResponse(500, b"oops")]
        _, error, _ = self.run_url()
        self.assert_refused(error, "HTTP 500")

    def test_200_with_a_tiny_body_is_not_a_lora(self):
        body = b"<html>please sign in</html>"
        self.responses = [FakeResponse(200, body, {"Content-Length": str(len(body))})]
        _, error, _ = self.run_url()
        self.assert_refused(error, "bytes")

    def test_200_without_a_safetensors_header(self):
        body = b"<html>" + b"x" * 8192 + b"</html>"
        self.responses = [FakeResponse(200, body, {"Content-Length": str(len(body))})]
        _, error, _ = self.run_url()
        self.assert_refused(error, "safetensors")

    def test_body_shorter_than_content_length(self):
        body = safetensors_bytes()
        self.responses = [FakeResponse(200, body, {"Content-Length": str(len(body) + 1000)})]
        _, error, _ = self.run_url()
        self.assert_refused(error, "ended early")

    def test_connection_dies_mid_download(self):
        body = safetensors_bytes(1 << 20)
        self.responses = [FakeResponse(200, body, {"Content-Length": str(len(body))}, fail_after=65536)]
        _, error, _ = self.run_url()
        self.assert_refused(error)

    def test_a_connect_error_never_carries_the_link(self):
        # requests puts the full URL, query string included, in its exception
        # text. The node must not pass that text on.
        self.responses = [ConnectionError(f"Max retries exceeded with url: {URL}")]
        _, error, _ = self.run_url()
        self.assertIsInstance(error, self.module.LoraDownloadError)
        self.assert_refused(error, "ConnectionError")
        self.assertTrue(error.__suppress_context__ or error.__context__ is None)


class AcceptsGoodResponses(UrlModeCase):
    def test_good_download_lands_whole_in_the_cache(self):
        good = safetensors_bytes(1 << 20)
        self.responses = [ok_response(good)]
        result, error, logs = self.run_url()
        self.assertIsNone(error)
        self.assertEqual(len(result), 1)
        name = result[0]
        self.assertRegex(name, URL_NAME)
        self.assertEqual(self.files_under(self.cache_dir), [name], "the temp file was not renamed away")
        self.assertEqual(self.read(self.cache_dir, name), good)
        self.assertEqual(self.files_under(self.volume_dir), [], "URL mode never writes the volume")
        self.assert_resolves_to(name, self.cache_dir)
        _, kwargs = self.calls[0]
        self.assertTrue(kwargs.get("stream"), "the body should stream to disk, not sit in memory")
        self.assertIsNotNone(kwargs.get("timeout"), "a stalled link must not hang the run forever")

    def test_good_download_without_content_length(self):
        good = safetensors_bytes()
        self.responses = [FakeResponse(200, good)]
        result, error, _ = self.run_url()
        self.assertIsNone(error)
        self.assertRegex(result[0], URL_NAME)

    def test_each_download_gets_its_own_name(self):
        good = safetensors_bytes()
        self.responses = [ok_response(good), ok_response(good)]
        first, _, _ = self.run_url()
        second, _, _ = self.run_url()
        self.assertNotEqual(first, second)

    def test_bearer_token_is_sent_and_never_logged(self):
        good = safetensors_bytes()
        self.responses = [ok_response(good)]
        _, error, logs = self.run_url(bearer="secret-bearer-value")
        self.assertIsNone(error)
        _, kwargs = self.calls[0]
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer secret-bearer-value")
        self.assertNotIn("secret-bearer-value", logs)


class UrlModeGate(UrlModeCase):
    def test_unset_allows_urls(self):
        os.environ.pop("MONET_LORA_ALLOW_URL", None)
        self.responses = [ok_response(safetensors_bytes())]
        result, error, _ = self.run_url()
        self.assertIsNone(error)
        self.assertRegex(result[0], URL_NAME)

    def test_one_allows_urls(self):
        os.environ["MONET_LORA_ALLOW_URL"] = "1"
        self.responses = [ok_response(safetensors_bytes())]
        result, error, _ = self.run_node("http://bucket.example/a.safetensors")
        self.assertIsNone(error)
        self.assertRegex(result[0], URL_NAME)

    def test_zero_and_any_other_value_refuse_urls(self):
        for value in ("0", "", "true", "yes", " 1", "1 ", "01"):
            with self.subTest(value=value):
                os.environ["MONET_LORA_ALLOW_URL"] = value
                self.responses = [ok_response(safetensors_bytes())]
                _, error, _ = self.run_url()
                self.assertIsInstance(error, self.module.LoraRefError)
                self.assertEqual(error.code, "not_a_ref")
                self.assertEqual(self.calls, [])
                self.assert_nothing_kept()

    def test_lora_save_name_is_refused(self):
        self.responses = [ok_response(safetensors_bytes())]
        _, error, _ = self.run_url(save_name="trained.safetensors")
        self.assertIsInstance(error, self.module.LoraRefError)
        self.assertEqual(self.calls, [])
        self.assert_nothing_kept()


if __name__ == "__main__":
    unittest.main()
