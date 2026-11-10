"""The External Lora node never writes a credential into a log or an error (NV2).

Run from the fork root: python3 -m unittest discover -s tests -v

ComfyUI reports a node's exception message and traceback to Deploy, and the
run log follows. `run_node` fails a test whenever the reference, its payload,
the inner presigned link or a bearer token shows up in either. On a real run
`input_id` holds the same value as `lora_url` (custom_routes swaps both in).
"""

import traceback
import unittest
from unittest import mock

from lora_harness import (
    REF_SOURCE,
    URL,
    FakeResponse,
    LoraTestCase,
    claims_for,
    make_ref,
    safetensors_bytes,
)


class BakedDefaultsAreNotPrinted(LoraTestCase):
    def test_a_baked_default_link_is_never_printed(self):
        result, error, logs = self.run_node(None, default=URL)
        self.assertIsNone(error)
        self.assertEqual(result, (URL,), "the default still goes to the loader as it is")
        self.assertIn("Ext Lora loading: https://bucket.r2.example/loras/abc.safetensors?[redacted]", logs)

    def test_a_baked_default_reference_is_never_printed(self):
        ref = make_ref(claims_for(safetensors_bytes()))
        self.secrets += [ref, ref.split(".")[1]]
        result, error, logs = self.run_node("", default=ref)
        self.assertIsNone(error)
        self.assertEqual(result, (ref,))
        self.assertIn("Ext Lora loading: mlr1.[redacted]", logs)


class ErrorsCarryNoCredential(LoraTestCase):
    def test_a_failing_close_cannot_replace_the_error_with_the_link(self):
        ref = make_ref(claims_for(safetensors_bytes()))
        answer = FakeResponse(403, b"denied")

        def close():
            raise ConnectionError(f"pool closed while reading {REF_SOURCE}")

        answer.close = close
        self.responses.append(answer)
        _result, error, _logs = self.run_node(ref, input_id=ref)
        self.assertIsInstance(error, self.module.LoraDownloadError)
        self.assertIn("failed with HTTP 403", str(error))

    def assert_scrubbed_error(self, error, *expected):
        self.assertIsInstance(error, RuntimeError)
        self.assertTrue(error.__suppress_context__, "the leaking original must not be chained")
        shown = "".join(traceback.format_exception(error))
        self.assert_no_leak(shown, "formatted traceback")
        for text in expected:
            self.assertIn(text, str(error))

    def test_an_unexpected_error_with_the_reference_is_scrubbed(self):
        ref = make_ref(claims_for(safetensors_bytes()))

        def boom(_requests, url, _headers, _destination, **_kw):
            raise ValueError(f"cannot fetch {ref} via {url}")

        with mock.patch.object(self.module, "_download_lora", boom):
            _result, error, _logs = self.run_node(ref, input_id=ref)
        self.assert_scrubbed_error(
            error,
            "cannot fetch mlr1.[redacted] via https://acct.r2.cloudflarestorage.com/monet-private/ws/loras/p/"
            "v.safetensors?[redacted]",
        )

    def test_an_unexpected_error_with_a_link_and_bearer_token_is_scrubbed(self):
        bearer = "tok-" + "b3arer" * 4
        self.secrets.append(bearer)

        def boom(_requests, url, headers, _destination, **_kw):
            raise RuntimeError(f"GET {url} with {headers} (token {bearer})")

        with mock.patch.object(self.module, "_download_lora", boom):
            _result, error, _logs = self.run_node(URL, bearer=bearer, input_id=URL)
        self.assert_scrubbed_error(
            error,
            "GET https://bucket.r2.example/loras/abc.safetensors?[redacted] with ",
            "'Authorization': '[redacted]'",
            "(token [redacted])",
        )

    def test_a_refusal_never_carries_a_value_even_when_handed_one(self):
        ref = make_ref(claims_for(safetensors_bytes()))
        self.secrets += [ref, ref.split(".")[1]]
        error = self.module.LoraRefError("path", f"{ref} via {REF_SOURCE}")
        self.assert_no_leak(str(error), "refusal")
        self.assertEqual(error.code, "path")
        self.assertIn("(path): mlr1.[redacted] via https://acct.r2.cloudflarestorage.com/", str(error))

    def test_a_clean_error_is_raised_unchanged(self):
        _result, error, _logs = self.run_node("not a reference")
        self.assertIsInstance(error, self.module.LoraRefError)
        self.assertEqual(error.code, "not_a_ref")
        self.assertEqual(
            str(error),
            "External Lora refused the value (not_a_ref): "
            "this Engine loads a LoRA only from a signed Monet reference.",
        )


if __name__ == "__main__":
    unittest.main()
