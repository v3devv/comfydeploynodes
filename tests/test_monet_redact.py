"""The scrubber the node and custom_routes share (NV2). Standard library only.

Run from the fork root: python3 -m unittest discover -s tests -v
"""

import logging
import os
import sys
import traceback
import types
import unittest
import unittest.mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import monet_redact as r  # noqa: E402
from lora_harness import REF_SOURCE, claims_for, load_fixture, make_ref, safetensors_bytes  # noqa: E402

REF = make_ref(claims_for(safetensors_bytes()))
PAYLOAD = REF.split(".")[1]
SIGNATURE = REF.split(".")[2]


class References(unittest.TestCase):
    def test_a_whole_reference_is_redacted(self):
        self.assertEqual(r.scrub_text(f"value {REF} refused"), "value mlr1.[redacted] refused")

    def test_a_reference_cut_short_is_redacted(self):
        for cut in (REF[:-1], REF[: len("mlr1.") + len(PAYLOAD) + 1], REF[: len("mlr1.") + len(PAYLOAD)],
                    REF[:40], REF[:6]):
            with self.subTest(cut=len(cut)):
                self.assertEqual(r.scrub_text(f"[{cut}]"), "[mlr1.[redacted]]")

    def test_every_fixture_reference_is_redacted(self):
        for case in load_fixture("ref-cases.json")["cases"]:
            ref = case.get("ref")
            if isinstance(ref, str) and ref.startswith("mlr1.") and len(ref) > 5:
                with self.subTest(case=case["name"]):
                    shown = r.scrub_text(f"x {ref} y")
                    self.assertTrue(shown.startswith("x mlr1.[redacted]"), shown)
                    self.assertNotIn(ref.split(".")[1][:12], shown)

    def test_scrubbing_is_idempotent_and_leaves_plain_text(self):
        once = r.scrub_text(f"{REF} {REF_SOURCE} Authorization: Bearer abcdefgh1234")
        self.assertEqual(r.scrub_text(once), once)
        for plain in ("Ext Lora: volume hit monet/a/b/c-0123456789ab.safetensors", "using bearer token",
                      "mlr1.", "a=b&c=d", "https://example.com/a?page=2&size=10", ""):
            with self.subTest(plain=plain):
                self.assertEqual(r.scrub_text(plain), plain)


class SignedLinks(unittest.TestCase):
    def test_a_signed_query_is_stripped_whole(self):
        cases = {
            REF_SOURCE: "https://acct.r2.cloudflarestorage.com/monet-private/ws/loras/p/v.safetensors?[redacted]",
            "https://b.s3.amazonaws.com/k?X-Amz-Credential=AKIA%2F2026&X-Amz-Date=1": "https://b.s3.amazonaws.com/k?[redacted]",
            "https://b.s3.amazonaws.com/k?X-Amz-Security-Token=FQoG": "https://b.s3.amazonaws.com/k?[redacted]",
            "https://d.cloudfront.net/k?Expires=1&Signature=abc~&Key-Pair-Id=K": "https://d.cloudfront.net/k?[redacted]",
            "https://storage.googleapis.com/b/k?X-Goog-Signature=0f": "https://storage.googleapis.com/b/k?[redacted]",
            "https://a.blob.core.windows.net/c/k?sv=2024&sig=abc%2B": "https://a.blob.core.windows.net/c/k?[redacted]",
            "http://h/k?token=abc#frag": "http://h/k?[redacted]",
            "https://h/k?access_token=abc": "https://h/k?[redacted]",
            "https://proxy/?u=https%3A%2F%2Fr2%2Fk%3FX-Amz-Signature%3Dabc": "https://proxy/?[redacted]",
        }
        for url, shown in cases.items():
            with self.subTest(url=url):
                self.assertEqual(r.scrub_text(f"GET '{url}' failed"), f"GET '{shown}' failed")

    def test_a_loose_signing_parameter_keeps_only_its_name(self):
        self.assertEqual(r.scrub_text("...&X-Amz-Signature=deadbeef&x=1"), "...&X-Amz-Signature=[redacted]&x=1")
        self.assertEqual(r.scrub_text("Credential=AKIA/2026, Signature=abc"),
                         "Credential=[redacted], Signature=[redacted]")
        self.assertEqual(r.scrub_text("sig%3Dabc"), "sig%3D[redacted]")


class BearerTokens(unittest.TestCase):
    def test_a_bearer_header_is_redacted(self):
        self.assertEqual(r.scrub_text("Authorization: Bearer eyJhbGci.x.y"), "Authorization: Bearer [redacted]")
        self.assertEqual(r.scrub_text("authorization: bearer abcdefgh"), "authorization: bearer [redacted]")

    def test_a_secret_field_in_a_repr_is_redacted(self):
        self.assertEqual(r.scrub_text("{'bearer_token': 'abc', 'x': 'y'}"), "{'bearer_token': '[redacted]', 'x': 'y'}")
        self.assertEqual(r.scrub_text('{"Authorization": "Basic Zm9v"}'), '{"Authorization": "[redacted]"}')
        self.assertEqual(r.scrub_text("bearer_token='q'"), "bearer_token='[redacted]'")
        self.assertEqual(r.scrub_text("{'bearer_token': ''}"), "{'bearer_token': ''}")
        escaped = '"{\\"bearer_token\\": \\"tok-abcdefgh\\", \\"n\\": 1}"'
        self.assertEqual(r.scrub_text(escaped), '"{\\"bearer_token\\": \\"[redacted]\\", \\"n\\": 1}"')

    def test_a_withheld_literal_is_redacted_wherever_it_occurs(self):
        self.assertEqual(r.scrub_text("token tok-123456 used", withheld=("tok-123456",)), "token [redacted] used")
        self.assertEqual(r.scrub_text("a short tok", withheld=("tok", None, "")), "a short tok")


class NeverRaises(unittest.TestCase):
    def test_any_value_becomes_text(self):
        class Unprintable:
            def __str__(self):
                raise ValueError(REF)

        self.assertEqual(r.scrub_text(None), "None")
        self.assertEqual(r.scrub_text(ValueError(REF)), "mlr1.[redacted]")
        self.assertEqual(r.scrub_text(Unprintable()), "<unprintable Unprintable>")


class Structures(unittest.TestCase):
    def test_a_structure_is_copied_with_every_string_scrubbed(self):
        original = {
            "current_inputs": {"lora_url": [REF], "bearer_token": ["tok-abcdefgh"], "lora_save_name": [""]},
            REF: ("a", REF_SOURCE),
            "n": 3,
            "none": None,
            "Authorization": "Bearer abcdefgh1234",
            "bearer_token": None,
        }
        frozen = repr(original)
        clean = r.scrub_value(original)
        self.assertEqual(repr(original), frozen, "the input is never mutated")
        self.assertEqual(clean["current_inputs"], {
            "lora_url": ["mlr1.[redacted]"], "bearer_token": ["[redacted]"], "lora_save_name": [""]})
        self.assertEqual(clean["mlr1.[redacted]"],
                         ["a", "https://acct.r2.cloudflarestorage.com/monet-private/ws/loras/p/v.safetensors?[redacted]"])
        self.assertEqual((clean["n"], clean["none"], clean["Authorization"], clean["bearer_token"]),
                         (3, None, "[redacted]", None))

    def test_deep_and_cyclic_structures_do_not_crash(self):
        deep = REF
        for _ in range(5000):
            deep = [deep]
        clean = r.scrub_value(deep)
        for _ in range(5000):
            clean = clean[0]
        self.assertEqual(clean, "mlr1.[redacted]")
        loop = {"v": REF}
        loop["self"] = loop
        clean = r.scrub_value(loop)
        self.assertIs(clean["self"], clean)
        self.assertEqual(clean["v"], "mlr1.[redacted]")

    def test_an_event_is_scrubbed_in_place_only_when_it_repeats_inputs(self):
        error = {"current_inputs": {"lora_url": [REF]}}
        same = error
        r.scrub_event_in_place("execution_error", error)
        self.assertIs(error, same)
        self.assertEqual(error["current_inputs"]["lora_url"], ["mlr1.[redacted]"])
        output = {"output": {"url": [REF_SOURCE]}}
        r.scrub_event_in_place("executed", output)
        self.assertEqual(output["output"]["url"], [REF_SOURCE])


class LogLines(unittest.TestCase):
    def test_render_follows_logging(self):
        self.assertEqual(r.render_log("n=%d %s", (3, REF)), "n=3 mlr1.[redacted]")
        self.assertEqual(r.render_log("%(a)s", ({"a": REF},)), "mlr1.[redacted]")
        self.assertEqual(r.render_log("no placeholder", ({"a": REF},)), "no placeholder")
        self.assertEqual(r.render_log("invalid prompt:", ("bad", REF)), "invalid prompt: bad mlr1.[redacted]")
        self.assertEqual(r.render_log({"bearer_token": "tok-abcdefgh"}, ()), "{'bearer_token': '[redacted]'}")
        self.assertEqual(r.render_log("100% done", ()), "100% done")

    def test_a_logger_writes_the_traceback_scrubbed(self):
        inner = logging.getLogger("monet-redact-test")
        logger = r.ScrubbingLogger(inner)
        with self.assertLogs("monet-redact-test", level=logging.INFO) as logs:
            try:
                raise RuntimeError(f"GET {REF_SOURCE}")
            except RuntimeError:
                logger.exception("upload failed")
            logger.info("plain %s", "line")
        self.assertEqual(logs.records[0].exc_info, None)
        self.assertIn("RuntimeError: GET https://acct.r2.cloudflarestorage.com/", logs.output[0])
        self.assertNotIn("X-Amz-Signature=feed", logs.output[0])
        self.assertEqual(logs.output[1], "INFO:monet-redact-test:plain line")
        self.assertEqual(logger.name, "monet-redact-test", "anything else is the wrapped logger's own")

    def test_a_disabled_level_renders_nothing(self):
        inner = logging.getLogger("monet-redact-quiet")
        inner.setLevel(logging.INFO)
        rendered = []
        with unittest.mock.patch.object(r, "render_log", side_effect=lambda *a: rendered.append(a) or ""):
            r.ScrubbingLogger(inner).debug("costly %s", REF)
        self.assertEqual(rendered, [])

    def test_a_logfire_style_logger_gets_scrubbed_attributes(self):
        seen = []
        logfire = types.SimpleNamespace(error=lambda msg, **attrs: seen.append((msg, attrs)))
        r.ScrubbingLogger(logfire).error("Streaming error", error=RuntimeError(f"x {REF}"), n=2)
        self.assertEqual(seen, [("Streaming error", {"error": "x mlr1.[redacted]", "n": 2})])


class RaisedErrors(unittest.TestCase):
    def test_a_clean_error_is_raised_unchanged(self):
        boom = KeyError("missing")

        @r.scrub_raised("bearer_token")
        def run(bearer_token=None):
            raise boom

        with self.assertRaises(KeyError) as caught:
            run(bearer_token="tok-abcdefgh")
        self.assertIs(caught.exception, boom)

    def test_a_leaking_error_is_replaced_and_not_chained(self):
        @r.scrub_raised("bearer_token")
        def run(lora_url, bearer_token=None):
            raise OSError(f"{lora_url} with {bearer_token}")

        for args, kwargs in (((REF, "tok-abcdefgh"), {}), ((REF,), {"bearer_token": "tok-abcdefgh"})):
            with self.subTest(args=args), self.assertRaises(r.ScrubbedError) as caught:
                run(*args, **kwargs)
            self.assertEqual(str(caught.exception), "mlr1.[redacted] with [redacted]")
            shown = "".join(traceback.format_exception(caught.exception))
            self.assertNotIn(PAYLOAD, shown)
            self.assertNotIn("tok-abcdefgh", shown)

    def test_an_unprintable_error_is_withheld(self):
        class Odd(Exception):
            def __str__(self):
                raise ValueError("no")

        @r.scrub_raised()
        def run():
            raise Odd()

        with self.assertRaises(r.ScrubbedError) as caught:
            run()
        self.assertEqual(str(caught.exception), "Odd, its message withheld")


if __name__ == "__main__":
    unittest.main()
