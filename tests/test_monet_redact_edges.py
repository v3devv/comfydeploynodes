"""Edges of the NV2 scrubber found in review. Standard library only.

Run from the fork root: python3 -m unittest discover -s tests -v

Each case is a form the first cut let through or rendered differently from
`logging`: a reference whose separator arrives escaped, a signing parameter
inside a link carried percent encoded in another link, a secret field whose
value is a one item list, a leaking exception chained under a clean one, and
log arguments that `%` formats by type.
"""

import os
import sys
import traceback
import unittest
from decimal import Decimal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import monet_redact as r  # noqa: E402
from lora_harness import REF_SOURCE, claims_for, make_ref, safetensors_bytes  # noqa: E402

REF = make_ref(claims_for(safetensors_bytes()))
PAYLOAD = REF.split(".")[1]
SIGNATURE = REF.split(".")[2]
TOKEN = "tok-" + "5ecret" * 4


class EscapedReferences(unittest.TestCase):
    def test_an_escaped_separator_still_hides_the_payload(self):
        for sep in ("\\u002e", "\\\\u002e", "\\u002E", "%2E", "%2e"):
            with self.subTest(sep=sep):
                shown = r.scrub_text(f"x mlr1{sep}{PAYLOAD}{sep}{SIGNATURE} y")
                self.assertEqual(shown, "x mlr1.[redacted] y")

    def test_an_upper_case_prefix_is_still_a_reference(self):
        self.assertEqual(r.scrub_text(f"MLR1.{PAYLOAD}.{SIGNATURE}"), "mlr1.[redacted]")

    def test_plain_text_and_scrubbed_text_stay_as_they_are(self):
        for plain in ("mlr1.", "mlr1%2E", "html1.x", "mlr1.[redacted]", "x mlr1.[redacted] y"):
            with self.subTest(plain=plain):
                self.assertEqual(r.scrub_text(plain), plain)


class NestedSignedLinks(unittest.TestCase):
    def test_a_link_inside_a_link_loses_its_signing_parameters(self):
        cases = {
            "https://proxy/?u=https%3A%2F%2Fa.blob%2Fc%3Fsv%3D2024%26sig%3D" + "c0ffee" * 6: "https://proxy/?[redacted]",
            "https://proxy/?u=https%3A%2F%2Fh%2Fk%3Ftoken%3D" + "7e57" * 6: "https://proxy/?[redacted]",
            "https://proxy/?u=https%3A%2F%2Fd.cloudfront.net%2Fk%3FExpires%3D1%26Signature%3Dabc": "https://proxy/?[redacted]",
        }
        for url, shown in cases.items():
            with self.subTest(url=url):
                self.assertEqual(r.scrub_text(f"GET {url} failed"), f"GET {shown} failed")

    def test_a_loose_encoded_parameter_keeps_only_its_name(self):
        self.assertEqual(r.scrub_text("q=1%26sig%3Dc0ffee"), "q=1%26sig%3D[redacted]")
        self.assertEqual(r.scrub_text("k%3FCredential%3DAKIA%2F2026"), "k%3FCredential%3D[redacted]")


class SecretFieldsAsText(unittest.TestCase):
    def test_a_one_item_list_value_is_withheld(self):
        self.assertEqual(r.scrub_text(f"{{'bearer_token': ['{TOKEN}'], 'n': 1}}"),
                         "{'bearer_token': ['[redacted]'], 'n': 1}")
        self.assertEqual(r.scrub_text(f'{{"Authorization": [ "Basic {TOKEN}" ]}}'),
                         '{"Authorization": [ "[redacted]" ]}')

    def test_a_tuple_keyword_value_never_shows_the_token(self):
        self.assertNotIn(TOKEN, r.scrub_text(f"bearer_token=('{TOKEN}',)"))

    def test_an_empty_list_value_is_kept(self):
        self.assertEqual(r.scrub_text("{'bearer_token': []}"), "{'bearer_token': []}")


class ChainedErrors(unittest.TestCase):
    def assert_shows_nothing(self, error):
        shown = "".join(traceback.format_exception(error))
        self.assertNotIn(PAYLOAD[:24], shown)
        self.assertNotIn("feedface", shown)
        self.assertNotIn(TOKEN, shown)
        return shown

    def test_a_leaking_cause_is_dropped_and_the_error_kept(self):
        @r.scrub_raised()
        def run():
            try:
                raise OSError(f"GET {REF_SOURCE}")
            except OSError as exc:
                raise LookupError("download failed") from exc

        with self.assertRaises(LookupError) as caught:
            run()
        shown = self.assert_shows_nothing(caught.exception)
        self.assertTrue(shown.rstrip().endswith("LookupError: download failed"))

    def test_a_leaking_context_is_dropped(self):
        @r.scrub_raised("bearer_token")
        def run(bearer_token=None):
            try:
                raise ValueError(f"{REF} with {bearer_token}")
            except ValueError:
                raise LookupError("refused")  # noqa: B904, the implicit context is the point

        with self.assertRaises(LookupError) as caught:
            run(bearer_token=TOKEN)
        self.assert_shows_nothing(caught.exception)

    def test_a_leaking_note_is_scrubbed(self):
        @r.scrub_raised()
        def run():
            error = LookupError("refused")
            error.__notes__ = [f"while fetching {REF}"]  # what add_note writes, on any Python
            raise error

        with self.assertRaises(LookupError) as caught:
            run()
        self.assert_shows_nothing(caught.exception)
        self.assertEqual(caught.exception.__notes__, ["while fetching mlr1.[redacted]"])

    def test_a_clean_chain_is_raised_unchanged(self):
        inner = KeyError("missing")

        @r.scrub_raised()
        def run():
            raise LookupError("refused") from inner

        with self.assertRaises(LookupError) as caught:
            run()
        self.assertIs(caught.exception.__cause__, inner)


class LogArgumentsFormatByType(unittest.TestCase):
    def test_a_number_like_argument_still_formats_as_a_number(self):
        self.assertEqual(r.render_log("n=%d", (Decimal(3),)), "n=3")
        self.assertEqual(r.render_log("%.1f s", (Decimal("2.25"),)), "2.2 s")

    def test_repr_of_a_non_text_argument_is_logging_s(self):
        self.assertEqual(r.render_log("%r", (b"raw",)), "b'raw'")
        self.assertEqual(r.render_log("%r", (REF.encode(),)), "b'mlr1.[redacted]'")

    def test_an_object_argument_is_still_scrubbed(self):
        self.assertEqual(r.render_log("failed: %s", (ValueError(f"GET {REF}"),)), "failed: GET mlr1.[redacted]")


if __name__ == "__main__":
    unittest.main()
