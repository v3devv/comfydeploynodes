"""parse_lora_ref against the frozen golden fixtures (contract C-REF, C-PATH).

Run from the fork root: python3 -m unittest discover -s tests -v

The fixtures in tests/fixtures/lora_volume/ are byte copies of Monet's
docs/fork-plan/todo/lora-volume-fixtures/ (contract section 15). The runner
holds no HMAC key, so the node runs verification steps 1, 2, 4 and 5 only:
a reference with a bad signature, an expired one or one for another
deployment still parses here; Deploy refused those before the run existed.
"""

import json
import re
import unittest

from lora_harness import (
    CLAIM_KEYS,
    DEPLOYMENT,
    LoraTestCase,
    REF_SOURCE,
    b64url,
    canonical_payload,
    load_fixture,
    make_ref,
)


class FixtureRefCases(LoraTestCase):
    def test_every_fixture_case_gets_the_node_answer(self):
        data = load_fixture("ref-cases.json")
        cases = data["cases"]
        self.assertEqual(len(cases), 29, "the fixture should hold 29 verification cases")
        for case in cases:
            with self.subTest(case=case["name"]):
                expect = case["expect"]["node"]
                ref = case["ref"]
                if expect == "ok":
                    claims = self.module.parse_lora_ref(ref)
                    self.assertEqual(list(claims), list(CLAIM_KEYS))
                    if "claims" in case:
                        self.assertEqual(claims, case["claims"])
                    self.assertIs(type(claims["n"]), int)
                    self.assertIs(type(claims["e"]), int)
                else:
                    with self.assertRaises(self.module.LoraRefError) as ctx:
                        self.module.parse_lora_ref(ref)
                    self.assertEqual(ctx.exception.code, expect)
                    message = str(ctx.exception)
                    if ref:
                        self.assertNotIn(ref, message)
                        parts = ref.split(".")
                        if len(parts) > 1 and len(parts[1]) > 8:
                            self.assertNotIn(parts[1], message, "the payload leaked into the error")
                    self.assertIsNone(ctx.exception.__cause__)
                    self.assertTrue(ctx.exception.__suppress_context__ or ctx.exception.__context__ is None)

    def test_node_ignores_signature_time_and_deployment(self):
        # Every case Monet or Deploy refuses for a reason that needs the key,
        # the clock or the run's deployment is a plain parse on the runner.
        keyed = {"signature", "expired", "lifetime", "deployment"}
        cases = load_fixture("ref-cases.json")["cases"]
        seen = 0
        for case in cases:
            reasons = {case["expect"]["monet"], case["expect"]["deploy"]}
            if reasons & keyed:
                seen += 1
                self.assertEqual(case["expect"]["node"], "ok", case["name"])
                self.module.parse_lora_ref(case["ref"])
        self.assertGreaterEqual(seen, 7)

    def test_mint_fixture_ref_parses_to_its_claims(self):
        mint = load_fixture("ref-cases.json")["mint"]
        self.assertEqual(self.module.parse_lora_ref(mint["ref"]), mint["claims"])


class ShapeRules(LoraTestCase):
    def valid(self):
        return load_fixture("ref-cases.json")["mint"]["ref"]

    def assert_code(self, value, code):
        with self.assertRaises(self.module.LoraRefError) as ctx:
            self.module.parse_lora_ref(value)
        self.assertEqual(ctx.exception.code, code)

    def test_values_that_are_not_strings_are_shape(self):
        for value in (None, 5, b"mlr1.a.b", ["mlr1.a.b"], {"p": "x"}):
            with self.subTest(value=type(value).__name__):
                self.assert_code(value, "shape")

    def test_a_trailing_newline_is_shape_not_a_match(self):
        # Python's `$` also matches before a final newline; the node must not.
        self.assert_code(self.valid() + "\n", "shape")
        self.assert_code(" " + self.valid(), "shape")

    def test_non_canonical_base64_is_shape(self):
        prefix, payload, sig = self.valid().split(".")
        # The last character of a 43 character segment carries 2 unused bits;
        # setting one gives the same bytes from a non canonical string.
        last = sig[-1]
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
        bumped = alphabet[alphabet.index(last) + 1]
        self.assert_code(f"{prefix}.{payload}.{sig[:-1]}{bumped}", "shape")
        self.assert_code(f"{prefix}.{payload}.{sig[:-1]}", "shape")
        self.assert_code(f"{prefix}.{payload}=.{sig}", "shape")
        self.assert_code(f"{prefix}.{payload[:-1]}+.{sig}", "shape")

    def test_payload_must_be_a_json_object_with_exactly_six_keys(self):
        claims = load_fixture("ref-cases.json")["mint"]["claims"]
        self.assert_code(make_ref("[1,2,3]"), "shape")
        self.assert_code(make_ref("not json"), "shape")
        self.assert_code(make_ref("[" * 1500 + "]" * 1500), "shape")
        missing = {k: claims[k] for k in CLAIM_KEYS if k != "u"}
        self.assert_code(make_ref(json.dumps(missing, separators=(",", ":"))), "shape")
        dup = canonical_payload(claims)[:-1] + ',"u":' + json.dumps(claims["u"]) + "}"
        self.assert_code(make_ref(dup), "shape")

    def test_claim_types(self):
        claims = load_fixture("ref-cases.json")["mint"]["claims"]
        bad = {
            "n_bool": dict(claims, n=True),
            "n_string": dict(claims, n=str(claims["n"])),
            "e_zero": dict(claims, e=0),
            "e_too_big": dict(claims, e=4294967296),
            "e_bool": dict(claims, e=True),
            "h_short": dict(claims, h=claims["h"][:63]),
            "d_braced": dict(claims, d="{" + DEPLOYMENT + "}"),
            "u_space": dict(claims, u="https://a.example/ x"),
            "u_long": dict(claims, u="https://a.example/" + "a" * 2031),
            "u_empty_host": dict(claims, u="https://"),
            "p_number": dict(claims, p=5),
        }
        for name, c in bad.items():
            with self.subTest(case=name):
                self.assert_code(make_ref(c), "shape")
        edge = dict(claims, u="https://a.example/" + "a" * 2030, n=16384, e=4294967295)
        self.assertEqual(len(edge["u"]), 2048)
        self.assertEqual(self.module.parse_lora_ref(make_ref(edge))["n"], 16384)
        top = dict(claims, n=629145600, e=1)
        self.assertEqual(self.module.parse_lora_ref(make_ref(top))["n"], 629145600)
        self.assert_code(make_ref(dict(claims, n=16383)), "shape")
        self.assert_code(make_ref(dict(claims, n=629145601)), "shape")

    def test_payload_bytes_outside_printable_ascii_are_shape(self):
        claims = load_fixture("ref-cases.json")["mint"]["claims"]
        payload = canonical_payload(claims).replace('"u":', '"u":\t', 1)
        sig = b64url(b"\0" * 32)
        self.assert_code("mlr1." + b64url(payload.encode()) + "." + sig, "shape")

    def test_an_escaped_slash_is_not_canonical(self):
        claims = load_fixture("ref-cases.json")["mint"]["claims"]
        payload = canonical_payload(claims).replace("https://", "https:\\/\\/", 1)
        self.assert_code(make_ref(payload), "shape")


class PathRules(LoraTestCase):
    def test_grammar_constant_is_the_frozen_one(self):
        grammar = load_fixture("rel-path-cases.json")["grammar"]
        self.assertEqual(self.module.REL_PATH.pattern, grammar)

    def ref_with_path(self, rel, sha=None):
        if sha is None:
            head = rel[-24:-12]
            head = head if re.fullmatch(r"[0-9a-f]{12}", head) else "0" * 12
            sha = head + "0" * 52
        return make_ref(
            {"p": rel, "h": sha, "n": 186646528, "d": DEPLOYMENT, "e": 1794319200, "u": REF_SOURCE}
        )

    def test_accepted_paths_parse(self):
        cases = load_fixture("rel-path-cases.json")
        for rel in cases["accept"]:
            with self.subTest(rel=rel):
                self.assertEqual(self.module.parse_lora_ref(self.ref_with_path(rel))["p"], rel)
        for derive in cases["derive"]:
            rel = derive["relPath"]
            sha = derive["input"]["loraSha256"]
            self.assertEqual(self.module.parse_lora_ref(self.ref_with_path(rel, sha))["p"], rel)

    def test_rejected_paths_are_path(self):
        for rel in load_fixture("rel-path-cases.json")["reject"]:
            with self.subTest(rel=rel):
                with self.assertRaises(self.module.LoraRefError) as ctx:
                    self.module.parse_lora_ref(self.ref_with_path(rel))
                self.assertEqual(ctx.exception.code, "path")

    def test_sha12_must_be_the_head_of_h(self):
        rel = load_fixture("rel-path-cases.json")["derive"][0]["relPath"]
        other = "1" * 64
        with self.assertRaises(self.module.LoraRefError) as ctx:
            self.module.parse_lora_ref(self.ref_with_path(rel, other))
        self.assertEqual(ctx.exception.code, "path")


class Constants(LoraTestCase):
    def test_frozen_names(self):
        self.assertEqual(self.module.REF_PREFIX, "mlr1.")
        self.assertEqual(self.module.RUN_CACHE_MAX_BYTES, 8 * 1024 * 1024 * 1024)
        self.assertTrue(issubclass(self.module.LoraRefError, RuntimeError))

    def test_error_codes(self):
        for code in ("shape", "path", "not_a_ref"):
            err = self.module.LoraRefError(code)
            self.assertEqual(err.code, code)
            self.assertIn(code, str(err))


if __name__ == "__main__":
    unittest.main()
