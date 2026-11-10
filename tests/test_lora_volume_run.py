"""The External Lora node on a runner with a read only LoRA volume (C-NODE).

Run from the fork root: python3 -m unittest discover -s tests -v

A signed reference names a LoRA by relPath. The node returns that name when
the volume copy passes its size and sha256 check; otherwise it serves the
bytes from the run cache, downloaded through the inner presigned link and
checked against the reference's size and sha256. A volume copy that fails its
check is never loaded: the node serves `shadow/{rel}` from the cache instead.
"""

import os
import re
import unittest
from unittest import mock

from lora_harness import (
    LoraTestCase,
    REF_SOURCE,
    claims_for,
    make_ref,
    ok_response,
    rel_for,
    safetensors_bytes,
    FakeResponse,
)


class ImportRegistersTheRunCache(LoraTestCase):
    def test_cache_root_is_created_and_appended_last(self):
        self.import_makedirs.assert_any_call("/tmp/monet-loras", exist_ok=True)
        self.assertIn(("loras", "/tmp/monet-loras", False), self.registered)
        self.assertEqual(
            self.folders_after_import,
            [self.volume_dir, self.public_dir, "/tmp/monet-loras"],
            "the run cache must be searched last, never become the default",
        )

    def test_import_survives_an_unwritable_tmp(self):
        # A node that fails to import disables every workflow that uses it;
        # the run fails later, with a named error, instead.
        import importlib.util

        from lora_harness import NODE_PATH

        spec = importlib.util.spec_from_file_location("external_lora_no_tmp", NODE_PATH)
        module = importlib.util.module_from_spec(spec)
        with mock.patch("os.makedirs", side_effect=PermissionError("read only")), mock.patch(
            "sys.stdout"
        ):
            spec.loader.exec_module(module)
        self.assertTrue(hasattr(module, "ComfyUIDeployExternalLora"))

    def test_listing_hides_volume_cache_and_url_names(self):
        good = safetensors_bytes()
        self.put(self.volume_dir, rel_for("a" * 64), good)
        self.put(self.public_dir, "style/film.safetensors", good)
        self.put(self.public_dir, "base.safetensors", good)
        self.put(self.cache_dir, "shadow/" + rel_for("b" * 64), good)
        self.put(self.cache_dir, "url/" + "c" * 32 + ".safetensors", good)
        self.put(self.cache_dir, rel_for("d" * 64), good)
        options = self.module.ComfyUIDeployExternalLora.INPUT_TYPES()["optional"]["default_lora_name"][0]
        self.assertEqual(sorted(options), ["base.safetensors", "style/film.safetensors"])


class VolumeHit(LoraTestCase):
    def test_hit_returns_the_relpath_without_a_request(self):
        data = safetensors_bytes()
        claims = claims_for(data)
        self.put(self.volume_dir, claims["p"], data)
        result, error, logs = self.run_node(make_ref(claims))
        self.assertIsNone(error)
        self.assertEqual(result, (claims["p"],))
        self.assertEqual(self.calls, [])
        self.assertEqual(self.files_under(self.cache_dir), [])
        self.assert_resolves_to(claims["p"], self.volume_dir)
        self.assertIn(claims["p"], logs)

    def test_sha_is_checked_once_per_file_version(self):
        data = safetensors_bytes()
        claims = claims_for(data)
        path = self.put(self.volume_dir, claims["p"], data)
        real = self.module._sha256_file
        counted = mock.MagicMock(side_effect=real)
        self.module._sha256_file = counted
        for _ in range(3):
            result, error, _ = self.run_node(make_ref(claims))
            self.assertIsNone(error)
            self.assertEqual(result, (claims["p"],))
        self.assertEqual(counted.call_count, 1)
        st = os.stat(path)
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
        self.run_node(make_ref(claims))
        self.assertEqual(counted.call_count, 2, "a changed file must be hashed again")

    def test_a_folder_searched_before_the_volume_is_refused(self):
        # The hit is checked at VOLUME_LORA_ROOT. If ComfyUI would find the
        # name in an earlier folder, it would load an unchecked file.
        data = safetensors_bytes()
        claims = claims_for(data)
        self.put(self.volume_dir, claims["p"], data)
        self.put(self.public_dir, claims["p"], safetensors_bytes(fill=b"\3"))
        self.loras_folders[:] = [self.public_dir, self.volume_dir, self.cache_dir]
        _, error, _ = self.run_node(make_ref(claims))
        self.assertIsInstance(error, self.module.LoraRefError)
        self.assertEqual(error.code, "path")

    def test_bearer_token_is_never_sent_with_a_reference(self):
        data = safetensors_bytes()
        claims = claims_for(data)
        self.responses = [ok_response(data)]
        result, error, _ = self.run_node(make_ref(claims), bearer="tok")
        self.assertIsNone(error)
        _, kwargs = self.calls[0]
        self.assertEqual(kwargs["headers"], {"User-Agent": "Mozilla/5.0"})


class VolumeMiss(LoraTestCase):
    def test_miss_downloads_into_the_cache_and_returns_the_relpath(self):
        data = safetensors_bytes(1 << 20)
        claims = claims_for(data)
        self.responses = [ok_response(data)]
        result, error, logs = self.run_node(make_ref(claims))
        self.assertIsNone(error)
        self.assertEqual(result, (claims["p"],))
        self.assertEqual(self.read(self.cache_dir, claims["p"]), data)
        self.assertEqual(self.files_under(self.volume_dir), [], "the volume is read only")
        self.assertEqual(self.files_under(self.cache_dir), [claims["p"]], "a temp file was left")
        self.assert_resolves_to(claims["p"], self.cache_dir)
        url, kwargs = self.calls[0]
        self.assertEqual(url, REF_SOURCE)
        self.assertTrue(kwargs.get("stream"))
        self.assertIsNotNone(kwargs.get("timeout"))

    def test_a_cached_copy_is_reused_without_a_request(self):
        data = safetensors_bytes()
        claims = claims_for(data)
        self.responses = [ok_response(data)]
        self.run_node(make_ref(claims))
        result, error, _ = self.run_node(make_ref(claims))
        self.assertIsNone(error)
        self.assertEqual(result, (claims["p"],))
        self.assertEqual(len(self.calls), 1)

    def test_a_corrupt_cached_copy_is_downloaded_again(self):
        data = safetensors_bytes()
        claims = claims_for(data)
        self.put(self.cache_dir, claims["p"], safetensors_bytes(fill=b"\1"))
        self.responses = [ok_response(data)]
        result, error, _ = self.run_node(make_ref(claims))
        self.assertIsNone(error)
        self.assertEqual(result, (claims["p"],))
        self.assertEqual(self.read(self.cache_dir, claims["p"]), data)

    def test_download_with_the_wrong_sha_is_refused_and_nothing_is_kept(self):
        data = safetensors_bytes()
        claims = claims_for(data)
        self.responses = [ok_response(safetensors_bytes(fill=b"\7"))]
        result, error, _ = self.run_node(make_ref(claims))
        self.assertIsInstance(error, self.module.LoraDownloadError)
        self.assertIn("sha256", str(error))
        self.assertEqual(self.files_under(self.cache_dir), [])

    def test_download_with_the_wrong_size_is_refused_and_nothing_is_kept(self):
        data = safetensors_bytes()
        claims = claims_for(data)
        bigger = safetensors_bytes(128 * 1024)
        for answer in (ok_response(bigger), FakeResponse(200, bigger)):
            with self.subTest(content_length=bool(answer.headers)):
                self.responses = [answer]
                _, error, _ = self.run_node(make_ref(claims))
                self.assertIsInstance(error, self.module.LoraDownloadError)
                self.assertEqual(self.files_under(self.cache_dir), [])

    def test_an_expired_inner_link_names_the_status_not_the_link(self):
        data = safetensors_bytes()
        claims = claims_for(data)
        self.responses = [FakeResponse(403, b"<Error>Request has expired</Error>")]
        _, error, _ = self.run_node(make_ref(claims))
        self.assertIsInstance(error, self.module.LoraDownloadError)
        self.assertIn("HTTP 403", str(error))
        self.assertEqual(self.files_under(self.cache_dir), [])

    def test_a_failed_download_never_shadows_a_later_good_one(self):
        data = safetensors_bytes()
        claims = claims_for(data)
        self.responses = [FakeResponse(403, b"<Error>expired</Error>"), ok_response(data)]
        self.run_node(make_ref(claims))
        result, error, _ = self.run_node(make_ref(claims))
        self.assertIsNone(error)
        self.assertEqual(result, (claims["p"],))
        self.assertEqual(self.read(self.cache_dir, claims["p"]), data)

    def test_a_core_volume_file_at_the_relpath_is_refused_not_loaded(self):
        # The cache is searched after the core volume. A file there under a
        # monet/ name would be what ComfyUI loads, not the checked cache copy.
        data = safetensors_bytes()
        claims = claims_for(data)
        self.put(self.public_dir, claims["p"], safetensors_bytes(fill=b"\5"))
        self.responses = [ok_response(data)]
        _, error, _ = self.run_node(make_ref(claims))
        self.assertIsInstance(error, self.module.LoraRefError)
        self.assertEqual(error.code, "path")
        self.assertEqual(self.calls, [], "nothing should be fetched for a name that cannot be served")

    def test_an_unregistered_run_cache_is_refused_before_the_loader(self):
        data = safetensors_bytes()
        claims = claims_for(data)
        self.loras_folders.remove(self.cache_dir)
        self.responses = [ok_response(data)]
        _, error, _ = self.run_node(make_ref(claims))
        self.assertIsInstance(error, self.module.LoraRefError)
        self.assertEqual(error.code, "path")


class VolumeCopyFailsItsCheck(LoraTestCase):
    def assert_shadow(self, claims, data, logs, result, error):
        self.assertIsNone(error)
        shadow = "shadow/" + claims["p"]
        self.assertEqual(result, (shadow,))
        self.assertIn(f"lora volume copy failed its check: {claims['p']}", logs)
        self.assertEqual(self.read(self.cache_dir, shadow), data)
        self.assert_resolves_to(shadow, self.cache_dir)

    def test_wrong_sha_on_the_volume_is_served_from_shadow(self):
        data = safetensors_bytes()
        claims = claims_for(data)
        bad = safetensors_bytes(fill=b"\2")
        self.assertEqual(len(bad), len(data))
        self.put(self.volume_dir, claims["p"], bad)
        self.responses = [ok_response(data)]
        result, error, logs = self.run_node(make_ref(claims))
        self.assert_shadow(claims, data, logs, result, error)
        self.assertEqual(self.read(self.volume_dir, claims["p"]), bad, "the volume is never written")

    def test_wrong_size_on_the_volume_is_served_from_shadow(self):
        data = safetensors_bytes()
        claims = claims_for(data)
        self.put(self.volume_dir, claims["p"], data + b"\0")
        self.responses = [ok_response(data)]
        result, error, logs = self.run_node(make_ref(claims))
        self.assert_shadow(claims, data, logs, result, error)

    def test_a_volume_symlink_out_of_the_root_is_never_a_hit(self):
        data = safetensors_bytes()
        claims = claims_for(data)
        target = self.put(self.outside_dir, "stolen.safetensors", data)
        link = os.path.join(self.volume_dir, claims["p"])
        os.makedirs(os.path.dirname(link))
        os.symlink(target, link)
        self.responses = [ok_response(data)]
        result, error, logs = self.run_node(make_ref(claims))
        self.assert_shadow(claims, data, logs, result, error)

    def test_a_good_shadow_copy_is_reused(self):
        data = safetensors_bytes()
        claims = claims_for(data)
        self.put(self.volume_dir, claims["p"], safetensors_bytes(fill=b"\2"))
        self.responses = [ok_response(data)]
        self.run_node(make_ref(claims))
        result, error, _ = self.run_node(make_ref(claims))
        self.assertEqual(result, ("shadow/" + claims["p"],))
        self.assertEqual(len(self.calls), 1)


class RefusedValues(LoraTestCase):
    def assert_refused(self, value, code, **kw):
        result, error, _ = self.run_node(value, **kw)
        self.assertIsInstance(error, self.module.LoraRefError, f"accepted {value!r}")
        self.assertEqual(error.code, code)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.files_under(self.cache_dir), [])

    def test_traversal_in_a_reference_is_path(self):
        data = safetensors_bytes()
        rel = f"monet/{'a' * 20}/../{'b' * 20}-" + claims_for(data)["h"][:12] + ".safetensors"
        self.assert_refused(make_ref(claims_for(data, rel=rel)), "path")

    def test_a_foreign_root_is_path(self):
        data = safetensors_bytes()
        rel = "loras/" + claims_for(data)["p"]
        self.assert_refused(make_ref(claims_for(data, rel=rel)), "path")

    def test_a_malformed_reference_is_shape(self):
        self.assert_refused("mlr1.not-a-ref", "shape")

    def test_a_bare_file_name_is_not_a_ref(self):
        self.assert_refused("trained.safetensors", "not_a_ref")
        self.assert_refused(rel_for("e" * 64), "not_a_ref")

    def test_values_that_are_not_strings_are_not_a_ref(self):
        self.assert_refused(5, "not_a_ref")
        self.assert_refused(["mlr1.a.b"], "not_a_ref")

    def test_case_and_scheme_must_match_exactly(self):
        self.assert_refused("HTTPS://bucket.example/x.safetensors", "not_a_ref")
        self.assert_refused("ftp://bucket.example/x.safetensors", "not_a_ref")
        self.assert_refused("httpx://bucket.example/x.safetensors", "not_a_ref")


class DefaultName(LoraTestCase):
    def test_empty_value_returns_the_default(self):
        for value in (None, ""):
            result, error, _ = self.run_node(value, default="style/film.safetensors")
            self.assertIsNone(error)
            self.assertEqual(result, ("style/film.safetensors",))

    def test_a_monet_default_is_refused(self):
        result, error, _ = self.run_node("", default=rel_for("f" * 64))
        self.assertIsInstance(error, self.module.LoraRefError)
        self.assertEqual(error.code, "path")

    def test_cache_names_are_refused_as_defaults(self):
        for default in ("shadow/" + rel_for("f" * 64), "url/" + "0" * 32 + ".safetensors"):
            with self.subTest(default=default):
                _, error, _ = self.run_node(None, default=default)
                self.assertIsInstance(error, self.module.LoraRefError)
                self.assertEqual(error.code, "path")


class RunCacheEviction(LoraTestCase):
    def aged(self, rel, data, age):
        path = self.put(self.cache_dir, rel, data)
        st = os.stat(path)
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns - age * 1_000_000_000))
        return path

    def test_oldest_files_go_until_the_new_one_fits(self):
        blob = safetensors_bytes()
        size = len(blob)
        oldest = self.aged(rel_for("1" * 64), blob, 300)
        middle = self.aged("url/" + "2" * 32 + ".safetensors", blob, 200)
        newest = self.aged(rel_for("3" * 64), blob, 100)
        self.module.RUN_CACHE_MAX_BYTES = 3 * size
        data = safetensors_bytes(fill=b"\4")
        claims = claims_for(data)
        self.responses = [ok_response(data)]
        result, error, _ = self.run_node(make_ref(claims))
        self.assertIsNone(error)
        self.assertFalse(os.path.exists(oldest))
        self.assertTrue(os.path.exists(middle))
        self.assertTrue(os.path.exists(newest))
        self.assertEqual(self.read(self.cache_dir, claims["p"]), data)

    def test_the_file_about_to_be_returned_is_never_evicted(self):
        blob = safetensors_bytes()
        keep = self.aged(rel_for("5" * 64), blob, 900)
        other = self.aged(rel_for("6" * 64), blob, 10)
        self.module.RUN_CACHE_MAX_BYTES = 1
        self.module._evict_run_cache(len(blob), keep)
        self.assertTrue(os.path.exists(keep))
        self.assertFalse(os.path.exists(other))

    def test_url_mode_downloads_also_respect_the_cap(self):
        old = self.aged("url/" + "8" * 32 + ".safetensors", safetensors_bytes(), 500)
        self.module.RUN_CACHE_MAX_BYTES = 1
        self.responses = [ok_response(safetensors_bytes())]
        result, error, _ = self.run_node("https://bucket.example/a.safetensors")
        self.assertIsNone(error)
        self.assertFalse(os.path.exists(old))
        self.assertEqual(self.files_under(self.cache_dir), [result[0]])

    def test_eviction_never_touches_the_volume(self):
        blob = safetensors_bytes()
        on_volume = self.put(self.volume_dir, rel_for("7" * 64), blob)
        self.module.RUN_CACHE_MAX_BYTES = 1
        self.module._evict_run_cache(len(blob), os.path.join(self.cache_dir, "x.safetensors"))
        self.assertTrue(os.path.exists(on_volume))


if __name__ == "__main__":
    unittest.main()
