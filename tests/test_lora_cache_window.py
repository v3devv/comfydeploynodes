"""Run cache eviction never removes a file the node handed to ComfyUI (C-NODE).

Run from the fork root: python3 -m unittest discover -s tests -v

The node returns a NAME; ComfyUI's LoraLoader opens the file later. A prompt
with two External Lora nodes can run both before either loader, and ComfyUI
reuses a cached node output in a later prompt without running the node again.
So a cache file the node returned stays protected from eviction for
RUN_CACHE_PROTECT_SECONDS after it was last returned. A cache hit keeps its
old mtime, so without this the oldest file, the one just returned, goes first.
Protected files still count toward the cap.
"""

import os
import unittest

from lora_harness import LoraTestCase, claims_for, make_ref, ok_response, safetensors_bytes


class ReturnedFilesSurviveEviction(LoraTestCase):
    def aged(self, path, age):
        st = os.stat(path)
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns - age * 1_000_000_000))

    def cached(self, fill, age):
        data = safetensors_bytes(fill=fill)
        claims = claims_for(data)
        self.aged(self.put(self.cache_dir, claims["p"], data), age)
        return data, claims

    def test_a_cache_hit_returned_earlier_survives_a_later_download(self):
        first_data, first = self.cached(b"\1", 900)
        other_data, other = self.cached(b"\2", 10)
        size = len(first_data)
        self.module.RUN_CACHE_MAX_BYTES = 2 * size
        result, error, _ = self.run_node(make_ref(first))
        self.assertIsNone(error)
        self.assertEqual(result, (first["p"],))
        self.assertEqual(self.calls, [], "the first LoRA is a cache hit")

        second_data = safetensors_bytes(fill=b"\3")
        second = claims_for(second_data)
        self.responses = [ok_response(second_data)]
        result, error, _ = self.run_node(make_ref(second))
        self.assertIsNone(error)
        first_path = os.path.join(self.cache_dir, first["p"])
        self.assertTrue(os.path.exists(first_path), "a returned file was evicted")
        self.assertEqual(self.read(self.cache_dir, first["p"]), first_data)
        self.assertFalse(os.path.exists(os.path.join(self.cache_dir, other["p"])), "the cap was not kept")
        self.assert_resolves_to(first["p"], self.cache_dir)
        self.assertEqual(self.read(self.cache_dir, second["p"]), second_data)

    def test_a_shadow_copy_returned_earlier_survives(self):
        data = safetensors_bytes()
        claims = claims_for(data)
        self.put(self.volume_dir, claims["p"], safetensors_bytes(fill=b"\2"))
        self.responses = [ok_response(data)]
        result, error, _ = self.run_node(make_ref(claims))
        self.assertIsNone(error)
        shadow = os.path.join(self.cache_dir, result[0])
        self.aged(shadow, 900)
        self.module.RUN_CACHE_MAX_BYTES = 1
        self.module._evict_run_cache(len(data), os.path.join(self.cache_dir, "next.safetensors"))
        self.assertTrue(os.path.exists(shadow), "a returned shadow copy was evicted")

    def test_a_url_download_returned_earlier_survives(self):
        self.responses = [ok_response(safetensors_bytes())]
        result, error, _ = self.run_node("https://bucket.example/a.safetensors")
        self.assertIsNone(error)
        path = os.path.join(self.cache_dir, result[0])
        self.aged(path, 900)
        self.module.RUN_CACHE_MAX_BYTES = 1
        self.module._evict_run_cache(0, os.path.join(self.cache_dir, "next.safetensors"))
        self.assertTrue(os.path.exists(path), "a returned URL download was evicted")

    def test_protection_ends_after_the_window(self):
        data, claims = self.cached(b"\1", 900)
        self.run_node(make_ref(claims))
        self.module.RUN_CACHE_PROTECT_SECONDS = 0
        self.module.RUN_CACHE_MAX_BYTES = 1
        self.module._evict_run_cache(len(data), os.path.join(self.cache_dir, "next.safetensors"))
        self.assertFalse(os.path.exists(os.path.join(self.cache_dir, claims["p"])))


if __name__ == "__main__":
    unittest.main()
