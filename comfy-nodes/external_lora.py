"""External Lora: loads a Monet LoRA from the read only LoRA volume (contract C-NODE).

`lora_url` holds a signed reference `mlr1.{payload}.{signature}`, a plain
http(s) URL (only while MONET_LORA_ALLOW_URL allows it) or nothing (use
`default_lora_name`). The runner holds no key: Deploy checked the signature,
expiry and deployment before the run existed, so this node checks shape and
path, then loads the relPath from the volume when its size and sha256 pass, or
from the run cache through the inner presigned link. It never writes the
volume. Never log a reference, its inner link or headers; a relPath may be.
"""

import base64
import binascii
import hashlib
import json
import os
import re
import stat
import time
import uuid

import folder_paths

REF_PREFIX = "mlr1."
REL_PATH = re.compile(r"^monet/[a-z0-9]{20,40}/[a-z0-9]{20,40}/[a-z0-9]{20,40}-[0-9a-f]{12}\.safetensors$")
RUN_CACHE_ROOT = "/tmp/monet-loras"
RUN_CACHE_MAX_BYTES = 8 * 1024 * 1024 * 1024
# A LoraLoader opens a returned cache file later, maybe after another External Lora node in the
# prompt evicted, or via a node output ComfyUI reuses. Eviction skips it this long (a ref's life).
RUN_CACHE_PROTECT_SECONDS = 7200
# The LoRA volume as ComfyUI's default loras folder sees it (it symlinks to
# /private_models). The node only ever reads it.
VOLUME_LORA_ROOT = "/comfyui/models/loras"
ALLOW_URL_ENV = "MONET_LORA_ALLOW_URL"
SHADOW_PREFIX = "shadow/"
URL_PREFIX = "url/"
# Names that only a reference may produce. They are hidden from the default
# list and refused as a baked default.
PRIVATE_PREFIXES = ("monet/", SHADOW_PREFIX, URL_PREFIX)

REF_MAX_CHARS = 4096
REF_MIN_BYTES = 16384
REF_MAX_BYTES = 629145600
CLAIM_KEYS = ("p", "h", "n", "d", "e", "u")
_SEGMENT = re.compile(r"[A-Za-z0-9_-]+")
_SIGNATURE = re.compile(r"[A-Za-z0-9_-]{43}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_DEPLOYMENT = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_SOURCE = re.compile(r"https://[\x21-\x7E]+")
_SOURCE_MAX_CHARS = 2048

# A LoRA smaller than this is an error page, not weights. The smallest real
# LoRAs are hundreds of KiB; the error bodies this catches are under a few KiB.
MIN_LORA_BYTES = 16 * 1024
# Larger than any LoRA we train or load; a body past this is not a LoRA either.
MAX_LORA_BYTES = 8 * 1024 * 1024 * 1024
# (connect, read between chunks) in seconds, so a stalled link fails the run
# instead of holding the GPU until the run timeout.
DOWNLOAD_TIMEOUT = (30, 120)
CHUNK_BYTES = 1024 * 1024

_MESSAGES = {
    "shape": "the value is not a well formed Monet LoRA reference",
    "path": "the LoRA path is outside the Monet volume grammar",
    "not_a_ref": "this Engine loads a LoRA only from a signed Monet reference",
}


class LoraRefError(RuntimeError):
    """A value this node will not load. `.code` is "shape", "path" or "not_a_ref"; the message
    never carries the value."""

    def __init__(self, code, detail=None):
        self.code = code
        super().__init__(f"External Lora refused the value ({code}): {detail or _MESSAGES.get(code, code)}.")


class LoraDownloadError(RuntimeError):
    """The link did not serve a usable LoRA. The message never carries the URL."""


def _is_claim_int(value, low, high):
    return type(value) is int and low <= value <= high


def _b64url_canonical(segment):
    """The bytes of a canonical unpadded base64url segment, else None."""
    try:
        raw = base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))
    except (binascii.Error, ValueError):
        return None
    if base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii") != segment:
        return None
    return raw


def parse_lora_ref(value):
    """Claims of a reference, checked for shape and path (C-REF steps 1, 2, 4
    and 5). The signature, expiry and deployment are Deploy's to check; this
    node holds no key. Raises LoraRefError("shape") or LoraRefError("path")."""
    # 1. A bounded string with the prefix and exactly three parts.
    if not isinstance(value, str) or len(value) > REF_MAX_CHARS or not value.startswith(REF_PREFIX):
        raise LoraRefError("shape")
    parts = value.split(".")
    if len(parts) != 3:
        raise LoraRefError("shape")
    _prefix, payload_segment, signature_segment = parts
    # 2. Canonical base64url segments, a 43 character signature.
    if not _SEGMENT.fullmatch(payload_segment) or not _SIGNATURE.fullmatch(signature_segment):
        raise LoraRefError("shape")
    payload = _b64url_canonical(payload_segment)
    if payload is None or _b64url_canonical(signature_segment) is None:
        raise LoraRefError("shape")
    # 3. The HMAC is Deploy's.
    # 4. Printable ASCII, exactly the six claims, each to its rule, and the
    #    canonical serialisation reproduces the payload byte for byte.
    if any(b < 0x20 or b > 0x7E for b in payload):
        raise LoraRefError("shape")
    text = payload.decode("ascii")
    try:
        obj = json.loads(text)
    except (ValueError, RecursionError):
        raise LoraRefError("shape") from None
    if not isinstance(obj, dict) or len(obj) != len(CLAIM_KEYS) or set(obj) != set(CLAIM_KEYS):
        raise LoraRefError("shape")
    h, n, d, e, u = obj["h"], obj["n"], obj["d"], obj["e"], obj["u"]
    if not (
        isinstance(obj["p"], str)
        and isinstance(h, str)
        and _SHA256.fullmatch(h)
        and _is_claim_int(n, REF_MIN_BYTES, REF_MAX_BYTES)
        and isinstance(d, str)
        and _DEPLOYMENT.fullmatch(d)
        and _is_claim_int(e, 1, 4294967295)
        and isinstance(u, str)
        and len(u) <= _SOURCE_MAX_CHARS
        and _SOURCE.fullmatch(u)
    ):
        raise LoraRefError("shape")
    claims = {k: obj[k] for k in CLAIM_KEYS}
    if json.dumps(claims, separators=(",", ":"), ensure_ascii=True) != text:
        raise LoraRefError("shape")
    # 5. The grammar, and the sha12 in the path is the head of h.
    rel = claims["p"]
    if not REL_PATH.fullmatch(rel) or rel[-24:-12] != h[:12]:
        raise LoraRefError("path")
    return claims


def _is_under(path, root):
    return path.startswith(root.rstrip(os.sep) + os.sep)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


_SHA_MEMO = {}
_SHA_MEMO_MAX = 4096


def _remember_sha(key, ok):
    if len(_SHA_MEMO) >= _SHA_MEMO_MAX:
        _SHA_MEMO.clear()
    _SHA_MEMO[key] = ok


def _verify_sha_once(path, sha256):
    """True when the file's sha256 is `sha256`. Memoised per process on
    (realpath, size, mtime, sha256), so each file version is hashed once."""
    real = os.path.realpath(path)
    try:
        before = os.stat(real)
    except OSError:
        return False
    key = (real, before.st_size, before.st_mtime_ns, sha256)
    if key in _SHA_MEMO:
        return _SHA_MEMO[key]
    try:
        ok = _sha256_file(real) == sha256
        after = os.stat(real)
    except OSError:
        return False
    if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
        return False  # changed while hashing: no verdict, and none remembered
    _remember_sha(key, ok)
    return ok


def _checked_file(root, name, size, sha256):
    """True when root/name resolves under root to a regular file of `size`
    bytes whose sha256 is `sha256`."""
    real_root = os.path.realpath(root)
    real = os.path.realpath(os.path.join(root, name))
    if not _is_under(real, real_root):
        return False
    try:
        st = os.stat(real)
    except OSError:
        return False
    if not stat.S_ISREG(st.st_mode) or st.st_size != size:
        return False
    return _verify_sha_once(real, sha256)


def _volume_hit(rel, size, sha256):
    """True when the read only volume holds rel with the reference's size and
    sha256, resolving under the volume's loras root."""
    return isinstance(rel, str) and bool(REL_PATH.fullmatch(rel)) and _checked_file(VOLUME_LORA_ROOT, rel, size, sha256)


_RETURNED = {}  # abspath of a returned cache file: time.monotonic() at its last return


def _mark_returned(name):
    _RETURNED[os.path.abspath(os.path.join(RUN_CACHE_ROOT, name))] = time.monotonic()


def _evict_run_cache(incoming_bytes, keep_path):
    """Delete the least recently modified run cache files until the cache plus `incoming_bytes`
    fits RUN_CACHE_MAX_BYTES. Never `keep_path`, nor a file returned in the last
    RUN_CACHE_PROTECT_SECONDS (those still count toward the cap)."""
    keep = os.path.abspath(keep_path)
    now = time.monotonic()
    for path, at in list(_RETURNED.items()):
        if now - at >= RUN_CACHE_PROTECT_SECONDS:
            del _RETURNED[path]
    entries = []
    total = 0
    for dirpath, _dirs, files in os.walk(RUN_CACHE_ROOT):
        for fn in files:
            path = os.path.abspath(os.path.join(dirpath, fn))
            if path == keep:
                continue
            try:
                st = os.lstat(path)
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            total += st.st_size
            if path not in _RETURNED:
                entries.append((st.st_mtime_ns, path, st.st_size))
    entries.sort()
    for _mtime, path, size in entries:
        if total + incoming_bytes <= RUN_CACHE_MAX_BYTES:
            break
        try:
            os.remove(path)
            total -= size
        except OSError:
            pass


def _check_safetensors_header(first_bytes, total):
    # safetensors: 8 byte little endian header length, then a JSON object.
    if len(first_bytes) < 9:
        return False
    header_len = int.from_bytes(first_bytes[:8], "little")
    return 0 < header_len <= total - 8 and first_bytes[8:9] == b"{"


def _download_lora(requests, url, headers, destination_path, expected_sha256=None, expected_size=None):
    """Download url to destination_path, or raise LoraDownloadError.

    The body goes to a hidden temp name in the same folder and is renamed into
    place only once it is whole, plausible and, when given, of the expected
    size and sha256, so a failed or partial download never sits under the
    LoRA's name (where the next run would reuse it).
    """
    folder = os.path.dirname(destination_path)
    name = os.path.basename(destination_path)
    temp_path = os.path.join(folder, f".{name}.{uuid.uuid4().hex}.part")

    def fail(text):
        return LoraDownloadError(f"External LoRA download for {name} {text}")

    try:
        response = requests.get(url, headers=headers, allow_redirects=True, stream=True, timeout=DOWNLOAD_TIMEOUT)
    except Exception as exc:
        # requests puts the full URL, signature included, in its message.
        raise fail(f"failed: {type(exc).__name__}.") from None
    try:
        status = response.status_code
        if not 200 <= status < 300:
            raise fail(
                f"failed with HTTP {status}. "
                "If the link was presigned it has likely expired; request a fresh link."
            )
        declared = None
        raw_length = response.headers.get("Content-Length")
        encoding = (response.headers.get("Content-Encoding") or "identity").lower()
        if raw_length is not None and encoding == "identity":
            try:
                declared = int(raw_length)
            except ValueError:
                declared = None
        if declared is not None and not MIN_LORA_BYTES <= declared <= MAX_LORA_BYTES:
            raise fail(f"answered HTTP {status} with {declared} bytes, which is not a plausible LoRA size.")
        if declared is not None and expected_size is not None and declared != expected_size:
            raise fail(f"answered HTTP {status} with {declared} bytes, expected {expected_size}.")
        limit = MAX_LORA_BYTES if expected_size is None else expected_size

        digest = hashlib.sha256() if expected_sha256 is not None else None
        received = 0
        head = b""
        with open(temp_path, "wb") as out_file:
            for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                if not chunk:
                    continue
                if len(head) < 9:
                    head += chunk[: 9 - len(head)]
                received += len(chunk)
                if received > limit:
                    raise fail(f"passed {limit} bytes, which is not the LoRA it should be.")
                if digest is not None:
                    digest.update(chunk)
                out_file.write(chunk)
            out_file.flush()
            os.fsync(out_file.fileno())

        if declared is not None and received != declared:
            raise fail(f"ended early: received {received} of {declared} bytes.")
        if received < MIN_LORA_BYTES:
            raise fail(f"answered HTTP {status} with only {received} bytes, which is not a plausible LoRA size.")
        if expected_size is not None and received != expected_size:
            raise fail(f"received {received} bytes, expected {expected_size}.")
        if name.endswith(".safetensors") and not _check_safetensors_header(head, received):
            raise fail(f"answered HTTP {status} but the body is not a safetensors file.")
        if digest is not None and digest.hexdigest() != expected_sha256:
            raise fail("did not match its sha256; nothing was kept.")
        os.replace(temp_path, destination_path)
    except LoraDownloadError:
        raise
    except Exception as exc:
        raise fail(f"failed: {type(exc).__name__}.") from None
    finally:
        response.close()
        if os.path.exists(temp_path):
            os.remove(temp_path)
    if expected_sha256 is not None:  # hashed on the way in, so the next run need not rehash
        try:
            real = os.path.realpath(destination_path)
            st = os.stat(real)
            _remember_sha((real, st.st_size, st.st_mtime_ns, expected_sha256), True)
        except OSError:
            pass


def _cache_destination(name):
    """RUN_CACHE_ROOT/name with its folder created, confined to the cache."""
    destination = os.path.join(RUN_CACHE_ROOT, name)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    real_root = os.path.realpath(RUN_CACHE_ROOT)
    real_folder = os.path.realpath(os.path.dirname(destination))
    if real_folder != real_root and not _is_under(real_folder, real_root):
        raise LoraRefError("path", "the run cache path escapes the cache")
    return destination


def _resolves_to(name, expected_real):
    """True when ComfyUI's loras lookup for `name` lands on `expected_real`."""
    resolved = folder_paths.get_full_path("loras", name)
    return resolved is not None and os.path.realpath(resolved) == expected_real


def _resolves_elsewhere(name):
    """True when ComfyUI would load `name` from a folder other than the cache."""
    resolved = folder_paths.get_full_path("loras", name)
    return resolved is not None and not _is_under(os.path.realpath(resolved), os.path.realpath(RUN_CACHE_ROOT))


def _serve_from_cache(requests, name, claims):
    """Serve `name` from the run cache: reuse a checked copy, else download it
    through the reference's inner link with its size and sha256 checked."""
    rel = claims["p"]
    if _resolves_elsewhere(name):
        # A folder searched before the cache already holds this name, so
        # ComfyUI would load that file, not the checked one.
        raise LoraRefError("path", f"{name} is shadowed by a folder searched before the run cache")
    if _checked_file(RUN_CACHE_ROOT, name, claims["n"], claims["h"]):
        print(f"Ext Lora: run cache hit {name}")
    else:
        destination = _cache_destination(name)
        _evict_run_cache(claims["n"], destination)
        print(f"Ext Lora: fetching {rel} into the run cache")
        _download_lora(requests, claims["u"], {"User-Agent": "Mozilla/5.0"}, destination,
                       expected_sha256=claims["h"], expected_size=claims["n"])
    return name


try:
    os.makedirs(RUN_CACHE_ROOT, exist_ok=True)
except OSError as exc:
    print(f"Ext Lora: could not create the run cache {RUN_CACHE_ROOT}: {type(exc).__name__}")
folder_paths.add_model_folder_path("loras", RUN_CACHE_ROOT)


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


WILDCARD = AnyType("*")


class ComfyUIDeployExternalLora:
    @classmethod
    def INPUT_TYPES(s):
        names = [n for n in folder_paths.get_filename_list("loras") if not n.startswith(PRIVATE_PREFIXES)]
        text = lambda multiline=False: ("STRING", {"multiline": multiline, "default": ""})
        return {
            "required": {"input_id": ("STRING", {"multiline": False, "default": "input_lora"})},
            "optional": {
                "default_lora_name": (names,),
                # Must stay empty: a downloaded LoRA is never saved under a chosen name.
                "lora_save_name": text(),
                "display_name": text(),
                "description": text(multiline=True),
                "lora_url": text(),
                "bearer_token": text(),
            },
        }

    RETURN_TYPES = (WILDCARD,)
    RETURN_NAMES = ("path",)
    FUNCTION = "run"
    CATEGORY = "🔗ComfyDeploy"

    def run(self, input_id, default_lora_name=None, lora_save_name=None, display_name=None,
            description=None, lora_url=None, bearer_token=None):
        import requests

        value = lora_url
        if value is None or value == "":
            # ComfyUI normalises a name before its lookup, so "./monet/x" loads "monet/x".
            if isinstance(default_lora_name, str) and any(n.startswith(PRIVATE_PREFIXES) for n in (
                    default_lora_name, os.path.relpath(os.path.join("/", default_lora_name), "/"))):
                raise LoraRefError("path", "a Monet LoRA loads only from a signed reference, never a default")
            print(f"Ext Lora loading: {default_lora_name}")
            return (default_lora_name,)

        if not isinstance(value, str):
            raise LoraRefError("not_a_ref")

        if value.startswith(REF_PREFIX):
            claims = parse_lora_ref(value)
            rel = claims["p"]
            if _volume_hit(rel, claims["n"], claims["h"]):
                volume_copy = os.path.realpath(os.path.join(VOLUME_LORA_ROOT, rel))
                if not _resolves_to(rel, volume_copy):
                    raise LoraRefError("path", f"{rel} does not resolve to the checked volume copy")
                print(f"Ext Lora: volume hit {rel}")
                return (rel,)
            if os.path.lexists(os.path.join(VOLUME_LORA_ROOT, rel)):
                print(f"lora volume copy failed its check: {rel}")
                name = SHADOW_PREFIX + rel
            else:
                print(f"Ext Lora: volume miss {rel}")
                name = rel
            _serve_from_cache(requests, name, claims)
            cached = os.path.realpath(os.path.join(RUN_CACHE_ROOT, name))
            if not _resolves_to(name, cached):
                raise LoraRefError("path", f"{name} does not resolve to the checked run cache copy")
            _mark_returned(name)
            print(f"Ext Lora loaded: {name}")
            return (name,)

        if value.startswith("https://") or value.startswith("http://"):
            if os.environ.get(ALLOW_URL_ENV) not in (None, "1"):
                raise LoraRefError("not_a_ref", f"plain URLs are off ({ALLOW_URL_ENV})")
            if lora_save_name:
                raise LoraRefError("not_a_ref", "lora_save_name must be empty; a URL LoRA is never saved by name")
            name = f"{URL_PREFIX}{uuid.uuid4().hex}.safetensors"
            destination = _cache_destination(name)
            _evict_run_cache(0, destination)
            # Never log the URL: a presigned link carries its signature in
            # the query string, so it is a credential until it expires.
            print(f"Ext Lora downloading to {destination}")
            headers = {"User-Agent": "Mozilla/5.0"}
            if bearer_token:
                headers["Authorization"] = f"Bearer {bearer_token}"
                print("using bearer token")
            _download_lora(requests, value, headers, destination)
            if not _resolves_to(name, os.path.realpath(destination)):
                raise LoraRefError("path", f"{name} does not resolve to the downloaded file")
            _mark_returned(name)
            print(f"Ext Lora loaded: {name}")
            return (name,)

        raise LoraRefError("not_a_ref")


NODE_CLASS_MAPPINGS = {"ComfyUIDeployExternalLora": ComfyUIDeployExternalLora}
NODE_DISPLAY_NAME_MAPPINGS = {"ComfyUIDeployExternalLora": "External Lora (ComfyUI Deploy)"}
