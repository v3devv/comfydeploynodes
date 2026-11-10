import os
import uuid

import folder_paths

# A LoRA smaller than this is an error page, not weights. The smallest real
# LoRAs are hundreds of KiB; the error bodies this catches are under a few KiB.
MIN_LORA_BYTES = 16 * 1024
# Larger than any LoRA we train or load; a body past this is not a LoRA either.
MAX_LORA_BYTES = 8 * 1024 * 1024 * 1024
# (connect, read between chunks) in seconds, so a stalled link fails the run
# instead of holding the GPU until the run timeout.
DOWNLOAD_TIMEOUT = (30, 120)
CHUNK_BYTES = 1024 * 1024


class LoraDownloadError(RuntimeError):
    """The link did not serve a usable LoRA. The message never carries the URL."""


def _check_safetensors_header(first_bytes, total):
    # safetensors: 8 byte little endian header length, then a JSON object.
    if len(first_bytes) < 9:
        return False
    header_len = int.from_bytes(first_bytes[:8], "little")
    return 0 < header_len <= total - 8 and first_bytes[8:9] == b"{"


def _download_lora(requests, url, headers, destination_path):
    """Download url to destination_path, or raise LoraDownloadError.

    The body goes to a hidden temp name in the same folder and is renamed into
    place only once it is whole and plausible, so a failed or partial download
    never sits under the LoRA's name (where the next run would reuse it).
    """
    folder = os.path.dirname(destination_path)
    name = os.path.basename(destination_path)
    temp_path = os.path.join(folder, f".{name}.{uuid.uuid4().hex}.part")
    response = requests.get(
        url,
        headers=headers,
        allow_redirects=True,
        stream=True,
        timeout=DOWNLOAD_TIMEOUT,
    )
    try:
        status = response.status_code
        if not 200 <= status < 300:
            raise LoraDownloadError(
                f"External LoRA download for {name} failed with HTTP {status}. "
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
            raise LoraDownloadError(
                f"External LoRA download for {name} answered HTTP {status} with "
                f"{declared} bytes, which is not a plausible LoRA size."
            )

        received = 0
        head = b""
        with open(temp_path, "wb") as out_file:
            for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                if not chunk:
                    continue
                if len(head) < 9:
                    head += chunk[: 9 - len(head)]
                received += len(chunk)
                if received > MAX_LORA_BYTES:
                    raise LoraDownloadError(
                        f"External LoRA download for {name} passed {MAX_LORA_BYTES} bytes, "
                        "which is not a plausible LoRA size."
                    )
                out_file.write(chunk)
            out_file.flush()
            os.fsync(out_file.fileno())

        if declared is not None and received != declared:
            raise LoraDownloadError(
                f"External LoRA download for {name} ended early: received "
                f"{received} of {declared} bytes."
            )
        if received < MIN_LORA_BYTES:
            raise LoraDownloadError(
                f"External LoRA download for {name} answered HTTP {status} with only "
                f"{received} bytes, which is not a plausible LoRA size."
            )
        if name.endswith(".safetensors") and not _check_safetensors_header(head, received):
            raise LoraDownloadError(
                f"External LoRA download for {name} answered HTTP {status} but the "
                "body is not a safetensors file."
            )
        os.replace(temp_path, destination_path)
    except LoraDownloadError:
        raise
    except Exception as exc:
        raise LoraDownloadError(
            f"External LoRA download for {name} failed: {type(exc).__name__}."
        ) from None
    finally:
        response.close()
        if os.path.exists(temp_path):
            os.remove(temp_path)


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


WILDCARD = AnyType("*")


class ComfyUIDeployExternalLora:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_id": (
                    "STRING",
                    {"multiline": False, "default": "input_lora"},
                ),
            },
            "optional": {
                "default_lora_name": (folder_paths.get_filename_list("loras"),),
                "lora_save_name": (  # if `default_lora_name` is a link to download a file, we will attempt to save it with this name
                    "STRING",
                    {"multiline": False, "default": ""},
                ),
                "display_name": (
                    "STRING",
                    {"multiline": False, "default": ""},
                ),
                "description": (
                    "STRING",
                    {"multiline": True, "default": ""},
                ),
                "lora_url": (
                    "STRING",
                    {"multiline": False, "default": ""},
                ),
                "bearer_token": (
                    "STRING",
                    {"multiline": False, "default": ""},
                ),
            },
        }

    RETURN_TYPES = (WILDCARD,)
    RETURN_NAMES = ("path",)
    FUNCTION = "run"
    CATEGORY = "🔗ComfyDeploy"

    def run(
        self,
        input_id,
        default_lora_name=None,
        lora_save_name=None,
        display_name=None,
        description=None,
        lora_url=None,
        bearer_token=None,
    ):
        import requests

        if lora_url:
            if lora_url.startswith("http"):
                if lora_save_name:
                    existing_loras = folder_paths.get_filename_list("loras")
                    # Check if lora_save_name exists in the list
                    if lora_save_name in existing_loras:
                        print(f"using lora: {lora_save_name}")
                        return (lora_save_name,)
                else:
                    lora_save_name = str(uuid.uuid4()) + ".safetensors"
                destination_path = os.path.join(
                    folder_paths.folder_names_and_paths["loras"][0][0], lora_save_name
                )
                # Never log lora_url: a presigned link carries its signature in
                # the query string, so it is a credential until it expires.
                print(f"Ext Lora downloading to {destination_path}")
                headers = {"User-Agent": "Mozilla/5.0"}
                if bearer_token:
                    headers["Authorization"] = f"Bearer {bearer_token}"
                    print("using bearer token")
                _download_lora(requests, lora_url, headers, destination_path)
                print(f"Ext Lora loaded: {lora_save_name}")
                return (lora_save_name,)
            else:
                print(f"Ext Lora loading: {lora_url}")
                return (lora_url,)
        else:
            print(f"Ext Lora loading: {default_lora_name}")
            return (default_lora_name,)


NODE_CLASS_MAPPINGS = {"ComfyUIDeployExternalLora": ComfyUIDeployExternalLora}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyUIDeployExternalLora": "External Lora (ComfyUI Deploy)"
}
