"""What custom_routes sends back to Deploy carries no credential (NV2).

Run from the fork root: python3 -m unittest discover -s tests -v

On an External Lora run the node's inputs are credentials: `lora_url` and
`input_id` hold the signed reference (or a presigned link) and `bearer_token`
a token. ComfyUI's `execution_error` repeats a failing node's inputs, its
validation errors repeat values, and the run log repeats whatever was printed.
Every one of those strings reaches Deploy, which stores it. They are scrubbed
here, at the source: a reference becomes `mlr1.[redacted]`, a signed query
string `?[redacted]`, a bearer token `[redacted]`.
"""

import asyncio
import contextlib
import os
import types
import unittest
from unittest import mock

from lora_harness import REF_SOURCE
from routes_harness import BEARER, PRESIGNED_PUT, REF, RUN_CREDENTIAL, SAS_URL, RoutesTestCase


def lora_error_event(prompt_id):
    """ComfyUI's `execution_error` for an External Lora node, its inputs swapped in."""
    return {
        "prompt_id": prompt_id,
        "node_id": "4",
        "node_type": "ComfyUIDeployExternalLora",
        "executed": ["1"],
        "exception_message": f"fetch failed: GET {REF_SOURCE} answered 403",
        "exception_type": "RuntimeError",
        "traceback": ['  File "external_lora.py", line 9, in run\n', f"    fetch({REF!r})\n"],
        "current_inputs": {
            "input_id": [REF],
            "lora_url": [REF],
            "bearer_token": [BEARER],
            "default_lora_name": [None],
            "lora_save_name": [""],
            "display_name": [f"see {SAS_URL}"],
            "description": [f"Authorization: Bearer {BEARER}"],
        },
        "current_outputs": [],
    }


class ExecutionErrorIsScrubbedAtTheSource(RoutesTestCase):
    def test_send_sync_scrubs_the_event_comfyui_keeps_in_its_history(self):
        # ComfyUI appends this same dict to the prompt's history before it
        # sends it, so it is scrubbed in place.
        data = lora_error_event(self.prompt_id)
        self.routes.swizzle_send_sync(self.server, "execution_error", data, "comfy_deploy_instance")
        event, seen, _sent = self.server.native[-1]
        self.assertEqual(event, "execution_error")
        self.assert_no_secret(seen, "ComfyUI's websocket")
        self.assert_no_secret(data, "ComfyUI's history copy of the event")
        inputs = data["current_inputs"]
        self.assertEqual(inputs["lora_url"], ["mlr1.[redacted]"])
        self.assertEqual(inputs["input_id"], ["mlr1.[redacted]"])
        self.assertEqual(inputs["bearer_token"], ["[redacted]"])
        self.assertEqual(inputs["default_lora_name"], [None])
        self.assertEqual(inputs["lora_save_name"], [""])
        self.assertEqual(inputs["display_name"], ["see https://acct.blob.example/c/a.safetensors?[redacted]"])
        self.assertEqual(data["node_type"], "ComfyUIDeployExternalLora")
        self.assertEqual(data["exception_message"],
                         "fetch failed: GET https://acct.r2.cloudflarestorage.com/monet-private/ws/loras/p/"
                         "v.safetensors?[redacted] answered 403")
        self.assertIn("    fetch('mlr1.[redacted]')\n", data["traceback"])

    def test_every_channel_to_deploy_gets_the_scrubbed_event(self):
        async def scenario():
            self.routes.comfy_message_queues[self.prompt_id] = asyncio.Queue()
            await self.server.send_json("execution_error", lora_error_event(self.prompt_id), "comfy_deploy_instance")
            for _ in range(5):
                await asyncio.sleep(0)
            return self.routes.comfy_message_queues[self.prompt_id].get_nowait()

        queued = asyncio.run(scenario())
        self.assert_no_secret(queued, "the streaming response")
        self.assert_no_secret(self.ws, "the ComfyDeploy websocket")
        self.assert_no_secret([seen for _e, seen, _d in self.server.native], "ComfyUI's websocket")
        bodies = [body for _m, _u, body in self.posts]
        self.assert_no_secret(bodies, "a body posted to Deploy")
        outputs = [b["output_data"] for b in bodies if "output_data" in b]
        self.assertEqual(len(outputs), 1, "the error is stored once, as the run's output")
        self.assertEqual(outputs[0]["current_inputs"]["bearer_token"], ["[redacted]"])
        self.assertEqual(outputs[0]["node_id"], "4")
        self.assertIn("failed", [b.get("status") for b in bodies])
        events = [b["ws_event"] for b in bodies if "ws_event" in b]
        self.assertEqual([e["event"] for e in events], ["execution_error"])
        self.assertEqual(queued["event"], "execution_error")

    def test_other_events_pass_through_unchanged(self):
        data = {"prompt_id": self.prompt_id, "node": "4", "output": {"text": [SAS_URL]}}
        self.routes.swizzle_send_sync(self.server, "executed", data, "comfy_deploy_instance")
        self.assertEqual(data["output"]["text"], [SAS_URL], "an output is the run's product, not an error")


class RunLogsAreScrubbed(RoutesTestCase):
    def write(self, name, text):
        path = os.path.join(self.tmp, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def test_the_run_log_sent_on_failure_is_scrubbed(self):
        self.routes.comfyui_file_path = self.write("comfyui.log", "".join([
            "got prompt\n",
            f"queue {REF}\n",
            f"GET {REF_SOURCE} 403\n",
            f"Authorization: Bearer {BEARER}\n",
            f"proxy {SAS_URL}\n",
            "Prompt executed in 1.20 seconds\n",
        ]))
        self.routes.cd_enable_run_log = True
        self.routes.last_read_line_number = 0
        self.run_async(self.routes.update_run(self.prompt_id, self.routes.Status.FAILED))
        logs = [b["log_data"][0]["logs"] for _m, _u, b in self.posts if "log_data" in b]
        self.assertEqual(len(logs), 1)
        self.assert_no_secret(logs[0], "the run log")
        self.assertIn("got prompt\nqueue mlr1.[redacted]\n", logs[0])
        self.assertIn("Authorization: Bearer [redacted]\n", logs[0])
        self.assertTrue(logs[0].endswith("Prompt executed in 1.20 seconds\n"))

    def test_logs_streamed_to_the_websocket_are_scrubbed(self):
        self.run_async(self.routes.send_logs_to_websocket(f"loading {REF}\n"))
        self.routes.log_file_path = self.write("comfy-deploy.log", f"GET {PRESIGNED_PUT}\nready\n")
        self.run_async(self.routes.send_first_time_log("sid-1"))
        self.assertEqual([e for e, _d in self.ws], ["LOGS", "LOGS"])
        self.assert_no_secret(self.ws, "the log websocket")
        self.assertEqual(self.ws[0][1], "loading mlr1.[redacted]\n")
        self.assertIn("ready\n", self.ws[1][1])


class RunRouteErrorsAreScrubbed(RoutesTestCase):
    def request(self):
        body = {
            "prompt_id": self.prompt_id,
            "workflow_api_raw": {
                "4": {"class_type": "ComfyUIDeployExternalLora",
                      "inputs": {"input_id": "input_lora", "lora_url": "", "bearer_token": BEARER}},
            },
            "inputs": {"input_lora": REF},
            "status_endpoint": "https://deploy.example/api/update-run",
            "file_upload_endpoint": None,
            "workflow": {},
        }

        async def read_json():
            return body

        return types.SimpleNamespace(json=read_json, headers={"Authorization": "Bearer " + RUN_CREDENTIAL})

    def test_a_queue_failure_answer_and_stored_error_are_scrubbed(self):
        self.routes.post_prompt = mock.AsyncMock(side_effect=RuntimeError(f"could not queue {REF} from {SAS_URL}"))
        with self.captured_logs() as logs:
            response = self.run_async(self.routes.comfy_deploy_run(self.request()))
        self.assertEqual(response.status, 500)
        self.assert_no_secret(response.reason, "the HTTP reason")
        self.assertTrue(response.reason.startswith("RuntimeError: could not queue mlr1.[redacted] from "))
        self.assert_no_secret([b for _m, _u, b in self.posts], "a body posted to Deploy")
        errors = [b["output_data"]["error"] for _m, _u, b in self.posts if "output_data" in b]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["error_type"], "RuntimeError")
        self.assertIn("could not queue mlr1.[redacted]", errors[0]["stack_trace"])
        self.assert_no_secret("\n".join(logs.output), "the routes log")

    def test_validation_errors_returned_and_stored_are_scrubbed(self):
        res = {
            "error": {"type": "prompt_outputs_failed_validation",
                      "message": "Prompt outputs failed validation", "details": f"bad {REF}"},
            "node_errors": {"4": {
                "errors": [{"type": "value_not_in_list", "message": "Value not in list",
                            "details": f"default_lora_name: '{SAS_URL}' not in []",
                            "extra_info": {"input_name": "default_lora_name", "received_value": SAS_URL}}],
                "dependent_outputs": ["9"],
                "class_type": "ComfyUIDeployExternalLora",
            }},
        }
        self.routes.post_prompt = mock.AsyncMock(return_value=res)
        response = self.run_async(self.routes.comfy_deploy_run(self.request()))
        self.assertEqual(response.status, 400)
        self.assert_no_secret(response.data, "the run answer")
        self.assertEqual(response.data["error"]["message"], "Prompt outputs failed validation")
        self.assertEqual(response.data["node_errors"]["4"]["dependent_outputs"], ["9"])
        self.assert_no_secret([b for _m, _u, b in self.posts], "a body posted to Deploy")
        self.assertEqual(len([b for _m, _u, b in self.posts if "output_data" in b]), 1)

    def test_the_streaming_route_answer_is_scrubbed(self):
        self.routes.post_prompt = mock.AsyncMock(
            return_value={"prompt_id": self.prompt_id, "number": 3, "node_errors": {"4": {"errors": [REF]}}})
        res = self.run_async(self.routes.stream_prompt(self.request_body(), RUN_CREDENTIAL))
        self.assert_no_secret(res, "the streaming answer")
        self.assertEqual(res["number"], 3)
        self.assertEqual(res["node_errors"]["4"]["errors"], ["mlr1.[redacted]"])

    def test_the_streaming_route_stored_error_is_scrubbed(self):
        self.routes.post_prompt = mock.AsyncMock(side_effect=RuntimeError(f"could not queue {REF}"))
        # Upstream then reads an answer it never got; only what was posted is checked.
        with self.captured_logs(), contextlib.suppress(Exception):
            self.run_async(self.routes.stream_prompt(self.request_body(), RUN_CREDENTIAL))
        errors = [b["output_data"]["error"] for _m, _u, b in self.posts if "output_data" in b]
        self.assertEqual(len(errors), 1)
        self.assert_no_secret(errors, "the stored streaming error")

    def request_body(self):
        return self.run_async(self.request().json())

    def test_a_successful_answer_is_unchanged(self):
        answer = {"prompt_id": self.prompt_id, "number": 7, "node_errors": {}}
        self.routes.post_prompt = mock.AsyncMock(return_value=dict(answer))
        response = self.run_async(self.routes.comfy_deploy_run(self.request()))
        self.assertEqual(response.status, 200)
        self.assertEqual(response.data, answer)
        self.assertEqual(self.posts, [])


class TheRoutesLoggerScrubs(RoutesTestCase):
    def test_logged_workflows_inputs_and_errors_carry_no_credential(self):
        workflow = {"4": {"class_type": "ComfyUIDeployExternalLora",
                          "inputs": {"input_id": REF, "lora_url": REF, "bearer_token": BEARER}}}
        with self.captured_logs() as logs:
            self.routes.logger.info(workflow)
            self.routes.logger.info("getting inputs", {"input_lora": REF})
            self.routes.logger.info(f"Got input: ${ {'input_lora': SAS_URL} }")
            self.routes.log("error", "Streaming error", error=RuntimeError(f"GET {REF_SOURCE}"))
            self.routes.logger.error(f"Request PUT : {PRESIGNED_PUT} failed after 5 attempts")
            self.routes.logger.info("Upload queue system initialized with max_concurrent=%d", 3)
        text = "\n".join(logs.output)
        self.assert_no_secret(text, "the routes log")
        self.assertIn("'bearer_token': '[redacted]'", text)
        # As logging does, a lone mapping argument with no placeholder adds nothing.
        self.assertIn("INFO:comfy-deploy:getting inputs", logs.output)
        self.assertIn("Request PUT : https://acct.r2.cloudflarestorage.com/outputs/run/a.png?[redacted] failed", text)
        self.assertIn("max_concurrent=3", text)
        self.assertEqual(len(logs.records), 6)
        self.assertEqual(logs.records[0].funcName, "test_logged_workflows_inputs_and_errors_carry_no_credential",
                         "a record still names the line that logged it")


class UploadFailuresAreScrubbed(RoutesTestCase):
    def test_a_failed_upload_never_reports_the_presigned_link(self):
        with open(os.path.join(self.tmp, "a.png"), "wb") as f:
            f.write(b"png")

        class _Ctx:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def read(self):
                return b"png"

        self.routes.aiofiles.open = lambda _path, _mode: _Ctx()
        self.routes.aiohttp.ClientSession = _Ctx

        async def failing_upload(_session, url, _headers, _data, **_kw):
            raise RuntimeError(f"PUT {url} answered 403")

        self.routes.upload_with_retry = failing_upload
        with self.captured_logs() as logs, self.assertRaises(RuntimeError):
            self.run_async(self.routes.upload_queue._process_upload(
                self.prompt_id, {"filename": "a.png", "type": "output"}, "9"))
        failed = [d for e, d in self.ws if e == "upload_failed"]
        self.assertEqual(len(failed), 1)
        self.assert_no_secret(failed[0], "the upload_failed event")
        self.assertEqual(failed[0]["filename"], "a.png")
        self.assertEqual(failed[0]["error"],
                         "PUT https://acct.r2.cloudflarestorage.com/outputs/run/a.png?[redacted] answered 403")
        self.assert_no_secret("\n".join(logs.output), "the routes log")


if __name__ == "__main__":
    unittest.main()
