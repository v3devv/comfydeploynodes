"""custom_routes.py's patches on ComfyUI, exercised through the real module.

Run from the fork root, with nothing but the standard library:

    python3 -m unittest discover -s tests -v

custom_routes.py cannot be imported without ComfyUI, torch, aiohttp, PIL and
pydantic, so each case runs it in a CHILD process in which those are stubbed,
`server.PromptServer` is a fake, and `execution.execute` is a fake with ComfyUI
0.37.0's twelve parameters. The code under test, custom_routes.py and
comfydeploy_execute_swizzle.py, is the real file. The child process keeps the
stubs out of every other test.

The fake PromptServer's send_sync / send_json / send_bytes carry 0.37.0's
parameter lists plus ONE appended parameter. The execute wrapper died on
exactly that kind of change (ComfyUI 0.36.0 appended `asset_manager`), and these
three wrappers sit on the same paths: send_sync runs on the prompt worker
thread, send_json and send_bytes in ComfyUI's single publish loop.
"""

import json
import os
import subprocess
import sys
import unittest

FORK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MARK = "CUSTOM-ROUTES-RESULT "


def _child(fork_root, mode):
    import asyncio
    import importlib.abc
    import importlib.machinery
    import logging
    import types
    from unittest import mock

    stubbed = {
        "torch", "comfy", "comfy_aimdo", "latent_preview", "nodes",
        "comfy_execution", "comfy_api", "app", "folder_paths", "model_management",
        "aiohttp", "aiofiles", "PIL", "psutil", "requests", "tqdm", "logfire",
    }

    class StubModule(types.ModuleType):
        def __init__(self, name):
            super().__init__(name)
            self.__path__ = []
            self._attrs = {}

        def __getattr__(self, attr):
            if attr.startswith("__") and attr.endswith("__"):
                raise AttributeError(attr)
            if attr not in self._attrs:
                self._attrs[attr] = mock.MagicMock(name=f"{self.__name__}.{attr}")
            return self._attrs[attr]

    class StubLoader(importlib.abc.Loader):
        def create_module(self, spec):
            return StubModule(spec.name)

        def exec_module(self, module):
            pass

    class StubFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in stubbed:
                return importlib.machinery.ModuleSpec(
                    name, StubLoader(), is_package=True
                )
            return None

    sys.meta_path.insert(0, StubFinder())
    pydantic = types.ModuleType("pydantic")

    class BaseModel:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    pydantic.BaseModel = BaseModel
    sys.modules["pydantic"] = pydantic
    import aiohttp

    aiohttp.ClientError = type("ClientError", (Exception,), {})

    received = []
    sent_data = []
    fail_original = {"on": False}

    class Routes:
        def _deco(self, *a, **k):
            return lambda f: f

        get = post = put = delete = patch = _deco

    class PromptServer:
        instance = None

        def __init__(self):
            self.client_id = "client-1"
            self.last_node_id = "4"
            self.last_prompt_id = None
            self.routes = Routes()
            self.app = types.SimpleNamespace(on_startup=[], router=mock.MagicMock())
            self.prompt_queue = mock.MagicMock()
            self.number = 0

        async def send_bytes(self, event, data, sid=None, broadcast=False):
            received.append(("send_bytes", event, sid, broadcast))

        async def send_json(self, event, data, sid=None, broadcast=False):
            received.append(("send_json", event, sid, broadcast))
            sent_data.append(data)
            if fail_original["on"]:
                raise RuntimeError("socket said no")

        def send_sync(self, event, data, sid=None, broadcast=False):
            received.append(("send_sync", event, sid, broadcast))

    PromptServer.instance = PromptServer()
    server_mod = types.ModuleType("server")
    server_mod.PromptServer = PromptServer
    server_mod.BinaryEventTypes = mock.MagicMock()
    sys.modules["server"] = server_mod

    execute_calls = []
    execution = types.ModuleType("execution")

    async def execute(server, dynprompt, caches, current_item, extra_data, executed,
                      prompt_id, execution_list, pending_subgraph_results,
                      pending_async_nodes, ui_outputs, asset_manager):
        execute_calls.append(asset_manager)
        return "SUCCESS"

    execution.execute = execute
    sys.modules["execution"] = execution

    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            text = record.getMessage()
            if record.exc_info:
                text += " | " + repr(record.exc_info[1])
            records.append((record.levelname, text))

    logging.getLogger("comfy-deploy").addHandler(Capture())

    if mode == "swizzle-import-broken":
        sys.modules["comfydeploy_execute_swizzle"] = None  # import -> ImportError

    sys.path.insert(0, fork_root)
    import custom_routes

    results = {}
    if not custom_routes.__file__.startswith(fork_root):
        results["imported the fork's custom_routes"] = ["fail", custom_routes.__file__]

    if mode == "swizzle-import-broken":
        errors = [t for lvl, t in records if lvl == "ERROR" and "execute timing patch" in t]
        results["a failed install is logged at ERROR"] = (
            ["ok", errors[0]] if errors else ["fail", repr(records)]
        )
        results["ComfyUI's execute is left in place"] = (
            ["ok", ""] if execution.execute is execute else ["fail", "replaced"]
        )
        print(_MARK + json.dumps(results), flush=True)
        return

    srv = PromptServer.instance

    if mode == "publish-guard":
        _publish_guard_cases(custom_routes, srv, received, sent_data,
                             fail_original, records, results)
        print(_MARK + json.dumps(results), flush=True)
        return

    if mode == "node-failure":
        _node_failure_cases(custom_routes, srv, records, results)
        print(_MARK + json.dumps(results), flush=True)
        return

    def case(label, fn, expect):
        received.clear()
        try:
            fn()
        except Exception as ex:
            results[label] = ["fail", f"raised {type(ex).__name__}: {ex}"]
            return
        if received == [expect]:
            results[label] = ["ok", repr(received[0])]
        else:
            results[label] = ["fail", f"original received {received!r}"]

    case("send_sync forwards an appended keyword",
         lambda: srv.send_sync("status", {}, "sid-1", broadcast=True),
         ("send_sync", "status", "sid-1", True))
    case("send_sync forwards an appended positional",
         lambda: srv.send_sync("status", {}, "sid-1", True),
         ("send_sync", "status", "sid-1", True))
    case("send_json forwards an appended keyword",
         lambda: asyncio.run(srv.send_json("status", {"prompt_id": None}, "sid-1", broadcast=True)),
         ("send_json", "status", "sid-1", True))
    case("send_bytes forwards an appended keyword",
         lambda: asyncio.run(srv.send_bytes(99, b"\0\0\0\1xx", "sid-1", broadcast=True)),
         ("send_bytes", 99, "sid-1", True))
    case("send_sync keeps ComfyUI 0.37.0's call shape",
         lambda: srv.send_sync("status", {}, "sid-1"),
         ("send_sync", "status", "sid-1", False))
    case("send_json keeps ComfyUI 0.37.0's call shape",
         lambda: asyncio.run(srv.send_json("status", {"prompt_id": None}, sid="sid-1")),
         ("send_json", "status", "sid-1", False))
    case("send_bytes keeps ComfyUI 0.37.0's call shape",
         lambda: asyncio.run(srv.send_bytes(99, b"\0\0\0\1xx", sid="sid-1")),
         ("send_bytes", 99, "sid-1", False))

    # The execute wrapper is installed by custom_routes and times a node end to
    # end: execution_start opens the run, executing starts the node's clock,
    # the 12-argument execute call closes it.
    srv.send_sync("execution_start", {"prompt_id": "p-1"}, "sid-1")
    srv.send_sync("executing", {"node": "5", "prompt_id": "p-1"}, "sid-1")
    dynprompt = types.SimpleNamespace(get_node=lambda uid: {"class_type": "SaveImage"})
    asset_manager = object()
    try:
        out = asyncio.run(execution.execute(
            srv, dynprompt, None, "5", {}, set(), "p-1", None, {}, {}, {}, asset_manager
        ))
        timed = custom_routes.NODE_EXECUTION_TIMES.get("5", {}).get("class_type")
        ok = (out == "SUCCESS" and execute_calls == [asset_manager]
              and execution.execute is not execute and timed == "SaveImage")
        results["custom_routes installs the execute wrapper and it times a 12-argument call"] = [
            "ok" if ok else "fail",
            f"returned={out!r} reached_original={len(execute_calls)} timed={timed!r}",
        ]
    except Exception as ex:
        results["custom_routes installs the execute wrapper and it times a 12-argument call"] = [
            "fail", f"raised {type(ex).__name__}: {ex}"
        ]

    # Timing that raises must not reach ComfyUI's caller: execute() would turn
    # it into a failed node. It must be logged, not swallowed.
    def boom():
        raise RuntimeError("cuda said no")

    real_reset = custom_routes.reset_peak_memory_record
    custom_routes.reset_peak_memory_record = boom
    records.clear()
    label = "send_sync timing that raises is logged and the event still goes out"
    case(label,
         lambda: srv.send_sync("executing", {"node": "6", "prompt_id": "p-1"}, "sid-1"),
         ("send_sync", "executing", "sid-1", False))
    custom_routes.reset_peak_memory_record = real_reset
    if results[label][0] == "ok" and not any("cuda said no" in t for _, t in records):
        results[label] = ["fail", "swallowed without a log line"]

    print(_MARK + json.dumps(results), flush=True)


def _publish_guard_cases(custom_routes, srv, received, sent_data, fail_original,
                         records, results):
    """send_json runs inside ComfyUI's single publish loop.

    That loop sits in the `asyncio.gather` in ComfyUI's main.py, so anything
    raised out of send_json ends the WHOLE server ("Exiting the application").
    A third-party node calling `PromptServer.instance.send_sync("x", "a string")`
    did exactly that on 0.37.0: the fork read `data.get("prompt_id")` off a str.
    """
    import asyncio
    import types

    def send(event, data):
        received.clear()
        sent_data.clear()
        asyncio.run(srv.send_json(event, data, "sid-1"))

    # 1. Payloads that are not a dict go to ComfyUI's send untouched, and are
    #    not treated as a failure: there is simply no prompt id to track.
    for what, payload in (("a string", "a plain string payload"),
                          ("a list", ["a", 1]),
                          ("None", None)):
        label = f"send_json hands {what} payload to ComfyUI's send unchanged"
        records.clear()
        # Failures are logged once per place; forget the earlier cases' so a
        # payload that fails where the one before it did still shows up here.
        getattr(custom_routes, "_send_json_failures_logged", set()).clear()
        try:
            send("reviewer-str", payload)
        except Exception as ex:
            results[label] = ["fail", f"raised {type(ex).__name__}: {ex}"]
            continue
        warned = [t for lvl, t in records if lvl in ("WARNING", "ERROR", "CRITICAL")]
        ok = (received == [("send_json", "reviewer-str", "sid-1", False)]
              and len(sent_data) == 1 and sent_data[0] is payload and not warned)
        results[label] = ["ok" if ok else "fail",
                          f"received={received!r} data={sent_data!r} warned={warned!r}"]

    # 2. The fork's own handling failing is logged, ONCE per distinct failure,
    #    and ComfyUI's send still goes out with the original arguments.
    def boom(*a, **k):
        raise RuntimeError("handler said no")

    real_done = custom_routes.mark_prompt_done
    custom_routes.mark_prompt_done = boom
    records.clear()
    label = "a failing handler is logged once and ComfyUI's send still runs every time"
    outcomes = []
    event_data = {"node": None, "prompt_id": "p-9"}
    for _ in range(3):
        try:
            send("executing", event_data)
            outcomes.append(received == [("send_json", "executing", "sid-1", False)]
                            and sent_data == [event_data] and sent_data[0] is event_data)
        except Exception as ex:
            outcomes.append(f"raised {type(ex).__name__}: {ex}")
    custom_routes.mark_prompt_done = real_done
    tracebacks = [t for lvl, t in records
                  if lvl == "WARNING" and "RuntimeError('handler said no')" in t]
    results[label] = [
        "ok" if outcomes == [True, True, True] and len(tracebacks) == 1 else "fail",
        f"outcomes={outcomes!r} tracebacks={tracebacks!r}",
    ]

    # 3. ComfyUI's own send raising is ComfyUI's business: it propagates.
    label = "an exception from ComfyUI's own send propagates"
    fail_original["on"] = True
    seen = []
    for payload in ({"prompt_id": None}, "a plain string payload"):
        try:
            send("status", payload)
            seen.append("nothing raised")
        except RuntimeError as ex:
            seen.append(str(ex))
        except Exception as ex:
            seen.append(f"raised {type(ex).__name__}: {ex}")
    fail_original["on"] = False
    results[label] = [
        "ok" if seen == ["socket said no", "socket said no"] else "fail", repr(seen)
    ]

    # 4. `executed` for a node that a custom node's graph EXPANSION created:
    #    ComfyUI 0.37.0 names it "<parent>.<call>.<graph>.<id>", which is not a
    #    key of the submitted workflow. Its output must still be uploaded.
    uploads = []

    async def fake_upload(prompt_id, data, node_id=None, node_meta=None,
                          gpu_event_id=None):
        uploads.append([prompt_id, node_id, node_meta])

    real_upload = custom_routes.update_run_with_output
    custom_routes.update_run_with_output = fake_upload
    custom_routes.prompt_metadata["p-7"] = types.SimpleNamespace(
        workflow_api={
            "3": {"class_type": "SaveImage", "inputs": {}},
            "4": {"class_type": "PreviewImage", "inputs": {}},
            "5": {"class_type": "ExpandsToSave", "inputs": {}},
        },
        status_endpoint=None, token=None, gpu_event_id=None, is_realtime=False,
        start_time=None, progress=set(), last_updated_node=None, status=None,
    )
    out = {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]}
    for label, node, expect in (
        ("executed for an expanded node id uploads its output",
         "5.0.0.1", [["p-7", "5.0.0.1", {"node_id": "5.0.0.1", "node_class": "ExpandsToSave"}]]),
        ("executed for a nested expanded node id uploads its output",
         "5.0.0.1.2.0.3", [["p-7", "5.0.0.1.2.0.3",
                            {"node_id": "5.0.0.1.2.0.3", "node_class": "ExpandsToSave"}]]),
        ("executed for a workflow node still uploads its output",
         "3", [["p-7", "3", {"node_id": "3", "node_class": "SaveImage"}]]),
        ("executed for a PreviewImage still uploads nothing", "4", []),
    ):
        uploads.clear()
        records.clear()
        payload = {"node": node, "display_node": node.split(".")[0], "output": out,
                   "prompt_id": "p-7"}
        try:
            send("executed", payload)
        except Exception as ex:
            results[label] = ["fail", f"raised {type(ex).__name__}: {ex}"]
            continue
        warned = [t for lvl, t in records if lvl in ("WARNING", "ERROR", "CRITICAL")]
        ok = (uploads == expect and not warned
              and received == [("send_json", "executed", "sid-1", False)])
        results[label] = ["ok" if ok else "fail",
                          f"uploads={uploads!r} warned={warned!r} received={received!r}"]
    custom_routes.update_run_with_output = real_upload
    del custom_routes.prompt_metadata["p-7"]


def _node_failure_cases(custom_routes, srv, records, results):
    """A failed node's reason reaches the engine BEFORE the run is marked failed.

    The engine fires its terminal webhook on the `failed` status POST, strips
    `execution_error` outputs, and writes `live_status` only when `progress`
    comes with it. So the reason travels as `live_status` + `progress` in its
    own POST, awaited, and `status: failed` follows in a separate POST. Every
    POST goes through `async_request_with_retry`, which is replaced here by a
    recorder: the real update_run_with_output, update_run_live_status and
    update_run build the bodies.
    """
    import asyncio
    import types

    posts = []
    fail_when = {"pred": lambda body: False}

    async def fake_request(method, url, disable_timeout=False, token=None, **kw):
        body = kw.get("json")
        posts.append(body)
        if fail_when["pred"](body):
            raise RuntimeError("engine said no")

    custom_routes.async_request_with_retry = fake_request

    def run(pid, data, progress=(), fail=lambda body: False):
        custom_routes.prompt_metadata[pid] = types.SimpleNamespace(
            workflow_api={str(i): {"class_type": "N", "inputs": {}} for i in range(1, 5)},
            status_endpoint="http://engine.invalid/status", file_upload_endpoint=None,
            token="t", gpu_event_id=None, is_realtime=False, start_time=None,
            progress=set(progress), last_updated_node=None,
            status=custom_routes.Status.RUNNING,
        )
        posts.clear()
        records.clear()
        fail_when["pred"] = fail
        try:
            asyncio.run(srv.send_json("execution_error", dict(data, prompt_id=pid), "sid-1"))
            raised = None
        except Exception as ex:
            raised = f"{type(ex).__name__}: {ex}"
        fail_when["pred"] = lambda body: False
        live = [b for b in posts if "live_status" in b]
        failed = [b for b in posts if b.get("status") == "failed"]
        # The `ws_event` mirror of the event is a background task of its own
        # and may land anywhere; only the three run-state POSTs are ordered.
        order = ["live" if "live_status" in b else "failed" if b.get("status") == "failed"
                 else "output" if "output_data" in b else "other" for b in posts]
        order = [o for o in order if o != "other"]
        return raised, live, failed, order

    full = {
        "node_id": "5", "node_type": "KSampler", "executed": ["1", "2"],
        "exception_message": "CUDA out of memory.\nTried to allocate 2 GiB",
        "exception_type": "torch.OutOfMemoryError",
        "traceback": ["TRACEBACK-MARKER line 1\n", "TRACEBACK-MARKER line 2\n"],
        "current_inputs": {"image": ["INPUT-MARKER"]},
        "current_outputs": [],
    }

    # 1. The reason goes out first, with progress, and `failed` follows it.
    label = "a failed node posts its reason with progress, then failed, in that order"
    raised, live, failed, order = run("p-f1", full, progress=["1", "2"])
    want = ("ComfyUI node failed: Node 5 (KSampler): torch.OutOfMemoryError: "
            "CUDA out of memory. Tried to allocate 2 GiB")
    ok = (raised is None and order == ["output", "live", "failed"]
          and len(live) == 1 and live[0]["live_status"] == want
          and live[0]["progress"] == 0.5 and "status" not in live[0]
          and len(failed) == 1 and "live_status" not in failed[0])
    results[label] = ["ok" if ok else "fail",
                      f"raised={raised} order={order} live={live!r}"]

    # 2. The reason carries no traceback and no inputs.
    label = "the reason carries no traceback and no node inputs"
    text = live[0]["live_status"] if live else ""
    ok = bool(text) and "TRACEBACK-MARKER" not in text and "INPUT-MARKER" not in text
    results[label] = ["ok" if ok else "fail", repr(text)]

    # 3. The whole string is capped at 1000 characters.
    label = "the reason is capped at 1000 characters"
    raised, live, failed, order = run("p-f2", dict(full, exception_message="x" * 5000))
    text = live[0]["live_status"] if live else ""
    ok = (raised is None and len(text) == 1000 and text.startswith(
        "ComfyUI node failed: Node 5 (KSampler): torch.OutOfMemoryError: xxx")
          and order[-1] == "failed")
    results[label] = ["ok" if ok else "fail", f"len={len(text)} order={order}"]

    # 4. Missing fields render as `?`, never `None`; unknown progress is 0.
    label = "missing fields render as ? and unknown progress as 0"
    raised, live, failed, order = run("p-f3", {"node_type": None})
    text = live[0]["live_status"] if live else ""
    ok = (raised is None and text == "ComfyUI node failed: Node ? (?): ?: ?"
          and live[0]["progress"] == 0 and order[-1] == "failed")
    results[label] = ["ok" if ok else "fail", f"text={text!r} live={live!r} order={order}"]

    # 5. The reason's POST failing is logged and the run still ends failed.
    label = "a failing reason POST is logged and failed is still posted"
    raised, live, failed, order = run(
        "p-f4", full, fail=lambda body: "live_status" in body)
    warned = [t for lvl, t in records if lvl == "WARNING" and "engine said no" in t]
    ok = (raised is None and order == ["output", "live", "failed"] and len(warned) == 1)
    results[label] = ["ok" if ok else "fail",
                      f"raised={raised} order={order} warned={warned!r}"]

    # 6. So does the raw-record POST failing before it.
    label = "a failing output POST is logged and the reason and failed still post"
    raised, live, failed, order = run(
        "p-f5", full, fail=lambda body: "output_data" in body)
    warned = [t for lvl, t in records if lvl == "WARNING" and "engine said no" in t]
    ok = (raised is None and order == ["output", "live", "failed"] and len(warned) == 1)
    results[label] = ["ok" if ok else "fail",
                      f"raised={raised} order={order} warned={warned!r}"]


def _run_child(mode):
    env = {k: v for k, v in os.environ.items()
           if k not in ("CD_ENABLE_LOG", "USE_LOGFIRE", "PYTHONPATH",
                        "CD_ENABLE_RUN_LOG", "CD_BYPASS_UPLOAD", "MAX_RETRIES")}
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--child", FORK_ROOT, mode],
        capture_output=True, text=True, env=env, timeout=120,
    )
    # custom_routes prints at import and from its atexit handler, so the
    # result is found by its marker rather than by position.
    found = [l[len(_MARK):] for l in proc.stdout.splitlines() if l.startswith(_MARK)]
    try:
        return json.loads(found[-1])
    except (IndexError, ValueError):
        raise AssertionError(
            f"child process produced no result (exit {proc.returncode})\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )


class CustomRoutesWrappers(unittest.TestCase):
    EXPECTED = {
        "send_sync forwards an appended keyword",
        "send_sync forwards an appended positional",
        "send_json forwards an appended keyword",
        "send_bytes forwards an appended keyword",
        "send_sync keeps ComfyUI 0.37.0's call shape",
        "send_json keeps ComfyUI 0.37.0's call shape",
        "send_bytes keeps ComfyUI 0.37.0's call shape",
        "custom_routes installs the execute wrapper and it times a 12-argument call",
        "send_sync timing that raises is logged and the event still goes out",
    }

    def _check(self, results, expected):
        # Every case must have REPORTED. A child that stopped early must not
        # read as a pass.
        self.assertEqual(set(results), expected)
        for label, (status, detail) in sorted(results.items()):
            with self.subTest(case=label):
                self.assertEqual(status, "ok", detail)

    def test_the_wrappers_pass_through_what_comfyui_adds(self):
        self._check(_run_child("normal"), self.EXPECTED)

    def test_send_json_never_raises_its_own_failure_into_the_publish_loop(self):
        self._check(_run_child("publish-guard"), {
            "send_json hands a string payload to ComfyUI's send unchanged",
            "send_json hands a list payload to ComfyUI's send unchanged",
            "send_json hands None payload to ComfyUI's send unchanged",
            "a failing handler is logged once and ComfyUI's send still runs every time",
            "an exception from ComfyUI's own send propagates",
            "executed for an expanded node id uploads its output",
            "executed for a nested expanded node id uploads its output",
            "executed for a workflow node still uploads its output",
            "executed for a PreviewImage still uploads nothing",
        })

    def test_a_failed_node_posts_its_reason_before_the_run_is_marked_failed(self):
        self._check(_run_child("node-failure"), {
            "a failed node posts its reason with progress, then failed, in that order",
            "the reason carries no traceback and no node inputs",
            "the reason is capped at 1000 characters",
            "missing fields render as ? and unknown progress as 0",
            "a failing reason POST is logged and failed is still posted",
            "a failing output POST is logged and the reason and failed still post",
        })

    def test_a_patch_that_fails_to_install_is_logged_and_comfyui_still_starts(self):
        self._check(
            _run_child("swizzle-import-broken"),
            {"a failed install is logged at ERROR", "ComfyUI's execute is left in place"},
        )


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--child":
        _child(sys.argv[2], sys.argv[3])
    else:
        unittest.main(verbosity=2)
