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


def _run_child(mode):
    env = {k: v for k, v in os.environ.items()
           if k not in ("CD_ENABLE_LOG", "USE_LOGFIRE", "PYTHONPATH")}
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
