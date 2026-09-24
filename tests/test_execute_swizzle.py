"""The execute swizzle must forward whatever ComfyUI passes, and must never cost a run.

Run from the fork root, with nothing but the standard library:

    python3 -m unittest discover -s tests -v

No ComfyUI install is needed. Each fake `execute` below carries the exact
parameter list of a real ComfyUI release, and REJECTS a call that does not fit
it, the way the real function would. The 12-parameter shape is ComfyUI 0.36.0
onward (0.37.0 included); the fixed 11-parameter wrapper this replaces raised

    TypeError: swizzle_execute() takes from 10 to 11 positional arguments but 12 were given

inside ComfyUI's prompt worker thread, which killed the thread and hung every
later run on the container.
"""

import asyncio
import inspect
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import comfydeploy_execute_swizzle as swizzle  # noqa: E402

_BASE = [
    "server",
    "dynprompt",
    "caches",
    "current_item",
    "extra_data",
    "executed",
    "prompt_id",
    "execution_list",
    "pending_subgraph_results",
]

# (is_async, parameter list) for every execute() shape ComfyUI has shipped since
# execution.execute existed, plus one that has not shipped yet.
SHAPES = {
    "sync 9 (v0.1.0 to v0.3.44)": (False, _BASE),
    "async 10 (v0.3.45 to v0.3.67)": (True, _BASE + ["pending_async_nodes"]),
    "async 11 (v0.3.68 to v0.35.x)": (
        True,
        _BASE + ["pending_async_nodes", "ui_outputs"],
    ),
    "async 12 (v0.36.0 onward, 0.37.0 included)": (
        True,
        _BASE + ["pending_async_nodes", "ui_outputs", "asset_manager"],
    ),
    "async 13 (a parameter ComfyUI has not added yet)": (
        True,
        _BASE
        + ["pending_async_nodes", "ui_outputs", "asset_manager", "next_param"],
    ),
}

RESULT = object()
NODE_ID = "7"
PROMPT_ID = "prompt-1"
CLASS_TYPE = "KSampler"
LAST_NODE_ID = "6"


class FakeServer:
    def __init__(self):
        self.last_node_id = LAST_NODE_ID


class FakeDynPrompt:
    def get_node(self, unique_id):
        return {"class_type": CLASS_TYPE, "inputs": {}}


class ExplodingDynPrompt:
    def get_node(self, unique_id):
        raise RuntimeError("get_node exploded")


class OriginFailure(Exception):
    pass


def make_origin(is_async, params, calls, raises=None):
    """A stand-in for execution.execute with a real ComfyUI parameter list.

    It records the raw (args, kwargs) it received, so a test can tell a
    positional from a keyword, and binds them against the declared signature
    first, so a call the real function would reject raises TypeError here too.
    """
    sig = inspect.Signature(
        [
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for name in params
        ]
    )

    def body(args, kwargs):
        sig.bind(*args, **kwargs)
        calls.append((args, kwargs))
        if raises is not None:
            raise raises
        return RESULT

    if is_async:

        async def execute(*args, **kwargs):
            await asyncio.sleep(0)
            return body(args, kwargs)

    else:

        def execute(*args, **kwargs):
            return body(args, kwargs)

    execute.__signature__ = sig
    return execute


def make_args(params, dynprompt=None, server=None):
    """One distinct object per parameter, so identity proves nothing was swapped."""
    values = {name: object() for name in params}
    values["server"] = server if server is not None else FakeServer()
    values["dynprompt"] = dynprompt if dynprompt is not None else FakeDynPrompt()
    values["current_item"] = NODE_ID
    values["prompt_id"] = PROMPT_ID
    return [values[name] for name in params]


def call(wrapper, *args, **kwargs):
    if inspect.iscoroutinefunction(wrapper):
        return asyncio.run(wrapper(*args, **kwargs))
    return wrapper(*args, **kwargs)


def assert_same_objects(test, got, expected):
    test.assertEqual(len(got), len(expected))
    for i, (g, e) in enumerate(zip(got, expected)):
        test.assertIs(g, e, f"argument {i} was replaced on the way through")


class ForwardsEveryShape(unittest.TestCase):
    def test_every_positional_shape_reaches_the_original_unchanged(self):
        for label, (is_async, params) in SHAPES.items():
            with self.subTest(shape=label):
                calls, done = [], []
                origin = make_origin(is_async, params, calls)
                wrapper = swizzle.make_swizzle_execute(
                    origin, lambda *a: done.append(a)
                )
                args = make_args(params)

                result = call(wrapper, *args)

                self.assertIs(result, RESULT)
                self.assertEqual(len(calls), 1)
                got_args, got_kwargs = calls[0]
                assert_same_objects(self, got_args, args)
                self.assertEqual(got_kwargs, {})
                # (b) the instrumentation ran, with what it reads by name
                self.assertEqual(
                    done, [(CLASS_TYPE, LAST_NODE_ID, PROMPT_ID, args[0], NODE_ID)]
                )

    def test_a_keyword_call_is_forwarded_as_keywords(self):
        is_async, params = SHAPES["async 12 (v0.36.0 onward, 0.37.0 included)"]
        calls, done = [], []
        origin = make_origin(is_async, params, calls)
        wrapper = swizzle.make_swizzle_execute(origin, lambda *a: done.append(a))
        args = make_args(params)
        split = params.index("prompt_id")
        positional = args[:split]
        keywords = dict(zip(params[split:], args[split:]))

        result = call(wrapper, *positional, **keywords)

        self.assertIs(result, RESULT)
        got_args, got_kwargs = calls[0]
        assert_same_objects(self, got_args, positional)
        self.assertEqual(set(got_kwargs), set(keywords))
        for name, value in keywords.items():
            self.assertIs(got_kwargs[name], value, f"{name} was replaced")
        self.assertEqual(
            done, [(CLASS_TYPE, LAST_NODE_ID, PROMPT_ID, args[0], NODE_ID)]
        )

    def test_an_all_keyword_call_still_finds_what_the_instrumentation_needs(self):
        is_async, params = SHAPES["async 12 (v0.36.0 onward, 0.37.0 included)"]
        calls, done = [], []
        origin = make_origin(is_async, params, calls)
        wrapper = swizzle.make_swizzle_execute(origin, lambda *a: done.append(a))
        args = make_args(params)
        keywords = dict(zip(params, args))

        result = call(wrapper, **keywords)

        self.assertIs(result, RESULT)
        self.assertEqual(calls[0][0], ())
        self.assertEqual(set(calls[0][1]), set(params))
        self.assertEqual(
            done, [(CLASS_TYPE, LAST_NODE_ID, PROMPT_ID, args[0], NODE_ID)]
        )

    def test_the_wrapper_keeps_the_originals_signature_and_kind(self):
        # Anything else that inspects execution.execute (another node pack, or
        # this fork on a second import) must see ComfyUI's parameters, not
        # (*args, **kwargs).
        for label, (is_async, params) in SHAPES.items():
            with self.subTest(shape=label):
                origin = make_origin(is_async, params, [])
                wrapper = swizzle.make_swizzle_execute(origin, lambda *a: None)
                self.assertEqual(
                    inspect.iscoroutinefunction(wrapper), is_async
                )
                self.assertEqual(
                    list(inspect.signature(wrapper).parameters), params
                )


class InstrumentationNeverCostsARun(unittest.TestCase):
    def _run_with_broken_instrumentation(self, label, dynprompt=None, server=None,
                                         on_node_done=None):
        is_async, params = SHAPES[label]
        calls = []
        origin = make_origin(is_async, params, calls)
        wrapper = swizzle.make_swizzle_execute(
            origin, on_node_done or (lambda *a: None)
        )
        args = make_args(params, dynprompt=dynprompt, server=server)
        with self.assertLogs("comfy-deploy", level="WARNING") as logs:
            result = call(wrapper, *args)
        self.assertIs(result, RESULT)
        self.assertEqual(len(calls), 1)
        assert_same_objects(self, calls[0][0], args)
        return logs

    def test_a_node_lookup_that_raises_is_logged_and_the_node_still_runs(self):
        for label in SHAPES:
            with self.subTest(shape=label):
                done = []
                logs = self._run_with_broken_instrumentation(
                    label,
                    dynprompt=ExplodingDynPrompt(),
                    on_node_done=lambda *a: done.append(a),
                )
                self.assertEqual(done, [])
                self.assertIn("get_node exploded", "\n".join(logs.output))

    def test_a_server_without_last_node_id_is_logged_and_the_node_still_runs(self):
        self._run_with_broken_instrumentation(
            "async 12 (v0.36.0 onward, 0.37.0 included)",
            server=types.SimpleNamespace(),
        )

    def test_a_timing_hook_that_raises_is_logged_and_the_result_still_returns(self):
        def boom(*a):
            raise ValueError("timing hook exploded")

        for label in SHAPES:
            with self.subTest(shape=label):
                logs = self._run_with_broken_instrumentation(label, on_node_done=boom)
                self.assertIn("timing hook exploded", "\n".join(logs.output))


class OriginalFailuresPropagate(unittest.TestCase):
    def test_the_originals_exception_reaches_the_caller_untouched(self):
        for label, (is_async, params) in SHAPES.items():
            with self.subTest(shape=label):
                failure = OriginFailure("the node itself failed")
                calls, done = [], []
                origin = make_origin(is_async, params, calls, raises=failure)
                wrapper = swizzle.make_swizzle_execute(
                    origin, lambda *a: done.append(a)
                )
                with self.assertRaises(OriginFailure) as caught:
                    call(wrapper, *make_args(params))
                self.assertIs(caught.exception, failure)
                self.assertEqual(len(calls), 1)
                self.assertEqual(done, [])


class Installation(unittest.TestCase):
    def test_install_replaces_execute_and_times_each_node_once(self):
        is_async, params = SHAPES["async 12 (v0.36.0 onward, 0.37.0 included)"]
        calls, done = [], []
        origin = make_origin(is_async, params, calls)
        module = types.SimpleNamespace(execute=origin)

        self.assertTrue(
            swizzle.install_execute_swizzle(module, lambda *a: done.append(a))
        )
        self.assertIsNot(module.execute, origin)
        # A second install (the module imported twice) must not wrap the
        # wrapper, which would time every node twice.
        first = module.execute
        swizzle.install_execute_swizzle(module, lambda *a: done.append(a))
        self.assertIs(module.execute, first)

        result = call(module.execute, *make_args(params))
        self.assertIs(result, RESULT)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(done), 1)

    def test_an_install_that_cannot_happen_is_logged_not_raised(self):
        module = types.SimpleNamespace()  # an execution module with no execute
        with self.assertLogs("comfy-deploy", level="ERROR"):
            self.assertFalse(
                swizzle.install_execute_swizzle(module, lambda *a: None)
            )
        self.assertFalse(hasattr(module, "execute"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
