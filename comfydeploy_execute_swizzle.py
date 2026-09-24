"""Per-node timing around ComfyUI's ``execution.execute``, without owning its signature.

``custom_routes`` wraps ``execution.execute`` so it can record how long each node
took and how much VRAM it used. The wrapper used to spell out ComfyUI's parameter
list by hand. ComfyUI 0.36.0 appended a twelfth parameter, ``asset_manager``, and
from then on every node call raised

    TypeError: swizzle_execute() takes from 10 to 11 positional arguments but 12 were given

before ComfyUI's own code ran. That exception escapes into ComfyUI's
``prompt_worker`` thread and kills it, so the container never processes another
prompt and nothing reports why: every later run hangs.

So the wrapper here owns none of the signature. It forwards exactly what it was
given, positional and keyword, to the original, and it reads the four values the
timing needs (``server``, ``dynprompt``, ``current_item``, ``prompt_id``) by NAME,
by binding the call against the original's own signature. Every shape ComfyUI has
shipped works unchanged: the sync nine-parameter form, the async ten, the eleven
with ``ui_outputs``, the twelve with ``asset_manager``, and whatever is appended
next.

The timing is instrumentation, not the run. Anything it does is caught, logged
and dropped; the original's result is returned, and the original's own
exceptions propagate exactly as if the wrapper were not there.

This module imports nothing from ComfyUI, so ``tests/test_execute_swizzle.py``
runs it with the standard library alone.
"""

import functools
import inspect
import logging
import sys

logger = logging.getLogger("comfy-deploy")

# Set on every wrapper this module builds, so a second install (the fork's
# modules imported twice) leaves the first one in place instead of timing every
# node twice.
MARKER = "__comfydeploy_execute_swizzle__"


def make_swizzle_execute(origin_execute, on_node_done):
    """Wrap ``origin_execute`` so ``on_node_done`` runs after each node.

    ``on_node_done(class_type, last_node_id, prompt_id, server, unique_id)`` is
    called after the original returns, with the same arguments the fixed-arity
    wrapper used to pass. It is not called when the original raises, and it is
    skipped (with a warning) when those values cannot be read from the call.
    """
    try:
        signature = inspect.signature(origin_execute)
    except (TypeError, ValueError):
        # Nothing to bind against. The node calls are still forwarded; only the
        # timing is lost, and that is said once here rather than per node.
        signature = None
        logger.warning(
            "comfy-deploy - execution.execute has no readable signature; "
            "runs are unaffected, per-node timings will be missing"
        )

    failures = {"count": 0}

    def report(stage):
        # Called from inside an `except` block. Every failure is logged and
        # names its exception; only the first carries the traceback, so a
        # broken read is visible without a traceback per node of every run.
        failures["count"] += 1
        logger.warning(
            "comfy-deploy - per-node timing failed while %s (%r); the node runs regardless",
            stage,
            sys.exc_info()[1],
            exc_info=failures["count"] == 1,
        )

    def before(args, kwargs):
        """What ``on_node_done`` will need, or None if it cannot be read."""
        if signature is None:
            return None
        try:
            bound = signature.bind_partial(*args, **kwargs).arguments
            server = bound["server"]
            dynprompt = bound["dynprompt"]
            unique_id = bound["current_item"]
            prompt_id = bound["prompt_id"]
            class_type = dynprompt.get_node(unique_id)["class_type"]
            last_node_id = server.last_node_id
            return (class_type, last_node_id, prompt_id, server, unique_id)
        except Exception:
            report("reading the node before it ran")
            return None

    def after(context):
        if context is None:
            return
        try:
            on_node_done(*context)
        except Exception:
            report("recording the node after it ran")

    if inspect.iscoroutinefunction(origin_execute):

        @functools.wraps(origin_execute)
        async def swizzle_execute(*args, **kwargs):
            context = before(args, kwargs)
            result = await origin_execute(*args, **kwargs)
            after(context)
            return result

    else:

        @functools.wraps(origin_execute)
        def swizzle_execute(*args, **kwargs):
            context = before(args, kwargs)
            result = origin_execute(*args, **kwargs)
            after(context)
            return result

    setattr(swizzle_execute, MARKER, True)
    return swizzle_execute


def install_execute_swizzle(execution_module, on_node_done):
    """Replace ``execution_module.execute`` with the timing wrapper.

    Returns True when the wrapper is in place (newly or already), False when it
    could not be installed. Never raises: ComfyUI must still start without the
    timing, but the failure is logged rather than passed over, because a patch
    that silently failed to install is indistinguishable from one that works.
    """
    try:
        origin_execute = execution_module.execute
        if getattr(origin_execute, MARKER, False):
            return True
        execution_module.execute = make_swizzle_execute(origin_execute, on_node_done)
        return True
    except Exception:
        logger.error(
            "comfy-deploy - could not wrap execution.execute; runs are unaffected, "
            "per-node timings will be missing",
            exc_info=True,
        )
        return False
