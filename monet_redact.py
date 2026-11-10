"""Credentials scrubbed out of what the runner reports (NV2, LORA-VOLUME-CONTRACT section 5).

On an External Lora run three node inputs are credentials. `lora_url` and
`input_id` (custom_routes writes the run input into both) hold the signed
reference `mlr1.{payload}.{signature}`, whose payload carries a presigned link
base64 encoded, or a plain presigned link; `bearer_token` holds a token.
ComfyUI repeats a failing node's inputs in `execution_error`, a validation
error repeats the value it refused, and a log repeats whatever was printed.
All of it reaches Deploy, which stores it, so the node and custom_routes pass
every error, status and log string through here before it leaves:

* a reference, also one cut short, becomes `mlr1.[redacted]`;
* a URL whose query carries a signing parameter (`X-Amz-Signature`,
  `X-Amz-Credential`, `Signature`, `sig`, `token`, ...) keeps scheme, host and
  path and its query becomes `?[redacted]`; a loose `name=value` of those
  parameters keeps the name only;
* `Bearer <token>`, and the value of a `bearer_token` or `Authorization` field,
  become `[redacted]`.

Standard library only, so the node, the routes and the tests share one copy.
Nothing here raises: a scrubber on an error path must not become the error.
"""

import builtins
import functools
import inspect
import logging
import re
import traceback
from collections.abc import Mapping

REDACTED = "[redacted]"
REDACTED_LORA_REF = "mlr1.[redacted]"
REDACTED_QUERY = "?" + REDACTED

# The contract's pattern is `mlr1\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]{43}`. This one
# also takes a reference cut short, since its payload alone carries the link
# (Deploy's `redact_secrets` stops there), and a separator that arrives escaped
# (`\u002e` in JSON, `%2E` in a URL) or a prefix in capitals, so an escaped
# copy of the payload never slips past.
_REF_DOT = r"(?:\.|\\+u002e|%2e)"
_LORA_REF = re.compile(r"(?i)mlr1" + _REF_DOT + r"[A-Za-z0-9_-]+(?:" + _REF_DOT + r"[A-Za-z0-9_-]*)?")

# Query parameters whose value is a credential: AWS SigV4 and R2, CloudFront,
# GCS (X-Goog-Signature ends in "Signature"), Azure SAS (`sig`), and tokens.
_SIGNING_PARAMETERS = (
    "x-amz-signature",
    "x-amz-credential",
    "x-amz-security-token",
    "signature",
    "credential",
    "token",
    "sig",
)
_PARAMETER = r"(?:" + "|".join(re.escape(p) for p in _SIGNING_PARAMETERS) + r")"
# Inside a URL carried in another URL the `?` or `&` before a name arrives
# percent encoded (`%3F`, `%26`), and so does the `=` after it.
_NAME_START = r"(?:(?<![a-z0-9])|(?<=%[0-9a-f]{2}))"
_SIGNED_QUERY = re.compile(r"(?i)" + _NAME_START + _PARAMETER + r"(?:=|%3d)")
_PARAMETER_VALUE = re.compile(r"(?i)" + _NAME_START + r"(" + _PARAMETER + r")(=|%3d)([^\s&,;\"'<>\\`]+)")
_URL = re.compile(r"(?i)\bhttps?://[^\s\"'<>\\`]+")
# Eight characters at least, so prose such as "using bearer token" survives.
_BEARER = re.compile(r"(?i)(?<![a-z0-9_])(bearer[ \t]+)([A-Za-z0-9._~+/=-]{8,})")
# `'bearer_token': 'x'`, `"Authorization": "x"`, `bearer_token='x'` in a repr,
# also inside a JSON string, where the quotes arrive escaped, and the first
# item of a list or tuple value, as ComfyUI's `current_inputs` holds one.
_SECRET_FIELD_TEXT = re.compile(
    r"""(?i)(\\?["']?)\b(bearer_token|authorization)\1(\s*[:=]\s*)([\[(]\s*)?(\\?["'])(.*?)\5""")

# Keys whose value is withheld whole wherever it sits in a structure.
SECRET_FIELDS = frozenset({"bearer_token", "authorization"})

# A withheld literal shorter than this would garble ordinary words.
_MIN_WITHHELD = 8

# ComfyUI events whose data repeats node inputs or error text.
SCRUBBED_EVENTS = frozenset({"execution_error", "execution_interrupted"})

# Python 3.11 and later; a traceback of a group shows its members too.
_GROUP = getattr(builtins, "BaseExceptionGroup", None)


def _strip_signed_query(match):
    url = match.group(0)
    start = url.find("?")
    if start < 0 or not _SIGNED_QUERY.search(url, start + 1):
        return url
    return url[:start] + REDACTED_QUERY


def _field(match):
    key_quote, name, separator, opener, value_quote, value = match.groups()
    if value == "":
        return match.group(0)
    return f"{key_quote}{name}{key_quote}{separator}{opener or ''}{value_quote}{REDACTED}{value_quote}"


def _as_text(value):
    if isinstance(value, str):
        return value
    try:
        return str(value)
    except Exception:  # noqa: BLE001, never raise on an error path
        return f"<unprintable {type(value).__name__}>"


def scrub_text(text, withheld=()):
    """`text` as a string with every credential above replaced.

    `withheld` lists literal values to hide wherever they occur (a node passes
    its own bearer token, which has no shape a pattern could find); values
    shorter than eight characters are ignored."""
    text = _as_text(text)
    try:
        for literal in withheld:
            if isinstance(literal, str) and len(literal) >= _MIN_WITHHELD:
                text = text.replace(literal, REDACTED)
        text = _LORA_REF.sub(REDACTED_LORA_REF, text)
        # Fields first: `bearer_token=('x',)` would otherwise lose only its
        # `(` to the `token=` parameter rule below and keep the token.
        text = _SECRET_FIELD_TEXT.sub(_field, text)
        text = _URL.sub(_strip_signed_query, text)
        text = _PARAMETER_VALUE.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", text)
        return _BEARER.sub(lambda m: f"{m.group(1)}{REDACTED}", text)
    except Exception:  # noqa: BLE001
        return "<unscrubbable text>"


def _blank(value):
    if value is None or value == "":
        return True
    if isinstance(value, (list, tuple)):
        return all(_blank(v) for v in value)
    return False


def _withhold(value):
    if isinstance(value, (list, tuple)):
        return [v if _blank(v) else REDACTED for v in value]
    return REDACTED


def scrub_value(value, withheld=()):
    """A copy of `value` with every string, keys included, through `scrub_text`.

    A `bearer_token` or `Authorization` key that holds anything has its value
    withheld whole. Dicts, lists and tuples (as lists) are copied; anything
    else is kept as it is. Iterative, so nesting deeper than the recursion
    limit cannot crash an error path, and the input is never mutated."""
    holder = [value]
    stack = [(holder, 0)]
    copies = {}  # id of an original container: its copy, so a cycle stays a cycle
    while stack:
        container, slot = stack.pop()
        item = container[slot]
        if isinstance(item, str):
            container[slot] = scrub_text(item, withheld)
        elif isinstance(item, (dict, list, tuple)):
            if id(item) in copies:
                container[slot] = copies[id(item)]
                continue
            if isinstance(item, dict):
                copied = {}
                copies[id(item)] = copied
                for key, inner in item.items():
                    new_key = scrub_text(key, withheld) if isinstance(key, str) else key
                    if isinstance(key, str) and key.lower() in SECRET_FIELDS and not _blank(inner):
                        copied[new_key] = _withhold(inner)
                    else:
                        copied[new_key] = inner
                        stack.append((copied, new_key))
            else:
                copied = list(item)
                copies[id(item)] = copied
                stack.extend((copied, i) for i in range(len(copied)))
            container[slot] = copied
    return holder[0]


def scrub_event_in_place(event, data):
    """Scrub a ComfyUI event's data in place when the event repeats inputs.

    In place because ComfyUI appends the same dict to the prompt's history
    before it sends it, so the history keeps the scrubbed copy too."""
    if event in SCRUBBED_EVENTS and isinstance(data, dict):
        clean = scrub_value(data)
        data.clear()
        data.update(clean)
    return data


def print_scrubbed(text):
    """`print` for a log line that may name a reference, a signed link or a token."""
    print(scrub_text(text))


class ScrubbedError(RuntimeError):
    """Raised in place of an exception whose message would carry a credential.
    Its message is the scrubbed one, and the original is not shown as context."""


def _message(exc):
    try:
        return str(exc)
    except Exception:  # noqa: BLE001, a formatted traceback shows a placeholder
        return None


def _notes(exc):
    notes = getattr(exc, "__notes__", None)
    return [n for n in notes if isinstance(n, str)] if isinstance(notes, list) else []


def _linked(exc):
    """Every exception other than `exc` a formatted traceback of it would show:
    its cause or unsuppressed context, theirs, and an exception group's members."""
    stack, seen = [exc], {id(exc)}
    while stack:
        current = stack.pop()
        follow = [current.__cause__ if current.__cause__ is not None
                  else None if current.__suppress_context__ else current.__context__]
        if _GROUP is not None and isinstance(current, _GROUP):
            follow.extend(current.exceptions)
        for linked in follow:
            if isinstance(linked, BaseException) and id(linked) not in seen:
                seen.add(id(linked))
                stack.append(linked)
                yield linked


def scrub_raised(*withheld_arguments):
    """Decorate a node's function so nothing it raises carries a credential.

    An exception whose message, notes and chain are clean is raised unchanged.
    One whose message would carry a reference, a signed link, a token or the
    value of one of `withheld_arguments` (named parameters of the function,
    withheld literally) is replaced by a ScrubbedError with the scrubbed
    message, raised `from None` so a formatted traceback does not show the
    original. A clean one whose notes leak keeps them scrubbed; one whose cause
    or context leaks (ComfyUI prints the whole chain to the run log) is raised
    with its chain dropped."""

    def decorate(function):
        signature = inspect.signature(function)

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except Exception as exc:
                try:
                    bound = signature.bind_partial(*args, **kwargs).arguments
                except TypeError:
                    bound = kwargs
                withheld = tuple(bound.get(name) for name in withheld_arguments)

                def leaks(text):
                    return text is not None and scrub_text(text, withheld) != text

                message = _message(exc)
                if message is None or leaks(message):
                    clean = (scrub_text(message, withheld) if message is not None
                             else f"{type(exc).__name__}, its message withheld")
                    raise ScrubbedError(clean) from None
                if any(leaks(note) for note in _notes(exc)):
                    exc.__notes__ = [scrub_text(n, withheld) if isinstance(n, str) else n for n in exc.__notes__]
                if any(leaks(_message(e)) or any(leaks(n) for n in _notes(e)) for e in _linked(exc)):
                    if _GROUP is not None and isinstance(exc, _GROUP):
                        raise ScrubbedError(message) from None  # a group's members cannot be taken off it
                    exc.__cause__ = None
                    exc.__context__ = None
                    exc.__suppress_context__ = True
                raise

        return wrapper

    return decorate


def _log_arg(value, keep_objects=False):
    """A log argument scrubbed. With `keep_objects` any other value is kept as
    it is, so `%d` or `%r` formats it as logging would; the rendered line is
    scrubbed whole after formatting anyway. Without it (an attribute a logger
    stores as given) it becomes its scrubbed text."""
    if value is None or isinstance(value, (bool, int, float, str, dict, list, tuple)):
        return scrub_value(value)
    return value if keep_objects else scrub_text(value)


def render_log(msg, args):
    """The message `logging` would render for `msg % args`, scrubbed.

    Follows `LogRecord.getMessage`: a single non empty mapping is the mapping.
    Where that formatting raises (logging would print the arguments to stderr
    unscrubbed), the arguments are appended instead."""
    if args and len(args) == 1 and isinstance(args[0], Mapping) and args[0]:
        args = args[0]
    text = msg if isinstance(msg, str) else _as_text(_log_arg(msg))
    if args:
        if isinstance(args, Mapping):
            clean = _log_arg(dict(args))
        else:
            clean = tuple(_log_arg(a, keep_objects=True) for a in args)
        try:
            text = text % clean
        except Exception:  # noqa: BLE001
            parts = clean.values() if isinstance(clean, Mapping) else clean
            text = " ".join([text] + [_as_text(p) for p in parts])
    return scrub_text(text)


def _format_exc_info(exc_info):
    """The traceback `exc_info` would add to a record, scrubbed."""
    try:
        if isinstance(exc_info, BaseException):
            lines = traceback.format_exception(exc_info)
        elif isinstance(exc_info, tuple):
            lines = traceback.format_exception(*exc_info)
        else:
            lines = [traceback.format_exc()]
        return scrub_text("".join(lines).rstrip())
    except Exception:  # noqa: BLE001
        return "<unformattable traceback>"


_LEVELS = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING,
           "error": logging.ERROR, "critical": logging.CRITICAL}


class ScrubbingLogger:
    """A logger whose every message and attribute is scrubbed before it is written.

    Wraps a `logging.Logger` or a logfire style module; anything it does not
    define is the wrapped logger's own."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def _write(self, level, msg, args, kwargs):
        if isinstance(self._inner, logging.Logger) and not self._inner.isEnabledFor(_LEVELS[level]):
            return None  # nothing is written, so nothing is rendered or scrubbed
        text = render_log(msg, args)
        exc_info = kwargs.pop("exc_info", None)
        if exc_info:
            text = f"{text}\n{_format_exc_info(exc_info)}"
        if isinstance(self._inner, logging.Logger):
            kwargs["stacklevel"] = kwargs.get("stacklevel", 1) + 2
        else:
            kwargs = {k: _log_arg(v) for k, v in kwargs.items()}
        return getattr(self._inner, level)(text, **kwargs)

    def debug(self, msg, *args, **kwargs):
        return self._write("debug", msg, args, kwargs)

    def info(self, msg, *args, **kwargs):
        return self._write("info", msg, args, kwargs)

    def warning(self, msg, *args, **kwargs):
        return self._write("warning", msg, args, kwargs)

    def error(self, msg, *args, **kwargs):
        return self._write("error", msg, args, kwargs)

    def critical(self, msg, *args, **kwargs):
        return self._write("critical", msg, args, kwargs)

    def exception(self, msg, *args, **kwargs):
        kwargs.setdefault("exc_info", True)
        return self._write("error", msg, args, kwargs)
