#!/usr/bin/env python3
"""handoff.py: the control plane of the handoff convention (packets v2, discovery, stages, sync).

The tools that take part are registered in contracts/tool.json (identical in every repository
that follows the convention). One tool hands a task to another as one JSON packet
(contracts/handoff-packet.schema.json, schema_version 2). The sender writes the packet into the
handoff/ folder of the RECEIVING repository, the receiver accepts it, runs the stage chain,
records the result in the same file, and the sender verifies the result. The file is the only
state. Every change is a compare and swap: the writer takes handoff/<id>.lock (a lock left by a
dead process on this host is broken), checks that the file still holds the revision it read and
replaces it through a temporary name with revision + 1.

  python tools/handoff.py new --to twixtor --task "invert twice" \\
      --input clip.raw --width 160 --height 96 --depth 8 --fps 24 --stage selftest-invert
  python tools/handoff.py list [--other | --peer ID] [--status new]
  python tools/handoff.py show ID [--other]
  python tools/handoff.py accept ID [--revision N]
  python tools/handoff.py done ID --output out.raw --run @cmd.json
  python tools/handoff.py verify ID [--other] [--replay [--trust]]
  python tools/handoff.py migrate (ID | --all)
  python tools/handoff.py stages [--provider FILE]
  python tools/handoff.py sync [--to ID] [--check] [PATH ...]
  python tools/handoff.py contracts
  python tools/handoff.py doctor [--conformance]

status:   new -> accepted -> done | failed -> verified. accept on an accepted packet is a no-op;
          done needs an accepted packet; verify can be repeated.
paths:    every path inside a v2 packet is a relative POSIX path (no leading slash, drive,
          backslash or .. segment) from the root of the repository that holds the packet. new
          copies an input or layer that lies outside that repository into handoff/<id>/inputs/
          or handoff/<id>/layers/. Paths given to done (--output, --scratch, the stdin, stdout and
          cwd of a command spec) are taken relative to the root and must lie inside it.
packets:  at most 1 MiB. schema_version 1 packets stay readable (list, show, verify); migrate
          rewrites them as version 2 before any other write.
exit:     the frames/2 column of contracts/error.schema.json: 0 ok, 1 stage_failed, 2 usage,
          3 unsupported, 4 bad_input, broken_pipe and partial_frame, 5 stage_timeout,
          6 tool_missing, 130 interrupted.
--json:   every result carries "contract": "frames/2"; every failure is
          {"ok": false, "contract", "error": {"code", "stage", "exit_code", "message",
          "stderr_tail"}} where exit_code is the failing child's status or null; verify and
          doctor add their checks next to it. Text mode prints one line
          `error: <code>: <message>` on stderr.

Which tool a root is: --self, HANDOFF_SELF, else the registry entry whose markers all exist under
the root. A peer is found through --peer-dir ID=DIR, HANDOFF_<ID>_DIR, else the one sibling folder
that holds all of the peer's markers; folder names are never used. A directory given by flag or
variable that does not exist is an error, not a fall-through.

verify --replay runs the recorded commands again (no shell) with every recorded output replaced by
a scratch path and compares the bytes. Without --trust only ffmpeg, ffprobe and registered stage
providers (the registry binaries, --provider files and HANDOFF_PROVIDERS manifests) are replayed,
each through the local program of that name in the scratch folder; an argument that names a path
outside the packet inputs and the scratch folder refuses the replay.

stdlib only (Python 3.12), runs on Windows and POSIX; run it with python -I.
"""

import argparse
import array
import fnmatch
import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.dirname(HERE)
CONTRACTS_DIR = os.path.join(DEFAULT_ROOT, "contracts")
SCHEMA_PATH = os.path.join(CONTRACTS_DIR, "handoff-packet.schema.json")
SCHEMA_V1_PATH = os.path.join(CONTRACTS_DIR, "handoff-packet-v1.schema.json")
TOOL_PATH = os.path.join(CONTRACTS_DIR, "tool.json")
TOOL_SCHEMA_PATH = os.path.join(CONTRACTS_DIR, "tool.schema.json")
MANIFEST_SCHEMA_PATH = os.path.join(CONTRACTS_DIR, "stage-manifest.schema.json")
HANDOFF_DIR = "handoff"

CONTRACT = "frames/2"
KNOWN_MAJORS = (1, 2)
SCHEMA_VERSION = 2
GEOMETRY_VALUES = ("preserve", "declared")
STATUSES = ("new", "accepted", "done", "failed", "verified")

DEPTH_PIX_FMT = {8: "rgba", 16: "rgba64le", 32: "gbrapf32le"}
BYTES_PER_PIXEL = {8: 4, 16: 8, 32: 16}
GEOMETRY = ("width", "height", "depth", "fps")

TOOL_ID = r"[a-z][a-z0-9_]{0,31}"
TOOL_ID_RE = re.compile(rf"^{TOOL_ID}$")
ID_RE = re.compile(rf"^([0-9]{{8}}T[0-9]{{6}}Z)-({TOOL_ID})-({TOOL_ID})-([0-9a-f]{{6}})$")
REL_RE = re.compile(r"^(?!/)(?![A-Za-z]:)(?!.*\\)(?!(.*/)?\.\.(/|$)).+$")
TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

MAX_PACKET_BYTES = 1 << 20
LOCK_WAIT = 10.0
LOCK_UNREADABLE_STALE = 60.0
MAX_SIBLINGS = 1000
TIMEOUT_EXIT = 124
MISSING_EXIT = 127
# seconds taskkill, and the read of what a killed tree left in its pipes, may take
KILL_TIMEOUT = 10.0
DEFAULT_TIMEOUT = 600.0
LIST_TIMEOUT = 60.0

EXIT_OK = 0
EXIT = {"stage_failed": 1, "usage": 2, "unsupported": 3, "bad_input": 4, "broken_pipe": 4,
        "partial_frame": 4, "stage_timeout": 5, "tool_missing": 6, "interrupted": 130}
TOOL_ENVS = {"ffmpeg": ("MEDIA_FFMPEG_PATH", "FFMPEG"), "ffprobe": ("MEDIA_FFPROBE_PATH", "FFPROBE")}


# ----------------------------------------------------------------------------
# errors
# ----------------------------------------------------------------------------

class HandoffError(Exception):
    """a failure with a code of the error table, the failing child's status and extra output.

    the process exit status follows from the code (EXIT). `extra` is merged into the --json
    failure document (verify and doctor put their checks there), `lines` is printed on standard
    output in text mode before the error line."""

    def __init__(self, code, message, stage=None, child_exit=None, tail=None, extra=None, lines=None):
        if code not in EXIT:
            raise ValueError(f"unknown error code {code}")
        message = str(message).replace("\r", " ").replace("\n", " ")
        super().__init__(message)
        self.code = code
        self.message = message
        self.exit_code = EXIT[code]
        self.stage = stage
        self.child_exit = child_exit
        self.tail = list(tail or [])
        self.extra = dict(extra or {})
        self.lines = list(lines or [])

    def to_object(self):
        return {"code": self.code, "stage": self.stage, "exit_code": self.child_exit,
                "message": self.message, "stderr_tail": self.tail[-5:]}

    def text_line(self):
        line = f"error: {self.code}: {self.message}"
        last = next((t for t in reversed(self.tail) if t.strip()), None)
        if last and last.strip() not in self.message:
            line += ": " + last.strip()
        return line


def usage_error(message):
    return HandoffError("usage", message)


def invalid_error(message, details=None):
    return HandoffError("bad_input", message, extra={"problems": list(details or [])})


# ----------------------------------------------------------------------------
# a small JSON Schema validator: the keywords the contract schemas use
# ----------------------------------------------------------------------------

SCHEMA_KEYWORDS = {"type", "required", "properties", "additionalProperties", "enum", "items",
                   "pattern", "minimum", "maximum", "minLength", "minItems", "const"}
SCHEMA_ANNOTATIONS = {"$schema", "$id", "$comment", "title", "description", "default", "examples"}


class SchemaError(Exception):
    pass


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_type(value, name):
    if name == "object":
        return isinstance(value, dict)
    if name == "array":
        return isinstance(value, list)
    if name == "string":
        return isinstance(value, str)
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return _is_number(value)
    if name == "boolean":
        return isinstance(value, bool)
    if name == "null":
        return value is None
    raise SchemaError(f"unknown type `{name}`")


def _type_name(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _json_equal(a, b):
    """JSON equality: true is not 1, 1 equals 1.0."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if _is_number(a) and _is_number(b):
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def check_schema(schema, path="$"):
    """raise SchemaError when `schema` uses a keyword this validator does not implement.
    annotations and keys that start with x- (tables such as x-codes) are allowed anywhere."""
    if not isinstance(schema, dict):
        raise SchemaError(f"{path}: a schema must be an object")
    for key, value in schema.items():
        if key in SCHEMA_ANNOTATIONS or key.startswith("x-"):
            continue
        if key not in SCHEMA_KEYWORDS:
            raise SchemaError(f"{path}: unsupported keyword `{key}`")
        if key == "properties":
            for name, sub in value.items():
                check_schema(sub, f"{path}.{name}")
        elif key == "items":
            check_schema(value, path + "[]")
        elif key == "additionalProperties" and isinstance(value, dict):
            check_schema(value, path + ".*")
        elif key == "pattern":
            try:
                re.compile(value)
            except re.error as exc:
                raise SchemaError(f"{path}: pattern does not compile: {exc}")


def validate(instance, schema, path="$"):
    """problems of `instance` against `schema` as a list of `$.path: message` strings."""
    problems = []
    _validate(instance, schema, path, problems)
    return problems


def _validate(value, schema, path, problems):
    if "type" in schema:
        names = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_is_type(value, n) for n in names):
            problems.append(f"{path}: expected {' or '.join(names)}, got {_type_name(value)}")
            return
    if "const" in schema and not _json_equal(value, schema["const"]):
        problems.append(f"{path}: must be {json.dumps(schema['const'])}")
    if "enum" in schema and not any(_json_equal(value, c) for c in schema["enum"]):
        problems.append(f"{path}: must be one of {json.dumps(schema['enum'])}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            problems.append(f"{path}: shorter than {schema['minLength']} characters")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            problems.append(f"{path}: does not match {schema['pattern']}")
    if _is_number(value):
        if "minimum" in schema and value < schema["minimum"]:
            problems.append(f"{path}: {value} is below the minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            problems.append(f"{path}: {value} is above the maximum {schema['maximum']}")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                problems.append(f"{path}: missing `{key}`")
        props = schema.get("properties", {})
        for key, sub in props.items():
            if key in value:
                _validate(value[key], sub, f"{path}.{key}", problems)
        extra = schema.get("additionalProperties", True)
        for key in value:
            if key in props:
                continue
            if extra is False:
                problems.append(f"{path}: unexpected property `{key}`")
            elif isinstance(extra, dict):
                _validate(value[key], extra, f"{path}.{key}", problems)
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            problems.append(f"{path}: fewer than {schema['minItems']} items")
        if "items" in schema:
            for i, item in enumerate(value):
                _validate(item, schema["items"], f"{path}[{i}]", problems)


def _reject_constant(name):
    raise ValueError(f"{name} is not allowed in JSON")


def read_json(path, what):
    """parse a JSON file (a UTF-8 byte order mark is allowed, NaN and Infinity are not)."""
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError as exc:
        raise HandoffError("tool_missing", f"cannot read {what} {path}: {exc.strerror or exc}")
    try:
        return json.loads(raw.decode("utf-8-sig"), parse_constant=_reject_constant)
    except (UnicodeDecodeError, ValueError) as exc:
        raise HandoffError("bad_input", f"{what} {path} is not valid JSON: {exc}")


_SCHEMAS = {}


def load_schema(path):
    """a contract schema, read once per path and checked against the validator's keywords."""
    if path not in _SCHEMAS:
        schema = read_json(path, "the schema")
        try:
            check_schema(schema)
        except SchemaError as exc:
            raise HandoffError("bad_input", f"the schema {path} cannot be used: {exc}")
        _SCHEMAS[path] = schema
    return _SCHEMAS[path]


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------

def now_iso():
    """the current time, ISO 8601 UTC at second precision."""
    return datetime.now(timezone.utc).strftime(TIME_FORMAT)


def parse_time(text):
    return datetime.strptime(text, TIME_FORMAT)


def hash_file(path):
    """(lowercase hex SHA-256, byte count) of a file, read in 1 MiB pieces."""
    digest, size = hashlib.sha256(), 0
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def file_digest_or_none(path):
    try:
        return hash_file(path)[0]
    except OSError:
        return None


def frame_size(width, height, depth):
    return width * height * BYTES_PER_PIXEL[depth]


def parse_fps(text):
    """'24', '29.97' or '30000/1001' as an exact 'N/D' in lowest terms; ValueError otherwise."""
    value = Fraction(str(text).strip())
    if value <= 0:
        raise ValueError("a frame rate must be positive")
    return f"{value.numerator}/{value.denominator}"


def resolve(path, base):
    """a path from a packet or a flag: absolute as it is, relative (either slash) to `base`."""
    return os.path.normpath(os.path.join(base, path))


def path_key(path, base):
    return os.path.normcase(resolve(path, base))


def same_path(a, b):
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def rel_or_none(path, root):
    """the relative POSIX path of `path` from `root`, or None when it lies outside the root."""
    real = os.path.normpath(os.path.abspath(path))
    base = os.path.normpath(os.path.abspath(root))
    try:
        rel = os.path.relpath(real, base)
    except ValueError:
        return None
    if rel in (os.curdir, os.pardir) or rel.startswith(os.pardir + os.sep) or os.path.isabs(rel):
        return None
    rel = rel.replace(os.sep, "/")
    return rel if REL_RE.match(rel) else None


def to_rel(path, root, what):
    rel = rel_or_none(path, root)
    if rel is None:
        raise usage_error(f"{what} must lie inside {os.path.abspath(root)}: {os.path.abspath(path)}")
    return rel


def tail_lines(data, count=3):
    text = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else (data or "")
    lines = [t.rstrip() for t in text.strip().splitlines()]
    return lines[-count:]


def discard(path):
    try:
        os.remove(path)
    except OSError:
        pass


def exe_name(name):
    return name + ".exe" if os.name == "nt" else name


def program_stem(program):
    """the program name of an argv[0]: last path part, lower case, without .exe."""
    base = re.split(r"[\\/]", program)[-1].lower()
    return base[:-4] if base.endswith(".exe") else base


def norm_word(word):
    return os.path.normcase(os.path.normpath(word)) if ("/" in word or "\\" in word) else word


# ----------------------------------------------------------------------------
# the tool registry, which tool a root is, and its peers
# ----------------------------------------------------------------------------

_REGISTRY = {}


def load_registry(path=None):
    """contracts/tool.json next to this script, validated against tool.schema.json."""
    path = path or TOOL_PATH
    if path not in _REGISTRY:
        data = read_json(path, "the tool registry")
        problems = validate(data, load_schema(os.path.join(os.path.dirname(path), "tool.schema.json")))
        if not problems:
            ids = [t["id"] for t in data["tools"]]
            problems += [f"duplicate tool id {i}" for i in sorted({i for i in ids if ids.count(i) > 1})]
            for tool in data["tools"]:
                names = {b["name"] for b in tool["binaries"]}
                problems += [f"tool {tool['id']}: provider binary {p['binary']} is not one of its binaries"
                             for p in tool["providers"] if p["binary"] not in names]
        if problems:
            raise invalid_error(f"the tool registry {path} is invalid: " + "; ".join(problems[:3]), problems)
        _REGISTRY[path] = data
    return _REGISTRY[path]


def tool_ids(registry=None):
    return [t["id"] for t in (registry or load_registry())["tools"]]


def tool_entry(tool_id, registry=None):
    return next((t for t in (registry or load_registry())["tools"] if t["id"] == tool_id), None)


def has_markers(folder, tool):
    return all(os.path.exists(os.path.join(folder, *m.split("/"))) for m in tool["markers"])


def detect_self(root, explicit=None, env=None, registry=None):
    """the registered tool id of the repository at `root`, None when it cannot be told.

    order: the explicit id, HANDOFF_SELF, the one registry entry whose markers all exist."""
    env = os.environ if env is None else env
    registry = registry or load_registry()
    ids = tool_ids(registry)
    for given, where in ((explicit, "--self"), (env.get("HANDOFF_SELF"), "HANDOFF_SELF")):
        if given:
            if given not in ids:
                raise usage_error(f"{where} must name a registered tool ({', '.join(ids)}), got `{given}`")
            return given
    hits = [t["id"] for t in registry["tools"] if has_markers(root, t)]
    return hits[0] if len(hits) == 1 else None


def peer_env(tool_id):
    return f"HANDOFF_{tool_id.upper()}_DIR"


def find_peer(tool_id, root, explicit=None, env=None, registry=None):
    """(folder, '') of the peer `tool_id`, or (None, why not).

    order: the explicit folder (--peer-dir), HANDOFF_<ID>_DIR, the one sibling folder of `root`
    that holds all of the peer's markers. a folder that is given and missing is an error."""
    env = os.environ if env is None else env
    registry = registry or load_registry()
    tool = tool_entry(tool_id, registry)
    if tool is None:
        return None, f"{tool_id} is not a registered tool ({', '.join(tool_ids(registry))})"
    var = peer_env(tool_id)
    for given, where in ((explicit, f"--peer-dir {tool_id}="), (env.get(var), var)):
        if given:
            path = os.path.abspath(given)
            if os.path.isdir(path):
                return path, ""
            return None, f"{where} points at a missing folder: {path}"
    here = os.path.abspath(root)
    parent = os.path.dirname(here)
    try:
        names = sorted(os.listdir(parent))[:MAX_SIBLINGS]
    except OSError:
        names = []
    hits = []
    for name in names:
        path = os.path.join(parent, name)
        if name.startswith(".") or same_path(path, here) or not os.path.isdir(path):
            continue
        if has_markers(path, tool):
            hits.append(path)
    if len(hits) == 1:
        return hits[0], ""
    if hits:
        return None, f"{len(hits)} folders next to {here} hold the markers of {tool_id}: {', '.join(hits)}; set {var}"
    return None, (f"no folder next to {here} holds the markers of {tool_id} ({', '.join(tool['markers'])}); "
                  f"set {var} or pass --peer-dir {tool_id}=DIR")


def local_root(args):
    return os.path.abspath(args.root) if getattr(args, "root", None) else DEFAULT_ROOT


def peer_dirs(args):
    """the --peer-dir ID=DIR flags as a table."""
    table, ids = {}, tool_ids()
    for item in getattr(args, "peer_dir", None) or []:
        key, sep, value = item.partition("=")
        if not sep or not key or not value:
            raise usage_error(f"--peer-dir expects ID=DIR, got `{item}`")
        if key not in ids:
            raise usage_error(f"--peer-dir names an unregistered tool `{key}` ({', '.join(ids)})")
        table[key] = value
    return table


def locate_peer(args, tool_id, root):
    path, reason = find_peer(tool_id, root, peer_dirs(args).get(tool_id))
    if path is None:
        raise HandoffError("tool_missing", f"the {tool_id} repository was not found: {reason}")
    return path


def target_peer(args, root):
    """the peer named by --peer, or with --other the only registered peer; None for this root."""
    ids = tool_ids()
    peer = getattr(args, "peer", None)
    if peer:
        if peer not in ids:
            raise usage_error(f"--peer must name a registered tool ({', '.join(ids)}), got `{peer}`")
        return peer
    if getattr(args, "other", False):
        me = detect_self(root, getattr(args, "self_name", None))
        if me is None:
            raise usage_error(f"cannot tell which registered tool {root} is; pass --self ID or --peer ID")
        others = [i for i in ids if i != me]
        if len(others) != 1:
            raise usage_error(f"--other needs exactly one peer, the registry has {len(others)}; pass --peer ID")
        return others[0]
    return None


def packet_root(args):
    """the repository whose handoff/ folder a command works on: this one, or a peer."""
    root = local_root(args)
    peer = target_peer(args, root)
    return root if peer is None else locate_peer(args, peer, root)


def actor(args, root, fallback):
    """who a note is by: the tool this root is, else `fallback`."""
    try:
        return detect_self(root, getattr(args, "self_name", None)) or fallback
    except HandoffError:
        return fallback


# ----------------------------------------------------------------------------
# the packet store: atomic writes, lock, compare and swap, validation
# ----------------------------------------------------------------------------

def temp_name(path):
    directory = os.path.dirname(os.path.abspath(path))
    return os.path.join(directory, f".{os.path.basename(path)}.{os.getpid()}.{secrets.token_hex(2)}.tmp")


def write_atomic(path, text):
    """write `text` through a temporary name in the same folder and a rename, so a reader sees
    the old file or the new one and never half of one. the temporary name starts with a dot and
    ends in .tmp, so listings skip it."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temp = temp_name(path)
    try:
        with open(temp, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    except BaseException:
        discard(temp)
        raise


def copy_atomic(source, target):
    """copy the bytes of `source` to `target` through a temporary name in the target's folder."""
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    temp = temp_name(target)
    try:
        shutil.copyfile(source, temp)
        os.replace(temp, target)
    except BaseException:
        discard(temp)
        raise


def packet_problems(packet, stem=None):
    """every reason a packet is invalid: the schema of its version first, then the rules a
    schema cannot say."""
    if not isinstance(packet, dict):
        return [f"$: expected object, got {_type_name(packet)}"]
    version = packet.get("schema_version")
    if isinstance(version, int) and not isinstance(version, bool) and version == SCHEMA_VERSION:
        problems = validate(packet, load_schema(SCHEMA_PATH))
    elif isinstance(version, int) and not isinstance(version, bool) and version == 1:
        problems = validate(packet, load_schema(SCHEMA_V1_PATH))
    else:
        return [f"$.schema_version: {json.dumps(version)} is not a readable version (1 or {SCHEMA_VERSION})"]
    if problems:
        return problems
    if stem is not None and packet["id"] != stem:
        problems.append(f"id {packet['id']} does not match the file name {stem}.json")
    match = ID_RE.match(packet["id"])
    if packet["from"] == packet["to"]:
        problems.append("from and to name the same tool")
    if match and (match.group(2), match.group(3)) != (packet["from"], packet["to"]):
        problems.append("the id names a different from or to than the packet")
    if version == SCHEMA_VERSION:
        ids = tool_ids()
        for key in ("from", "to"):
            if packet[key] not in ids:
                problems.append(f"{key} {packet[key]} is not a registered tool ({', '.join(ids)})")
    times = {}
    for key in ("created", "updated"):
        try:
            times[key] = parse_time(packet[key])
        except ValueError:
            problems.append(f"{key} is not a real time: {packet[key]}")
    if len(times) == 2 and times["updated"] < times["created"]:
        problems.append("updated is earlier than created")
    for where, records in (("inputs", packet["inputs"]),
                           ("result.outputs", (packet["result"] or {}).get("outputs", []))):
        for i, rec in enumerate(records):
            geometry = [rec[k] is not None for k in ("width", "height", "depth")]
            if any(geometry) and not all(geometry):
                problems.append(f"{where}[{i}]: width, height and depth are all set or all null")
            elif all(geometry) and rec["pix_fmt"] != DEPTH_PIX_FMT[rec["depth"]]:
                problems.append(f"{where}[{i}]: pix_fmt {rec['pix_fmt']} does not match depth {rec['depth']}")
            elif not any(geometry) and (rec["pix_fmt"] is not None or rec["frames"] is not None):
                problems.append(f"{where}[{i}]: pix_fmt and frames need width, height and depth")
    if packet["status"] in ("done", "verified") and packet["result"] is None:
        problems.append(f"status {packet['status']} needs a result")
    for i, cmd in enumerate((packet["result"] or {}).get("commands", [])):
        if not cmd["argv"] or not cmd["argv"][0]:
            problems.append(f"result.commands[{i}]: argv needs a program name")
    return problems


def load_packet(path):
    """read and check a packet file of a readable version; bad_input for anything else."""
    try:
        with open(path, "rb") as handle:
            raw = handle.read(MAX_PACKET_BYTES + 1)
    except OSError as exc:
        raise usage_error(f"cannot read {path}: {exc.strerror or exc}")
    if len(raw) > MAX_PACKET_BYTES:
        raise invalid_error(f"{path} is larger than the 1 MiB packet cap")
    try:
        packet = json.loads(raw.decode("utf-8-sig"), parse_constant=_reject_constant)
    except (UnicodeDecodeError, ValueError) as exc:
        raise invalid_error(f"{path} is not valid JSON: {exc}")
    stem = os.path.basename(path)[:-5] if path.endswith(".json") else None
    problems = packet_problems(packet, stem)
    if problems:
        shown = "; ".join(problems[:3]) + (f"; and {len(problems) - 3} more" if len(problems) > 3 else "")
        raise invalid_error(f"{path} is not a valid packet: {shown}", problems)
    return packet


def save_packet(path, packet):
    """replace the packet file atomically. an invalid or oversized packet is never written."""
    problems = packet_problems(packet, os.path.basename(path)[:-5])
    if problems:
        raise invalid_error("refusing to write an invalid packet: " + "; ".join(problems[:3]), problems)
    text = json.dumps(packet, indent=2, ensure_ascii=False) + "\n"
    size = len(text.encode("utf-8"))
    if size > MAX_PACKET_BYTES:
        raise invalid_error(f"refusing to write {packet['id']}: {size} bytes is above the 1 MiB packet cap")
    write_atomic(path, text)


def require_v2(packet):
    if packet["schema_version"] != SCHEMA_VERSION:
        raise usage_error(f"packet {packet['id']} is schema_version {packet['schema_version']}; "
                          f"run `handoff.py migrate {packet['id']}` first")


def check_revision(packet, expected):
    if expected is not None and packet.get("revision") != expected:
        raise HandoffError("bad_input", f"revision conflict on {packet['id']}: expected {expected}, "
                                        f"the file holds {packet.get('revision')}")


def pid_alive(pid):
    """whether process `pid` runs on this host (an unknown pid counts as gone)."""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == 259
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return not unreaped(pid)


def unreaped(pid):
    """whether a pid that still answers has exited and only waits to be reaped: a killed child whose
    parent is gone, under a new parent that never reaps (Linux /proc; False elsewhere)."""
    try:
        with open(f"/proc/{int(pid)}/stat", "rb") as f:
            stat = f.read()
    except OSError:
        return False
    return stat[stat.rfind(b")") + 2:][:1] in (b"Z", b"X")


def read_lock(path):
    try:
        with open(path, "rb") as handle:
            info = json.loads(handle.read().decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return info if isinstance(info, dict) else None


def lock_is_stale(path):
    """a lock is stale when its process is gone on this host, or it is unreadable and older than
    a minute. a lock of another host is never stale (its pid cannot be checked from here)."""
    info = read_lock(path)
    if info is None:
        try:
            return time.time() - os.path.getmtime(path) > LOCK_UNREADABLE_STALE
        except OSError:
            return False
    if info.get("host") != socket.gethostname():
        return False
    return not pid_alive(info.get("pid"))


class PacketLock:
    """handoff/<id>.lock, created exclusively; holds {pid, host, at}; removed on exit."""

    def __init__(self, packet_file, wait=LOCK_WAIT):
        stem = packet_file[:-5] if packet_file.endswith(".json") else packet_file
        self.path = stem + ".lock"
        self.wait = LOCK_WAIT if wait is None else wait
        self.held = False

    def __enter__(self):
        deadline = time.monotonic() + self.wait
        body = json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "at": now_iso()})
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if lock_is_stale(self.path):
                    discard(self.path)
                    if not os.path.exists(self.path):
                        continue
                if time.monotonic() >= deadline:
                    info = read_lock(self.path) or {}
                    raise HandoffError("stage_timeout",
                                       f"{os.path.basename(self.path)} is held by pid {info.get('pid')} on "
                                       f"{info.get('host')} since {info.get('at')}; waited {self.wait:g} seconds",
                                       stage="lock")
                time.sleep(0.05)
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(body)
            self.held = True
            return self

    def __exit__(self, *exc):
        if self.held:
            discard(self.path)
            self.held = False
        return False


def commit(path, packet, read_revision, wait=None):
    """compare and swap: under the lock, write `packet` as revision read_revision + 1 when the
    file still holds read_revision; a bad_input revision conflict otherwise."""
    with PacketLock(path, wait):
        current = load_packet(path)
        if current.get("revision") != read_revision:
            raise HandoffError("bad_input", f"revision conflict on {packet['id']}: read revision "
                                            f"{read_revision}, the file now holds {current.get('revision')}; "
                                            f"read it again and retry")
        packet["revision"] = read_revision + 1
        touch(packet)
        save_packet(path, packet)


def touch(packet):
    """set `updated` to now (never earlier than `created`)."""
    packet["updated"] = max(now_iso(), packet["created"])


def add_note(packet, by, text):
    packet["notes"].append({"at": max(now_iso(), packet["created"]), "by": by, "text": text})


def handoff_dir(root):
    return os.path.join(root, HANDOFF_DIR)


def list_packet_files(directory):
    if not os.path.isdir(directory):
        return []
    return sorted(n for n in os.listdir(directory)
                  if n.endswith(".json") and not n.startswith(".")
                  and os.path.isfile(os.path.join(directory, n)))


def packet_path(directory, ref):
    """the file for an id, a unique id prefix or the path of a .json file."""
    if not ref:
        raise usage_error("a packet id is required")
    if ref.endswith(".json") and os.path.isfile(ref):
        return os.path.abspath(ref)
    if "/" in ref or "\\" in ref:
        raise usage_error(f"no such packet file: {ref}")
    stems = [n[:-5] for n in list_packet_files(directory)]
    if ref in stems:
        return os.path.join(directory, ref + ".json")
    hits = [s for s in stems if s.startswith(ref)]
    if not hits:
        raise usage_error(f"no packet matches `{ref}` in {directory}")
    if len(hits) > 1:
        raise usage_error(f"`{ref}` matches {len(hits)} packets: " + ", ".join(hits[:5]))
    return os.path.join(directory, hits[0] + ".json")


# ----------------------------------------------------------------------------
# file records
# ----------------------------------------------------------------------------

def blank_record(path, sha256):
    return {"path": path, "sha256": sha256, "width": None, "height": None, "depth": None,
            "pix_fmt": None, "fps": None, "frames": None}


def file_record(path, geometry, base):
    """the record of a file (with its absolute path): hash, and when width, height and depth are
    given, the frame count. a raw file that is not a whole number of frames is refused the way
    vfx refuses it."""
    real = resolve(path, base)
    if not os.path.isfile(real):
        raise usage_error(f"no such file: {real}")
    try:
        digest, size = hash_file(real)
    except OSError as exc:
        raise usage_error(f"cannot read {real}: {exc.strerror or exc}")
    record = blank_record(os.path.abspath(real), digest)
    dims = [geometry.get(k) for k in ("width", "height", "depth")]
    if any(d is not None for d in dims) and not all(d is not None for d in dims):
        raise usage_error("--width, --height and --depth go together for a raw frame file")
    if all(d is not None for d in dims):
        width, height, depth = dims
        whole = frame_size(width, height, depth)
        if size % whole:
            raise usage_error(f"{real} ends with a partial frame ({size % whole} of {whole} bytes) "
                              f"after {size // whole} whole frames")
        record.update(width=width, height=height, depth=depth, pix_fmt=DEPTH_PIX_FMT[depth],
                      frames=size // whole)
    if geometry.get("fps") is not None:
        record["fps"] = geometry["fps"]
    return record


def place_file(real, dest, packet_id, kind, used):
    """the packet path of a file for the repository `dest`: its relative path when it lies
    inside, else a copy under handoff/<id>/<kind>/ (an existing copy with other bytes is refused)."""
    rel = rel_or_none(real, dest)
    if rel is not None:
        return rel
    name = os.path.basename(real)
    candidate, n = name, 1
    while (kind, candidate.lower()) in used:
        candidate, n = f"{n}-{name}", n + 1
    used.add((kind, candidate.lower()))
    rel = f"{HANDOFF_DIR}/{packet_id}/{kind}/{candidate}"
    if not REL_RE.match(rel):
        raise usage_error(f"cannot place {real} in the packet: {rel} is not a portable path")
    target = resolve(rel, dest)
    if os.path.isfile(target):
        if hash_file(target)[0] != hash_file(real)[0]:
            raise usage_error(f"{target} already exists with other bytes; use another --id")
        return rel
    copy_atomic(real, target)
    return rel


def frames_on_disk(path, record):
    """whole frames in the file at `path` by the geometry of `record`; None when it has no
    geometry or the size is not a whole number of frames."""
    if record["width"] is None:
        return None
    whole = frame_size(record["width"], record["height"], record["depth"])
    size = os.path.getsize(path)
    return size // whole if size % whole == 0 else None


# ----------------------------------------------------------------------------
# running commands (done --run, verify --replay)
# ----------------------------------------------------------------------------

SPEC_KEYS = {"argv", "stdin", "stdout", "cwd", "exit_code", "seconds"}


def parse_specs(text, recorded):
    """command specs from JSON text or @file: one object or a list of objects with argv and,
    optionally, stdin, stdout and cwd. a recorded command (--command) also needs exit_code and
    seconds; a command to run (--run) must not have them."""
    if text.startswith("@"):
        try:
            with open(text[1:], "rb") as handle:
                text = handle.read().decode("utf-8-sig")
        except OSError as exc:
            raise usage_error(f"cannot read {text[1:]}: {exc.strerror or exc}")
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise usage_error(f"a command spec must be JSON (or @file): {exc}")
    items = data if isinstance(data, list) else [data]
    specs = []
    for item in items:
        if not isinstance(item, dict):
            raise usage_error("a command spec is a JSON object with argv")
        unknown = sorted(set(item) - SPEC_KEYS)
        if unknown:
            raise usage_error("unknown command spec key: " + ", ".join(unknown))
        argv = item.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv) or not argv[0]:
            raise usage_error("a command spec needs argv, a non-empty list of strings")
        spec = {"argv": argv}
        for key in ("stdin", "stdout", "cwd"):
            value = item.get(key)
            if value is not None and (not isinstance(value, str) or not value):
                raise usage_error(f"command spec `{key}` is a non-empty string or null")
            spec[key] = value
        has_result = "exit_code" in item or "seconds" in item
        if recorded:
            code, secs = item.get("exit_code"), item.get("seconds")
            if isinstance(code, bool) or not isinstance(code, int):
                raise usage_error("a recorded command needs an integer exit_code")
            if isinstance(secs, bool) or not isinstance(secs, (int, float)) or secs < 0:
                raise usage_error("a recorded command needs seconds, a number of 0 or more")
            spec["exit_code"], spec["seconds"] = code, secs
        elif has_result:
            raise usage_error("--run measures exit_code and seconds itself; use --command to record them")
        specs.append(spec)
    return specs


def spec_record(spec, root):
    """(packet form of a spec, absolute cwd, stdin, stdout): the paths of a spec are taken relative
    to the root and must lie inside it; a cwd equal to the root is stored as null."""
    cwd = resolve(spec["cwd"], root) if spec.get("cwd") else root
    stdin = resolve(spec["stdin"], root) if spec.get("stdin") else None
    stdout = resolve(spec["stdout"], root) if spec.get("stdout") else None
    record = dict(spec)
    record["cwd"] = None if same_path(cwd, root) else to_rel(cwd, root, "a command's cwd")
    record["stdin"] = to_rel(stdin, root, "a command's stdin") if stdin else None
    record["stdout"] = to_rel(stdout, root, "a command's stdout") if stdout else None
    return record, cwd, stdin, stdout


def kill_tree(proc):
    """end a child and everything it started: taskkill /T on Windows, the child's process group on
    POSIX (run_tree starts it in a new session, so the group is signalled even after the leader has
    exited), then the child itself."""
    if os.name == "nt":
        if proc.poll() is None:
            try:
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], shell=False,
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=KILL_TIMEOUT)
            except (OSError, subprocess.TimeoutExpired):
                pass
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
    if proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass


def run_tree(argv, timeout, cwd=None, stdin=None, stdout=None, stderr=None):
    """subprocess.run without a shell whose deadline ends the whole process tree, not only the direct
    child: the child starts in a new session on POSIX, and a timeout or an interrupt calls kill_tree.
    returns a CompletedProcess; a timeout raises TimeoutExpired with the output read so far."""
    with subprocess.Popen(argv, shell=False, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr,
                          start_new_session=os.name != "nt") as proc:
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            kill_tree(proc)
            try:
                out, err = proc.communicate(timeout=KILL_TIMEOUT)
            except subprocess.TimeoutExpired:
                out = err = None
            raise subprocess.TimeoutExpired(argv, timeout, output=out, stderr=err) from None
        except BaseException:
            kill_tree(proc)
            raise
    return subprocess.CompletedProcess(argv, proc.returncode, out, err)


def execute(argv, stdin, stdout, cwd, timeout):
    """run one command without a shell, stdin and stdout being file paths or None (discarded).
    returns exit_code, seconds, timed_out and the last lines of standard error. a program that
    cannot start gives exit code 127, a timeout 124."""
    started = time.monotonic()
    tail, timed_out = [], False
    stdin_handle = stdout_handle = None
    try:
        stdin_handle = open(stdin, "rb") if stdin else None
        stdout_handle = open(stdout, "wb") if stdout else None
        proc = run_tree(argv, timeout, cwd=cwd,
                        stdin=stdin_handle if stdin_handle else subprocess.DEVNULL,
                        stdout=stdout_handle if stdout_handle else subprocess.DEVNULL,
                        stderr=subprocess.PIPE)
        code, tail = proc.returncode, tail_lines(proc.stderr)
    except subprocess.TimeoutExpired as exc:
        code, timed_out = TIMEOUT_EXIT, True
        tail = tail_lines(exc.stderr or b"") + [f"timed out after {timeout:g} seconds"]
    except OSError as exc:
        code, tail = MISSING_EXIT, [f"cannot run {argv[0]}: {exc.strerror or exc}"]
    finally:
        for handle in (stdin_handle, stdout_handle):
            if handle is not None:
                handle.close()
    return {"exit_code": code, "seconds": round(time.monotonic() - started, 3),
            "timed_out": timed_out, "stderr_tail": tail}


def command_cwd(spec, root):
    return resolve(spec["cwd"], root) if spec.get("cwd") else root


# ----------------------------------------------------------------------------
# comparing replayed bytes
# ----------------------------------------------------------------------------

def _samples(chunk, depth):
    if depth == 8:
        return chunk
    values = array.array("H" if depth == 16 else "f")
    values.frombytes(chunk)
    if sys.byteorder == "big":
        values.byteswap()
    return values


def compare_samples(path_a, path_b, depth):
    """sample by sample difference of two raw frame files of one depth (8: bytes, 16: unsigned
    little endian, 32: little endian floats). returns the sample count, how many differ, the
    largest difference and whether the lengths match."""
    width = {8: 1, 16: 2, 32: 4}[depth]
    size_a, size_b = os.path.getsize(path_a), os.path.getsize(path_b)
    report = {"same_length": size_a == size_b, "samples": min(size_a, size_b) // width,
              "differing": 0, "max_abs_diff": 0}
    if size_a != size_b:
        return report
    with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
        while True:
            ca, cb = fa.read(1 << 20), fb.read(1 << 20)
            if not ca:
                break
            if ca == cb:
                continue
            usable = len(ca) - len(ca) % width
            for x, y in zip(_samples(ca[:usable], depth), _samples(cb[:usable], depth)):
                if x == y or (x != x and y != y):
                    continue
                diff = abs(x - y)
                report["differing"] += 1
                if diff != diff or diff == float("inf"):
                    diff = sys.float_info.max
                if diff > report["max_abs_diff"]:
                    report["max_abs_diff"] = diff
    return report


def within_tolerance(report, tolerance):
    if not report["same_length"]:
        return False
    fraction = report["differing"] / report["samples"] if report["samples"] else 0.0
    return report["max_abs_diff"] <= tolerance["max_abs_diff"] and fraction <= tolerance["max_diff_fraction"]


# ----------------------------------------------------------------------------
# programs: ffmpeg, ffprobe, built binaries, stage providers
# ----------------------------------------------------------------------------

def resolve_tool(name, env_names):
    """the path of ffmpeg or ffprobe, or (None, why): an environment override that is missing is
    an error, then PATH, then the places a fresh Windows install lands."""
    for var in env_names:
        value = os.environ.get(var)
        if value:
            found = shutil.which(value)
            return (found, "") if found else (None, f"{var} points at a missing program: {value}")
    found = shutil.which(name)
    if found:
        return found, ""
    candidates = []
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidates.append(os.path.join(local, "Microsoft", "WinGet", "Links", exe_name(name)))
    for var in ("ProgramFiles", "ProgramFiles(x86)"):
        base = os.environ.get(var)
        if base:
            candidates.append(os.path.join(base, "ffmpeg", "bin", exe_name(name)))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate, ""
    return None, f"{name} is not on PATH (set MEDIA_{name.upper()}_PATH or {name.upper()})"


def find_binary(name, env_var, extra_dirs):
    """(path, '') of a built tool: an environment override, $CARGO_TARGET_DIR/release, the given
    folders, then PATH; (None, why) when not found."""
    if env_var and os.environ.get(env_var):
        value = os.environ[env_var]
        return (os.path.abspath(value), "") if os.path.isfile(value) else \
            (None, f"{env_var} points at a missing file: {value}")
    folders = []
    if os.environ.get("CARGO_TARGET_DIR"):
        folders.append(os.path.join(os.environ["CARGO_TARGET_DIR"], "release"))
    folders += extra_dirs
    for folder in folders:
        candidate = os.path.join(folder, exe_name(name))
        if os.path.isfile(candidate):
            return candidate, ""
    found = shutil.which(name)
    if found:
        return found, ""
    return None, f"{exe_name(name)} not found in " + ", ".join(folders + ["PATH"])


def tool_binary(tool, binary, tool_root):
    """find_binary for a registry binary, its dirs taken under the tool's root."""
    dirs = [resolve(d, tool_root) for d in binary["dirs"]] if tool_root else []
    return find_binary(binary["name"], binary["env"], dirs)


def first_line(command, timeout=20):
    """first line of a command's standard output (or standard error), '' when it cannot run."""
    try:
        proc = run_tree(command, timeout, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    text = (proc.stdout or proc.stderr or b"").decode("utf-8", errors="replace").strip()
    return text.splitlines()[0].split(" Copyright")[0][:100] if text else ""


def run_capture(command, timeout=LIST_TIMEOUT):
    """(CompletedProcess or None, why): run without a shell, stdin closed, output captured."""
    try:
        return run_tree(command, timeout, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE), ""
    except subprocess.TimeoutExpired:
        return None, f"ran past {timeout:g} seconds"
    except OSError as exc:
        return None, f"did not start: {exc.strerror or exc}"


def contract_major(text):
    match = re.match(r"^frames/([0-9]+)$", text or "")
    return int(match.group(1)) if match else None


def reported_contract(command, adapter="native", timeout=20):
    """(contract text or None, note): the contract a provider's list --json reports."""
    if adapter == "vfx-list":
        return "frames/1", "implied by the vfx-list adapter (vfx list --json has no contract field)"
    proc, why = run_capture(command, timeout)
    if proc is None:
        return None, why
    if proc.returncode != 0:
        return None, f"exit code {proc.returncode}"
    try:
        data = json.loads(proc.stdout.decode("utf-8", errors="replace"))
    except ValueError:
        return None, "output is not JSON"
    if isinstance(data, dict) and isinstance(data.get("contract"), str):
        return data["contract"], ""
    return None, "no contract field"


def adapt_vfx_list(entries, tool_id, version, command):
    """the stage manifest of `vfx list --json`, a bare array of effect entries. every vfx effect
    keeps its frame geometry; a temporal effect spools its whole input before it writes."""
    if not isinstance(entries, list):
        raise HandoffError("bad_input", "vfx list --json did not print an array")
    stages = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise HandoffError("bad_input", "vfx list --json printed an entry without an id")
        kind = entry.get("kind")
        layers = [dict(layer, optional=bool(layer.get("optional", False)))
                  for layer in entry.get("layers", []) if isinstance(layer, dict)]
        stage = {"id": entry["id"], "kind": kind, "spool": kind == "temporal", "geometry": "preserve",
                 "depths": [8, 16, 32], "params": list(entry.get("params", [])), "layers": layers}
        for key in ("name", "family", "hidden"):
            if key in entry:
                stage[key] = entry[key]
        stages.append(stage)
    return {"contract": "frames/1",
            "provider": {"id": tool_id, "version": version, "command": list(command), "adapter": "vfx-list"},
            "stages": stages}


def check_manifest(data, source):
    problems = validate(data, load_schema(MANIFEST_SCHEMA_PATH))
    if problems:
        raise invalid_error(f"{source} is not a valid stage manifest: " + "; ".join(problems[:3]), problems)
    return data


def load_manifest_file(path):
    """a *.stage.json file; a relative program path in its command is relative to its folder."""
    data = check_manifest(read_json(path, "the stage manifest"), path)
    command = data["provider"]["command"]
    if ("/" in command[0] or "\\" in command[0]) and not os.path.isabs(command[0]):
        command[0] = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(path)), command[0]))
    return data


def manifest_files(args):
    """([(manifest, source)], problems) of --provider files and HANDOFF_PROVIDERS folders. an
    explicit file that fails is an error; a file in a folder that fails is reported."""
    found, problems = [], []
    for path in getattr(args, "providers", None) or []:
        found.append((load_manifest_file(path), path))
    for folder in filter(None, os.environ.get("HANDOFF_PROVIDERS", "").split(os.pathsep)):
        if not os.path.isdir(folder):
            problems.append(f"HANDOFF_PROVIDERS names a missing folder: {folder}")
            continue
        for name in sorted(os.listdir(folder)):
            if name.endswith(".stage.json"):
                path = os.path.join(folder, name)
                try:
                    found.append((load_manifest_file(path), path))
                except HandoffError as exc:
                    problems.append(exc.message)
    return found, problems


def provider_manifest(tool, provider, binary, program):
    """run the provider's list command and return its stage manifest (adapted when needed)."""
    command = [program] + provider["prefix"]
    listing = command + provider["list_args"]
    proc, why = run_capture(listing)
    if proc is not None and proc.returncode != 0 and "--all" in provider["list_args"]:
        listing = command + [a for a in provider["list_args"] if a != "--all"]
        proc, why = run_capture(listing)
    if proc is None:
        raise HandoffError("tool_missing", f"{' '.join(listing)} {why}")
    if proc.returncode != 0:
        raise HandoffError("stage_failed", f"{' '.join(listing)} exited with code {proc.returncode}",
                           child_exit=proc.returncode, tail=tail_lines(proc.stderr))
    try:
        data = json.loads(proc.stdout.decode("utf-8", errors="replace"))
    except ValueError:
        raise HandoffError("bad_input", f"{' '.join(listing)} printed no JSON")
    if provider["adapter"] == "vfx-list":
        version = first_line([program] + binary["version_args"]) or "unknown"
        data = adapt_vfx_list(data, tool["id"], version, command)
    return check_manifest(data, " ".join(listing))


def registry_manifests(args, root):
    """([(manifest, source)], problems) of every registry provider whose binary is found."""
    registry = load_registry()
    me = detect_self(root, getattr(args, "self_name", None))
    dirs = peer_dirs(args)
    found, problems = [], []
    for tool in registry["tools"]:
        if not tool["providers"]:
            continue
        if tool["id"] == me:
            tool_root, why = root, ""
        else:
            tool_root, why = find_peer(tool["id"], root, dirs.get(tool["id"]))
        binaries = {b["name"]: b for b in tool["binaries"]}
        for provider in tool["providers"]:
            binary = binaries[provider["binary"]]
            program, reason = tool_binary(tool, binary, tool_root)
            if program is None:
                problems.append(f"{tool['id']}: {reason}" + (f" ({why})" if why else ""))
                continue
            try:
                found.append((provider_manifest(tool, provider, binary, program), f"{tool['id']}:{binary['name']}"))
            except HandoffError as exc:
                problems.append(f"{tool['id']}: {exc.message}")
    return found, problems


def quote_word(word):
    if word and not re.search(r"[\s\"']", word):
        return word
    return '"' + word.replace('"', '\\"') + '"'


def stage_string(command, stage_id):
    """a ready --stage / --pre-stage / --post-stage command with the wrapper's placeholders."""
    words = list(command) + ["apply", stage_id, "--width", "{width}", "--height", "{height}",
                             "--depth", "{depth}", "--fps", "{fps}"]
    return " ".join(quote_word(w) for w in words)


def trusted_commands(args, root):
    """the commands a replay runs without --trust: ffmpeg, ffprobe, every registry binary that
    provides stages, and every manifest provider (its whole command prefix must match)."""
    entries = []
    for name in ("ffmpeg", "ffprobe"):
        path, _ = resolve_tool(name, TOOL_ENVS[name])
        entries.append({"stem": name, "rest": [], "program": path, "source": name})
    registry = load_registry()
    me = detect_self(root, getattr(args, "self_name", None))
    dirs = peer_dirs(args)
    for tool in registry["tools"]:
        names = {p["binary"] for p in tool["providers"]}
        if not names:
            continue
        tool_root = root if tool["id"] == me else find_peer(tool["id"], root, dirs.get(tool["id"]))[0]
        for binary in tool["binaries"]:
            if binary["name"] in names:
                path, _ = tool_binary(tool, binary, tool_root)
                entries.append({"stem": binary["name"].lower(), "rest": [], "program": path, "source": tool["id"]})
    for manifest, source in manifest_files(args)[0]:
        command = manifest["provider"]["command"]
        entries.append({"stem": program_stem(command[0]), "rest": [norm_word(w) for w in command[1:]],
                        "program": command[0], "source": source})
    return entries


def match_trusted(argv, entries):
    stem = program_stem(argv[0])
    for entry in entries:
        rest = entry["rest"]
        if entry["stem"] == stem and [norm_word(w) for w in argv[1:1 + len(rest)]] == rest:
            return entry
    return None


def names_a_place(text, cwd):
    """whether a replay argument names a file: absolute, or with a folder part whose folder exists
    (so rates like 30000/1001 and expressions like PTS/2 are words, not paths). only the separators
    of this system count, so an escaped character in a filter is not a folder on POSIX."""
    separators = [s for s in (os.sep, os.altsep) if s]
    if not text or not (os.path.isabs(text) or any(s in text for s in separators)):
        return False
    return os.path.isdir(os.path.dirname(resolve(text, cwd)) or cwd)


def inside(path, folder):
    """whether `path` lies in `folder`; a path on another drive does not."""
    path, folder = os.path.normcase(path), os.path.normcase(folder)
    try:
        return os.path.commonpath([path, folder]) == folder
    except ValueError:
        return False


def confine(value, cwd, scratch, inputs):
    """an argument of a replay without --trust: a packet input becomes its absolute path, a path in
    the scratch folder stays, a word stays; any other path gives None, so the command is refused.
    the text after every = in the word is judged the same way, so key=value=PATH is too."""
    texts = [("", value)] + [(value[:i + 1], value[i + 1:]) for i, ch in enumerate(value) if ch == "=" and value[i + 1:]]
    for prefix, text in texts:
        key = path_key(text, cwd)
        if key in inputs:
            return prefix + inputs[key]
        if names_a_place(text, cwd) and not inside(resolve(text, cwd), scratch):
            return None
    return value


def check_replay_programs(packet, entries):
    names = sorted({e["stem"] for e in entries})
    for n, cmd in enumerate(packet["result"]["commands"]):
        if match_trusted(cmd["argv"], entries) is None:
            raise HandoffError("unsupported", f"replay refuses command[{n}] `{cmd['argv'][0]}`: only "
                                              f"{', '.join(names)} (ffmpeg, ffprobe and registered "
                                              f"providers) replay without --trust", stage="replay")


# ----------------------------------------------------------------------------
# verify
# ----------------------------------------------------------------------------

def verify_checks(packet, root, replay, timeout, trusted=None):
    """run every check on a packet. returns (checks, replay report or None).

    a check is {name, ok, target, detail, code, exit_code}; code is the error code of a failing
    check: bad_input (a hash or an expectation), partial_frame (not a whole number of frames),
    stage_failed (a command exited non-zero or replayed differently), stage_timeout and
    tool_missing (a replayed command timed out or could not start)."""
    checks = []

    def add(name, ok, target, detail, code=None, exit_code=None):
        checks.append({"name": name, "ok": bool(ok), "target": target, "detail": detail,
                       "code": None if ok else (code or "bad_input"), "exit_code": exit_code})

    def check_file(kind, index, record):
        real = resolve(record["path"], root)
        label = f"{kind}[{index}]"
        if not os.path.isfile(real):
            add(f"{kind}-exists", False, real, f"{label}: missing file {real}")
            return None
        digest, size = hash_file(real)
        add(f"{kind}-sha256", digest == record["sha256"], real,
            f"{label}: sha256 matches" if digest == record["sha256"] else
            f"{label}: sha256 is {digest}, the packet says {record['sha256']}")
        if record["width"] is not None:
            whole = frame_size(record["width"], record["height"], record["depth"])
            frames = size // whole if size % whole == 0 else None
            if frames is None:
                add(f"{kind}-frames", False, real,
                    f"{label}: {size} bytes is not a whole number of {record['width']}x{record['height']} "
                    f"depth {record['depth']} frames ({whole} bytes each)", "partial_frame")
            else:
                add(f"{kind}-frames", frames == record["frames"], real,
                    f"{label}: {frames} frames" if frames == record["frames"] else
                    f"{label}: the file holds {frames} frames, the packet says {record['frames']}")
        return real

    for i, record in enumerate(packet["inputs"]):
        check_file("input", i, record)

    result, status = packet["result"], packet["status"]
    report = None
    if result is not None:
        output_paths = [check_file("output", i, record) for i, record in enumerate(result["outputs"])]
        if status in ("done", "verified"):
            expect = packet["expect"]
            for i, record in enumerate(result["outputs"]):
                real = output_paths[i]
                if real is None:
                    continue
                if expect["size"] is not None:
                    size = os.path.getsize(real)
                    add("expect-size", size == expect["size"], real,
                        f"output[{i}]: {size} bytes" if size == expect["size"] else
                        f"output[{i}]: {size} bytes, expected {expect['size']}")
                if expect["frames"] is not None:
                    frames = frames_on_disk(real, record)
                    if record["width"] is None:
                        add("expect-frames", False, real,
                            f"output[{i}]: no frame geometry recorded, cannot count frames")
                    elif frames is None:
                        add("expect-frames", False, real,
                            f"output[{i}]: not a whole number of frames", "partial_frame")
                    else:
                        add("expect-frames", frames == expect["frames"], real,
                            f"output[{i}]: {frames} frames" if frames == expect["frames"] else
                            f"output[{i}]: {frames} frames, expected {expect['frames']}")
            for i, cmd in enumerate(result["commands"]):
                name = program_stem(cmd["argv"][0])
                add("command-exit", cmd["exit_code"] == 0, cmd["argv"][0],
                    f"command[{i}] {name}: exit code 0" if cmd["exit_code"] == 0 else
                    f"command[{i}] {name}: recorded exit code {cmd['exit_code']}",
                    "stage_failed", cmd["exit_code"])
        if replay:
            report = replay_commands(packet, root, timeout, add, trusted)
    return checks, report


def replay_commands(packet, root, timeout, add, trusted=None):
    """run the recorded commands again in a scratch folder and compare their outputs.

    every recorded output path, and every file a command writes to standard output, is replaced
    by a path in the scratch folder (also inside `--flag=PATH`), so nothing recorded is
    overwritten. with `trusted`, argv[0] becomes the local program of the matching trusted entry.
    the exit codes must equal the recorded ones; the bytes must equal the recorded output
    (determinism exact) or stay within the tolerance (tolerance)."""
    result, expect = packet["result"], packet["expect"]
    scratch = tempfile.mkdtemp(prefix="handoff_replay_")
    report = {"commands": [], "outputs": []}
    try:
        mapping, aliases = {}, []
        inputs = {path_key(resolve(r["path"], root), root): resolve(r["path"], root) for r in packet["inputs"]}
        for i, record in enumerate(result["outputs"]):
            real = resolve(record["path"], root)
            alias = os.path.join(scratch, f"out{i}_{os.path.basename(real)}")
            mapping[path_key(real, root)] = alias
            aliases.append((record, real, alias))
        aux = 0
        for n, cmd in enumerate(result["commands"]):
            cwd = command_cwd(cmd, root)

            def swap(value, cwd=cwd):
                hit = mapping.get(path_key(value, cwd))
                if hit:
                    return hit
                head, sep, rest = value.partition("=")
                if sep and rest:
                    hit = mapping.get(path_key(rest, cwd))
                    if hit:
                        return head + sep + hit
                return value

            stdout = resolve(cmd["stdout"], root) if cmd["stdout"] is not None else None
            if stdout is not None:
                key = path_key(stdout, root)
                if key not in mapping:
                    aux += 1
                    mapping[key] = os.path.join(scratch, f"aux{aux}_{os.path.basename(stdout)}")
            argv = [cmd["argv"][0]] + [swap(a) for a in cmd["argv"][1:]]
            name = program_stem(cmd["argv"][0])
            if trusted is not None:
                entry = match_trusted(cmd["argv"], trusted)
                program = entry["program"]
                if program is None:
                    add("replay-command", False, cmd["argv"][0],
                        f"replay command[{n}] {name}: the local {name} was not found", "tool_missing")
                    report["commands"].append({"argv": argv, "exit_code": None, "recorded_exit_code": cmd["exit_code"],
                                               "seconds": 0, "stderr_tail": []})
                    continue
                # without --trust a command reaches no path but the packet inputs and the scratch
                # folder, and runs in the scratch folder, so a bare file name lands there too
                fixed = 1 + len(entry["rest"])
                confined = [confine(a, cwd, scratch, inputs) for a in argv[fixed:]]
                outside = next((a for a, c in zip(argv[fixed:], confined) if c is None), None)
                if outside is not None:
                    add("replay-command", False, cmd["argv"][0],
                        f"replay command[{n}] {name}: `{outside}` names a path outside the packet; replay it with "
                        f"--trust", "unsupported")
                    report["commands"].append({"argv": argv, "exit_code": None, "recorded_exit_code": cmd["exit_code"],
                                               "seconds": 0, "stderr_tail": []})
                    continue
                argv = [program] + argv[1:fixed] + confined
                cwd = scratch
            stdin = resolve(cmd["stdin"], root) if cmd["stdin"] is not None else None
            stdin = swap(stdin) if stdin is not None else None
            stdout_path = mapping[path_key(stdout, root)] if stdout is not None else None
            sys.stderr.write("[replay] " + " ".join(argv) + "\n")
            ran = execute(argv, stdin, stdout_path, cwd, timeout)
            report["commands"].append({"argv": argv, "exit_code": ran["exit_code"],
                                       "recorded_exit_code": cmd["exit_code"], "seconds": ran["seconds"],
                                       "stderr_tail": ran["stderr_tail"]})
            if ran["timed_out"]:
                add("replay-command", False, cmd["argv"][0], f"replay command[{n}] {name}: timed out",
                    "stage_timeout")
            elif ran["exit_code"] == MISSING_EXIT and cmd["exit_code"] != MISSING_EXIT:
                add("replay-command", False, cmd["argv"][0],
                    f"replay command[{n}] {name}: " + (ran["stderr_tail"][-1] if ran["stderr_tail"] else "did not start"),
                    "tool_missing")
            else:
                same = ran["exit_code"] == cmd["exit_code"]
                add("replay-command", same, cmd["argv"][0],
                    f"replay command[{n}] {name}: exit code {ran['exit_code']}" if same else
                    f"replay command[{n}] {name}: exit code {ran['exit_code']}, recorded {cmd['exit_code']}",
                    "stage_failed", ran["exit_code"])
        depth = next((r["depth"] for r in result["outputs"] + packet["inputs"] if r["depth"] is not None), None)
        for i, (record, real, alias) in enumerate(aliases):
            entry = {"path": record["path"], "recorded_sha256": record["sha256"], "replay_sha256": None,
                     "match": False}
            if not os.path.isfile(alias):
                add("replay-output", False, real,
                    f"output[{i}]: the replay did not write it (is its path an argument or the stdout of a command?)")
                report["outputs"].append(entry)
                continue
            entry["replay_sha256"] = hash_file(alias)[0]
            rule = expect["determinism"]
            if rule == "none":
                entry["match"] = True
                add("replay-output", True, real, f"output[{i}]: written (determinism none, bytes not compared)")
            elif rule == "exact" or depth is None:
                entry["match"] = entry["replay_sha256"] == record["sha256"]
                if rule == "tolerance":
                    detail = f"output[{i}]: tolerance needs a frame depth in the packet; compared exactly"
                elif entry["match"]:
                    detail = f"output[{i}]: replay sha256 equals the recorded one"
                else:
                    detail = f"output[{i}]: replay sha256 {entry['replay_sha256']}, recorded {record['sha256']}"
                add("replay-output", entry["match"], real, detail)
            else:
                diff = compare_samples(real, alias, depth)
                entry["compare"] = diff
                entry["match"] = within_tolerance(diff, expect["tolerance"])
                detail = (f"output[{i}]: {diff['differing']} of {diff['samples']} samples differ, "
                          f"largest difference {diff['max_abs_diff']:g}, "
                          f"tolerance {expect['tolerance']['max_abs_diff']:g} and "
                          f"{expect['tolerance']['max_diff_fraction']:g}" if diff["same_length"] else
                          f"output[{i}]: the replay is {os.path.getsize(alias)} bytes, the output "
                          f"{os.path.getsize(real)}")
                add("replay-output", entry["match"], real, detail)
            report["outputs"].append(entry)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return report


# ----------------------------------------------------------------------------
# events: options whose order matters (--input then its --width, --stage then its --param)
# ----------------------------------------------------------------------------

class Event(argparse.Action):
    """append (dest, value) to namespace.events, so the command line order survives."""

    def __call__(self, parser, namespace, values, option_string=None):
        events = getattr(namespace, "events", None)
        if events is None:
            events = []
            setattr(namespace, "events", events)
        events.append((self.dest, values))


def split_events(events, owner_kinds):
    """(defaults, items): geometry flags before the first owner are defaults for every owner,
    a geometry flag after an owner belongs to that owner. items are the other events in order."""
    defaults, items, last = {}, [], None
    for kind, value in events or []:
        if kind in GEOMETRY:
            (last["geom"] if last is not None else defaults)[kind] = value
        else:
            item = {"kind": kind, "value": value, "geom": {}}
            items.append(item)
            if kind in owner_kinds:
                last = item
    return defaults, items


def positive_int(text):
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value < 1:
        raise argparse.ArgumentTypeError("expected a positive whole number")
    return value


def fps_arg(text):
    try:
        return parse_fps(text)
    except (ValueError, ZeroDivisionError):
        raise argparse.ArgumentTypeError("expected a positive rate such as 24, 29.97 or 30000/1001")


def count_or_any(text):
    if text == "any":
        return "any"
    try:
        value = int(text)
    except ValueError:
        value = -1
    if value < 0:
        raise argparse.ArgumentTypeError("expected a whole number of 0 or more, or `any`")
    return value


def nonneg_number(text):
    try:
        value = float(text)
    except ValueError:
        value = -1.0
    if not value >= 0 or value == float("inf"):
        raise argparse.ArgumentTypeError("expected a number of 0 or more")
    return int(value) if value.is_integer() else value


def fraction_arg(text):
    value = nonneg_number(text)
    if value > 1:
        raise argparse.ArgumentTypeError("expected a fraction from 0 to 1")
    return value


def seconds_arg(text):
    try:
        value = float(text)
    except ValueError:
        value = 0.0
    if not value > 0:
        raise argparse.ArgumentTypeError("expected a positive number of seconds")
    return value


def wait_arg(text):
    try:
        value = float(text)
    except ValueError:
        value = -1.0
    if not value >= 0 or value == float("inf"):
        raise argparse.ArgumentTypeError("expected a number of seconds, 0 or more")
    return value


# ----------------------------------------------------------------------------
# commands
# ----------------------------------------------------------------------------

def summary(packet, path):
    return {"id": packet["id"], "schema_version": packet["schema_version"], "revision": packet.get("revision"),
            "status": packet["status"], "from": packet["from"], "to": packet["to"], "task": packet["task"],
            "created": packet["created"], "updated": packet["updated"], "path": path}


def cmd_new(args):
    root = local_root(args)
    ids = tool_ids()
    me = detect_self(root, args.self_name)
    sender = args.from_name or me
    if sender is None:
        raise usage_error(f"cannot tell which registered tool {root} is; pass --from ID ({', '.join(ids)})")
    for flag, value in (("--from", sender), ("--to", args.to_name)):
        if value not in ids:
            raise usage_error(f"{flag} must name a registered tool ({', '.join(ids)}), got `{value}`")
    if sender == args.to_name:
        raise usage_error("--from and --to must name different tools")
    if args.root or me == args.to_name:
        dest = root
    else:
        dest = locate_peer(args, args.to_name, root)
    task = args.task.strip()
    if not task:
        raise usage_error("--task must not be empty")
    stamp = now_iso()
    packet_id = args.packet_id
    if packet_id is not None:
        match = ID_RE.match(packet_id)
        if not match or (match.group(2), match.group(3)) != (sender, args.to_name):
            raise usage_error(f"--id must look like <YYYYMMDDTHHMMSSZ>-{sender}-{args.to_name}-<6 hex digits>")
    else:
        compact = stamp.replace("-", "").replace(":", "")
        packet_id = f"{compact}-{sender}-{args.to_name}-{secrets.token_hex(3)}"

    cwd = os.getcwd()
    defaults, items = split_events(getattr(args, "events", None), ("input", "file"))
    records, stages = [], []
    for item in items:
        kind, value = item["kind"], item["value"]
        if kind == "file":
            if item["geom"]:
                raise usage_error("--width, --height, --depth and --fps do not apply to --file")
            records.append(file_record(value, {}, cwd))
        elif kind == "input":
            geometry = dict(defaults)
            geometry.update(item["geom"])
            records.append(file_record(value, geometry, cwd))
        elif kind == "stage":
            stages.append({"stage": value, "params": {}, "layers": {}})
        elif kind in ("param", "layer"):
            if not stages:
                raise usage_error(f"--{kind} needs a --stage before it")
            key, sep, val = value.partition("=")
            if not sep or not key or (kind == "layer" and not val):
                raise usage_error(f"--{kind} expects " + ("K=V" if kind == "param" else "ID=PATH") + f", got `{value}`")
            table = stages[-1]["params" if kind == "param" else "layers"]
            if key in table:
                raise usage_error(f"--{kind} `{key}` is given twice for stage {stages[-1]['stage']}")
            if kind == "layer":
                real = resolve(val, cwd)
                if not os.path.isfile(real):
                    real = resolve(val, dest)
                if not os.path.isfile(real):
                    raise usage_error(f"--layer {key}: no such file: {val}")
                val = real
            table[key] = val

    first = next((r for r in records if r["frames"] is not None), None)
    frames = args.expect_frames
    if frames is None:
        frames = first["frames"] if first else None
    elif frames == "any":
        frames = None
    size = args.expect_size
    if size is None:
        size = frames * frame_size(first["width"], first["height"], first["depth"]) \
            if (first is not None and frames is not None) else None
    elif size == "any":
        size = None
    tolerant = args.tolerance_abs is not None or args.tolerance_fraction is not None
    rule = args.determinism or ("tolerance" if tolerant else "exact")
    if rule != "tolerance" and tolerant:
        raise usage_error("--tolerance-abs and --tolerance-fraction need --determinism tolerance")

    used = set()
    for record in records:
        record["path"] = place_file(record["path"], dest, packet_id, "inputs", used)
    for stage in stages:
        for key in list(stage["layers"]):
            stage["layers"][key] = place_file(stage["layers"][key], dest, packet_id, "layers", used)
    packet = {
        "schema_version": SCHEMA_VERSION, "id": packet_id, "revision": 1, "created": stamp, "updated": stamp,
        "from": sender, "to": args.to_name, "task": task, "status": "new",
        "inputs": records, "request": {"stages": stages},
        "expect": {"frames": frames, "size": size, "determinism": rule,
                   "tolerance": {"max_abs_diff": args.tolerance_abs or 0,
                                 "max_diff_fraction": args.tolerance_fraction or 0}},
        "result": None, "notes": [],
    }
    add_note(packet, sender, "created")
    for text in args.notes or []:
        add_note(packet, sender, text)

    directory = handoff_dir(dest)
    path = os.path.join(directory, packet_id + ".json")
    os.makedirs(directory, exist_ok=True)
    with PacketLock(path, args.lock_wait):
        if os.path.exists(path):
            existing = load_packet(path)
            keys = ("from", "to", "task", "inputs", "request", "expect")
            if all(existing.get(k) == packet[k] for k in keys):
                return ({"id": packet_id, "path": path, "changed": False, "packet": existing},
                        [packet_id, path])
            raise usage_error(f"packet {packet_id} already exists with different content: {path}")
        save_packet(path, packet)
    return {"id": packet_id, "path": path, "changed": True, "packet": packet}, [packet_id, path]


def cmd_list(args):
    root = packet_root(args)
    directory = handoff_dir(root)
    rows, invalid = [], []
    for name in list_packet_files(directory):
        path = os.path.join(directory, name)
        try:
            packet = load_packet(path)
        except HandoffError as exc:
            invalid.append({"path": path, "error": exc.message})
            continue
        if any(want and packet[key] != want for key, want in
               (("status", args.status), ("from", args.from_filter), ("to", args.to_filter))):
            continue
        rows.append(summary(packet, path))
    if args.status or args.from_filter or args.to_filter:
        invalid = []
    lines = []
    if rows:
        width = max(len(r["id"]) for r in rows)
        for r in rows:
            task = r["task"] if len(r["task"]) <= 50 else r["task"][:47] + "..."
            lines.append(f"{r['id'].ljust(width)}  {r['status'].ljust(8)}  {r['from']} -> {r['to']}  "
                         f"r{r['revision'] or 0}  {r['updated']}  {task}")
    for bad in invalid:
        lines.append(f"invalid: {bad['path']}: {bad['error']}")
    lines.append(f"{len(rows)} packet(s) in {directory}")
    return {"dir": directory, "packets": rows, "invalid": invalid}, lines


def describe_record(record):
    bits = [f"{record['path']}", f"sha256 {record['sha256'][:12]}"]
    if record["width"] is not None:
        bits.append(f"{record['width']}x{record['height']} depth {record['depth']} ({record['pix_fmt']})")
    if record["fps"]:
        bits.append(f"{record['fps']} fps")
    if record["frames"] is not None:
        bits.append(f"{record['frames']} frames")
    return ", ".join(bits)


def cmd_show(args):
    root = packet_root(args)
    path = packet_path(handoff_dir(root), args.packet)
    packet = load_packet(path)
    lines = [f"id:       {packet['id']}", f"version:  {packet['schema_version']}, revision {packet.get('revision')}",
             f"status:   {packet['status']}", f"route:    {packet['from']} -> {packet['to']}",
             f"created:  {packet['created']}", f"updated:  {packet['updated']}",
             f"task:     {packet['task']}", f"inputs:   {len(packet['inputs'])}"]
    lines += [f"  [{i}] {describe_record(r)}" for i, r in enumerate(packet["inputs"])]
    stages = packet["request"]["stages"]
    lines.append(f"stages:   {len(stages)}")
    for i, st in enumerate(stages):
        extra = "".join(f" --param {k}={v}" for k, v in st["params"].items())
        extra += "".join(f" --layer {k}={v}" for k, v in st["layers"].items())
        lines.append(f"  [{i}] {st['stage']}{extra}")
    ex = packet["expect"]
    lines.append(f"expect:   frames {ex['frames']}, size {ex['size']}, determinism {ex['determinism']}, "
                 f"tolerance {ex['tolerance']['max_abs_diff']} / {ex['tolerance']['max_diff_fraction']}")
    result = packet["result"]
    if result is None:
        lines.append("result:   none yet")
    else:
        lines.append(f"result:   {len(result['outputs'])} output(s), {len(result['commands'])} command(s), "
                     f"scratch {result['scratch']}")
        lines += [f"  output[{i}] {describe_record(r)}" for i, r in enumerate(result["outputs"])]
        for i, c in enumerate(result["commands"]):
            lines.append(f"  command[{i}] exit {c['exit_code']}, {c['seconds']}s: " + " ".join(c["argv"]))
    lines.append(f"notes:    {len(packet['notes'])}")
    lines += [f"  {n['at']} {n['by']}: {n['text']}" for n in packet["notes"]]
    return {"path": path, "packet": packet}, lines


def cmd_accept(args):
    root = local_root(args)
    path = packet_path(handoff_dir(root), args.packet)
    packet = load_packet(path)
    require_v2(packet)
    check_revision(packet, args.revision)
    read_revision = packet["revision"]
    status = packet["status"]
    if status == "accepted" and not args.notes:
        return ({"id": packet["id"], "path": path, "status": status, "revision": read_revision,
                 "changed": False, "packet": packet}, [f"{packet['id']}: already accepted"])
    if status not in ("new", "accepted"):
        raise usage_error(f"packet {packet['id']} is {status}; only a new packet can be accepted")
    by = actor(args, root, packet["to"])
    if status == "new":
        packet["status"] = "accepted"
        add_note(packet, by, "accepted")
    for text in args.notes or []:
        add_note(packet, by, text)
    commit(path, packet, read_revision, args.lock_wait)
    return ({"id": packet["id"], "path": path, "status": packet["status"], "revision": packet["revision"],
             "changed": True, "packet": packet}, [f"{packet['id']}: {packet['status']} (revision {packet['revision']})"])


def cmd_done(args):
    root = local_root(args)
    path = packet_path(handoff_dir(root), args.packet)
    packet = load_packet(path)
    require_v2(packet)
    check_revision(packet, args.revision)
    read_revision = packet["revision"]
    if packet["status"] != "accepted":
        raise usage_error(f"packet {packet['id']} is {packet['status']}; done needs an accepted packet "
                          f"(run accept first)")
    defaults, items = split_events(getattr(args, "events", None), ("output",))
    parsed = []
    for item in items:
        if item["kind"] == "command_spec":
            parsed += [(False, spec) for spec in parse_specs(item["value"], recorded=True)]
        elif item["kind"] == "run_spec":
            parsed += [(True, spec) for spec in parse_specs(item["value"], recorded=False)]
    planned = [(run_it,) + spec_record(spec, root) for run_it, spec in parsed]
    scratch = None
    if args.scratch:
        scratch = to_rel(resolve(args.scratch, root), root, "--scratch")
    outputs_given = []
    for item in items:
        if item["kind"] == "output":
            real = resolve(item["value"], root)
            to_rel(real, root, "--output")
            outputs_given.append((real, item["geom"]))
    commands = []
    for run_it, record, cwd, stdin, stdout in planned:
        if run_it:
            ran = execute(record["argv"], stdin, stdout, cwd, args.timeout)
            record["exit_code"], record["seconds"] = ran["exit_code"], ran["seconds"]
        commands.append(record)
    failing = bool(args.failed) or any(c["exit_code"] != 0 for c in commands)

    inherited = next(({k: r[k] for k in ("width", "height", "depth", "fps")}
                      for r in packet["inputs"] if r["width"] is not None), {})
    outputs = []
    for real, geom in outputs_given:
        if not os.path.isfile(real):
            if failing:
                continue
            raise usage_error(f"no such output file: {real}")
        flags = dict(defaults)
        flags.update(geom)
        if any(k in flags for k in ("width", "height", "depth")):
            record = file_record(real, flags, root)
        else:
            guess = dict(inherited)
            guess.update(flags)
            try:
                record = file_record(real, guess, root)
            except HandoffError:
                record = file_record(real, {"fps": flags.get("fps")}, root)
        record["path"] = to_rel(real, root, "--output")
        outputs.append(record)
    by = actor(args, root, packet["to"])
    packet["result"] = {"outputs": outputs, "commands": commands, "scratch": scratch}
    packet["status"] = "failed" if failing else "done"
    add_note(packet, by, packet["status"])
    for text in args.notes or []:
        add_note(packet, by, text)
    commit(path, packet, read_revision, args.lock_wait)
    return ({"id": packet["id"], "path": path, "status": packet["status"], "revision": packet["revision"],
             "changed": True, "packet": packet},
            [f"{packet['id']}: {packet['status']} ({len(outputs)} output(s), {len(commands)} command(s))"])


def check_lines(checks):
    return [("[ok]   " if c["ok"] else "[fail] ") + c["detail"] for c in checks]


def cmd_verify(args):
    root = packet_root(args)
    path = packet_path(handoff_dir(root), args.packet)
    packet = load_packet(path)
    check_revision(packet, args.revision)
    read_revision = packet.get("revision")
    if args.replay and packet["result"] is None:
        raise usage_error(f"packet {packet['id']} is {packet['status']} and has no recorded commands to replay")
    trusted = None
    if args.replay and not args.trust:
        trusted = trusted_commands(args, root)
        check_replay_programs(packet, trusted)
    checks, report = verify_checks(packet, root, args.replay, args.timeout, trusted)
    problems = [c for c in checks if not c["ok"]]
    before = packet["status"]
    data = {"id": packet["id"], "path": path, "status_before": before, "status": before, "changed": False,
            "revision": read_revision, "checks": checks, "problems": [c["detail"] for c in problems],
            "replay": report, "scratch": (packet["result"] or {}).get("scratch")}
    lines = check_lines(checks)
    if problems:
        first = problems[0]
        stage = "replay" if first["name"].startswith("replay-") else "verify"
        tail = []
        if report is not None:
            for c in report["commands"]:
                if c["exit_code"] != c["recorded_exit_code"]:
                    tail = c.get("stderr_tail", [])
                    break
        raise HandoffError(first["code"] or "bad_input",
                           f"{len(problems)} verification problem(s) in {packet['id']}: {first['detail']}",
                           stage=stage, child_exit=first.get("exit_code"), tail=tail, extra=data, lines=lines)
    if before == "done" and not args.no_write:
        if packet["schema_version"] != SCHEMA_VERSION:
            lines.append(f"{packet['id']}: schema_version {packet['schema_version']} is read only; "
                         f"run migrate to record verified")
        else:
            by = actor(args, root, packet["from"])
            packet["status"] = "verified"
            add_note(packet, by, "verified" + (" with replay" if args.replay else ""))
            for text in args.notes or []:
                add_note(packet, by, text)
            commit(path, packet, read_revision, args.lock_wait)
            data.update(status="verified", changed=True, revision=packet["revision"])
    lines.append(f"{packet['id']}: {data['status']}" + ("" if checks else " (nothing to check)"))
    return data, lines


def migrate_packet(old, root, by):
    """a schema_version 1 packet as version 2: revision 1, every path relative to the root.
    an input, output, layer, stdin or stdout outside the root is refused (bad_input); a scratch
    folder outside the root is dropped with a note."""
    outside = []

    def rel(path, base=None):
        real = resolve(path, base or root)
        found = rel_or_none(real, root)
        if found is None:
            outside.append(path)
        return found

    def record(rec):
        out = dict(rec)
        out["path"] = rel(rec["path"])
        return out

    request = {"stages": [{"stage": st["stage"], "params": dict(st["params"]),
                           "layers": {k: rel(v) for k, v in st["layers"].items()}}
                          for st in old["request"]["stages"]]}
    result, dropped = None, None
    if old["result"] is not None:
        commands = []
        for cmd in old["result"]["commands"]:
            cwd = resolve(cmd["cwd"], root) if cmd["cwd"] else root
            new = dict(cmd)
            new["cwd"] = None if same_path(cwd, root) else rel(cwd)
            new["stdin"] = rel(cmd["stdin"], cwd) if cmd["stdin"] else None
            new["stdout"] = rel(cmd["stdout"], cwd) if cmd["stdout"] else None
            commands.append(new)
        scratch = old["result"]["scratch"]
        scratch_rel = rel_or_none(resolve(scratch, root), root) if scratch else None
        if scratch and scratch_rel is None:
            dropped = scratch
        result = {"outputs": [record(r) for r in old["result"]["outputs"]], "commands": commands,
                  "scratch": scratch_rel}
    inputs = [record(r) for r in old["inputs"]]
    if outside:
        raise HandoffError("bad_input", f"packet {old['id']} names {len(outside)} path(s) outside {root}, "
                                        f"first {outside[0]}; move the files inside and edit the packet, "
                                        f"or keep it as version 1", extra={"outside": outside})
    packet = {"schema_version": SCHEMA_VERSION, "id": old["id"], "revision": 1, "created": old["created"],
              "updated": old["updated"], "from": old["from"], "to": old["to"], "task": old["task"],
              "status": old["status"], "inputs": inputs, "request": request, "expect": old["expect"],
              "result": result, "notes": list(old["notes"])}
    add_note(packet, by, "migrated from schema_version 1")
    if dropped:
        add_note(packet, by, f"scratch {dropped} lies outside the repository and was dropped")
    touch(packet)
    return packet


def cmd_migrate(args):
    root = local_root(args)
    directory = handoff_dir(root)
    if args.all == bool(args.packet):
        raise usage_error("migrate takes one packet id or --all")
    paths = [os.path.join(directory, n) for n in list_packet_files(directory)] if args.all \
        else [packet_path(directory, args.packet)]
    migrated, unchanged, lines = [], [], []
    for path in paths:
        packet = load_packet(path)
        if packet["schema_version"] == SCHEMA_VERSION:
            unchanged.append(packet["id"])
            lines.append(f"{packet['id']}: already schema_version {SCHEMA_VERSION}")
            continue
        new = migrate_packet(packet, root, actor(args, root, packet["to"]))
        with PacketLock(path, args.lock_wait):
            if load_packet(path) != packet:
                raise HandoffError("bad_input", f"revision conflict on {packet['id']}: the file changed during migrate")
            save_packet(path, new)
        migrated.append(packet["id"])
        lines.append(f"{packet['id']}: migrated to schema_version {SCHEMA_VERSION}")
    lines.append(f"migrate: {len(migrated)} migrated, {len(unchanged)} already current")
    return {"migrated": migrated, "unchanged": unchanged, "changed": bool(migrated)}, lines


def cmd_stages(args):
    root = local_root(args)
    listed, problems = manifest_files(args)
    if not args.no_registry:
        found, more = registry_manifests(args, root)
        listed += found
        problems += more
    providers, rows = [], []
    for manifest, source in listed:
        prov = manifest["provider"]
        providers.append({"id": prov["id"], "version": prov["version"], "command": prov["command"],
                          "contract": manifest["contract"], "source": source, "stages": len(manifest["stages"])})
        if contract_major(manifest["contract"]) not in KNOWN_MAJORS:
            problems.append(f"{source}: contract {manifest['contract']} is not frames/1 or frames/2; "
                            f"its stages are refused")
            continue
        for st in manifest["stages"]:
            if st["geometry"] not in GEOMETRY_VALUES:
                problems.append(f"{source}: stage {st['id']} has geometry {st['geometry']}, which this "
                                f"tool does not know; refused")
                continue
            row = {"provider": prov["id"], "contract": manifest["contract"], "id": st["id"], "kind": st["kind"],
                   "spool": st["spool"], "geometry": st["geometry"], "depths": st["depths"],
                   "hidden": bool(st.get("hidden", False)), "params": st["params"], "layers": st["layers"],
                   "stage": stage_string(prov["command"], st["id"])}
            rows.append(row)
    lines = [f"{r['provider']:<10} {r['id']:<20} {r['kind']:<8} {r['geometry']:<8} {r['stage']}" for r in rows]
    lines += [f"problem: {p}" for p in problems]
    lines.append(f"{len(rows)} stage(s) from {len(providers)} provider(s)")
    return {"providers": providers, "stages": rows, "problems": problems}, lines


def shared_files(root, patterns):
    """the relative paths under `root` that the registry's shared patterns name (one folder deep)."""
    found = set()
    for pattern in patterns:
        folder, _, name = pattern.rpartition("/")
        base = resolve(folder, root) if folder else root
        if not any(c in name for c in "*?["):
            if os.path.isfile(os.path.join(base, name)):
                found.add(pattern)
            continue
        if not os.path.isdir(base):
            continue
        for entry in sorted(os.listdir(base)):
            if not entry.startswith(".") and fnmatch.fnmatchcase(entry, name) \
                    and os.path.isfile(os.path.join(base, entry)):
                found.add(f"{folder}/{entry}" if folder else entry)
    return sorted(found)


def cmd_sync(args):
    root = local_root(args)
    registry = load_registry()
    ids = tool_ids(registry)
    me = detect_self(root, args.self_name)
    if args.to:
        if args.to not in ids:
            raise usage_error(f"--to must name a registered tool ({', '.join(ids)}), got `{args.to}`")
        if args.to == me:
            raise usage_error(f"--to names this repository ({me})")
        targets = [args.to]
    else:
        if me is None:
            raise usage_error(f"cannot tell which registered tool {root} is; pass --self ID or --to ID")
        targets = [i for i in ids if i != me]
    patterns = registry["shared"]
    if args.paths:
        files = []
        for given in args.paths:
            rel = given.replace("\\", "/")
            if not REL_RE.match(rel) or not any(fnmatch.fnmatchcase(rel, p) for p in patterns):
                raise usage_error(f"{given} is not a shared file (the shared list of contracts/tool.json)")
            if not os.path.isfile(resolve(rel, root)):
                raise usage_error(f"no such shared file here: {resolve(rel, root)}")
            files.append(rel)
    else:
        files = shared_files(root, patterns)
    results = []
    for peer in targets:
        peer_root = locate_peer(args, peer, root)
        if same_path(peer_root, root):
            raise usage_error(f"the {peer} repository is this repository: {root}")
        for rel in files:
            src, dst = resolve(rel, root), resolve(rel, peer_root)
            ours = hash_file(src)[0]
            theirs = file_digest_or_none(dst)
            if theirs == ours:
                status = "same"
            elif args.check:
                status = "missing" if theirs is None else "differs"
            else:
                copy_atomic(src, dst)
                theirs = file_digest_or_none(dst)
                status = "copied" if theirs == ours else "failed"
            results.append({"peer": peer, "peer_root": peer_root, "path": rel, "status": status,
                            "sha256": ours, "peer_sha256": theirs})
    lines = [f"{r['status']:<8} {r['peer']}: {r['path']} {r['sha256'][:16]}" for r in results]
    data = {"files": results, "changed": any(r["status"] == "copied" for r in results)}
    bad = [r for r in results if r["status"] in ("differs", "missing", "failed")]
    if bad:
        raise HandoffError("bad_input" if args.check else "stage_failed",
                           f"{len(bad)} shared file(s) are not byte equal, first {bad[0]['path']} in {bad[0]['peer']}",
                           stage="sync", extra=data, lines=lines)
    copied = sum(r["status"] == "copied" for r in results)
    lines.append(f"sync: {len(results)} file(s) byte equal by sha256, {copied} copied")
    return data, lines


def contract_checks(root):
    """every contracts/*.json parses; every *.schema.json uses only the validator's keywords and
    validates its own examples; tool.json validates against tool.schema.json; the error table of
    error.schema.json equals the exit codes of this script."""
    checks = []
    folder = os.path.join(root, "contracts")

    def add(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    names = sorted(n for n in os.listdir(folder) if n.endswith(".json")) if os.path.isdir(folder) else []
    if not names:
        add("contracts", False, f"no JSON files in {folder}")
    docs = {}
    for name in names:
        try:
            docs[name] = read_json(os.path.join(folder, name), name)
        except HandoffError as exc:
            add(name, False, exc.message)
    for name, data in docs.items():
        if not name.endswith(".schema.json"):
            continue
        try:
            check_schema(data)
        except SchemaError as exc:
            add(name, False, f"{name}: {exc}")
            continue
        examples = data.get("examples", [])
        bad = []
        for i, example in enumerate(examples):
            found = validate(example, data)
            if found:
                bad.append(f"examples[{i}]: {found[0]}")
        add(name, examples and not bad, f"{name}: {len(examples)} example(s) valid" if examples and not bad else
            f"{name}: " + ("; ".join(bad[:2]) if bad else "no examples"))
    if "tool.json" in docs and "tool.schema.json" in docs:
        found = validate(docs["tool.json"], docs["tool.schema.json"])
        add("tool.json", not found, "tool.json: valid registry, tools " +
            ", ".join(t["id"] for t in docs["tool.json"].get("tools", [])) if not found else "; ".join(found[:2]))
    if "error.schema.json" in docs:
        rows = docs["error.schema.json"].get("x-codes", [])
        table = {r.get("code"): r.get("exit", {}).get("frames/2") for r in rows if isinstance(r, dict)}
        enum = docs["error.schema.json"]["properties"]["error"]["properties"]["code"].get("enum", [])
        ok = table == EXIT and sorted(enum) == sorted(EXIT)
        add("error-table", ok, f"x-codes: {len(rows)} codes, frames/2 exits equal this script's" if ok else
            f"x-codes {table} and enum {enum} differ from this script's {EXIT}")
    return checks


def cmd_contracts(args):
    root = local_root(args)
    checks = contract_checks(root)
    lines = check_lines(checks)
    failed = [c for c in checks if not c["ok"]]
    data = {"root": root, "checks": checks}
    if failed:
        raise HandoffError("bad_input", f"{len(failed)} contract check(s) failed: {failed[0]['detail']}",
                           stage="contracts", extra=data, lines=lines)
    lines.append(f"contracts: {len(checks)} check(s) passed")
    return data, lines


def cmd_doctor(args):
    root = local_root(args)
    registry = load_registry()
    me = detect_self(root, args.self_name)
    checks = []

    def add(name, ok, detail, required=False):
        checks.append({"name": name, "ok": bool(ok), "detail": detail, "required": required})

    add("repository", me is not None,
        f"this is {me} at {root}" if me else f"no registered tool's markers match {root}; pass --self")
    dirs = peer_dirs(args)
    peers = {}
    others = [t for t in registry["tools"] if t["id"] != me]
    for tool in others:
        path, why = find_peer(tool["id"], root, dirs.get(tool["id"]))
        add(f"peer:{tool['id']}", path is not None, f"{tool['id']} at {path}" if path else why)
        if path:
            peers[tool["id"]] = path
    add("peers", bool(peers), f"{len(peers)} of {len(others)} registered peer(s) found: " +
        (", ".join(peers) or "none"), required=True)

    for name in ("ffmpeg", "ffprobe"):
        path, reason = resolve_tool(name, TOOL_ENVS[name])
        version = first_line([path, "-version"]) if path else ""
        add(name, path is not None, f"{path}: {version}" if path else reason, required=True)

    roots = dict(peers)
    if me:
        roots[me] = root
    for tool in registry["tools"]:
        tool_root = roots.get(tool["id"])
        for binary in tool["binaries"]:
            path, reason = tool_binary(tool, binary, tool_root)
            add(f"binary:{binary['name']}", path is not None,
                f"{path}: {first_line([path] + binary['version_args'])}" if path else reason)
            if path is None:
                continue
            for provider in tool["providers"]:
                if provider["binary"] != binary["name"]:
                    continue
                contract, note = reported_contract([path] + provider["prefix"] + provider["list_args"],
                                                   provider["adapter"])
                if contract is None:
                    add(f"contract:{binary['name']}", True, f"not reported ({note})")
                else:
                    major = contract_major(contract)
                    add(f"contract:{binary['name']}", major in KNOWN_MAJORS,
                        f"reports {contract}" + (f" ({note})" if note else "") +
                        ("" if major in KNOWN_MAJORS else ", which this tool does not know"))

    found = contract_checks(root)
    bad = [c for c in found if not c["ok"]]
    add("contracts-valid", not bad, f"{len(found)} contract check(s) passed" if not bad else bad[0]["detail"],
        required=True)
    files = shared_files(root, registry["shared"])
    contract_files = [f for f in files if f.startswith("contracts/") and f.count("/") == 1 and f.endswith(".json")]
    other_files = [f for f in files if f not in contract_files]
    for peer, peer_root in peers.items():
        for label, group, required in (("contracts-equal", contract_files, True), ("shared-equal", other_files, False)):
            differ = [f for f in group if file_digest_or_none(resolve(f, root)) != file_digest_or_none(resolve(f, peer_root))]
            add(f"{label}:{peer}", not differ,
                f"{len(group)} file(s) byte equal with {peer}" if not differ else
                f"{len(differ)} of {len(group)} file(s) differ from {peer}: {', '.join(differ[:4])}", required)

    ignore = os.path.join(root, ".gitignore")
    try:
        with open(ignore, encoding="utf-8", errors="replace") as handle:
            ignored = any(line.strip() in ("handoff/", "/handoff/", "handoff", "/handoff") for line in handle)
    except OSError:
        ignored = False
    add("gitignore", ignored, "handoff/ is git-ignored" if ignored else "handoff/ is not listed in .gitignore")

    conformance_exit = None
    if args.conformance:
        script = os.path.join(root, "tools", "conformance.py")
        if not os.path.isfile(script):
            add("conformance", True, "not run: tools/conformance.py is not present")
        else:
            ran = execute([sys.executable, "-I", script, "--quick"], None, None, root, args.timeout)
            conformance_exit = ran["exit_code"]
            add("conformance", ran["exit_code"] == 0,
                f"tools/conformance.py --quick exited {ran['exit_code']} in {ran['seconds']}s"
                + (": " + ran["stderr_tail"][-1] if ran["stderr_tail"] and ran["exit_code"] else ""),
                required=True)

    lines = []
    for c in checks:
        mark = "[ok]   " if c["ok"] else ("[FAIL] " if c["required"] else "[warn] ")
        lines.append(f"{mark}{c['name']}: {c['detail']}")
    failed = [c for c in checks if c["required"] and not c["ok"]]
    data = {"self": me, "root": root, "peers": peers, "python": sys.version.split()[0],
            "platform": sys.platform, "checks": checks}
    if failed:
        first = failed[0]["name"]
        code = "stage_failed" if first == "conformance" else (
            "bad_input" if first.startswith("contracts-") else "tool_missing")
        raise HandoffError(code, f"{len(failed)} required check(s) failed: " + ", ".join(c["name"] for c in failed),
                           stage="doctor", child_exit=conformance_exit if code == "stage_failed" else None,
                           extra=data, lines=lines)
    lines.append("doctor: all required checks passed")
    return data, lines


# ----------------------------------------------------------------------------
# command line
# ----------------------------------------------------------------------------

class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise HandoffError("usage", f"{self.prog}: {message}", stage="args")


EPILOG = """\
exit codes (contracts/error.schema.json, frames/2): 0 ok, 1 stage_failed, 2 usage, 3 unsupported,
4 bad_input, broken_pipe and partial_frame, 5 stage_timeout, 6 tool_missing, 130 interrupted.
json: every result has "contract": "frames/2"; a failure is {"ok": false, "error": {...}}.
"""


def add_geometry(parser, what):
    parser.add_argument("--width", type=positive_int, action=Event, dest="width", metavar="W",
                        help=f"frame width of {what}")
    parser.add_argument("--height", type=positive_int, action=Event, dest="height", metavar="H",
                        help=f"frame height of {what}")
    parser.add_argument("--depth", type=int, choices=sorted(DEPTH_PIX_FMT), action=Event, dest="depth",
                        metavar="D", help=f"bits per channel of {what}: 8 rgba, 16 rgba64le, 32 gbrapf32le")
    parser.add_argument("--fps", type=fps_arg, action=Event, dest="fps", metavar="N[/D]",
                        help=f"exact frame rate of {what}, stored as N/D")


def build_parser():
    common = Parser(add_help=False)
    common.add_argument("--root", metavar="DIR", help="repository root whose handoff/ folder is used (default: the "
                        "folder above this script's); for new it is the folder the packet is written into")
    common.add_argument("--self", dest="self_name", metavar="ID",
                        help="which registered tool the root is (default: HANDOFF_SELF, then the markers)")
    common.add_argument("--peer-dir", action="append", dest="peer_dir", metavar="ID=DIR",
                        help="the folder of peer ID, repeatable (default: HANDOFF_<ID>_DIR, then the one "
                             "sibling folder holding the peer's markers)")
    common.add_argument("--lock-wait", type=wait_arg, default=LOCK_WAIT, metavar="SECONDS",
                        help="how long a write waits for handoff/<id>.lock (default 10)")
    common.add_argument("--json", action="store_true", dest="as_json", help="print the result as JSON")

    def note_arg(p):
        p.add_argument("--note", action="append", dest="notes", metavar="TEXT", help="add a note, repeatable")

    def revision_arg(p):
        p.add_argument("--revision", type=positive_int, metavar="N",
                       help="compare and swap: refuse (bad_input) unless the packet holds revision N")

    def peer_args(p, what):
        p.add_argument("--other", action="store_true", help=f"{what} the only registered peer's folder")
        p.add_argument("--peer", metavar="ID", help=f"{what} the folder of peer ID")

    parser = Parser(prog="handoff.py", description="hand a task from one registered tool to another as one "
                    "JSON packet, list the stages of every provider and keep the shared files in sync",
                    epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", metavar="COMMAND", parser_class=Parser)

    p = sub.add_parser("new", parents=[common], allow_abbrev=False,
                       help="write a new packet into the receiving repository's handoff/ folder",
                       description="write handoff/<id>.json into the repository of the tool named by --to (found "
                                   "as a peer unless --root is given). Geometry flags after an --input describe "
                                   "that input; before the first --input they are defaults for all of them. A "
                                   "--param or --layer belongs to the --stage before it. An input or layer outside "
                                   "the receiving repository is copied into handoff/<id>/.")
    p.add_argument("--from", dest="from_name", metavar="ID", help="the sending tool (default: this root's tool)")
    p.add_argument("--to", dest="to_name", required=True, metavar="ID", help="the receiving tool, which keeps the packet")
    p.add_argument("--task", required=True, metavar="TEXT", help="what the receiver is asked to do")
    p.add_argument("--id", dest="packet_id", metavar="ID",
                   help="use this id; repeating new with the same id and arguments changes nothing")
    p.add_argument("--input", action=Event, dest="input", metavar="PATH",
                   help="an input file, repeatable; with width, height and depth it is raw frames and its "
                        "frame count is computed")
    p.add_argument("--file", action=Event, dest="file", metavar="PATH",
                   help="an input file that is never raw frames (hash only), repeatable")
    add_geometry(p, "the input before it")
    p.add_argument("--stage", action=Event, dest="stage", metavar="ID", help="a stage of the chain, repeatable, in order")
    p.add_argument("--param", action=Event, dest="param", metavar="K=V", help="a parameter of the stage before it")
    p.add_argument("--layer", action=Event, dest="layer", metavar="ID=PATH", help="a layer of the stage before it")
    p.add_argument("--expect-frames", type=count_or_any, metavar="N|any",
                   help="frames every output must hold (default: the frames of the first raw input)")
    p.add_argument("--expect-size", type=count_or_any, metavar="BYTES|any",
                   help="bytes of every output (default: expected frames times the first raw input's frame size)")
    p.add_argument("--determinism", choices=("exact", "tolerance", "none"), metavar="RULE",
                   help="exact, tolerance or none (default: exact, or tolerance when a tolerance is given)")
    p.add_argument("--tolerance-abs", type=nonneg_number, metavar="X", help="largest allowed difference of one sample")
    p.add_argument("--tolerance-fraction", type=fraction_arg, metavar="F", help="largest allowed fraction of differing samples")
    note_arg(p)

    p = sub.add_parser("list", parents=[common], allow_abbrev=False, help="list the packets of a handoff/ folder")
    peer_args(p, "list")
    p.add_argument("--status", choices=STATUSES, metavar="STATUS", help="only packets with this status")
    p.add_argument("--from", dest="from_filter", metavar="ID", help="only packets sent by this tool")
    p.add_argument("--to", dest="to_filter", metavar="ID", help="only packets addressed to this tool")

    p = sub.add_parser("show", parents=[common], allow_abbrev=False, help="print one packet")
    p.add_argument("packet", metavar="ID", help="a packet id, a unique prefix of one, or a .json path")
    peer_args(p, "read from")

    p = sub.add_parser("accept", parents=[common], allow_abbrev=False,
                       help="take a new packet; repeating it on an accepted packet changes nothing")
    p.add_argument("packet", metavar="ID", help="a packet id or a unique prefix of one")
    revision_arg(p)
    note_arg(p)

    p = sub.add_parser("done", parents=[common], allow_abbrev=False,
                       help="record the result of an accepted packet",
                       description="record outputs (hashed here) and commands in the packet. --run SPEC runs a "
                                   "command now and records its exit code and time; --command SPEC records one "
                                   "that already ran. SPEC is JSON (an object or a list) or @file with argv and "
                                   "optionally stdin, stdout, cwd (relative to the root, inside it); --command adds "
                                   "exit_code and seconds. Geometry flags after an --output describe it; without "
                                   "them it inherits the first raw input's geometry when it is a whole number of frames.")
    p.add_argument("packet", metavar="ID", help="a packet id or a unique prefix of one")
    p.add_argument("--output", action=Event, dest="output", metavar="PATH",
                   help="an output file inside the root, repeatable")
    add_geometry(p, "the output before it")
    p.add_argument("--command", action=Event, dest="command_spec", metavar="SPEC", help="record a command that ran")
    p.add_argument("--run", action=Event, dest="run_spec", metavar="SPEC", help="run a command and record it")
    p.add_argument("--failed", action="store_true", help="mark the packet failed (a non-zero exit code does too)")
    p.add_argument("--scratch", metavar="DIR", help="the folder inside the root the run kept its intermediate files in")
    p.add_argument("--timeout", type=seconds_arg, default=DEFAULT_TIMEOUT, metavar="SECONDS",
                   help="limit for each --run command (default 600)")
    revision_arg(p)
    note_arg(p)

    p = sub.add_parser("verify", parents=[common], allow_abbrev=False,
                       help="recompute hashes, check frames and size against expect, optionally replay",
                       description="check the hashes of the inputs and recorded outputs, the frame counts, "
                                   "expect.frames and expect.size, and the recorded exit codes. A done packet "
                                   "that passes becomes verified. --replay also runs the recorded commands "
                                   "again and compares the bytes; without --trust only ffmpeg, ffprobe and "
                                   "registered providers are replayed, through the local programs.")
    p.add_argument("packet", metavar="ID", help="a packet id, a unique prefix of one, or a .json path")
    peer_args(p, "verify a packet in")
    p.add_argument("--replay", action="store_true", help="run the recorded commands again and compare the outputs")
    p.add_argument("--trust", action="store_true", help="replay any recorded program exactly as recorded")
    p.add_argument("--provider", action="append", dest="providers", metavar="FILE",
                   help="a *.stage.json manifest whose command is trusted for replay, repeatable")
    p.add_argument("--no-write", action="store_true", help="do not move a passing packet to verified")
    p.add_argument("--timeout", type=seconds_arg, default=DEFAULT_TIMEOUT, metavar="SECONDS",
                   help="limit for each replayed command (default 600)")
    revision_arg(p)
    note_arg(p)

    p = sub.add_parser("migrate", parents=[common], allow_abbrev=False,
                       help="rewrite schema_version 1 packets as version 2",
                       description="rewrite a schema_version 1 packet as version 2: revision 1 and every path "
                                   "relative to the root. A packet with an input, output, layer, stdin or stdout "
                                   "outside the root is refused; repeating migrate changes nothing.")
    p.add_argument("packet", nargs="?", metavar="ID", help="a packet id or a unique prefix of one")
    p.add_argument("--all", action="store_true", help="every packet of the handoff/ folder")

    p = sub.add_parser("stages", parents=[common], allow_abbrev=False,
                       help="list every stage of every provider with a ready --stage command",
                       description="providers: --provider files, the *.stage.json files of the HANDOFF_PROVIDERS "
                                   "folders, and the providers of every registered tool that is found (its list "
                                   "--json, or vfx list --json through the vfx-list adapter). A manifest of an "
                                   "unknown contract major and a stage of an unknown geometry are refused.")
    p.add_argument("--provider", action="append", dest="providers", metavar="FILE", help="a *.stage.json file, repeatable")
    p.add_argument("--no-registry", action="store_true", help="skip the providers of the registered tools")

    p = sub.add_parser("sync", parents=[common], allow_abbrev=False,
                       help="copy the shared files byte for byte to the peers",
                       description="copy the files named by the shared list of contracts/tool.json from this "
                                   "root to every peer (or --to ID) and confirm byte equality by SHA-256. --check "
                                   "only compares (bad_input when a file differs).")
    p.add_argument("paths", nargs="*", metavar="PATH", help="only these shared files (relative POSIX paths)")
    p.add_argument("--to", metavar="ID", help="only this peer")
    p.add_argument("--check", action="store_true", help="compare only, write nothing")

    sub.add_parser("contracts", parents=[common], allow_abbrev=False,
                   help="check that every contract file parses and every schema validates its own examples")

    p = sub.add_parser("doctor", parents=[common], allow_abbrev=False,
                       help="check the peers, ffmpeg, ffprobe, the binaries, the contracts and the shared files",
                       description="one report: the peers, ffmpeg and ffprobe, every registry binary with its "
                                   "version and contract, the contract files, and whether the shared files are "
                                   "byte equal with each peer. Exit 0 only when a peer, ffmpeg, ffprobe, valid "
                                   "contracts and contract files equal with every found peer are present; the rest "
                                   "is advisory. Binaries are looked up in their environment variable, "
                                   "$CARGO_TARGET_DIR/release, the registry dirs under the tool's root and PATH.")
    p.add_argument("--conformance", action="store_true",
                   help="also run tools/conformance.py --quick when that file exists")
    p.add_argument("--timeout", type=seconds_arg, default=DEFAULT_TIMEOUT, metavar="SECONDS",
                   help="limit for the conformance run (default 600)")
    return parser


HANDLERS = {"new": cmd_new, "list": cmd_list, "show": cmd_show, "accept": cmd_accept, "done": cmd_done,
            "verify": cmd_verify, "migrate": cmd_migrate, "stages": cmd_stages, "sync": cmd_sync,
            "contracts": cmd_contracts, "doctor": cmd_doctor}


def emit(document):
    print(json.dumps(document, indent=2, ensure_ascii=False))


def fail(err, as_json, command):
    if err.stage is None:
        err.stage = command
    if as_json:
        document = {"ok": False, "contract": CONTRACT, "error": err.to_object()}
        document.update(err.extra)
        emit(document)
    else:
        for line in err.lines:
            print(line)
    print(err.text_line(), file=sys.stderr)
    return err.exit_code


def main(argv=None):
    """run one command line and return the exit code (the script exits with it)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in argv
    command = None
    try:
        args = build_parser().parse_args(argv)
        command = args.command
        if command is None:
            raise HandoffError("usage", "a command is required: " + ", ".join(HANDLERS), stage="args")
        data, lines = HANDLERS[command](args)
        if args.as_json:
            document = {"ok": True, "contract": CONTRACT}
            document.update(data)
            emit(document)
        else:
            for line in lines:
                print(line)
        return EXIT_OK
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else EXIT["usage"]
    except HandoffError as err:
        return fail(err, as_json, command)
    except OSError as exc:
        return fail(HandoffError("stage_failed", f"I/O error: {exc.strerror or exc}"), as_json, command)
    except KeyboardInterrupt:
        return fail(HandoffError("interrupted", "interrupted"), as_json, command)


if __name__ == "__main__":
    # a pipe gets the ansi code page; a path it cannot spell must not crash the report line
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass
    sys.exit(main())
