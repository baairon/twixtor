#!/usr/bin/env python3
"""tests for tools/handoff.py and the contracts/ files. stdlib unittest, no network.

    python -I tools/test_handoff.py

the suite makes its own scratch folders with tempfile (set TEMP to choose where) and removes them.
every test that builds a layout runs with the HANDOFF_* variables removed, so it never looks at a
real repository. the tests of the real peers read the environment the suite started with: they
find a peer through HANDOFF_<ID>_DIR or the sibling folder; when no peer is found they check that
the absence is explained, and with HANDOFF_REQUIRE_PEERS=1 an absent peer is a failure. nothing
is skipped. the same file runs in every repository that follows the convention.
"""

import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# no bytecode next to the tools, here or in the script the suite runs
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, HERE)
import handoff  # noqa: E402

REAL_ENV = dict(os.environ)
REQUIRE_PEERS = REAL_ENV.get("HANDOFF_REQUIRE_PEERS") == "1"
REGISTRY = handoff.load_registry()
DOCS_REL = os.path.join("docs", "handoff.md")

# commands the tests record and replay: invert every byte, add k to every byte, write a fixed file
INVERT = "import sys; d = sys.stdin.buffer.read(); sys.stdout.buffer.write(bytes(255 - b for b in d))"
SHIFT = ("import sys; d = sys.stdin.buffer.read(); k = int(sys.argv[1]); "
         "sys.stdout.buffer.write(bytes(min(255, b + k) for b in d))")
WRITE_TO_ARG = "import sys; open(sys.argv[1].split('=', 1)[1], 'wb').write(b'x' * 16)"
NOISE = "import os, sys; sys.stdout.buffer.write(os.urandom(96))"
INVERT_SCRIPT = "import sys\nd = sys.stdin.buffer.read()\nsys.stdout.buffer.write(bytes(255 - b for b in d))\n"


def clean_env():
    """the environment without anything that points the script at a real repository or binary."""
    drop = ("TWIXTOR_DIR", "SAPPHIRE_DIR", "SAPPHIRE_PATH", "VFX_PATH", "CARGO_TARGET_DIR")
    return {k: v for k, v in os.environ.items() if not k.startswith("HANDOFF_") and k not in drop}


def run_cli(argv):
    """call handoff.main in this process; returns (exit code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = handoff.main(list(argv))
    return code, out.getvalue(), err.getvalue()


def read_bytes(path):
    with open(path, "rb") as handle:
        return handle.read()


def write_bytes(path, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(data)


def sha(path):
    return hashlib.sha256(read_bytes(path)).hexdigest()


def raw_bytes(frames, width=4, height=2, depth=8):
    """deterministic frame data with every byte at most 250, so adding 1 never clips."""
    size = frames * width * height * handoff.BYTES_PER_PIXEL[depth]
    return bytes((i * 7 + 3) % 251 for i in range(size))


def make_tool(folder, tool_id):
    """a folder that holds the markers of a registered tool."""
    for marker in handoff.tool_entry(tool_id)["markers"]:
        write_bytes(os.path.join(folder, *marker.split("/")), b"")
    return folder


def copy_shared(folder):
    """the shared files of this repository, copied into `folder`."""
    for rel in handoff.shared_files(REPO, REGISTRY["shared"]):
        target = os.path.join(folder, *rel.split("/"))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copyfile(os.path.join(REPO, *rel.split("/")), target)


def files_in(folder):
    return sorted(n for n in os.listdir(folder) if os.path.isfile(os.path.join(folder, n)))


class Clock:
    """a now_iso that moves one second per call."""

    def __init__(self):
        self.n = 0

    def __call__(self):
        self.n += 1
        return f"2026-10-26T12:{self.n // 60:02d}:{self.n % 60:02d}Z"


class Case(unittest.TestCase):
    """a scratch folder with a receiving repository at `self.recv` and an input outside it."""

    def setUp(self):
        patcher = mock.patch.dict(os.environ, clean_env(), clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.tmp = tempfile.mkdtemp(prefix="handoff_test_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.recv = os.path.join(self.tmp, "recv")
        os.makedirs(self.recv)
        self.raw = os.path.join(self.tmp, "in.raw")
        write_bytes(self.raw, raw_bytes(3))

    # -- helpers --

    def cli(self, *argv, root=True):
        argv = list(argv)
        if root:
            argv += ["--root", self.recv if root is True else root]
        return run_cli(argv)

    def cli_json(self, *argv, root=True):
        code, out, err = self.cli(*argv, "--json", root=root)
        return code, json.loads(out), err

    def new(self, *extra, task="invert the clip"):
        code, doc, err = self.cli_json(
            "new", "--from", "sapphire", "--to", "twixtor", "--task", task,
            "--input", self.raw, "--width", "4", "--height", "2", "--depth", "8", "--fps", "24",
            "--stage", "selftest-invert", *extra)
        self.assertEqual(code, 0, err)
        return doc["id"]

    def path(self, packet_id, root=None):
        return os.path.join(root or self.recv, "handoff", packet_id + ".json")

    def load(self, packet_id):
        with open(self.path(packet_id), encoding="utf-8") as handle:
            return json.load(handle)

    def edit(self, packet_id, change):
        packet = self.load(packet_id)
        change(packet)
        with open(self.path(packet_id), "w", encoding="utf-8") as handle:
            json.dump(packet, handle, indent=2)
        return packet

    def input_path(self, packet_id, index=0):
        return handoff.resolve(self.load(packet_id)["inputs"][index]["path"], self.recv)

    def spec_file(self, name, spec):
        path = os.path.join(self.tmp, name)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(spec, handle)
        return "@" + path

    def done_packet(self, *new_extra, code=INVERT, argv_extra=(), output=None):
        """a packet run through new, accept and done with one recorded command that inverts the input."""
        packet_id = self.new(*new_extra)
        self.assertEqual(self.cli("accept", packet_id)[0], 0)
        output = output or os.path.join(self.recv, "out.raw")
        spec = self.spec_file("run.json", {"argv": [sys.executable, "-c", code, *argv_extra],
                                           "stdin": self.input_path(packet_id), "stdout": output})
        os.makedirs(os.path.join(self.recv, "scratch"), exist_ok=True)
        status, out, err = self.cli("done", packet_id, "--output", output, "--run", spec,
                                    "--scratch", os.path.join(self.recv, "scratch"))
        self.assertEqual(status, 0, err)
        self.assertEqual(self.load(packet_id)["status"], "done")
        return packet_id, output

    def assert_error(self, doc, code, exit_code=None):
        self.assertEqual(doc["contract"], "frames/2")
        self.assertIs(doc["ok"], False)
        error = doc["error"]
        self.assertEqual(sorted(error), ["code", "exit_code", "message", "stage", "stderr_tail"])
        self.assertEqual(error["code"], code)
        if exit_code is not None:
            self.assertEqual(handoff.EXIT[code], exit_code)
        self.assertIsInstance(error["message"], str)
        self.assertNotIn("\n", error["message"])
        self.assertIsInstance(error["stderr_tail"], list)
        self.assertEqual(handoff.validate(doc, handoff.load_schema(os.path.join(REPO, "contracts", "error.schema.json"))), [])

    def v1_packet(self, raw_path, packet_id="20261026T120000Z-sapphire-twixtor-0a0b0c", status="new", result=None):
        return {"schema_version": 1, "id": packet_id, "created": "2026-10-26T12:00:00Z",
                "updated": "2026-10-26T12:00:00Z", "from": "sapphire", "to": "twixtor", "task": "old",
                "status": status,
                "inputs": [{"path": raw_path, "sha256": sha(raw_path), "width": 4, "height": 2, "depth": 8,
                            "pix_fmt": "rgba", "fps": "24/1", "frames": 3}],
                "request": {"stages": [{"stage": "selftest-invert", "params": {}, "layers": {}}]},
                "expect": {"frames": 3, "size": 96, "determinism": "exact",
                           "tolerance": {"max_abs_diff": 0, "max_diff_fraction": 0}},
                "result": result, "notes": [{"at": "2026-10-26T12:00:00Z", "by": "sapphire", "text": "created"}]}

    def write_packet(self, packet):
        path = self.path(packet["id"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(packet, handle, indent=2)
        return path


# ----------------------------------------------------------------------------
# the validator
# ----------------------------------------------------------------------------

class ValidatorTests(unittest.TestCase):
    def check(self, instance, schema):
        return handoff.validate(instance, schema)

    def test_type(self):
        self.assertEqual(self.check(1, {"type": "integer"}), [])
        self.assertTrue(self.check(1.5, {"type": "integer"}))
        self.assertEqual(self.check(None, {"type": ["string", "null"]}), [])
        self.assertTrue(self.check([], {"type": "object"}))

    def test_boolean_is_not_a_number(self):
        self.assertTrue(self.check(True, {"type": "integer"}))
        self.assertTrue(self.check(True, {"type": "number"}))
        self.assertTrue(self.check(1, {"const": True}))
        self.assertTrue(self.check(True, {"enum": [1]}))

    def test_required_properties_additional(self):
        schema = {"type": "object", "required": ["a"], "additionalProperties": False,
                  "properties": {"a": {"type": "string"}}}
        self.assertEqual(self.check({"a": "x"}, schema), [])
        self.assertTrue(self.check({}, schema))
        self.assertTrue(self.check({"a": "x", "b": 1}, schema))
        self.assertTrue(self.check({"a": 1}, schema))
        typed = {"type": "object", "additionalProperties": {"type": "string"}}
        self.assertTrue(self.check({"k": 1}, typed))

    def test_enum_const_items(self):
        self.assertEqual(self.check(8, {"enum": [8, 16, None]}), [])
        self.assertTrue(self.check(12, {"enum": [8, 16, None]}))
        self.assertEqual(self.check(1.0, {"const": 1}), [])
        self.assertTrue(self.check([1, "x"], {"type": "array", "items": {"type": "integer"}}))

    def test_string_and_number_limits(self):
        self.assertTrue(self.check("", {"minLength": 1}))
        self.assertTrue(self.check("ab", {"pattern": "^a$"}))
        self.assertTrue(self.check(-1, {"minimum": 0}))
        self.assertTrue(self.check(2, {"maximum": 1}))
        self.assertEqual(self.check(1, {"minimum": 0, "maximum": 1}), [])

    def test_min_items(self):
        self.assertTrue(self.check([], {"type": "array", "minItems": 1}))
        self.assertEqual(self.check(["a"], {"type": "array", "minItems": 1}), [])

    def test_schema_keywords_are_checked(self):
        with self.assertRaises(handoff.SchemaError):
            handoff.check_schema({"oneOf": []})
        with self.assertRaises(handoff.SchemaError):
            handoff.check_schema({"properties": {"a": {"anyOf": []}}})
        with self.assertRaises(handoff.SchemaError):
            handoff.check_schema({"pattern": "("})
        handoff.check_schema({"type": "object", "title": "t", "examples": [], "x-codes": [{"anything": 1}]})


# ----------------------------------------------------------------------------
# the contract files
# ----------------------------------------------------------------------------

class ContractFileTests(Case):
    def schema(self, name):
        return handoff.read_json(os.path.join(REPO, "contracts", name), name)

    def test_every_contract_file_parses_and_every_schema_validates_its_examples(self):
        checks = handoff.contract_checks(REPO)
        failed = [c["detail"] for c in checks if not c["ok"]]
        self.assertEqual(failed, [])
        names = {c["name"] for c in checks}
        for name in ("error.schema.json", "stage-manifest.schema.json", "handoff-packet.schema.json",
                     "handoff-packet-v1.schema.json", "tool.schema.json", "describe.schema.json",
                     "run-report.schema.json", "tool.json", "error-table"):
            self.assertIn(name, names)

    def test_the_contracts_command(self):
        code, doc, err = self.cli_json("contracts", root=REPO)
        self.assertEqual(code, 0, err)
        self.assertTrue(all(c["ok"] for c in doc["checks"]))

    def test_a_broken_example_fails_the_contracts_command(self):
        root = os.path.join(self.tmp, "copy")
        shutil.copytree(os.path.join(REPO, "contracts"), os.path.join(root, "contracts"))
        path = os.path.join(root, "contracts", "describe.schema.json")
        data = handoff.read_json(path, "x")
        data["examples"][0]["width"] = 0
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
        code, doc, _ = self.cli_json("contracts", root=root)
        self.assertEqual(code, 4)
        self.assert_error(doc, "bad_input", 4)
        self.assertIn("describe.schema.json", doc["error"]["message"])

    def test_the_error_table_matches_the_exit_codes(self):
        data = self.schema("error.schema.json")
        rows = {r["code"]: r["exit"] for r in data["x-codes"]}
        self.assertEqual({c: e["frames/2"] for c, e in rows.items()}, handoff.EXIT)
        self.assertEqual(sorted(data["properties"]["error"]["properties"]["code"]["enum"]), sorted(handoff.EXIT))
        for code, exits in rows.items():
            self.assertEqual(exits["frames/1"], 130 if code == "interrupted" else 1, code)
        for code, value in handoff.EXIT.items():
            self.assertEqual(handoff.HandoffError(code, "x").exit_code, value)

    def test_the_code_table_is_in_docs_handoff(self):
        with open(os.path.join(REPO, DOCS_REL), encoding="utf-8") as handle:
            text = handle.read()
        for row in self.schema("error.schema.json")["x-codes"]:
            pattern = rf"^\| `{row['code']}` \| {row['exit']['frames/2']} \| {row['exit']['frames/1']} \|"
            self.assertRegex(text, re.compile(pattern, re.M), row["code"])

    def test_the_relative_path_rule_is_one_pattern(self):
        schema = self.schema("handoff-packet.schema.json")
        pattern = schema["properties"]["inputs"]["items"]["properties"]["path"]["pattern"]
        self.assertEqual(pattern, handoff.REL_RE.pattern)
        markers = self.schema("tool.schema.json")["properties"]["tools"]["items"]["properties"]["markers"]["items"]
        self.assertEqual(markers["pattern"], handoff.REL_RE.pattern)
        for good in ("a", "handoff/x/inputs/in.raw", "a..b/c", "x/..."):
            self.assertTrue(handoff.REL_RE.match(good), good)
        for bad in ("/abs", "C:/x", "c:x", "a/../b", "..", "../a", "a\\b", "a/.."):
            self.assertIsNone(handoff.REL_RE.match(bad), bad)

    def test_the_v1_schema_is_kept_for_reading(self):
        self.assertEqual(self.schema("handoff-packet-v1.schema.json")["properties"]["schema_version"]["const"], 1)
        self.assertEqual(self.schema("handoff-packet.schema.json")["properties"]["schema_version"]["const"], 2)

    def test_the_registry(self):
        self.assertEqual(handoff.validate(REGISTRY, handoff.load_schema(handoff.TOOL_SCHEMA_PATH)), [])
        self.assertEqual(sorted(handoff.tool_ids()), ["sapphire", "twixtor"])
        for tool in REGISTRY["tools"]:
            self.assertRegex(tool["id"], handoff.TOOL_ID_RE)
        for pattern in ("contracts/*.json", "docs/handoff.md", "tools/handoff.py", "tools/test_handoff.py"):
            self.assertIn(pattern, REGISTRY["shared"])

    def test_a_registry_with_a_duplicate_id_is_refused(self):
        folder = os.path.join(self.tmp, "reg")
        os.makedirs(folder)
        shutil.copyfile(handoff.TOOL_SCHEMA_PATH, os.path.join(folder, "tool.schema.json"))
        data = json.loads(json.dumps(REGISTRY))
        data["tools"].append(dict(data["tools"][0]))
        path = os.path.join(folder, "tool.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
        with self.assertRaises(handoff.HandoffError) as caught:
            handoff.load_registry(path)
        self.assertEqual(caught.exception.code, "bad_input")
        self.assertIn("duplicate tool id", caught.exception.message)


# ----------------------------------------------------------------------------
# packet validation
# ----------------------------------------------------------------------------

class PacketValidationTests(Case):
    def packet(self):
        return self.load(self.new())

    def problems(self, packet):
        return handoff.packet_problems(packet)

    def test_a_new_packet_is_valid(self):
        packet = self.packet()
        self.assertEqual(self.problems(packet), [])
        self.assertEqual((packet["schema_version"], packet["revision"]), (2, 1))

    def test_a_done_packet_is_valid(self):
        packet_id, _ = self.done_packet()
        self.assertEqual(self.problems(self.load(packet_id)), [])

    def test_bad_packets(self):
        def drop(key):
            return lambda p: p.pop(key)

        cases = {
            "missing task": drop("task"),
            "missing revision": drop("revision"),
            "revision 0": lambda p: p.update(revision=0),
            "extra field": lambda p: p.update(extra=1),
            "version 3": lambda p: p.update(schema_version=3),
            "bad status": lambda p: p.update(status="finished"),
            "bad sha256": lambda p: p["inputs"][0].update(sha256="abc"),
            "upper case sha256": lambda p: p["inputs"][0].update(sha256="A" * 64),
            "fps not a fraction": lambda p: p["inputs"][0].update(fps="24"),
            "negative frames": lambda p: p["inputs"][0].update(frames=-1),
            "depth 12": lambda p: p["inputs"][0].update(depth=12),
            "pix_fmt does not fit depth": lambda p: p["inputs"][0].update(pix_fmt="rgba64le"),
            "geometry half set": lambda p: p["inputs"][0].update(width=None),
            "empty task": lambda p: p.update(task=""),
            "bad id": lambda p: p.update(id="nope"),
            "id names other route": lambda p: p.update(id=p["id"].replace("-sapphire-twixtor-", "-twixtor-sapphire-")),
            "same from and to": lambda p: p.update(**{"from": "twixtor"}),
            "unregistered tool": lambda p: p.update(**{"from": "nobody"}),
            "bad timestamp": lambda p: p.update(created="2026-13-45T00:00:00Z"),
            "updated before created": lambda p: p.update(updated="2000-01-01T00:00:00Z"),
            "tolerance fraction above 1": lambda p: p["expect"]["tolerance"].update(max_diff_fraction=2),
            "unknown determinism": lambda p: p["expect"].update(determinism="mostly"),
            "param not a string": lambda p: p["request"]["stages"][0]["params"].update(k=1),
            "done without result": lambda p: p.update(status="done"),
            "note without text": lambda p: p["notes"].append({"at": p["created"], "by": "x", "text": ""}),
        }
        for name, change in cases.items():
            with self.subTest(name):
                packet = self.packet()
                change(packet)
                self.assertTrue(self.problems(packet), name)

    def test_paths_must_be_relative_posix_without_dotdot(self):
        for bad in ("/abs/in.raw", "C:/in.raw", "a\\in.raw", "a/../in.raw", "../in.raw"):
            with self.subTest(bad):
                packet = self.packet()
                packet["inputs"][0]["path"] = bad
                self.assertTrue(self.problems(packet), bad)
        packet = self.packet()
        packet["request"]["stages"][0]["layers"]["l"] = "/abs"
        self.assertTrue(self.problems(packet))

    def test_bad_result_shapes(self):
        packet_id, _ = self.done_packet()
        for name, change in {
            "command without exit code": lambda p: p["result"]["commands"][0].pop("exit_code"),
            "negative seconds": lambda p: p["result"]["commands"][0].update(seconds=-1),
            "argv not strings": lambda p: p["result"]["commands"][0].update(argv=[1]),
            "empty argv": lambda p: p["result"]["commands"][0].update(argv=[]),
            "empty program": lambda p: p["result"]["commands"][0].update(argv=[""]),
            "absolute stdin": lambda p: p["result"]["commands"][0].update(stdin="/x"),
            "output sha256": lambda p: p["result"]["outputs"][0].update(sha256="0"),
            "scratch missing": lambda p: p["result"].pop("scratch"),
            "scratch outside": lambda p: p["result"].update(scratch="../x"),
        }.items():
            with self.subTest(name):
                packet = self.load(packet_id)
                change(packet)
                self.assertTrue(self.problems(packet), name)

    def test_load_rejects_bad_json_and_nan(self):
        packet_id = self.new()
        path = self.path(packet_id)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("{not json")
        code, doc, _ = self.cli_json("show", packet_id)
        self.assertEqual(code, 4)
        self.assert_error(doc, "bad_input", 4)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"x": 1}).replace("1", "NaN"))
        self.assertEqual(self.cli("show", packet_id)[0], 4)

    def test_load_accepts_a_bom(self):
        packet_id = self.new()
        path = self.path(packet_id)
        data = read_bytes(path)
        write_bytes(path, b"\xef\xbb\xbf" + data)
        self.assertEqual(self.cli("show", packet_id)[0], 0)

    def test_a_file_above_one_mib_is_refused_on_load(self):
        packet_id = self.new()
        self.edit(packet_id, lambda p: p["notes"].append({"at": p["created"], "by": "x", "text": "y" * (1 << 20)}))
        code, doc, _ = self.cli_json("show", packet_id)
        self.assertEqual(code, 4)
        self.assertIn("1 MiB", doc["error"]["message"])

    def test_a_packet_above_one_mib_is_never_written(self):
        packet = self.load(self.new())
        packet["task"] = "t" * (1 << 20)
        before = read_bytes(self.path(packet["id"]))
        with self.assertRaises(handoff.HandoffError) as caught:
            handoff.save_packet(self.path(packet["id"]), packet)
        self.assertEqual(caught.exception.code, "bad_input")
        self.assertEqual(read_bytes(self.path(packet["id"])), before)


# ----------------------------------------------------------------------------
# new
# ----------------------------------------------------------------------------

class NewTests(Case):
    def test_new_writes_a_valid_packet(self):
        code, doc, err = self.cli_json(
            "new", "--from", "sapphire", "--to", "twixtor", "--task", "invert twice", "--input", self.raw,
            "--width", "4", "--height", "2", "--depth", "8", "--fps", "24", "--stage", "selftest-invert")
        self.assertEqual(code, 0, err)
        self.assertEqual((doc["contract"], doc["ok"], doc["changed"]), ("frames/2", True, True))
        packet = self.load(doc["id"])
        self.assertEqual(packet["status"], "new")
        self.assertEqual((packet["from"], packet["to"], packet["schema_version"], packet["revision"]),
                         ("sapphire", "twixtor", 2, 1))
        self.assertRegex(packet["id"], r"^\d{8}T\d{6}Z-sapphire-twixtor-[0-9a-f]{6}$")
        self.assertEqual(packet["created"], packet["updated"])
        rec = packet["inputs"][0]
        self.assertEqual(rec["sha256"], sha(self.raw))
        self.assertEqual((rec["width"], rec["height"], rec["depth"], rec["pix_fmt"], rec["fps"], rec["frames"]),
                         (4, 2, 8, "rgba", "24/1", 3))
        self.assertEqual(packet["expect"], {"frames": 3, "size": 96, "determinism": "exact",
                                            "tolerance": {"max_abs_diff": 0, "max_diff_fraction": 0}})
        self.assertEqual(packet["request"]["stages"], [{"stage": "selftest-invert", "params": {}, "layers": {}}])
        self.assertIsNone(packet["result"])
        self.assertEqual(handoff.packet_problems(packet), [])

    def test_an_input_outside_the_receiver_is_copied_into_the_packet_folder(self):
        packet_id = self.new()
        rec = self.load(packet_id)["inputs"][0]
        self.assertEqual(rec["path"], f"handoff/{packet_id}/inputs/in.raw")
        self.assertEqual(sha(self.input_path(packet_id)), sha(self.raw))

    def test_an_input_inside_the_receiver_keeps_its_relative_path(self):
        inside = os.path.join(self.recv, "clips", "a.raw")
        write_bytes(inside, raw_bytes(2))
        code, doc, err = self.cli_json("new", "--from", "sapphire", "--to", "twixtor", "--task", "t",
                                       "--input", inside, "--width", "4", "--height", "2", "--depth", "8")
        self.assertEqual(code, 0, err)
        self.assertEqual(doc["packet"]["inputs"][0]["path"], "clips/a.raw")
        self.assertFalse(os.path.exists(os.path.join(self.recv, "handoff", doc["id"])))

    def test_two_inputs_with_one_name_get_two_copies(self):
        other = os.path.join(self.tmp, "sub", "in.raw")
        write_bytes(other, b"abc")
        code, doc, err = self.cli_json("new", "--from", "sapphire", "--to", "twixtor", "--task", "t",
                                       "--file", self.raw, "--file", other)
        self.assertEqual(code, 0, err)
        paths = [r["path"] for r in doc["packet"]["inputs"]]
        self.assertEqual(len(set(paths)), 2)
        self.assertEqual(sha(handoff.resolve(paths[1], self.recv)), sha(other))

    def test_depth_decides_frame_size_and_pix_fmt(self):
        for depth, pix_fmt, per_pixel in ((8, "rgba", 4), (16, "rgba64le", 8), (32, "gbrapf32le", 16)):
            with self.subTest(depth=depth):
                path = os.path.join(self.tmp, f"d{depth}.raw")
                write_bytes(path, bytes(2 * 3 * per_pixel * 5))
                code, doc, err = self.cli_json("new", "--from", "twixtor", "--to", "sapphire", "--task", "t",
                                               "--input", path, "--width", "2", "--height", "3",
                                               "--depth", str(depth), "--stage", "s")
                self.assertEqual(code, 0, err)
                rec = doc["packet"]["inputs"][0]
                self.assertEqual((rec["pix_fmt"], rec["frames"]), (pix_fmt, 5))

    def test_fps_forms_become_exact_fractions(self):
        for given, expected in (("24", "24/1"), ("29.97", "2997/100"), ("30000/1001", "30000/1001"),
                                ("60/2", "30/1")):
            with self.subTest(given):
                code, doc, err = self.cli_json("new", "--from", "sapphire", "--to", "twixtor", "--task", "t",
                                               "--input", self.raw, "--width", "4", "--height", "2",
                                               "--depth", "8", "--fps", given, "--stage", "s")
                self.assertEqual(code, 0, err)
                self.assertEqual(doc["packet"]["inputs"][0]["fps"], expected)

    def test_bad_fps_is_a_usage_error(self):
        for given in ("0", "-1", "abc", "1/0", "24/"):
            with self.subTest(given):
                code, doc, _ = self.cli_json("new", "--from", "sapphire", "--to", "twixtor", "--task", "t",
                                             "--input", self.raw, "--fps", given)
                self.assertEqual(code, 2)
                self.assert_error(doc, "usage", 2)

    def test_partial_frame_is_refused_like_vfx(self):
        with open(self.raw, "ab") as handle:
            handle.write(b"\x00" * 5)
        code, doc, _ = self.cli_json("new", "--from", "sapphire", "--to", "twixtor", "--task", "t",
                                     "--input", self.raw, "--width", "4", "--height", "2", "--depth", "8")
        self.assertEqual(code, 2)
        self.assert_error(doc, "usage", 2)
        self.assertIn("partial frame (5 of 32 bytes) after 3 whole frames", doc["error"]["message"])
        self.assertFalse(os.path.exists(os.path.join(self.recv, "handoff")))

    def test_missing_input_and_half_geometry(self):
        base = ["new", "--from", "sapphire", "--to", "twixtor", "--task", "t"]
        self.assertEqual(self.cli(*base, "--input", os.path.join(self.tmp, "missing.raw"))[0], 2)
        self.assertEqual(self.cli(*base, "--input", self.raw, "--width", "4")[0], 2)

    def test_geometry_defaults_before_the_first_input_and_per_input_after(self):
        other = os.path.join(self.tmp, "b.raw")
        write_bytes(other, bytes(8 * 2 * 4 * 2))
        code, doc, err = self.cli_json(
            "new", "--from", "sapphire", "--to", "twixtor", "--task", "t",
            "--width", "4", "--height", "2", "--depth", "8", "--fps", "24",
            "--input", self.raw, "--input", other, "--width", "8", "--stage", "s")
        self.assertEqual(code, 0, err)
        first, second = doc["packet"]["inputs"]
        self.assertEqual((first["width"], first["frames"]), (4, 3))
        self.assertEqual((second["width"], second["height"], second["frames"], second["fps"]), (8, 2, 2, "24/1"))

    def test_file_input_is_hash_only(self):
        note = os.path.join(self.tmp, "notes.txt")
        write_bytes(note, b"hello")
        code, doc, err = self.cli_json("new", "--from", "sapphire", "--to", "twixtor", "--task", "t",
                                       "--width", "4", "--height", "2", "--depth", "8",
                                       "--input", self.raw, "--file", note)
        self.assertEqual(code, 0, err)
        rec = doc["packet"]["inputs"][1]
        self.assertEqual(rec["sha256"], sha(note))
        self.assertEqual((rec["width"], rec["frames"], rec["pix_fmt"]), (None, None, None))

    def test_stage_param_and_layer_bind_in_order(self):
        code, doc, err = self.cli_json(
            "new", "--from", "sapphire", "--to", "twixtor", "--task", "t", "--input", self.raw,
            "--stage", "retime", "--param", "speed_percent=100@0;50@24", "--param", "frame_interp=2",
            "--layer", "alt_source_a=" + self.raw, "--stage", "selftest-invert")
        self.assertEqual(code, 0, err)
        stages = doc["packet"]["request"]["stages"]
        self.assertEqual(stages[0]["params"], {"speed_percent": "100@0;50@24", "frame_interp": "2"})
        self.assertEqual(stages[0]["layers"], {"alt_source_a": f"handoff/{doc['id']}/layers/in.raw"})
        self.assertEqual(sha(handoff.resolve(stages[0]["layers"]["alt_source_a"], self.recv)), sha(self.raw))
        self.assertEqual(stages[1], {"stage": "selftest-invert", "params": {}, "layers": {}})

    def test_param_without_a_stage_or_twice_and_a_missing_layer_are_refused(self):
        base = ["new", "--from", "sapphire", "--to", "twixtor", "--task", "t"]
        self.assertEqual(self.cli(*base, "--param", "k=v")[0], 2)
        self.assertEqual(self.cli(*base, "--stage", "s", "--param", "k=1", "--param", "k=2")[0], 2)
        self.assertEqual(self.cli(*base, "--stage", "s", "--param", "novalue")[0], 2)
        self.assertEqual(self.cli(*base, "--stage", "s", "--layer", "l=" + os.path.join(self.tmp, "gone"))[0], 2)

    def test_expect_flags(self):
        code, doc, err = self.cli_json(
            "new", "--from", "sapphire", "--to", "twixtor", "--task", "t", "--input", self.raw,
            "--width", "4", "--height", "2", "--depth", "8", "--expect-frames", "6",
            "--tolerance-abs", "1", "--tolerance-fraction", "0.5")
        self.assertEqual(code, 0, err)
        self.assertEqual(doc["packet"]["expect"], {"frames": 6, "size": 192, "determinism": "tolerance",
                                                   "tolerance": {"max_abs_diff": 1, "max_diff_fraction": 0.5}})
        code, doc, err = self.cli_json(
            "new", "--from", "sapphire", "--to", "twixtor", "--task", "t", "--input", self.raw,
            "--width", "4", "--height", "2", "--depth", "8", "--expect-frames", "any", "--determinism", "none")
        self.assertEqual((doc["packet"]["expect"]["frames"], doc["packet"]["expect"]["size"]), (None, None))
        code, doc, _ = self.cli_json(
            "new", "--from", "sapphire", "--to", "twixtor", "--task", "t", "--determinism", "exact",
            "--tolerance-abs", "1")
        self.assertEqual(code, 2)

    def test_same_route_and_unregistered_tools_are_refused(self):
        for route in (("twixtor", "twixtor"), ("sapphire", "nobody"), ("nobody", "twixtor")):
            with self.subTest(route):
                code, doc, _ = self.cli_json("new", "--from", route[0], "--to", route[1], "--task", "t")
                self.assertEqual(code, 2)
                self.assert_error(doc, "usage", 2)

    def test_from_defaults_to_the_tool_of_the_root(self):
        make_tool(self.recv, "sapphire")
        code, doc, err = self.cli_json("new", "--to", "twixtor", "--task", "t")
        self.assertEqual(code, 0, err)
        self.assertEqual(doc["packet"]["from"], "sapphire")
        plain = os.path.join(self.tmp, "plain")
        os.makedirs(plain)
        code, doc, _ = self.cli_json("new", "--to", "twixtor", "--task", "t", root=plain)
        self.assertEqual(code, 2)
        self.assertIn("pass --from", doc["error"]["message"])

    def test_new_with_an_id_is_idempotent(self):
        fixed = "20261026T120000Z-sapphire-twixtor-abcdef"
        args = ["new", "--from", "sapphire", "--to", "twixtor", "--task", "same", "--input", self.raw,
                "--width", "4", "--height", "2", "--depth", "8", "--stage", "s", "--id", fixed]
        code, doc, err = self.cli_json(*args)
        self.assertEqual((code, doc["changed"]), (0, True), err)
        before = read_bytes(self.path(fixed))
        code, doc, err = self.cli_json(*args)
        self.assertEqual((code, doc["changed"]), (0, False), err)
        self.assertEqual(read_bytes(self.path(fixed)), before)
        changed = list(args)
        changed[changed.index("same")] = "different"
        code, doc, _ = self.cli_json(*changed)
        self.assertEqual(code, 2)
        self.assert_error(doc, "usage", 2)
        self.assertEqual(read_bytes(self.path(fixed)), before)

    def test_id_must_match_the_route(self):
        code, doc, _ = self.cli_json("new", "--from", "sapphire", "--to", "twixtor", "--task", "t",
                                     "--id", "20261026T120000Z-twixtor-sapphire-abcdef")
        self.assertEqual(code, 2)

    def test_note_is_recorded(self):
        packet_id = self.new("--note", "please run at depth 8")
        notes = self.load(packet_id)["notes"]
        self.assertEqual([n["text"] for n in notes], ["created", "please run at depth 8"])
        self.assertTrue(all(n["by"] == "sapphire" for n in notes))

    def test_text_mode_prints_the_id_and_the_path(self):
        code, out, err = self.cli("new", "--from", "sapphire", "--to", "twixtor", "--task", "t")
        self.assertEqual(code, 0, err)
        lines = out.strip().splitlines()
        self.assertEqual(lines[1], os.path.join(self.recv, "handoff", lines[0] + ".json"))


# ----------------------------------------------------------------------------
# accept, done, revisions, atomic writes
# ----------------------------------------------------------------------------

class LifecycleTests(Case):
    def test_accept_then_accept_again_is_a_no_op(self):
        packet_id = self.new()
        code, doc, err = self.cli_json("accept", packet_id)
        self.assertEqual((code, doc["status"], doc["changed"], doc["revision"]), (0, "accepted", True, 2), err)
        packet = self.load(packet_id)
        self.assertEqual([n["text"] for n in packet["notes"]], ["created", "accepted"])
        before = read_bytes(self.path(packet_id))
        code, doc, err = self.cli_json("accept", packet_id)
        self.assertEqual((code, doc["status"], doc["changed"]), (0, "accepted", False), err)
        self.assertEqual(read_bytes(self.path(packet_id)), before)

    def test_accept_by_unique_prefix(self):
        packet_id = self.new()
        self.assertEqual(self.cli("accept", packet_id[:14])[0], 0)
        self.assertEqual(self.load(packet_id)["status"], "accepted")
        code, doc, _ = self.cli_json("accept", "nothing")
        self.assertEqual(code, 2)
        self.assert_error(doc, "usage", 2)

    def test_ambiguous_prefix_is_refused(self):
        with mock.patch.object(handoff, "now_iso", return_value="2026-10-26T12:00:00Z"):
            first, second = self.new(), self.new()
        self.assertNotEqual(first, second)
        code, doc, _ = self.cli_json("accept", "20261026T120000Z")
        self.assertEqual(code, 2)
        self.assertIn("matches 2 packets", doc["error"]["message"])

    def test_done_needs_an_accepted_packet(self):
        packet_id = self.new()
        code, doc, _ = self.cli_json("done", packet_id)
        self.assertEqual(code, 2)
        self.assertIn("needs an accepted packet", doc["error"]["message"])
        self.assertEqual(self.load(packet_id)["status"], "new")

    def test_done_twice_and_accept_after_done_are_refused(self):
        packet_id, _ = self.done_packet()
        self.assertEqual(self.cli("done", packet_id)[0], 2)
        code, doc, _ = self.cli_json("accept", packet_id)
        self.assertEqual(code, 2)
        self.assertIn("only a new packet", doc["error"]["message"])

    def test_done_records_outputs_commands_and_scratch_as_relative_paths(self):
        packet_id, output = self.done_packet()
        packet = self.load(packet_id)
        result = packet["result"]
        out = result["outputs"][0]
        self.assertEqual((out["path"], out["sha256"]), ("out.raw", sha(output)))
        self.assertEqual((out["width"], out["height"], out["depth"], out["pix_fmt"], out["fps"], out["frames"]),
                         (4, 2, 8, "rgba", "24/1", 3))
        cmd = result["commands"][0]
        self.assertEqual(cmd["exit_code"], 0)
        self.assertGreaterEqual(cmd["seconds"], 0)
        self.assertEqual(cmd["argv"][:2], [sys.executable, "-c"])
        self.assertEqual((cmd["stdin"], cmd["stdout"], cmd["cwd"]),
                         (f"handoff/{packet_id}/inputs/in.raw", "out.raw", None))
        self.assertEqual(result["scratch"], "scratch")
        self.assertEqual(bytes(255 - x for x in read_bytes(self.raw)), read_bytes(output))
        self.assertEqual([n["text"] for n in packet["notes"]], ["created", "accepted", "done"])
        self.assertEqual(packet["revision"], 3)

    def test_done_with_a_non_zero_exit_marks_the_packet_failed(self):
        packet_id = self.new()
        self.cli("accept", packet_id)
        spec = self.spec_file("bad.json", {"argv": [sys.executable, "-c", "import sys; sys.exit(3)"]})
        code, doc, err = self.cli_json("done", packet_id, "--run", spec)
        self.assertEqual((code, doc["status"]), (0, "failed"), err)
        self.assertEqual(self.load(packet_id)["result"]["commands"][0]["exit_code"], 3)

    def test_done_failed_flag_skips_missing_outputs(self):
        packet_id = self.new()
        self.cli("accept", packet_id)
        code, doc, err = self.cli_json("done", packet_id, "--failed", "--output", os.path.join(self.recv, "nope"),
                                       "--note", "stage crashed")
        self.assertEqual((code, doc["status"]), (0, "failed"), err)
        packet = self.load(packet_id)
        self.assertEqual(packet["result"]["outputs"], [])
        self.assertEqual(packet["notes"][-1]["text"], "stage crashed")

    def test_done_missing_output_is_a_usage_error(self):
        packet_id = self.new()
        self.cli("accept", packet_id)
        code, doc, _ = self.cli_json("done", packet_id, "--output", os.path.join(self.recv, "nope"))
        self.assertEqual(code, 2)
        self.assertEqual(self.load(packet_id)["status"], "accepted")

    def test_an_output_or_a_spec_path_outside_the_root_is_refused_before_anything_runs(self):
        packet_id = self.new()
        self.cli("accept", packet_id)
        outside = os.path.join(self.tmp, "outside.raw")
        write_bytes(outside, raw_bytes(3))
        code, doc, _ = self.cli_json("done", packet_id, "--output", outside)
        self.assertEqual(code, 2)
        self.assertIn("must lie inside", doc["error"]["message"])
        marker = os.path.join(self.recv, "ran.txt")
        spec = self.spec_file("out.json", {"argv": [sys.executable, "-c", f"open({marker!r}, 'w')"],
                                           "stdout": outside})
        self.assertEqual(self.cli("done", packet_id, "--run", spec)[0], 2)
        self.assertFalse(os.path.exists(marker))
        self.assertEqual(self.cli("done", packet_id, "--scratch", self.tmp)[0], 2)
        self.assertEqual(self.load(packet_id)["status"], "accepted")

    def test_done_records_a_command_that_already_ran(self):
        packet_id = self.new()
        self.cli("accept", packet_id)
        spec = self.spec_file("rec.json", [{"argv": ["vfx", "apply", "selftest-invert"], "stdin": "in.raw",
                                           "stdout": "out.raw", "exit_code": 0, "seconds": 1.5}])
        out = os.path.join(self.recv, "out.raw")
        shutil.copyfile(self.raw, out)
        code, doc, err = self.cli_json("done", packet_id, "--output", out, "--command", spec)
        self.assertEqual(code, 0, err)
        cmd = self.load(packet_id)["result"]["commands"][0]
        self.assertEqual((cmd["argv"][0], cmd["exit_code"], cmd["seconds"], cmd["cwd"], cmd["stdin"]),
                         ("vfx", 0, 1.5, None, "in.raw"))

    def test_bad_command_specs_are_refused(self):
        packet_id = self.new()
        self.cli("accept", packet_id)
        for name, spec in {
            "no argv": {"stdin": "x"},
            "empty argv": {"argv": []},
            "unknown key": {"argv": ["a"], "shell": True},
            "run with exit code": {"argv": ["a"], "exit_code": 0},
        }.items():
            with self.subTest(name):
                self.assertEqual(self.cli("done", packet_id, "--run", self.spec_file("s.json", spec))[0], 2)
        self.assertEqual(self.cli("done", packet_id, "--command", self.spec_file("s.json", {"argv": ["a"]}))[0], 2)
        self.assertEqual(self.cli("done", packet_id, "--run", "{not json")[0], 2)
        self.assertEqual(self.load(packet_id)["status"], "accepted")

    def test_a_program_that_cannot_start_is_recorded_as_exit_127(self):
        packet_id = self.new()
        self.cli("accept", packet_id)
        spec = self.spec_file("missing.json", {"argv": [os.path.join(self.tmp, "no-such-program")]})
        code, doc, err = self.cli_json("done", packet_id, "--run", spec)
        self.assertEqual((code, doc["status"]), (0, "failed"), err)
        self.assertEqual(self.load(packet_id)["result"]["commands"][0]["exit_code"], 127)

    def test_updated_moves_and_created_stays(self):
        with mock.patch.object(handoff, "now_iso", Clock()):
            packet_id = self.new()
            created = self.load(packet_id)["created"]
            self.cli("accept", packet_id)
        packet = self.load(packet_id)
        self.assertEqual(packet["created"], created)
        self.assertGreater(packet["updated"], created)

    def test_every_write_increments_the_revision(self):
        packet_id, _ = self.done_packet()
        self.assertEqual(self.load(packet_id)["revision"], 3)
        code, doc, err = self.cli_json("verify", packet_id)
        self.assertEqual((code, doc["revision"]), (0, 4), err)
        self.assertEqual(self.load(packet_id)["revision"], 4)

    def test_the_revision_flag_is_a_compare_and_swap(self):
        packet_id = self.new()
        code, doc, _ = self.cli_json("accept", packet_id, "--revision", "2")
        self.assertEqual(code, 4)
        self.assert_error(doc, "bad_input", 4)
        self.assertIn("revision conflict", doc["error"]["message"])
        self.assertEqual(self.load(packet_id)["status"], "new")
        self.assertEqual(self.cli("accept", packet_id, "--revision", "1")[0], 0)

    def test_a_change_between_read_and_write_is_a_conflict(self):
        packet_id = self.new()
        path = self.path(packet_id)
        mine = handoff.load_packet(path)
        self.edit(packet_id, lambda p: p.update(revision=2, task="changed by another writer"))
        mine["status"] = "accepted"
        with self.assertRaises(handoff.HandoffError) as caught:
            handoff.commit(path, mine, 1)
        self.assertEqual(caught.exception.code, "bad_input")
        self.assertIn("revision conflict", caught.exception.message)
        self.assertEqual(self.load(packet_id)["task"], "changed by another writer")
        self.assertFalse(os.path.exists(path[:-5] + ".lock"))

    def test_atomic_write_goes_through_a_temporary_name_in_the_same_folder(self):
        packet_id = self.new()
        path = self.path(packet_id)
        seen = []
        real = os.replace

        def spy(src, dst):
            with open(src, encoding="utf-8") as a, open(dst, encoding="utf-8") as b:
                seen.append((os.path.basename(src), os.path.dirname(src) == os.path.dirname(dst),
                             json.load(a)["status"], json.load(b)["status"]))
            real(src, dst)

        with mock.patch.object(handoff.os, "replace", spy):
            self.assertEqual(self.cli("accept", packet_id)[0], 0)
        self.assertEqual(len(seen), 1)
        name, same_folder, new_status, old_status = seen[0]
        self.assertTrue(name.startswith(".") and name.endswith(".tmp"), name)
        self.assertTrue(same_folder)
        self.assertEqual((new_status, old_status), ("accepted", "new"))
        self.assertEqual(files_in(os.path.dirname(path)), [packet_id + ".json"])

    def test_a_failed_write_leaves_the_old_packet_no_temporary_file_and_no_lock(self):
        packet_id = self.new()
        path = self.path(packet_id)
        before = read_bytes(path)
        with mock.patch.object(handoff.os, "replace", side_effect=OSError("disk full")):
            code, out, err = self.cli("accept", packet_id)
        self.assertEqual(code, 1)
        self.assertIn("error: stage_failed:", err)
        self.assertEqual(read_bytes(path), before)
        self.assertEqual(files_in(os.path.dirname(path)), [packet_id + ".json"])

    def test_an_invalid_packet_is_never_written(self):
        packet = self.load(self.new())
        packet["status"] = "bogus"
        path = self.path(packet["id"])
        before = read_bytes(path)
        with self.assertRaises(handoff.HandoffError) as caught:
            handoff.save_packet(path, packet)
        self.assertEqual(caught.exception.exit_code, 4)
        self.assertEqual(read_bytes(path), before)


# ----------------------------------------------------------------------------
# the lock
# ----------------------------------------------------------------------------

class LockTests(Case):
    def lock_file(self, packet_id, body):
        path = self.path(packet_id)[:-5] + ".lock"
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(body if isinstance(body, str) else json.dumps(body))
        return path

    def dead_pid(self):
        proc = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True)
        return int(proc.stdout.strip())

    def test_pid_alive(self):
        self.assertTrue(handoff.pid_alive(os.getpid()))
        self.assertFalse(handoff.pid_alive(self.dead_pid()))
        self.assertFalse(handoff.pid_alive(0))
        self.assertFalse(handoff.pid_alive(True))

    def test_a_live_lock_times_out_as_stage_timeout(self):
        packet_id = self.new()
        lock = self.lock_file(packet_id, {"pid": os.getpid(), "host": socket.gethostname(), "at": "now"})
        code, doc, _ = self.cli_json("accept", packet_id, "--lock-wait", "0.2")
        self.assertEqual(code, 5)
        self.assert_error(doc, "stage_timeout", 5)
        self.assertEqual(doc["error"]["stage"], "lock")
        self.assertTrue(os.path.exists(lock))
        self.assertEqual(self.load(packet_id)["status"], "new")

    def test_the_lock_of_a_dead_process_is_broken(self):
        packet_id = self.new()
        lock = self.lock_file(packet_id, {"pid": self.dead_pid(), "host": socket.gethostname(), "at": "then"})
        self.assertEqual(self.cli("accept", packet_id, "--lock-wait", "0")[0], 0)
        self.assertFalse(os.path.exists(lock))
        self.assertEqual(self.load(packet_id)["status"], "accepted")

    def test_a_lock_of_another_host_is_respected(self):
        packet_id = self.new()
        self.lock_file(packet_id, {"pid": self.dead_pid(), "host": "another-host.invalid", "at": "then"})
        self.assertEqual(self.cli("accept", packet_id, "--lock-wait", "0.1")[0], 5)

    def test_an_unreadable_lock_is_stale_only_when_old(self):
        packet_id = self.new()
        lock = self.lock_file(packet_id, "garbage")
        self.assertEqual(self.cli("accept", packet_id, "--lock-wait", "0.1")[0], 5)
        old = os.path.getmtime(lock) - 3600
        os.utime(lock, (old, old))
        self.assertEqual(self.cli("accept", packet_id, "--lock-wait", "0")[0], 0)
        self.assertFalse(os.path.exists(lock))

    def test_the_lock_holds_pid_host_and_time_while_held(self):
        packet_id = self.new()
        path = self.path(packet_id)
        with handoff.PacketLock(path, 0) as held:
            info = handoff.read_lock(held.path)
            self.assertEqual((info["pid"], info["host"]), (os.getpid(), socket.gethostname()))
            with self.assertRaises(handoff.HandoffError):
                with handoff.PacketLock(path, 0):
                    pass
        self.assertFalse(os.path.exists(path[:-5] + ".lock"))


# ----------------------------------------------------------------------------
# verify and replay
# ----------------------------------------------------------------------------

class VerifyTests(Case):
    def failing(self, doc, name):
        return [c for c in doc["checks"] if not c["ok"] and c["name"] == name]

    def test_verify_passes_and_marks_verified(self):
        packet_id, _ = self.done_packet()
        code, doc, err = self.cli_json("verify", packet_id)
        self.assertEqual(code, 0, err)
        self.assertTrue(all(c["ok"] for c in doc["checks"]))
        self.assertEqual((doc["status_before"], doc["status"], doc["changed"]), ("done", "verified", True))
        packet = self.load(packet_id)
        self.assertEqual((packet["status"], packet["notes"][-1]["text"]), ("verified", "verified"))
        names = {c["name"] for c in doc["checks"]}
        self.assertTrue({"input-sha256", "input-frames", "output-sha256", "output-frames", "expect-size",
                         "expect-frames", "command-exit"} <= names)
        again = self.cli_json("verify", packet_id)
        self.assertEqual((again[0], again[1]["changed"], again[1]["status"]), (0, False, "verified"))

    def test_no_write_keeps_the_status(self):
        packet_id, _ = self.done_packet()
        self.assertEqual(self.cli("verify", packet_id, "--no-write")[0], 0)
        self.assertEqual(self.load(packet_id)["status"], "done")

    def test_verify_before_done_only_checks_inputs(self):
        packet_id = self.new()
        code, doc, err = self.cli_json("verify", packet_id)
        self.assertEqual(code, 0, err)
        self.assertEqual({c["name"] for c in doc["checks"]}, {"input-sha256", "input-frames"})
        self.assertEqual(self.load(packet_id)["status"], "new")

    def test_hash_mismatch_of_an_input_is_detected(self):
        packet_id, _ = self.done_packet()
        with open(self.input_path(packet_id), "r+b") as handle:
            handle.write(b"\xff")
        code, doc, err = self.cli_json("verify", packet_id)
        self.assertEqual(code, 4)
        self.assert_error(doc, "bad_input", 4)
        self.assertEqual(doc["error"]["stage"], "verify")
        self.assertTrue(self.failing(doc, "input-sha256"))
        self.assertEqual(self.load(packet_id)["status"], "done")

    def test_hash_mismatch_of_an_output_is_detected(self):
        packet_id, output = self.done_packet()
        with open(output, "r+b") as handle:
            handle.write(b"\x01\x02")
        code, doc, _ = self.cli_json("verify", packet_id)
        self.assertEqual(code, 4)
        self.assertTrue(self.failing(doc, "output-sha256"))
        code, out, err = self.cli("verify", packet_id)
        self.assertIn("[fail]", out)
        self.assertRegex(err.strip(), r"^error: bad_input: 1 verification problem\(s\) in ")

    def test_a_missing_output_is_detected(self):
        packet_id, output = self.done_packet()
        os.remove(output)
        code, doc, _ = self.cli_json("verify", packet_id)
        self.assertEqual(code, 4)
        self.assertTrue(self.failing(doc, "output-exists"))

    def test_frame_count_and_size_are_checked_against_expect(self):
        packet_id, _ = self.done_packet()
        self.edit(packet_id, lambda p: p["expect"].update(frames=4))
        code, doc, _ = self.cli_json("verify", packet_id)
        self.assertEqual(code, 4)
        self.assertTrue(self.failing(doc, "expect-frames"))
        self.edit(packet_id, lambda p: p["expect"].update(frames=3, size=95))
        code, doc, _ = self.cli_json("verify", packet_id)
        self.assertTrue(self.failing(doc, "expect-size"))
        self.assertFalse(self.failing(doc, "expect-frames"))

    def test_an_output_that_is_not_whole_frames_is_a_partial_frame(self):
        packet_id, output = self.done_packet()
        write_bytes(output, read_bytes(output)[:-5])
        self.edit(packet_id, lambda p: p["result"]["outputs"][0].update(sha256=sha(output)))
        code, doc, _ = self.cli_json("verify", packet_id)
        self.assertEqual(code, 4)
        self.assert_error(doc, "partial_frame", 4)

    def test_a_recorded_non_zero_exit_fails_a_done_packet(self):
        packet_id, _ = self.done_packet()
        self.edit(packet_id, lambda p: p["result"]["commands"][0].update(exit_code=9))
        code, doc, _ = self.cli_json("verify", packet_id)
        self.assertEqual(code, 1)
        self.assert_error(doc, "stage_failed", 1)
        self.assertEqual(doc["error"]["exit_code"], 9)

    def test_a_failed_packet_is_checked_for_hashes_only(self):
        packet_id = self.new()
        self.cli("accept", packet_id)
        spec = self.spec_file("bad.json", {"argv": [sys.executable, "-c", "import sys; sys.exit(3)"]})
        self.assertEqual(self.cli("done", packet_id, "--run", spec)[0], 0)
        code, doc, err = self.cli_json("verify", packet_id)
        self.assertEqual((code, doc["status"]), (0, "failed"), err)

    def test_replay_of_an_unregistered_program_needs_trust(self):
        packet_id, output = self.done_packet()
        before = sha(output)
        code, doc, _ = self.cli_json("verify", packet_id, "--replay")
        self.assertEqual(code, 3)
        self.assert_error(doc, "unsupported", 3)
        self.assertEqual(doc["error"]["stage"], "replay")
        self.assertIn("--trust", doc["error"]["message"])
        self.assertEqual((sha(output), self.load(packet_id)["status"]), (before, "done"))

    def test_replay_exact_with_trust_passes_and_changes_no_recorded_file(self):
        packet_id, output = self.done_packet()
        before = (sha(output), os.path.getmtime(output))
        with mock.patch.object(tempfile, "tempdir", self.tmp):
            code, doc, err = self.cli_json("verify", packet_id, "--replay", "--trust")
        self.assertEqual(code, 0, err)
        self.assertEqual(doc["replay"]["outputs"][0]["replay_sha256"], sha(output))
        self.assertTrue(doc["replay"]["outputs"][0]["match"])
        self.assertEqual((sha(output), os.path.getmtime(output)), before)
        self.assertEqual(self.load(packet_id)["notes"][-1]["text"], "verified with replay")
        self.assertEqual([n for n in os.listdir(self.tmp) if n.startswith("handoff_replay_")], [])

    def test_replay_of_a_registered_manifest_provider_runs_without_trust(self):
        script = os.path.join(self.tmp, "prov", "invert.py")
        write_bytes(script, INVERT_SCRIPT.encode())
        manifest = os.path.join(self.tmp, "prov", "invert.stage.json")
        with open(manifest, "w", encoding="utf-8") as handle:
            json.dump({"contract": "frames/2", "provider": {"id": "fixture", "version": "1", "command": [sys.executable, script]},
                       "stages": [{"id": "fixture-a", "kind": "frame", "spool": False, "geometry": "preserve",
                                   "depths": [8], "params": [], "layers": []}]}, handle)
        packet_id = self.new()
        self.cli("accept", packet_id)
        output = os.path.join(self.recv, "out.raw")
        spec = self.spec_file("p.json", {"argv": [sys.executable, script], "stdin": self.input_path(packet_id),
                                         "stdout": output})
        self.assertEqual(self.cli("done", packet_id, "--output", output, "--run", spec)[0], 0)
        self.assertEqual(self.cli("verify", packet_id, "--replay", "--no-write")[0], 3)
        code, doc, err = self.cli_json("verify", packet_id, "--replay", "--provider", manifest)
        self.assertEqual(code, 0, err)
        with mock.patch.dict(os.environ, {"HANDOFF_PROVIDERS": os.path.dirname(manifest)}):
            self.assertEqual(self.cli("verify", packet_id, "--replay", "--no-write")[0], 0)

    def test_replay_runs_the_local_ffmpeg_for_a_recorded_ffmpeg(self):
        packet_id = self.new("--expect-frames", "any")
        self.cli("accept", packet_id)
        spec = self.spec_file("ff.json", {"argv": ["C:/elsewhere/ffmpeg.exe", "-c", "pass"], "exit_code": 0, "seconds": 0})
        self.assertEqual(self.cli("done", packet_id, "--command", spec)[0], 0)
        with mock.patch.object(handoff, "resolve_tool", lambda name, envs: (sys.executable, "")):
            code, doc, err = self.cli_json("verify", packet_id, "--replay")
        self.assertEqual(code, 0, err)
        self.assertEqual(doc["replay"]["commands"][0]["argv"][0], sys.executable)
        with mock.patch.object(handoff, "resolve_tool", lambda name, envs: (None, "not on PATH")):
            code, doc, _ = self.cli_json("verify", packet_id, "--replay")
        self.assertEqual(code, 6)
        self.assert_error(doc, "tool_missing", 6)

    def test_replay_without_trust_reaches_no_path_outside_the_packet(self):
        packet_id = self.new("--expect-frames", "any")
        self.cli("accept", packet_id)
        outside = os.path.join(self.tmp, "outside.txt")
        write = "import sys; open(sys.argv[1], 'w').write('x')"
        spec = self.spec_file("ff.json", {"argv": ["C:/elsewhere/ffmpeg.exe", "-c", write, outside], "exit_code": 0,
                                          "seconds": 0})
        self.assertEqual(self.cli("done", packet_id, "--command", spec)[0], 0)
        with mock.patch.object(handoff, "resolve_tool", lambda name, envs: (sys.executable, "")):
            code, doc, err = self.cli_json("verify", packet_id, "--replay")
        self.assertNotEqual(code, 0, err)
        self.assertIn("names a path outside the packet", json.dumps(doc))
        self.assertFalse(os.path.exists(outside))
        bare = self.spec_file("bare.json", {"argv": ["C:/elsewhere/ffmpeg.exe", "-c", write, "side.txt"], "exit_code": 0,
                                            "seconds": 0})
        other = self.new("--expect-frames", "any")
        self.cli("accept", other)
        self.assertEqual(self.cli("done", other, "--command", bare)[0], 0)
        with mock.patch.object(handoff, "resolve_tool", lambda name, envs: (sys.executable, "")):
            code, doc, err = self.cli_json("verify", other, "--replay")
        self.assertEqual(code, 0, err)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "side.txt")))

    def test_confine_judges_every_value_and_only_real_separators(self):
        scratch = os.path.join(self.tmp, "scratch")
        os.makedirs(scratch, exist_ok=True)
        outside = os.path.join(self.tmp, "x.cube")
        self.assertIsNone(handoff.confine("lut3d=file=" + outside, self.tmp, scratch, {}))
        self.assertIsNone(handoff.confine("--out=" + outside, self.tmp, scratch, {}))
        kept = "--out=" + os.path.join(scratch, "o.raw")
        self.assertEqual(handoff.confine(kept, self.tmp, scratch, {}), kept)
        for word in ("select='eq(n\\,0)'", "30000/1001", "setpts=PTS/2", "rgba"):
            self.assertEqual(handoff.confine(word, self.tmp, scratch, {}), word, word)
        self.assertTrue(handoff.inside(os.path.join(scratch, "a"), scratch))
        self.assertFalse(handoff.inside(outside, scratch))
        if os.name == "nt":
            other = "Z:" if not self.tmp.upper().startswith("Z:") else "Y:"
            self.assertFalse(handoff.inside(other + "\\x", scratch))

    def test_replay_of_a_command_that_is_not_deterministic_fails_exact(self):
        packet_id, _ = self.done_packet(code=NOISE)
        self.edit(packet_id, lambda p: p["expect"].update(frames=None, size=None))
        code, doc, _ = self.cli_json("verify", packet_id, "--replay", "--trust")
        self.assertEqual(code, 4)
        self.assert_error(doc, "bad_input", 4)
        self.assertEqual(doc["error"]["stage"], "replay")
        self.assertFalse(doc["replay"]["outputs"][0]["match"])
        self.assertEqual(self.load(packet_id)["status"], "done")

    def test_replay_with_determinism_none_does_not_compare_bytes(self):
        packet_id, _ = self.done_packet(code=NOISE)
        self.edit(packet_id, lambda p: p["expect"].update(frames=None, size=None, determinism="none"))
        self.assertEqual(self.cli("verify", packet_id, "--replay", "--trust")[0], 0)

    def test_replay_tolerance(self):
        packet_id, _ = self.done_packet(code=SHIFT, argv_extra=("0",))

        def shift_by_one(packet):
            packet["result"]["commands"][0]["argv"][-1] = "1"

        self.edit(packet_id, shift_by_one)
        replay = ("verify", packet_id, "--replay", "--trust", "--no-write")
        self.assertEqual(self.cli(*replay)[0], 4)
        self.edit(packet_id, lambda p: p["expect"].update(
            determinism="tolerance", tolerance={"max_abs_diff": 1, "max_diff_fraction": 1}))
        code, doc, err = self.cli_json(*replay)
        self.assertEqual(code, 0, err)
        compare = doc["replay"]["outputs"][0]["compare"]
        self.assertEqual((compare["differing"], compare["samples"], compare["max_abs_diff"]), (96, 96, 1))
        self.edit(packet_id, lambda p: p["expect"].update(tolerance={"max_abs_diff": 0, "max_diff_fraction": 1}))
        self.assertEqual(self.cli(*replay)[0], 4)
        self.edit(packet_id, lambda p: p["expect"].update(tolerance={"max_abs_diff": 1, "max_diff_fraction": 0.5}))
        self.assertEqual(self.cli(*replay)[0], 4)

    def test_replay_substitutes_an_output_path_inside_a_flag(self):
        output = os.path.join(self.recv, "flagged.bin")
        packet_id = self.new("--expect-frames", "any")
        self.cli("accept", packet_id)
        spec = self.spec_file("flag.json", {"argv": [sys.executable, "-c", WRITE_TO_ARG, "--out=" + output]})
        self.assertEqual(self.cli("done", packet_id, "--output", output, "--run", spec)[0], 0)
        self.assertIsNone(self.load(packet_id)["result"]["outputs"][0]["frames"])
        before = (sha(output), os.path.getmtime(output))
        code, doc, err = self.cli_json("verify", packet_id, "--replay", "--trust")
        self.assertEqual(code, 0, err)
        self.assertEqual((sha(output), os.path.getmtime(output)), before)

    def test_replay_stdout_to_a_file_that_is_not_an_output_is_redirected(self):
        packet_id = self.new()
        self.cli("accept", packet_id)
        side, output = os.path.join(self.recv, "side.bin"), os.path.join(self.recv, "out.raw")
        stdin = self.input_path(packet_id)
        spec = self.spec_file("side.json", [
            {"argv": [sys.executable, "-c", INVERT], "stdin": stdin, "stdout": output},
            {"argv": [sys.executable, "-c", INVERT], "stdin": stdin, "stdout": side}])
        self.assertEqual(self.cli("done", packet_id, "--output", output, "--run", spec)[0], 0)
        write_bytes(side, b"keep")
        self.assertEqual(self.cli("verify", packet_id, "--replay", "--trust")[0], 0)
        self.assertEqual(read_bytes(side), b"keep")

    def test_replay_timeout_is_a_stage_timeout(self):
        packet_id = self.new("--expect-frames", "any")
        self.cli("accept", packet_id)
        spec = self.spec_file("slow.json", {"argv": [sys.executable, "-c", "import time; time.sleep(30)"],
                                           "exit_code": 0, "seconds": 30})
        self.assertEqual(self.cli("done", packet_id, "--command", spec)[0], 0)
        code, doc, _ = self.cli_json("verify", packet_id, "--replay", "--trust", "--timeout", "0.5")
        self.assertEqual(code, 5)
        self.assert_error(doc, "stage_timeout", 5)

    def test_replay_of_a_missing_program_is_tool_missing(self):
        packet_id = self.new("--expect-frames", "any")
        self.cli("accept", packet_id)
        spec = self.spec_file("gone.json", {"argv": [os.path.join(self.tmp, "gone-program")],
                                           "exit_code": 0, "seconds": 0})
        self.assertEqual(self.cli("done", packet_id, "--command", spec)[0], 0)
        code, doc, _ = self.cli_json("verify", packet_id, "--replay", "--trust")
        self.assertEqual(code, 6)
        self.assert_error(doc, "tool_missing", 6)

    def test_replay_exit_code_must_equal_the_recorded_one(self):
        packet_id = self.new("--expect-frames", "any")
        self.cli("accept", packet_id)
        spec = self.spec_file("three.json", {"argv": [sys.executable, "-c", "import sys; sys.exit(3)"]})
        self.assertEqual(self.cli("done", packet_id, "--run", spec)[0], 0)
        self.assertEqual(self.cli("verify", packet_id, "--replay", "--trust")[0], 0)
        self.edit(packet_id, lambda p: p["result"]["commands"][0].update(exit_code=0))
        code, doc, _ = self.cli_json("verify", packet_id, "--replay", "--trust")
        self.assertEqual(code, 1)
        self.assert_error(doc, "stage_failed", 1)
        self.assertEqual(doc["error"]["exit_code"], 3)

    def test_replay_needs_a_recorded_result(self):
        packet_id = self.new()
        code, doc, _ = self.cli_json("verify", packet_id, "--replay")
        self.assertEqual(code, 2)
        self.assert_error(doc, "usage", 2)

    def test_compare_samples(self):
        a, b = os.path.join(self.tmp, "a"), os.path.join(self.tmp, "b")
        write_bytes(a, bytes([1, 2, 3, 4]))
        write_bytes(b, bytes([1, 2, 9, 4]))
        report = handoff.compare_samples(a, b, 8)
        self.assertEqual((report["samples"], report["differing"], report["max_abs_diff"]), (4, 1, 6))
        write_bytes(a, struct.pack("<2H", 100, 65535))
        write_bytes(b, struct.pack("<2H", 101, 65534))
        report = handoff.compare_samples(a, b, 16)
        self.assertEqual((report["samples"], report["differing"], report["max_abs_diff"]), (2, 2, 1))
        write_bytes(a, struct.pack("<2f", 0.5, float("nan")))
        write_bytes(b, struct.pack("<2f", 0.75, float("nan")))
        report = handoff.compare_samples(a, b, 32)
        self.assertEqual((report["samples"], report["differing"], report["max_abs_diff"]), (2, 1, 0.25))
        write_bytes(b, b"\x00" * 4)
        self.assertFalse(handoff.compare_samples(a, b, 32)["same_length"])


# ----------------------------------------------------------------------------
# version 1 packets and migrate
# ----------------------------------------------------------------------------

class MigrateTests(Case):
    def inside_raw(self):
        inside = os.path.join(self.recv, "clips", "in.raw")
        write_bytes(inside, raw_bytes(3))
        return inside

    def test_a_v1_packet_is_readable_but_not_writable(self):
        packet = self.v1_packet(self.inside_raw())
        self.write_packet(packet)
        code, doc, err = self.cli_json("list")
        self.assertEqual(code, 0, err)
        self.assertEqual([(p["schema_version"], p["revision"]) for p in doc["packets"]], [(1, None)])
        self.assertEqual(self.cli("show", packet["id"])[0], 0)
        self.assertEqual(self.cli("verify", packet["id"])[0], 0)
        code, doc, _ = self.cli_json("accept", packet["id"])
        self.assertEqual(code, 2)
        self.assertIn("migrate", doc["error"]["message"])

    def test_migrate_rewrites_paths_and_is_idempotent(self):
        inside = self.inside_raw()
        out = os.path.join(self.recv, "out.raw")
        write_bytes(out, raw_bytes(3))
        result = {"outputs": [dict(self.v1_packet(out)["inputs"][0])],
                  "commands": [{"argv": ["vfx"], "stdin": inside, "stdout": "out.raw", "cwd": None,
                                "exit_code": 0, "seconds": 1}],
                  "scratch": self.tmp}
        packet = self.v1_packet(inside, status="done", result=result)
        self.write_packet(packet)
        code, doc, err = self.cli_json("migrate", packet["id"])
        self.assertEqual((code, doc["migrated"]), (0, [packet["id"]]), err)
        new = self.load(packet["id"])
        self.assertEqual((new["schema_version"], new["revision"]), (2, 1))
        self.assertEqual(new["inputs"][0]["path"], "clips/in.raw")
        self.assertEqual(new["result"]["outputs"][0]["path"], "out.raw")
        self.assertEqual((new["result"]["commands"][0]["stdin"], new["result"]["commands"][0]["stdout"]),
                         ("clips/in.raw", "out.raw"))
        self.assertIsNone(new["result"]["scratch"])
        texts = [n["text"] for n in new["notes"]]
        self.assertIn("migrated from schema_version 1", texts)
        self.assertTrue(any("was dropped" in t for t in texts))
        self.assertEqual(handoff.packet_problems(new), [])
        before = read_bytes(self.path(packet["id"]))
        code, doc, err = self.cli_json("migrate", packet["id"])
        self.assertEqual((code, doc["unchanged"], doc["changed"]), (0, [packet["id"]], False), err)
        self.assertEqual(read_bytes(self.path(packet["id"])), before)
        self.assertEqual(self.cli("verify", packet["id"])[0], 0)
        self.assertEqual(self.load(packet["id"])["status"], "verified")

    def test_migrate_refuses_a_path_outside_the_root(self):
        packet = self.v1_packet(self.raw)
        path = self.write_packet(packet)
        before = read_bytes(path)
        code, doc, _ = self.cli_json("migrate", packet["id"])
        self.assertEqual(code, 4)
        self.assert_error(doc, "bad_input", 4)
        self.assertEqual(read_bytes(path), before)

    def test_migrate_all_and_its_arguments(self):
        self.write_packet(self.v1_packet(self.inside_raw()))
        self.new()
        code, doc, err = self.cli_json("migrate", "--all")
        self.assertEqual((code, len(doc["migrated"]), len(doc["unchanged"])), (0, 1, 1), err)
        self.assertEqual(self.cli("migrate")[0], 2)


# ----------------------------------------------------------------------------
# list, show, finding peers
# ----------------------------------------------------------------------------

class ListTests(Case):
    def layout(self):
        """two sibling folders with neutral names: one holds the sapphire markers, one the twixtor markers."""
        work = os.path.join(self.tmp, "work")
        mine = make_tool(os.path.join(work, "alpha"), "sapphire")
        theirs = make_tool(os.path.join(work, "beta"), "twixtor")
        return mine, theirs

    def test_list_is_sorted_filtered_and_reports_invalid_files(self):
        with mock.patch.object(handoff, "now_iso", Clock()):
            first, second = self.new(task="first"), self.new(task="second")
        self.cli("accept", second)
        folder = os.path.join(self.recv, "handoff")
        for name, text in (("broken.json", "{}"), (".hidden.json.1.tmp", "{}"), ("notes.txt", "x"),
                           (first + ".lock", "{}")):
            with open(os.path.join(folder, name), "w", encoding="utf-8") as handle:
                handle.write(text)
        code, doc, err = self.cli_json("list")
        self.assertEqual(code, 0, err)
        self.assertEqual([p["id"] for p in doc["packets"]], sorted([first, second]))
        self.assertEqual(len(doc["invalid"]), 1)
        self.assertTrue(doc["invalid"][0]["path"].endswith("broken.json"))
        code, doc, _ = self.cli_json("list", "--status", "accepted")
        self.assertEqual(([p["id"] for p in doc["packets"]], doc["invalid"]), ([second], []))
        self.assertEqual(len(self.cli_json("list", "--from", "sapphire", "--to", "twixtor")[1]["packets"]), 2)
        self.assertEqual(self.cli_json("list", "--from", "twixtor")[1]["packets"], [])
        code, out, _ = self.cli("list")
        self.assertIn("2 packet(s) in", out)
        self.assertIn("invalid:", out)

    def test_list_of_a_repository_without_packets(self):
        code, doc, _ = self.cli_json("list")
        self.assertEqual((code, doc["packets"], doc["invalid"]), (0, [], []))

    def test_show_prints_the_packet(self):
        packet_id, _ = self.done_packet()
        code, doc, _ = self.cli_json("show", packet_id)
        self.assertEqual((code, doc["packet"]["id"], doc["packet"]["status"]), (0, packet_id, "done"))
        code, out, _ = self.cli("show", packet_id)
        for needle in ("status:   done", "route:    sapphire -> twixtor", "command[0] exit 0", "revision 3"):
            self.assertIn(needle, out)
        code, doc, _ = self.cli_json("show", self.path(packet_id), root=False)
        self.assertEqual(doc["packet"]["id"], packet_id)

    def test_list_other_finds_the_peer_by_its_markers(self):
        mine, theirs = self.layout()
        code, sent, err = self.cli_json("new", "--from", "sapphire", "--to", "twixtor", "--task", "sent", root=theirs)
        self.assertEqual(code, 0, err)
        code, doc, _ = self.cli_json("new", "--to", "twixtor", "--task", "to itself", root=theirs)
        self.assertEqual(code, 2)
        for flags in (("--other",), ("--peer", "twixtor")):
            with self.subTest(flags):
                code, listed, err = self.cli_json("list", *flags, root=mine)
                self.assertEqual(code, 0, err)
                self.assertEqual([p["id"] for p in listed["packets"]], [sent["id"]])
        self.assertEqual(self.cli_json("list", root=mine)[1]["packets"], [])
        code, shown, _ = self.cli_json("show", sent["id"], "--other", root=mine)
        self.assertEqual(shown["packet"]["task"], "sent")
        self.assertEqual(self.cli("verify", sent["id"], "--other", root=mine)[0], 0)

    def test_new_from_a_marked_root_writes_into_the_peer(self):
        mine, theirs = self.layout()
        argv = ["new", "--to", "twixtor", "--task", "t", "--input", self.raw, "--json"]
        code, out, err = run_cli(argv + ["--self", "sapphire", "--peer-dir", f"twixtor={theirs}"])
        self.assertEqual(code, 0, err)
        doc = json.loads(out)
        self.assertEqual(os.path.dirname(doc["path"]), os.path.join(theirs, "handoff"))
        self.assertEqual(sha(handoff.resolve(doc["packet"]["inputs"][0]["path"], theirs)), sha(self.raw))

    def test_a_sibling_with_the_right_name_but_no_markers_is_not_a_peer(self):
        work = os.path.join(self.tmp, "named")
        mine = make_tool(os.path.join(work, "sapphire"), "sapphire")
        os.makedirs(os.path.join(work, "twixtor"))
        code, doc, _ = self.cli_json("list", "--other", root=mine)
        self.assertEqual(code, 6)
        self.assert_error(doc, "tool_missing", 6)
        self.assertIn("holds the markers of twixtor", doc["error"]["message"])

    def test_list_other_without_the_peer(self):
        mine, theirs = self.layout()
        shutil.rmtree(theirs)
        code, doc, _ = self.cli_json("list", "--other", root=mine)
        self.assertEqual(code, 6)
        self.assertIn("twixtor repository was not found", doc["error"]["message"])
        code, doc, _ = self.cli_json("list", "--other", "--peer-dir", "twixtor=" + os.path.join(self.tmp, "nowhere"),
                                     root=mine)
        self.assertEqual(code, 6)
        self.assertIn("points at a missing folder", doc["error"]["message"])
        self.assertEqual(self.cli("list", "--other", "--peer-dir", "nobody=x", root=mine)[0], 2)

    def test_list_other_with_an_explicit_directory(self):
        mine, _ = self.layout()
        elsewhere = os.path.join(self.tmp, "elsewhere", "any-name")
        os.makedirs(elsewhere)
        self.assertEqual(self.cli("new", "--from", "sapphire", "--to", "twixtor", "--task", "t", root=elsewhere)[0], 0)
        code, doc, err = self.cli_json("list", "--other", "--peer-dir", "twixtor=" + elsewhere, root=mine)
        self.assertEqual((code, len(doc["packets"])), (0, 1), err)
        with mock.patch.dict(os.environ, {"HANDOFF_TWIXTOR_DIR": elsewhere}):
            code, doc, err = self.cli_json("list", "--other", root=mine)
        self.assertEqual((code, len(doc["packets"])), (0, 1), err)

    def test_other_needs_a_known_self(self):
        plain = os.path.join(self.tmp, "plain")
        os.makedirs(plain)
        code, doc, _ = self.cli_json("list", "--other", root=plain)
        self.assertEqual(code, 2)
        self.assertIn("--self", doc["error"]["message"])

    def test_a_copy_of_the_script_finds_its_root_and_the_peer(self):
        """run a copy of the script from <work>/alpha/tools; the packet lands in <work>/beta."""
        mine, theirs = self.layout()
        copy_shared(mine)
        script = os.path.join(mine, "tools", "handoff.py")
        env = clean_env()
        proc = subprocess.run([sys.executable, "-I", script, "new", "--to", "twixtor", "--task", "from a copy",
                               "--json"], capture_output=True, text=True, env=env)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        doc = json.loads(proc.stdout)
        self.assertEqual(os.path.dirname(doc["path"]), os.path.join(theirs, "handoff"))
        self.assertFalse(os.path.exists(os.path.join(mine, "handoff")))
        proc = subprocess.run([sys.executable, "-I", script, "list", "--other", "--json"],
                              capture_output=True, text=True, env=env)
        self.assertEqual([p["id"] for p in json.loads(proc.stdout)["packets"]], [doc["id"]])


class LocateTests(Case):
    def test_detect_self_by_markers_flag_and_environment(self):
        base = os.path.join(self.tmp, "detect")
        self.assertEqual(handoff.detect_self(make_tool(os.path.join(base, "x1"), "twixtor"), env={}), "twixtor")
        self.assertEqual(handoff.detect_self(make_tool(os.path.join(base, "x2"), "sapphire"), env={}), "sapphire")
        both = make_tool(make_tool(os.path.join(base, "x3"), "sapphire"), "twixtor")
        self.assertIsNone(handoff.detect_self(both, env={}))
        for name in ("twixtor", "sapphire", "renamed-sapphire"):
            os.makedirs(os.path.join(base, name))
            self.assertIsNone(handoff.detect_self(os.path.join(base, name), env={}), name)
        plain = os.path.join(base, "renamed-sapphire")
        self.assertEqual(handoff.detect_self(plain, "twixtor", env={}), "twixtor")
        self.assertEqual(handoff.detect_self(plain, env={"HANDOFF_SELF": "sapphire"}), "sapphire")
        with self.assertRaises(handoff.HandoffError):
            handoff.detect_self(plain, env={"HANDOFF_SELF": "other"})

    def test_find_peer_order(self):
        work = os.path.join(self.tmp, "find")
        here = os.path.join(work, "x")
        os.makedirs(here)
        sibling = make_tool(os.path.join(work, "some-folder"), "twixtor")
        chosen, from_env = os.path.join(work, "chosen"), os.path.join(work, "from-env")
        os.makedirs(chosen)
        os.makedirs(from_env)
        self.assertEqual(handoff.find_peer("twixtor", here, env={})[0], sibling)
        env = {"HANDOFF_TWIXTOR_DIR": from_env}
        self.assertEqual(handoff.find_peer("twixtor", here, env=env)[0], from_env)
        self.assertEqual(handoff.find_peer("twixtor", here, chosen, env=env)[0], chosen)
        self.assertIsNone(handoff.find_peer("nobody", here, env={})[0])
        self.assertEqual(handoff.peer_env("twixtor"), "HANDOFF_TWIXTOR_DIR")

    def test_a_given_directory_that_is_missing_is_an_error_not_a_fall_through(self):
        work = os.path.join(self.tmp, "find2")
        here = os.path.join(work, "x")
        os.makedirs(here)
        make_tool(os.path.join(work, "t"), "twixtor")
        path, reason = handoff.find_peer("twixtor", here, os.path.join(work, "gone"), env={})
        self.assertIsNone(path)
        self.assertIn("--peer-dir twixtor= points at a missing folder", reason)
        path, reason = handoff.find_peer("twixtor", here, env={"HANDOFF_TWIXTOR_DIR": os.path.join(work, "gone")})
        self.assertIsNone(path)
        self.assertIn("HANDOFF_TWIXTOR_DIR points at a missing folder", reason)

    def test_two_siblings_with_the_markers_are_ambiguous(self):
        work = os.path.join(self.tmp, "find3")
        here = os.path.join(work, "x")
        os.makedirs(here)
        make_tool(os.path.join(work, "t1"), "twixtor")
        make_tool(os.path.join(work, "t2"), "twixtor")
        path, reason = handoff.find_peer("twixtor", here, env={})
        self.assertIsNone(path)
        self.assertIn("2 folders", reason)

    def test_relative_paths(self):
        root = self.recv
        self.assertEqual(handoff.rel_or_none(os.path.join(root, "a", "b.raw"), root), "a/b.raw")
        self.assertIsNone(handoff.rel_or_none(self.raw, root))
        self.assertIsNone(handoff.rel_or_none(root, root))
        with self.assertRaises(handoff.HandoffError):
            handoff.to_rel(self.raw, root, "x")
        self.assertEqual(handoff.program_stem("C:\\x\\FFmpeg.EXE"), "ffmpeg")
        self.assertEqual(handoff.program_stem("/usr/bin/vfx"), "vfx")


# ----------------------------------------------------------------------------
# stages and providers
# ----------------------------------------------------------------------------

class StagesTests(Case):
    def manifest(self, folder, name, command, stages, contract="frames/2"):
        path = os.path.join(folder, name)
        os.makedirs(folder, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"contract": contract, "provider": {"id": "plug", "version": "1.0", "command": command},
                       "stages": stages}, handle)
        return path

    def stage(self, stage_id, geometry="preserve"):
        return {"id": stage_id, "kind": "frame", "spool": False, "geometry": geometry, "depths": [8, 16],
                "params": [{"id": "fixture_param", "kind": "float"}], "layers": []}

    def test_vfx_list_adapter(self):
        entries = [{"id": "retime", "name": "Retime", "kind": "temporal", "hidden": False,
                    "params": [{"id": "speed_percent", "kind": "float", "default": 100.0}],
                    "layers": [{"id": "alt_source_a", "name": "Alt Source A", "optional": True}]},
                   {"id": "selftest-invert", "kind": "frame", "hidden": True, "params": [], "layers": []}]
        manifest = handoff.check_manifest(handoff.adapt_vfx_list(entries, "twixtor", "vfx 0.1.0", ["vfx"]), "x")
        self.assertEqual(manifest["contract"], "frames/1")
        retime, invert = manifest["stages"]
        self.assertEqual((retime["spool"], retime["geometry"], retime["depths"]), (True, "preserve", [8, 16, 32]))
        self.assertEqual((invert["spool"], invert["hidden"]), (False, True))
        self.assertIs(retime["layers"][0]["optional"], True)
        with self.assertRaises(handoff.HandoffError):
            handoff.adapt_vfx_list({"not": "a list"}, "twixtor", "v", ["vfx"])

    def test_a_provider_file_gives_ready_stage_commands(self):
        folder = os.path.join(self.tmp, "plug dir")
        path = self.manifest(folder, "plug.stage.json", ["bin/plug", "stage"], [self.stage("fixture-a")])
        code, doc, err = self.cli_json("stages", "--provider", path, "--no-registry")
        self.assertEqual(code, 0, err)
        row = doc["stages"][0]
        program = os.path.normpath(os.path.join(folder, "bin", "plug"))
        self.assertEqual(row["stage"], f'"{program}" stage apply fixture-a --width {{width}} --height {{height}} '
                                       f'--depth {{depth}} --fps {{fps}}')
        self.assertEqual((row["provider"], row["kind"], row["geometry"], row["depths"]), ("plug", "frame", "preserve", [8, 16]))
        self.assertEqual(doc["providers"][0]["source"], path)

    def test_handoff_providers_folders_and_their_problems(self):
        folder = os.path.join(self.tmp, "providers")
        self.manifest(folder, "a.stage.json", ["a"], [self.stage("fixture-a"), self.stage("fixture-b")])
        write_bytes(os.path.join(folder, "b.stage.json"), b"{broken")
        write_bytes(os.path.join(folder, "ignored.json"), b"{}")
        env = {"HANDOFF_PROVIDERS": folder + os.pathsep + os.path.join(self.tmp, "gone")}
        with mock.patch.dict(os.environ, env):
            code, doc, err = self.cli_json("stages", "--no-registry")
        self.assertEqual(code, 0, err)
        self.assertEqual([r["id"] for r in doc["stages"]], ["fixture-a", "fixture-b"])
        self.assertEqual(len(doc["problems"]), 2)

    def test_unknown_geometry_and_unknown_contract_major_are_refused(self):
        folder = os.path.join(self.tmp, "p")
        first = self.manifest(folder, "g.stage.json", ["g"], [self.stage("fixture-a"), self.stage("fixture-b", "stretch")])
        second = self.manifest(folder, "c.stage.json", ["c"], [self.stage("fixture-c")], contract="frames/3")
        code, doc, err = self.cli_json("stages", "--provider", first, "--provider", second, "--no-registry")
        self.assertEqual(code, 0, err)
        self.assertEqual([r["id"] for r in doc["stages"]], ["fixture-a"])
        self.assertTrue(any("stretch" in p for p in doc["problems"]))
        self.assertTrue(any("frames/3" in p for p in doc["problems"]))

    def test_an_invalid_provider_file_is_bad_input(self):
        folder = os.path.join(self.tmp, "p2")
        path = self.manifest(folder, "x.stage.json", [], [self.stage("s")])
        code, doc, _ = self.cli_json("stages", "--provider", path, "--no-registry")
        self.assertEqual(code, 4)
        self.assert_error(doc, "bad_input", 4)

    def test_registry_providers_native_and_vfx_list(self):
        root = make_tool(os.path.join(self.tmp, "reg", "s"), "sapphire")
        make_tool(os.path.join(self.tmp, "reg", "t"), "twixtor")
        native = {"contract": "frames/2", "provider": {"id": "sapphire", "version": "0.1", "command": ["nh", "stage"]},
                  "stages": [self.stage("selftest-invert")]}
        listing = [{"id": "retime", "kind": "temporal", "params": [], "layers": []}]
        calls = []

        def fake_run(command, timeout=60):
            calls.append(command)
            body = native if program_is(command, "sapphire") else listing
            return subprocess.CompletedProcess(command, 0, json.dumps(body).encode(), b""), ""

        def program_is(command, name):
            return command[0] == "/bin/" + name

        with mock.patch.object(handoff, "find_binary", lambda name, env, dirs: ("/bin/" + name, "")), \
                mock.patch.object(handoff, "run_capture", fake_run), \
                mock.patch.object(handoff, "first_line", lambda command: "vfx 0.1.0"):
            code, doc, err = self.cli_json("stages", root=root)
        self.assertEqual(code, 0, err)
        self.assertEqual(sorted(r["id"] for r in doc["stages"]), ["retime", "selftest-invert"])
        self.assertIn(["/bin/sapphire", "stage", "list", "--json"], calls)
        self.assertIn(["/bin/vfx", "list", "--json", "--all"], calls)
        retime = next(r for r in doc["stages"] if r["id"] == "retime")
        self.assertEqual((retime["contract"], retime["spool"]), ("frames/1", True))
        self.assertTrue(retime["stage"].startswith("/bin/vfx apply retime "))

    def test_a_missing_binary_is_a_problem_not_a_failure(self):
        root = make_tool(os.path.join(self.tmp, "reg2", "s"), "sapphire")
        with mock.patch.object(handoff, "find_binary", lambda name, env, dirs: (None, f"{name} not built")):
            code, doc, err = self.cli_json("stages", root=root)
        self.assertEqual((code, doc["stages"]), (0, []), err)
        self.assertEqual(len(doc["problems"]), 2)

    def test_quote_word(self):
        self.assertEqual(handoff.quote_word("plain"), "plain")
        self.assertEqual(handoff.quote_word("a b"), '"a b"')
        self.assertEqual(handoff.quote_word(""), '""')


# ----------------------------------------------------------------------------
# sync
# ----------------------------------------------------------------------------

class SyncTests(Case):
    def repos(self):
        work = os.path.join(self.tmp, "sync")
        mine = make_tool(os.path.join(work, "a"), "sapphire")
        theirs = make_tool(os.path.join(work, "b"), "twixtor")
        copy_shared(mine)
        return mine, theirs

    def test_sync_copies_every_shared_file_and_confirms_sha256(self):
        mine, theirs = self.repos()
        code, doc, err = self.cli_json("sync", root=mine)
        self.assertEqual(code, 0, err)
        expected = handoff.shared_files(REPO, REGISTRY["shared"])
        self.assertEqual(sorted(r["path"] for r in doc["files"]), expected)
        self.assertTrue(all(r["status"] == "copied" and r["sha256"] == r["peer_sha256"] for r in doc["files"]))
        for rel in expected:
            self.assertEqual(sha(os.path.join(mine, *rel.split("/"))), sha(os.path.join(theirs, *rel.split("/"))))
        code, doc, err = self.cli_json("sync", root=mine)
        self.assertEqual((code, {r["status"] for r in doc["files"]}, doc["changed"]), (0, {"same"}, False), err)

    def test_sync_check_reports_a_difference_and_writes_nothing(self):
        mine, theirs = self.repos()
        self.assertEqual(self.cli("sync", root=mine)[0], 0)
        target = os.path.join(theirs, "tools", "handoff.py")
        with open(target, "ab") as handle:
            handle.write(b"# local edit\n")
        before = read_bytes(target)
        code, doc, _ = self.cli_json("sync", "--check", root=mine)
        self.assertEqual(code, 4)
        self.assert_error(doc, "bad_input", 4)
        self.assertEqual([r["path"] for r in doc["files"] if r["status"] == "differs"], ["tools/handoff.py"])
        self.assertEqual(read_bytes(target), before)

    def test_sync_of_named_files_only_accepts_shared_paths(self):
        mine, theirs = self.repos()
        code, doc, err = self.cli_json("sync", "tools/handoff.py", "--to", "twixtor", root=mine)
        self.assertEqual((code, [r["path"] for r in doc["files"]]), (0, ["tools/handoff.py"]), err)
        write_bytes(os.path.join(mine, "README.md"), b"x\n")
        for bad in ("README.md", "../x", "contracts/missing.json"):
            with self.subTest(bad):
                self.assertEqual(self.cli("sync", bad, root=mine)[0], 2)
        self.assertEqual(self.cli("sync", "--to", "sapphire", root=mine)[0], 2)

    def test_sync_without_the_peer_is_tool_missing(self):
        mine, theirs = self.repos()
        shutil.rmtree(theirs)
        code, doc, _ = self.cli_json("sync", root=mine)
        self.assertEqual(code, 6)
        self.assert_error(doc, "tool_missing", 6)


# ----------------------------------------------------------------------------
# the json contract and the error object
# ----------------------------------------------------------------------------

class ContractTests(Case):
    def test_every_json_result_carries_the_contract(self):
        packet_id = self.new()
        with mock.patch.object(handoff, "resolve_tool", lambda name, envs: (sys.executable, "")):
            calls = [("list",), ("show", packet_id), ("accept", packet_id), ("verify", packet_id),
                     ("migrate", packet_id), ("stages", "--no-registry"), ("doctor",)]
            for call in calls:
                with self.subTest(call[0]):
                    document = json.loads(self.cli(*call, "--json")[1])
                    self.assertEqual(document["contract"], "frames/2")
                    self.assertIn("ok", document)
        code, doc, _ = self.cli_json("contracts", root=REPO)
        self.assertEqual((doc["contract"], doc["ok"]), ("frames/2", True))

    def test_usage_errors_use_the_error_object(self):
        for argv in (["list", "--bogus"], ["accept"], ["new", "--from", "sapphire"], [], ["frobnicate"]):
            with self.subTest(argv):
                code, out, err = run_cli(argv + ["--json"])
                self.assertEqual(code, 2)
                doc = json.loads(out)
                self.assert_error(doc, "usage", 2)
                self.assertEqual((doc["error"]["stage"], doc["error"]["exit_code"]), ("args", None))
                self.assertRegex(err.strip(), r"^error: usage: ")

    def test_text_mode_error_is_one_line(self):
        code, out, err = self.cli("accept", "no-such-packet")
        self.assertEqual((code, out), (2, ""))
        self.assertEqual(len(err.strip().splitlines()), 1)
        self.assertRegex(err.strip(), r"^error: usage: no packet matches `no-such-packet` in ")

    def test_help_exits_zero(self):
        for argv in (["--help"], ["new", "--help"], ["sync", "-h"], ["stages", "-h"], ["doctor", "-h"]):
            with self.subTest(argv):
                code, out, _ = run_cli(argv)
                self.assertEqual(code, 0)
                self.assertIn("usage:", out)

    def test_invalid_packet_exits_4(self):
        packet_id = self.new()
        self.edit(packet_id, lambda p: p.update(status="bogus"))
        for command in ("show", "accept", "verify"):
            with self.subTest(command):
                code, doc, _ = self.cli_json(command, packet_id)
                self.assertEqual(code, 4)
                self.assert_error(doc, "bad_input", 4)

    def test_unknown_schema_version_is_refused(self):
        packet_id = self.new()
        self.edit(packet_id, lambda p: p.update(schema_version=3))
        code, doc, _ = self.cli_json("show", packet_id)
        self.assertEqual(code, 4)
        self.assertIn("not a readable version", doc["error"]["message"])

    def test_an_interrupt_exits_130(self):
        def interrupted(args):
            raise KeyboardInterrupt

        with mock.patch.dict(handoff.HANDLERS, {"list": interrupted}):
            code, doc, _ = self.cli_json("list")
        self.assertEqual(code, 130)
        self.assert_error(doc, "interrupted", 130)

    def test_the_script_runs_as_a_program(self):
        script = os.path.join(HERE, "handoff.py")
        env = clean_env()
        proc = subprocess.run([sys.executable, "-I", script, "list", "--json", "--root", self.recv],
                              capture_output=True, text=True, env=env)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["contract"], "frames/2")
        proc = subprocess.run([sys.executable, "-I", script], capture_output=True, text=True, env=env)
        self.assertEqual(proc.returncode, 2)
        self.assertRegex(proc.stderr, r"^error: usage: ")


# ----------------------------------------------------------------------------
# doctor
# ----------------------------------------------------------------------------

class TreeKillTests(unittest.TestCase):
    """a timeout ends the whole process tree. the grandchild holds the stderr pipe, so a kill of the
    direct child alone would leave the read waiting until the grandchild ends by itself."""
    SLEEP = 20

    def tree(self):
        inner = f"import time; time.sleep({self.SLEEP})"
        outer = ("import subprocess, sys, time; "
                 f"subprocess.Popen([sys.executable, '-c', {inner!r}], stderr=sys.stderr); time.sleep({self.SLEEP})")
        return [sys.executable, "-c", outer]

    def test_a_timeout_kills_the_children_of_the_child_too(self):
        started = time.monotonic()
        ran = handoff.execute(self.tree(), None, None, None, 1)
        self.assertTrue(ran["timed_out"])
        self.assertEqual(ran["exit_code"], handoff.TIMEOUT_EXIT)
        self.assertLess(time.monotonic() - started, self.SLEEP / 2)
        started = time.monotonic()
        proc, why = handoff.run_capture(self.tree(), timeout=1)
        self.assertIsNone(proc)
        self.assertIn("ran past", why)
        self.assertLess(time.monotonic() - started, self.SLEEP / 2)
        started = time.monotonic()
        self.assertEqual(handoff.first_line(self.tree(), timeout=1), "")
        self.assertLess(time.monotonic() - started, self.SLEEP / 2)


class DoctorTests(Case):
    def repos(self):
        work = os.path.join(self.tmp, "doctor")
        mine = make_tool(os.path.join(work, "m"), "sapphire")
        theirs = make_tool(os.path.join(work, "t"), "twixtor")
        for repo in (mine, theirs):
            copy_shared(repo)
            write_bytes(os.path.join(repo, ".gitignore"), b"target/\nhandoff/\n")
        return mine, theirs

    def doctor(self, mine, *extra, binaries=None):
        found = binaries or (lambda name, env, dirs: (None, f"{name} not built"))
        with mock.patch.object(handoff, "resolve_tool", lambda name, envs: (sys.executable, "")), \
                mock.patch.object(handoff, "find_binary", found):
            return self.cli_json("doctor", *extra, root=mine)

    def check(self, doc, name):
        found = [c for c in doc["checks"] if c["name"] == name]
        self.assertEqual(len(found), 1, name)
        return found[0]

    def test_doctor_passes_when_everything_needed_is_there(self):
        mine, theirs = self.repos()
        code, doc, err = self.doctor(mine)
        self.assertEqual(code, 0, err)
        self.assertEqual((doc["contract"], doc["ok"], doc["self"], doc["peers"]),
                         ("frames/2", True, "sapphire", {"twixtor": theirs}))
        for c in doc["checks"]:
            self.assertEqual(sorted(c), ["detail", "name", "ok", "required"])
        required = {c["name"] for c in doc["checks"] if c["required"]}
        self.assertEqual(required, {"peers", "ffmpeg", "ffprobe", "contracts-valid", "contracts-equal:twixtor"})
        self.assertTrue(all(self.check(doc, n)["ok"] for n in required))
        self.assertTrue(self.check(doc, "shared-equal:twixtor")["ok"])
        self.assertTrue(self.check(doc, "gitignore")["ok"])
        self.assertTrue(self.check(doc, "ffmpeg")["detail"].startswith(sys.executable + ": "))

    def test_doctor_text_mode(self):
        mine, _ = self.repos()
        with mock.patch.object(handoff, "resolve_tool", lambda name, envs: (sys.executable, "")), \
                mock.patch.object(handoff, "find_binary", lambda name, env, dirs: (None, "not built")):
            code, out, err = self.cli("doctor", root=mine)
        self.assertEqual(code, 0, err)
        self.assertIn("[ok]   peers:", out)
        self.assertIn("doctor: all required checks passed", out)

    def test_doctor_fails_without_a_peer(self):
        mine, theirs = self.repos()
        shutil.rmtree(theirs)
        code, doc, _ = self.doctor(mine)
        self.assertEqual(code, 6)
        self.assert_error(doc, "tool_missing", 6)
        self.assertEqual(doc["error"]["stage"], "doctor")
        self.assertFalse(self.check(doc, "peers")["ok"])

    def test_doctor_fails_when_ffmpeg_is_missing(self):
        mine, _ = self.repos()

        def tools(name, envs):
            return (None, f"{name} is not on PATH") if name == "ffmpeg" else (sys.executable, "")

        with mock.patch.object(handoff, "resolve_tool", tools), \
                mock.patch.object(handoff, "find_binary", lambda name, env, dirs: (None, "not built")):
            code, doc, _ = self.cli_json("doctor", root=mine)
        self.assertEqual(code, 6)
        self.assertFalse(self.check(doc, "ffmpeg")["ok"])
        self.assertTrue(self.check(doc, "ffprobe")["ok"])

    def test_doctor_fails_when_the_contract_copies_differ(self):
        mine, theirs = self.repos()
        with open(os.path.join(theirs, "contracts", "error.schema.json"), "ab") as handle:
            handle.write(b" ")
        code, doc, _ = self.doctor(mine)
        self.assertEqual(code, 4)
        self.assert_error(doc, "bad_input", 4)
        self.assertFalse(self.check(doc, "contracts-equal:twixtor")["ok"])

    def test_a_script_difference_is_advisory(self):
        mine, theirs = self.repos()
        with open(os.path.join(theirs, "tools", "handoff.py"), "ab") as handle:
            handle.write(b"\n")
        code, doc, err = self.doctor(mine)
        self.assertEqual(code, 0, err)
        shared = self.check(doc, "shared-equal:twixtor")
        self.assertEqual((shared["ok"], shared["required"]), (False, False))

    def test_missing_binaries_are_findings_not_crashes(self):
        mine, _ = self.repos()
        code, doc, err = self.doctor(mine)
        self.assertEqual(code, 0, err)
        for name in ("binary:vfx", "binary:sapphire"):
            found = self.check(doc, name)
            self.assertEqual((found["ok"], found["required"]), (False, False))
            self.assertIn("not built", found["detail"])

    def test_contract_versions_are_reported_and_compared(self):
        mine, _ = self.repos()
        exe = lambda name, env, dirs: (sys.executable, "")  # noqa: E731
        for reported, ok in ((("frames/2", ""), True), (("frames/9", ""), False), ((None, "no contract field"), True)):
            with self.subTest(reported), mock.patch.object(handoff, "reported_contract", lambda command, adapter: reported):
                code, doc, err = self.doctor(mine, binaries=exe)
                self.assertEqual(code, 0, err)
                found = self.check(doc, "contract:sapphire")
                self.assertEqual((found["ok"], found["required"]), (ok, False))
        self.assertEqual(handoff.reported_contract(["unused"], "vfx-list")[0], "frames/1")

    def test_the_binaries_are_looked_up_in_the_registry_dirs_of_each_root(self):
        mine, theirs = self.repos()
        vfx = os.path.join(theirs, "target", "release", handoff.exe_name("vfx"))
        built = os.path.join(mine, "target", "release", handoff.exe_name("sapphire"))
        for path in (vfx, built):
            write_bytes(path, b"")
        with mock.patch.object(handoff, "resolve_tool", lambda name, envs: (sys.executable, "")), \
                mock.patch.object(handoff, "first_line", lambda command: "fake 1.0"), \
                mock.patch.object(handoff, "reported_contract", lambda command, adapter: (None, "no contract field")):
            code, doc, err = self.cli_json("doctor", root=mine)
        self.assertEqual(code, 0, err)
        self.assertIn(vfx, self.check(doc, "binary:vfx")["detail"])
        self.assertIn(built, self.check(doc, "binary:sapphire")["detail"])

    def test_an_environment_override_that_is_missing_is_reported(self):
        with mock.patch.dict(os.environ, {"VFX_PATH": os.path.join(self.tmp, "gone")}):
            path, reason = handoff.find_binary("vfx", "VFX_PATH", [])
        self.assertIsNone(path)
        self.assertIn("VFX_PATH points at a missing file", reason)

    def test_conformance_hook(self):
        mine, _ = self.repos()
        script = os.path.join(mine, "tools", "conformance.py")
        if os.path.exists(script):
            os.remove(script)
        code, doc, err = self.doctor(mine, "--conformance")
        self.assertEqual(code, 0, err)
        self.assertIn("not run", self.check(doc, "conformance")["detail"])
        seen = os.path.join(self.tmp, "argv.txt")
        script = os.path.join(mine, "tools", "conformance.py")
        write_bytes(script, ("import sys\nopen(%r, 'w').write(' '.join(sys.argv[1:]))\n" % seen).encode())
        code, doc, err = self.doctor(mine, "--conformance")
        self.assertEqual(code, 0, err)
        self.assertTrue(self.check(doc, "conformance")["required"])
        self.assertEqual(read_bytes(seen), b"--quick")
        write_bytes(script, b"import sys\nprint('matrix failed', file=sys.stderr)\nsys.exit(1)\n")
        code, doc, _ = self.doctor(mine, "--conformance")
        self.assertEqual(code, 1)
        self.assert_error(doc, "stage_failed", 1)
        self.assertEqual(doc["error"]["exit_code"], 1)
        self.assertIn("matrix failed", self.check(doc, "conformance")["detail"])
        code, doc, _ = self.doctor(mine)
        self.assertNotIn("conformance", [c["name"] for c in doc["checks"]])


# ----------------------------------------------------------------------------
# the real repositories (no skips: an absent peer must be explained, or fails with HANDOFF_REQUIRE_PEERS=1)
# ----------------------------------------------------------------------------

def real_peers():
    me = handoff.detect_self(REPO, env=REAL_ENV)
    found, missing = {}, {}
    for tool_id in handoff.tool_ids():
        if tool_id == me:
            continue
        path, reason = handoff.find_peer(tool_id, REPO, env=REAL_ENV)
        (found if path else missing)[tool_id] = path or reason
    return me, found, missing


class RealRepositoryTests(Case):
    def peers_or_explained(self):
        me, found, missing = real_peers()
        self.assertIsNotNone(me, f"{REPO} holds no registered tool's markers")
        if REQUIRE_PEERS:
            self.assertEqual(missing, {}, "HANDOFF_REQUIRE_PEERS=1 and a peer is absent")
        for tool_id, reason in missing.items():
            print(f"[peer absent] {tool_id}: {reason}", file=sys.stderr)
            self.assertRegex(reason, r"markers|missing folder|folders next to")
        return me, found

    def test_this_repository_is_a_registered_tool(self):
        me = handoff.detect_self(REPO, env={})
        self.assertIn(me, handoff.tool_ids())

    def test_shared_files_are_byte_identical_in_every_found_peer(self):
        me, found = self.peers_or_explained()
        files = handoff.shared_files(REPO, REGISTRY["shared"])
        self.assertIn("tools/handoff.py", files)
        for tool_id, peer in found.items():
            for rel in files:
                with self.subTest(peer=tool_id, file=rel):
                    theirs = os.path.join(peer, *rel.split("/"))
                    self.assertTrue(os.path.isfile(theirs), f"{tool_id} has no copy of {rel}")
                    self.assertEqual(sha(os.path.join(REPO, *rel.split("/"))), sha(theirs), rel)

    def test_doctor_on_the_real_repositories(self):
        me, found = self.peers_or_explained()
        flags = [f"--peer-dir={tool_id}={path}" for tool_id, path in found.items()]
        code, out, err = run_cli(["doctor", "--json", "--root", REPO] + flags)
        document = json.loads(out)
        self.assertEqual((document["contract"], document["self"]), ("frames/2", me))
        self.assertTrue(self.check_named(document, "contracts-valid")["ok"])
        if found:
            for tool_id in found:
                self.assertTrue(self.check_named(document, f"contracts-equal:{tool_id}")["ok"])
        self.assertEqual(code, 0 if document["ok"] else handoff.EXIT[document["error"]["code"]])

    def check_named(self, document, name):
        return next(c for c in document["checks"] if c["name"] == name)

    def test_invert_twice_through_the_real_vfx_and_replay_without_trust(self):
        me, found = self.peers_or_explained()
        twixtor = REPO if me == "twixtor" else found.get("twixtor")
        if twixtor is None:
            return
        tool = handoff.tool_entry("twixtor")
        vfx, reason = handoff.tool_binary(tool, tool["binaries"][0], twixtor)
        if vfx is None:
            self.assertFalse(REQUIRE_PEERS, f"vfx is not built: {reason}")
            print(f"[vfx absent] {reason}", file=sys.stderr)
            return
        opaque = bytearray()
        for i in range(4 * 2 * 3):
            opaque += bytes(((i * 5) % 256, (i * 11 + 1) % 256, (i * 17 + 2) % 256, 255))
        write_bytes(self.raw, bytes(opaque))
        packet_id = self.new()
        self.assertEqual(self.cli("accept", packet_id)[0], 0)
        once, twice = os.path.join(self.recv, "once.raw"), os.path.join(self.recv, "twice.raw")
        flags = ["--width", "4", "--height", "2", "--depth", "8", "--fps", "24/1"]
        spec = self.spec_file("vfx.json", [
            {"argv": [vfx, "apply", "selftest-invert", *flags], "stdin": self.input_path(packet_id), "stdout": once},
            {"argv": [vfx, "apply", "selftest-invert", *flags], "stdin": once, "stdout": twice}])
        code, doc, err = self.cli_json("done", packet_id, "--output", twice, "--run", spec)
        self.assertEqual((code, doc["status"]), (0, "done"), err)
        self.assertEqual(sha(twice), sha(self.raw))
        self.assertNotEqual(sha(once), sha(self.raw))
        code, doc, err = self.cli_json("verify", packet_id, "--replay", "--peer-dir", f"twixtor={twixtor}")
        self.assertEqual(code, 0, err)
        self.assertEqual(doc["status"], "verified")
        self.assertEqual([c["exit_code"] for c in doc["replay"]["commands"]], [0, 0])


class SourceHygieneTests(unittest.TestCase):
    def test_shared_files_are_ascii_without_dashes_and_end_with_a_newline(self):
        for rel in handoff.shared_files(REPO, REGISTRY["shared"]):
            with self.subTest(rel):
                data = read_bytes(os.path.join(REPO, *rel.split("/")))
                data.decode("ascii")
                self.assertTrue(data.endswith(b"\n"))
                self.assertNotIn(b"\r", data)

    def test_docs_handoff_is_short_and_has_no_source_lines(self):
        with open(os.path.join(REPO, DOCS_REL), encoding="ascii") as handle:
            text = handle.read()
        self.assertLess(len(text.splitlines()), 400)
        sections = re.findall(r"^## .+$", text, re.M)
        self.assertGreaterEqual(len(sections), 8)
        self.assertIsNone(re.search(r"^Source:", text, re.M))
        self.assertNotIn("AGENTS.md", text)
        prose = re.sub(r"```.*?```", "", re.sub(r"`[^`\n]*`", "", text), flags=re.S)
        self.assertIsNone(re.search(r"\b(I|we|my|our|us)\b", prose))


if __name__ == "__main__":
    unittest.main(verbosity=2)
