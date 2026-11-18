#!/usr/bin/env python3
"""Preview or export an ECM-ELF v3 configuration as v4 (Python 3.11+)."""

import argparse
import json
from pathlib import Path
import re
import sys
import tomllib


SCHEMA = Path(__file__).resolve().parents[2] / "config" / "ecm_options.json"
V3_KEYS = Path(__file__).with_name("config_v3_keys.json")
TABLE = re.compile(r"\[([^\]]+)\](\s*(?:#.*)?)")
KEY_PART = r"""[A-Za-z0-9_-]+|"(?:[^"\\]|\\.)*"|'[^']*'"""
KEY_PATH = rf"(?:{KEY_PART})(?:\s*\.\s*(?:{KEY_PART}))*"
ASSIGNMENT = re.compile(rf"({KEY_PATH})(\s*=\s*)(.*)")
WORKER_TABLE = re.compile(r"worker\.([1-9][0-9]{0,6})(?:\.(.+))?")
GUI_UI_NAMES = {
    "language": "display_language",
    "font_size": "body_font_size",
    "dock": "workspace_layout",
    "worker": "selected_worker",
    "stage2_tune_file": "stage2_timing_profile",
}


def current_options(schema):
    options = {}
    for item in schema["options"]:
        owner = item["owner"]
        table = "worker.1" if owner == "worker" else "shared" if item.get("shared_with") else owner
        options[owner, item["key"]] = (item.get("toml_table", table), item.get("toml_key", item["key"]))
    return options


def load_toml(source, path):
    try:
        return tomllib.loads(source)
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"{path}: {error}") from error


def literal(value):
    if isinstance(value, (str, int, float, bool)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    raise ValueError("GUI preferences must be strings, numbers or booleans")


def key_path(key):
    parts = []
    for part in re.findall(KEY_PART, key):
        parts.append(tomllib.loads(f"value = {part}")["value"] if part[0] in "'\"" else part)
    return ".".join(parts)


def sidecar_entries(source, path, supported_keys):
    data = load_toml(source, path)
    for group, fields in data.items():
        if group == "state":
            continue
        if group not in ("ui", "window", "pet", "worker") or not isinstance(fields, dict):
            raise ValueError(f"{path}: unknown GUI preference table [{group}]")
        for key, value in fields.items():
            table = "gui"
            if group == "ui":
                new_key = GUI_UI_NAMES.get(key, key)
            elif group == "worker":
                if key == "count":
                    if value == 0:
                        continue
                    table, new_key = "shared", "worker_count"
                else:
                    new_key = key
            else:
                new_key = f"{group}_{key}"
            if new_key not in supported_keys:
                raise ValueError(f"{path}: unknown GUI preference {group}.{key}")
            yield table, new_key, literal(value)


def destination(section, source_key, v3_keys, options):
    full = section + "." + source_key
    if worker := WORKER_TABLE.fullmatch(full):
        number, suffix = worker.groups()
        if int(number) > 1_000_000:
            raise ValueError("worker ID must be in 1..1000000")
        if suffix.split(".", 1)[0] in ("shared", "stage1", "stage2"):
            table, key = suffix.split(".", 1)
        else:
            table, key = "worker.1", suffix
        worker_table = f"worker.{number}"
    else:
        table, key = full.split(".", 1)
        worker_table = ""
    option = v3_keys.get(table, {}).get(key)
    if option is None:
        raise ValueError(f"unknown v3 option {full}")
    bind = tuple(option)
    if bind == ("stage1", "stage1_threads"):
        return [(worker_table or "stage1", key, False) for key in ("stage1_cpu_threads", "gpu_init_threads")]
    if bind == ("stage1", "affinity"):
        return [(worker_table or "shared", "cpu_affinity", False)]
    if bind == ("gui", "NumWorkers"):
        return [("shared", "worker_count", False)]
    new_table, new_key = options.get(bind, ("", ""))
    if not new_key:
        raise ValueError(f"no v4 equivalent for {full}")
    return [(worker_table or new_table, new_key, True)]


def migrate(source, schema, path, *, gui_state=None):
    parsed = load_toml(source, path)
    version = parsed.get("config_version")
    if type(version) is not int or version not in (3, 4):
        raise ValueError(f"{path}: expected an ECM-ELF config_version = 3 file")
    if version == 4:
        if gui_state:
            raise ValueError(f"{path}: GUI sidecar import is only available when converting v3")
        return source

    v3_keys = json.loads(V3_KEYS.read_text(encoding="utf-8"))
    options = current_options(schema)
    sections, active, pending, header = {}, {}, [], []
    section = ""
    marker_seen = False
    lines = iter(enumerate(source.splitlines(), 1))
    for number, original in lines:
        line = original.strip()
        if not line:
            pending.append(original)
            continue
        commented = line.startswith("#")
        candidate = line[1:].lstrip() if commented else line
        if not commented and (match := TABLE.fullmatch(line)):
            if not marker_seen:
                raise ValueError(f"{path}:{number}: config_version must precede the first table")
            section = key_path(match[1])
            if match[2].strip():
                pending.append(match[2].strip())
            continue
        if not (match := ASSIGNMENT.fullmatch(candidate)):
            if commented:
                pending.append(original)
                continue
            raise ValueError(f"{path}:{number}: expected an ECM-ELF v3 assignment")
        key, spacing, rhs = match.groups()
        key = key_path(key)
        if not section:
            if not commented and key == "config_version":
                header.extend(pending)
                pending.clear()
                header.append(re.sub(r"(?<=\s|=)\+?3\b", "4", original, count=1))
                marker_seen = True
                continue
            if commented:
                pending.append(original)
                continue
            raise ValueError(f"{path}:{number}: expected config_version = 3 before the first table")
        try:
            targets = destination(section, key, v3_keys, options)
        except ValueError as error:
            if commented:
                pending.append(original)
                continue
            raise ValueError(f"{path}:{number}: {error}") from error
        if not commented:
            # A v3 string array can span lines. Keep the values and their comments verbatim.
            while True:
                try:
                    tomllib.loads("value = " + rhs)
                except tomllib.TOMLDecodeError:
                    continuation = next(lines, None)
                    if continuation is None:
                        raise ValueError(f"{path}:{number}: incomplete multiline value")
                    rhs += "\n" + continuation[1]
                else:
                    break
        for table, new_key, preferred in targets:
            target = (table, new_key)
            if not commented:
                old_preferred = active.get(target)
                if old_preferred is not None and preferred == old_preferred:
                    raise ValueError(f"{path}:{number}: more than one value for [{table}] {new_key}")
                if old_preferred and not preferred:
                    continue
                active[target] = preferred
            entries = sections.setdefault(table, [])
            entries.extend((None, text) for text in pending)
            pending.clear()
            entries.append((None if commented else target, f'{"# " if commented else ""}{new_key}{spacing}{rhs}'))

    if gui_state:
        state_path, state_source = gui_state
        supported_keys = {key for table, key in options.values() if table in ("gui", "shared")}
        supported_keys.update(field["key"] for field in schema["gui_preferences"]["fields"])
        for table, key, value in sidecar_entries(state_source, state_path, supported_keys):
            target = (table, key)
            if target in active:
                continue
            sections.setdefault(table, []).append((target, f"{key} = {value}"))
            active[target] = True

    output = header
    for table, entries in sections.items():
        output.extend(("", f"[{table}]"))
        keep = {}
        for index, (target, _) in enumerate(entries):
            if target is not None:
                keep[target] = index
        output.extend(text for index, (target, text) in enumerate(entries)
                      if target is None or keep[target] == index)
    output.extend(pending)
    result = "\n".join(output).rstrip() + "\n"
    load_toml(result, path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="v3 ecm.toml")
    parser.add_argument("--gui-state", type=Path, help="optional previous GUI v2 preference file")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preview", action="store_true", help="write UTF-8 TOML to stdout")
    mode.add_argument("--output", type=Path, help="new file beside input, defaults to ecm.v4.toml")
    args = parser.parse_args()
    try:
        source = args.input.resolve()
        destination = args.output.resolve() if args.output else source.with_name(f"{source.stem}.v4{source.suffix}")
        if not args.preview and destination.parent != source.parent:
            raise ValueError("output must be beside the input to preserve relative paths")
        if not args.preview and destination == source:
            raise ValueError("output must differ from the input")
        raw = source.read_bytes()
        original = raw.decode("utf-8-sig")
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        gui_state = (args.gui_state, args.gui_state.read_text(encoding="utf-8-sig")) if args.gui_state else None
        migrated = migrate(original, schema, source, gui_state=gui_state)
        if args.preview:
            sys.stdout.buffer.write(migrated.encode("utf-8"))
        elif migrated == original:
            print(f"Already v4: {source}")
        else:
            newline = "\r\n" if b"\r\n" in raw else "\n"
            output = destination.open("xb")
            try:
                with output:
                    output.write(migrated.replace("\n", newline).encode("utf-8"))
            except BaseException:
                destination.unlink()
                raise
            print(f"Wrote {destination}")
    except (OSError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
