#!/usr/bin/env python3
"""Preview or export an ECM-ELF configuration upgrade (Python 3.11+)."""

import argparse
from decimal import Decimal
import json
import os
from pathlib import Path
import re
import sys
import tomllib


SCHEMA = Path(__file__).resolve().parents[2] / "config" / "ecm_options.json"
ASSIGNMENT = re.compile(r"([A-Za-z_][A-Za-z_0-9]*)(\s*=\s*)(.*)")
VERSION = re.compile(r"""(?:config_version|"config_version"|'config_version')\s*=\s*(.*)""")
TABLE = re.compile(r"\[([^\]]+)\](\s*(?:#.*)?)")
WORKER = re.compile(r"workers\.([1-9][0-9]{0,6})(?:\.(shared|stage1|stage2))?")
DECIMAL_NUMBER = re.compile(r"[+-]?(?:0|[1-9](?:_?[0-9])*)(?:\.[0-9](?:_?[0-9])*)?(?:[eE][+-]?[0-9](?:_?[0-9])*)?")
UNSIGNED_NUMBER = re.compile(r"\+?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE]([+-]?[0-9]+))?")
GLOBAL_TABLES = {"shared", "stage1", "stage2", "gui"}


class MigrationError(ValueError):
    pass


def options_by_key(schema):
    return {
        ("shared" if item.get("shared_with") else item["owner"], item["key"]): item
        for item in schema["options"] if not item.get("v3_only")
    }


def split_comment(text):
    quote = ""
    escaped = False
    for i, ch in enumerate(text):
        if escaped:
            escaped = False
        elif quote == '"' and ch == "\\":
            escaped = True
        elif quote and ch == quote:
            quote = ""
        elif not quote and ch in "\"'":
            quote = ch
        elif not quote and ch == "#":
            value = text[:i].rstrip()
            return value, text[len(value):]
    return text, ""


def unsigned_mode(value):
    match = UNSIGNED_NUMBER.fullmatch(value)
    if len(value) > 128 or not match or (match[1] and abs(int(match[1])) > 1000):
        raise MigrationError("expected auto or an integer")
    return Decimal(value)


def convert_value(value, raw, option):
    """Retain the source value spelling except for declared conversions."""
    parser = option.get("parser_override", option["parser"])
    declared = option["type"]
    if parser == "auto_integer":
        declared = "int|string"
    elif parser in ("log_level", "gui_font"):
        declared += "|string"

    is_integer = type(value) is int
    is_bool = type(value) is bool
    is_number = is_integer or isinstance(value, Decimal)
    allowed = {
        "string": isinstance(value, str),
        "int": is_integer, "u32": is_integer, "u64": is_integer, "i64": is_integer,
        "float": is_number, "double": is_number,
        "int|string": is_number or isinstance(value, str),
        "float|string": is_number or isinstance(value, str),
        "bool": is_bool or (is_integer and value in (0, 1)),
        "bool|string": is_bool or (is_integer and value in (0, 1)) or isinstance(value, str),
        "rect": isinstance(value, list) and all(type(x) is int for x in value),
    }
    if not allowed.get(declared, False):
        raise MigrationError(f"expected v1 type {declared}")

    # The worker parser accepts decimal scalar syntax. Keep converted output
    # readable by the same parser even when tomllib accepts a broader syntax.
    if is_number and not DECIMAL_NUMBER.fullmatch(raw.strip()):
        raise MigrationError("expected a decimal number supported by the worker")
    if parser == "enum" and value not in option["values"]:
        raise MigrationError(f"expected {option['domain']}")
    if declared == "rect":
        try:
            json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MigrationError("expected a v1 integer array") from exc

    conversion = option.get("v2_conversion", "")
    if not conversion:
        if declared in ("bool", "bool|string") and is_integer:
            return "true" if value else "false"
        if declared == "bool|string" and isinstance(value, str):
            if value in ("0", "1", "false", "true"):
                return "true" if value in ("1", "true") else "false"
            if value != "auto":
                raise MigrationError("expected auto, true or false")
        return raw
    if conversion == "bool01":
        if value not in (False, True, 0, 1) or (
            parser == "optional_integer" and raw.lstrip().startswith("-")
        ):
            raise MigrationError("expected 0 or 1")
        return "true" if value else "false"
    if conversion == "nonzero_bool":
        # The old GUI falls back to its default when stoll cannot parse the
        # integer, and treats every other nonzero autostart value as enabled.
        if not -(1 << 63) <= value < (1 << 63):
            value = option["default"]
        return "true" if value else "false"
    if conversion.startswith("enum:"):
        if value == "auto" and parser == "auto_integer":
            value = 0
        elif parser == "auto_integer":
            value = unsigned_mode(value.strip() if isinstance(value, str) else raw.strip().replace("_", ""))
        elif parser == "optional_integer":
            value = unsigned_mode(raw.strip().replace("_", ""))
        names = conversion[5:].split("|")
        if any(value == i for i in range(len(names))):
            return json.dumps(names[int(value)], ensure_ascii=False)
        raise MigrationError(f"expected legacy mode 0..{len(names) - 1}")
    raise MigrationError(f"unknown schema conversion {conversion}")


def config_version(data, source, path):
    version = data.get("config_version", 1)
    if type(version) is not int or version not in (1, 2, 3):
        raise MigrationError(f"{path}: config_version must be integer 1, 2 or 3")
    if "config_version" not in data:
        return 1
    for number, original in enumerate(source.split("\n"), 1):
        line = original.strip()
        if line.startswith("["):
            break
        if marker := VERSION.fullmatch(line):
            value, _ = split_comment(marker[1])
            if value.strip().removeprefix("+") == str(version):
                return version
            raise MigrationError(f"{path}:{number}: config_version must be decimal {version}")
    raise MigrationError(f"{path}: config_version must be declared before the first table")


def migrate(source, schema, path):
    try:
        data = tomllib.loads(source)
    except tomllib.TOMLDecodeError as exc:
        raise MigrationError(f"{path}: {exc}") from exc
    version = config_version(data, source, path)
    if version == 2:
        return source
    if version == 3:
        raise MigrationError(f'{path}: cannot downgrade configuration v3 to v2')

    options = options_by_key(schema)
    output_sections = {}
    seen = {}
    pending = []
    prefix = []
    old_table = new_table = ""
    owner = worker_prefix = ""
    for number, original in enumerate(source.split("\n"), 1):
        original = original.removesuffix("\r")
        line = original.lstrip()
        if not line or line.startswith("#"):
            pending.append(original)
            continue

        if table := TABLE.fullmatch(line):
            previous_table = old_table
            old_table = table[1].strip()
            suffix = table[2]
            worker_prefix = ""
            if old_table in GLOBAL_TABLES:
                owner = new_table = old_table
            elif (worker := WORKER.fullmatch(old_table)) and int(worker[1]) <= 1_000_000:
                new_table = "worker." + worker[1]
                owner = worker[2] or "worker"
                if worker[2]:
                    worker_prefix = worker[2] + "."
            else:
                raise MigrationError(f"{path}:{number}: unknown v1 table [{old_table}]")
            if new_table in output_sections:
                output_sections[new_table].extend(pending)
                if suffix.strip():
                    output_sections[new_table].append(suffix.strip())
            else:
                if not previous_table:
                    prefix.extend(pending)
                    pending.clear()
                output_sections[new_table] = [*pending, f"[{new_table}]{suffix}"]
            pending.clear()
            continue

        marker = VERSION.fullmatch(line)
        if not old_table and marker:
            prefix.extend(pending)
            pending.clear()
            _, comment = split_comment(marker[1])
            prefix.append(f"config_version = 2{comment}")
            continue
        assignment = ASSIGNMENT.fullmatch(line)
        if not assignment:
            raise MigrationError(f"{path}:{number}: expected a v1 key and a single-line value")
        old_key, spacing, rhs = assignment.groups()

        option = options.get((owner, old_key))
        if not option:
            qualified = f"{old_table}.{old_key}" if old_table else old_key
            raise MigrationError(f"{path}:{number}: no v2 mapping for {qualified}")
        new_key = worker_prefix + option.get("v2_key", old_key)
        identity = new_table, new_key
        if identity in seen:
            raise MigrationError(f"{path}:{number}: {new_table}.{new_key} also defined on line {seen[identity]}")
        seen[identity] = number
        value, comment = split_comment(rhs)
        try:
            decoded = tomllib.loads("value = " + value, parse_float=Decimal)["value"]
            converted = convert_value(decoded, value, option)
        except ValueError as exc:
            raise MigrationError(f"{path}:{number}: {old_table}.{old_key}: {exc}") from exc
        indent = original[:len(original) - len(line)]
        output_sections[new_table].extend(pending)
        output_sections[new_table].append(f"{indent}{new_key}{spacing}{converted}{comment}")
        pending.clear()

    if "config_version" not in data:
        prefix.append("config_version = 2")
        if output_sections:
            prefix.append("")
    if not output_sections:
        prefix.extend(pending)
    else:
        output_sections[new_table].extend(pending)
    lines = prefix
    for section in output_sections.values():
        if lines and lines[-1] and section and section[0]:
            lines.append("")
        lines.extend(section)
    migrated = "\n".join(lines).rstrip("\n") + "\n"
    try:
        tomllib.loads(migrated)
    except tomllib.TOMLDecodeError as exc:
        raise MigrationError(f"{path}: v2 paths conflict: {exc}") from exc
    return migrated


def upgrade_v3(source, schema, path, *, log_file=None, finished_file=None, handoff_mode=None,
               stage1_directory=None, tune_file=None):
    try:
        data = tomllib.loads(source, parse_float=Decimal)
    except tomllib.TOMLDecodeError as exc:
        raise MigrationError(f"{path}: {exc}") from exc
    version = config_version(data, source, path)
    if version == 1:
        source = migrate(source, schema, path)
        data = tomllib.loads(source, parse_float=Decimal)

    def leaves(table, prefix=()):
        for name, value in table.items():
            key = (*prefix, name)
            if isinstance(value, dict):
                yield from leaves(value, key)
            else:
                yield key, value

    options = {
        ('shared' if o.get('shared_with') else o['owner'], o.get('v2_key', o['key'])): o
        for o in schema['options'] if not o.get('v3_only')
    }
    workers = data.get('worker', {})
    global_values = {table: {'.'.join(key): val for key, val in leaves(data.get(table, {}))}
                     for table in GLOBAL_TABLES}
    overrides = {worker: {table: {'.'.join(key): val for key, val in leaves(settings.get(table, {}))}
                          for table in ('shared', 'stage1', 'stage2')}
                 for worker, settings in workers.items()}
    def effective(worker, table, key, default=None):
        return overrides.get(worker, {}).get(table, {}).get(key, global_values[table].get(key, default))

    directory = Path(path).absolute().parent
    tune_keys = {('stage1', 'ntt.tune_file'), ('stage2', 'tune.profile')} if version < 3 else {
        ('shared', 'tune_file'), ('stage1', 'tune_file'), ('stage2', 'tune_file')
    }
    if version == 3 and tune_file is None and not any(
        'tune_file' in settings[stage]
        for settings in [global_values, *overrides.values()] for stage in ('stage1', 'stage2')
    ):
        return source

    def old_stage_tune(worker, stage):
        if version == 3:
            worker_values = overrides.get(worker, {})
            for values in (worker_values, global_values):
                for table in (stage, 'shared'):
                    if 'tune_file' in values.get(table, {}):
                        return values[table]['tune_file']
            if stage == 'stage2' and effective(worker, stage, 'tune.legacy_cost_profile', ''):
                return ''
        elif stage == 'stage1':
            # The unqualified worker spelling was ignored by the v2 runtime.
            return effective(worker, stage, 'ntt.tune_file', 'tune.toml')
        else:
            configured = effective(worker, stage, 'tune.profile')
            if configured is not None and not isinstance(configured, str):
                raise MigrationError('Stage2 tune.profile must be a path string')
            if configured:
                return configured
            if effective(worker, stage, 'tune.legacy_cost_profile', ''):
                return ''
        # Preserve the old Stage2 auto-discovery rule for comparison. Only v3
        # explicit empty paths disable persistence directly in both stages.
        if stage == 'stage2' and not (directory / 'tune.toml').is_file() and \
                (directory / 'stage2_ecm_tune.toml').is_file():
            return 'stage2_ecm_tune.toml'
        return 'tune.toml'

    def tune_identity(value):
        if not isinstance(value, str):
            raise MigrationError('tune_file must be a path string')
        return os.path.normcase(str((directory / value).resolve())) if value else ''

    def common_tune(worker):
        if tune_file is not None:
            return tune_file
        stage1, stage2 = (old_stage_tune(worker, stage) for stage in ('stage1', 'stage2'))
        if tune_identity(stage1) != tune_identity(stage2):
            scope = f'worker {worker}' if worker else 'global settings'
            raise MigrationError(f'{scope} has different CUDA tune paths (Stage1 {stage1!r}, '
                                 f'Stage2 {stage2!r}); choose --tune-file')
        return stage1

    def migrated_param(worker):
        curve = effective(worker, 'stage1', 'curve', 'mont').lower()
        target = effective(worker, 'stage1', 'target', 'gpu').lower()
        if curve == 'ed':
            return 'e'
        if curve != 'mont' or target not in ('cpu', 'gpu'):
            raise MigrationError('invalid old Stage1 curve or target')
        param = effective(worker, 'stage1', 'gpu_montgomery_param')
        if param is not None and (type(param) is not int or param not in (0, 2, 3)):
            raise MigrationError('invalid old gpu_montgomery_param')
        if target == 'cpu':
            return 'm0'
        return 'auto' if param is None else f'm{param}'

    def migrated_handoff(worker):
        enabled = effective(worker, 'stage1', 'handoff.enabled', False)
        edwards = effective(worker, 'stage1', 'handoff.edwards_only_enabled', False)
        prime95 = bool(effective(worker, 'stage1', 'prime95.worktodo_file', ''))
        if handoff_mode:
            return handoff_mode, 'ecm-elf'
        scope = f'worker {worker}' if worker else 'global Stage1'
        if enabled and prime95:
            raise MigrationError(f'{scope} enables both routes; choose --handoff-mode')
        if edwards and not enabled:
            receiver = effective(worker, 'stage1', 'handoff.edwards_only_worker', 0)
            if not prime95 or receiver or effective(worker, 'stage1', 'handoff.workers', '') or \
                    effective(worker, 'stage1', 'handoff.worktodo_file', ''):
                raise MigrationError(f'{scope} has an Edwards-only receiver; choose --handoff-mode and review the local receiver')
            return 'prime95', 'ecm-elf'
        return ('ecm-elf' if enabled else 'prime95' if prime95 else 'off'), 'keep-save'

    def common_output(key, chosen):
        if chosen is not None:
            return chosen
        if not any(effective(worker, stage, key) is not None for worker in [None, *workers]
                   for stage in ('stage1', 'stage2')):
            return None  # Adopt the v3 default when both stages left the output unset.
        explicit = []
        for worker in [None, *workers]:
            for stage in ('stage1', 'stage2'):
                value = effective(worker, stage, key)
                if value is None:
                    if key == 'log_file':
                        stem = 'screen' if stage == 'stage1' else 'stage2_screen'
                        value = f'{stem}_{worker}.log' if worker and worker != '1' else f'{stem}.log'
                    else:
                        value = 'worktodo.finished.txt' if stage == 'stage1' else 'worktodo.stage2.finished.txt'
                candidate = Path(value)
                if value and not candidate.is_absolute():
                    if stage == 'stage1' and stage1_directory is None:
                        raise MigrationError(f'{key} has a relative Stage1 path; pass --stage1-directory or --{key.replace("_", "-")}')
                    candidate = (stage1_directory if stage == 'stage1' else directory) / candidate
                explicit.append(str(candidate.resolve()) if value else '')
        if len({os.path.normcase(name) for name in explicit}) > 1:
            raise MigrationError(f'the effective {key} paths differ; choose --{key.replace("_", "-")}')
        return explicit[0] if explicit else None

    if version < 3:
        log_file = common_output('log_file', log_file)
        finished_file = common_output('finished_file', finished_file)
        if finished_file == '':
            raise MigrationError('finished_file must name a file')
        default_param = migrated_param(None)
        mode, fallback = migrated_handoff(None)
    default_tune = common_tune(None)
    worker_tunes = {worker: common_tune(worker) for worker in workers}
    sections = {}
    header_comments = []
    pending = []
    section = ()

    def emit(table, key, rhs, comments=()):
        bucket = sections.setdefault(table, [])
        bucket.extend(comments)
        bucket.append(f'{key} = {rhs}')

    for number, line in enumerate(source.splitlines(), 1):
        content = line.strip()
        if not content or content.startswith('#'):
            pending.append(line)
            continue
        if content.startswith('['):
            declaration = tomllib.loads(content + '\n__ecm_marker__ = true', parse_float=Decimal)
            section, _ = next(leaves(declaration))
            section = section[:-1]
            if section and section[0] == 'worker':
                if len(section) >= 2:
                    sections.setdefault('.'.join(section[:2]), [])
            elif section:
                sections.setdefault(section[0], [])
            _, comment = split_comment(content)
            if comment.strip():
                pending.append(comment.strip())
            continue
        lhs, separator, rhs = content.partition('=')
        if not separator:
            raise MigrationError(f'{path}:{number}: expected a single-line assignment')
        name, _ = next(leaves(tomllib.loads(lhs + '= true')))
        full = (*section, *name)
        if full == ('config_version',):
            header_comments.extend(pending)
            pending.clear()
            continue
        if full[0] == 'worker' and len(full) >= 3:
            destination = '.'.join(full[:2])
            owner = full[2] if full[2] in ('shared', 'stage1', 'stage2') else 'worker'
            source_key = '.'.join(full[3:] if owner != 'worker' else full[2:])
            if owner == 'worker' and source_key == 'ntt.tune_file':
                # Earlier hand-edited files sometimes omitted the Stage1 qualifier.
                owner = 'stage1'
        else:
            destination = owner = full[0]
            source_key = '.'.join(full[1:])
        if (owner, source_key) in tune_keys:
            comments = sections.setdefault(destination, [])
            comments.extend(pending)
            _, comment = split_comment(rhs)
            if comment.strip():
                comments.append(comment.strip())
            pending.clear()
            continue
        if version == 3:
            # Keep every unrelated v3 assignment, including arbitrary tables and
            # lexical values. Explicit subsections become dotted assignments.
            base = 2 if full[0] == 'worker' else 1
            key = '.'.join(part if re.fullmatch(r'[A-Za-z0-9_-]+', part) else
                           json.dumps(part, ensure_ascii=False) for part in full[base:])
            emit(destination, key, rhs.strip(), pending)
            pending.clear()
            continue
        option = options.get((owner, source_key))
        if not option:
            raise MigrationError(f'{path}:{number}: no v3 mapping for {".".join(full)}')
        key = option.get('v3_key', source_key)
        if key:
            scope = option.get('v3_table', owner)
            if destination.startswith('worker.'):
                key = f'{scope}.{key}' if owner != 'worker' else key
            else:
                destination = scope
            emit(destination, key, rhs.strip(), pending)
        else:
            # Keep attached comments for folded keys without retaining obsolete assignments.
            sections.setdefault(destination, []).extend(pending)
            _, comment = split_comment(rhs)
            if comment.strip():
                sections[destination].append(comment.strip())
        pending.clear()

    if tune_file is not None or tune_identity(default_tune) != tune_identity('tune.toml'):
        emit('shared', 'tune_file', json.dumps(default_tune, ensure_ascii=False))
    for worker, value in worker_tunes.items():
        if tune_identity(value) != tune_identity(default_tune):
            emit(f'worker.{worker}', 'shared.tune_file', json.dumps(value, ensure_ascii=False))
    if version < 3:
        for key, value in [('log_file', log_file), ('finished_file', finished_file)]:
            if value is not None:
                emit('shared', key, json.dumps(value, ensure_ascii=False))
        emit('stage1', 'param', json.dumps(default_param))
        emit('stage1', 'handoff.mode', json.dumps(mode))
        if mode == 'prime95':
            emit('stage1', 'handoff.unsupported', json.dumps(fallback))
        for worker in workers:
            param = migrated_param(worker)
            worker_mode, worker_fallback = migrated_handoff(worker)
            if param != default_param:
                emit(f'worker.{worker}', 'stage1.param', json.dumps(param))
            if worker_mode != mode:
                emit(f'worker.{worker}', 'stage1.handoff.mode', json.dumps(worker_mode))
            if worker_mode == 'prime95' and worker_fallback != fallback:
                emit(f'worker.{worker}', 'stage1.handoff.unsupported', json.dumps(worker_fallback))
    output = [*header_comments, 'config_version = 3']
    for table, body in sections.items():
        output.extend(['', f'[{table}]', *body])
    output.extend(pending)
    result = '\n'.join(output).rstrip() + '\n'
    try:
        tomllib.loads(result)
    except tomllib.TOMLDecodeError as exc:
        raise MigrationError(f'{path}: v3 paths conflict: {exc}') from exc
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="ecm.toml")
    parser.add_argument('--to', type=int, choices=(2, 3), default=3)
    parser.add_argument('--log-file', help='explicit shared log destination for v3')
    parser.add_argument('--finished-file', help='explicit shared finished destination for v3')
    parser.add_argument('--handoff-mode', choices=('off', 'ecm-elf', 'prime95'))
    parser.add_argument('--stage1-directory', type=Path, help='legacy Stage1 executable directory')
    parser.add_argument('--tune-file', help='use this shared CUDA tune path for all workers in v3')
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--preview", action="store_true", help="write UTF-8 TOML to stdout")
    output_mode.add_argument("--output", type=Path, help="new file beside the input")
    args = parser.parse_args()
    if args.to == 2 and args.tune_file is not None:
        parser.error('--tune-file requires v3 output')
    source = Path(os.path.abspath(args.input))
    destination = Path(os.path.abspath(args.output)) if args.output else source.with_name(
        f"{source.stem}.v{args.to}{source.suffix}"
    )
    try:
        if not args.preview and destination.parent != source.parent:
            raise MigrationError("output must be beside the input to preserve relative paths")
        if not args.preview and destination == source:
            raise MigrationError("output must differ from the input")
        raw = source.read_bytes()
        original = raw.decode("utf-8-sig")
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        migrated = migrate(original, schema, source) if args.to == 2 else upgrade_v3(
            original, schema, source, log_file=args.log_file, finished_file=args.finished_file,
            handoff_mode=args.handoff_mode, stage1_directory=args.stage1_directory, tune_file=args.tune_file)
        if args.preview:
            sys.stdout.buffer.write(migrated.encode("utf-8"))
        elif migrated == original:
            print(f"Already v{args.to}: {source}")
        else:
            newline = "\r\n" if b"\r\n" in raw else "\n"
            payload = migrated.replace("\n", newline).encode("utf-8")
            output = destination.open("xb")
            try:
                with output:
                    output.write(payload)
            except BaseException:
                destination.unlink()
                raise
            print(f"Wrote {destination}")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    main()
