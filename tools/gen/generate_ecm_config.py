#!/usr/bin/env python3
"""Generate runtime bindings, TOML templates and reference from one schema.

No third-party dependencies. --check is read-only and fails on stale output.
Generated files are committed; building/running the C++ programs needs no Python.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "config/ecm_options.json"
SELF = Path(__file__).resolve()
TYPES = {"string": "std::string", "int": "int", "u32": "uint32_t",
         "u64": "uint64_t", "i64": "int64_t", "double": "double",
         "float": "float", "bool": "bool", "rect": "std::array<int,4>"}


def literal(value):
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=True)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "{{" + ",".join(map(str, value)) + "}}"
    return str(value)


def text(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return ",".join(map(str, value))
    return str(value)


def load_schema():
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    # Shared keys have one authoritative definition and one TOML entry.
    originals = list(schema["options"])
    for option in originals:
        for owner in option.get("shared_with", []):
            schema["options"].append(dict(option, owner=owner, _mirror=True))
    if schema["schema_version"] != 2:
        raise ValueError("unsupported configuration schema")
    seen = set()
    fields = set()
    for o in schema["options"]:
        key = o["owner"], o["key"]
        field = o["owner"], o["field"]
        if key in seen or field in fields or o["type"] not in TYPES:
            raise ValueError(f"duplicate/invalid option: {key}")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", o["field"]):
            raise ValueError(f"invalid C++ field: {field}")
        if o["owner"] not in schema["owners"]:
            raise ValueError(f"invalid owner: {key}")
        seen.add(key)
        fields.add(field)
    if schema.get("aliases"):
        raise ValueError("legacy configuration aliases are not supported")
    for entry in schema["options"]:
        description = entry.get("description")
        if not isinstance(description, list) or len(description) != 2 or any(
                not isinstance(p, str) or not p.strip() for p in description):
            raise ValueError(f"description must contain English then Chinese: {entry['owner']}.{entry['key']}")
    return schema


def description_lines(entry, width=100):
    """Preserve explicit newlines and wrap prose by display columns (CJK=2).

    Description entries are ordered English first, then Chinese. Wrap each
    entry separately so translations never share a TOML comment/Markdown line.
    """
    def columns(value):
        return sum(0 if unicodedata.combining(c) else
                   2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in value)

    for paragraph in entry["description"]:
        for explicit_line in paragraph.splitlines():
            current = ""
            # Keep identifiers/English words intact; Chinese permits character breaks.
            tokens = re.findall(r"[^\s\u2e80-\u9fff\uff00-\uffef]+|[\u2e80-\u9fff\uff00-\uffef]|\s+", explicit_line)
            for token in tokens:
                if columns(current + token) > width and current.strip():
                    yield current.rstrip()
                    current = ""
                # Split even an unusually long URL/identifier to bound the line width.
                for char in token.lstrip() if not current else token:
                    if columns(current + char) > width:
                        yield current.rstrip()
                        current = ""
                    current += char
            yield current.rstrip()


def assignment(o, key=None):
    key = key or o["key"]
    dst = "c." + o["field"]
    kind = o.get("parser_override", o["parser"])
    lo = str(o.get("minimum", 0))
    hi = str(o.get("maximum", "UINT64_MAX"))
    if kind == "legacy":
        code = f"assign_value({dst},v);"
        if o.get("value_aliases"):
            code = f"{dst}=canonical_{o['owner']}_{o['key']}(v);"
    elif kind == "gpu_param":
        valid="&&".join(f"{dst}!={x}" for x in o["values"])
        code = f"assign_value({dst},v);if({valid}){{std::fprintf(stderr,\"[ecm] WARNING: {key}=%d unsupported; using {o['default']}\\n\",{dst});{dst}={o['default']};}}"
    elif kind == "integer_bool":
        code = f"{dst}=boolean(v,{literal(key)});"
    elif kind == "signed_integer":
        code = f"{dst}=signed_integer(v,{literal(key)},{lo},{hi});"
    elif kind == "nonnegative":
        code = f"{dst}=bounded_integer(v,{literal(key)},{lo},{hi});"
    elif kind == "positive":
        code = f"{dst}=positive(v,{literal(key)});"
    elif kind == "optional_integer":
        code = f"{dst}=v.empty()?0:static_cast<{TYPES[o['type']]}>(bounded_integer(v,{literal(key)},{lo},{hi}));"
    elif kind == "log_level":
        code = f"{dst}=log_level(v);"
    elif kind == "gui_font":
        code = f"{dst}=gui_font(v,{lo},{hi});"
    elif kind == "gui_snap":
        code = f'{dst}=!(v=="0"||v=="off"||v=="false"||v=="no");'
    elif kind == "gui_enum":
        expr = "||".join("v==" + literal(x) for x in o["values"])
        code = f"{dst}=({expr})?v:{literal(o['default'])};"
    elif kind == "gui":
        if o["type"] == "int":
            code = f"{dst}=gui_integer(v,{literal(o['default'])},{o.get('minimum','INT32_MIN')},{o.get('maximum','INT32_MAX')});"
        elif o["type"] == "i64":
            code = f"{dst}=gui_long(v,{literal(o['default'])},{lo},{hi});"
        elif o["type"] == "rect":
            code = f"gui_rect({dst},v);"
        else:
            code = f"{dst}=v;"
    elif kind == "strict":
        if o["type"] == "string":
            code = f"{dst}=v;"
        elif o["type"] == "bool":
            code = f"{dst}=boolean(v,{literal(key)});"
        else:
            code = f"{dst}=static_cast<{TYPES[o['type']]}>(bounded_integer(v,{literal(key)},{lo},{hi}));"
    else:
        raise ValueError(f"unsupported parser: {kind}")
    if o.get("present"):
        code += "c." + o["present"] + "=true;"
    return code


def defaults_header(schema):
    out = ["// Generated by tools/gen/generate_ecm_config.py; edit config/ecm_options.json.",
           "#pragma once", "#include <cstdint>", "#include <array>", "namespace ecm_config { namespace defaults {"]
    for o in schema["options"]:
        name=f"{o['owner']}_{o['key']}"
        if o["type"] == "string":
            out.append(f"inline constexpr char {name}[]={literal(o.get('effective_default',o['default']))};")
        else:
            out.append(f"inline constexpr {TYPES[o['type']]} {name}={literal(o['default'])};")
    return "\n".join(out + ["} } // namespace ecm_config::defaults", ""])


def header(schema):
    out = ["// Generated by tools/gen/generate_ecm_config.py; edit config/ecm_options.json.",
           "#pragma once", '#include "../../../../io/config_format.h"',
           '#include "../../../platform/generated/configuration_defaults.h"', "#include <array>",
           "namespace ecm_config {", f"inline constexpr int layout_version={schema['layout_version']};"]
    out.append("namespace limits {")
    for o in schema["options"]:
        for prop in ("minimum", "maximum"):
            if prop in o:
                out.append(f"inline constexpr int64_t {o['owner']}_{o['key']}_{prop}={o[prop]};")
    out.append("} // namespace limits")
    for o in schema["options"]:
        if not o.get("value_aliases"):
            continue
        out.append(f"inline std::string canonical_{o['owner']}_{o['key']}(const std::string &value){{")
        out.append("    const auto v=lower(value);")
        for canonical, aliases in o["value_aliases"].items():
            expr="||".join("v=="+literal(x) for x in aliases)
            out.append(f"    if({expr})return {literal(canonical)};")
        out.extend(["    return v;", "}"])
    index = {(o["owner"], o["key"]): o for o in schema["options"]}
    for owner, cls in schema["owners"].items():
        opts = [o for o in schema["options"] if o["owner"] == owner]
        out.append(f"struct {cls} {{")
        for o in opts:
            out.append(f"    {TYPES[o['type']]} {o['field']}={literal(o['default'])};")
        for flag in dict.fromkeys(o["present"] for o in opts if "present" in o):
            out.append(f"    bool {flag}=false;")
        out.extend(["};", f"inline const Binding<{cls}> {owner}_bindings[]={{"])
        for o in opts:
            copy = f"c.{o['field']}=v.{o['field']};"
            if o.get("present"):
                copy += f"c.{o['present']}=v.{o['present']};"
            out.append(f"    {{{literal(o['key'])},[]({cls}&c,const std::string&v){{{assignment(o)}}},[]({cls}&c,const {cls}&v){{{copy}}}}},")
        out.extend(["};", f"inline {cls} read_{owner}(const Entries &entries){{{cls} c;apply(c,entries,{owner}_bindings);return c;}}"])
    out.append("} // namespace ecm_config")
    return "\n".join(out) + "\n"


def annotation(o):
    default = o.get("default_display", o.get("effective_default", o["default"]))
    if o.get("parser_override") == "integer_bool" or (isinstance(default, bool) and o["domain"] == "[0|1]"):
        default = int(default)
    if "implicit" in o:
        default = "@" + o["implicit"]
    parts = [f"default={text(default) if default != '' else chr(34)*2}"]
    for prop in ("unit", "empty", "zero", "path_base"):
        if prop in o:
            parts.append(f"{prop}={o[prop]}")
    if o.get("worker_suffix"):
        parts.append("worker(N>1)=_N")
    if "release_value" in o:
        parts.append("release=" + text(o["release_value"]))
    if "first_run_value" in o:
        parts.append("first_run=" + text(o["first_run_value"]))
    parts += o.get("rules", [])
    for canonical, aliases in o.get("value_aliases", {}).items():
        parts.append("[" + "|".join(aliases) + "]=>" + canonical)
    return "; ".join(parts)


def config_table(option):
    if option["owner"] == "worker":
        return "workers.1"
    return "shared" if option.get("shared_with") else option["owner"]


def toml_literal(option):
    value = option["default"]
    if option["type"] == "string":
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def toml_template(schema, example):
    out = ["# GENERATED from config/ecm_options.json / 由 config/ecm_options.json 生成",
           "# See ECM_TOML_REFERENCE.md for ranges and worker overrides.",
           "# 取值范围和 worker 覆盖规则见 ECM_TOML_REFERENCE.md。", ""]
    originals = [o for o in schema["options"] if not o.get("_mirror")]
    for table in ("shared", "stage1", "stage2", "gui", "workers.1"):
        out.extend([f"[{table}]", ""])
        for option in sorted((o for o in originals if config_table(o) == table), key=lambda o: o["key"].lower()):
            en, zh = option["description"]
            setting = f"{option['key']} = {toml_literal(option)}"
            if option.get("present"):
                out.append("# " + setting)
            else:
                out.append(setting)
            out.extend("# EN: " + line for line in description_lines({"description": [en]}))
            out.extend("# 中文：" + line for line in description_lines({"description": [zh]}))
            if option.get("present"):
                out.extend(["# EN: Uncommenting marks this value explicit; omission preserves implicit worker/path/environment behavior.",
                            "# 中文：取消注释会将此值标记为显式设置；保留注释则沿用 worker、路径或环境变量的隐式规则。"])
            out.append("")
    if example:
        out.extend(["# Worker overrides / worker 覆盖示例：",
                    "# [workers.1.shared]", "# priority = 1",
                    "# EN: Worker 1 selects Stage1 tasks first.",
                    "# 中文：worker 1 优先选取 Stage1 任务。", "",
                    "# [workers.1.stage1]", "# device = 0",
                    "# EN: Worker 1 uses GPU device 0 for Stage1.",
                    "# 中文：worker 1 的 Stage1 使用 GPU 设备 0。", "",
                    "# [workers.1.stage2]", "# stage2_device = 1",
                    "# EN: Worker 1 uses GPU device 1 for Stage2.",
                    "# 中文：worker 1 的 Stage2 使用 GPU 设备 1。", ""])
    return "\n".join(out).rstrip() + "\n\n"


def toml_reference(schema):
    out = ["# ecm.toml Reference / ecm.toml 配置参考", "",
           "<!-- GENERATED: config/ecm_options.json; do not edit. -->", "",
           "Stage1, Stage2 and the GUI read the same `ecm.toml` file. Strings are quoted, booleans are `true`/`false`, and `window` is an integer array.", "",
           "Stage1、Stage2 和 GUI 共用 `ecm.toml`。字符串须加引号，布尔值使用 `true`/`false`，`window` 使用整数数组。", "",
           "## Tables and precedence / 分区与优先级", "",
           "- `[shared]`: settings used by both stages. / 两阶段共用。",
           "- `[stage1]`, `[stage2]`: stage-specific settings. / 阶段专用。",
           "- `[gui]`: GUI-wide settings; these do not have worker overrides. / GUI 全局设置，不支持 worker 覆盖。",
           "- `[workers.N]`: GUI settings for worker N (1..1000000). / 第 N 个 GUI worker 的设置。",
           "- `[workers.N.shared]`, `[workers.N.stage1]`, `[workers.N.stage2]`: worker overrides for the matching table. / 对应分区的 worker 覆盖。", "",
           "Precedence: built-in default < global table < matching worker table < CLI. A key may appear only once in a table; unknown tables, duplicate keys and invalid TOML values are errors. `worktodo.txt` retains its separate `[Worker #N]` queue sections. INI files are not read.", "",
           "优先级：内置默认值 < 全局分区 < 对应 worker 分区 < 命令行。同一分区中的键不能重复；未知分区、重复键和无效 TOML 值会报错。`worktodo.txt` 继续使用独立的 `[Worker #N]` 队列分区。程序不读取 INI。", "",
           "Compute programs skip GUI-only values in `[gui]` and `[workers.N]`, warning about unknown keys in those tables. The exception is `gui.NumWorkers`: it remains validated and available as the Stage1 routing fallback when a destination queue has no worker sections. Unknown compute keys/tables, invalid compute types, malformed TOML and duplicate keys are still errors. GUI readers retain strict schema validation. This compatibility boundary does not permit arbitrary unknown tables or unsupported TOML value syntax.", "",
           "计算程序跳过 `[gui]` 和 `[workers.N]` 中的纯界面值，对其中未知键给出警告。例外是 `gui.NumWorkers`：目标队列没有 worker 分区时，Stage1 分发用它作为后备数量，仍保留并校验。计算键和分区未知、计算值类型错误、TOML 语法错误和重复键仍拒绝；GUI 读取器继续严格校验 schema。兼容范围不包括任意未知分区或尚不支持的 TOML 值语法。", "",
           "The shared worktodo path and Stage2 relative paths use the TOML directory; other Stage1 relative paths use the executable directory. CLI relative paths use the current directory.", "",
           "共用 worktodo 与 Stage2 的相对路径基于 TOML 所在目录；其他 Stage1 相对路径基于可执行文件目录；命令行相对路径基于当前目录。", ""]
    originals = [o for o in schema["options"] if not o.get("_mirror")]
    for table in ("shared", "stage1", "stage2", "gui", "workers.1"):
        out.extend([f"## [{table}]", ""])
        for option in sorted((o for o in originals if config_table(o) == table), key=lambda o: o["key"].lower()):
            worker = table in ("shared", "stage1", "stage2")
            override = f"[workers.N.{table}]" if worker else "no / 不支持"
            out.extend([f"### {option['key']}", "",
                        "```toml", f"{option['key']} = {toml_literal(option)}", "```", "",
                        f"Type / 类型: `{option['type']}`; range / 范围: `{option['domain']}`; worker override / worker 覆盖: `{override}`.", ""])
            if option.get("present"):
                out.extend(["Writing this key explicitly can override implicit worker suffixes, paths or environment settings even when its value equals the built-in default.", "",
                            "显式写入此键时，即使数值与内置默认值相同，也可能覆盖 worker 后缀、路径或环境变量的隐式规则。", ""])
            out.extend([option["description"][0], "", option["description"][1], ""])
    out.extend(["## Maintenance / 维护", "",
                "Edit `config/ecm_options.json`, then run `python tools/gen/generate_ecm_config.py`; commit the schema and generated files together.", "",
                "修改 `config/ecm_options.json` 后运行生成器，并将定义与生成物一同提交。", ""])
    return "\n".join(out)


def toml_schema_header(schema):
    out = ["// Generated by tools/gen/generate_ecm_config.py; edit config/ecm_options.json.",
           "#pragma once", "namespace ecm_config {",
           "struct TomlOptionMeta { const char *table, *key, *type, *domain; };",
           "inline constexpr TomlOptionMeta toml_options[]={"]
    for option in schema["options"]:
        if not option.get("_mirror"):
            kind = option["type"]
            if option.get("parser_override") in ("log_level", "gui_font"):
                kind += "|string"
            out.append("    {" + ",".join(literal(x) for x in (config_table(option),option["key"],kind,option.get("domain", "valid scalar"))) + "},")
    out.extend(["};", "} // namespace ecm_config", ""])
    return "\n".join(out)


def template_header(value):
    # A character initializer list preserves the UTF-8 bytes and constexpr array
    # API without MSVC 14.16's 65535-byte concatenated string-literal limit.
    lines = ["// Generated; edit config/ecm_options.json.", "#pragma once",
             "namespace ecm_config {", "inline constexpr const char default_toml[]={"]
    encoded=value.encode('utf-8')
    for offset in range(0,len(encoded),24):
        lines.append("    "+','.join(f"'\\x{byte:02x}'" for byte in encoded[offset:offset+24])+',')
    return "\n".join(lines + ["    '\\0'", "};", "}", ""])


def products(schema):
    default = toml_template(schema, False)
    files = {"src/ecm/application/configuration/generated/ecm_config_generated.h": header(schema),
             "src/ecm/platform/generated/configuration_defaults.h": defaults_header(schema),
             "src/ecm/application/configuration/generated/ecm_toml_schema.h": toml_schema_header(schema),
             "src/ecm/application/configuration/generated/ecm_toml_template.h": template_header(default),
             "config/ecm.toml.default": default,
             "config/ecm.toml.example": toml_template(schema, True),
             "docs/ECM_TOML_REFERENCE.md": toml_reference(schema)}
    hashes = {name: hashlib.sha256(content.encode()).hexdigest() for name, content in files.items()}
    for path in (SCHEMA, SELF):
        hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    files["config/ecm_config.generated.json"] = json.dumps({"schema_version": 2, "sha256": hashes}, indent=2) + "\n"
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    files = products(load_schema())
    stale = []
    for name, content in files.items():
        path = ROOT / name
        if args.check:
            # Generated output is explicitly LF in .gitattributes.
            if not path.exists() or path.read_bytes() != content.encode():
                stale.append(name)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists() or path.read_bytes() != content.encode():
                path.write_text(content, encoding="utf-8", newline="\n")
    if stale:
        print("Stale generated config; run python tools/gen/generate_ecm_config.py:\n" + "\n".join(stale), file=sys.stderr)
        return 1
    print(f"Configuration {'current' if args.check else 'generated'}: {len(files)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
