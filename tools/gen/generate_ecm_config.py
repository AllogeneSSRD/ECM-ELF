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
         "float": "float", "bool": "bool", "bool|string": "std::string", "int|string": "std::string", "rect": "std::array<int,4>"}


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
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate schema property: {key}")
            result[key] = value
        return result

    schema = json.loads(SCHEMA.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    # Shared keys have one authoritative definition and one TOML entry.
    originals = list(schema["options"])
    for option in originals:
        for owner in option.get("shared_with", []):
            schema["options"].append(dict(option, owner=owner, _mirror=True))
    if schema["schema_version"] != 2:
        raise ValueError("unsupported configuration schema")
    if schema["config_version"] != 3:
        raise ValueError("unsupported business TOML version")
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
        if o["type"] == "bool" and o.get("domain") != "[false|true|0|1]":
            raise ValueError(f"TOML boolean requires domain [false|true|0|1]: {key}")
        if o["type"] == "bool|string" and o.get("domain") != "[auto|false|true|0|1]":
            raise ValueError(f"TOML automatic boolean requires domain [auto|false|true|0|1]: {key}")
        seen.add(key)
        fields.add(field)
    if schema.get("aliases"):
        raise ValueError("legacy configuration aliases are not supported")
    paths = set()
    compute_bindings = set()
    for option in originals:
        table, key = v2_table(option), v2_key(option)
        if table in ("shared", "stage1", "stage2"):
            binding = option["key"]
            if binding in compute_bindings:
                raise ValueError(f"duplicate compute binding: {binding}")
            compute_bindings.add(binding)
        if not re.fullmatch(r"[a-z][a-zA-Z0-9_]*(?:\.[a-z][a-zA-Z0-9_]*)?", key):
            raise ValueError(f"invalid v2 key: {table}.{key}")
        path = f"{table}.{key}"
        if path in paths or any(path.startswith(prior + ".") or prior.startswith(path + ".")
                                for prior in paths):
            raise ValueError(f"conflicting v2 key: {path}")
        paths.add(path)
        conversion = option.get("v2_conversion", "")
        if conversion not in ("", "bool01", "nonzero_bool") and not re.fullmatch(
                r"enum:[a-z][a-z0-9_]*(?:\|[a-z][a-z0-9_]*)+", conversion):
            raise ValueError(f"invalid v2 conversion: {path}")
        if option.get("template_level", "advanced") not in ("core", "advanced", "legacy"):
            raise ValueError(f"invalid template level: {path}")
    for entry in schema["options"]:
        description = entry.get("description")
        if not isinstance(description, list) or len(description) != 2 or any(
                not isinstance(p, str) or not p.strip() for p in description):
            raise ValueError(f"description must contain English then Chinese: {entry['owner']}.{entry['key']}")
    return schema


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
    elif kind == "auto_bool":
        code = f"{dst}=automatic_boolean_text(v,{literal(key)});"
    elif kind == "signed_integer":
        code = f"{dst}=signed_integer(v,{literal(key)},{lo},{hi});"
    elif kind == "nonnegative":
        code = f"{dst}=bounded_integer(v,{literal(key)},{lo},{hi});"
    elif kind == "positive":
        code = f"{dst}=positive(v,{literal(key)});"
    elif kind == "bounded_real":
        code = f"assign_value({dst},v);if({dst}<{lo}||{dst}>{hi})throw std::runtime_error({literal(key+' must be within ['+lo+','+hi+']')});"
    elif kind == "optional_integer":
        code = f"{dst}=v.empty()?0:static_cast<{TYPES[o['type']]}>(bounded_integer(v,{literal(key)},{lo},{hi}));"
    elif kind == "auto_integer":
        code = f'{dst}=v=="auto"?0:static_cast<{TYPES[o["type"]]}>(bounded_integer(v,{literal(key)},{lo},{hi}));'
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
    elif kind == "enum":
        valid="&&".join("v!="+literal(x) for x in o["values"])
        code=f"if({valid})throw std::runtime_error({literal(key+' must be '+o['domain'])});{dst}=v;"
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
        if o["type"] in ("string", "bool|string", "int|string"):
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
    if o.get("parser_override") == "integer_bool" and o["type"] != "bool":
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


def v2_table(option):
    return "worker.1" if option["owner"] == "worker" else config_table(option)


def v2_key(option):
    return option.get("v2_key", option["key"])


def v3_table(option):
    return option.get("v3_table", v2_table(option))


def v3_key(option):
    return option.get("v3_key", v2_key(option))


def v2_description_type(option):
    conversion = option.get("v2_conversion", "")
    if conversion.startswith("enum:"):
        return "string", "[" + conversion[5:] + "]"
    if conversion in ("bool01", "nonzero_bool"):
        return "bool", "[false|true]"
    kind = option["type"]
    if option.get("parser_override", option["parser"]) in ("auto_integer", "log_level", "gui_font"):
        kind += "|string"
    return kind, option["domain"]


def toml_literal(option, *, example=False):
    if example and "template_value" in option:
        value = option["template_value"]
        if not (isinstance(value, str) and value.startswith("<") and value.endswith(">")):
            return json.dumps(value, ensure_ascii=False)
    value = option["default"]
    conversion = option.get("v2_conversion", "")
    if conversion.startswith("enum:"):
        return json.dumps(conversion[5:].split("|")[value])
    if conversion in ("bool01", "nonzero_bool"):
        return "true" if value else "false"
    if option["type"] in ("string", "bool|string", "int|string"):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def wrap_toml_comment(description, width=100):
    line = ""
    columns = 2  # '# '
    space = ""
    for token in re.findall(r"\s+|[\u3000-\u9fff][，。；：！？、）】》」』]*|[^\s\u3000-\u9fff]+", description):
        if token.isspace():
            space = " "
            continue
        length = sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in token)
        if line and columns + len(space) + length > width:
            yield "# " + line
            line, columns = "", 2
        separator = space if line else ""
        line += separator + token
        columns += len(separator) + length
        space = ""
    if line:
        yield "# " + line


def toml_option_lines(option, assignment, languages=(0, 1), inline_comment=""):
    kind, domain = v2_description_type(option)
    metadata = f"{kind}: {domain}"
    if inline_comment:
        metadata += " | " + inline_comment
    yield f"{assignment}  # {metadata}"
    for language in languages:
        for paragraph in option["description"][language].splitlines():
            yield from wrap_toml_comment(paragraph)


def toml_template(schema, example):
    out = ["# GENERATED from config/ecm_options.json / 由 config/ecm_options.json 生成",
           "# ECM_TOML_REFERENCE.en.md lists advanced settings and worker overrides.",
           "# 高级设置和 worker 覆盖见 ECM_TOML_REFERENCE.zh-CN.md。",
           f"config_version = {schema['config_version']}", ""]
    originals = [o for o in schema["options"] if not o.get("_mirror")]
    for table in ("shared", "stage1", "stage2", "gui", "worker.1"):
        out.append(f"[{table}]")
        for option in (o for o in originals if v3_key(o) and v3_table(o) == table and
                       (example or o.get("template_level") == "core")):
            # Direct Stage1 uses different defaults when tmp_dir or verbose is absent.
            active = option["key"] in ("worktodo", "curve", "target", "tmp_dir", "verbose")
            assignment = ("" if active else "# ") + f"{v3_key(option)} = {toml_literal(option, example=True)}"
            out.extend(toml_option_lines(option, assignment))
            out.append("")
    if example:
        out.extend(["# Worker 1 overrides / worker 1 覆盖",
                    '# shared.queue_order = "stage1_first"', "# stage1.gpu_device = 0",
                    "# stage2.gpu_device = 0", '# shared.tune_file = "tune_gpu0.toml"', "",
                    "# [worker.2]", "# stage1.gpu_device = 1", "# stage2.gpu_device = 1",
                    '# shared.tune_file = "tune_gpu1.toml"'])
    return "\n".join(out).rstrip() + "\n\n"


REFERENCE_TEXT = {
    "en": {
        "description_index": 0,
        "title": "ecm.toml Reference",
        "other_language": "[简体中文](ECM_TOML_REFERENCE.zh-CN.md)",
        "intro": [
            "Put `config_version = 3` before the first table. A file without a version uses the legacy format. See [configuration migration](architecture/CONFIGURATION.md#升级旧配置).", "",
            "## Tables and precedence", "",
            "- `[shared]` contains settings used by both stages.",
            "- `[stage1]` and `[stage2]` contain settings specific to each stage.",
            "- `[gui]` contains GUI-wide settings. Workers cannot override them.",
            "- `[worker.N]` configures worker N (1–1000000). Prefix an override with `shared.`, `stage1.` or `stage2.`.", "",
            "From lowest to highest priority, the program applies built-in defaults, the global table, the matching worker table and CLI arguments. Duplicate keys or tables, unknown tables, missing values and unclosed strings cause errors. `worktodo.txt` uses separate `[Worker #N]` sections. The program does not read INI files.", "",
            "Each worker validates shared settings and its own stage. The GUI validates shared, GUI and worker settings and previews recognized compute options. See [configuration ownership](architecture/CONFIGURATION.md#按程序职责读取).", "",
            "The shared worktodo path and Stage2 relative paths use the TOML directory. Stage1 usually resolves its paths from the executable directory. See [runtime paths](architecture/CONFIGURATION.md#运行时限制) for details.",
        ],
        "metadata": "Type `{kind}` · Range `{domain}` · Worker override `{override}` · Level `{level}`.",
        "levels": {"core": "core", "advanced": "advanced", "legacy": "legacy"},
        "no_override": "not supported",
        "legacy": "Legacy v1 key",
        "present": "Writing this key explicitly can override implicit worker suffixes, paths or environment settings even when its value equals the built-in default.",
        "maintenance": "Maintenance",
        "regenerate": "Edit `config/ecm_options.json` and run `python tools/gen/generate_ecm_config.py`. Commit the schema and generated files together.",
    },
    "zh-CN": {
        "description_index": 1,
        "title": "ecm.toml 配置参考",
        "other_language": "[English](ECM_TOML_REFERENCE.en.md)",
        "intro": [
            "在首个表头前写 `config_version = 3`。无版本字段的文件沿用旧格式。见[配置迁移](architecture/CONFIGURATION.md#升级旧配置)。", "",
            "## 分区与优先级", "",
            "- `[shared]` 存放两阶段共用的设置。",
            "- `[stage1]` 和 `[stage2]` 各自存放本阶段的设置。",
            "- `[gui]` 存放 GUI 全局设置，worker 无法覆盖。",
            "- `[worker.N]` 存放第 N 个 worker 的设置，N 为 1–1000000。覆盖项使用 `shared.`、`stage1.` 或 `stage2.` 前缀。", "",
            "程序依次应用内置默认值、全局分区、对应 worker 分区和命令行参数，后者优先。重复键或表、未知分区、缺失值及未闭合字符串会报错。`worktodo.txt` 使用独立的 `[Worker #N]` 分区。程序不读取 INI 文件。", "",
            "worker 校验共享和本阶段设置。GUI 校验共享、GUI 和 worker 设置，并预览能识别的计算参数。见[配置读取职责](architecture/CONFIGURATION.md#按程序职责读取)。", "",
            "共用 worktodo 与 Stage2 的相对路径基于 TOML 所在目录，其他 Stage1 路径通常基于可执行文件目录。具体规则见[运行时限制](architecture/CONFIGURATION.md#运行时限制)。",
        ],
        "metadata": "类型 `{kind}` · 范围 `{domain}` · Worker 覆盖 `{override}` · 级别 `{level}`。",
        "levels": {"core": "常用", "advanced": "高级", "legacy": "旧版"},
        "no_override": "不支持",
        "legacy": "v1 旧键",
        "present": "显式写入此键时，即使数值与内置默认值相同，也可能覆盖 worker 后缀、路径或环境变量的隐式规则。",
        "maintenance": "维护",
        "regenerate": "修改 `config/ecm_options.json` 后运行 `python tools/gen/generate_ecm_config.py`，并将定义与生成物一同提交。",
    },
}


def toml_reference(schema, language):
    labels = REFERENCE_TEXT[language]
    out = [f"# {labels['title']}", "", "<!-- Generated from config/ecm_options.json. -->", "",
           labels["other_language"], "", *labels["intro"], ""]
    originals = [o for o in schema["options"] if not o.get("_mirror") and v3_key(o)]
    occupied_anchors = {v3_key(option).lower() for option in originals}
    for table in ("shared", "stage1", "stage2", "gui", "worker.1"):
        out.extend([f"## [{table}]", ""])
        for option in sorted((o for o in originals if v3_table(o) == table), key=v3_key):
            worker = table in ("shared", "stage1", "stage2") and not option.get("v3_global_only")
            key = v3_key(option)
            override = f"[worker.N] {table}.{key}" if worker else labels["no_override"]
            kind, domain = v2_description_type(option)
            legacy_anchor = option["key"].lower()
            if legacy_anchor not in occupied_anchors:
                occupied_anchors.add(legacy_anchor)
                out.extend([f'<a id="{legacy_anchor}"></a>', ""])
            level = labels["levels"][option.get("template_level", "advanced")]
            out.extend([f"### {key}", "",
                        "```toml", f"{key} = {toml_literal(option)}", "```", "",
                        labels["metadata"].format(kind=kind, domain=domain, override=override, level=level), ""])
            if not option.get("v3_only") and (key != option["key"] or table == "worker.1"):
                out.extend([f"{labels['legacy']} `{config_table(option)}.{option['key']}`", ""])
            if option.get("present"):
                out.extend([labels["present"], ""])
            out.extend([option["description"][labels["description_index"]], ""])
    out.extend([f"## {labels['maintenance']}", "", labels["regenerate"], ""])
    return "\n".join(out)


def toml_schema_header(schema):
    out = ["// Generated by tools/gen/generate_ecm_config.py; edit config/ecm_options.json.",
           "#pragma once", "namespace ecm_config {",
           f"inline constexpr int business_config_version={schema['config_version']};",
           "struct TomlOptionMeta { const char *table, *key, *type, *domain, *v2_table, *v2_key, *v2_conversion, *v3_table, *v3_key; bool v3_only, v3_global_only; };",
           "inline constexpr TomlOptionMeta toml_options[]={"]
    for option in schema["options"]:
        if not option.get("_mirror"):
            kind = option["type"]
            parser = option.get("parser_override", option["parser"])
            if parser == "auto_integer":
                kind = "int|string"
            elif parser in ("log_level", "gui_font"):
                kind += "|string"
            out.append("    {" + ",".join(literal(x) for x in
                       (config_table(option), option["key"], kind, option.get("domain", "valid scalar"),
                        config_table(option), v2_key(option), option.get("v2_conversion", ""),
                        option.get("v3_table", config_table(option)), v3_key(option),
                        option.get("v3_only", False), option.get("v3_global_only", False))) + "},")
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
             "config/ecm.toml.example": toml_template(schema, True)}
    files.update({f"docs/ECM_TOML_REFERENCE.{language}.md": toml_reference(schema, language)
                  for language in REFERENCE_TEXT})
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
