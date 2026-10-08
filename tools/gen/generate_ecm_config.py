#!/usr/bin/env python3
"""Generate runtime bindings, INI templates and symbolic reference from one schema.

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
    if schema["schema_version"] != 1:
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
    for a in schema["aliases"]:
        key = a["owner"], a["key"]
        if key in seen or (a["owner"], a["target"]) not in seen:
            raise ValueError(f"duplicate/invalid alias: {key}")
        seen.add(key)
    for entry in schema["options"] + schema["aliases"]:
        description = entry.get("description")
        if not isinstance(description, list) or len(description) != 2 or any(
                not isinstance(p, str) or not p.strip() for p in description):
            raise ValueError(f"description must contain English then Chinese: {entry['owner']}.{entry['key']}")
    return schema


def description_lines(entry, width=100):
    """Preserve explicit newlines and wrap prose by display columns (CJK=2).

    Description entries are ordered English first, then Chinese. Wrap each
    entry separately so translations never share an INI comment/Markdown line.
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
        code = f"legacy_assign({dst},v);"
        if o.get("value_aliases"):
            code = f"{dst}=canonical_{o['owner']}_{o['key']}(v);"
    elif kind == "gpu_param":
        valid="&&".join(f"{dst}!={x}" for x in o["values"])
        code = f"legacy_assign({dst},v);if({valid}){{std::fprintf(stderr,\"[ecm] WARNING: {key}=%d unsupported; using {o['default']}\\n\",{dst});{dst}={o['default']};}}"
    elif kind == "integer_bool":
        code = f"{dst}=bounded_integer(v,{literal(key)},{lo},{hi})!=0;"
    elif kind == "positive":
        code = f"{dst}=positive(v,{literal(key)});"
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


def header(schema):
    out = ["// Generated by tools/gen/generate_ecm_config.py; edit config/ecm_options.json.",
           "#pragma once", '#include "../ecm_ini.h"', "#include <array>",
           "namespace ecm_config {", f"inline constexpr int layout_version={schema['layout_version']};"]
    out.append("namespace defaults {")
    for o in schema["options"]:
        name=f"{o['owner']}_{o['key']}"
        if o["type"] == "string":
            out.append(f"inline constexpr char {name}[]={literal(o.get('effective_default',o['default']))};")
        else:
            out.append(f"inline constexpr {TYPES[o['type']]} {name}={literal(o['default'])};")
    out.append("} // namespace defaults")
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
            out.append(f"    {{{literal(o['key'])},[]({cls}&c,const std::string&v){{{assignment(o)}}},false,nullptr}},")
        for a in schema["aliases"]:
            if a["owner"] != owner:
                continue
            o = index[owner, a["target"]]
            dst = "c." + o["field"]
            transform = a.get("transform")
            if transform == "nonzero":
                body = f"int n=0;legacy_assign(n,v);if(n){dst}={literal(a['value'])};"
            elif transform == "integer_choice":
                body = f"int n=0;legacy_assign(n,v);{dst}=n=={a['equals']}?{literal(a['value'])}:{literal(a['otherwise'])};"
            elif transform == "map":
                body = f"const auto x=lower(v);{dst}={literal(a['otherwise'])};"
                for k, value in a["mapping"].items():
                    body += f"if(x=={literal(k)}){dst}={literal(value)};"
            elif transform is None:
                body = assignment(o, a["key"])
            else:
                raise ValueError(f"unsupported alias transform: {transform}")
            fallback = literal(a["target"]) if a.get("fallback_only") else "nullptr"
            out.append(f"    {{{literal(a['key'])},[]({cls}&c,const std::string&v){{{body}}},{literal(a.get('deprecated',False))},{fallback}}},")
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


def template(schema, release):
    out = ["# GENERATED: config/ecm_options.json -> tools/gen/generate_ecm_config.py",
           "# Reference: ECM_INI_REFERENCE.md (package), docs/ECM_INI_REFERENCE.md (repository)",
           "# CLI: global < Worker #N < CLI; GUI: global / [GUI] / [Worker #N]",
           "# S1.path_base=exe_dir; S2.path_base=ini_dir; CLI.path_base=CWD",
           "# AutoB2(current_release)=uncalibrated; use explicit B2",
           "# MiB=2^20 B; arena+fold+batch != process_peak", ""]
    for owner in ("stage1", "stage2", "gui"):
        out.extend((["[GUI]"] if owner == "gui" else ["# " + owner.upper()]))
        for o in schema["options"]:
            if o["owner"] != owner:
                continue
            value = o.get("template_value", o["default"])
            override = "release_value" if release else "first_run_value"
            if override in o:
                value = o[override]
            if o.get("effective_default") and o["type"] == "string":
                value = o["effective_default"]
            if o.get("implicit") and "template_value" not in o:
                value = o.get("minimum", value)
            if o.get("parser_override") == "integer_bool":
                value = int(value)
            out.append("# " + o["domain"] + "; " + annotation(o))
            out.extend("# " + line for line in description_lines(o))
            prefix = "# " if o.get("template_comment") or o.get("worker_suffix") else ""
            out.append(f"{prefix}{o['key']} = {text(value)}".rstrip())
        out.append("")
    out.extend(["# [Worker #2]", "# device = 1", "# worktodo = worktodo_2.txt", "# stage2_worktodo = stage2_worktodo_2.txt", ""])
    return "\n".join(out)


def reference(schema):
    out = ["# ecm.ini Reference / ecm.ini 配置参考", "", "<!-- GENERATED: config/ecm_options.json; do not edit. -->", "",
           "Add or edit the configuration lines below in `ecm.ini`. Stage1, Stage2, and the GUI share this file; each group identifies the applicable settings.<br>",
           "在 `ecm.ini` 中添加或修改下面的配置行。Stage1、Stage2 和 GUI 共用此文件，各选项的适用范围在分组中注明。", "",
           "Configuration lines use symbolic notation; the accompanying descriptions explain purpose, value effects, and when to adjust a setting.<br>",
           "配置行保留符号写法，后面的文字说明用途、取值效果和需要调整的情况。", "",
           "## Notation and general rules / 记号与通用规则", "", "```text",
           "<x> = value; [x] = optional; [a|b] = one of; a..b = inclusive range",
           "<x> = 值; [x] = 可选; [a|b] = 任选其一; a..b = 含端点的范围",
           'default = built-in default; "" = empty; @x = inherited or derived value',
           'default = 内置默认值; "" = 空值; @x = 继承或派生值',
           "Z = integer; R = real; N = worker_index (1..1000000)",
           "S1 = Stage1; S2 = Stage2; GUI = ecm_gui; P95 = Prime95",
           "MiB = 2^20 B; s = seconds; ms = milliseconds",
           "key=<domain>; default=<default>; [unit=<unit>]; [empty=<policy>]",
           "INI: key=value; comment=[#|;]...; inline_comment=unsupported; path_quotes=none",
           "key_case: S1=sensitive; S2=insensitive; GUI=sensitive",
           "bool(S1,S2): [true|false|yes|no|on|off|1|0]; case=insensitive",
           "integer(S2): decimal|scientific; exact_integer; >=0",
           "override: built_in < global < Worker#N < CLI",
           "B2: CLI(nonzero) > task(nonzero) > stage2_b2 > AutoB2",
           "CLI.sections: Worker#N=scope; other[]=label; label!=scope_reset",
           "GUI.sections: global|GUI|Worker#N; other[]=distinct_section",
           "global_keys => before_first_section; grouping => #comment",
           "duplicate_key: last_in_layer; unknown_key: ignore",
           "invalid: S1=legacy_fallback; S2=error; GUI=fallback_or_clamp",
           "path_base: S1=exe_dir; S2.INI=ini_dir; CLI=CWD; exceptions=path_base(key)",
           "queue: consumers=1; parallel_workers=>distinct(queue,outputs)",
           "S1.INI => queue_mode; single_run => CLI",
           "AutoB2(current_release)=uncalibrated; required=explicit_B2",
           "memory: arena+fold+batch != process_peak; budget!=total_VRAM_cap", "```", "",
           "Write only `key=value` to the INI. `default`, `unit`, and similar annotations are not additional keys. `release` and `first_run` identify deliberate template values that may differ from built-in defaults; existing INI files are not updated automatically.<br>",
           "只把 `key=value` 写入 INI，`default`、`unit` 等为说明记号，不是额外配置键。`release` 和 `first_run` 表示模板有意采用的值，可能与内置默认值不同；已有 INI 不会自动改为模板值。", "",
           "Place shared keys before the first section header and worker overrides under `[Worker #N]`. CLI programs treat only worker headers as scopes; other headers are labels and do not leave an active worker scope. The GUI reads actual sections. Use `#` comments to group shared keys for compatibility.<br>",
           "公共键放在任何分区标题之前；专用 worker 值放在 `[Worker #N]` 下。命令行程序只把 worker 标题当作作用域，其他标题仅是标签，也不会退出已有 worker 作用域；GUI 则按真实分区读取。为兼容两者，公共键应使用 `#` 注释分组。", "",
           "The last duplicate key in each layer wins; worker settings override global settings, and CLI overrides INI. Boolean keys accept the forms above, but Stage2 integer switches accept only `0|1`. Do not quote INI paths or append inline comments to configuration values.<br>",
           "同一层重复键以最后一次出现为准，worker 设置覆盖全局，命令行覆盖 INI。布尔键可以使用上列布尔写法，但标为整数开关的 Stage2 键只接受 `0|1`。路径不要加引号，不支持在配置行末尾追加注释。", "",
           "Stage1 relative paths normally use the executable directory. Stage2 INI paths use the INI directory; CLI paths use the current working directory. Individual keys document any exceptions. Concurrent processes need separate queues, progress files, and output paths.<br>",
           "Stage1 相对路径通常以可执行文件目录为基准；Stage2 的 INI 相对路径以 INI 目录为基准，命令行相对路径以当前工作目录为基准。各键另有约定时见该项说明。多进程并行运行应使用独立队列、进度及输出路径。", ""]
    for owner, title in (("stage1", "Stage1 and shared settings / Stage1 与共享选项 — global or [Worker #N] / 全局或 [Worker #N]"),
                         ("stage2", "Stage2 — global or [Worker #N] / 全局或 [Worker #N]"),
                         ("gui", "GUI settings / GUI 配置 — [GUI]"), ("worker", "GUI worker settings / GUI worker 配置 — [Worker #N]")):
        out += ["## " + title, ""]
        for o in schema["options"]:
            if o["owner"] == owner:
                out += ["### " + o["key"], "", "```text",
                        o["key"] + "=" + o["domain"] + "; " + annotation(o), "```", ""]
                out += ["<br>\n".join(description_lines(o)), ""]
    out += ["## Compatibility aliases / 兼容别名", "",
            "Legacy names remain readable; use canonical keys in new configurations. Direct aliases share the canonical value rules and have no separate defaults; conversion switches use the mappings below.<br>",
            "旧名称仍可读取，新配置建议使用对应的正式键。直接别名沿用正式键的取值规则，不另设默认值；转换开关按下面列出的映射处理。", ""]
    for a in schema["aliases"]:
        line = a["key"] + " => " + a["target"]
        if a.get("transform") == "nonzero":
            line += f"; domain=<x:Z>; x!=0:{a['value']}; x=0:no_change"
        elif a.get("transform") == "integer_choice":
            line += f"; domain=<x:Z>; x={a['equals']}:{a['value']}; else:{a['otherwise']}"
        elif a.get("transform") == "map":
            line += "; domain=<enum>; " + ",".join(k + ":" + v for k, v in a["mapping"].items()) + "; else:" + a["otherwise"]
        if a.get("fallback_only"):
            line += "; only_if(target=absent)"
        out += ["### " + a["key"], "", "```text", line, "```", ""]
        out += ["<br>\n".join(description_lines(a)), ""]
    out += ["## Repository maintenance / 源码仓库维护", "",
            "`config/ecm_options.json` -> `tools/gen/generate_ecm_config.py`", "",
            "See `docs/DEV_ECM_CONFIG_SCHEMA.md` in the source repository; release packages include only this configuration reference.<br>",
            "维护流程见源码仓库的 `docs/DEV_ECM_CONFIG_SCHEMA.md`；发布包仅附本配置说明。", ""]
    return "\n".join(out)


def template_header(value):
    # Keep the C++ source ASCII while preserving exact UTF-8 bytes in the INI.
    # Literal boundaries prevent a following ASCII hex digit extending a \x escape.
    lines = ["// Generated; edit config/ecm_options.json.", "#pragma once",
             "namespace ecm_config {", "inline constexpr const char default_ini[]="]
    for line in value.splitlines(keepends=True):
        chunks = re.split(r"([^\x00-\x7f]+)", line)
        literals = [literal(c) if c.isascii() else '"' + ''.join(f"\\x{b:02x}" for b in c.encode("utf-8")) + '"'
                    for c in chunks if c]
        lines.append("    " + " ".join(literals))
    return "\n".join(lines + [";", "}", ""])


def products(schema):
    default = template(schema, False)
    files = {"src/core/generated/ecm_config_generated.h": header(schema),
             "src/core/generated/ecm_ini_template.h": template_header(default),
             "config/ecm.ini.default": default,
             "config/ecm.ini.example": template(schema, True),
             "docs/ECM_INI_REFERENCE.md": reference(schema)}
    hashes = {name: hashlib.sha256(content.encode()).hexdigest() for name, content in files.items()}
    for path in (SCHEMA, SELF):
        hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    files["config/ecm_config.generated.json"] = json.dumps({"schema_version": 1, "sha256": hashes}, indent=2) + "\n"
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
