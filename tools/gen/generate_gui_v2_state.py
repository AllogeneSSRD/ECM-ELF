"""Generate v2 sidecar defaults/validation/reference from ecm_options.json."""
from pathlib import Path
import argparse
import hashlib
import json
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / 'config/ecm_options.json'


def products():
    full_schema = json.loads(SCHEMA.read_text(encoding='utf-8'))
    schema = full_schema['gui_v2_sidecar']
    fields = schema['fields']
    # A TOML table may only be emitted once, even after fields are appended to the schema.
    tables = list(dict.fromkeys(field['table'] for field in fields))
    fields = [field for table in tables for field in fields if field['table'] == table]
    types = {'string': 'std::string', 'int': 'int', 'float': 'float', 'bool': 'bool'}
    lines = ['// Generated; edit config/ecm_options.json, not this file.', '#pragma once', '#include <string>', '#include <cmath>', '#include <stdexcept>', '#include <algorithm>',
             'namespace ecmgui_v2 {', 'struct UiValues {']
    for field in fields:
        assert len(field['description']) == 2 and field['type'] in types
        value = json.dumps(field['default'], ensure_ascii=True)
        if value == '-2147483648': value = '(-2147483647 - 1)'
        if field['type'] == 'float': value = f"{float(field['default'])}f"
        lines.append(f"    {types[field['type']]} {field['field']} = {value};")
    lines += ['};', 'namespace ui_limits {']
    for field in fields:
        if field['type'] not in ['int', 'float']: continue
        for bound, suffix in [('minimum', 'min'), ('maximum', 'max')]:
            if bound not in field: continue
            value = field[bound]
            literal = '(-2147483647 - 1)' if value == -2147483648 else str(value)
            if field['type'] == 'float': literal = f'{float(value)}f'
            lines.append(f'inline constexpr {types[field["type"]]} {field["field"]}_{suffix} = {literal};')
    lines += ['}', 'template<class V,class F> void visit_ui_fields(V& value,F&& field) {']
    for field in fields:
        lines.append(f'    field("{field["table"]}","{field["key"]}",value.{field["field"]});')
    lines += ['}', 'inline void validate_ui(const UiValues& v) {']
    for field in fields:
        name = f'v.{field["field"]}'
        tests = []
        if 'permutation_of' in field:
            assert field['type'] == 'string' and sorted(field['default']) == sorted(field['permutation_of'])
            lines.append(f'    auto sorted_{field["field"]} = {name}; std::sort(sorted_{field["field"]}.begin(),sorted_{field["field"]}.end());')
            tests.append(f'sorted_{field["field"]} != {json.dumps("".join(sorted(field["permutation_of"])))}')
        if 'minimum' in field: tests.append(f'{name} < {"(-2147483647 - 1)" if field["minimum"] == -2147483648 else field["minimum"]}')
        if 'maximum' in field: tests.append(f'{name} > {field["maximum"]}')
        if field['type'] == 'float': tests.append(f'!std::isfinite({name})')
        if 'values' in field: tests.append('(' + ' && '.join(f'{name} != {json.dumps(x)}' for x in field['values']) + ')')
        if field['type'] == 'string': tests.append(f'{name}.size() > {field.get("max_length", 4096)}')
        if tests: lines.append(f'    if ({" || ".join(tests)}) throw std::runtime_error("Invalid UI state: {field["table"]}.{field["key"]}");')
    lines += ['}', '}']
    reference = ['# GUI v2 侧车配置参考', '', '由 `config/ecm_options.json` 的 `gui_v2_sidecar` 生成。独立 UI 文件不属于业务 TOML；不会写回旧 `[gui]`。', '']
    for field in fields:
        reference += [f'## {field["table"]}.{field["key"]}', '', f'默认值：`{json.dumps(field["default"], ensure_ascii=False)}`；类型：`{field["type"]}`。', '', *field['description'], '']
    result = {'src/ecm/application/gui_v2/generated/ui_values.h': '\n'.join(lines) + '\n',
              'docs/GUI_V2_STATE_REFERENCE.md': '\n'.join(reference)}
    meta=['// Generated; edit config/ecm_options.json.', '#pragma once', 'namespace ecmgui_v2 {',
          'struct BusinessOption { const char *table,*key,*domain,*description_key,*type; bool has_min,has_max; double minimum,maximum; };',
          'inline constexpr BusinessOption business_options[]={']
    for option in full_schema['options']:
        if option['owner']=='gui': continue
        table='workers.1' if option['owner']=='worker' else 'shared' if option.get('shared_with') else option['owner']
        strings=[table,option['key'],option.get('domain',''),'option_'+table+'_'+option['key'],option['type']]
        meta.append('    {'+','.join(json.dumps(v,ensure_ascii=True) for v in strings)+','+
                    ','.join(['true' if 'minimum' in option else 'false','true' if 'maximum' in option else 'false',str(option.get('minimum',0)),str(option.get('maximum',0))])+'},')
    meta+=['};','}']
    result['src/ecm/application/gui_v2/generated/business_options.h']='\n'.join(meta)+'\n'
    # Configuration prose stays schema-owned, but is loaded as XML at runtime.
    for language,index in [('english',0),('chineseSimplified',1)]:
        root=ET.Element('EcmGui');native=ET.SubElement(root,'Native-Langue',name=language)
        panel=ET.SubElement(native,'Panel',id='ui')
        for option in full_schema['options']:
            if option['owner']=='gui':continue
            table='workers.1' if option['owner']=='worker' else 'shared' if option.get('shared_with') else option['owner']
            ET.SubElement(panel,'Item',id='option_'+table+'_'+option['key'],name=option['description'][index])
        ET.indent(root,space='  ')
        result[f'src/ecm/application/gui_v2/localization/options.{language}.xml']='<?xml version="1.0" encoding="utf-8"?>\n<!-- Generated; edit config/ecm_options.json. -->\n'+ET.tostring(root,encoding='unicode')+'\n'
    hashes = {path: hashlib.sha256(value.encode()).hexdigest() for path, value in result.items()}
    for path in [SCHEMA, Path(__file__)]: hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    result['config/gui_v2_state.generated.json'] = json.dumps({'sha256': hashes}, indent=2) + '\n'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    for name, content in products().items():
        path = ROOT / name
        if args.check:
            assert path.exists() and path.read_bytes() == content.encode(), f'Stale v2 UI generated file: {name}'
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8', newline='\n')
    print('GUI v2 state definitions current' if args.check else 'GUI v2 state definitions generated')


if __name__ == '__main__': main()
