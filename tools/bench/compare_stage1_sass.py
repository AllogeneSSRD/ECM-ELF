#!/usr/bin/env python3
"""Compare complete exported Stage1 SASS, including scheduling encodings.

Inputs are directories from inventory_stage1_sass.py. Exported files are read
without accessing the original object (its build path may now hold a new one).
Only line endings and trailing blank lines between functions are normalized.
This is static evidence, not a benchmark or a GPU correctness gate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from inventory_stage1_sass import FUNCTION


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def functions(path, requested):
    result = {}; name = None; state = None; count = 0; blank = []

    def finish():
        if name is not None:
            if name in result:
                raise ValueError(f'Repeated function {name}: select one architecture/object')
            result[name] = dict(complete_lines=count, sha256=state.hexdigest())

    with path.open(encoding='utf-8', errors='strict') as source:
        for line in source:
            match = FUNCTION.search(line)
            if match:
                finish(); name = match[1] if match[1] in requested else None
                state = hashlib.sha256(); count = 0; blank = []
            if name is None:
                continue
            if not line.strip():
                blank.append(line)
                continue
            for pending in blank:
                state.update(pending.encode('utf-8')); count += 1
            blank.clear()
            state.update(line.encode('utf-8')); count += 1
    finish()
    if set(result) != set(requested):
        raise ValueError(f'Missing functions: {set(requested) - set(result)}')
    return result


def resources(path):
    result = {}; name = None
    for line in path.read_text(encoding='utf-8').splitlines():
        match = re.match(r'\s*Function (\S+):$', line)
        if match:
            name = match[1]
            if name in result: raise ValueError('Repeated resource function')
        elif name is not None and 'REG:' in line:
            result[name] = line.strip(); name = None
    return result


def compare(old, new, modes):
    reports = [json.loads((p/'summary.json').read_text(encoding='utf-8')) for p in (old,new)]
    selected = []
    for report in reports:
        rows = {row['mode']:row for row in report['kernels']}
        if any(mode not in rows for mode in modes): raise ValueError('Missing requested MODE')
        selected.append({mode:rows[mode] for mode in modes})
    if any(selected[0][m]['function'] != selected[1][m]['function'] for m in modes):
        raise ValueError('Selected entry names/TPI/container do not match')
    names = [selected[0][m]['function'] for m in modes]
    full = [functions(p/'all.sass',names) for p in (old,new)]
    usage = [resources(p/'resources.txt') for p in (old,new)]
    kernels = []
    for mode in modes:
        name = selected[0][mode]['function']
        if any(name not in r for r in usage): raise ValueError('Missing resource entry')
        kernels.append(dict(mode=mode, function=name, old=full[0][name], new=full[1][name],
            complete_function_lines_equal=full[0][name]==full[1][name],
            resources_equal=usage[0][name]==usage[1][name],
            old_resources=usage[0][name],new_resources=usage[1][name],
            old_instruction_count=selected[0][mode]['instruction_count'],
            new_instruction_count=selected[1][mode]['instruction_count'],
            old_text_bytes=selected[0][mode]['text_span_bytes'],
            new_text_bytes=selected[1][mode]['text_span_bytes']))
    return dict(scope='Complete selected static functions including scheduling rows; not runtime throughput',
        inputs=[dict(directory=str(p.resolve()),object_sha256=r['object_sha256'],
                     exports={n:digest(p/n) for n in ('all.sass','resources.txt','summary.json')})
                for p,r in zip((old,new),reports)], kernels=kernels,
        all_selected_equal=all(k['complete_function_lines_equal'] and k['resources_equal'] for k in kernels))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old',type=Path,required=True);p.add_argument('--new',type=Path,required=True)
    p.add_argument('--modes',type=int,nargs='+',default=[10,11,12,13])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if len(set(a.modes))!=len(a.modes):p.error('Duplicate modes')
    result=compare(a.old,a.new,a.modes)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k['mode']:dict(equal=k['complete_function_lines_equal'],
        resources_equal=k['resources_equal'],old_instructions=k['old_instruction_count'],
        new_instructions=k['new_instruction_count']) for k in result['kernels']}))


if __name__=='__main__':main()
