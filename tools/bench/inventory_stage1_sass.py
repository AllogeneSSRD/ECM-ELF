#!/usr/bin/env python3
"""Inventory Stage1 kernel and shared callee text, never dynamic work or cache size."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

FUNCTION = re.compile(r'Function\s*:\s*(\S+)')
INSTRUCTION = re.compile(r'^\s*/\*([0-9a-f]+)\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]+)')


def inventory(path):
    functions = []
    current = None
    with path.open(encoding='utf-8', errors='replace') as source:
        for line in source:
            match = FUNCTION.search(line)
            if match:
                current = dict(function=match[1], instruction_count=0, text_span_bytes=0,
                               opcode_counts=Counter())
                functions.append(current)
            match = INSTRUCTION.match(line)
            if match and current:
                current['instruction_count'] += 1
                current['text_span_bytes'] = max(current['text_span_bytes'], int(match[1],16)+16)
                current['opcode_counts'][match[2].split('.')[0]] += 1
    if not functions or any(not f['instruction_count'] for f in functions):
        raise ValueError('Missing SASS function/instruction records')
    if len({f['function'] for f in functions}) != len(functions):
        raise ValueError('Repeated function names: select one architecture/object before comparing')
    return functions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--object', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--bits', type=int, default=4608)
    p.add_argument('--tpi', type=int, default=16)
    p.add_argument('--modes', type=int, nargs='+', default=[4,5,8,9])
    p.add_argument('--kernel-family', choices=['kernel_suyama_domain','kernel_suyama_constants'], default='kernel_suyama_domain')
    p.add_argument('--cuobjdump', type=Path, default=Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/cuobjdump.exe'))
    a = p.parse_args(); obj = a.object.resolve(strict=True)
    object_hash = hashlib.sha256(obj.read_bytes()).hexdigest()
    root = a.output.resolve(); root.mkdir(parents=True, exist_ok=False)
    for flag,name in (('--dump-sass','all.sass'),('--dump-resource-usage','resources.txt'),
                      ('--dump-elf-symbols','symbols.txt')):
        with (root/name).open('wb') as output:
            subprocess.run([str(a.cuobjdump),flag,str(obj)],stdout=output,stderr=subprocess.PIPE,check=True,timeout=120)
    if hashlib.sha256(obj.read_bytes()).hexdigest() != object_hash:
        raise ValueError('Object changed during SASS/resource export')
    functions = inventory(root/'all.sass')
    kernels = []
    for mode in a.modes:
        matches = [f for f in functions if f'ILj{a.tpi}ELj{a.bits}EELi{mode}EE' in f['function'] and a.kernel_family in f['function']]
        if len(matches)!=1: raise ValueError(f'Expected exactly one requested MODE{mode}')
        kernels.append(dict(mode=mode,**matches[0]))
    callees = [f for f in functions if 'prac_add_outlined' in f['function']]
    # Non-RDC ptxas can embed a local callee inside each entry's text section.
    # Its instructions are already included in the corresponding kernel span.
    embedded = [line.split()[-1] for line in (root/'symbols.txt').read_text(encoding='utf-8').splitlines()
                if line.startswith('STT_FUNC') and 'prac_add_outlined' in line and '$_Z20kernel_suyama_domain' in line]
    report = dict(object=str(obj),object_sha256=object_hash,
                  scope='static entry text includes embedded local callees; separately emitted callees listed separately; not retired instructions or a cache working set',
                  kernels=kernels,outlined_callees=callees,embedded_outlined_symbols=embedded,
                  all_function_count=len(functions))
    (root/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({str(k['mode']):{key:k[key] for key in ('instruction_count','text_span_bytes')} for k in kernels}))
    print('separate callees:',len(callees),'embedded callee symbols:',len(embedded))


if __name__=='__main__': main()
