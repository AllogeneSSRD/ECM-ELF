"""Read current NTT declarations/definitions for the existing CPU CUDA doubles.

This is test instrumentation only. Native builds link the module inventory.
"""
from pathlib import Path
import re
from test_cuda_p2_structure import mask, balanced

ROOT = Path(__file__).resolve().parents[2]
FILES = [ROOT/'src/ecm/backend/cuda/ntt/runtime.h', ROOT/'src/ecm/backend/cuda/ntt/runtime.cpp']


def declaration(text, name):
    clean = mask(text)
    found = re.search(r'^struct '+name+r'\b[^;{]*\{', clean, re.M)
    end = balanced(clean, clean.index('{', found.start()), '{', '}')+2
    return text[found.start():end]


def function(text, name):
    clean = mask(text)
    for found in re.finditer(r'\b'+name+r'\s*\(', clean):
        end = balanced(clean, clean.index('(', found.start()), '(', ')')+1
        while clean[end].isspace(): end += 1
        if clean[end] != '{': continue
        start = clean.rfind('\n', 0, found.start())+1
        return text[start:balanced(clean, end, '{', '}')+1]
    raise ValueError('missing current NTT function: '+name)


def arena_source():
    header, runtime = [p.read_text(encoding='utf-8-sig') for p in FILES]
    names = ['ntt_arena_bufs', 'ntt_arena_pin_smem', 'ntt_arena_fuse', 'ntt_trim_fuse_headroom']
    prototypes = []
    for name in names:
        # The canonical defaults live only in the module header.
        found = re.search(r'^[^\n;{}]*\b'+name+r'\([^;]+;', header, re.M)
        prototypes.append(found[0])
    methods = runtime[runtime.index('void NttArena::drop_carry_scratch('):runtime.index('unsigned int ntt_gl_reduce_requested_mode(')]
    return '\n'.join([declaration(header, 'NttWorkspacePolicy'), declaration(header, 'NttArena'),
                      *prototypes, methods, *(function(runtime, name) for name in names)])


def fuse_source():
    header, runtime = [p.read_text(encoding='utf-8-sig') for p in FILES]
    begin = runtime.index('void fuse_base_allocate(')
    end = runtime.index('unsigned long long fuse_grid(')
    functions = runtime[begin:end]
    # Retain the header's defaults after moving the definitions out of line.
    prototypes = []
    for name in ['fuse_describe', 'fuse_init']:
        prototypes.append(re.search(r'^void '+name+r'\([^;]+;', header, re.M)[0])
    return '\n'.join([declaration(header, 'FuseCtx'), declaration(header, 'FuseBaseStats'),
                      'bool fuse_compact_scratch();',
                      'FuseBaseStats g_fuse_base;', *prototypes, functions])


def remove_table_launches(text):
    clean = mask(text); spans = []
    for found in re.finditer(r'\becm_launch_build_\w+_kernel\s*\(', clean):
        end = balanced(clean, clean.index('(', found.start()), '(', ')')+1
        if clean[end] != ';': raise ValueError('unexpected table launch statement')
        spans.append((found.start(), end+1))
    for begin, end in reversed(spans): text = text[:begin]+'(void)0;'+text[end:]
    return text, len(spans)
