"""Select unchanged CUDA definitions into one independently loadable fatbin per entry.

The original .cu explicit-instantiation inventory and .cuh implementations remain
authoritative. Generated mirrors only add selection guards and constants binding;
they never rewrite arithmetic expressions or manufacture a second capacity list.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CUDA = ROOT / 'src/ecm/backend/cuda'


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text(encoding='utf-8-sig') != text:
        path.write_text(text, encoding='utf-8-sig' if path.suffix in ('.cu','.cuh') else 'utf-8')


def definition(text, start):
    """Balanced C++ definition, ignoring strings/comments while counting braces."""
    opening = text.index('{', start)
    token = re.compile(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[{}]')
    depth = 0
    for match in token.finditer(text, opening):
        if match[0] == '{':
            depth += 1
        elif match[0] == '}':
            depth -= 1
            if depth == 0:
                return opening, match.end()
    raise ValueError(f'Unbalanced definition at {start}')


def prefix(text, start):
    # Include a preceding template declaration inside the selection guard.
    before = text[:start].rstrip()
    match = re.search(r'(?m)^template[^;{}]*>\s*\Z', before)
    return match.start() if match else start


def symbol(text):
    return re.sub(r'[^A-Za-z0-9_]', '_', text)


def generate(out, *, emit=True):
    out = out.resolve()
    headers = sorted(set((CUDA / 'stage2/device').glob('*.cuh')) | set((CUDA / 'ntt/device').glob('*.cuh')))
    wrappers = {}
    edits = {}
    contents = {}
    def uses_constants(path, seen=None):
        seen = set() if seen is None else seen
        if path in seen or not path.is_file():
            return False
        seen.add(path)
        if path == CUDA / 'stage2/device/point.cuh':
            return True
        return any(uses_constants((path.parent / name).resolve(), seen)
                   for name in re.findall(r'#include\s*"([^"\n]+)"', path.read_text(encoding='utf-8-sig')))
    kernel_pattern = re.compile(r'(?m)^__global__\s+(?:__launch_bounds__\([^\n]*\)\s*)?void\s+(\w+)\s*\(')
    wrapper_pattern = re.compile(r'(?m)^void\s+((?:ecm_launch_|ecm_ntt_tune_)\w+)\s*\(')
    for path in headers:
        text = path.read_text(encoding='utf-8-sig')
        contents[path] = text
        replacements = []
        for match in kernel_pattern.finditer(text):
            begin = prefix(text, match.start())
            _, end = definition(text, match.start())
            key = match[1]
            body = text[begin:end]
            replacements.append((begin, end, f'#if defined(ECM_MODULE_{key})\n{body}\n#endif'))
        for match in wrapper_pattern.finditer(text):
            begin = prefix(text, match.start())
            opening, end = definition(text, match.start())
            body = text[begin:end]
            launch = re.search(r'(\w+)(?:<[^;\n]*>)?\s*<<<', body)
            if not launch:
                raise ValueError(f'Wrapper without one launch: {path}:{match[1]}')
            key = launch[1]
            if match[1] in wrappers:
                raise ValueError(f'Duplicate wrapper: {match[1]}')
            wrappers[match[1]] = dict(kernel=key, header=path, templated=begin != match.start())
            if uses_constants(path) or 'stage2/device' in path.as_posix():
                index = opening - begin + 1
                body = body[:index] + '\n    ecm_stage2_bind_module_constants();' + body[index:]
            replacements.append((begin, end, f'#if defined(ECM_MODULE_{key})\n{body}\n#endif'))
        edits[path] = replacements

    mirrors = {path: out / 'headers' / path.relative_to(CUDA) for path in headers}
    for path, text in contents.items():
        for begin, end, replacement in sorted(edits[path], reverse=True):
            text = text[:begin] + replacement + text[end:]
        def include(match):
            target = (path.parent / match[1]).resolve()
            return '#include "' + mirrors.get(target, target).as_posix() + '"'
        text = re.sub(r'#include\s*"([^"\n]+)"', include, text)
        if emit:write(mirrors[path], text)

    modules = []
    for group, source in [('arithmetic', CUDA / 'stage2/device/arithmetic_kernels.cu'),
                          ('ntt', CUDA / 'ntt/device/transform_kernels.cu')]:
        original = source.read_text(encoding='utf-8-sig')
        includes = [(source.parent / name).resolve() for name in re.findall(r'^#include "([^"]+)"', original, re.M)]
        instantiations = {}
        namespace = ''
        for line in original.splitlines():
            scope = re.match(r'namespace (\w+)\s*\{',line)
            if scope:namespace=scope[1]
            elif line.strip()=='}':namespace=''
            match = re.match(r'template void ((?:\w+::)?(ecm_launch_\w+))<([^\n]+)>\(', line)
            if match:
                if namespace and '::' not in match[1]:
                    line=line.replace('template void ','template void '+namespace+'::',1)
                instantiations.setdefault(match[2], []).append((match[3], line))
        for name, wrapper in wrappers.items():
            if (group == 'arithmetic') != ('stage2/device' in wrapper['header'].as_posix()):
                continue
            if wrapper['templated'] and name not in instantiations:
                raise ValueError(f'Missing explicit instantiation: {name}')
            entries = instantiations.get(name, [('', '')])
            for arguments, instantiation in entries:
                identity = f'{group}_{name}_{symbol(arguments)}'.rstrip('_')
                path = out / f'{identity}.cu'
                text = '#define ECM_CUDA_ISOLATED_DEVICE 1\n'
                text += f'#define ECM_MODULE_{wrapper["kernel"]} 1\n'
                text += ''.join(f'#include "{mirrors.get(header,header).as_posix()}"\n' for header in includes)
                text += instantiation + '\n'
                if name == 'ecm_launch_tile_kernel':
                    text += f'const void *ecm_ntt_tile_{symbol(arguments)}() {{ return (const void*)tile_kernel<{arguments}>; }}\n'
                if emit:write(path, text)
                modules.append(dict(source=path.as_posix(), label=identity, kernel=wrapper['kernel'],
                                    arguments=arguments, implementation=wrapper['header'].relative_to(ROOT).as_posix()))
        missing = set(instantiations) - set(wrappers)
        if missing:
            raise ValueError(f'Unresolved instantiation inventory: {missing}')
    expected = sum(len(re.findall(r'^template void ', (CUDA / p).read_text(encoding='utf-8-sig'), re.M))
                   for p in ['stage2/device/arithmetic_kernels.cu', 'ntt/device/transform_kernels.cu'])
    assert sum(bool(m['arguments']) for m in modules) == expected, 'Instantiation membership changed'
    inputs = set(headers) | {CUDA / 'stage2/device/arithmetic_kernels.cu', CUDA / 'ntt/device/transform_kernels.cu', Path(__file__).resolve()}
    manifest = dict(schema=1, modules=modules,
                    inputs_sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(inputs)})
    if emit:
        write(out / 'modules.json', json.dumps(manifest, indent=2) + '\n')
        print(f'{len(modules)} isolated Stage2/NTT modules: {out}')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    generate(parser.parse_args().out)
