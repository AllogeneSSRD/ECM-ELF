"""Audit the current production include graph and coarse workflow boundaries.

This checks source dependencies, not runtime support or frozen extraction
equivalence. Actual compiler dependency and worker tests provide those gates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
SUFFIXES = {'.h', '.hpp', '.cuh', '.cu', '.cpp', '.inl', '.inc', '.def'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    files = sorted(p.resolve() for folder in ('include/ecm', 'src/ecm')
                   for p in (ROOT/folder).rglob('*') if p.suffix in SUFFIXES)
    source_set = set(files)
    roots = [ROOT/p for p in ('include', 'src', 'src/ecm/backend/cuda',
                             'src/ecm/backend/cuda/runtime')]
    graph = {}
    public = []
    edges = []
    for path in files:
        name = path.relative_to(ROOT).as_posix()
        exposed = path.is_relative_to(ROOT/'include/ecm')
        if exposed:
            public.append(name)
            # Every exposed family has a corresponding implementation family;
            # header-only models need not invent a same-named .cpp.
            assert (ROOT/'src/ecm'/path.parent.relative_to(ROOT/'include/ecm')).is_dir(), name
        graph[path] = []
        for inc in re.findall(r'(?m)^\s*#\s*include\s*["<]([^">]+)[">]',
                              path.read_text(encoding='utf-8-sig')):
            assert not any(part in inc for part in ('tools/test/', 'tools/bench/',
                'tools/experiments/', '.refactor/', 'kernels/cuda/')), (name, inc)
            target = next((p.resolve() for p in [path.parent/inc]+[r/inc for r in roots]
                           if p.is_file()), None)
            if inc.startswith('ecm/'):
                assert target is not None, (name, inc)
            if target not in source_set:
                continue  # SDK/system/third-party dependency; checked by builds.
            if exposed:
                assert target.is_relative_to(ROOT/'include/ecm'), (name, inc)
            graph[path].append(target)
            edges.append([name, target.relative_to(ROOT).as_posix()])
    visited = set()
    active = []

    def visit(path):
        assert path not in active, [str(p.relative_to(ROOT)) for p in active+[path]]
        if path in visited:
            return
        active.append(path)
        for child in graph[path]:
            visit(child)
        active.pop()
        visited.add(path)

    for path in graph:
        visit(path)
    for old in ('kernels/cuda', 'src/core', 'src/cpu', 'src/cuda', 'src/gui'):
        assert not [p for p in (ROOT/old).rglob('*') if p.is_file()], old
    task = (ROOT/'src/ecm/service/stage1/task.cpp').read_text(encoding='utf-8-sig')
    driver = (ROOT/'src/ecm/application/cli/ecm_driver.cpp').read_text(encoding='utf-8-sig')
    queue = (ROOT/'src/ecm/service/work/queue_workflow.cpp').read_text(encoding='utf-8-sig')
    assert len(re.findall(r'^TaskResult run_stage1_task\(', task, re.M)) == 1
    assert len(re.findall(r'^int run_queue_workflow\(', queue, re.M)) == 1
    assert 'run_stage1_task({' in driver and 'run_stage1_task({' in queue
    assert not re.search(r'run_stage1_impl|Stage1TaskSession|class .*Runner', driver)
    session = task[task.index('class Stage1TaskSession'):task.index('TaskResult run_stage1_task')]
    assert not re.search(r'\brun\s*\(', session)
    result = dict(complete=True, files=len(files), public_headers=len(public),
                  include_edges=len(edges), include_cycles=0, public_private_edges=0,
                  task_entry_points=1, queue_entry_points=1,
                  gpu_runs=0, scope='project textual include graph; system/SDK edges are build-validated',
                  sources={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in files}, edges=edges)
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources', 'edges')}))


if __name__ == '__main__':
    main()
