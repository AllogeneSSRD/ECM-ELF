"""Exercise non-GPU function-body edits through formal local build scripts.

Run with no other production build active: the probe temporarily edits shared
source files, preserves their original bytes in evidence, and restores them in
finally. It never rewrites objects, dependency records or build manifests.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('stage1-build', 'stage2-build', 'gui-build', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    s1, s2 = args.stage1_build.resolve(), args.stage2_build.resolve()
    first = read(s1/'stage1_local_manifest.json')
    second = read(s2/'build_manifest.json')
    assert first['tiers'] == '384,1024' and first['role'] == second['role'] == 'local'
    objects = {**{str(s1/k):v.lower() for k,v in first['objects'].items()},
               **{v['entry']['output']:v['object_sha256'].lower()
                  for k,v in second['compile_units'].items() if k.endswith('.cu')}}
    assert len(objects) == 33 and all(sha(Path(p)) == h for p,h in objects.items())
    probes = [
        ('configuration_read', 'src/ecm/application/configuration/ecm_toml.h', 'inline bool read_toml('),
        ('configuration_write', 'src/ecm/application/gui/toml_file.cpp', 'bool TomlFile::save('),
        ('curve_cpu_math', 'src/ecm/backend/cuda/stage1/montgomery/host_math.cpp', 'void to_mpz('),
        ('checkpoint', 'src/ecm/storage/checkpoint/ecm_stage1_checkpoint_runtime.cpp', 'bool replace_file('),
        ('save', 'src/ecm/storage/save/ecm_save.cpp', 'std::string build_who_field('),
        ('data', 'src/ecm/storage/results/publication.cpp', 'std::string internal_json('),
        ('result', 'src/ecm/storage/results/publication.cpp', 'std::string submission_json('),
        ('tune_toml', 'src/ecm/service/planning/ecm_stage2_tune_format.h', 'inline std::string table('),
        ('screen', 'src/ecm/platform/stage1_progress.cpp', 'std::string line('),
    ]
    original = {name:(ROOT/name).read_bytes() for _,name,_ in probes}
    report = dict(complete=False, categories=[], original_sha256={}, builds=[],
                  cuda_objects_before=objects, source_restored=False,
                  scope='combined independent body probes; CUDA runtime/contract edits excluded')
    # A failed previous study may already have built the same body probe. Bind
    # each new study to its fresh output path so content-based reuse cannot turn
    # the edited phase into an unexercised compile check.
    probe_id = hashlib.sha256(str(out).encode()).hexdigest()[:16]
    report['probe_id'] = probe_id
    for name,data in original.items():
        target = out/'originals'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        report['original_sha256'][name] = sha(target)

    def persist():
        (out/'result.json').write_text(json.dumps(report, indent=2)+'\n')

    env = dict(os.environ)
    env['PSModulePath'] = str(Path(env['WINDIR'])/'System32/WindowsPowerShell/v1.0/Modules')

    def build(phase, kind):
        if kind == 'stage1':
            script, flags = 'build_stage1_local.ps1', ['-BuildDir', s1, '-Tiers', '384,1024', '-HostOnly']
        elif kind == 'stage2':
            script, flags = 'build_stage2_local.ps1', ['-Build', s2, '-HostOnly']
        else:
            script, flags = 'dev/build_gui.ps1', ['-BuildDir', args.gui_build.resolve()]
        command = list(map(str, ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass',
                                '-File', ROOT/'tools/build'/script, *flags]))
        log = out/(phase+'_'+kind+'.log')
        (out/(phase+'_'+kind+'.command.json')).write_text(json.dumps(command, indent=2)+'\n')
        with log.open('wb') as stream:
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                     stderr=subprocess.STDOUT, timeout=3600)
        text = log.read_text(encoding='utf-8-sig', errors='replace')
        assert process.returncode == 0, (kind, phase, text[-2000:])
        starts = re.findall(r'(?m)^\s*start \[\d+/\d+\] (\S+)', text)
        assert all(not path.endswith('.cu') for path in starts), starts
        if kind == 'stage1':
            assert not re.search(r'Building CUDA object|parallel nvcc: [1-9]|compiled [1-9]\d* TU', text)
        count = len(starts) if kind == 'stage2' else len(re.findall('Building CXX object', text))
        if phase != 'repeat':
            assert count > 0, (phase, kind, 'probe did not compile')
        else:
            assert count == 0 and not re.search(r'Linking CXX executable|== link ', text), (kind, text[-1500:])
        assert all(sha(Path(p)) == h for p,h in objects.items()), 'CUDA object changed'
        report['builds'].append(dict(phase=phase, kind=kind, CXX=count, CU=0, log_sha256=sha(log)))
        persist()
        print(phase, kind, 'CXX='+str(count), 'CU=0', flush=True)

    try:
        for category,name,anchor in probes:
            path = ROOT/name
            data = path.read_bytes()
            assert data.count(anchor.encode()) == 1, (name, anchor)
            brace = data.index(b'{', data.index(anchor.encode()))+1
            injection = ('\n    static_assert(true, "ECM host incremental '+category+' '+probe_id+'");\n').encode()
            path.write_bytes(data[:brace]+injection+data[brace:])
            report['categories'].append(dict(category=category, path=name, anchor=anchor))
        persist()
        for kind in ('stage1', 'stage2', 'gui'):
            build('edited', kind)
    finally:
        restore_errors = []
        restored = set()
        for name,data in original.items():
            for attempt in range(3):
                try:
                    if (ROOT/name).read_bytes() != data:
                        (ROOT/name).write_bytes(data)
                    assert (ROOT/name).read_bytes() == data, name
                    restored.add(name)
                    break
                except OSError as error:
                    restore_errors.append(dict(path=name, attempt=attempt+1, error=str(error)))
                    if attempt < 2:
                        time.sleep(0.2)
        report['restoration_attempt_errors'] = restore_errors
        report['source_restored'] = len(restored) == len(original)
        persist()
        assert report['source_restored'], 'Source restoration failed; inspect original-byte backups before continuing'
    for kind in ('stage1', 'stage2', 'gui'):
        build('restored', kind)
    for kind in ('stage1', 'stage2', 'gui'):
        build('repeat', kind)
    report['complete'] = True
    persist()


if __name__ == '__main__':
    main()
