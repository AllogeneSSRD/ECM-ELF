"""GPU gates for NTT lengths/batches, publication and production ECM regression."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools/bench'))
from analyze_stage2_tune_workload import ntt_batch_samples


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('exe', 'save', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--device', type=int, required=True)
    p.add_argument('--carrier', type=int, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    exe, save = a.exe.resolve(), a.save.resolve()
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path = exe.parent/'build_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    assert sha(exe) == manifest['sha256'].lower()
    sources = {}
    for entry in manifest['sources']:
        name, _, digest = entry.rpartition('=')
        path = ROOT/name
        if len(digest) != 64 or not path.is_file():
            continue
        assert sha(path) == digest.lower(), name
        frozen = out/'frozen_sources'/name
        frozen.parent.mkdir(parents=True, exist_ok=True)
        frozen.write_bytes(path.read_bytes())
        sources[str(path)] = digest.lower()
    assert len(sources) >= 49
    for path in (exe, manifest_path, exe.parent/'gmp-10.dll', save, Path(__file__),
                 ROOT/'tools/bench/analyze_stage2_tune_workload.py',
                 ROOT/'tools/test/test_stage2_tune_tail_runtime.py'):
        sources[str(path)] = sha(path)
        frozen = out/'frozen_inputs'/path.name
        frozen.parent.mkdir(parents=True, exist_ok=True)
        frozen.write_bytes(path.read_bytes())
    state = dict(complete=False, identities=sources, cases=[])
    publish = lambda: (out/'state.json').write_text(json.dumps(state, indent=2)+'\n', encoding='utf-8')
    ini = out/'ecm.ini'
    ini.write_text('verbose=false\nstage2_debug_log=false\n', encoding='utf-8')
    common = [str(exe), '--ini', str(ini), '--device', str(a.device),
              '--batch-mb', '256', '--arena-mb', '6300', '--owner-budget-mb', '640']

    def execute(name, args, success=True):
        command = common+args
        state['current'] = dict(name=name, command=command)
        publish()
        proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                              encoding='utf-8', errors='replace', timeout=1800)
        (out/(name+'.log')).write_text(proc.stdout+proc.stderr, encoding='utf-8')
        assert (proc.returncode == 0) == success, (name, proc.returncode, proc.stderr[-1500:])
        state['cases'].append(dict(name=name, returncode=proc.returncode))
        print('ntt_batch_gate:', name, proc.returncode, flush=True)
        publish()
        return proc

    def tune(name, args, expected_slices):
        destination = out/(name+'.toml')
        execute(name, ['--tune', 'ntt', '--tune-file', str(destination)]+args)
        profile = tomllib.loads(destination.read_text(encoding='utf-8-sig'))
        measured = ntt_batch_samples(profile)
        assert profile['profile']['slices'] == expected_slices
        assert profile['device']['gl_add_sub_mask'] == 1
        assert profile['policy']['accounting'] == 'cuda_events_two_forward_product_inverse_v1'
        text = destination.read_text(encoding='utf-8-sig')
        assert str(ROOT) not in text and 'sha256' not in text and 'build_manifest' not in text
        return profile, measured

    publish()
    telemetry_file = (out/'telemetry.csv').open('wb')
    monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,uuid,utilization.gpu,clocks.sm,temperature.gpu,power.draw,memory.used',
        '--format=csv', '-lms', '1000'], stdout=telemetry_file, stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        bad = [['--tune-slices', value] for value in ('0', '65536', '1,1', '1,', 'true')]
        bad += [['--length-log2', value] for value in ('2:3', '3:28')]
        for i, flags in enumerate(bad):
            destination = out/f'bad_{i}.toml'
            execute(f'bad_{i}', ['--tune', 'ntt', '--tune-file', str(destination)]+flags, False)
            assert not destination.exists()
        for i, flags in enumerate((['--tune','ecm','--tune-slices','1'],
                                   ['--tune','ecm','--tune-merge','missing.toml','--tune-slices','1'])):
            execute(f'conflict_{i}', flags, False)
        profile, measured = tune('small', ['--tune-level','7','--length-log2','3:12',
            '--tune-slices','1,3,16','--tune-repeats','3','--tune-memory-mb','128'], [1,3,16])
        assert len(measured) == 30 and profile['summary']['skipped'] == 0
        tune('reverse_flags', ['--length-log2','3:5','--tune-slices','2,5',
             '--tune-repeats','2','--tune-level','7','--tune-memory-mb','64'], [2,5])
        tune('maximum_slices', ['--length-log2','3:3','--tune-slices','65535',
             '--tune-repeats','2','--tune-memory-mb','64'], [65535])
        profile, measured = tune('production_lengths', ['--length-log2','11:23',
            '--tune-slices','1,3,16,64,256,2880','--tune-repeats','3','--tune-memory-mb','1024'],
            [1,3,16,64,256,2880])
        assert profile['summary']['skipped'] > 0 and len(measured) > 20
        # Level presets remain observable; explicit flags override independent of order.
        profile, measured = tune('preset_level1', ['--tune-level','1','--tune-memory-mb','128'], [1])
        assert profile['profile']['min_log2'] == 3 and profile['profile']['max_log2'] == 20
        assert profile['profile']['repeats'] == 3
        destination = out/'callbacks.jsonl'
        execute('jsonl', ['--tune','ntt','--length-log2','3:4','--tune-slices','3',
                         '--tune-repeats','2','--tune-file',str(destination)])
        records = [json.loads(line) for line in destination.read_text(encoding='utf-8-sig').splitlines()]
        assert records[0]['schema'] == 2
        assert next(r for r in records if r['type']=='policy')['slices'] == [3]
        assert sum(r['type']=='sample' and r['status']=='measured' for r in records) == 2
        # Failed all-skip tune retains the existing destination and partial evidence.
        destination = out/'all_skip.toml'
        sentinel = b'# existing profile must survive\n'
        destination.write_bytes(sentinel)
        execute('all_skip', ['--tune','ntt','--length-log2','27:27','--tune-slices','65535',
            '--tune-repeats','1','--tune-memory-mb','1','--tune-file',str(destination)], False)
        assert destination.read_bytes() == sentinel
        assert list(out.glob('all_skip.toml.partial.*'))
        # Full production curves verify both arithmetic paths and mandatory GMP checks.
        command = [sys.executable, str(ROOT/'tools/test/test_stage2_tune_tail_runtime.py'),
            '--exe',str(exe),'--save',str(save),'--device',str(a.device),'--carrier',str(a.carrier),
            '--level','3','--ds','120120','--b2','12000000000','--repeats','1',
            '--output',str(out/'production_ecm')]
        state['current'] = dict(name='production_ecm', command=command)
        publish()
        proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                              encoding='utf-8', errors='replace', timeout=1800)
        (out/'production_ecm.log').write_text(proc.stdout+proc.stderr, encoding='utf-8')
        assert proc.returncode == 0, proc.stderr[-2000:]
        receipt = json.loads((out/'production_ecm/state.json').read_text(encoding='utf-8'))
        assert receipt['complete'] and receipt['counts']['formal'] == 2 and receipt['counts']['warmup'] == 2
        state['ecm_counts'] = receipt['counts']
        assert all(sha(Path(path)) == digest for path, digest in sources.items())
        state['complete'] = True
        publish()
        print('PASS: GPU batch/minimum-length/publication gates and both production ECM paths', flush=True)
    except Exception as error:
        state['failure'] = repr(error)
        publish()
        raise
    finally:
        monitor.terminate()
        monitor.wait(timeout=10)
        telemetry_file.close()


if __name__ == '__main__':
    main()
