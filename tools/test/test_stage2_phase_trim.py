"""Verify native phase reclamation preserves the resident-fallback contract.

Consumes a completed workspace/output-lifetime check with an independent unit-case CPU oracle.
Replays its candidate invocation, injecting each allocation fallback. Retains
commands, environments, complete target-leaf checks and frozen identities.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools/bench'))
from bench_stage2_production import fields, freeze, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--reference-check', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    exe, reference, out = a.exe.resolve(), a.reference_check.resolve(), a.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('use a fresh output directory')
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads(reference.read_text(encoding='utf-8'))
    identity, tool_sha, ref_sha = freeze(exe), sha(__file__), sha(reference)
    if (not data['complete'] or data['mode'] != 'check' or
        data['comparison'] not in ('workspace-bq','phase-output') or not data['oracle'] or not data['oracle']['unit'] or
        identity != data['identity']):
        raise ValueError('a matching completed workspace/output-lifetime check with an independent unit oracle is required')
    row = next(r for r in data['runs'] if r['key'] ==
               ('trimmed_output' if data['comparison']=='phase-output' else 'two_buffer'))
    expected = {k: str(v) for k, v in data['oracle']['expected_leaf'].items()}
    if row['leaf'] != expected or row['environment'].get('NTT_PHASE_TRIM_RAW') != '1':
        raise ValueError('reference must exercise phase reclamation and match its CPU leaf oracle')
    for key in ('log', 'debug_log'):
        if sha(row[key]) != row['log_sha256' if key == 'log' else 'debug_sha256']:
            raise ValueError('reference raw log changed')
    save = Path(data['input']['save'])
    if sha(save) != data['input']['save_sha256']:
        raise ValueError('reference Stage1 save changed')
    if any(row[k]['enabled'] != '1' for k in ('fold', 'root', 'frontier')):
        raise ValueError('reference must use the resident path before failure injection')
    rows = []
    result = dict(complete=False, identity=identity, reference_check=str(reference),
                  reference_check_sha256=ref_sha, tool_sha256=tool_sha, rows=rows)
    try:
        for name, key, prefix in (
            ('frontier', 'NTT_SCALED_FRONTIER_ALLOC_FAIL', 'scaled_frontier_device'),
            ('fold', 'NTT_FOLD_DEVICE_ALLOC_FAIL', 'real_batched_folddevice')):
            command = list(row['command'])
            command[0] = str(exe)
            log, debug, output = out/(name+'.log'), out/(name+'.debug.log'), out/(name+'.jsonl')
            for flag, path in (('--log', log), ('--debug-log-file', debug), ('--results', output)):
                command[command.index(flag)+1] = str(path)
            env = {k: v for k, v in os.environ.items() if not k.startswith('NTT_')}
            env.update(row['environment']); env[key] = '1'
            proc = subprocess.run(command, env=env, capture_output=True, timeout=60)
            (out/(name+'_driver.log')).write_bytes(proc.stdout+proc.stderr)
            if proc.returncode:
                raise ValueError(name+' failed; raw output retained')
            text = log.read_text(encoding='utf-8')+'\n'+debug.read_text(encoding='utf-8')
            state, leaf = fields(text, prefix), fields(text, 'target_descent_values')
            native = json.loads(output.read_text(encoding='utf-8'))
            if (state['enabled'] != '0' or state['fallback'] != 'allocation_fixture' or
                leaf != expected or native['bad_factors'] or native['factors'] != row['result']['factors']):
                raise ValueError(name+': fallback or independent target-leaf oracle differs')
            if any(t not in text for t in ('gmp_check_bad=0', 'gmp_selftest_bad=0', 'pending=0')):
                raise ValueError(name+': mandatory arithmetic checks missing')
            if (sha(exe) != identity['binary_sha256'] or sha(reference) != ref_sha or
                sha(__file__) != tool_sha or sha(save) != data['input']['save_sha256']):
                raise ValueError('build, reference, save or gate changed during checks')
            rows.append(dict(name=name, command=command, environment=row['environment']|{key: '1'},
                returncode=proc.returncode, leaf=leaf, fallback=state,
                log_sha256=sha(log), debug_sha256=sha(debug), result_sha256=sha(output)))
            print(name, 'fallback and independent full target leaves OK', flush=True)
        result['complete'] = True
    except Exception as exc:
        result['error'] = str(exc)
        raise
    finally:
        (out/'checks.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
