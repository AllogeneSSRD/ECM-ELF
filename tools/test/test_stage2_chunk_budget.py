"""Exercise chunk controls/fallbacks and prove deferred verdicts cannot be erased.

Requires a matching completed chunk/output-lifetime check with an independent unit-case CPU
leaf oracle. Forced single-slice chunks exercise interior carry groups even
when the small reference curve would normally fit in one chunk.
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
    exe, ref, out = a.exe.resolve(), a.reference_check.resolve(), a.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('use a fresh output directory')
    out.mkdir(parents=True, exist_ok=True)
    identity, ref_sha, tool_sha = freeze(exe), sha(ref), sha(__file__)
    data = json.loads(ref.read_text(encoding='utf-8'))
    if (not data['complete'] or data['mode'] != 'check' or data['comparison'] not in ('chunk','phase-output','owner-cache') or
        not data['oracle'] or not data['oracle']['unit'] or identity != data['identity']):
        raise ValueError('matching completed chunk/output-lifetime check with independent unit oracle required')
    row = next(r for r in data['runs'] if r['key'] ==
               ({'phase-output':'trimmed_output','owner-cache':'trimmed_cache'}.get(data['comparison'],'workspace_chunk')))
    expected = {k: str(v) for k, v in data['oracle']['expected_leaf'].items()}
    save = Path(data['input']['save'])
    if row['leaf'] != expected or sha(save) != data['input']['save_sha256']:
        raise ValueError('reference/input differs from CPU oracle')
    for k in ('log', 'debug_log'):
        if sha(row[k]) != row['log_sha256' if k == 'log' else 'debug_sha256']:
            raise ValueError('reference raw log changed')
    rows = []
    result = dict(complete=False, identity=identity, reference_sha256=ref_sha,
                  tool_sha256=tool_sha, rows=rows)
    policies = [
        ('pool_off', {'NTT_ARENA_WORKSPACE_POOL': '0'}, False),
        ('three_buffers', {'NTT_WORKSPACE_REUSE_BQ': '0'}, False),
        ('arena_refusal', {'NTT_ARENA_CAP_KB': '1'}, False),
        ('carry_fault', {'NTT_S4_CARRY_TEST_BAD': '1'}, True),
    ]
    try:
        for name, override, fatal in policies:
            command = list(row['command']); command[0] = str(exe)
            if name == 'arena_refusal':
                # Zero CLI leaves the explicit environment cap untouched.
                command[command.index('--arena-mb')+1] = '0'
            log, debug, output = out/(name+'.log'), out/(name+'.debug.log'), out/(name+'.jsonl')
            for flag, path in (('--log', log), ('--debug-log-file', debug), ('--results', output)):
                command[command.index(flag)+1] = str(path)
            env = {k: v for k, v in os.environ.items() if not k.startswith('NTT_')}
            env.update(row['environment']); env.update(NTT_S4_CHUNK_MAX='1', **override)
            proc = subprocess.run(command, env=env, capture_output=True, timeout=90)
            driver = out/(name+'_driver.log'); driver.write_bytes(proc.stdout+proc.stderr)
            text = driver.read_text(encoding='utf-8', errors='replace')
            for path in (log, debug):
                if path.exists(): text += '\n'+path.read_text(encoding='utf-8')
            state = dict(name=name, command=command, returncode=proc.returncode,
                environment={k:v for k,v in env.items() if k.startswith('NTT_')},
                log_sha256=sha(log) if log.exists() else None,
                debug_sha256=sha(debug) if debug.exists() else None, driver_sha256=sha(driver))
            rows.append(state)
            if fatal:
                if (proc.returncode == 0 or 's4_carry_fault:' not in text or
                    'deferred carry check of the chunked multiply failed' not in text or
                    (output.exists() and output.read_text(encoding='utf-8').strip())):
                    raise ValueError('poisoned interior carry verdict was not rejected before publishing a result')
                state['verified_fatal'] = True
            else:
                if proc.returncode: raise ValueError(name+' failed; raw logs retained')
                native = json.loads(output.read_text(encoding='utf-8'))
                leaf = fields(text, 'target_descent_values')
                if leaf != expected or native['bad_factors'] or native['factors'] != row['result']['factors']:
                    raise ValueError(name+': independent full target leaves or factors differ')
                if any(t not in text for t in ('gmp_check_bad=0','gmp_selftest_bad=0','pending=0')):
                    raise ValueError(name+': required arithmetic check missing')
                state.update(leaf=leaf, chunk_plan=fields(text,'s4_chunk_plan'), result_sha256=sha(output))
                if name == 'arena_refusal' and fields(text,'real_batched_breakdown')['arena_overflow'] == '0':
                    raise ValueError('arena refusal did not reach per-call fallback')
            if sha(exe) != identity['binary_sha256'] or sha(ref) != ref_sha or sha(__file__) != tool_sha or sha(save) != data['input']['save_sha256']:
                raise ValueError('build, reference, collector or save changed')
            print(name, 'OK', flush=True)
        result['complete'] = True
    except Exception as exc:
        result['error'] = str(exc); raise
    finally:
        (out/'checks.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
