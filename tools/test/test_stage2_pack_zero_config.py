"""Exercise generated INI / worker / environment / CLI pack-zero precedence on real curves."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--save', type=Path, required=True)
    p.add_argument('--device', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    exe = a.exe.resolve(); save = out/'input.save'; save.write_bytes(a.save.read_bytes())
    root = Path(__file__).resolve().parents[2]
    build=json.loads((exe.parent/'build_manifest.json').read_text(encoding='utf-8-sig'))
    if sha(exe)!=build['sha256'].lower():
        raise ValueError('binary/build identity mismatch')
    schema=exe.parent/'sources/config/ecm_options.json'
    if not schema.exists():schema=root/'config/ecm_options.json'
    if sha(schema)!=build['source_hashes']['config/ecm_options.json'].lower():
        raise ValueError('compiled schema identity mismatch')
    expected_default=int(next(o['default'] for o in json.loads(schema.read_text(encoding='utf-8'))['options']
                              if o['owner']=='stage2' and o['key']=='stage2_pack_zero'))
    # B1=20, sigma=26, 318-bit cofactor saves are used in the documented fixture.
    tests = [
        ('default', '', {}, [], expected_default),
        ('ini_off', 'stage2_pack_zero=false\n', {}, [], 0),
        ('ini_on', 'stage2_pack_zero=true\n', {}, [], 1),
        ('worker_off', 'stage2_pack_zero=true\n[Worker #2]\nstage2_pack_zero=false\n', {}, ['--worker','2'], 0),
        ('worker_on', 'stage2_pack_zero=false\n[Worker #2]\nstage2_pack_zero=true\n', {}, ['--worker','2'], 1),
        ('env_off', '', {'NTT_S4_PACK_ZERO':'0'}, [], 0),
        ('env_on', '', {'NTT_S4_PACK_ZERO':'1'}, [], 1),
        ('ini_above_env', 'stage2_pack_zero=true\n', {'NTT_S4_PACK_ZERO':'0'}, [], 1),
        ('cli_off', 'stage2_pack_zero=true\n', {'NTT_S4_PACK_ZERO':'1'}, ['--pack-zero','0'], 0),
        ('cli_on', 'stage2_pack_zero=false\n', {'NTT_S4_PACK_ZERO':'0'}, ['--pack-zero','1'], 1),
        ('legacy_implicit_off', '', {}, ['--cost-profile',str(out/'unused.cprof')], 0),
        ('bad_ini', 'stage2_pack_zero=2\n', {}, [], None),
        ('bad_env', '', {'NTT_S4_PACK_ZERO':'invalid'}, [], None),
        ('bad_cli', '', {}, ['--pack-zero','2'], None),
    ]
    report = dict(binary_sha256=sha(exe), save_sha256=sha(save), tool_sha256=sha(__file__),
                  schema_sha256=sha(schema), expected_default=expected_default,
                  device=a.device, complete=False, runs=[])
    for name, text, changes, extra, expected in tests:
        if sha(exe)!=report['binary_sha256'] or sha(save)!=report['save_sha256']:
            raise ValueError('binary/input changed')
        ini = out/(name+'.ini')
        ini.write_text('stage2_short_calibration=false\nstage2_tune_runtime=false\n'+text)
        log, result = out/(name+'.engine.log'), out/(name+'.jsonl')
        env = {k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
        env.update(changes)
        cmd = [str(exe), '--ini',str(ini),'--save',str(save),'--device',str(a.device),
               '--curves','1','--b2','260000000','--d','30030','--short-calibration','0',
               '--factor-only','--log-level','phases','--log',str(log),'--results',str(result),*extra]
        run = subprocess.run(cmd, env=env, cwd=root, capture_output=True, text=True,
                             encoding='utf-8', errors='replace', timeout=120,
                             creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        (out/(name+'.stdout.log')).write_text(run.stdout+run.stderr, encoding='utf-8')
        if expected is None:
            if run.returncode == 0 or result.exists():
                raise ValueError(name+': invalid option was not rejected before a curve')
        else:
            if run.returncode:
                raise ValueError(name+': curve failed')
            text = log.read_text(encoding='utf-8')
            stats = re.findall(r'^s4_pack_zero_stats: (.*)$', text, re.M)
            if len(stats)!=1:
                raise ValueError(name+': missing child statistics')
            fields = dict(re.findall(r'(\w+)=(\S+)', stats[0]))
            if int(fields['requested'])!=expected or bool(int(fields['words']))!=bool(expected):
                raise ValueError(name+': resolved value did not reach the curve worker')
            s4 = re.search(r'^s4_multiply_stats: (.*)$', text, re.M)
            f = dict(re.findall(r'(\w+)=(\S+)', s4[1]))
            if (f['gmp_selftest_bad']!='0' or f['gmp_check_bad']!='0'
                    or not int(f['gmp_checked']) or not int(f['gmp_selftest_cases'])):
                raise ValueError(name+': arithmetic checks failed')
            receipts=[json.loads(line) for line in result.read_text(encoding='utf-8').splitlines()]
            if len(receipts)!=1 or receipts[0]['status']!='stage2_completed' or receipts[0]['bad_factors']:
                raise ValueError(name+': missing or invalid completion receipt')
        report['runs'].append(dict(name=name, expected=expected, returncode=run.returncode,
                                   command=cmd, environment=changes))
        (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        print(name+': PASS',flush=True)
    report['complete']=True
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
