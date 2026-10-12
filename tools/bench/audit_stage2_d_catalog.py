"""Compare native default/tuned plans and budget routes of frozen D catalogues.

Planning is not performance evidence. Every invocation is retained; GPU settings,
queues, input saves and performance profiles are never modified.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from bench_stage2_n_scaling import read, sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-study', type=Path, required=True)
    p.add_argument('--candidate-study', type=Path, required=True)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', type=int, default=0)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / 'audit_tool.py')
    studies = {arm: read(src / 'measurements.json') for arm, src in
               [('baseline', a.baseline_study), ('candidate', a.candidate_study)]}
    for study in studies.values():
        if sha(study['executable']) != study['executable_sha256']:
            raise ValueError('frozen binary changed')
        for case in study['cases']:
            if sha(case['save']) != case['save_sha256']:
                raise ValueError('frozen save changed')
    if studies['baseline']['dll_sha256'] != studies['candidate']['dll_sha256']:
        raise ValueError('DLL identity differs')
    profile = out / 'measured.toml'
    shutil.copy2(a.profile, profile)
    profile_hash = sha(profile)
    ini = out / 'audit.ini'
    ini.write_text('stage2_short_calibration=0\nstage2_debug_log=false\n', encoding='ascii')
    report = dict(binary_sha256={k: v['executable_sha256'] for k, v in studies.items()},
                  profile_sha256=profile_hash, audit_tool_sha256=sha(__file__),
                  plan_pairs=[], planning_only=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith('NTT_') and k != 'CUDA_VISIBLE_DEVICES'}
    env.update(NTT_NO_PROGRESS='1', CUDA_LAUNCH_BLOCKING='0')
    cases = [c for c in studies['baseline']['cases'] if c['variant'] == 'cofactor']
    requests = []
    for case in cases:
        for b2 in studies['baseline']['bounds']:
            for tuned in (False, True):
                requests.append((case, b2, 6300, 640, tuned))
    for case in cases:
        if case['exponent'] not in (503, 5003, 7001, 8011):
            continue
        for arena, fold in [(256, 48), (1024, 128), (1536, 256), (4096, 640), (6300, 0)]:
            requests.append((case, max(studies['baseline']['bounds']), arena, fold, False))
    for index, (case, b2, arena, fold, tuned) in enumerate(requests):
        pair = dict(exponent=case['exponent'], B2=b2, arena_mb=arena, fold_mb=fold,
                    tuned=tuned, arms={})
        for arm, study in studies.items():
            c = next(c for c in study['cases'] if c['id'] == case['id'])
            for key in ('N_hex', 'Q_sha256', 'save_sha256', 'B1', 'sigma'):
                if c[key] != case[key]:
                    raise ValueError('paired input differs: ' + key)
            command = [study['executable'], '--ini', str(ini), '--save', c['save'], '--b2', str(b2),
                       '--d', '0', '--plan-only', '--device', str(a.device), '--arena-mb', str(arena),
                       '--batch-mb', '256', '--owner-budget-mb', str(fold), '--carrier-exponent', '0',
                       '--short-calibration', '0', '--tune-profile', str(profile if tuned else out / 'absent.toml')]
            prefix = out / f'{index:03d}_{arm}'
            write(prefix.with_suffix('.command.json'), command)
            result = subprocess.run(command, cwd=out, env=env, capture_output=True,
                                    text=True, errors='replace', timeout=90)
            prefix.with_suffix('.log').write_text(result.stdout + result.stderr, encoding='utf-8')
            records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
            native = next((x for x in records if x.get('type') == 'stage2_plan'), None)
            selection = next((x for x in records if x.get('type') == 'tune_selection'), None)
            if native and (native.get('curves_executed') != 0 or 'request_program' not in native
                           or 'curve_workspace_memory' not in native):
                raise ValueError('native plan contract missing or a curve was executed')
            pair['arms'][arm] = dict(exit_code=result.returncode, plan=native, selection=selection)
        b, c = pair['arms']['baseline'], pair['arms']['candidate']
        pair['same_exit'] = b['exit_code'] == c['exit_code']
        fields = ['D', 'P', 'I', 'G', 'fold_length', 'tree_length', 'owner_reuse', 'request_program']
        pair['same_geometry'] = bool(b['plan'] and c['plan']) and all(b['plan'].get(k) == c['plan'].get(k) for k in fields)
        route = ('valid', 'finished', 'reason', 'fold_enabled', 'frontier_enabled', 'required_free_bytes')
        pair['same_memory_route'] = bool(b['plan'] and c['plan']) and all(
            b['plan']['curve_workspace_memory'].get(k) == c['plan']['curve_workspace_memory'].get(k)
            for k in route)
        pair['same_selection'] = b['selection'] == c['selection']
        report['plan_pairs'].append(pair)
        write(out / 'report.json', report)
        if sha(profile) != profile_hash:
            raise ValueError('planning changed frozen profile')
    report['default_pairs'] = sum(not x['tuned'] and x['arena_mb'] == 6300 and x['fold_mb'] == 640 for x in report['plan_pairs'])
    report['tuned_pairs'] = sum(x['tuned'] for x in report['plan_pairs'])
    report['budget_pairs'] = len(requests) - report['default_pairs'] - report['tuned_pairs']
    report['all_same_exit'] = all(x['same_exit'] for x in report['plan_pairs'])
    report['all_same_geometry'] = all(x['same_geometry'] for x in report['plan_pairs'] if x['arms']['baseline']['exit_code'] == 0)
    report['all_same_memory_route'] = all(x['same_memory_route'] for x in report['plan_pairs'] if x['arms']['baseline']['exit_code'] == 0)
    report['all_same_selection'] = all(x['same_selection'] for x in report['plan_pairs'])
    write(out / 'report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'plan_pairs'}))
    if not all(report[k] for k in ('all_same_exit', 'all_same_geometry',
                                  'all_same_memory_route', 'all_same_selection')):
        raise SystemExit('native plan drift; inspect the retained report before performance acceptance')


if __name__ == '__main__':
    main()
