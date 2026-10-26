"""Compare real PARAM0/Edwards continuation across frozen and current layouts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from test_result_output_consistency import crc_valid


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'grouped', 'isolated', 'mont_save', 'ed_save', 'output'):
        p.add_argument('--'+name.replace('_', '-'), type=Path, required=True)
    p.add_argument('--device', type=int, default=0)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(complete=False, device=a.device, b2=4000, curves=2, cases=[])
    for model, source in [('mont', a.mont_save), ('ed', a.ed_save)]:
        expected = None
        for label in ['baseline', 'grouped', 'isolated']:
            exe = getattr(a, label).resolve()
            work = out/(model+'_'+label)
            work.mkdir()
            save = work/'points.save'
            shutil.copy2(source, save)
            ini = work/'ecm.toml'
            ini.write_text('[shared]\nuser=""\ncomputer=""\n[stage2]\nstage2_log_file=""\n')
            command = [str(exe), '--config', str(ini), '--save', str(save), '--b2', '4000',
                       '--curves', '2', '--device', str(a.device), '--d', '90',
                       '--short-calibration', '0', '--tune-runtime', '0', '--auto-carrier', '0',
                       '--results', str(work/'internal.jsonl'), '--submission', str(work/'results.txt')]
            run = subprocess.run(command, cwd=work, capture_output=True, timeout=180)
            (work/'run.log').write_bytes(run.stdout+run.stderr)
            assert run.returncode == 0, (model, label, run.returncode, (run.stdout+run.stderr)[-2500:])
            rows = [json.loads(line) for line in (work/'internal.jsonl').read_text().splitlines()]
            assert len(rows) == 2 and all(crc_valid(row) for row in rows)
            if label != 'baseline':
                assert all(row['program']['version'] == '0.53.4' for row in rows)
            sigma = lambda row: row.get('Edwards', {}).get('sigma', row.get('sigma'))
            assert [sigma(row) for row in rows] == [1000, 1001]
            assert all(('Edwards' in row) == (model == 'ed') for row in rows)
            # A factor already present in the Stage1 X has stage=1 and no B2.
            projection = [(sigma(row), row['status'], row.get('factors', []), row['b1'],
                           row.get('b2'), row.get('stage')) for row in rows]
            if expected is None:
                expected = projection
            assert projection == expected, ('Stage2 mathematical result differs', model, label)
            report['cases'].append(dict(model=model, layout=label, records=2,
                                        binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest()))
            print('PASS', model, label, flush=True)
            (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    report['complete'] = True
    (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
