"""Check the profit auditor against completed evidence and damaged private copies.

No extra curves are executed. Original evidence is never edited. Corruptions
live in explicitly labelled synthetic fixtures under a new ignored directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/bench'))
from audit_stage2_tune_auto_profit import audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    evidence = a.evidence.resolve()
    originals = {path.relative_to(evidence): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in evidence.rglob('*') if path.is_file()}
    positive = audit(evidence, True)
    report = json.loads((evidence / 'result.json').read_text())
    selected = next(c for c in report['candidates'] if c['selected'])
    stem = f"candidate_{selected['index']}_repeat1"

    def change_json(folder, name, mutate):
        path = folder / name
        value = json.loads(path.read_text(encoding='utf-8-sig'))
        mutate(value)
        path.write_text(json.dumps(value) + '\n')

    def change_check(folder):
        path = folder / (stem + '.log')
        text = path.read_text()
        assert 'gmp_check_bad=0' in text
        path.write_text(text.replace('gmp_check_bad=0', 'gmp_check_bad=1', 1))

    def change_modules(folder):
        def mutate(value):
            processes = [value] if isinstance(value, dict) else value
            processes[0]['modules'][0]['sha256'] = '0' * 64
        change_json(folder, 'loaded_modules.json', mutate)

    def change_frozen(folder):
        original, expected = next(iter(report['source_identities'].items()))
        path = folder / 'inputs' / expected / Path(original).name
        path.write_bytes(path.read_bytes() + b'\n')

    def change_nvml(folder):
        path = folder / 'telemetry.csv'
        text = path.read_text()
        import csv
        rows = list(csv.reader(text.splitlines()))
        uuid = next(row[2].strip() for row in rows[1:] if len(row) == 8 and row[1].strip() == str(positive['device']))
        path.write_text(text.replace(uuid, 'GPU-' + '0' * 32))

    mutations = {
        'incomplete': lambda f: change_json(f, 'result.json', lambda r: r.update(complete=False)),
        'wrong_t1': lambda f: change_json(f, 'result.json', lambda r: r.update(t1_seconds=r['t1_seconds'] * 2)),
        'wrong_score': lambda f: change_json(f, 'result.json', lambda r: r['candidates'][0].update(actual_score=1)),
        'wrong_receipt_target': lambda f: change_json(f, stem + '.jsonl', lambda r: r.update(N_hex='7')),
        'arithmetic_bad': change_check,
        'wrong_loaded_module': change_modules,
        'changed_frozen_source': change_frozen,
        'wrong_nvml_uuid': change_nvml,
    }
    rejected = []
    for name, mutate in mutations.items():
        folder = out / ('synthetic_' + name)
        shutil.copytree(evidence, folder)
        mutate(folder)
        try:
            audit(folder, True)
        except AssertionError as error:
            rejected.append(name)
            (out / (name + '.error.txt')).write_text(repr(error))
        else:
            raise AssertionError('damaged evidence accepted: ' + name)
    assert all(hashlib.sha256((evidence / name).read_bytes()).hexdigest() == expected
               for name, expected in originals.items())
    result = dict(complete=True, audit_curves=positive['counts']['curves'],
                  rejected_synthetic_fixtures=rejected, extra_gpu_curves=0,
                  original_evidence_unchanged=True)
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
