"""Check native tune serialization and the lossless formatter against a frozen profile."""
import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/bench'))
from format_stage2_tune import format_rows
from stage2_tune_matrix import NTT_COLUMN_ORDER, literal, matrix_row_key
from stage2_tune_tables import summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--fixture-exe', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    text = a.profile.read_text(encoding='utf-8-sig')
    def reverse_rows(match):
        header,*lines = match[1].splitlines(keepends=True)
        lines.reverse()
        return 'matrix = [\n' + header + ''.join(lines) + ']'
    ordered, _ = format_rows(text)
    shuffled = re.sub(r'matrix\s*=\s*\[\s*\n(.*?)^\]', reverse_rows, ordered, flags=re.S | re.M)
    path = a.output / 'shuffled.toml'
    path.write_text(shuffled, encoding='utf-8')
    original = summaries(tomllib.loads(shuffled), 'ntt')
    key = lambda row: matrix_row_key(dict({'ntt_type':'stage2_convolution'}, **row),'ntt')
    expected = sorted(original, key=key)
    formatted, changed = format_rows(shuffled)
    assert changed > 0
    assert summaries(tomllib.loads(formatted), 'ntt') == expected
    assert format_rows(formatted) == (formatted, 0)
    native = subprocess.check_output([str(a.fixture_exe.resolve()), '--roundtrip', str(path.resolve())], text=True)
    document=tomllib.loads(native)
    actual=summaries(document,'ntt')
    assert len(actual)==len(expected)
    for row,old in zip(actual,expected):
        for k,v in old.items():
            if k in ('verified','measurement_ids'):continue
            if isinstance(v,float):assert math.isclose(row[k],v,rel_tol=1e-12),k
            else:assert row.get(k)==v,k
        assert row['log2_length']==row['length'].bit_length()-1
        assert math.isclose(row['iter_per_s'],row['batch']/row['mean_seconds'],rel_tol=1e-12)
        assert math.isclose(row['batch_iter_per_s'],1/row['mean_seconds'],rel_tol=1e-12)
        if row.get('max_radix_bits') and 0<=row.get('tile',-1)<=row['log2_length']:
            remaining=row['log2_length']-row['tile'];radix=row['max_radix_bits']
            assert row['passes_forward']==(remaining+radix-1)//radix+int(row['tile']>0)
    assert tuple(json.loads(subprocess.check_output([str(a.fixture_exe.resolve()),'--ntt-columns'],text=True)))==NTT_COLUMN_ORDER
    assert set(document['ntt'])=={'matrix'}
    assert document['ntt']['matrix'][0][:4]==['length','iter_per_s','batch','ntt_type']
    assert 'log2_length' not in document['ntt']['matrix'][0]
    assert not {'measurement_ids','verified','mean_seconds','batch_iter_per_s'} & set(document['ntt']['matrix'][0])
    (a.output / 'native.toml').write_text(native, encoding='utf-8')
    assert subprocess.check_output([str(a.fixture_exe.resolve()),'--roundtrip',str((a.output/'native.toml').resolve())],text=True)==native
    assert format_rows(native)==(native,0)
    # An old table may provide log2_length. The reader validates it before migration.
    legacy = tomllib.loads(text)['ntt']
    legacy = legacy if 'matrix' in legacy else next(iter(legacy.values()))
    old_cols, *old_rows = legacy['matrix']
    if 'log2_length' in old_cols:
        col = old_cols.index('log2_length')
        old_rows[0][col] += 1
        bad = a.output / 'legacy_log2_bad.toml'
        bad.write_text(text.rsplit('matrix = [', 1)[0] + 'matrix = [\n' + ''.join(
            json.dumps(r) + ',\n' for r in [old_cols, *old_rows]) + ']\n', encoding='utf-8')
        assert subprocess.run([str(a.fixture_exe.resolve()), '--roundtrip', str(bad)], capture_output=True).returncode
    # Reject inconsistent shape and non-positive throughput.
    columns,*values=document['ntt']['matrix']
    for field in ('iter_per_s','passes_forward'):
        damaged=[list(row) for row in values];column=columns.index(field)
        assert isinstance(damaged[0][column],(int,float))
        damaged[0][column]=0 if field=='iter_per_s' else damaged[0][column]+1
        text=native.rsplit('matrix = [',1)[0]+'matrix = [\n'+''.join(
            '['+', '.join(map(literal,row))+'],\n' for row in [columns,*damaged])+']\n'
        bad=a.output/(field+'_bad.toml');bad.write_text(text,encoding='utf-8')
        assert subprocess.run([str(a.fixture_exe.resolve()),'--roundtrip',str(bad.resolve())],capture_output=True).returncode!=0,field
    assert not any(k in row for row in expected for k in ('setup_seconds','warmup_seconds','verification_seconds'))
    print(f'PASS: {len(expected)} NTT rows, shared columns, derived metadata, numeric order, statistics preserved, corrupt metadata rejected')


if __name__ == '__main__':
    main()
