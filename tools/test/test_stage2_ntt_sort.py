"""Check native tune serialization and the lossless formatter against a frozen profile."""
import argparse
import json
import math
from pathlib import Path
import random
import re
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/bench'))
from format_stage2_tune import format_rows
from stage2_tune_matrix import NTT_COLUMN_ORDER, literal
from stage2_tune_tables import summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--fixture-exe', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    text = a.profile.read_text(encoding='utf-8-sig')
    rng = random.Random(17)
    def shuffle(match):
        header,*lines = match[1].splitlines(keepends=True)
        rng.shuffle(lines)
        return 'matrix = [\n' + header + ''.join(lines) + ']'
    shuffled = re.sub(r'matrix\s*=\s*\[\s*\n(.*?)^\]', shuffle, text, flags=re.S | re.M)
    path = a.output / 'shuffled.toml'
    path.write_text(shuffled, encoding='utf-8')
    original = summaries(tomllib.loads(shuffled), 'ntt')
    key = lambda row: tuple(row.get(k, 0) for k in ('log2_length','length','batch','tile'))
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
        assert all(row.get(k)==v for k,v in old.items()),'existing statistics/shape/provenance changed'
        assert row['log2_length']==row['length'].bit_length()-1
        assert math.isclose(row['iter_per_s'],row['batch']/row['mean_seconds'],rel_tol=1e-12)
        assert math.isclose(row['batch_iter_per_s'],1/row['mean_seconds'],rel_tol=1e-12)
        if row.get('max_radix_bits') and 0<=row.get('tile',-1)<=row['log2_length']:
            remaining=row['log2_length']-row['tile'];radix=row['max_radix_bits']
            assert row['passes_forward']==(remaining+radix-1)//radix+int(row['tile']>0)
    assert tuple(json.loads(subprocess.check_output([str(a.fixture_exe.resolve()),'--ntt-columns'],text=True)))==NTT_COLUMN_ORDER
    for table in document['ntt'].values():
        assert table['matrix'][0][:4]==['log2_length','length','iter_per_s','mean_seconds']
    (a.output / 'native.toml').write_text(native, encoding='utf-8')
    assert subprocess.check_output([str(a.fixture_exe.resolve()),'--roundtrip',str((a.output/'native.toml').resolve())],text=True)==native
    assert format_rows(native)==(native,0)
    # Derived display fields must not hide stale or corrupt metadata on reload.
    assert len(document['ntt'])==1,'this gate expects one frozen NTT condition'
    table=next(iter(document['ntt'].values()));columns,*values=table['matrix']
    for field in ('log2_length','iter_per_s','batch_iter_per_s','passes_forward'):
        damaged=[list(row) for row in values];column=columns.index(field)
        assert isinstance(damaged[0][column],(int,float))
        damaged[0][column]+=1
        text=native.split('matrix = [',1)[0]+'matrix = [\n'+''.join(
            '['+', '.join(map(literal,row))+'],\n' for row in [columns,*damaged])+']\n'
        bad=a.output/(field+'_bad.toml');bad.write_text(text,encoding='utf-8')
        assert subprocess.run([str(a.fixture_exe.resolve()),'--roundtrip',str(bad.resolve())],capture_output=True).returncode!=0,field
    assert not any(k in row for row in expected for k in ('setup_seconds','warmup_seconds','verification_seconds'))
    print(f'PASS: {len(expected)} NTT rows, shared columns, derived metadata, numeric order, statistics preserved, corrupt metadata rejected')


if __name__ == '__main__':
    main()
