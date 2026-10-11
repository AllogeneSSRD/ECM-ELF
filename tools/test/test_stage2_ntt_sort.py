"""Check native tune serialization and the lossless formatter against a frozen profile."""
import argparse
from pathlib import Path
import random
import re
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/bench'))
from format_stage2_tune import format_rows
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
        lines = match[1].splitlines(keepends=True)
        rng.shuffle(lines)
        return 'rows = [\n' + ''.join(lines) + ']'
    shuffled = re.sub(r'rows\s*=\s*\[\s*\n(.*?)^\]', shuffle, text, flags=re.S | re.M)
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
    assert summaries(tomllib.loads(native), 'ntt') == expected
    (a.output / 'native.toml').write_text(native, encoding='utf-8')
    assert not any(k in row for row in expected for k in ('setup_seconds','warmup_seconds','verification_seconds'))
    print(f'PASS: {len(expected)} NTT rows, numeric order, measurements unchanged, formatter idempotent')


if __name__ == '__main__':
    main()
