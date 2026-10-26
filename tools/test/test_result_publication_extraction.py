"""Prove result encoding, CRC and ledger ordering moved without protocol changes."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from test_stage1_host_modules import normalize
from source_layout import frozen_source

ROOT=Path(__file__).resolve().parents[2]


def body(text,name):
    # Empty default arguments contain braces; find the definition after its
    # balanced parameter list rather than treating those defaults as a body.
    masked=re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|/\*.*?\*/|//[^\n]*',lambda m:' '*len(m[0]),text,flags=re.S)
    def end(at,left,right):
        depth=0
        for i in range(at,len(masked)):
            depth+=(masked[i]==left)-(masked[i]==right)
            if not depth:return i
        raise AssertionError(name)
    for m in re.finditer(r'\b'+name+r'\s*\(',masked):
        at=masked.index('(',m.start());i=end(at,'(',')')+1
        while i<len(masked) and masked[i].isspace():i+=1
        if i<len(masked) and masked[i]=='{':return normalize(text[i:end(i,'{','}')+1])
    raise AssertionError(name)


def main():
    from test_stage2_planning_extraction import balanced
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    before = frozen_source('6dad3d1', 'src/io/results.cpp')
    after = (ROOT/'src/io/results.cpp').read_text(encoding='utf-8-sig')
    # M1 removes only the redundant writer. Undo its twelve call-site changes
    # before comparing every encoder, CRC, lock, ledger and ordering token.
    start = before.index('void atomic_text(')
    opening = before.index('{', start)
    before = before[:start]+before[balanced(before, opening, '{', '}'):]
    pattern = re.compile(r'ecm_io::write_atomic\s*\(')
    changes = []
    for match in pattern.finditer(after):
        opening = after.index('(', match.start())
        end = balanced(after, opening, '(', ')')
        args = after[opening+1:end-1]
        suffix = re.search(r',\s*"\.new"\s*$', args)
        if not suffix:
            continue  # The pre-existing append helper uses write_atomic's default suffix.
        changes.append((match.start(), end, 'atomic_text('+args[:suffix.start()]+')'))
    assert len(changes) == 12
    for start, end, text in reversed(changes):
        after = after[:start]+text+after[end:]
    strip = lambda text: normalize(re.sub(r'(?m)^\s*#\s*include[^\n]*', '', text))
    assert strip(before) == strip(after), 'result encoding/CRC/ledger/order changed'
    public = (ROOT/'include/ecm/io/results.h').read_text(encoding='utf-8-sig')
    assert strip(public) == strip(frozen_source('6dad3d1', 'include/ecm/io/results.h'))
    assert not re.search(r'#\s*include[^\n]*(?:windows|src/|cuda|cgbn)', public)
    assert not re.search(r'inline |std::ofstream|std::ifstream|directory_iterator|WriteFile\(', public)
    report = dict(complete=True, baseline='6dad3d1', encoders_crc_ledger_order_unchanged=True,
                  persistent_writer='ecm_io::write_atomic', compatibility_suffix='.new',
                  redirected_call_sites=12, public_layout_unchanged=True, gpu_runs=0)
    (a.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
