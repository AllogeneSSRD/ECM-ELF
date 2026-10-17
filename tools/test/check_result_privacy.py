"""Reject staged additions containing real local result identities/checksums.

Values are inspected only in memory and are never printed. Public schemas and
field names remain versioned; local configuration and result payloads do not.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-results', type=Path, action='append', default=[])
    p.add_argument('--ini', type=Path, action='append', default=[])
    args = p.parse_args();private = set()
    keys = {'user', 'computer', 'aid', 'security-code', 'checksum'}
    def inspect(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in keys and isinstance(v, str) and v:private.add(v)
                inspect(v)
        elif isinstance(obj, list):
            for v in obj:inspect(v)
    for path in args.reference_results:
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            if line.strip():inspect(json.loads(line))
    for path in args.ini:
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            if '=' not in line:continue
            key, value = map(str.strip, line.split('=', 1))
            if key.lower() in keys and value:private.add(value)
    diff = subprocess.check_output(['git', 'diff', '--cached', '--no-ext-diff', '--unified=0'], encoding='utf-8')
    additions = '\n'.join(line[1:] for line in diff.splitlines() if line.startswith('+') and not line.startswith('+++'))
    for value in private:
        # Check raw and JSON-escaped spellings. Do not reveal matching content.
        for spelling in {value, json.dumps(value, ensure_ascii=True)[1:-1], json.dumps(value, ensure_ascii=False)[1:-1]}:
            if spelling and spelling in additions:raise SystemExit('FAIL: staged additions contain a private local result value')
    staged = subprocess.check_output(['git', 'diff', '--cached', '--name-only'], encoding='utf-8').splitlines()
    if any('.ecm-ledger' in name.lower() or re.search(r'\.(?:jsonl|json(?:_\d+)?\.txt|local\.ini|ckpt)(?:\.|$)', name, re.I) for name in staged):
        raise SystemExit('FAIL: runtime result/configuration artifact is staged')
    print('PASS: staged additions contain no inspected private result values or runtime artifacts')


if __name__ == '__main__':main()
