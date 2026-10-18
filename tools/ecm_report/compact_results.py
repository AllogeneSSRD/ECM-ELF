"""Create review copies of legacy results; never overwrite inputs or submit them."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
from result_format import modulus
from rewrite_stage2_results import checksum


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def compact(old, targets, os_name, architecture):
    if 'N_hex' not in old:
        return old
    n = int(old['N_hex'], 16)
    if old.get('status') not in ('stage2_completed', 'factor_in_saved_X') or old.get('bad_factors', 0):
        raise ValueError('Legacy row is not a successful checked result')
    if any(not 1 < int(f) < n or n % int(f) for f in old.get('factors', [])):
        raise ValueError('Legacy result contains an invalid factor')
    target = targets.get(n)
    if target is None:
        raise ValueError('No verified compact target identity for legacy modulus')
    hit = bool(old.get('factors'))
    stage = 1 if old['status'] == 'factor_in_saved_X' else 2
    row = dict(status='F' if hit else 'NF', exponent=target['exponent'], worktype='ECM', b1=old['B1'])
    if not hit or stage != 1:
        row['b2'] = old['B2']
    if old.get('d') and stage != 1:
        row.update(d=old['d'], **{'poly-size': old['poly-size']})
    row['curves'] = 1
    if old['curve'] == 'ed':
        row['Edwards'] = dict(sigma=old['sigma'])
    else:
        row.update(sigma=old['sigma'], param=old.get('param', 0))
    row['fft-length'] = 0
    if hit:
        row.update(stage=stage, factors=old['factors'])
    if target.get('known-factors'):
        row['known-factors'] = target['known-factors']
    row.update(errors=dict(invariant=0), program=old['program'],
               **{'source-build': 'unknown (legacy record)'},
               os=dict(os=os_name, architecture=architecture),
               timestamp=dt.datetime.fromtimestamp(old['timestamp_ms']/1000, dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S'))
    for key in ('user', 'computer', 'aid'):
        if key in old:
            row[key] = old[key]
    row['queue_receipt'] = old.get('queue_receipt', '')
    excluded = {'N_hex', 'B1', 'B2', 'sigma', 'curve', 'param', 'factors', 'program',
                'timestamp_ms', 'submission_file', 'reference_fft_table', 'queue_receipt',
                'tune_observation', 'd', 'poly-size', 'user', 'computer', 'aid'}
    if not old.get('relocation'):
        excluded.update(k for k in old if k.startswith('relocation') or k.startswith('giant_'))
    if not old.get('auto_b2'):
        excluded.update(('auto_b2', 'auto_planning_seconds', 'auto_plan'))
    if not old.get('factorization_seconds'):
        excluded.add('factorization_seconds')
    if not old.get('requested_factor_only'):
        excluded.add('requested_factor_only')
    row['stage2-engine'] = dict(name='cuda-ntt', **{k: v for k, v in old.items() if k not in excluded})
    if 'tune_observation' in old:
        row['tune_observation'] = old['tune_observation']
    row['checksum'] = checksum(row)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('stage1', 'save', 'stage2', 'submission', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--os', default='Windows', help='Source operating system for legacy records')
    p.add_argument('--architecture', default='x86_64', help='Source architecture for legacy records')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    first, second, submitted = rows(a.stage1), rows(a.stage2), rows(a.submission)
    sigmas = [int(re.search(r'\bSIGMA=(\d+)', line)[1]) for line in a.save.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    index = {r.get('Edwards', {}).get('sigma', r.get('sigma')): r for r in first}
    if len(index) != len(first) or len(set(sigmas)) != len(sigmas) or set(index) != set(sigmas):
        raise ValueError('Stage1/save sigma sets are not a one-to-one batch')
    # Verify N and B1 against every save row, in addition to sigma set equality.
    targets = {}
    for line, sigma in zip(a.save.read_text(encoding='utf-8-sig').splitlines(), sigmas):
        fields = dict(token.strip().split('=', 1) for token in line.split(';') if token.strip())
        row = index[sigma]
        if modulus(row) != int(fields['N'], 0) or row['b1'] != int(fields['B1']):
            raise ValueError('Stage1/save target or bound mismatch')
        targets[modulus(row)] = row
    converted = [compact(r, targets, a.os, a.architecture) for r in second]
    repaired = []
    for old in submitted:
        row = dict(old)
        if modulus(row) not in targets:
            raise ValueError('Submission target does not match the audited inputs')
        row.setdefault('errors', dict(invariant=0))
        row.setdefault('os', dict(os=a.os, architecture=a.architecture))
        row['checksum'] = checksum(row)
        repaired.append(row)
    for name, records in [('stage1_results.jsonl', [index[s] for s in sigmas]),
                          ('stage2_results.jsonl', converted), ('results.json_2.txt', repaired)]:
        (a.output/name).write_text(''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':'))+'\n' for r in records), encoding='utf-8')
    audit = dict(stage1_rows=len(first), stage2_rows=len(second), submission_rows=len(submitted),
                 source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.stage1, a.save, a.stage2, a.submission)},
                 stage2_input_bytes=a.stage2.stat().st_size,
                 stage2_output_bytes=(a.output/'stage2_results.jsonl').stat().st_size,
                 original_program_versions_preserved=True, submitted=False)
    (a.output/'audit.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in audit.items() if k != 'source_hashes'}))


if __name__ == '__main__':
    main()
