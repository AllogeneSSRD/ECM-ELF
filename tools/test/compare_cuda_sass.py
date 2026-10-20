"""Compare complete CUDA instruction text, allowing only NVCC anonymous file identities."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def anonymous_name(name):
    found = re.match(r'_ZN(\d+)', name)
    if found:
        start = found.end(); end = start+int(found[1])
        if name[start:end].startswith('_GLOBAL__N__'):
            return '_ZNANONYMOUS'+name[end:]
    return name


def fingerprints(path):
    result = {}; current = None; digest = None; count = 0
    def finish():
        if current is not None:
            key = anonymous_name(current)
            assert key not in result, 'anonymous-name collision: '+key
            result[key] = dict(symbol=current, sha256=digest.hexdigest(), instructions=count)
    with path.open(encoding='utf-8-sig', errors='strict') as stream:
        for line in stream:
            if 'Function : ' in line:
                finish(); current=line.split('Function : ', 1)[1].strip(); digest=hashlib.sha256(); count=0
            elif current is not None:
                found = re.match(r'\s*/\*[0-9a-f]+\*/\s*(.*?)\s*/\*', line)
                if found and found[1]:
                    digest.update((found[1]+'\n').encode()); count += 1
    finish(); return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--baseline-sass', type=Path)
    p.add_argument('--cuobjdump', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); out=a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    old = a.baseline_sass.resolve() if a.baseline_sass else out/'baseline.sass'
    new = out/'candidate.sass'
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    identities = dict(baseline=sha(a.baseline), candidate=sha(a.candidate))
    for exe, path in [(a.baseline, old), (a.candidate, new)]:
        if exe == a.baseline and a.baseline_sass: continue
        with path.open('wb') as stdout, (out/(path.stem+'.stderr.log')).open('wb') as stderr:
            subprocess.run([str(a.cuobjdump), '--dump-sass', str(exe.resolve())], stdout=stdout, stderr=stderr, check=True)
    before, after = fingerprints(old), fingerprints(new)
    missing=sorted(before.keys()-after.keys()); added=sorted(after.keys()-before.keys())
    changed=[name for name in before.keys() & after.keys() if before[name]['sha256'] != after[name]['sha256']]
    assert identities == dict(baseline=sha(a.baseline), candidate=sha(a.candidate)), 'binary changed during comparison'
    result=dict(complete=not (missing or added or changed), identity=identities,
                baseline_functions=len(before),candidate_functions=len(after),missing=missing,added=added,changed=sorted(changed),
                baseline=before,candidate=after)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(complete=result['complete'],baseline_functions=len(before),candidate_functions=len(after),
                         missing=missing[:4],added=added[:4],changed=sorted(changed)[:6])))
    if not result['complete']: raise SystemExit(1)


if __name__ == '__main__': main()
