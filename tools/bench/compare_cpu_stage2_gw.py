"""Paired static/DLL Edwards Stage2 measurement from a single validated save."""
import argparse
import hashlib
import json
import platform
import statistics
import shutil
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('static', 'dll', 'save', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('b2', 'd'):
        parser.add_argument('--' + name, type=int, required=True)
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 4])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--exponent', type=int)
    parser.add_argument('--known-factor', action='append', default=[])
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    save = out / 'input.save'
    shutil.copy2(args.save, save)
    engines = {'static': args.static.resolve(), 'dll': args.dll.resolve()}
    config = out / 'ecm.toml'
    config.write_text('config_version = 4\n', encoding='utf-8')
    report = dict(os=platform.platform(), save_sha256=digest(save), b2=args.b2, d=args.d,
                  power='unmeasured', affinity='inherited', repeats=args.repeats, binaries={}, runs=[], cells=[])
    for name, exe in engines.items():
        paths = [exe, exe.parent / 'gmp-10.dll']
        if name == 'dll':
            paths.append(exe.parent / 'gwnum.dll')
        snapshot = out / name
        snapshot.mkdir()
        for path in paths:
            shutil.copy2(path, snapshot / path.name)
        engines[name] = snapshot / exe.name
        report['binaries'][name] = {p.name: digest(snapshot / p.name) for p in paths}
    reference = None
    for threads in args.threads:
        times = {name: [] for name in engines}
        for repeat in range(args.repeats + 1):
            for name in (list(engines) if repeat % 2 else list(reversed(engines))):
                tag = f'{name}_t{threads}_r{repeat}'
                cmd = [engines[name], '--config', config, '--save', save, '--curves', '1',
                       '--backend', 'gwpoly', '--b2', str(args.b2), '--threads', str(threads),
                       '--d', str(args.d), '--results', out / (tag + '.jsonl'),
                       '--submission', out / (tag + '.json.txt')]
                if args.exponent:
                    cmd += ['--exponent', str(args.exponent)]
                for factor in args.known_factor:
                    cmd += ['--known-factor', factor]
                if repeat == 0:
                    cmd.append('--gw-check')
                with (out / (tag + '.log')).open('wb') as log:
                    subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=out, check=True)
                row = json.loads((out / (tag + '.jsonl')).read_bytes())
                stats = row['stage2-engine']
                geometry = [row[k] for k in ('b1', 'b2', 'Edwards', 'd', 'poly-size', 'fft-length')]
                geometry.extend([stats['fft-description'], stats['polymult-isa']])
                if reference is None:
                    reference = geometry
                assert geometry == reference and row['status'] == 'NF'
                assert stats['gw-build'].startswith('static' if name == 'static' else 'gwnum.dll')
                assert stats['max-roundoff'] <= .40 and stats['checked-values'] == (3 if repeat == 0 else 0)
                report['runs'].append(dict(tag=tag, threads=threads, geometry=geometry, **stats))
                if repeat:
                    times[name].append(stats['seconds'])
                (out / 'summary.json').write_text(json.dumps(report, indent=2))
        medians = {name: statistics.median(values) for name, values in times.items()}
        report['cells'].append(dict(threads=threads, medians=medians, ratio=medians['dll']/medians['static']))
    (out / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report['cells']))


if __name__ == '__main__':
    main()
