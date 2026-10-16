"""Complete Stage1 baseline: Edwards vs Montgomery PARAM0/3 on one CUDA GPU.

Process wall includes startup, scalar/curve setup, GPU execution, GCD and saves.
Every saved point is stable across repeats; first/last points use an independent
ordinary-integer x-only ladder. Oracle time is outside the measured boundary.
Use a fresh output directory; raw logs and failed samples are never overwritten.
"""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools/test'))
from test_edwards_gpu_pipeline import curve, checksum, mont


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def integer_expression(text):
    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) is int: return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub): return -evaluate(node.operand)
        if isinstance(node, ast.BinOp):
            x, y = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add): return x+y
            if isinstance(node.op, ast.Sub): return x-y
            if isinstance(node.op, ast.Mult): return x*y
            if isinstance(node.op, ast.Pow): return x**y
            if isinstance(node.op, (ast.Div, ast.FloorDiv)):
                assert x % y == 0
                return x//y
        raise ValueError('unsupported saved N expression')
    return evaluate(ast.parse(text.replace('^', '**'), mode='eval').body)


def reference(n, sigma, b1, model):
    s = mont.lcm_1_to(b1)
    if model in ('ed', 'ed-5', 'ed-binary'):
        a24, x = curve(n, sigma); z = 1; s *= 48
    elif model == 'mont0':
        _, a24, x, z = mont.suyama_curve(sigma, n)
    else:
        a24 = sigma*pow(1 << 32, -1, n) % n; x, z = 2, 1
    x, z = mont.ladder(s, x, z, a24, n)
    g = math.gcd(z, n)
    return ('POINT', x*pow(z, -1, n) % n) if g == 1 else ('FACTOR' if g < n else 'IDENTITY', g)


def validate(path, n, b1, count, sigma, model):
    rows = []
    for line in path.read_text().splitlines():
        if 'SIGMA=' not in line: continue
        fields = dict(t.strip().split('=', 1) for t in line.split(';') if t.strip())
        sg = int(fields['SIGMA']); x = int(fields['X'], 0)
        saved_n = integer_expression(fields['N'])
        assert saved_n > 0 and n % saved_n == 0 and float(fields['B1']) == b1
        assert sg == sigma+len(rows)
        if model in ('ed', 'ed-5', 'ed-binary'):
            assert saved_n == n
            status = fields['STATUS']
            assert fields['METHOD'] == 'ECM-EDWARDS-AM'
            assert fields['SCALAR'] == '48LCM' and fields['VERSION'] == '1'
            assert int(fields['CHECKSUM']) == checksum(n, x, sg, b1, status)
            if status == 'FACTOR': assert 1 < x < n and n % x == 0
        else:
            assert fields['METHOD'] == 'ECM' and int(fields.get('PARAM', '0')) == int(model[-1])
            assert int(fields['CHECKSUM']) == b1*sg*n*x*(int(model[-1])+1) % 4294967291
            # Historical Mont saves have no status field and PARAM3 may rewrite
            # N after a batch hit. All endpoint checks use the actual input N.
            status = 'FACTOR' if 1 < x < n and n % x == 0 else ('IDENTITY' if x == 0 else 'POINT')
        assert 0 <= x < n
        rows.append((sg, status, x))
    assert len(rows) == count, (path, len(rows), count)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--bits', type=int, nargs='+', default=[347, 677, 991, 2203, 4003, 8191])
    p.add_argument('--b1', type=int, default=10000)
    p.add_argument('--curves', type=int, default=128)
    p.add_argument('--sigma', type=int, default=20260922)
    p.add_argument('--device', type=int, default=1)
    p.add_argument('--models', nargs='+', choices=['ed', 'ed-5', 'ed-binary', 'mont0', 'mont3'], default=['ed-binary', 'ed', 'mont0', 'mont3'])
    p.add_argument('--naf-w', type=int)
    p.add_argument('--doubling', type=int, choices=[4, 5])
    p.add_argument('--source-archive', type=Path, help='Prior frozen source snapshot for an already-built baseline')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert a.repeats >= 3 and a.curves > 0 and a.b1 >= 2
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    exe = a.exe.absolute()
    if not exe.is_file(): raise FileNotFoundError(exe)
    frozen = out/'bin'; frozen.mkdir()
    shutil.copy2(exe, frozen/exe.name)
    for dll in exe.parent.glob('*.dll'): shutil.copy2(dll, frozen/dll.name)
    exe = frozen/exe.name
    with zipfile.ZipFile(out/'tools.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in [Path(__file__).resolve(), ROOT/'tools/test/test_edwards_gpu_pipeline.py', ROOT/'tools/stat/suyama_mont_ref.py']:
            archive.write(path, path.relative_to(ROOT).as_posix())
    tracked = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', 'src', 'kernels', 'include', 'cgbn/include', 'config', 'CMakeLists.txt', 'tools/build'], cwd=ROOT).decode().splitlines()
    tracked += [str(path.relative_to(ROOT)).replace('\\', '/') for base in ['cgbn/include', 'third_party/gmp-zen3/dist/include'] for path in (ROOT/base).rglob('*') if path.is_file()]
    tracked = sorted(set(tracked))
    if a.source_archive:
        shutil.copy2(a.source_archive, out/'sources.zip')
        with zipfile.ZipFile(out/'sources.zip') as archive:
            sources = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()}
    else:
        sources = {path: sha(ROOT/path) for path in tracked if (ROOT/path).is_file()}
        with zipfile.ZipFile(out/'sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in sources: archive.write(ROOT/path, path)
            archive.write(__file__, 'tools/bench/bench_edwards_gpu_stage1.py')
    for name in ['CMakeCache.txt', 'compile_commands.json']:
        path = a.exe.absolute().parent/name
        if path.exists(): shutil.copy2(path, out/name)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('ECM_', 'NTT_'))}
    env['ECM_GPU_STAGE1_ALGO'] = 'ladder'
    if a.doubling is not None: env['ECM_EDWARDS_DBL_ALGO'] = str(a.doubling)
    info = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,uuid,driver_version,power.limit', '--format=csv']).decode()
    (out/'device.csv').write_text(info)
    report = dict(schema=1, command=sys.argv, binary_sha256=sha(exe), test_sha256=sha(__file__),
                  runtime_sha256={path.name: sha(path) for path in frozen.iterdir() if path.is_file()},
                  device=a.device, device_info=info, timing='complete process wall; no oracle time',
                  exponent={'ed': '48*lcm', 'ed-5': '48*lcm', 'ed-binary': '48*lcm', 'mont0': 'lcm', 'mont3': 'lcm'},
                  source_sha256=sources, source_archive_sha256=sha(out/'sources.zip'),
                  tools_archive_sha256=sha(out/'tools.zip'),
                  worktree_commit_at_collection=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
                  source_origin='supplied frozen archive' if a.source_archive else 'current worktree snapshot',
                  checks='every record and repeat stability; first/last independent ladder',
                  samples=[], complete=False)
    stop = threading.Event(); telemetry_errors = []
    def monitor():
        with (out/'telemetry.csv').open('wb') as log:
            while not stop.is_set():
                try:
                    r = subprocess.run(['nvidia-smi', '-i', str(a.device),
                        '--query-gpu=timestamp,index,uuid,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu',
                        '--format=csv,noheader,nounits'], capture_output=True, timeout=10)
                    if r.returncode or not r.stdout: raise RuntimeError('empty/failed GPU telemetry')
                    log.write(r.stdout); log.flush()
                except Exception as error: telemetry_errors.append(str(error))
                stop.wait(0.5)
    thread = threading.Thread(target=monitor, daemon=True); thread.start()
    signatures = {}; family_signatures = {}; oracle = {}
    try:
        for bits in a.bits:
            n = (1 << bits)-1
            for repeat in range(-1, a.repeats):
                for model in (a.models if repeat % 2 == 0 else list(reversed(a.models))):
                    name = f'm{bits}_{model}_r{repeat}'
                    folder = out/name; folder.mkdir()
                    save = folder/'completed.save'
                    edwards = model in ('ed', 'ed-5', 'ed-binary')
                    selected_doubling = 5 if model == 'ed-5' else (a.doubling if model == 'ed' else None)
                    command = [exe, '--curve', 'ed' if edwards else 'mont', '--target', 'gpu',
                        '-d', a.device, '-sigma', a.sigma, '-gpucurves', a.curves, '--ckpt', 0,
                        '--exp-cache', out/'exp_cache', '--tmp-dir', folder/'scratch', '-save', save, '-v']
                    if not edwards: command += ['--gpu-param', int(model[-1])]
                    elif model != 'ed-binary' and a.naf_w is not None: command += ['--naf-w', a.naf_w]
                    command += [a.b1, 0]
                    variant_env = dict(env)
                    if edwards:
                        variant_env['ECM_EDWARDS_STAGE1_ALGO'] = 'binary' if model == 'ed-binary' else 'naf'
                        if selected_doubling is not None: variant_env['ECM_EDWARDS_DBL_ALGO'] = str(selected_doubling)
                    if model == 'ed-binary': variant_env.pop('ECM_EDWARDS_DBL_ALGO', None)
                    print('START', name, flush=True)
                    t0 = time.perf_counter()
                    r = subprocess.run(list(map(str, command)), input=f'(2^{bits}-1)\n'.encode(),
                        capture_output=True, cwd=folder, env=variant_env, timeout=3600)
                    wall = time.perf_counter()-t0
                    raw = r.stdout+r.stderr; log = folder/'run.log'; log.write_bytes(raw)
                    assert r.returncode == 0, (name, r.returncode, raw[-2000:])
                    rows = validate(save, n, a.b1, a.curves, a.sigma, model)
                    key = (bits, model)
                    if key not in signatures:
                        signatures[key] = rows
                        family = (bits, 'ed' if edwards else model)
                        if family in family_signatures:
                            assert rows == family_signatures[family], ('binary/NAF batch mismatch', name)
                            oracle[str(key)] = 'entire batch matches independently sampled binary/NAF family'
                        else:
                            for index in sorted({0, len(rows)-1}):
                                expected = reference(n, rows[index][0], a.b1, model)
                                assert rows[index][1:] == expected, (name, index, rows[index], expected)
                            family_signatures[family] = rows
                            oracle[str(key)] = 'first and last matched independent ladder'
                    else: assert rows == signatures[key], ('repeat point mismatch', name)
                    text = raw.decode('utf-8', 'replace')
                    matches = re.findall(r'CGBN<(\d+),\s*(\d+)> kernel, N is (\d+) bits \((\d+) blocks x (\d+) threads\)', text)
                    geometry = [dict(tpi=int(t), container_bits=int(w), bits=int(nbits), blocks=int(blocks), tpb=int(tpb), basis='worker log')
                                for t, w, nbits, blocks, tpb in matches]
                    arithmetic = re.findall(r'GPU Edwards arithmetic: algorithm=(\w+) api_seconds=([\d.]+)', text)
                    selection = re.findall(r'GPU Edwards Stage1:.*?window=(\d+) doubling=(\d+)', text)
                    naf_metrics = re.findall(r'GPU Edwards NAF: columns=(\d+) additions=(\d+) container_bits=(\d+) tpi=(\d+) dictionary_bytes=(\d+) dictionary_gpu_ms=([\d.]+) loop_gpu_ms=([\d.]+) launches=(\d+)', text)
                    if edwards:
                        assert ('GPU Edwards NAF:' in text) == (model != 'ed-binary'), ('unexpected Edwards execution mode', name)
                        reported = re.findall(r'GPU Edwards geometry: container_bits=(\d+) tpi=(\d+) tpb=(\d+) blocks=(\d+)', text)
                        # Old frozen binaries used seven fixed tiers and no geometry line.
                        # New split builds report their actually compiled dispatch result.
                        if reported:
                            assert len(reported) == 1
                            w,t,tpb,blocks = map(int,reported[0]);tier=(w,t)
                            assert bits < w and tpb == 128 and blocks == (a.curves+128//t-1)//(128//t)
                        else:
                            tier = next((w, t) for w, t in [(256,4),(512,4),(1024,8),(2048,16),(4096,16),(8192,32),(17408,32)] if bits < w)
                        geometry = [dict(container_bits=tier[0], tpi=tier[1], tpb=128, blocks=(a.curves+(128//tier[1])-1)//(128//tier[1]), basis='worker geometry' if reported else 'checked dispatcher source')]
                        # An old baseline has no API timer; never manufacture it.
                        if arithmetic: assert len(arithmetic) == 1 and arithmetic[0][0] == ('binary' if model == 'ed-binary' else 'naf')
                        if model != 'ed-binary':
                            assert len(naf_metrics) == 1 and len(selection) == 1
                            if selected_doubling is not None: assert int(selection[0][1]) == selected_doubling
                            assert (int(naf_metrics[0][2]), int(naf_metrics[0][3])) == tier
                            geometry[0]['basis'] = 'worker NAF metrics'
                    assert len(geometry) == 1, ('missing/ambiguous GPU geometry', name)
                    report['samples'].append(dict(bits=bits, b1=a.b1, curves=a.curves, sigma=a.sigma,
                        model=model, naf_w=a.naf_w if edwards and model != 'ed-binary' else None, doubling=selected_doubling,
                        arithmetic_api_seconds=float(arithmetic[0][1]) if arithmetic else None,
                        actual_edwards_selection=[dict(window=int(w), doubling=int(d)) for w, d in selection],
                        naf_metrics=[dict(columns=int(c), additions=int(ad), container_bits=int(w), tpi=int(t),
                            dictionary_bytes=int(db), dictionary_gpu_ms=float(dm), loop_gpu_ms=float(lm), launches=int(l))
                            for c, ad, w, t, db, dm, lm, l in naf_metrics],
                        mont_reported_gpu_ms=[float(x) for x in re.findall(r'GPU stage1 returned: .*?gputime=([\d.eE+-]+) ms', text)],
                        repeat=repeat, warmup=repeat < 0,
                        process_seconds=wall, seconds_per_curve=wall/a.curves, geometry=geometry,
                        command=list(map(str, command)), log_sha256=sha(log), save_sha256=sha(save)))
                    (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
                    print('PASS', name, round(wall, 6), 'seconds', flush=True)
        report['aggregates'] = []
        for bits in a.bits:
            for model in a.models:
                values = [s['process_seconds'] for s in report['samples'] if s['bits'] == bits and s['model'] == model and not s['warmup']]
                report['aggregates'].append(dict(bits=bits, model=model, count=len(values),
                    median_seconds=statistics.median(values), min_seconds=min(values), max_seconds=max(values),
                    median_seconds_per_curve=statistics.median(values)/a.curves))
                api = [s['arithmetic_api_seconds'] for s in report['samples'] if s['bits'] == bits and s['model'] == model and not s['warmup'] and s['arithmetic_api_seconds'] is not None]
                if api: report['aggregates'][-1]['median_arithmetic_api_seconds'] = statistics.median(api)
        report['oracle'] = oracle; report['complete'] = True
    finally:
        stop.set(); thread.join(timeout=10)
        report['telemetry_errors'] = telemetry_errors
        if telemetry_errors or thread.is_alive(): report['complete'] = False
        (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    assert report['complete'], ('incomplete benchmark', telemetry_errors)
    print(out/'summary.json', flush=True)


if __name__ == '__main__': main()
