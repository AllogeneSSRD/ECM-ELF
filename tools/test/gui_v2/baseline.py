"""Freeze and measure the unchanged DX11 GUI using isolated, non-autostart fixtures."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from win32_probe import window, capture, close, process_sample, user

ROOT = Path(__file__).resolve().parents[3]
PROTECTED = ['src/ecm/application/gui', 'third_party/imgui', 'tools/test/gui',
             'tools/build/build_gui.bat', 'tools/build/dev/build_gui.ps1', 'config/ecm_options.json']


def hashes():
    paths = []
    for name in PROTECTED:
        path = ROOT / name
        paths.extend(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else paths.append(path)
    paths.extend((ROOT / 'tools/test').glob('test_gui*'))
    return {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths) if p.is_file()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exe', type=Path, default=ROOT / 'build_gui_local/ecm_gui.exe')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--check', type=Path)
    args = parser.parse_args()
    if args.check:
        expected = json.loads(args.check.read_text(encoding='utf-8'))['protected']
        actual = hashes()
        assert actual == expected, [p for p in set(actual) | set(expected) if actual.get(p) != expected.get(p)]
        print(f'PASS: {len(actual)} protected files unchanged')
        return
    out = args.out.resolve()
    assert not out.exists(), 'Use a fresh evidence directory'
    out.mkdir(parents=True)
    before = hashes()
    original = args.exe.resolve()
    shutil.copy2(original, out / 'ecm_gui.exe')
    shutil.copytree(ROOT / 'src/ecm/application/gui/localization', out / 'localization')
    shutil.copy2(original.parent / 'ecm_gui_fake_worker.exe', out / 'fake_worker.exe')
    exe = out / 'ecm_gui.exe'
    config = out / 'workflow.toml'
    config.write_text('[gui]\nNumWorkers=1\nrefresh_hz=10\ngpu_poll_ms=500\n'
                      'exe="fake_worker.exe"\nstage2_exe="fake_worker.exe"\n'
                      '[shared]\nworktodo="worktodo.txt"\n[workers.1]\nautostart=0\n', encoding='utf-8')
    (out / 'worktodo.txt').write_text('[Worker #1]\nECMSTAGE1=1,2,12011,-1,1000,10000,4\n', encoding='utf-8')
    (out / 'fixture.log').write_bytes(subprocess.run([str(out / 'fake_worker.exe'), '--scenario', 'ok', '--delay-ms', '0'], capture_output=True, check=True).stdout)
    version = subprocess.run([str(exe), '--version'], capture_output=True, check=True).stdout.decode().strip()
    tests = {}
    for name, command in [
        ('selftest', [exe, '--selftest', '--config', config]),
        ('logs', [original.parent / 'ecm_gui_log_parse_test.exe']),
        ('generator', [original.parent / 'ecm_gui_gen_test.exe']),
        ('results', [original.parent / 'ecm_gui_results_test.exe', out / 'results']),
        ('controller', [original.parent / 'ecm_gui_controller_test.exe', out / 'controller']),
    ]:
        run = subprocess.run(list(map(str, command)), cwd=out, capture_output=True, timeout=90)
        (out / f'{name}.log').write_bytes(run.stdout + run.stderr)
        assert run.returncode == 0, (name, run.stdout[-1000:], run.stderr[-1000:])
        tests[name] = {'returncode': run.returncode, 'log': f'{name}.log'}
    samples = []
    for index in range(5):
        startup = subprocess.STARTUPINFO()
        startup.dwFlags = subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 0
        language = 'chineseSimplified' if index == 1 else 'english'
        process = subprocess.Popen([str(exe), '--config', str(config), '--language', language, '--trace'],
                                   cwd=out, startupinfo=startup, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        hwnd = window(process.pid, 'ecm_gui')
        try:
            time.sleep(1)
            begin = process_sample(process)
            start = time.monotonic()
            time.sleep(2)
            end = process_sample(process)
            seconds = time.monotonic() - start
            samples.append({'language': language, 'duration_seconds': seconds,
                            'cpu_seconds': end['cpu_seconds'] - begin['cpu_seconds'],
                            'one_core_cpu_percent': 100 * (end['cpu_seconds'] - begin['cpu_seconds']) / seconds,
                            'working_set_bytes': end['working_set_bytes'], 'private_bytes': end['private_bytes'],
                            'dpi': user.GetDpiForWindow(hwnd)})
            if index < 2:
                user.ShowWindow(hwnd, 4)
                time.sleep(.3)
                capture(hwnd, out / f'{language}.png')
                user.ShowWindow(hwnd, 0)
        finally:
            close(process, hwnd)
    assert before == hashes()
    result = {'phase': 'P0', 'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
              'legacy_version': version, 'legacy_sha256': hashlib.sha256(exe.read_bytes()).hexdigest(),
              'protected': before, 'tests': tests, 'idle_samples': samples,
              'measurement': 'Win32 GetProcessTimes/GetProcessMemoryInfo; hidden normal HWND, 1s warmup, 2s interval; no real workers; 10Hz; NVML 500ms',
              'limitations': ['Not an input-latency or GPU/power benchmark', 'P3-P6 functionality is not qualified']}
    (out / 'baseline.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'passed': True, 'protected_files': len(before), 'tests': list(tests), 'idle_runs': len(samples), 'evidence': str(out)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
