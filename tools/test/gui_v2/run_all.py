"""Sequential DX11 suite for the existing GUI v2 binary; no Vulkan tests."""
import argparse
import ctypes as c
from ctypes import wintypes as w
from datetime import datetime
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


class ProcessTree:
    """Close the Windows job after each test, including on failure or timeout."""

    def __init__(self):
        class BasicLimits(c.Structure):
            _fields_ = [('process_time', c.c_longlong), ('job_time', c.c_longlong),
                        ('flags', w.DWORD), ('min_ws', c.c_size_t), ('max_ws', c.c_size_t),
                        ('active', w.DWORD), ('affinity', c.c_size_t),
                        ('priority', w.DWORD), ('scheduling', w.DWORD)]

        class ExtendedLimits(c.Structure):
            _fields_ = [('basic', BasicLimits), ('io', c.c_ulonglong * 6),
                        ('process_memory', c.c_size_t), ('job_memory', c.c_size_t),
                        ('peak_process_memory', c.c_size_t), ('peak_job_memory', c.c_size_t)]

        self.kernel = c.WinDLL('kernel32', use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [c.c_void_p, w.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = w.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [w.HANDLE, c.c_int, c.c_void_p, w.DWORD]
        self.kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        self.kernel.CloseHandle.argtypes = [w.HANDLE]
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise c.WinError(c.get_last_error())
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.kernel.SetInformationJobObject(self.handle, 9, c.byref(limits), c.sizeof(limits)):
            error = c.WinError(c.get_last_error())
            self.close()
            raise error

    def assign(self, process):
        if not self.kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise c.WinError(c.get_last_error())

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def plan(exe, out):
    build = exe.parent
    worker = build / 'ecm_gui_v2_fake_worker.exe'
    probe = build / 'ecm_gui_v2_owner_probe.exe'
    tests = []

    def native(name, binary):
        tests.append((name, [build / binary, out / name / 'fixture'], [build / binary]))

    def python(name, script, *, uses_worker=False, uses_layers=False, extra=()):
        command = [sys.executable, '-u', HERE / script, '--exe', exe, '--out', out / name]
        required = [exe]
        if uses_worker:
            command += ['--worker', worker]
            required.append(worker)
        if uses_layers:
            # Shared scripts require this argument even on DX11. No layer is loaded.
            command += ['--renderer', 'dx11', '--layers', out / 'unused-layers']
        tests.append((name, command + list(extra), required))

    native('state', 'ecm_gui_v2_state_test.exe')
    native('app', 'ecm_gui_v2_app_test.exe')
    tests.append(('ownership', [sys.executable, '-u', HERE / 'test_ownership.py',
                               '--probe', probe, '--out', out / 'ownership'], [probe]))
    python('startup-dispatch', 'test_startup_dispatch.py', uses_worker=True)
    python('dx11-host', 'test_dx11_host.py')
    python('pages', 'test_pages.py', uses_layers=True)
    python('interaction', 'test_interaction.py', uses_worker=True, uses_layers=True)
    python('usability', 'test_usability.py', uses_layers=True)
    python('shell', 'test_shell.py', uses_worker=True, uses_layers=True)
    for language in ('zh', 'en'):
        for theme in ('dark', 'light'):
            python(f'presentation-{language}-{theme}', 'test_presentation.py', uses_worker=True,
                   extra=['--language', language] + (['--light'] if theme == 'light' else []))
    python('pet-runtime', 'test_pet_runtime.py', uses_layers=True)
    python('pet-skins', 'test_pet_skins.py')
    for mode in ('single', 'multi'):
        python(f'chart-{mode}-viewport', 'test_chart_layout.py', uses_layers=True,
               extra=[f'--{mode}-viewport'])
    python('minimized', 'test_minimized.py')
    return tests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, default=ROOT / 'build_gui_v2_local/ecm_gui_v2.exe')
    parser.add_argument('--out', type=Path, help='New evidence directory (must not exist)')
    parser.add_argument('--only', default='*', help='Test name wildcard, e.g. presentation-*')
    parser.add_argument('--list', action='store_true', help='List selected tests without running')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout per test in seconds')
    parser.add_argument('--no-pause', action='store_true', help='Do not pause the BAT on completion')
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    exe = args.exe.resolve()
    out = (args.out or ROOT / 'data/experiments/gui_v2' /
           ('dx11_suite_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8])).resolve()
    selected = [test for test in plan(exe, out) if fnmatch.fnmatchcase(test[0], args.only)]
    if not selected:
        parser.error('No tests match --only')
    if args.list:
        for name, _, _ in selected:
            print(name)
        return 0
    if os.name != 'nt' or sys.version_info < (3, 11):
        parser.error('Windows and Python 3.11+ are required')
    missing = sorted({str(path) for _, _, paths in selected for path in paths if not path.is_file()})
    if not exe.is_file():
        missing.append(str(exe))
    if missing:
        print('ERROR: Missing executable(s):\n  ' + '\n  '.join(missing))
        print('Build helpers with:\n  powershell -NoProfile -ExecutionPolicy Bypass -File '
              'tools\\build\\test\\build_gui_v2.ps1 -BuildDir "' + str(exe.parent) + '" -TestHelpers')
        return 2
    if out.exists():
        parser.error('--out must be a new directory; previous evidence is preserved')
    out.mkdir(parents=True)
    # Several existing scripts omit --config. Isolate their adjacent auto-discovery
    # from the user's real workflow by running byte-identical executable copies.
    binaries = out / 'bin'
    binaries.mkdir()
    required = {exe} | {path for _, _, paths in selected for path in paths}
    for path in required | set(exe.parent.glob('*.dll')):
        shutil.copy2(path, binaries / path.name)
    if (exe.parent / 'assets').is_dir():
        shutil.copytree(exe.parent / 'assets', binaries / 'assets')
    selected = [test for test in plan(binaries / exe.name, out)
                if fnmatch.fnmatchcase(test[0], args.only)]
    report = {'renderer': 'dx11', 'source_exe': str(exe), 'tested_exe': str(binaries / exe.name),
              'exe_sha256': hashlib.sha256((binaries / exe.name).read_bytes()).hexdigest(),
              'python': sys.version, 'timeout_seconds': args.timeout, 'tests': []}
    print(f'DX11 suite: {len(selected)} tests\nExecutable: {exe}\nEvidence: {out}', flush=True)
    print('Tests control real windows. Leave the desktop idle until completion.', flush=True)
    env = os.environ.copy()
    env['PYTHONUTF8'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    env.pop('PYTHONOPTIMIZE', None)  # Existing scripts use assertions as test gates.
    interrupted = False
    for index, (name, command, _) in enumerate(selected, 1):
        log = out / (name + '.log')
        command = list(map(str, command))
        if name in ('state', 'app'):
            (out / name).mkdir()
        print(f'[{index}/{len(selected)}] {name} ...', flush=True)
        start = time.monotonic()
        status, code, error = 'FAIL', None, None
        tree, process = None, None
        try:
            tree = ProcessTree()
            with log.open('wb') as output:
                process = subprocess.Popen(command, cwd=out, env=env, stdout=output, stderr=subprocess.STDOUT)
                tree.assign(process)
                code = process.wait(timeout=args.timeout)
                status = 'PASS' if code == 0 else 'FAIL'
        except subprocess.TimeoutExpired:
            status, error = 'TIMEOUT', f'Exceeded {args.timeout} seconds'
        except KeyboardInterrupt:
            status, error, interrupted = 'INTERRUPTED', 'Stopped by user', True
        except Exception as exc:
            error = str(exc)
        finally:
            if tree:
                tree.close()
            if process and process.poll() is None:
                process.kill()
                process.wait()
        row = dict(name=name, status=status, returncode=code, seconds=round(time.monotonic() - start, 2),
                   log=str(log), command=command, error=error)
        report['tests'].append(row)
        report['passed'] = len(report['tests']) == len(selected) and all(
            item['status'] == 'PASS' for item in report['tests'])
        (out / 'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'  {status} ({row["seconds"]}s)' + (f': {error}' if error else ''), flush=True)
        if status != 'PASS':
            print(f'  Log: {log}', flush=True)
        if interrupted:
            break
    passed = sum(row['status'] == 'PASS' for row in report['tests'])
    print(f'\nPassed: {passed}/{len(selected)}; failed: {len(report["tests"]) - passed}; '
          f'not run: {len(selected) - len(report["tests"])}\nSummary: {out / "summary.json"}', flush=True)
    for row in report['tests']:
        if row['status'] != 'PASS':
            print(f'  {row["status"]}: {row["name"]} -> {row["log"]}', flush=True)
    return 130 if interrupted else (0 if report['passed'] else 1)


if __name__ == '__main__':
    raise SystemExit(main())
