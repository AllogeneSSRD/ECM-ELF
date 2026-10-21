"""Exercise the actual v2 Vulkan host, native lifecycle, and explicit failures."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from win32_probe import window, capture, close, user, process_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--layers', type=Path)
    args = parser.parse_args()
    exe, out = args.exe.resolve(), args.out.resolve()
    assert not out.exists(), 'Use a fresh evidence directory'
    out.mkdir(parents=True)
    env = os.environ.copy()
    if args.layers:
        env['VK_LAYER_PATH'] = str(args.layers.resolve())
    startup = subprocess.STARTUPINFO()
    startup.dwFlags = subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 0
    cases = []
    for name, switches, expected in [
        ('normal', ['--frames', '12'], 0),
        ('no-loader', ['--no-loader'], 1),
        ('device-lost', ['--inject-failure', 'device-lost'], 1),
        ('surface-lost', ['--inject-failure', 'surface-lost'], 1),
        ('invalid-option', ['--unknown'], 1),
    ]:
        command = [str(exe), '--diagnostics', str(out / (name + '.log')), '--ui-state', str(out / (name + '.toml')), '--no-nvml', *switches]
        run = subprocess.run(command, startupinfo=startup, env=env, cwd=out, capture_output=True, timeout=30)
        (out / (name + '.stdout')).write_bytes(run.stdout + run.stderr)
        assert run.returncode == expected, (name, run.returncode)
        if name != 'invalid-option':
            text = (out / (name + '.log')).read_text()
            assert 'validation_summary: errors=0' in text, text
            assert ('error:' in text) == (expected != 0), text
            if name in ('device-lost', 'surface-lost'):
                assert 'injected renderer failure' in text, text
        cases.append({'name': name, 'exit': run.returncode, 'injected': name in ('device-lost', 'surface-lost', 'no-loader')})
    log = out / 'lifecycle.log'
    command = [str(exe), '--diagnostics', str(log), '--ui-state', str(out / 'lifecycle.toml'), '--no-nvml', '--validation']
    process = subprocess.Popen(command, startupinfo=startup, env=env, cwd=out,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    hwnd = window(process.pid, 'ecm_gui_v2')
    try:
        end = time.monotonic() + 15
        while time.monotonic() < end:
            if process.poll() is not None:
                raise RuntimeError(log.read_text())
            if log.exists() and 'frame: 1' in log.read_text():
                break
            time.sleep(.05)
        else:
            raise RuntimeError('Vulkan first frame was not submitted')
        for size in [(800, 600), (1280, 800), (900, 650), (1100, 760)]:
            assert user.SetWindowPos(hwnd, None, 0, 0, *size, 0x0016)
            time.sleep(.3)
        user.ShowWindow(hwnd, 6)
        time.sleep(.4)
        user.ShowWindow(hwnd, 4)
        time.sleep(.5)
        begin = process_sample(process)
        time.sleep(2)
        end_sample = process_sample(process)
        capture(hwnd, out / 'host.png')
        measurement = {'duration_seconds': 2, 'cpu_seconds': end_sample['cpu_seconds'] - begin['cpu_seconds'],
                       'working_set_bytes': end_sample['working_set_bytes'], 'private_bytes': end_sample['private_bytes']}
    finally:
        if process.poll() is None:
            close(process, hwnd)
    text = log.read_text()
    assert process.returncode == 0, text
    assert 'validation=1' in text and 'validation_summary: errors=0 warnings=0' in text, text
    assert len(re.findall(r'swapchain: rebuild=', text)) >= 4, text
    assert 'worker_processes=0' in text, text
    cases.append({'name': 'validation-lifecycle', 'exit': process.returncode, 'resize_minimize_restore': True})
    result = {'passed': True, 'exe': str(exe), 'sha256': hashlib.sha256(exe.read_bytes()).hexdigest(),
              'cases': cases, 'validation': True, 'measurement': measurement,
              'limits': ['P1 single-window milestone; full multi-viewport qualification is P3',
                         'Lost-device/surface paths are injected, not an actual hardware failure']}
    (out / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
