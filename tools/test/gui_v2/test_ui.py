"""Exercise theme/font changes in the actual Vulkan host and ImGui controls.

WM_APP+32 is accepted only with --audit-ui: bits 0..1=scale, 2=zh, 3=light;
lParam is an acknowledgement sequence. Audit tracks explicit VkDeviceMemory,
not the driver's total residency, and is disabled during ordinary operation.
WM_APP+33 injects ImGui mouse input after the native backend: wParam 0=move,
1=down, 2=up, 3=disable; lParam is packed client pixels. It deliberately avoids
desktop cursor/focus dependence. Text and keys still use Win32 messages.
"""
import argparse
import ctypes as c
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
    parser.add_argument('--layers', type=Path, required=True)
    args = parser.parse_args()
    # Capture and input coordinates must use the same physical pixels as the host.
    user.SetThreadDpiAwarenessContext.argtypes = [c.c_void_p]
    user.SetThreadDpiAwarenessContext.restype = c.c_void_p
    assert user.SetThreadDpiAwarenessContext(c.c_void_p(-4))
    exe, out = args.exe.resolve(), args.out.resolve()
    assert not out.exists(), 'Use a fresh evidence directory'
    out.mkdir(parents=True)
    env = os.environ.copy()
    env['VK_LAYER_PATH'] = str(args.layers.resolve())
    env.pop('VULKAN_SDK', None)
    startup = subprocess.STARTUPINFO()
    startup.dwFlags = subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 0
    log = out / 'matrix.log'
    process = subprocess.Popen([str(exe), '--diagnostics', str(log), '--ui-state', str(out / 'matrix.toml'), '--no-nvml', '--renderer','vulkan','--validation', '--audit-ui', '--ui-scale', '1'],
                               startupinfo=startup, env=env, cwd=out, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    hwnd = window(process.pid, 'ecm_gui_v2')
    sequence = 0
    def state(code):
        nonlocal sequence
        sequence += 1
        assert user.PostMessageW(hwnd, 0x8020, code, sequence)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            text = log.read_text(encoding='utf-8') if log.exists() else ''
            if process.poll() is not None:
                raise RuntimeError(f'Host exited {process.returncode}\n{text}')
            match = re.search(rf'ui_sample: seq={sequence} (.+)\nui_controls: seq={sequence} (.+)', text)
            if match:
                fields = dict(re.findall(r'(\w+)=(-?[\d.]+)', ' '.join(match.groups())))
                return {key: float(value) for key, value in fields.items()}
            time.sleep(.025)
        raise RuntimeError('No UI state acknowledgement\n' + text)
    samples = []
    def click(x, y):
        point = int(x) | (int(y) << 16)
        user.PostMessageW(hwnd, 0x8021, 0, point)
        time.sleep(.2)
        user.PostMessageW(hwnd, 0x8021, 1, point)
        time.sleep(.2)
        user.PostMessageW(hwnd, 0x8021, 2, point)
        time.sleep(.2)
    try:
        assert user.SetWindowPos(hwnd, None, 0, 0, 1500, 1150, 0x0016)
        user.ShowWindow(hwnd, 4)
        # Prime the entire transition sequence before comparing identical histories.
        for code in range(16):
            state(code)
        for cycle in range(3):
            for code in range(16):
                sample = state(code)
                scale = [1, 1.25, 1.5, 2][code & 3]
                assert sample['scale'] == scale and sample['light'] == bool(code & 8) and sample['zh'] == bool(code & 4), sample
                for role, size in [('body', 15), ('caption', 13), ('heading', 22), ('mono', 15)]:
                    assert abs(sample[role] - size * scale) < 1.01, (role, sample)
                assert abs(sample['padding'] - 20 * scale) < .01, sample
                assert sample['cjk'] == sample['mono_cjk'] == 1, sample
                assert abs(sample['mono_i'] - sample['mono_w']) < .01, sample
                assert sample['fonts'] == 3 and 0 < sample['textures'] <= 2, sample
                assert sample['allocations'] > 0 and sample['gpu_bytes'] > 0, sample
                sample.update(cycle=cycle, code=code)
                samples.append(sample)
                if cycle == 0:
                    capture(hwnd, out / f"{'light' if code & 8 else 'dark'}-{'zh' if code & 4 else 'en'}-{int(scale * 100)}.png")
        # Compare the same settled state over full repeated reload cycles.
        for code in range(16):
            equivalent = [sample for sample in samples if sample['code'] == code]
            assert len({(s['allocations'], s['gpu_bytes'], s['textures']) for s in equivalent}) == 1, equivalent
        current = state(0)
        user.PostMessageW(hwnd, 0x7, 0, 0)  # native focus event for scoped input test
        click(current['button_x'], current['button_y'])
        clicked = state(0)
        assert clicked['clicks'] == current['clicks'] + 1, clicked
        click(clicked['input_x'], clicked['input_y'])
        user.PostMessageW(hwnd, 0x102, ord('Z'), 1)
        edited = state(0)
        assert edited['path_length'] == clicked['path_length'] + 1, edited
        for key in [0x1b, 0x09]:  # Escape leaves the field, Tab enables keyboard navigation.
            user.PostMessageW(hwnd, 0x100, key, 1)
            time.sleep(.1)
            user.PostMessageW(hwnd, 0x101, key, 0xc0000001)
            time.sleep(.1)
        assert state(0)['nav_visible'] == 1
        # A narrow surface uses stacked labels, child scroll and wrapping states.
        assert user.SetWindowPos(hwnd, None, 0, 0, 760, 900, 0x0016)
        state(4)
        capture(hwnd, out / 'narrow-zh-100.png')
        measurement = process_sample(process)
    finally:
        if process.poll() is None:
            close(process, hwnd)
    text = log.read_text(encoding='utf-8')
    assert process.returncode == 0, text
    assert 'validation_summary: errors=0 warnings=0' in text, text
    assert 'memory_audit_end: bytes=0 allocations=0' in text, text
    # Deliberately remove CJK font discovery; English rescue remains readable.
    rescue = out / 'rescue.log'
    run = subprocess.run([str(exe), '--diagnostics', str(rescue), '--ui-state', str(out / 'rescue.toml'), '--no-nvml', '--language', 'zh', '--simulate-missing-cjk', '--frames', '8', '--renderer','vulkan','--validation'],
                         startupinfo=startup, env=env, cwd=out, timeout=30)
    rescue_text = rescue.read_text(encoding='utf-8')
    assert run.returncode == 0 and 'language=en cjk=0' in rescue_text and 'Chinese font unavailable' in rescue_text, rescue_text
    assert 'validation_summary: errors=0 warnings=0' in rescue_text, rescue_text
    result = {'passed': True, 'exe': str(exe), 'sha256': hashlib.sha256(exe.read_bytes()).hexdigest(),
              'combinations': 16, 'cycles': 3, 'samples': samples, 'imgui_mouse_injection': True, 'win32_text_and_keys': True,
              'keyboard_focus': True, 'cjk_rescue': True, 'explicit_vulkan_allocations_released': True, 'process_sample': measurement,
              'limits': ['Explicit allocation audit excludes driver/swapchain residency; no total VRAM or performance claim',
                         'Mouse injection exercises ImGui controls; it does not qualify desktop pointer/focus or IME behavior',
                         'Scale is applied in one HWND; physical multi-monitor DPI and IME qualification remains P3']}
    (out / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key != 'samples'}))


if __name__ == '__main__':
    main()
