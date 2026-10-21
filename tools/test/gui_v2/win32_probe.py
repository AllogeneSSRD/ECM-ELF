"""Native window and process measurements for isolated GUI acceptance runs."""
import ctypes as c
from ctypes import wintypes as w
from pathlib import Path
import time

user = c.WinDLL('user32', use_last_error=True)
gdi = c.WinDLL('gdi32', use_last_error=True)
kernel = c.WinDLL('kernel32', use_last_error=True)
psapi = c.WinDLL('psapi', use_last_error=True)
for name, args, result in [
    ('GetWindowThreadProcessId', [w.HWND, c.POINTER(w.DWORD)], w.DWORD),
    ('GetClassNameW', [w.HWND, w.LPWSTR, c.c_int], c.c_int),
    ('GetWindowRect', [w.HWND, c.POINTER(w.RECT)], w.BOOL),
    ('GetClientRect', [w.HWND, c.POINTER(w.RECT)], w.BOOL),
    ('GetDpiForWindow', [w.HWND], w.UINT),
    ('PostMessageW', [w.HWND, w.UINT, w.WPARAM, w.LPARAM], w.BOOL),
    ('ShowWindow', [w.HWND, c.c_int], w.BOOL),
    ('SetWindowPos', [w.HWND, w.HWND, c.c_int, c.c_int, c.c_int, c.c_int, w.UINT], w.BOOL),
    ('GetDC', [w.HWND], w.HDC), ('ReleaseDC', [w.HWND, w.HDC], c.c_int),
    ('PrintWindow', [w.HWND, w.HDC, w.UINT], w.BOOL),
]:
    fn = getattr(user, name)
    fn.argtypes, fn.restype = args, result
callback_type = c.WINFUNCTYPE(w.BOOL, w.HWND, w.LPARAM)
user.EnumWindows.argtypes = [callback_type, w.LPARAM]
gdi.CreateCompatibleDC.argtypes, gdi.CreateCompatibleDC.restype = [w.HDC], w.HDC
gdi.CreateCompatibleBitmap.argtypes, gdi.CreateCompatibleBitmap.restype = [w.HDC, c.c_int, c.c_int], w.HBITMAP
gdi.SelectObject.argtypes, gdi.SelectObject.restype = [w.HDC, w.HGDIOBJ], w.HGDIOBJ
gdi.DeleteObject.argtypes = [w.HGDIOBJ]
gdi.DeleteDC.argtypes = [w.HDC]


def window(pid, class_name, timeout=15):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        found = []
        @callback_type
        def visit(hwnd, _):
            owner = w.DWORD()
            user.GetWindowThreadProcessId(hwnd, c.byref(owner))
            name = c.create_unicode_buffer(256)
            user.GetClassNameW(hwnd, name, len(name))
            if owner.value == pid and name.value == class_name:
                found.append(hwnd)
            return True
        user.EnumWindows(visit, 0)
        if found:
            return found[0]
        time.sleep(.05)
    raise RuntimeError(f'No {class_name} window for process {pid}')


class Memory(c.Structure):
    _fields_ = [('cb', w.DWORD), ('PageFaultCount', w.DWORD)] + [(name, c.c_size_t) for name in
        ['PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage',
         'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage', 'PrivateUsage']]


def process_sample(process):
    times = [w.FILETIME() for _ in range(4)]
    kernel.GetProcessTimes.argtypes = [w.HANDLE] + [c.POINTER(w.FILETIME)] * 4
    if not kernel.GetProcessTimes(process._handle, *(c.byref(t) for t in times)):
        raise c.WinError(c.get_last_error())
    ticks = lambda t: (t.dwHighDateTime << 32) | t.dwLowDateTime
    mem = Memory()
    mem.cb = c.sizeof(mem)
    psapi.GetProcessMemoryInfo.argtypes = [w.HANDLE, c.POINTER(Memory), w.DWORD]
    if not psapi.GetProcessMemoryInfo(process._handle, c.byref(mem), mem.cb):
        raise c.WinError(c.get_last_error())
    return {'cpu_seconds': (ticks(times[2]) + ticks(times[3])) / 1e7,
            'working_set_bytes': mem.WorkingSetSize, 'private_bytes': mem.PrivateUsage}


def capture(hwnd, path):
    from PIL import Image
    rect = w.RECT()
    user.GetWindowRect(hwnd, c.byref(rect))
    width, height = rect.right - rect.left, rect.bottom - rect.top
    dc = user.GetDC(hwnd)
    compatible = gdi.CreateCompatibleDC(dc)
    bitmap = gdi.CreateCompatibleBitmap(dc, width, height)
    old = gdi.SelectObject(compatible, bitmap)
    try:
        if not user.PrintWindow(hwnd, compatible, 2):
            raise RuntimeError('PrintWindow failed')
        class Header(c.Structure):
            _fields_ = [('size', w.DWORD), ('width', w.LONG), ('height', w.LONG), ('planes', w.WORD),
                        ('bits', w.WORD), ('compression', w.DWORD), ('image_size', w.DWORD),
                        ('xppm', w.LONG), ('yppm', w.LONG), ('used', w.DWORD), ('important', w.DWORD)]
        header = Header(c.sizeof(Header), width, -height, 1, 32)
        pixels = c.create_string_buffer(width * height * 4)
        gdi.SelectObject(compatible, old)
        gdi.GetDIBits.argtypes = [w.HDC, w.HBITMAP, w.UINT, w.UINT, c.c_void_p, c.c_void_p, w.UINT]
        if gdi.GetDIBits(compatible, bitmap, 0, height, pixels, c.byref(header), 0) != height:
            raise RuntimeError('GetDIBits failed')
        Image.frombytes('RGB', (width, height), pixels.raw, 'raw', 'BGRX').save(Path(path))
    finally:
        gdi.SelectObject(compatible, old)
        gdi.DeleteObject(bitmap)
        gdi.DeleteDC(compatible)
        user.ReleaseDC(hwnd, dc)


def close(process, hwnd):
    user.PostMessageW(hwnd, 0x10, 0, 0)
    try:
        code = process.wait(timeout=15)
    except Exception:
        process.kill()
        process.wait()
        raise
    if code:
        raise RuntimeError(f'GUI exited with {code}')
