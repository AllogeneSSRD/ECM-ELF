"""Accounting for an isolated Windows benchmark process tree.

PeakJobMemoryUsed is the simultaneous job commit high-water mark, not a sum of
process peaks, GPU allocations, working sets, or physical RAM consumption.
"""
import ctypes as ct
from ctypes import wintypes as wt
import time


class BasicAccounting(ct.Structure):
    _fields_ = [(n, ct.c_int64) for n in ('user', 'kernel', 'period_user', 'period_kernel')] + [
        (n, wt.DWORD) for n in ('page_faults', 'processes', 'active', 'terminated')]


class BasicLimits(ct.Structure):
    _fields_ = [('process_time', ct.c_int64), ('job_time', ct.c_int64), ('flags', wt.DWORD),
                ('min_working_set', ct.c_size_t), ('max_working_set', ct.c_size_t),
                ('active_process_limit', wt.DWORD), ('affinity', ct.c_size_t),
                ('priority', wt.DWORD), ('scheduling', wt.DWORD)]


class ExtendedLimits(ct.Structure):
    _fields_ = [('basic', BasicLimits), ('io', ct.c_uint64 * 6),
                ('process_memory_limit', ct.c_size_t), ('job_memory_limit', ct.c_size_t),
                ('peak_process_memory', ct.c_size_t), ('peak_job_memory', ct.c_size_t)]


class WindowsJob:
    def __init__(self):
        self.api = ct.WinDLL('kernel32', use_last_error=True)
        self.api.CreateJobObjectW.argtypes = [ct.c_void_p, wt.LPCWSTR]
        self.api.CreateJobObjectW.restype = wt.HANDLE
        self.api.AssignProcessToJobObject.argtypes = [wt.HANDLE, wt.HANDLE]
        self.api.QueryInformationJobObject.argtypes = [wt.HANDLE, ct.c_int, ct.c_void_p, wt.DWORD, ct.c_void_p]
        self.api.TerminateJobObject.argtypes = [wt.HANDLE, wt.UINT]
        self.api.CloseHandle.argtypes = [wt.HANDLE]
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ct.WinError(ct.get_last_error())

    def assign(self, process):
        # Popen's Windows process handle avoids a second OpenProcess race.
        # Attach immediately; subsequent descendants inherit the job. Startup
        # before this assignment is outside the accounting boundary.
        if not self.api.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise ct.WinError(ct.get_last_error())

    def accounting(self):
        basic, extended = BasicAccounting(), ExtendedLimits()
        for kind, value in ((1, basic), (9, extended)):
            if not self.api.QueryInformationJobObject(self.handle, kind, ct.byref(value), ct.sizeof(value), None):
                raise ct.WinError(ct.get_last_error())
        return {'cpu_seconds': (basic.user + basic.kernel) * 1e-7,
                'user_seconds': basic.user * 1e-7, 'kernel_seconds': basic.kernel * 1e-7,
                'processes': basic.processes, 'active_processes': basic.active,
                'peak_job_commit_bytes': extended.peak_job_memory}

    def terminate(self):
        if not self.api.TerminateJobObject(self.handle, 129):
            raise ct.WinError(ct.get_last_error())

    def drain(self, timeout=5):
        # A signalled root handle can precede the job's final exit accounting.
        # Wait for the whole scoped tree, rather than rejecting that race.
        end = time.perf_counter() + timeout
        while True:
            value = self.accounting()
            if not value['active_processes']:
                return value
            if time.perf_counter() >= end:
                raise TimeoutError('Benchmark job descendants did not exit')
            time.sleep(.02)

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None
