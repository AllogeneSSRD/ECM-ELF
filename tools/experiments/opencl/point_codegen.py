"""Inspect native point program resources and benchmark compiler-hint variants.

Only experiment copies of points.cl are changed. Production sources/defaults
remain intact. Every timed output is checked, including optional CUDA outputs.
"""
import argparse
import ctypes as ct
import hashlib
import json
import math
from pathlib import Path
import re
import statistics as st
import sys
import time

if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)
U64 = ct.c_uint64
HANDLE = ct.c_void_p
SIZE = ct.c_size_t


def check(code, operation):
    if code:
        raise RuntimeError(f'{operation}: OpenCL {code}')


class OpenCL:
    def __init__(self, index):
        self.cl = ct.WinDLL('OpenCL.dll')
        signatures = {
            'clGetPlatformIDs': ([ct.c_uint, ct.c_void_p, ct.c_void_p], ct.c_int),
            'clGetDeviceIDs': ([HANDLE, U64, ct.c_uint, ct.c_void_p, ct.c_void_p], ct.c_int),
            'clGetDeviceInfo': ([HANDLE, ct.c_uint, SIZE, ct.c_void_p, ct.c_void_p], ct.c_int),
            'clCreateContext': ([ct.c_void_p, ct.c_uint, ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p], HANDLE),
            'clCreateCommandQueue': ([HANDLE, HANDLE, U64, ct.c_void_p], HANDLE),
            'clCreateProgramWithSource': ([HANDLE, ct.c_uint, ct.c_void_p, ct.c_void_p, ct.c_void_p], HANDLE),
            'clBuildProgram': ([HANDLE, ct.c_uint, ct.c_void_p, ct.c_char_p, ct.c_void_p, ct.c_void_p], ct.c_int),
            'clGetProgramBuildInfo': ([HANDLE, HANDLE, ct.c_uint, SIZE, ct.c_void_p, ct.c_void_p], ct.c_int),
            'clGetProgramInfo': ([HANDLE, ct.c_uint, SIZE, ct.c_void_p, ct.c_void_p], ct.c_int),
            'clCreateKernel': ([HANDLE, ct.c_char_p, ct.c_void_p], HANDLE),
            'clGetKernelWorkGroupInfo': ([HANDLE, HANDLE, ct.c_uint, SIZE, ct.c_void_p, ct.c_void_p], ct.c_int),
            'clCreateBuffer': ([HANDLE, U64, SIZE, ct.c_void_p, ct.c_void_p], HANDLE),
            'clEnqueueWriteBuffer': ([HANDLE, HANDLE, ct.c_uint, SIZE, SIZE, ct.c_void_p, ct.c_uint, ct.c_void_p, ct.c_void_p], ct.c_int),
            'clEnqueueReadBuffer': ([HANDLE, HANDLE, ct.c_uint, SIZE, SIZE, ct.c_void_p, ct.c_uint, ct.c_void_p, ct.c_void_p], ct.c_int),
            'clSetKernelArg': ([HANDLE, ct.c_uint, SIZE, ct.c_void_p], ct.c_int),
            'clEnqueueNDRangeKernel': ([HANDLE, HANDLE, ct.c_uint, ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_uint, ct.c_void_p, ct.c_void_p], ct.c_int),
            'clWaitForEvents': ([ct.c_uint, ct.c_void_p], ct.c_int),
            'clGetEventInfo': ([HANDLE, ct.c_uint, SIZE, ct.c_void_p, ct.c_void_p], ct.c_int),
            'clFlush': ([HANDLE], ct.c_int),
            'clGetEventProfilingInfo': ([HANDLE, ct.c_uint, SIZE, ct.c_void_p, ct.c_void_p], ct.c_int),
            'clFinish': ([HANDLE], ct.c_int)}
        for name in ('clReleaseContext', 'clReleaseCommandQueue', 'clReleaseProgram', 'clReleaseKernel', 'clReleaseMemObject', 'clReleaseEvent'):
            signatures[name] = ([HANDLE], ct.c_int)
        for name, (args, result) in signatures.items():
            f = getattr(self.cl, name); f.argtypes = args; f.restype = result
        count = ct.c_uint(); check(self.cl.clGetPlatformIDs(0, None, ct.byref(count)), 'platform count')
        platforms = (HANDLE * count.value)(); check(self.cl.clGetPlatformIDs(count, platforms, None), 'platforms')
        devices = []
        for platform in platforms:
            code = self.cl.clGetDeviceIDs(platform, 0xffffffff, 0, None, ct.byref(count))
            if code == -1:
                continue
            check(code, 'device count'); values = (HANDLE * count.value)()
            check(self.cl.clGetDeviceIDs(platform, 0xffffffff, count, values, None), 'devices')
            devices.extend((platform, d) for d in values)
        platform, self.device = devices[index]
        self.name, self.vendor, self.driver = (self.device_text(key) for key in (0x102b, 0x102c, 0x102d))
        properties = (ct.c_ssize_t * 3)(0x1084, platform, 0)
        device = HANDLE(self.device); error = ct.c_int()
        self.context = self.cl.clCreateContext(properties, 1, ct.byref(device), None, None, ct.byref(error)); check(error.value, 'context')
        self.queue = self.cl.clCreateCommandQueue(self.context, self.device, 2, ct.byref(error)); check(error.value, 'queue')
        self.buffers = []

    def device_text(self, key):
        size = SIZE(); check(self.cl.clGetDeviceInfo(self.device, key, 0, None, ct.byref(size)), 'device text size')
        data = ct.create_string_buffer(size.value); check(self.cl.clGetDeviceInfo(self.device, key, size, data, None), 'device text')
        return data.value.decode('utf-8')

    def buffer(self, data_or_words):
        data = data_or_words if not isinstance(data_or_words, int) else None
        words = len(data) if data is not None else data_or_words
        error = ct.c_int(); mem = self.cl.clCreateBuffer(self.context, 1, words * 8, None, ct.byref(error)); check(error.value, 'buffer')
        self.buffers.append(mem)
        if data is not None:
            array = (U64 * words)(*data); check(self.cl.clEnqueueWriteBuffer(self.queue, mem, 1, 0, words * 8, array, 0, None, None), 'upload')
        return mem

    def close(self):
        self.cl.clFinish(self.queue)
        for mem in self.buffers:
            self.cl.clReleaseMemObject(mem)
        self.cl.clReleaseCommandQueue(self.queue); self.cl.clReleaseContext(self.context)


def ladder(x, k, a24, n):
    base = (x, 1)
    def twice(p):
        a = (p[0] + p[1]) ** 2 % n; b = (p[0] - p[1]) ** 2 % n; e = (a - b) % n
        return a * b % n, e * (b + a24 * e) % n
    p, q = base, twice(base)
    for bit in bin(k)[3:]:
        a = (q[0] - q[1]) * (p[0] + p[1]) % n; b = (q[0] + q[1]) * (p[0] - p[1]) % n
        s = (a + b) ** 2 % n, x * (a - b) ** 2 % n
        p, q = (s, twice(q)) if bit == '1' else (twice(p), s)
    return p


class CudaInput(ct.Structure):
    _fields_ = [(s, ct.c_int) for s in ('device', 'nw', 'count', 'block')] + [('ninv', U64)] + [
        (s, ct.POINTER(U64)) for s in ('n', 'x', 'z', 'a24', 'mone', 'scalars', 'values')]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--save', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--source', type=Path, default=Path('src/ecm/opencl/resources/stage2/points.cl'))
    p.add_argument('--variants', default='baseline,inline,unroll2,unroll4,inline2')
    p.add_argument('--device', type=int, default=1)
    p.add_argument('--cuda-device', type=int, default=1)
    p.add_argument('--cuda-dll', type=Path)
    p.add_argument('--count', type=int, default=240)
    p.add_argument('--scalar-bits', type=int, default=11)
    p.add_argument('--group', type=int, default=32)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--wait', choices=('event', 'sleep'), default='event')
    a = p.parse_args()
    if a.out.exists() or not 1 <= a.count <= 4096 or not 1 <= a.scalar_bits <= 63 or a.repeats < 1:
        p.error('Choose fresh output and valid count/width/repeats')
    a.out.mkdir(parents=True)
    line = a.save.read_text(encoding='utf-8-sig')
    n, x = [int(re.search(r'\b' + key + '=0x([0-9a-fA-F]+)', line)[1], 16) for key in ('N', 'X')]
    sigma = int(re.search(r'\bSIGMA=(\d+)', line)[1]); u = (sigma * sigma - 5) % n; v = 4 * sigma % n
    a24 = pow(v - u, 3, n) * (3 * u + v) * pow(16 * pow(u, 3, n) * v, -1, n) % n
    nw = (n.bit_length() + 63) // 64; mask = (1 << 64) - 1
    words = lambda value: [(value >> (i * 64)) & mask for i in range(nw)]
    r = pow(2, 64 * nw, n); ninv = -pow(n & mask, -1, 1 << 64) & mask
    scalars = [(1 << (a.scalar_bits - 1)) + i % (1 << (a.scalar_bits - 1)) for i in range(a.count)]
    reference = [ladder(x, k, a24, n) for k in scalars]
    cl = OpenCL(a.device)
    if 'Advanced Micro Devices' in cl.vendor and nw > 8:
        cl.close(); raise RuntimeError('Wide AMD point qualification guard remains in force')
    report = {'device': cl.name, 'vendor': cl.vendor, 'driver': cl.driver, 'bits': n.bit_length(), 'count': a.count,
              'scalar_bits': a.scalar_bits, 'group': a.group, 'repeats': a.repeats,
              'save_sha256': hashlib.sha256(a.save.read_bytes()).hexdigest(), 'wait': a.wait, 'rows': []}
    constants = cl.buffer(words(n) + words(r) + words(r * r % n) + words(x) + words(1) + words(a24))
    ks = cl.buffer(scalars); ox = cl.buffer(a.count * nw); oz = cl.buffer(a.count * nw)
    baseline = None
    def verify(xx, zz):
        for i, (rx, rz) in enumerate(reference):
            px = sum(int(xx[i * nw + j]) << (64 * j) for j in range(nw))
            pz = sum(int(zz[i * nw + j]) << (64 * j) for j in range(nw))
            if (px * rz - pz * rx) % n or math.gcd(pz, n) != math.gcd(rz, n) or (px == pz == 0 and (rx or rz)):
                raise RuntimeError(f'Projective/GCD oracle mismatch at {i}')
    try:
        original = a.source.read_text(encoding='utf-8')
        for variant in a.variants.split(','):
            if variant not in ('baseline', 'inline', 'unroll2', 'unroll4', 'inline2'):
                raise ValueError('Unknown variant')
            source = original
            if variant.startswith('inline'):
                needle = '__attribute__((noinline))\nvoid mont'
                if source.count(needle) != 1:
                    raise RuntimeError('Montgomery hint boundary changed')
                source = source.replace(needle, '__attribute__((always_inline))\nvoid mont')
            if variant in ('unroll2', 'inline2', 'unroll4'):
                needle = '#pragma unroll 1\n        for(int j='
                if source.count(needle) != 2:
                    raise RuntimeError('Inner loop boundary changed')
                source = source.replace(needle, '#pragma unroll ' + ('4' if variant == 'unroll4' else '2') + '\n        for(int j=')
            (a.out / (variant + '.cl')).write_text(source, encoding='utf-8')
            encoded = source.encode(); text = ct.c_char_p(encoded); size = SIZE(len(encoded)); error = ct.c_int()
            program = cl.cl.clCreateProgramWithSource(cl.context, 1, ct.byref(text), ct.byref(size), ct.byref(error)); check(error.value, 'program')
            kernel = None
            try:
                device = HANDLE(cl.device); start = time.perf_counter()
                code = cl.cl.clBuildProgram(program, 1, ct.byref(device), f'-cl-std=CL1.2 -DECM_POINT_NW={nw}'.encode(), None, None)
                build = time.perf_counter() - start
                size = SIZE(); cl.cl.clGetProgramBuildInfo(program, cl.device, 0x1183, 0, None, ct.byref(size))
                log = ct.create_string_buffer(size.value); cl.cl.clGetProgramBuildInfo(program, cl.device, 0x1183, size, log, None)
                (a.out / (variant + '.build.log')).write_bytes(log.value)
                check(code, 'build')
                kernel = cl.cl.clCreateKernel(program, b'point_ladder', ct.byref(error)); check(error.value, 'kernel')
                resources = {}
                for key, name in [(0x11b0, 'max_group'), (0x11b2, 'local_bytes'), (0x11b3, 'preferred_group_multiple'), (0x11b4, 'private_bytes')]:
                    value = U64(); check(cl.cl.clGetKernelWorkGroupInfo(kernel, cl.device, key, 8, ct.byref(value), None), name); resources[name] = value.value
                if a.group > resources['max_group'] or a.group < 1:
                    raise RuntimeError('Invalid group')
                size = SIZE(); check(cl.cl.clGetProgramInfo(program, 0x1165, ct.sizeof(size), ct.byref(size), None), 'binary size')
                data = (ct.c_ubyte * size.value)(); pointers = (HANDLE * 1)(ct.addressof(data))
                check(cl.cl.clGetProgramInfo(program, 0x1166, ct.sizeof(pointers), pointers, None), 'program binary')
                binary = bytes(data); (a.out / (variant + '.bin')).write_bytes(binary)
                decoded = binary.decode('utf-8', errors='replace')
                # NVIDIA returns PTX here. Match declarations, not the suffix
                # of ld.local/st.local instructions. Counts are program-wide,
                # not hardware registers, dynamic traffic or spill counters.
                resources.update(binary_bytes=len(binary), ptx_local_declarations=re.findall(r'(?m)^\s*\.local[^;]*;', decoded),
                                 ptx_ld_local=len(re.findall(r'ld\.local', decoded)), ptx_st_local=len(re.findall(r'st\.local', decoded)))
                args = [ct.c_int(nw), HANDLE(constants), U64(ninv), HANDLE(ks), U64(a.count), HANDLE(ox), HANDLE(oz)]
                for i, value in enumerate(args):
                    check(cl.cl.clSetKernelArg(kernel, i, ct.sizeof(value), ct.byref(value)), 'argument')
                samples = []
                for repeat in range(a.repeats):
                    event = HANDLE(); global_size = SIZE(((a.count + a.group - 1) // a.group) * a.group); local = SIZE(a.group)
                    start = time.perf_counter(); cpu_start = time.process_time()
                    check(cl.cl.clEnqueueNDRangeKernel(cl.queue, kernel, 1, None, ct.byref(global_size), ct.byref(local), 0, None, ct.byref(event)), 'enqueue')
                    try:
                        if a.wait == 'sleep':
                            check(cl.cl.clFlush(cl.queue), 'flush')
                            while True:
                                status = ct.c_int(); check(cl.cl.clGetEventInfo(event, 0x11d3, ct.sizeof(status), ct.byref(status), None), 'event status')
                                if status.value <= 0:
                                    check(status.value, 'execution'); break
                                time.sleep(.001)
                        check(cl.cl.clWaitForEvents(1, ct.byref(event)), 'wait')
                        times = []
                        for key in (0x1282, 0x1283):
                            value = U64(); check(cl.cl.clGetEventProfilingInfo(event, key, 8, ct.byref(value), None), 'event profile'); times.append(value.value)
                    finally:
                        cl.cl.clReleaseEvent(event)
                    xx, zz = (U64 * (a.count * nw))(), (U64 * (a.count * nw))()
                    for buffer, dest in ((ox, xx), (oz, zz)):
                        check(cl.cl.clEnqueueReadBuffer(cl.queue, buffer, 1, 0, ct.sizeof(dest), dest, 0, None, None), 'download')
                    wall = time.perf_counter() - start; cpu_seconds = time.process_time() - cpu_start
                    verify(xx, zz)
                    output = bytes(xx) + bytes(zz)
                    if baseline is None:
                        baseline = output
                    if output != baseline:
                        raise RuntimeError('Compiler hints changed exact X/Z outputs')
                    samples.append({'kernel_seconds': (times[1] - times[0]) * 1e-9, 'call_seconds': wall, 'cpu_seconds': cpu_seconds})
                row = {'variant': variant, 'source_sha256': hashlib.sha256(encoded).hexdigest(), 'build_seconds': build,
                       'resources': resources, 'samples': samples, 'median_kernel_seconds': st.median(s['kernel_seconds'] for s in samples),
                       'output_sha256': hashlib.sha256(baseline).hexdigest(), 'checked_words': a.repeats * a.count * nw * 2, 'bad': 0}
                report['rows'].append(row); (a.out / 'partial.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
                print(f'{variant}: build={build:.3f}s kernel={row["median_kernel_seconds"]:.6f}s private={resources["private_bytes"]} bytes', flush=True)
            finally:
                if kernel:
                    cl.cl.clReleaseKernel(kernel)
                cl.cl.clReleaseProgram(program)
        if a.cuda_dll:
            dll = ct.CDLL(str(a.cuda_dll.resolve())); run = dll.ecm_cuda_point_run
            run.argtypes = [ct.POINTER(CudaInput), ct.c_int, ct.c_void_p, ct.c_void_p, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.c_void_p, ct.c_int]
            arrays = [(U64 * nw)(*words(value)) for value in (n, x * r % n, r, a24 * r % n, r)]
            scalars_array = (U64 * a.count)(*scalars)
            inputs = CudaInput(a.cuda_device, nw, a.count, 64, ninv, *arrays, scalars_array, None); samples = []
            for repeat in range(a.repeats):
                xx, zz = (U64 * (a.count * nw))(), (U64 * (a.count * nw))(); kernel, wall = ct.c_double(), ct.c_double(); error = ct.create_string_buffer(2048)
                if run(ct.byref(inputs), 0, xx, zz, ct.byref(kernel), ct.byref(wall), error, len(error)):
                    raise RuntimeError(error.value.decode())
                verify(xx, zz)
                if bytes(xx) + bytes(zz) != baseline:
                    raise RuntimeError('CUDA and OpenCL XADD6 scale differs')
                samples.append({'kernel_seconds': kernel.value, 'call_seconds': wall.value})
            report['cuda'] = {'dll_sha256': hashlib.sha256(a.cuda_dll.read_bytes()).hexdigest(), 'samples': samples,
                              'median_kernel_seconds': st.median(s['kernel_seconds'] for s in samples), 'bad': 0}
    finally:
        cl.close()
    report['status'] = 'passed'; (a.out / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
