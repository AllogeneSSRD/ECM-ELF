"""Check and measure the production field source on a GPU with Python integer oracles."""
import argparse
import ctypes as C
import ctypes.util
import json
import os
from pathlib import Path
import random
import statistics

ptr, uint, ulong, sz, sint = C.c_void_p, C.c_uint32, C.c_uint64, C.c_size_t, C.c_int
cl = C.WinDLL("OpenCL.dll") if os.name == "nt" else C.CDLL(C.util.find_library("OpenCL"))


def api(name, output, *inputs):
    fn = getattr(cl, "cl" + name)
    fn.restype = output
    fn.argtypes = inputs
    return fn


platforms = api("GetPlatformIDs", sint, uint, ptr, ptr)
devices = api("GetDeviceIDs", sint, ptr, ulong, uint, ptr, ptr)
info = api("GetDeviceInfo", sint, ptr, uint, sz, ptr, ptr)
context = api("CreateContext", ptr, ptr, uint, ptr, ptr, ptr, ptr)
queue = api("CreateCommandQueue", ptr, ptr, ptr, ulong, ptr)
program = api("CreateProgramWithSource", ptr, ptr, uint, ptr, ptr, ptr)
build = api("BuildProgram", sint, ptr, uint, ptr, C.c_char_p, ptr, ptr)
build_log = api("GetProgramBuildInfo", sint, ptr, ptr, uint, sz, ptr, ptr)
kernel = api("CreateKernel", ptr, ptr, C.c_char_p, ptr)
kernel_info = api("GetKernelWorkGroupInfo", sint, ptr, ptr, uint, sz, ptr, ptr)
buffer = api("CreateBuffer", ptr, ptr, ulong, sz, ptr, ptr)
read = api("EnqueueReadBuffer", sint, ptr, ptr, uint, sz, sz, ptr, uint, ptr, ptr)
arg = api("SetKernelArg", sint, ptr, uint, sz, ptr)
launch = api("EnqueueNDRangeKernel", sint, ptr, ptr, uint, ptr, ptr, ptr, uint, ptr, ptr)
wait = api("WaitForEvents", sint, uint, ptr)
profile = api("GetEventProfilingInfo", sint, ptr, uint, sz, ptr, ptr)


def check(status):
    if status:
        raise RuntimeError(f"OpenCL error {status}")


arr = C.byref


def create(fn, *args):
    rc = sint()
    handle = fn(*args, arr(rc))
    check(rc.value)
    return ptr(handle)


def release(kind, value):
    check(api("Release" + kind, sint, ptr)(value))


def text_info(fn, *args):
    size = sz()
    check(fn(*args, 0, None, arr(size)))
    out = C.create_string_buffer(size.value)
    check(fn(*args, size, out, None))
    return out.value.decode(errors="replace")


def enumerate_gpu():
    count = uint()
    check(platforms(0, None, arr(count)))
    pp = (ptr * count.value)()
    check(platforms(count, pp, None))
    for p in pp:
        rc = devices(p, 4, 0, None, arr(count))
        if rc == -1:
            continue
        check(rc)
        dd = (ptr * count.value)()
        check(devices(p, 4, count, dd, None))
        yield from dd


BENCH = r"""
__kernel void mul_chain(__global const uint *constants, __global const uint *inputs,
                       __global uint *outputs, uint np0, uint repeats) {
    const size_t lane = get_global_id(0), count = get_global_size(0);
    const fe n = fe_load(constants, 1);
    fe a = fe_load(inputs + lane, count);
    const fe b = fe_load(inputs + LIMBS * count + lane, count);
    for (uint i = 0; i < repeats; ++i)
        a = fe_mul(a, b, n, np0);
    fe_store(outputs + lane, count, a);
}
__kernel void sqr_chain(__global const uint *constants, __global const uint *inputs,
                       __global uint *outputs, uint np0, uint repeats) {
    const size_t lane = get_global_id(0), count = get_global_size(0);
    const fe n = fe_load(constants, 1);
    fe a = fe_load(inputs + lane, count);
    for (uint i = 0; i < repeats; ++i)
        a = fe_sqr(a, n, np0);
    fe_store(outputs + lane, count, a);
}
"""
MODES = {"cios":(0,0,0), "fips":(1,0,0), "fips_square":(1,1,0),
         "ptx":(1,0,1), "ptx_square":(1,1,1)}
MASK = (1 << 32) - 1


def inputs(n, count):
    rng = random.Random(n)
    edge = (0, 1, 2, n-1, n-2, n//2, min(MASK, n-1))
    pairs = [(a, b) for a in edge for b in edge]
    pairs.extend((rng.randrange(n), rng.randrange(n)) for _ in range(count-len(pairs)))
    return pairs


def trial(ctx, cmdq, k, words, n, items, rounds, group, times, square):
    count = len(items)
    def alloc(values):
        raw = (uint * len(values))(*values)
        return create(buffer, ctx, 4 | 32, C.sizeof(raw), raw)
    const = alloc([(n >> (32*j)) & MASK for j in range(words)])
    source = alloc([(ab[pos] >> (32*j)) & MASK for pos in range(2)
                    for j in range(words) for ab in items])
    out = (uint * (count * words))()
    output = create(buffer, ctx, 1, C.sizeof(out), None)
    np0 = (-pow(n, -1, 1 << 32)) & MASK
    for i, value in enumerate((const, source, output, uint(np0), uint(rounds))):
        check(arg(k, i, C.sizeof(value), arr(value)))
    g, local = sz(count), sz(group)
    elapsed = []
    for _ in range(times):
        event = ptr()
        check(launch(cmdq, k, 1, None, arr(g), arr(local), 0, None, arr(event)))
        check(wait(1, arr(event)))
        start, end = ulong(), ulong()
        for token, v in ((0x1282, start), (0x1283, end)):
            check(profile(event, token, C.sizeof(v), arr(v), None))
        elapsed.append((end.value - start.value) * 1e-9)
        release("Event", event)
    check(read(cmdq, output, 1, 0, C.sizeof(out), out, 0, None, None))
    for m in (const, source, output):
        release("MemObject", m)
    inv = pow(1 << (32 * words), -1, n)
    check_count = count if rounds == 1 else 12
    for lane in range(check_count):
        x, y = items[lane]
        if square:
            for _ in range(rounds):
                x = x*x*inv % n
        else:
            x = x*pow(y*inv % n, rounds, n) % n
        got = sum(out[j*count + lane] << (32*j) for j in range(words))
        if got != x:
            raise AssertionError((words, n, lane, square, got, x))
    return elapsed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", type=int, default=1)
    p.add_argument("--widths", nargs="+", type=int, default=[128,192,256,384,512])
    p.add_argument("--modes", nargs="+", choices=MODES)
    p.add_argument("--rounds", type=int, default=1024)
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()
    if any(bits not in (128, 192, 256, 384, 512) for bits in args.widths) or args.rounds < 1:
        p.error("specify supported field widths and a positive iteration count")
    field_file = Path(__file__).resolve().parents[2] / "src/ecm/opencl/resources/edwards/field.clh"
    source = C.c_char_p((field_file.read_text() + BENCH).encode())
    dev = ptr(list(enumerate_gpu())[args.device])
    name = text_info(info, dev, 0x102B)
    driver = text_info(info, dev, 0x102D)
    vendor_id = uint()
    check(info(dev, 0x1001, C.sizeof(vendor_id), arr(vendor_id), None))
    modes = args.modes or [m for m, (_, _, ptx) in MODES.items() if not ptx or vendor_id.value == 0x10DE]
    if any(MODES[m][2] for m in modes) and vendor_id.value != 0x10DE:
        p.error("the PTX modes require NVIDIA OpenCL")
    ctx = create(context, None, 1, arr(dev), None, None)
    cmdq = create(queue, ctx, dev, 2)
    results = []
    for bits in args.widths:
        words = bits//32
        moduli = [(1<<bits)-1, (1<<bits)-159, (1<<(bits-1))+1, (1<<(bits-31))+1]
        for mode in modes:
            mul, sqr, ptx = MODES[mode]
            prog = create(program, ctx, 1, arr(source), None)
            options = (f"-cl-std=CL1.2 -DLIMBS={words} -DEDWARDS_UNROLL=1 -DEDWARDS_FIPS={mul} "
                       f"-DEDWARDS_FIPS_SQUARE={sqr} -DEDWARDS_FIPS_PTX={ptx}")
            if build(prog, 1, arr(dev), options.encode(), None, None):
                raise RuntimeError(text_info(build_log, prog, dev, 0x1183))
            for square in (False, True):
                if (sqr and not square) or (mul and not sqr and square):
                    continue
                k = create(kernel, prog, b"sqr_chain" if square else b"mul_chain")
                private = ulong()
                check(kernel_info(k, dev, 0x11B4, C.sizeof(private), arr(private), None))
                for n in moduli:
                    trial(ctx, cmdq, k, words, n, inputs(n, 2048), 1, 128, 1, square)
                elapsed = []
                if not args.verify_only:
                    n = moduli[1]
                    elapsed = trial(ctx, cmdq, k, words, n, inputs(n,32768), args.rounds, 128, 6, square)[1:]
                median = statistics.median(elapsed) if elapsed else None
                results.append(dict(bits=bits, mode=mode, square=square, seconds=elapsed,
                                    private_bytes=private.value, million_per_second=32768*args.rounds/median/1e6 if median else None))
                release("Kernel", k)
                print(name, bits, mode, "sqr" if square else "mul", f"private={private.value}", results[-1]["million_per_second"], flush=True)
            release("Program", prog)
        report = dict(device=name, driver=driver, threads=32768, workgroup=128, iterations=args.rounds, rows=results)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    release("CommandQueue", cmdq)
    release("Context", ctx)


if __name__ == "__main__":
    main()
