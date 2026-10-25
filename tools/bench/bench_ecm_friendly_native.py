"""Run paired native-GMP CPU or optional CUDA Stage2 research benchmarks.

Each repeat alternates family order. No production configuration is modified.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import winreg


def main(args):
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    cpu = winreg.QueryValueEx(winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"), "ProcessorNameString")[0].strip()
    metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(), cpu=cpu, platform=platform.platform(),
                    frequency="user previously confirmed fixed 4 GHz; not independently remeasured",
                    mode=args.mode, probe=str(args.probe.resolve()), warmups_discarded=0,
                    manifest=json.loads((args.probe.parent/"manifest.json").read_text(encoding="utf-8-sig")))
    if args.mode == "cpu":
        handle = ctypes.windll.kernel32.GetCurrentProcess
        handle.restype = ctypes.c_void_p
        setmask = ctypes.windll.kernel32.SetProcessAffinityMask
        setmask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        if not setmask(handle(), 1 << args.logical_cpu):
            raise ctypes.WinError()
        metadata["logical_cpu"] = args.logical_cpu
    else:
        query = subprocess.run(["nvidia-smi", f"--id={args.device}",
             "--query-gpu=index,name,power.limit,clocks.max.graphics,driver_version", "--format=csv"],
             text=True, capture_output=True, check=True)
        metadata["gpu"] = query.stdout
        metadata["stage2_backend_version"] = args.backend_version
        metadata["B2"] = args.b2; metadata["D"] = args.d
    rows = []
    for bits in args.bits:
        for b1 in args.b1:
            name = f"bits{bits}_b1{b1}"
            if args.mode == "cpu":
                request = None
                command = [str(args.probe), "--benchmark", str(bits), str(b1), str(args.curves), str(args.repeats)]
            else:
                command = [str(args.probe), "--stage2"]
                request = ""
                for repeat in range(args.repeats):
                    for family in ((1, 2) if repeat % 2 == 0 else (2, 1)):
                        for seed in range(1, args.curves+1):
                            request += f"{family} {seed} {2**bits-1} {b1} 48 {args.b2} {args.d} {args.device}\n"
                (out/(name+".input.txt")).write_text(request, encoding="utf-8")
            with (out/(name+".log")).open("w", encoding="utf-8") as log, (out/(name+".errors.log")).open("w", encoding="utf-8") as err:
                result = subprocess.run(command, input=request, text=True, stdout=log, stderr=err, timeout=args.timeout)
            if result.returncode:
                raise RuntimeError(f"{name}: native probe failed; see raw log")
            current = []
            for line in (out/(name+".log")).read_text(encoding="utf-8").splitlines():
                if args.mode == "gpu":
                    if not line.startswith("FRIENDLY_GPU|"): continue
                    line = line.split("|", 1)[1]
                r = json.loads(line); current.append(r)
                if args.mode == "gpu":
                    assert r["bad"] == r["bad_factors"] == 0 and not r["factors"], r
                    r["bits"] = bits
            expected = 2*args.repeats*(1 if args.mode == "cpu" else args.curves)
            assert len(current) == expected, (name, len(current), expected)
            rows.extend(current)
            (out/"timings.json").write_text(json.dumps(dict(metadata, rows=rows), indent=2)+"\n", encoding="utf-8")
            print(f"Completed {name}: {len(current)} paired-family records", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("cpu", "gpu"), default="cpu")
    p.add_argument("--bits", type=int, nargs="+", default=[521, 2203, 4423])
    p.add_argument("--b1", type=int, nargs="+", default=[4096])
    p.add_argument("--curves", type=int, default=32)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--logical-cpu", type=int, default=1)
    p.add_argument("--device", type=int, default=1)
    p.add_argument("--b2", type=int, default=26000000)
    p.add_argument("--d", type=int, default=30030)
    p.add_argument("--backend-version", default="unspecified")
    p.add_argument("--timeout", type=int, default=1800)
    a = p.parse_args()
    if a.curves < 1 or a.repeats < 1 or min(a.bits) < 4 or min(a.b1) < 2:
        p.error("positive matrix dimensions and B1>=2 required")
    if a.mode == "gpu" and a.backend_version == "unspecified":
        p.error("record --backend-version for reused engine objects")
    main(a)
