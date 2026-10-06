#!/usr/bin/env python3
"""Summarize exported Stage1 NCU CSV or Nsight Systems SQLite without replaying GPU work."""
import argparse
import csv
import json
from pathlib import Path
import sqlite3


def ncu(path):
    with path.open(encoding='utf-8-sig', newline='') as source:
        records = list(csv.reader(source))
    header = next(i for i, row in enumerate(records) if row and row[0] == 'ID')
    names, units = records[header:header + 2]
    kernels = []
    for row in records[header + 2:]:
        if len(row) != len(names) or not row[0].isdigit():
            continue
        data = dict(zip(names, row))
        keys = [k for k in names if k.startswith(('launch__', 'smsp__')) and
                any(s in k for s in ('register', 'occupancy_limit_registers',
                                    'warps_active.avg.per_cycle_active',
                                    'warps_eligible.avg.per_cycle_active',
                                    'issue_active.avg.pct', 'per_issue_active.ratio'))]
        keys += ['gpu__time_duration.sum', 'sm__throughput.avg.pct_of_peak_sustained_elapsed',
                 'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed',
                 'sm__warps_active.avg.pct_of_peak_sustained_active']
        kernels.append(dict(kernel=data['Kernel Name'], device=data['Device'],
            grid=data['Grid Size'], block=data['Block Size'],
            metrics={k: dict(value=data[k], unit=units[names.index(k)]) for k in keys if k in data}))
    if not kernels:
        raise ValueError('No captured kernel rows in NCU raw CSV')
    return dict(tool='ncu', scope='replayed kernel counters; timings are not benchmark rates', kernels=kernels)


def union_ns(intervals):
    total = 0
    start = end = None
    for a, b in sorted(intervals):
        if start is None:
            start, end = a, b
        elif a <= end:
            end = max(end, b)
        else:
            total += end - start
            start, end = a, b
    return total + (end - start if start is not None else 0)


def nsys(path, mode):
    with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        kernels = db.execute('''SELECT k.start,k.end,k.deviceId,s.value
            FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName
            WHERE s.value LIKE '%kernel_suyama_domain%' ''').fetchall()
        # MODE is part of the demangled template signature; INIT/EXPORT are excluded.
        chosen = [r for r in kernels if f'(int){mode}>' in r[3]]
        if not chosen:
            raise ValueError('No Stage1 kernels with the requested template mode')
        left = min(r[0] for r in chosen); right = max(r[1] for r in chosen)
        busy = union_ns([(r[0], r[1]) for r in chosen])
        result = dict(tool='nsys', device_ids=sorted({r[2] for r in chosen}),
            template_mode=mode, kernel_count=len(chosen),
            scope='first to last selected Stage1 kernel; excludes preparation and checkpoint',
            span_seconds=(right-left)/1e9, kernel_union_seconds=busy/1e9,
            no_own_kernel_seconds=(right-left-busy)/1e9, kernel_busy_fraction=busy/(right-left))
        if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables:
            result['copies'] = [dict(kind=r[0], count=r[1], bytes=r[2], seconds=r[3]/1e9)
                for r in db.execute('''SELECT copyKind,count(*),sum(bytes),sum(end-start)
                    FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind''')]
        result['runtime_api'] = [dict(name=r[0], count=r[1], seconds=r[2]/1e9)
            for r in db.execute('''SELECT s.value,count(*),sum(r.end-r.start) AS ns
                FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId
                GROUP BY r.nameId ORDER BY ns DESC LIMIT 10''')]
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path, help='NCU --csv --page raw export, or NSYS SQLite export')
    p.add_argument('--mode', type=int, choices=[2, 3, 4, 5, 6, 7], default=4)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = ncu(a.input) if a.input.suffix.lower() == '.csv' else nsys(a.input, a.mode)
    a.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(a.output)


if __name__ == '__main__':
    main()
