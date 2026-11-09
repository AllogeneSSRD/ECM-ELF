"""Attribute native CUDA kernels to Stage2 NTT phases in a Systems SQLite export.

Capture CUDA and NVTX with NTT_HOST_TRACE=1 and native submission. GPU kernel
durations exclude host launch gaps, copies and CUDA API waits.
"""
import argparse
from bisect import bisect_right
from collections import defaultdict
import json
from pathlib import Path
import re
import sqlite3


def enclosing(ranges, start):
    index = bisect_right(ranges, (start, float('inf'))) - 1
    return ranges[index] if index >= 0 and start < ranges[index][1] else None


def phase(name):
    if 'pack_zero_kernel<' in name:
        return 'pack'
    if 'outer_fwd_kernel<' in name:
        return 'forward_outer'
    if 'outer_inv_kernel<' in name:
        return 'inverse_outer'
    match = re.search(r'(tile_kernel|outer_coop_kernel)<(?:\(int\)\d+, )?\(bool\)([01])', name)
    if match:
        direction = 'inverse' if match[2] == '1' else 'forward'
        return direction + ('_tile' if match[1] == 'tile_kernel' else '_outer')
    return None


def analyze(db, device):
    with sqlite3.connect(db.resolve().as_uri() + '?mode=ro', uri=True) as conn:
        spans = conn.execute("select globalTid,start,end,text from NVTX_EVENTS where end is not null "
                             "and (text='NTT.request' or text like 'NTT.shape %') order by start").fetchall()
        kernels = conn.execute('select k.globalPid,k.start,k.end,s.value,r.globalTid,r.start,k.graphNodeId '
                               'from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName '
                               'left join CUPTI_ACTIVITY_KIND_RUNTIME r on r.correlationId=k.correlationId '
                               'and (r.globalTid >> 24)=(k.globalPid >> 24) where k.deviceId=?', (device,)).fetchall()
    requests, shapes = defaultdict(list), defaultdict(list)
    request_lengths = {}
    for thread, start, end, label in spans:
        if label == 'NTT.request':
            requests[thread].append((start, end))
            continue
        _, length, batch = label.split()
        shapes[thread].append((start, end, int(length), int(batch)))
        request = enclosing(requests[thread], start)
        if request:
            key = (thread, request[0])
            if request_lengths.setdefault(key, int(length)) != int(length):
                raise ValueError('Mixed lengths in one NTT pack request')
    if not shapes or not kernels or len({row[0] for row in kernels}) != 1:
        raise ValueError('Expected one Stage2 GPU process with NTT shape ranges')

    totals = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    unmatched = defaultdict(int)
    gpu_ns = 0
    for _, start, end, name, thread, launch, graph in kernels:
        if graph:
            raise ValueError('Use native CUDA submissions for this phase analysis')
        gpu_ns += end - start
        category = phase(name)
        if not category:
            continue
        length = None
        if launch is not None:
            if category == 'pack':
                span = enclosing(requests[thread], launch)
                if span:
                    length = request_lengths.get((thread, span[0]))
            else:
                span = enclosing(shapes[thread], launch)
                if span:
                    length = span[2]
        if length is None:
            unmatched[category] += 1
            continue
        row = totals[length][category]
        row[0] += 1
        row[1] += end - start
    return dict(device=device, kernel_seconds=gpu_ns / 1e9, unmatched=dict(unmatched),
                by_length={n: {k: dict(launches=v[0], seconds=v[1] / 1e9) for k, v in stats.items()}
                           for n, stats in sorted(totals.items())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sqlite', type=Path)
    parser.add_argument('--device', type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(analyze(args.sqlite, args.device), indent=2))


if __name__ == '__main__':
    main()
