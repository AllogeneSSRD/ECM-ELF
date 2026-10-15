"""Partition own-process GPU event gaps by concurrent host CUDA API intervals."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(path, device):
    with sqlite3.connect('file:'+path.as_posix()+'?mode=ro', uri=True) as connection:
        connection.row_factory = sqlite3.Row
        connection.text_factory = lambda b: b.decode('utf-8', 'replace')
        tables = {r[0] for r in connection.execute("select name from sqlite_master where type='table'")}
        strings = dict(connection.execute('select id,value from StringIds'))
        events = []
        for table, kind in [('CUPTI_ACTIVITY_KIND_KERNEL', 'kernel'), ('CUPTI_ACTIVITY_KIND_MEMCPY', 'copy'),
                            ('CUPTI_ACTIVITY_KIND_MEMSET', 'memset')]:
            if table not in tables:
                continue
            for row in connection.execute('select * from '+table+' where deviceId=?', (device,)):
                value = dict(row)
                value['kind'] = kind
                value['name'] = strings.get(value.get('demangledName'), kind)
                events.append(value)
        if not events:
            raise ValueError('no selected device events')
        owners = {r['globalPid'] for r in events}
        if len(owners) != 1:
            raise ValueError('expected one application process')
        owner = owners.pop()
        apis = [dict(r) for r in connection.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME')
                if r['globalTid'] >> 24 == owner >> 24]
        ranges = []
        if 'NVTX_EVENTS' in tables:
            for row in connection.execute('select * from NVTX_EVENTS where eventType in (59,70) and end is not null'):
                value = dict(row)
                if value['globalTid'] >> 24 == owner >> 24 and value['end'] > value['start']:
                    value['name'] = strings.get(value.get('textId')) or value.get('text') or '(unnamed)'
                    ranges.append(value)
    for row in apis:
        row['name'] = strings[row['nameId']]
    events.sort(key=lambda r: (r['start'], r['end']))
    merged = []
    for row in events:
        if not merged or row['start'] > merged[-1][1]:
            merged.append([row['start'], row['end']])
        else:
            merged[-1][1] = max(merged[-1][1], row['end'])
    first, last = merged[0][0], merged[-1][1]
    # GPU intervals are already disjoint. Runtime APIs can overlap on different
    # threads; count such slices once under an explicit ambiguous category.
    changes = []
    for start, end in merged:
        changes.extend([(start, 'gpu', 1), (end, 'gpu', -1)])
    for row in apis:
        start, end = max(first, row['start']), min(last, row['end'])
        if start < end:
            changes.extend([(start, row['name'], 1), (end, row['name'], -1)])
    range_keys = {}
    for index, row in enumerate(ranges):
        start, end = max(first, row['start']), min(last, row['end'])
        if start < end:
            key = '_NVTX#'+str(index)
            range_keys[key] = row
            changes.extend([(start, key, 1), (end, key, -1)])
    changes.sort()
    active = Counter()
    partition = Counter()
    outside_by_range, gap_by_range = Counter(), Counter()
    previous = first
    for timestamp, name, delta in changes:
        if timestamp > previous and not active['gpu']:
            names = [k for k, count in active.items() if count and k != 'gpu' and k not in range_keys]
            key = names[0] if len(names) == 1 else 'outside_runtime_API' if not names else 'overlapping_runtime_APIs'
            partition[key] += timestamp-previous
            named = [range_keys[k] for k, count in active.items() if count and k in range_keys]
            if len({r['globalTid'] for r in named}) > 1:
                owner_name = 'overlapping_NVTX_threads'
            else:
                owner_name = max(named, key=lambda r: r['start'])['name'] if named else 'unannotated'
            gap_by_range[owner_name] += timestamp-previous
            if key == 'outside_runtime_API':
                outside_by_range[owner_name] += timestamp-previous
        active[name] += delta
        if not active[name]:
            del active[name]
        previous = timestamp
    gaps = [(right[0]-left[1], left[1], right[0]) for left, right in zip(merged, merged[1:])]
    gap_ns = sum(end-start for _, start, end in gaps)
    if sum(partition.values()) != gap_ns:
        raise ValueError('gap partition is not exhaustive/disjoint')
    if sum(gap_by_range.values()) != gap_ns or sum(outside_by_range.values()) != partition['outside_runtime_API']:
        raise ValueError('NVTX gap ownership partition failed')
    api_stats = {}
    for row in apis:
        item = api_stats.setdefault(row['name'], dict(count=0, total_seconds=0))
        item['count'] += 1
        item['total_seconds'] += (row['end']-row['start'])/1e9
    # Attribute asynchronous GPU memset work by the host enqueue's NVTX range,
    # never by the later GPU execution timestamp.
    by_correlation = {r['correlationId']: r for r in apis}
    memset_stats = {}
    for row in events:
        if row['kind'] != 'memset':
            continue
        call = by_correlation.get(row['correlationId'])
        parents = [r for r in ranges if call and r['globalTid'] == call['globalTid']
                   and r['start'] <= call['start'] < r['end']]
        name = max(parents, key=lambda r: r['start'])['name'] if parents else 'unannotated'
        item = memset_stats.setdefault(name, dict(count=0, bytes=0, gpu_seconds=0))
        item['count'] += 1
        item['bytes'] += row['bytes']
        item['gpu_seconds'] += (row['end']-row['start'])/1e9
    largest = []
    for duration, start, end in sorted(gaps, reverse=True)[:20]:
        before = max((r for r in events if r['end'] <= start), key=lambda r: r['end'])
        after = min((r for r in events if r['start'] >= end), key=lambda r: r['start'])
        compact = lambda r: {k: r[k] for k in ('name', 'kind', 'start', 'end', 'correlationId', 'bytes') if k in r}
        calls = [dict(name=r['name'], seconds_in_gap=(min(end, r['end'])-max(start, r['start']))/1e9,
                      start=r['start'], end=r['end'], correlationId=r['correlationId'])
                 for r in apis if r['start'] < end and r['end'] > start]
        largest.append(dict(seconds=duration/1e9, start_ns=start, end_ns=end,
                            previous_gpu=compact(before), next_gpu=compact(after), runtime_calls=calls))
    return dict(device=device, owner_global_pid=owner, runtime_threads=sorted({r['globalTid'] for r in apis}),
                window_seconds=(last-first)/1e9, gpu_event_union_seconds=sum(b-a for a, b in merged)/1e9,
                gap_seconds=gap_ns/1e9, partition_seconds={k: v/1e9 for k, v in partition.most_common()},
                nvtx_ranges=len(ranges), outside_runtime_by_range_seconds={k: v/1e9 for k, v in outside_by_range.most_common()},
                gap_by_range_seconds={k: v/1e9 for k, v in gap_by_range.most_common()},
                memset_by_enqueue_range=memset_stats,
                api_stats=api_stats, largest_gaps=largest, trace_sha256=sha(path),
                scope='Own process kernel/copy/memset union; intervals outside runtime API may include '
                      'CPU work, driver calls, scheduler delay or instrumentation. API overlap is not causation. '
                      'Profiler times are excluded from performance A/B.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--device', type=int, default=1)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = analyze(a.trace.resolve(), a.device)
    report['tool_sha256'] = sha(Path(__file__))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('window_seconds', 'gap_seconds', 'partition_seconds')}, indent=2))


if __name__ == '__main__':
    main()
