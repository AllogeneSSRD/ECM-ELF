"""Read a bounded snapshot of a running Prime95 ECM log; export timing evidence.

Standard library only. Never changes the Prime95 directory or controls its process.
Re-running rebuilds the exports from the current snapshot, without appending duplicates.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import re
import statistics

from parser import LINE_RE, TS_FMT, parse_log, worker_number

NUMBER = r'\d+(?:\.\d+)?'
SHAPE = re.compile(r'Using (\d+)MB of memory\.\s+D:\s*(\d+), degree-(\d+) polynomials\.\s+Ftree polys in memory:\s*(\d+)')
FACTOR = re.compile(r'M(\d+) has a factor:\s*(\d+) \(ECM curve (\d+), B1=(\d+), B2=(\d+)\)')
COMPLETE = re.compile(r'M(\d+) completed (\d+) ECM (?:Edwards|Montgomery) curves?, B1=(\d+), B2=(\d+)')
COUNTERS = re.compile(r'(Stage 1 complete|Stage 2 init complete|Stage 2 complete)\.\s*(\d+) transforms(?:, (\d+) modular inverses)?')
PHASE = re.compile(r'^(nQx complete|PolyR built|Poly compress|PolyG built|PolyH built|H\(X\) scaled|PolyF up|PolyF down|gg = mul H\(X\))\.\s*Time:\s*('+NUMBER+r') sec')
TREE_BUILD = re.compile(r'^Round off:.*avg poly_size:.*Time:\s*('+NUMBER+r') sec')
PROGRESS = re.compile(r'curve \d+ stage 2 at B2=(\d+) \[('+NUMBER+r')%\]')
FFT_CANDIDATE = re.compile(r'^FFT:\s*(\d+), B2:\s*(\d+)/(\d+), numvals:\s*(\d+)/(\d+), poly:\s*(\d+), efficiency:\s*('+NUMBER+r')')
ERROR = re.compile(r'FATAL ERROR|Possible hardware failure|ILLEGAL SUMOUT|SUMOUT mismatch|rounding error', re.I)
STOP = re.compile(r'Worker stopp(?:ed|ing)|Execution halted|Stopping worker', re.I)


def integer(value):
    number = Decimal(value.strip())
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError('Expected an integer: ' + value)
    return int(number)


def snapshot(path, encoding, drop_tail=True):
    """Read at most the size observed at open; ignore a concurrently written tail."""
    with path.open('rb') as handle:
        before = os.fstat(handle.fileno())
        raw = handle.read(before.st_size)
    after = path.stat()
    used = raw
    dropped = 0
    if drop_tail and used and not used.endswith((b'\n', b'\r')):
        end = used.rfind(b'\n') + 1
        dropped = len(used) - end
        used = used[:end]
    text = used.decode(encoding, errors='replace').lstrip('\ufeff')
    return text, used, dict(path=str(path), observed_bytes=before.st_size,
                           read_bytes=len(raw), parsed_bytes=len(used),
                           dropped_unterminated_tail_bytes=dropped,
                           sha256=hashlib.sha256(used).hexdigest(),
                           size_after_read=after.st_size,
                           changed_during_read=before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns,
                           replaced_during_read=before.st_ino != after.st_ino,
                           modified_ns=before.st_mtime_ns,
                           replacement_characters=text.count('\ufffd'))


def parse_worktodo(text):
    tasks, warnings = [], []
    worker = 1
    for line_no, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith(('#', ';', '//')):
            continue
        section = re.fullmatch(r'\[Worker(?: #(\d+))?\]', line, re.I)
        if section:
            worker = int(section.group(1) or 1)
            continue
        match = re.match(r'^(ECM|ECM2|ECMSTAGE2)\s*=(.*)$', line, re.I)
        if not match:
            continue
        try:
            parts = next(csv.reader([match.group(2)], skipinitialspace=True, strict=True))
            parts = [part.strip() for part in parts]
            tokens = re.findall(r'(?:^|,)\s*("(?:[^"]|"")*"|[^,]*)', match.group(2))
            quoted = [token.startswith('"') for token in tokens]
            if len(quoted) != len(parts):
                raise ValueError('Could not preserve quoted optional fields')
            aid = None
            try:
                integer(parts[0])
            except (ValueError, InvalidOperation):
                aid = parts.pop(0)
                quoted.pop(0)
            if len(parts) < 5:
                raise ValueError('Expected k,b,n,c and B1/save')
            kind = match.group(1).upper()
            k, b, exponent, c = [integer(part) for part in parts[:4]]
            task = dict(line=line_no, worker=worker, kind=kind, aid=aid, raw=raw,
                        k=k, b=b, exponent=exponent, c=c, b1=None, save=None,
                        b2=0, curves_remaining=100 if kind != 'ECMSTAGE2' else 0,
                        specific_sigma=None, skip_curves=0, known_factors=[],
                        input_bits=None, input_bits_source=None)
            if kind == 'ECMSTAGE2':
                task['save'] = parts[4]
                names = ('b2', 'skip_curves', 'curves_remaining')
            else:
                task['b1'] = integer(parts[4])
                names = ('b2', 'curves_remaining', 'specific_sigma')
            tail = parts[5:]
            tail_quoted = quoted[5:]
            # Quoted factor lists arrive as one CSV field; positions stay optional.
            # A lone integer factor in the final optional field is also accepted.
            for name in names:
                if not tail or tail_quoted[0]:
                    break
                part = tail.pop(0)
                tail_quoted.pop(0)
                if part:
                    task[name] = integer(part)
            if tail:
                factors = ','.join(tail)
                task['known_factors'] = [str(integer(part)) for part in factors.split(',') if part.strip()]
            if k == 1 and b == 2 and c == -1 and 0 < exponent <= 1_000_000:
                modulus = (1 << exponent) - 1
                for factor in task['known_factors']:
                    factor_int = int(factor)
                    if factor_int <= 1 or modulus % factor_int:
                        raise ValueError('Known factors do not divide the current Mersenne input')
                    modulus //= factor_int
                task['input_bits'] = modulus.bit_length()
                task['input_bits_source'] = 'current_worktodo_only'
            tasks.append(task)
        except (ValueError, InvalidOperation, csv.Error, IndexError) as exc:
            warnings.append(dict(line=line_no, message=str(exc), raw=raw))
    return tasks, warnings


def bucket(bound, targets, max_overshoot):
    if not bound or not targets:
        return None
    candidates = [target for target in targets if target <= bound <= target * (1 + max_overshoot)]
    return max(candidates) if candidates else None


def rich_runs(text, tasks, targets, max_overshoot):
    rows = parse_log(text)
    lines = text.splitlines()
    last_for_worker = {row['worker']: row['start_line'] for row in rows}
    epochs = defaultdict(int)
    prior = {}
    for row in rows:
        w = row['worker']
        row.update(run_id=f"w{w}:line{row['start_line']}", analysis_status=None,
                   factors=[], phase_events=[], fft_candidates=[], diagnostics=[],
                   polymult_helper_cpu_ids=[], roundoff_max=None,
                   D=None, poly_degree=None, ftree_polys_in_memory=None,
                   estimated_s2_s1_ratio=None, progress_b2=None, progress_percent=None,
                   completed_task=None, finished_ts=None, resumed=False,
                   actual_input_bits=None, input_bits_reason='Historical initial known factors are not printed in ECM start lines')
        init_seen = False
        stopped = False
        for line_no in range(row['start_line'], row['end_line'] + 1):
            match = LINE_RE.match(lines[line_no - 1].rstrip('\r'))
            if not match or worker_number(match.group('worker')) != w:
                continue
            msg, ts = match.group('msg'), match.group('ts')
            if 'Stage 2 init complete.' in msg:
                init_seen = True
            for counter in COUNTERS.finditer(msg):
                prefix = {'Stage 1 complete': 's1', 'Stage 2 init complete': 's2_init', 'Stage 2 complete': 's2_main'}[counter.group(1)]
                row[prefix + '_transforms'] = int(counter.group(2))
                row[prefix + '_inverses'] = int(counter.group(3)) if counter.group(3) is not None else None
            shape = SHAPE.search(msg)
            if shape:
                row['D'], row['poly_degree'], row['ftree_polys_in_memory'] = map(int, shape.groups()[1:])
            ratio = re.search(r'Estimated stage 2 vs\. stage 1 runtime ratio:\s*('+NUMBER+')', msg)
            if ratio:
                row['estimated_s2_s1_ratio'] = float(ratio.group(1))
            candidate = FFT_CANDIDATE.search(msg)
            if candidate:
                row['fft_candidates'].append(dict(zip(
                    ('fft_length', 'b2_target', 'b2_actual', 'numvals_used', 'numvals_available', 'poly_degree', 'efficiency'),
                    [int(value) if i < 6 else float(value) for i, value in enumerate(candidate.groups())])))
            phase = PHASE.search(msg)
            tree = TREE_BUILD.search(msg)
            if phase or tree:
                phase_name = phase.group(1) if phase else 'Ftree build level'
                row['phase_events'].append(dict(line=line_no, timestamp=ts,
                    scope='init' if not init_seen else 'main',
                    phase=phase_name,
                    timer_kind=('slice_extrapolation_in_reference' if phase_name in ('PolyF up', 'PolyF down')
                                else 'reported_detail'),
                    seconds=float(phase.group(2) if phase else tree.group(1))))
            factor = FACTOR.search(msg)
            if factor and int(factor.group(1)) == row['exponent']:
                row['factors'].append(dict(value=factor.group(2), line=line_no, timestamp=ts,
                                          curve=int(factor.group(3)), b1=int(factor.group(4)), b2=int(factor.group(5))))
                row['finished_ts'] = ts
            completed = COMPLETE.search(msg)
            if completed and int(completed.group(1)) == row['exponent']:
                row['completed_task'] = dict(curves=int(completed.group(2)), b1=int(completed.group(3)),
                                             reported_average_b2=int(completed.group(4)), line=line_no)
            if 'Stage 2 GCD complete.' in msg:
                row['finished_ts'] = ts
            progress = PROGRESS.search(msg)
            if progress:
                row['progress_b2'], row['progress_percent'] = int(progress.group(1)), float(progress.group(2))
            if ERROR.search(msg):
                row['diagnostics'].append(dict(line=line_no, message=msg))
            helper = re.search(r'polymult helper thread on logical CPU (\d+)', msg)
            if helper:
                row['polymult_helper_cpu_ids'] = sorted(set(row['polymult_helper_cpu_ids']) | {int(helper.group(1))})
            roundoff = re.search(r'Round off:\s*([\d.eE+-]+)', msg)
            if roundoff:
                value = float(roundoff.group(1))
                row['roundoff_max'] = max(value, row['roundoff_max'] if row['roundoff_max'] is not None else value)
            # The final row extends to EOF. Prime95 can print "Resuming." while
            # stopping an idle worker after the task/GCD has already completed;
            # that message does not invalidate the completed curve's timers.
            if not row['finished_ts'] and re.search(r'\b(?:Resuming|Restarting)\b', msg, re.I):
                row['resumed'] = True
            if STOP.search(msg):
                stopped = True

        complete = all(row.get(key) is not None for key in ('s2_init_time', 's2_time', 's2_gcd_time'))
        if row['diagnostics']:
            status = 'error'
        elif row['resumed']:
            status = 'resumed_or_restarted'
        elif complete:
            status = 'complete_stage2'
        elif row['factors']:
            status = 'factor_early'
        elif stopped:
            status = 'interrupted'
        elif row['start_line'] == last_for_worker[w]:
            status = 'open_at_snapshot'
        else:
            status = 'incomplete'
        row['analysis_status'] = status
        row['benchmark_eligible'] = status == 'complete_stage2'
        row['b2_target_bucket'] = bucket(row['b2_requested'] or row['b2'], targets, max_overshoot)
        row['b2_overshoot_percent'] = (100 * (row['b2'] / row['b2_requested'] - 1)
                                      if row['b2'] and row['b2_requested'] else None)
        scope = (row['exponent'], row['b1'], row['curve_type'], row['b2_target_bucket'] or row['b2_requested'])
        previous = prior.get(w)
        if previous and (previous['scope'] != scope or previous['row']['factors'] or previous['row']['completed_task']):
            epochs[w] += 1
        row['modulus_epoch'] = f"w{w}:epoch{epochs[w]}"
        row['modulus_epoch_note'] = 'Conservative factor/task-boundary segment, not a recovered modulus identity'
        prior[w] = dict(scope=scope, row=row)
        row['queue_candidate_lines'] = [task['line'] for task in tasks
            if task['worker'] == w and (task['k'], task['b'], task['c']) == (1, 2, -1)
            and task['exponent'] == row['exponent'] and (task['b1'] is None or task['b1'] == row['b1'])
            and (task['b2'] == 0 or task['b2'] == row['b2_requested']
                 or (row['b2_target_bucket'] is not None and bucket(task['b2'], targets, max_overshoot) == row['b2_target_bucket']))]
        row['queue_match_note'] = 'Current queue candidates only; not evidence of historical known factors'
        if complete:
            row['s2_total_time'] = row['s2_init_time'] + row['s2_time'] + row['s2_gcd_time']
            row['logged_curve_time'] = row['s1_time'] + row['s2_total_time'] if row['s1_time'] is not None else None
            row['phase_totals'] = dict(s2_init=row['s2_init_time'], s2_main=row['s2_time'], gcd=row['s2_gcd_time'])
            main = defaultdict(float)
            init = defaultdict(float)
            for event in row['phase_events']:
                (init if event['scope'] == 'init' else main)[event['phase']] += event['seconds']
            row['phase_detail_totals'] = dict(init=dict(init), main=dict(main))
            row['init_detail_residual'] = row['s2_init_time'] - sum(init.values())
            row['main_detail_residual'] = row['s2_time'] - sum(main.values())
            row['detail_timers_additive'] = False
        else:
            row['s2_total_time'] = row['logged_curve_time'] = None
            row['phase_totals'] = row['phase_detail_totals'] = None
        finish = row['finished_ts']
        row['timestamp_elapsed_seconds'] = ((datetime.strptime(finish, TS_FMT) - datetime.strptime(row['start_ts'], TS_FMT)).total_seconds()
                                            if finish else None)
    return rows


def stats(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
                min=min(values), max=max(values),
                stdev=statistics.stdev(values) if len(values) > 1 else None)


GROUP_KEYS = ('worker', 'exponent', 'b1', 'curve_type', 'modulus_epoch', 'b2_target_bucket',
              'D', 'poly_degree', 's1_fft', 's1_fft_type', 's2_fft', 's2_fft_type', 'using_mem')


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row.get(name) for name in GROUP_KEYS)
        # Without explicit buckets, do not silently merge distinct requested bounds.
        key += (None if row['b2_target_bucket'] is not None else row['b2_requested'],)
        groups[key].append(row)
    result = []
    for key, members in groups.items():
        good = [row for row in members if row['benchmark_eligible']]
        item = dict(zip(GROUP_KEYS, key[:-1]))
        item.update(requested_b2_exact_group=key[-1], attempts=len(members), complete_curves=len(good),
                    statuses=dict(Counter(row['analysis_status'] for row in members)),
                    run_ids=[row['run_id'] for row in members],
                    actual_b2_min=min((row['b2'] for row in members if row['b2'] is not None), default=None),
                    actual_b2_max=max((row['b2'] for row in members if row['b2'] is not None), default=None))
        for name in ('s1_time', 's2_init_time', 's2_time', 's2_gcd_time', 's2_total_time', 'logged_curve_time'):
            item[name] = stats(row.get(name) for row in good)
        item['mean_phase_percent'] = ({name: 100 * statistics.mean(row['phase_totals'][name] for row in good) / item['s2_total_time']['mean']
                                       for name in ('s2_init', 's2_main', 'gcd')}
                                      if good and item['s2_total_time']['mean'] > 0 else None)
        result.append(item)
    return result


def csv_file(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8-sig')
        return
    names = list(dict.fromkeys(name for row in rows for name in row))
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: json.dumps(value, ensure_ascii=False, separators=(',', ':'))
                             if isinstance(value, (dict, list)) else value for name, value in row.items()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--screen', type=Path, required=True)
    p.add_argument('--worktodo', type=Path, help='Defaults to worktodo.txt beside screen.log')
    p.add_argument('--output', type=Path, required=True, help='Export directory outside the Prime95 directory')
    p.add_argument('--encoding', default='utf-8-sig')
    p.add_argument('--b2-targets', type=integer, nargs='+', default=[], help='Explicit nominal B2 tiers; reported actual B2 is always retained')
    p.add_argument('--max-b2-overshoot', type=float, default=0.10, help='Maximum tier-match overshoot fraction; default 0.10')
    a = p.parse_args()
    a.screen = a.screen.resolve()
    a.worktodo = (a.worktodo or a.screen.with_name('worktodo.txt')).resolve()
    a.output = a.output.resolve()
    if a.screen.parent == a.output or a.screen.parent in a.output.parents or a.worktodo.parent == a.output or a.worktodo.parent in a.output.parents:
        p.error('Output must be outside both input directories')
    if not 0 <= a.max_b2_overshoot < 1 or any(value <= 0 for value in a.b2_targets):
        p.error('B2 targets must be positive and overshoot must be in [0,1)')
    a.output.mkdir(parents=True, exist_ok=True)
    text, raw_log, log_meta = snapshot(a.screen, a.encoding)
    todo_text, raw_todo, todo_meta = snapshot(a.worktodo, a.encoding, drop_tail=False)
    tasks, warnings = parse_worktodo(todo_text)
    rows = rich_runs(text, tasks, sorted(set(a.b2_targets)), a.max_b2_overshoot)
    observed = sum(bool(re.search(r'ECM on .*curve #', match.group('msg'))) for line in text.splitlines()
                   if (match := LINE_RE.match(line)) and worker_number(match.group('worker')) is not None)
    if observed != len(rows):
        warnings.append(dict(message=f'{observed} ECM starts observed, {len(rows)} supported Mersenne runs parsed'))
    if any(row.get('init_detail_residual', 0) < -0.02 or row.get('main_detail_residual', 0) < -0.02 for row in rows):
        warnings.append(dict(message='Some verbose detail timers exceed the enclosing timer. Reference Prime95 source scales PolyF up/down from one slice; detail totals are diagnostic estimates, not an additive phase breakdown'))
    groups = summarize(rows)
    status_counts = dict(Counter(row['analysis_status'] for row in rows))
    data = dict(schema_version=1, generated_utc=datetime.now(timezone.utc).isoformat(),
                source_log=log_meta, source_worktodo=todo_meta,
                b2_targets=sorted(set(a.b2_targets)), max_b2_overshoot=a.max_b2_overshoot,
                metadata_lines=[dict(line=i, text=line) for i, line in enumerate(text.splitlines(), 1)
                                if 'Optimizing for CPU architecture:' in line or 'Setting affinity to run worker on logical CPU' in line],
                counts=dict(runs=len(rows), statuses=status_counts, queue_tasks=len(tasks),
                            complete_stage2=sum(row['benchmark_eligible'] for row in rows)),
                timing_semantics='s2_total_time=init+main+GCD; logged_curve_time=S1+s2_total. '
                    'Verbose child timers are nested; PolyF up/down are slice extrapolations in the reference source. '
                    'Never add child timers to their parents or normalize detailed estimates as measured shares. Planning, save I/O and PRP may be absent. '
                    'Timestamp elapsed is coarse (one-second timestamps), not a replacement for reported stage timers.',
                modulus_semantics='Exponent is a label, not guaranteed actual modulus bit width. '
                    'Current queue known factors cannot establish historical inputs. Epochs conservatively separate factor/task changes.',
                memory_semantics='Available/Using memory are the values printed by Prime95, not measured process peaks.',
                snapshot_semantics='Inputs are read-only and independently snapshotted, not an atomic pair. '
                    'open_at_snapshot means no terminal result observed; it does not prove the process is running.',
                warnings=warnings, tasks=tasks, runs=rows, groups=groups)
    (a.output / 'screen.snapshot.log').write_bytes(raw_log)
    (a.output / 'worktodo.snapshot.txt').write_bytes(raw_todo)
    (a.output / 'analysis.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    csv_file(a.output / 'runs.csv', [{k: v for k, v in row.items() if k not in ('phase_events', 'fft_candidates')} for row in rows])
    flattened = []
    for group in groups:
        flat = {name: value for name, value in group.items() if not isinstance(value, (list, dict))}
        flat['statuses'] = group['statuses']
        for name in ('s1_time', 's2_init_time', 's2_time', 's2_gcd_time', 's2_total_time', 'logged_curve_time'):
            for stat in ('n', 'mean', 'median', 'min', 'max', 'stdev'):
                flat[name + '_' + stat] = group[name][stat] if group[name] else None
        for name in ('s2_init', 's2_main', 'gcd'):
            flat[name + '_percent'] = group['mean_phase_percent'][name] if group['mean_phase_percent'] else None
        flattened.append(flat)
    csv_file(a.output / 'groups.csv', flattened)
    csv_file(a.output / 'worktodo.csv', tasks)
    csv_file(a.output / 'phases.csv', [dict(run_id=row['run_id'], worker=row['worker'], exponent=row['exponent'],
        b1=row['b1'], b2_actual=row['b2'], b2_target_bucket=row['b2_target_bucket'],
        analysis_status=row['analysis_status'], benchmark_eligible=row['benchmark_eligible'], **event)
        for row in rows for event in row['phase_events']])
    print(json.dumps(dict(output=str(a.output), **data['counts'], warnings=warnings), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
