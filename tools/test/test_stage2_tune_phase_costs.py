"""Check exclusive engine partitions and paired worker cost contracts (CPU only)."""
import argparse
import copy
import json
import math
from pathlib import Path
import statistics
import subprocess
import tomllib

NAMES = ('shape', 'setup', 'baby', 'ftree', 'main_setup', 'inverse_setup',
         'giant_loop', 'descent', 'accum', 'finalize')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--large-profile', action='store_true', help='Check level-10 paired-array capacity and the 64 MiB limit')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    exe = a.fixture_dir/'fixture.exe'
    seed = (a.fixture_dir/'valid.toml').read_text(encoding='utf-8')
    accepted = rejected = partitions = 0

    def load(name, fields, valid=True, remove=()):
        nonlocal accepted, rejected
        text = seed
        for key, value in fields.items():
            token = json.dumps(value, allow_nan=False)
            old = next((line for line in text.splitlines() if line.startswith(key+' = ')), None)
            if old:
                text = text.replace(old, key+' = '+token)
            else:
                text = text.replace('[summary]', key+' = '+token+'\n[summary]')
        text = '\n'.join(line for line in text.splitlines()
                         if not any(line.startswith(key+' = ') for key in remove))+'\n'
        file = a.output/(name+'.toml')
        file.write_text(text, encoding='utf-8')
        proc = subprocess.run([str(exe), '--load', str(file)], capture_output=True, text=True)
        (a.output/(name+'.log')).write_text(proc.stdout+proc.stderr, encoding='utf-8')
        assert (proc.returncode == 0) == valid, (name, proc.stdout, proc.stderr)
        if valid:
            accepted += 1
        else:
            rejected += 1
        return file

    def partition(boundaries, expected=None):
        nonlocal partitions
        result = json.loads(subprocess.check_output(
            [str(exe), '--phases', *map(str, boundaries)], text=True))
        assert result['complete'] == (expected is not None), (boundaries, result)
        if expected is not None:
            assert all(math.isclose(x, y, abs_tol=1e-9) for x, y in zip(result['seconds'], expected))
            assert math.isclose(result['total'], sum(expected), abs_tol=1e-9)
        else:
            assert result['seconds'] == [0.]*10
        partitions += 1

    boundary = [.2, 100., 101., 103., 106., 110., 111., 113., 117., 122., 123., 125.]
    partition(boundary, [.2, 1., 2., 3., 1., 2., 4., 5., 1., 2.])
    partition([1., *([0.]*11)], [1., *([0.]*9)])
    partition([0., *([0.]*11)])
    large = [boundary[0], *[x+1e8 for x in boundary[1:]]]
    partition(large, [.2, 1., 2., 3., 1., 2., 4., 5., 1., 2.])
    for i in range(12):
        for value in (float('nan'), float('inf'), -1.):
            bad = boundary.copy()
            bad[i] = value
            partition(bad)
    for i, value in ((2, 99.), (3, 100.), (4, 102.), (5, 105.), (6, 109.),
                     (7, 110.), (8, 112.), (9, 116.), (10, 121.), (11, 122.)):
        bad = boundary.copy()
        bad[i] = value
        partition(bad)
    partition(boundary[:6]+[0.]*6)  # Early return cannot publish complete phases.

    phase = dict(phase_accounting='exclusive_engine_v1',
                 init_samples=[1., 2., 1.], main_samples=[1., 1., 3.],
                 init_seconds=1., main_seconds=1.)
    # Independent medians do not add to median(total): only paired sums must.
    phase_arrays = [[0., 0., 0.], [1., 2., 1.], [0., 0., 0.], [0., 0., 0.],
                    [1., 1., 3.], *[[0., 0., 0.] for _ in range(5)]]
    for name, values in zip(NAMES, phase_arrays):
        phase['phase_'+name+'_samples'] = values
        phase['phase_'+name+'_seconds'] = statistics.median(values)
    worker = dict(worker_accounting='spawn_wait_exit_v1', worker_samples=[2.4, 3.8, 4.2],
                  worker_seconds=3.8, worker_mad_seconds=.4,
                  worker_overhead_samples=[.4, .8, .2], worker_overhead_seconds=.4)
    load('legacy', {})
    load('phases', phase)
    load('worker', worker)
    both = dict(phase, **worker)
    newest = load('both', both)
    for i, (key, value) in enumerate([
        ('phase_accounting', 'unknown'), ('worker_accounting', 'unknown'),
        ('init_samples', [1., 2.]), ('main_samples', [1., -1., 3.]),
        ('phase_setup_samples', [1., 2., 2.]), ('phase_setup_seconds', 2.),
        ('phase_setup_samples', [1., 2., 1., 2.]), ('phase_baby_samples', [-1., 0., 0.]),
        ('worker_samples', [2.4, 3.8]), ('worker_samples', [2.4, 3.8, 4.3]),
        ('worker_seconds', 4.), ('worker_mad_seconds', .3),
        ('worker_overhead_samples', [-.1, .8, .2]), ('worker_overhead_seconds', .2),
        ('init_seconds', 2.), ('main_seconds', 2.),
    ]):
        load('bad_'+str(i), dict(both, **{key: value}), False)
    for i, key in enumerate(both):
        bad = copy.deepcopy(both)
        del bad[key]
        load('missing_'+str(i), bad, False, remove=(key,))
    # Compatible old/new inputs retain per-sample contracts during native merge.
    legacy = a.output/'legacy.toml'
    old = legacy.read_text(encoding='utf-8').replace('b2 = 26000000000', 'b2 = 26000000001')
    # Keep exact geometry consistent: floor(B2/D)+2 unchanged.
    assert tomllib.loads(old)['ecm']['sample_0']['giant_points'] == 144302
    legacy.write_text(old, encoding='utf-8')
    merged = subprocess.check_output([str(exe), '--merge', str(legacy), str(newest)], text=True)
    rows = tomllib.loads(merged)['ecm']
    assert len(rows) == 2 and sum('phase_accounting' in s for s in rows.values()) == 1
    (a.output/'merged.toml').write_text(merged, encoding='utf-8')
    capacity = None
    if a.large_profile:
        document = tomllib.loads(newest.read_text(encoding='utf-8'))
        document['profile'].update(effort_level=10, repeats=21, max_batches=0)
        document['summary']['measured'] = 3094
        values = document['ecm']['sample_0'].copy()
        unit = math.pi/10
        total = 10*unit
        values.update(repeats=21, seconds=[total]*21, median_seconds=total, mad_seconds=0.,
                      init_samples=[4*unit]*21, main_samples=[6*unit]*21,
                      init_seconds=4*unit, main_seconds=6*unit,
                      worker_samples=[total+.6]*21, worker_seconds=total+.6,
                      worker_mad_seconds=0., worker_overhead_samples=[.6]*21,
                      worker_overhead_seconds=.6)
        for name in NAMES:
            values['phase_'+name+'_samples'] = [unit]*21
            values['phase_'+name+'_seconds'] = unit
        def table(name, fields):
            return '\n['+name+']\n'+''.join(k+' = '+json.dumps(v)+'\n' for k, v in fields.items())
        large = a.output/'level10_capacity.toml'
        with large.open('w', encoding='utf-8') as stream:
            for group in ('profile', 'device', 'policy'):
                stream.write(table(group, {k:v for k,v in document[group].items() if not isinstance(v, dict)}))
            stream.write(table('policy.environment', document['policy']['environment']))
            for index in range(3094):
                values['b2'] = 26000000000+index
                values['giant_points'] = values['b2']//values['d']+2
                stream.write(table('ecm.sample_'+str(index), values))
            stream.write(table('summary', document['summary']))
        assert 16*1048576 < large.stat().st_size < 64*1048576
        assert subprocess.check_output([str(exe), '--load', str(large)], text=True).strip() == '3094'
        too_large = a.output/'over_limit.toml'
        with too_large.open('wb') as stream:
            stream.truncate(64*1048576+1)
        assert subprocess.run([str(exe), '--load', str(too_large)], capture_output=True).returncode != 0
        capacity = dict(scopes=3094, repeats=21, bytes=large.stat().st_size, bounded_rejection=True)
    result = dict(complete=True, partitions=partitions, accepted=accepted, rejected=rejected,
                  paired_medians_not_additive=True, mixed_contract_merge=True)
    if capacity:result['capacity'] = capacity
    (a.output/'result.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
