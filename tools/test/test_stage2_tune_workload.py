"""CPU gates for compressed NTT work features and paired timing evidence."""
import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, action='append', default=[])
    a = parser.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('work', ROOT/'tools/bench/analyze_stage2_tune_workload.py')
    work = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(work)
    plan = dict(D=6, P=2, I=11, workspace_buffers=2,
        request_program=dict(version=2, valid=True, supported=True, residency_required=True,
            full_fold_degree_required=True, no_eviction_model=True, process_peak_complete=False,
            admission_model=False, blocks=[dict(repeat=3, requests=[
                dict(phase=1, ma=2, mb=2, pairs=5, first=1, count=2, input='tree_raw')])],
            phases=[dict(phase=phase, groups=3 if i==1 else 0, pairs=15 if i==1 else 0,
                         chunks=9 if i==1 else 0) for i, phase in enumerate(work.PHASES)]),
        s4_memory=dict(valid=True, shapes=[dict(operand=2, N=16, slots=3)]),
        tree_workspace=dict(supported=True, batch_bytes=600, chunk_max=0, physical_chunks=True))
    rows = work.workload(plan)
    # 296 bytes per slice, chunk budget reduces 5 -> 2. Each repetition has 2x2 + 1x1 slices.
    assert rows == [dict(phase='gtrees', length=16, slices=1, request_occurrences=3,
                        calls=3, pairs=3, output_coefficients=6, field_words=48, field_nlog2n=192),
                    dict(phase='gtrees', length=16, slices=2, request_occurrences=3,
                        calls=6, pairs=12, output_coefficients=24, field_words=192, field_nlog2n=768)]
    huge = copy.deepcopy(plan)
    huge['request_program']['blocks'][0]['repeat'] = 10**12
    huge['request_program']['phases'][1].update(groups=10**12, pairs=5*10**12, chunks=3*10**12)
    huge_rows = work.workload(huge)
    assert sum(r['pairs'] for r in huge_rows) == 5*10**12
    assert sum(r['calls'] for r in huge_rows) == 3*10**12
    assert len(huge_rows) == 2  # No expansion of a trillion middle batches.
    limited = copy.deepcopy(plan)
    limited['tree_workspace']['chunk_max'] = 1
    limited['request_program']['phases'][1]['chunks'] = 15
    assert work.workload(limited)[0]['calls'] == 15
    rejects = 0

    def rejected(fn):
        nonlocal rejects
        try:
            fn()
        except (ValueError, KeyError):
            rejects += 1
        else:
            raise AssertionError('invalid evidence accepted')

    for group, key, value in [('request_program', 'supported', False),
                              ('request_program', 'process_peak_complete', True),
                              ('request_program', 'version', 3),
                              ('s4_memory', 'valid', False),
                              ('tree_workspace', 'batch_bytes', 0),
                              ('tree_workspace', 'physical_chunks', 1)]:
        bad = copy.deepcopy(plan)
        bad[group][key] = value
        rejected(lambda: work.workload(bad))
    for key, value in [('phase', 5), ('pairs', 0), ('first', 2), ('ma', 4),
                       ('input', 'invalid'), ('pairs', True)]:
        bad = copy.deepcopy(plan)
        bad['request_program']['blocks'][0]['requests'][0][key] = value
        rejected(lambda: work.workload(bad))
    for key, value in [('N', 15), ('slots', 4), ('operand', True)]:
        bad = copy.deepcopy(plan)
        bad['s4_memory']['shapes'][0][key] = value
        rejected(lambda: work.workload(bad))
    bad = copy.deepcopy(plan)
    bad['request_program']['phases'][1]['chunks'] += 1
    rejected(lambda: work.workload(bad))
    bad = copy.deepcopy(plan)
    bad['s4_memory']['shapes'] *= 2
    rejected(lambda: work.workload(bad))
    receipt = dict(d=6, p=2, giant_points=11, total_seconds=2., init_seconds=.5, main_seconds=1.5,
        giant_seconds=.3, gtrees_seconds=.4, fold_seconds=.2, descent_seconds=.3,
        inverse_seconds=.1, accum_seconds=.1, hits=0, bad=0, clean=1, fold_resident=1,
        frontier_resident=1, selftest_cases=10, checked=10)
    assert work.paired_trials(plan, [receipt])['total_seconds'] == [2.]
    exclusive = dict(receipt, phase_accounting='exclusive_engine_v1')
    phase_values = [.0, .1, .2, .2, .1, .2, .8, .2, .1, .1]
    for name, value in zip(work.ENGINE_PHASES, phase_values):
        exclusive['phase_'+name+'_seconds'] = value
    timers = work.paired_trials(plan, [exclusive])
    sample_costs = dict(phase_accounting='exclusive_engine_v1', init_samples=[.5], main_samples=[1.5],
                        init_seconds=.5, main_seconds=1.5,
                        worker_accounting='spawn_wait_exit_v1', worker_samples=[2.5],
                        worker_seconds=2.5, worker_mad_seconds=0.,
                        worker_overhead_samples=[.5], worker_overhead_seconds=.5)
    for name, value in zip(work.ENGINE_PHASES, phase_values):
        sample_costs['phase_'+name+'_samples'] = [value]
        sample_costs['phase_'+name+'_seconds'] = value
    assert work.published_costs(sample_costs, timers)['worker_samples'] == [2.5]
    rejected(lambda: work.paired_trials(plan, [receipt, exclusive]))
    for key, value in [('phase_accounting', 'unknown'), ('phase_setup_seconds', .2),
                       ('phase_accum_seconds', -.1), ('phase_finalize_seconds', True)]:
        bad = dict(exclusive, **{key: value})
        rejected(lambda: work.paired_trials(plan, [bad]))
    missing = dict(exclusive)
    del missing['phase_accounting']
    rejected(lambda: work.paired_trials(plan, [missing]))
    for key, value in [('phase_accounting', 'unknown'), ('init_samples', [.4]),
                       ('phase_setup_samples', [.2]), ('phase_setup_seconds', .2),
                       ('worker_samples', [2.6]), ('worker_mad_seconds', .1),
                       ('worker_samples', [2.5, 2.5]), ('worker_overhead_samples', [-.5]),
                       ('worker_accounting', 'unknown'), ('worker_seconds', 2.4)]:
        bad = dict(sample_costs, **{key: value})
        rejected(lambda: work.published_costs(bad, timers))
    for key in ('phase_accounting', 'worker_accounting', 'phase_setup_samples', 'worker_samples'):
        bad = dict(sample_costs)
        del bad[key]
        rejected(lambda: work.published_costs(bad, timers))
    for key, value in [('total_seconds', 3.), ('fold_resident', 0), ('giant_seconds', float('nan')),
                       ('descent_seconds', -.1), ('hits', 1), ('d', 8)]:
        bad = dict(receipt, **{key: value})
        rejected(lambda: work.paired_trials(plan, [bad]))
    ntt = dict(profile=dict(format=1, unit='field_convolution', repeats=3),
        summary=dict(failed=0, usable=True, measured=1), ntt={'length_16':
            dict(length=16, status='measured', unit='field_convolution', batch=1, bad=0,
                 verified_words_per_sample=16, seconds=[.1, .2, .3], median_seconds=.2,
                 conv_iter_per_s=5.)})
    assert work.ntt_samples(ntt) == {16: .2}
    for key, value in [('batch', 2), ('seconds', [.1]), ('median_seconds', .3),
                       ('conv_iter_per_s', 6), ('bad', 1), ('length', 17)]:
        bad = copy.deepcopy(ntt)
        bad['ntt']['length_16'][key] = value
        rejected(lambda: work.ntt_samples(bad))
    batch_ntt = dict(profile=dict(format=2, unit='field_convolution', repeats=3,
                                 min_log2=4, max_log2=4, slices=[1,2,3]),
        summary=dict(failed=0, usable=True, measured=2, skipped=1), ntt={'length_16': {}})
    for b in (1,2):
        batch_ntt['ntt']['length_16'][f'slices_{b}'] = dict(ntt['ntt']['length_16'],
            batch=b, log2_length=4, verified_words_per_sample=16*b,
            conv_iter_per_s=5.*b, batch_iter_per_s=5., reference_kind='gmp_3x3_distinct_constant_v1')
    batch_ntt['ntt']['length_16']['slices_3'] = dict(length=16, log2_length=4,
                                                  batch=3, status='skipped_memory')
    measured = work.ntt_batch_samples(batch_ntt)
    assert measured == {(16,1): .2, (16,2): .2}
    assert work.ntt_samples(batch_ntt) == {16: .2}
    coverage, annotated = work.ntt_workload_features(rows, measured)
    assert coverage['ntt_exact_batch_covered_pairs'] == 15
    assert coverage['ntt_exact_batch_covered_calls'] == 9
    assert coverage['ntt_missing_batch_bins'] == 0
    assert math.isclose(sum(r['exact_batch_reference_seconds'] for r in annotated), 1.8)
    coverage, annotated = work.ntt_workload_features(rows, {(16,1): .2})
    assert coverage['ntt_missing_batch_bins'] == 1
    assert 'ntt_batch_seconds' not in annotated[1]
    batch_rejects = 0
    for section, key, value in [('profile','slices',[1,2,2]), ('profile','slices',[0,2,3]),
            ('profile','slices',[True,2,3]), ('profile','slices',[1,2,65536]),
            ('profile','min_log2',3), ('profile','max_log2',28),
            ('summary','skipped',0), ('summary','measured',3), ('summary','failed',1)]:
        bad = copy.deepcopy(batch_ntt)
        bad[section][key] = value
        rejected(lambda: work.ntt_batch_samples(bad))
        batch_rejects += 1
    for key, value in [('batch',3), ('log2_length',3), ('verified_words_per_sample',16),
            ('conv_iter_per_s',5.), ('batch_iter_per_s',10.), ('reference_kind','gpu_self'),
            ('seconds',[.1]), ('bad',True), ('length',17)]:
        bad = copy.deepcopy(batch_ntt)
        bad['ntt']['length_16']['slices_2'][key] = value
        rejected(lambda: work.ntt_batch_samples(bad))
        batch_rejects += 1
    bad = copy.deepcopy(batch_ntt)
    del bad['ntt']['length_16']['slices_3']
    rejected(lambda: work.ntt_batch_samples(bad))
    batch_rejects += 1
    # Legacy data has no mask/environment qualification. Matching format 2 is
    # qualified for comparison only, never for ranking or total ECM prediction.
    identity = dict(zip(work.IDENTITY, ['0'*32,8,9,13030,13030,3,0]))
    ecm_policy = dict(device=dict(identity, add_sub_mask=1),
                      policy=dict(environment=dict(arena_cap_kb=1024, fuse_warp_tail=1)))
    batch_ntt['device'] = dict(identity, gl_add_sub_mask=1)
    batch_ntt['policy'] = dict(accounting='cuda_events_two_forward_product_inverse_v1',
                              environment=dict(ecm_policy['policy']['environment']))
    assert not work.ntt_policy_matches(ecm_policy, ntt)
    assert work.ntt_policy_matches(ecm_policy, batch_ntt)
    for section, key, value in [('device','gl_add_sub_mask',0), ('device','cuda_driver',12060),
            ('policy','accounting','unknown'), ('policy','environment',{}),
            ('policy','environment',dict(arena_cap_kb=1024,fuse_warp_tail=0)),
            ('policy','environment',dict(arena_cap_kb=1024,fuse_warp_tail=True))]:
        mismatch = copy.deepcopy(batch_ntt)
        mismatch[section][key] = value
        assert not work.ntt_policy_matches(ecm_policy,mismatch)
    (a.output/'ntt_batches.json').write_text(json.dumps(dict(complete=True, measured_shapes=2,
        skipped_shapes=1, rejected=batch_rejects, policy_mismatches=7)), encoding='utf-8')
    with tempfile.TemporaryDirectory(dir=a.output) as tmp:
        path = Path(tmp)/'duplicate.jsonl'
        path.write_text('{"D":1,"D":2}\n', encoding='utf-8')
        rejected(lambda: work.one_json(path))
        root = Path(tmp)
        raw = root/'raw'
        raw.mkdir()
        native = dict(plan, target_bits=16, bits=16, carrier_exponent=0, B1=20, B2=60,
                      curve_workspace_memory=dict(valid=True, finished=True))
        (raw/'case_1.plan.jsonl').write_text(json.dumps(native)+'\n', encoding='utf-8')
        (raw/'case_1_1.jsonl').write_text(json.dumps(receipt)+'\n', encoding='utf-8')
        sample = dict(target_bits=16, arithmetic_bits=16, carrier_exponent=0, modulus_kind='generic',
            b1=20, b2=60, d=6, p=2, giant_points=11, repeats=1, seconds=[2.], median_seconds=2.)
        device = dict(zip(work.IDENTITY, ['0123456789abcdef0123456789abcdef', 8, 9, 13030, 13030, 3, 0]))
        profile_text = work.table('profile', dict(format=3, unit='full_stage2', repeats=1))
        profile_text += work.table('device', device)+work.table('policy', dict(batch_mb=1))
        profile_text += work.table('ecm.sample_0', sample)
        profile_text += work.table('summary', dict(complete=1, failed=0, measured=1))
        profile = root/'ecm.toml'
        profile.write_text(profile_text, encoding='utf-8')
        command = [sys.executable, str(ROOT/'tools/bench/analyze_stage2_tune_workload.py'),
                   '--evidence', str(raw), '--profile', str(profile)]
        good = root/'good'
        subprocess.run(command+['--output', str(good)], capture_output=True, check=True, timeout=30)
        parsed = tomllib.loads((good/'workload.toml').read_text(encoding='utf-8'))
        assert parsed['summary']['complete'] and parsed['ecm']['sample_0']['total_pairs'] == 15
        assert not parsed['profile']['ranking_qualified']
        assert 'sha256' not in (good/'workload.toml').read_text(encoding='utf-8')
        assert str(root) not in (good/'workload.toml').read_text(encoding='utf-8')
        cli_rejects = 0

        def cli_rejected(name, args):
            nonlocal cli_rejects
            out = root/name
            result = subprocess.run(command+['--output', str(out)]+args, capture_output=True, timeout=30)
            assert result.returncode != 0 and not (out/'workload.toml').exists()
            assert not json.loads((out/'evidence.json').read_text(encoding='utf-8'))['complete']
            cli_rejects += 1

        empty = root/'empty'
        empty.mkdir()
        cli_rejected('missing_scope', ['--evidence', str(empty)])
        wrong = dict(receipt, total_seconds=3.)
        (raw/'case_1_1.jsonl').write_text(json.dumps(wrong)+'\n', encoding='utf-8')
        cli_rejected('wrong_receipt', [])
        (raw/'case_1_1.jsonl').write_text(json.dumps(receipt)+'\n', encoding='utf-8')
        ntt_text = work.table('profile', ntt['profile'])
        ntt_text += work.table('device', dict(device, cuda_driver=12060))
        ntt_text += work.table('ntt.length_16', ntt['ntt']['length_16'])
        ntt_text += work.table('summary', ntt['summary'])
        wrong_ntt = root/'wrong_ntt.toml'
        wrong_ntt.write_text(ntt_text, encoding='utf-8')
        cli_rejected('wrong_identity', ['--ntt-profile', str(wrong_ntt)])
        batch_ntt['device'] = dict(device, gl_add_sub_mask=1)
        batch_text = work.table('profile', batch_ntt['profile'])
        batch_text += work.table('device', batch_ntt['device'])
        batch_text += work.table('policy', dict(accounting=batch_ntt['policy']['accounting']))
        batch_text += work.table('policy.environment', batch_ntt['policy']['environment'])
        for key, measurement in batch_ntt['ntt']['length_16'].items():
            batch_text += work.table(f'ntt.length_16.{key}', measurement)
        batch_text += work.table('summary', batch_ntt['summary'])
        batch_path = root/'batch.toml'
        batch_path.write_text(batch_text, encoding='utf-8')
        subprocess.run(command+['--ntt-profile', str(batch_path), '--output', str(root/'batch')],
                       capture_output=True, check=True, timeout=30)
        batch_output = tomllib.loads((root/'batch/workload.toml').read_text(encoding='utf-8'))
        assert batch_output['ecm']['sample_0']['ntt_exact_batch_covered_pairs'] == 15
        assert batch_output['ecm']['sample_0']['ntt_exact_batch_covered_calls'] == 9
        assert not batch_output['profile']['ntt_policy_qualified']
        qualified = profile_text.replace(work.table('device', device),
                                        work.table('device', dict(device, add_sub_mask=1)))
        qualified += work.table('policy.environment', batch_ntt['policy']['environment'])
        profile.write_text(qualified, encoding='utf-8')
        subprocess.run(command+['--ntt-profile', str(batch_path), '--output', str(root/'qualified')],
                       capture_output=True, check=True, timeout=30)
        batch_output = tomllib.loads((root/'qualified/workload.toml').read_text(encoding='utf-8'))
        assert batch_output['profile']['ntt_policy_qualified']
        assert not batch_output['profile']['ranking_qualified']
        assert not batch_output['profile']['ntt_feature_is_time_prediction']
        profile.write_text(profile_text, encoding='utf-8')
        # Published worker intervals come from the parent, exclusive phases from raw children.
        (raw/'case_1_1.jsonl').write_text(json.dumps(exclusive)+'\n', encoding='utf-8')
        current = profile_text.replace(work.table('ecm.sample_0', sample),
                                       work.table('ecm.sample_0', dict(sample, **sample_costs)))
        profile.write_text(current, encoding='utf-8')
        subprocess.run(command+['--output', str(root/'exclusive')], capture_output=True, check=True, timeout=30)
        parsed = tomllib.loads((root/'exclusive/workload.toml').read_text(encoding='utf-8'))
        assert parsed['ecm']['sample_0']['phase_accounting'] == 'exclusive_engine_v1'
        assert parsed['ecm']['sample_0']['worker_samples'] == [2.5]
        bad = dict(sample_costs, worker_overhead_samples=[.6])
        profile.write_text(profile_text.replace(work.table('ecm.sample_0', sample),
                                               work.table('ecm.sample_0', dict(sample, **bad))), encoding='utf-8')
        cli_rejected('bad_worker_pair', [])
        profile.write_text(current, encoding='utf-8')
        (raw/'case_1_1.jsonl').write_text(json.dumps(receipt)+'\n', encoding='utf-8')
        cli_rejected('mismatched_contract', [])
    # Dense reference exists independently of the production request program.
    topo_spec = importlib.util.spec_from_file_location('topo', ROOT/'tools/test/test_stage2_request_program.py')
    topo = importlib.util.module_from_spec(topo_spec)
    topo_spec.loader.exec_module(topo)
    replay = []
    for directory in a.evidence:
        count = pairs = calls = bins = 0
        for path in directory.glob('case_*.plan.jsonl'):
            native = work.one_json(path)
            if not native['request_program']['supported']:
                continue
            topo.verify_topology(native)
            r = work.workload(native)
            count += 1
            pairs += sum(x['pairs'] for x in r)
            calls += sum(x['calls'] for x in r)
            bins += len(r)
        assert count
        replay.append(dict(directory=str(directory), plans=count, pairs=pairs, calls=calls, bins=bins))
    result = dict(complete=True, synthetic_cases=3, paired_timing_cases=2, ntt_cases=3,
                  ntt_batch_rejected=batch_rejects,
                  rejected=rejects, cli_roundtrips=4, cli_rejected=cli_rejects, replay=replay)
    (a.output/'result.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
