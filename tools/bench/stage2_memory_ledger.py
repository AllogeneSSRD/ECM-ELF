"""Parse complete device-allocation lifecycle evidence (not driver VRAM usage)."""
from bench_stage2_production import fields


def parse(text):
    checkpoints = [fields(line, 'stage2_memory_ledger') for line in text.splitlines()
                   if line.startswith('stage2_memory_ledger:')]
    sites = [fields(line, 'stage2_memory_site') for line in text.splitlines()
             if line.startswith('stage2_memory_site:')]
    final = [c for c in checkpoints if c['snapshot'] == 'final']
    if len(final) != 1 or final[0]['version'] != '1' or final[0]['payload_only'] != '1':
        raise ValueError('one completed device-payload ledger is required')
    final = final[0]
    if (int(final['unknown_frees']) or final['live_bytes']!=final['persistent_bytes'] or
            final['live_allocations']!=final['persistent_allocations']):
        raise ValueError('curve-local device payload leaked or an untracked free occurred')
    if int(final['allocations'])+int(final['baseline_allocations'])-int(final['frees'])!=int(final['live_allocations']):
        raise ValueError('allocation/free lifecycle mismatch')
    names = [c['snapshot'] for c in checkpoints]
    if len(set(names)) != len(names):
        raise ValueError('use one curve per ledger matrix run')
    for checkpoint in checkpoints:
        live = sum(int(s['bytes']) for s in sites
                   if s['snapshot'] == checkpoint['snapshot'] and s['scope'] == 'live')
        if live != int(checkpoint['live_bytes']):
            raise ValueError('allocation-site sum differs from simultaneous live payload')
        interval = sum(int(s['bytes']) for s in sites
                       if s['snapshot'] == checkpoint['snapshot'] and s['scope'] == 'interval_peak')
        if interval != int(checkpoint['interval_peak_bytes']):
            raise ValueError('interval peak allocation-site composition differs from ledger')
    peak_sites = [s for s in sites if s['snapshot'] == 'final' and s['scope'] == 'global_peak']
    if sum(int(s['bytes']) for s in peak_sites) != int(final['peak_bytes']):
        raise ValueError('global peak allocation-site composition differs from ledger')
    if any(int(c['live_bytes']) > int(c['peak_bytes']) for c in checkpoints):
        raise ValueError('live payload exceeds cumulative peak')
    return dict(final=final, checkpoints=checkpoints, sites=sites)
