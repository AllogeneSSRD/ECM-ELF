"""Serial GPU1 gate for the optional NTT warp tail; independently checks GMP spectra."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exe', required=True)
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--output', required=True)
    parser.add_argument('--short-reduce', type=int, choices=(0, 1))
    args = parser.parse_args()
    exe = Path(args.exe).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    base_env = {k: v for k, v in os.environ.items() if not k.startswith('NTT_')}
    if args.short_reduce is not None:
        base_env['NTT_GL_SHORT_REDUCE'] = str(args.short_reduce)
    rows = []
    for enabled in ('0', '1'):
        env = dict(base_env, NTT_FUSE_WARP_TAIL=enabled, NTT_FUSE_WARP_TEST='1',
                   NTT_FUSE_LIFETIME_TEST='1', NTT_S4_OLDTAIL='0', NTT_NAME_MAX='1')
        cmd = [str(exe), '--real', '--n-hex', 'ffffffffffffffc5', '--sigma', '26',
               '--b1', '20', '--b2', '1000', '--d', '210', '--device', str(args.device)]
        run = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             timeout=180)
        text = run.stdout.decode('utf-8', errors='replace')
        (out / f'warp_{enabled}.log').write_text(text, encoding='utf-8')
        checks = {
            'exit': run.returncode == 0,
            'independent_gmp_dft_and_inverse':
                'ntt_fuse_capacity_check: cases=216 words=6854400 bad=0' in text,
            'tile_boundaries_and_three_slices':
                f'ntt_fuse_warp_check: requested={enabled} cases=216 batch=3 t=4,5,6,7,8,11,12 bad=0' in text,
            'lifetime': bool(re.search(r'ntt_fuse_lifetime_check: calls=14 bad=0 .*leaked_bytes=0', text)),
            'cached_mode_and_shared_capacity_switch':
                'ntt_fuse_warp_switch_check: calls=4 words=98304 bad=0' in text,
            'no_local_spill': bool(re.search(
                rf'ntt_fuse_warp_resources: requested={enabled} fwd_regs=\d+ inv_regs=\d+ fwd_local=0 inv_local=0 fwd_max_blocks=[1-9] inv_max_blocks=[1-9]', text)),
        }
        if args.short_reduce is not None:
            checks['selected_goldilocks_mode'] = (
                f'ntt_gl_reduce_mode: device={args.device} short={args.short_reduce}' in text)
        rows.append({'warp_tail': int(enabled), 'returncode': run.returncode, 'checks': checks,
                     'command': cmd, 'ntt_env': {k: v for k, v in env.items() if k.startswith('NTT_')}})
        print(f'warp_tail={enabled}: ' + ', '.join(f'{k}={v}' for k, v in checks.items()), flush=True)
    result = {'exe': str(exe), 'sha256': hashlib.sha256(exe.read_bytes()).hexdigest(),
              'device': args.device, 'short_reduce': args.short_reduce, 'runs': rows,
              'passed': sum(sum(r['checks'].values()) for r in rows),
              'failed': sum(sum(not v for v in r['checks'].values()) for r in rows)}
    (out / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return int(result['failed'] != 0)


if __name__ == '__main__':
    raise SystemExit(main())
