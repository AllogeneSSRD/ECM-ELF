"""Offline carrier/geometry research; no executable launch or CUDA allocation.

Reuses the existing integer reproduction of choose_cfg. Outputs analytical
payloads, not a measured process peak or a runtime-admissible production plan.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from calibrate_stage2_d import phi, shape

MIB = 1 << 20
DEFAULT_DS = (510510, 690690, 810810, 1021020, 1141140, 1381380, 1531530)
SOURCES = (
    'src/core/ecm_stage2_geometry.h', 'src/core/ecm_stage2_cost_profile.h',
    'src/core/ecm_cuda_stage2.h', 'src/core/ecm_cuda_stage2_main.cpp',
    'src/cuda/ecm_cuda_stage2.cu', 'src/cuda/stage2/ntt_runtime.cuh',
    'src/cuda/stage2/stage2_d_model.cuh',
    'src/cuda/stage2/stage2_point_mersenne.cuh',
    'tools/bench/calibrate_stage2_d.py',
    'tools/bench/analyze_stage2_carrier_plan.py',
)


def owner_bytes(p, w, reuse=3):
    # Exact expansion of fold_owner_layout, including constants and metadata.
    source = (4 if reuse & 2 else 5) * (p + 1) * w
    result = ((3 if reuse & 1 else 4) * p + 2) * w
    return 8 * (source + result + w) + 48


def baby_bytes(p, w):
    # Exact temporary payload from d_baby_payload_bytes, independent of owner.
    count, nodes = p, 0
    for _ in range(8):
        count = (count+1)//2
        nodes += count
    return 8*((3*p+5)*w+p+nodes*w)+count


def geometry(d, bits, b2, fold_mib, free_mib, reserve_mib, arena_mib,
             workspace_buffers=3, baby_mib=512):
    p, w = phi(d) // 2, (bits + 63) // 64
    i = b2 // d + 2
    if i <= p:
        raise ValueError('Concurrent lower-bound model requires repeated G trees (I > P)')
    nf, bpw, sw = shape(p + 1, bits)
    nt = shape(p // 2 + 1, bits)[0]
    # The existing padded binary tree can have a larger child than P/2.
    child = 1 << ((p - 1).bit_length() - 1) if p > 1 else 1
    padded_nt = shape(child + 1, bits)[0]
    old_arena = 8 * ((3 * nf + 2 * p + 1) +
                     2 * (3 * nt + 2 * (p // 2 + 1) - 1))
    pool = 8 * workspace_buffers * nf
    owner = owner_bytes(p, w)
    # build_groot_device retains both buffers: 2P and P+ceil(P/2) coefficients.
    raw = 8 * w * (3 * p + (p + 1) // 2)
    # The existing 256MiB point chunk rounds UP to a multiple of P.
    k = max(p, (256 * MIB) // (16 * w))
    chunk = p * ((k + p - 1) // p)
    chunk = min(chunk, i)
    coords = 16 * w * chunk
    # These allocations coexist in the repeated G-tree/fold phase. Omitted:
    # tables, base plans, S4 output/staging, seed/index/product buffers, etc.
    lower = pool + owner + raw + coords
    return dict(D=d, P=p, bits=bits, words=w, I=i, G=(i+p-1)//p,
                fold_log2=nf.bit_length()-1, fold_length=nf,
                packing_bpw=bpw, slot_words=sw,
                nominal_tree_length=nt, padded_tree_length=padded_nt,
                legacy_arena_mib=old_arena/MIB,
                fold_big_mib=pool/MIB, owner_mib=owner/MIB,
                workspace_buffers=workspace_buffers,
                baby_mib=baby_bytes(p,w)/MIB,
                baby_budget_fits=baby_bytes(p,w) <= baby_mib*MIB,
                legacy_owner_mib=owner_bytes(p, w, reuse=0)/MIB,
                raw_g_mib=raw/MIB, coord_mib=coords/MIB,
                giant_chunk_points=chunk, concurrent_lower_mib=lower/MIB,
                owner_budget_fits=owner <= fold_mib*MIB,
                legacy_arena_fits=old_arena <= arena_mib*MIB,
                lower_fits_free=lower <= free_mib*MIB,
                lower_fits_after_reserve=lower <= max(0, free_mib-reserve_mib)*MIB)


def largest_p(bits, max_log2):
    lo, hi = 1, 1 << 22
    while lo < hi:
        mid = (lo + hi + 1) // 2
        try:
            fits = shape(mid + 1, bits)[0] <= 1 << max_log2
        except ValueError:
            fits = False
        if fits:
            lo = mid
        else:
            hi = mid - 1
    return lo


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bits', type=int, default=7995)
    ap.add_argument('--carrier-bits', type=int, default=8011)
    ap.add_argument('--b2', type=int, default=2600000000000)
    ap.add_argument('--d', type=int, nargs='+', default=DEFAULT_DS)
    ap.add_argument('--free-mib', type=int, default=7106,
                    help='Historical free memory after CUDA context, not current GPU query')
    ap.add_argument('--reserve-mib', type=int, default=768)
    ap.add_argument('--arena-mib', type=int, default=6300)
    ap.add_argument('--fold-mib', type=int, default=640)
    ap.add_argument('--workspace-buffers', type=int, choices=(2,3), default=3,
                    help='Physical shared big-buffer count; exported/fallback paths still use 3')
    ap.add_argument('--baby-mib', type=int, default=512)
    ap.add_argument('--gpu-analysis', type=Path,
                    help='Optional completed N-scaling analysis; auto-use local 20261008 analysis if present')
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    if not 2 <= a.bits <= a.carrier_bits <= 16384 or a.b2 <= 0:
        ap.error('Need 2 <= bits <= carrier-bits <= 16384, B2 > 0')
    if any(d < 6 or d % 2 for d in a.d) or min(a.free_mib, a.reserve_mib, a.arena_mib, a.fold_mib,a.baby_mib) < 0:
        ap.error('Need even D >= 6 and nonnegative budgets')
    root = Path(__file__).resolve().parents[2]
    rows = [geometry(d, bits, a.b2, a.fold_mib, a.free_mib,
                     a.reserve_mib, a.arena_mib,a.workspace_buffers,a.baby_mib)
            for bits in sorted({a.bits, a.carrier_bits}) for d in a.d]
    reference = a.gpu_analysis
    if reference is None:
        historical = root/'data/benchmarks/stage2_n_scaling_20261008_analysis.json'
        reference = historical if historical.is_file() else None
    evidence = json.loads(reference.read_text(encoding='utf-8')) if reference else None
    if evidence is not None and not evidence.get('complete'):
        raise ValueError('Completed GPU analysis required')
    carriers = []
    for s in evidence['summary'] if evidence else []:
        if s['variant'] != 'cofactor':
            continue
        n, m, p = int(s['bits']), int(s['exponent']), int(s['P'])
        old, new = shape(p+1, n), shape(p+1, m)
        carriers.append(dict(exponent=m, bits=n, B2=s['B2'], D=s['D'], P=p,
                             target_words=(n+63)//64, carrier_words=(m+63)//64,
                             target_fold_log2=old[0].bit_length()-1,
                             carrier_fold_log2=new[0].bit_length()-1,
                             ntt_length_ratio=new[0]/old[0],
                             target_owner_mib=owner_bytes(p,(n+63)//64)/MIB,
                             carrier_owner_mib=owner_bytes(p,(m+63)//64)/MIB))
    result = dict(schema=1, measured=False, b2=a.b2,
                  target_bits=a.bits, carrier_bits=a.carrier_bits,
                  free_mib=a.free_mib, reserve_mib=a.reserve_mib,
                  arena_mib=a.arena_mib, fold_mib=a.fold_mib,
                  workspace_buffers=a.workspace_buffers,baby_budget_mib=a.baby_mib,
                  note='Offline integer geometry. Lower bound is NOT full VRAM peak; passing does not establish feasibility.',
                  source_sha256={s: hashlib.sha256((root/s).read_bytes()).hexdigest() for s in SOURCES},
                  gpu_analysis_source=str(reference) if reference else None,
                  gpu_analysis_sha256=hashlib.sha256(reference.read_bytes()).hexdigest() if reference else None,
                  fold_boundaries={str(b): {str(k): largest_p(b,k) for k in (26,27,28)}
                                   for b in sorted({a.bits,a.carrier_bits})},
                  rows=rows, carrier_cases=carriers)
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output/'analysis.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    for name, data in (('geometry',rows), ('carrier_cases',carriers)):
        if not data:
            continue
        with (a.output/(name+'.csv')).open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f, fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    print(json.dumps({'rows':len(rows),'carrier_cases':len(carriers),
                      'fold_boundaries':result['fold_boundaries'],
                      'ntt_jump_cases':[r for r in carriers if r['ntt_length_ratio']>1]},indent=2))


if __name__ == '__main__':
    main()
