"""Check the Stage2 module boundary and frozen arithmetic, independently of builds."""
from source_layout import current_source,current_files,relocation_text
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def mask(source):
    return re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
                  lambda m: ''.join('\n' if c == '\n' else ' ' for c in m[0]), source, flags=re.S)


def balanced(source, start, left, right):
    depth = 0
    for i in range(start, len(source)):
        if source[i] == left: depth += 1
        elif source[i] == right:
            depth -= 1
            if not depth: return i
    raise AssertionError('unbalanced source')


def body(source, name):
    clean = mask(source)
    for found in re.finditer(r'\b'+re.escape(name)+r'\s*\(', clean):
        opening = clean.index('(', found.start())
        end = balanced(clean, opening, '(', ')')+1
        while clean[end].isspace(): end += 1
        if clean[end] != '{': continue
        finish = balanced(clean, end, '{', '}')
        return re.sub(r'\s+', '', clean[end:finish+1])
    raise AssertionError('missing definition: '+name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', default='5e3ed63')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    current = lambda path: relocation_text(path)
    frozen = lambda path: subprocess.check_output(['git', 'show', a.baseline+':'+path], cwd=ROOT).decode('utf-8-sig')
    original = frozen('src/cuda/ecm_cuda_stage2.cu')
    groups = [
        ('point.cuh', 's2g_mac s2g_sub_n s2g_ge_n s2g_addmod s2g_submod s2g_mont_mul s2g_xdbl s2g_halfmod s2g_xadd s2g_ladder s2g_ladder_kernel s2g_mont_test_kernel s2g_xadd_test_kernel s2g_seed_pair_kernel s2g_segprod_kernel s2g_projective_leaf_kernel s2g_block_prod_kernel'),
        ('s4_reduce.cuh', 's4_mersenne_fixture_kernel s4_reduce_kernel'),
        ('coefficient_ops.cuh', 's4_scatter_result_kernel groot_input_digest_kernel scaled_root_reverse_kernel fold_reverse_kernel fold_subtract_kernel'),
    ]
    checked = []
    def compare(old, new, names):
        for name in names.split():
            before, after = body(old, name), body(new, name)
            assert before == after, 'changed arithmetic: '+name
            checked.append(dict(function=name, sha256=hashlib.sha256(after.encode()).hexdigest()))
    for path, names in groups: compare(original, current('kernels/cuda/stage2/'+path), names)
    compare(original, current('kernels/cuda/shared/stage2_scalar.cuh'), 's2g_mul64 s2g_udiv_2by1 s4_div_rem s4_mersenne_rem groot_mix64')
    for old, new, names in [
        ('s4_input_pack.cuh', 'pack.cuh', 'quotient pack_zero_kernel s4_pack_batch_kernel s4_pack_gather_kernel s4_pack_gather_reverse_kernel'),
        ('stage2_baby_device.cuh', 'baby_giant.cuh', 's2g_baby_negative s2g_baby_product_kernel s2g_baby_inverse_kernel s2g_baby_leaf_kernel'),
        ('scale_plain.cuh', 'plain_scale.cuh', 's2g_plain_scale_kernel'),
    ]:
        implementation=current('kernels/cuda/stage2/'+new)
        if new=='pack.cuh':implementation+='\n'+current('kernels/cuda/shared/pack_index.cuh')
        compare(frozen('src/cuda/stage2/'+old), implementation, names)
    assert frozen('src/cuda/stage2/stage2_point_mersenne.cuh') == current('kernels/cuda/stage2/point_mersenne.cuh')
    chain=current('kernels/cuda/stage2/giant_chain.cuh').split('\ntemplate<int NW,bool XADD6>\nvoid ecm_launch_')[0].replace('#include "ecm/cuda/stage2_launch.h"\n', '')
    # The probe-only rotation has no production instantiation. P4 isolates it
    # without changing a byte of its arithmetic or the real scalar chain.
    if 's2g_chain_rotate_kernel' not in chain:
        variant=current('tools/experiments/cuda/stage2_tree/giant_chain_variant.cuh')
        rotation=variant[variant.index('// Probe-only buffer rotation'):]
        chain=chain.rstrip()+'\n\n\n'+rotation
    assert frozen('kernels/cuda/stage2_giant_chain.cuh') == chain.rstrip()+'\n'
    device_paths = sorted(current_files('kernels/cuda/stage2'))
    ntt_original = frozen('src/cuda/stage2/ntt_runtime.cuh')
    ntt_device = '\n'.join(path.read_text(encoding='utf-8-sig')
        for folder in ('kernels/cuda/ntt', 'kernels/cuda/shared')
        for path in sorted(current_files(folder)) if path.is_file())
    compare(ntt_original, ntt_device, 'gl_reduce gl_mod gl_mul mul_64x64_hi_lo gl_mul_host gl_add gl_sub gl_mod_dev gl_sub_dev gl_add_dev gl_pow_host gl_add_host gl_sub_host bitlen_u64 bits_of gl_twiddle gl_mul_mod_n gl_mul_mod_n_host pointwise_twiddle_kernel ntt_stage dit_stage dit_stage_kernel ntt_stage_kernel bitrev_kernel bitrev_swap_kernel carry_extract_kernel carry_add_kernel carry_residual_kernel slot_assemble_kernel scale_kernel carry_cone_value carry_cone_kernel build_tile_table_kernel build_pass_table_kernel build_radix_table_kernel outer_fwd_kernel outer_inv_kernel tile_radix4_block tile_radix4_block_inv tile_radix2_stage tile_warp_twiddle tile_warp5 tile_warp6_pair tile_kernel gl_mul_test_kernel gl_twiddle_test_kernel dft_direct_kernel')
    for old, names in [
        ('ntt_coop_outer.cuh', 'outer_coop_kernel'),
        ('ntt_carry_partial.cuh', 'carry_partial_store carry_partial_kernel carry_cone_check_kernel carry_partial_finish_kernel'),
    ]:
        compare(frozen('src/cuda/stage2/'+old), ntt_device, names)
    compare(frozen('src/cuda/ecm_stage2_tune.cuh'), ntt_device, 'seed verdict')
    for old, new in [('ntt_goldilocks_reduce.cuh', 'goldilocks_reduce.cuh'),
                     ('ntt_goldilocks_ptx.cuh', 'goldilocks_ptx.cuh'),
                     ('ntt_goldilocks_sub.cuh', 'goldilocks_sub.cuh')]:
        assert frozen('src/cuda/stage2/'+old) == current('kernels/cuda/shared/'+new)
    assert frozen('src/cuda/stage2/device_memory_ledger.cuh') == current('src/cuda/common/device_memory_ledger.h')
    assert frozen('src/cuda/stage2/ntt_host_trace.cuh') == current('src/cuda/ntt/host_trace.h')
    for path in current_files('src/cuda/ntt'):
        text = mask(path.read_text(encoding='utf-8-sig'))
        assert not re.search(r'__global__|__device__|<<<|ecm_stage2_(?:tree|fold|gcd)|S4Ctx', text), path
    for path in ['src/cuda/stage2/engine.cpp', 'src/cuda/stage2/tune.cpp']:
        assert not re.search(r'__global__|__device__|<<<', mask(current(path))), path
    device_paths += sorted(current_files('kernels/cuda/ntt'))
    for path in device_paths:
        text = mask(path.read_text(encoding='utf-8-sig'))
        assert not re.search(r'\b(?:mpz_\w*|gmp_\w*|checkpoint\w*)\b', text), path
        includes = re.findall(r'#\s*include\s*["<]([^">]+)', path.read_text(encoding='utf-8-sig'))
        assert not any('/src/' in name or '/tools/' in name or 'gmp' in name for name in includes), path
    result = dict(complete=True, arithmetic_functions=checked, device_files=len(device_paths), gpu_runs=0)
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(complete=True, arithmetic_functions=len(checked), device_files=len(device_paths))))


if __name__ == '__main__': main()
