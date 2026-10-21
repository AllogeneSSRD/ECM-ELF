"""Compare moved host bodies, allowing only the specified thin launch adapters.

Strings, checks, allocation/synchronization order and launch arguments remain
significant. This complements GPU checks; it does not qualify timing behavior.
"""
from source_layout import current_source,current_files
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from test_cuda_p2_structure import ROOT, mask, balanced
from ntt_native_fixture import declaration


def raw_body(text, name):
    clean = mask(text)
    for found in re.finditer(r'\b'+re.escape(name)+r'\s*\(', clean):
        end = balanced(clean, clean.index('(', found.start()), '(', ')')+1
        suffix = re.match(r'\s*(?:const\s*|noexcept\s*)*', clean[end:])
        end += suffix.end()
        if clean[end] == '{': return text[end:balanced(clean, end, '{', '}')+1]
    raise AssertionError('missing host body: '+name)


def split_arguments(text):
    clean = mask(text); start = 0; stack = []; result = []
    for i, char in enumerate(clean):
        if char in '({[': stack.append(char)
        elif char in ')}]': stack.pop()
        elif char == ',' and not stack: result.append(text[start:i].strip()); start = i+1
    result.append(text[start:].strip())
    return result


def canonical(text):
    # Keep the original argument text. CUDA's omitted shared/stream are zero.
    clean = mask(text); edits = []
    for found in re.finditer(r'\b(\w+)(<[^;\n{}]+?>)?\s*<<<', clean):
        end = clean.index('>>>', found.end())
        config = split_arguments(text[found.end():end])
        opening = clean.index('(', end); finish = balanced(clean, opening, '(', ')')
        config += ['0']*(4-len(config))
        edits.append((found.start(), finish+1,
                      '__launch__('+found[1]+(found[2] or '')+','+','.join(config)+','+text[opening+1:finish]+')'))
    for found in re.finditer(r'\becm_launch_(\w+)(<[^;\n{}]+?>)?\s*\(', clean):
        opening = clean.index('(', found.start()); finish = balanced(clean, opening, '(', ')')
        launch = clean.index('EcmCudaLaunch', opening)+len('EcmCudaLaunch')
        while clean[launch].isspace(): launch += 1
        end = balanced(clean, launch, clean[launch], ')' if clean[launch] == '(' else '}')
        config = split_arguments(text[launch+1:end]); config += ['0']*(4-len(config))
        arguments = text[end+1:finish].lstrip().removeprefix(',')
        edits.append((found.start(), finish+1,
                      '__launch__('+found[1]+(found[2] or '')+','+','.join(config)+','+arguments+')'))
    for start, end, value in sorted(edits, reverse=True): text = text[:start]+value+text[end:]
    text = text.replace('cudaMemcpyToSymbol(ntt_gl_short_reduce_device,&enabled,sizeof(enabled))', 'ecm_ntt_copy_reduce_mode(enabled)')
    for inverse in ['false', 'true']:
        for warp in ['false', 'true']:
            old = r'\(const void\s*\*\)tile_kernel<'+inverse+(','+warp if warp == 'true' else '')+'>'
            text = re.sub(old, 'ecm_ntt_tile_handle('+inverse+','+warp+')', text)
    # Tokenization ignores comments/spacing while retaining literal contents.
    return re.findall(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*.*?\*/|\w+|[^\s]', text, re.S)


def tokens(text):
    return [token for token in canonical(text) if not token.startswith(('//', '/*'))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='5e3ed63')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    frozen = lambda path: subprocess.check_output(['git', 'show', args.baseline+':'+path], cwd=ROOT).decode('utf-8-sig')
    old = frozen('src/cuda/stage2/ntt_runtime.cuh')
    new = '\n'.join(p.read_text(encoding='utf-8-sig') for p in sorted(current_files('src/cuda/ntt')))
    names = 'ntt_gl_reduce_requested_mode ntt_gl_reduce_configure now_s ntt_bitrev_inplace ntt_dif ntt_bitrev_perm ntt_forward ntt_inverse ntt_carry_check_requested ntt_tile_attributes fuse_base_allocate fuse_base_free fuse_plan_stages fuse_env_ull fuse_compact_scratch fuse_shape_device_supported fuse_shape_policy_supported fuse_joint_policy_supported fuse_selected_tile fuse_outer_max fuse_describe fuse_planned_table_words fuse_planned_base_words fuse_init fuse_release fuse_base_words ntt_fuse_cache_tables fuse_grid fuse_trace_on fuse_mark ntt_forward_fused ntt_inverse_fused ntt_arena_bufs ntt_arena_pin_smem ntt_arena_fuse ntt_trim_fuse_headroom gl_selftest exact_ok_terms exact_ok_nterms mpz_log2d mpz_log2d_p coeff_bound_bits_terms bound_over_p_terms coeff_bound_bits choose_cfg max_conv_coeff_gmp digit_sq_double mpz_add_u64 dump_cmp ntt_host_break_on extract_exact_coeffs ntt_shape_query ntt_shape_plan ntt_shape_plan_uncached ntt_shape_exactness ntt_pack_operand ntt_shape_maxcoeff ntt_shape_print ntt_run_passes ntt_pass_read_marks ntt_hook_window ntt_poly_mul_batch_host ntt_poly_mul_batch_dev ntt_batch_carry_finish ntt_workspace_check ntt_fuse_lifetime_check fuse_fixture_env ntt_fuse_capacity_check ntt_fuse_coop_check ntt_poly_mul_host'
    checked = []
    for name in names.split():
        before, after = tokens(raw_body(old, name)), tokens(raw_body(new, name))
        if before != after:
            (out/(name+'.json')).write_text(json.dumps(dict(before=before, after=after), indent=2)+'\n')
            raise AssertionError('changed NTT host body: '+name)
        checked.append(dict(function=name, sha256=hashlib.sha256(json.dumps(after).encode()).hexdigest()))
    for class_name, source, methods in [
        ('NttArena', old, 'drop_carry_scratch carry_workspace try_malloc drop_workspace input_span payload drop_cold_fuse update_peaks print_workspace_stats fuse_table_words fuse_drop_tables evict_other_shapes NttArena release mb'),
        ('NttGraphCache', frozen('src/cuda/stage2/ntt_graph.cuh'), 'destroy clear release NttGraphCache recoverable accept capture lookup print'),
    ]:
        old_class = declaration(source, class_name)
        new_class = declaration(new, class_name)
        new_methods = current_source('src/cuda/ntt/runtime.cpp' if class_name == 'NttArena' else 'src/cuda/ntt/graph.cpp').read_text(encoding='utf-8-sig')
        for name in methods.split():
            before_body = raw_body(old_class, name)
            before, after = tokens(before_body), tokens(raw_body(new_methods, name))
            assert before == after, 'changed method: '+class_name+'::'+name
            checked.append(dict(function=class_name+'::'+name, sha256=hashlib.sha256(json.dumps(after).encode()).hexdigest()))
            old_class = old_class.replace(before_body, ';', 1)
        assert tokens(old_class) == tokens(new_class), 'changed fields/defaults: '+class_name
    compare = lambda before, after: tokens(before) == tokens(after)
    assert compare(raw_body(frozen('src/cuda/stage2/ntt_coop_outer.cuh'), 'launch_outer_coop'), raw_body(new, 'launch_outer_coop'))
    result = dict(complete=True, functions=checked, class_layouts=2, gpu_runs=0)
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(complete=True, host_functions=len(checked))))


if __name__ == '__main__': main()
