"""P4 reachability and retained-source proof against the P3 commit.

Only the finite, justified reductions below are allowed. Literal contents and
all other statements (including CUDA calls, checks and fallbacks) are compared.
This is a source proof, separate from the paired executable/GMP gates.
"""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from test_cuda_p2_host import tokens
from test_cuda_p2_structure import mask, balanced
from stage2_native_source import ROOT, units, signature

UNUSED = {'read_hex_mpz', 'read_f_dump', 'words_for_bits', 'hex_of_words', 'flat_truncate'}


def remove_units(text, names):
    edits = []
    for kind, a, b, value in units(text):
        if kind == 'func' and signature(value)[0] in names:
            edits.append((a, b))
        elif kind == 'decl' and any(re.search(r'\b'+n+r'\s*\(', mask(value)) for n in names):
            edits.append((a, b))
        elif kind == 'type' and re.match(r'struct\s+FDump\b', mask(value)):
            edits.append((a, b))
    for a, b in reversed(edits): text = text[:a]+text[b:]
    return text


def drop_if(text, condition, keep=False):
    """Reduce exactly a braced if with a proved constant condition."""
    pattern = re.compile(r'\bif\s*\(\s*'+re.escape(condition)+r'\s*\)\s*\{')
    while found := pattern.search(mask(text)):
        a = found.start(); b = mask(text).index('{', found.start())
        end = balanced(mask(text), b, '{', '}')
        # A true branch keeps its scope so vectors/owners still die before the
        # following statement. Removing the condition must not extend a lease.
        text = text[:a]+(text[b:end+1] if keep else '')+text[end+1:]
    return text


def reduce_batch(text):
    clean = mask(text)
    assert len(re.findall(r'\bdev_leaves\s*=(?!=)', clean)) == 1
    assert re.search(r'bool\s+dev_leaves\s*=\s*false\s*;', clean)
    assert '&dev_leaves' not in clean
    text = text.replace('bool dev_leaves = false;', '')
    text = drop_if(text, 'dev_leaves')
    text = drop_if(text, 'dev_leaves && values.empty()')
    text = drop_if(text, '!dev_leaves || R.s5_readback', True)
    text = drop_if(text, '!dev_leaves', True)
    assert 'dev_leaves' not in mask(text)
    return text


def reduce_plan(text):
    # Both assignments independently prove the model is disabled for every
    # entry/configuration. No later statement assigns true or exposes its address.
    assert 'const bool cache_rates_valid=false;' in text
    assert re.search(r'\n    calibrated=false;', text)
    assignments = re.findall(r'(?<![.\w])calibrated\s*=\s*([^;]+);', mask(text))
    assert len(assignments) > 10 and all(x.strip() == 'false' for x in assignments[1:])
    a = text.index('    const auto outer_mode=')
    b = text.index('    const double d_scan_begin=', a)
    # These inputs are still used by the live memory plan or debug output.
    text = text[:a]+'''    const bool baby_requested=fuse_env_ull("NTT_BABY_DEVICE",0)!=0;
    const auto baby_budget=fuse_env_ull("NTT_BABY_DEVICE_MAX_MB",512)*1024*1024;
    const bool gl_short=(ntt_gl_reduce_requested_mode()&1u)!=0;
    const bool gl_shift_scale=fuse_env_ull("NTT_GL_SHIFT_SCALE",0)!=0;
    const bool gl_ptx=NTT_GL_FIXED_MODE==3 || fuse_env_ull("NTT_GL_PTX_REDUCE",0)!=0;
    const auto fold_budget=fuse_env_ull("NTT_FOLD_DEVICE_MAX_MB",640)*1024*1024;
    stage2_log::print(stage2_log::debug, "ntt_addsub_arithmetic: mask=1\\n");
    if(model_requested)
        stage2_log::print(stage2_log::debug, "d_model_scope: calibrated=0 reason=cache_payload_v2_unmeasured\\n");
'''+text[b:]
    text=text.replace('    const bool baby_requested=',
        '    if(fuse_env_ull("NTT_FUSE_COOP_OUTER",0)==2)(void)fuse_shape_policy_supported(12);\n    const bool baby_requested=',1)
    a = text.index('    auto &phase_model=')
    b = text.index('    stage2_log::print', a)
    text = text[:a]+'    const char *model_version="legacy_56_1";\n'+text[b:]
    text = drop_if(text, 'calibrated')
    a = text.index('        if(calibrated) stage2_log::print')
    b = text.index('else stage2_log::print', a)
    text = text[:a]+text[b+5:]
    text = re.sub(r'if\(calibrated && \(owner_bytes\(c.P\)>fold_budget \|\| c.total>=1e80\)\)continue;', '', text)
    text = re.sub(r'if\(calibrated\)phase_model.print\([^;]+;', '', text)
    text = text.replace('(int)calibrated', '0').replace('p.calibrated=calibrated;', 'p.calibrated=false;')
    text = text.replace('d_baby_payload_bytes(', 'ecm_stage2::baby_payload_bytes(')
    return text


def reduce_engine(text):
    # The only run caller is the saved-curve API. Query already has its own path.
    old_call = 'false, 1, x_hex, &tail,false,nullptr,carrier_exponent,a24_hex'
    assert text.count(old_call) == 1
    assert len(re.findall(r'\brun_real\s*\(', mask(text))) == 2
    text = text.replace(old_call, 'x_hex, &tail,carrier_exponent,a24_hex')
    for line in ('    bool run_s2,\n', '    int curves,\n', '    bool d_plan_only,\n',
                 '    ecm_stage2::Plan *plan_result,\n'):
        assert text.count(line) == 1; text = text.replace(line, '')
    text = text.replace('g_ntt_counts_enabled && !d_plan_only', 'g_ntt_counts_enabled')
    text = text.replace('choose_d,curves,d_plan_only,plan_result,planning', 'choose_d,1,false,nullptr,planning')
    text = text.replace('if(plan_code || d_plan_only)', 'if(plan_code)')
    text = text.replace('!stage2_extra_fixtures && !run_s2 && curves==1', '!stage2_extra_fixtures')
    text = text.replace('cv+1,stage2_shape_seconds', '1,stage2_shape_seconds')
    loop = 'for (int cv = 0; cv < curves; ++cv) {'
    assert text.count(loop) == 1
    a = text.index(loop); b = text.index('{', a); end = balanced(mask(text),b,'{','}')
    # Exactly one iteration, with the original block/destructor boundary.
    text = text[:a]+text[b:end+1]+text[end+1:]
    return text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', default='2a2606e')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    frozen = lambda path: subprocess.check_output(['git','show',a.baseline+':'+path],cwd=ROOT).decode('utf-8-sig')
    paths = subprocess.check_output(['git','ls-tree','-r','--name-only',a.baseline,'src/cuda/stage2'],cwd=ROOT).decode().splitlines()
    before = '\n'.join(frozen(path) for path in paths)
    # Unused symbols have only a definition and a private declaration. Whole
    # repository consumers must not refer to those production functions.
    for name in UNUSED:
        assert len(re.findall(r'\b'+name+r'\b',mask(before))) == 2, name
    assert 'static constexpr bool d_shape_rates_valid=true;' in frozen('src/cuda/stage2/stage2_d_model.cuh')
    assert len(re.findall(r'\brun_real\b',mask(before)))==3, 'curve entry address escaped its private call/declaration'
    checked=[]
    for path in paths:
        if path.endswith('stage2_d_model.cuh'): continue  # frozen into experiments
        old = frozen(path)
        if path.endswith(('diagnostics.cpp','diagnostics.h','remainder.cpp','remainder.h')):
            old = remove_units(old, UNUSED)
        if path.endswith('context.h'): old = old.replace('inline constexpr bool g_scaled_descent = true;', '')
        if path.endswith('batch.cpp'): old = reduce_batch(old)
        if path.endswith('plan.cpp'): old = reduce_plan(old)
        if path.endswith('engine.cpp'): old = reduce_engine(old)
        if path.endswith('engine.h'):
            old = old.replace('bool choose_d, bool run_s2, int curves,','bool choose_d,')
            old = old.replace(', bool d_plan_only = false', '')
            old = old.replace('ecm_stage2::Plan *plan_result = nullptr, ', '')
        if path.endswith('plan_runtime.h'):
            old = old.replace('#include "stage2_d_model.cuh"','').replace('#include <optional>','')
            old = old.replace('std::optional<DPhaseModel> phase_model;', '')
        current = (ROOT/path).read_text(encoding='utf-8-sig')
        if path.endswith('product.cpp'):
            from test_cuda_p2_host import raw_body
            header=(ROOT/'src/cuda/stage2/pack_launch.h').read_text()
            names={'supported','launch','s4_launch_pack_batch'}
            def functions(text):
                result={}
                for kind,_,_,value in units(text):
                    if kind=='namespace':
                        op=mask(value).index('{');cl=balanced(mask(value),op,'{','}')
                        result.update(functions(value[op+1:cl]))
                    elif kind=='func' and signature(value)[0] in names:result[signature(value)[0]]=value
                return result
            expected=functions(old);actual=functions(header)
            assert set(expected)==set(actual)==names
            for name in names:
                assert tokens(expected[name])==tokens(actual[name]),'pack launch contract changed: '+name
                old=old.replace(expected[name],'')
            current=current.replace('#include "pack_launch.h"','')
        assert tokens(old)==tokens(current), 'unapproved retained-source change: '+path
        checked.append(path)
    result=dict(complete=True,baseline=a.baseline,checked=checked,unused=sorted(UNUSED),
                proof=['dev_leaves single false assignment; no address escape',
                       'run_real only caller: run_s2=false curves=1 d_plan_only=false plan_result=nullptr',
                       'calibrated false initializer and unconditional false assignment; no true assignment',
                       'live NTT policy-cache warmup retained at original planning boundary',
                       'three bounded pack launch bodies/defaults unchanged'])
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(complete=True,modules=len(checked),unused=len(UNUSED))))


if __name__=='__main__': main()
