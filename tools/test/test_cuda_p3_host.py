"""P3 host-flow/resource/check identity against the frozen P2 commit.

Retain literal contents, checks, allocation/synchronization order, defaults and
constructor initializers. This proves source relocation, not GPU correctness.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import re
import subprocess
from test_cuda_p2_host import tokens, raw_body
from stage2_native_source import ROOT, STAGE2, units, signature, class_declaration
from test_cuda_p2_structure import mask, balanced


def collect(text):
    functions = defaultdict(list); classes = {}; initializers = {}
    def visit(source, parent=''):
        for kind, _, _, value in units(source):
            if kind == 'func':
                name, _, ep, bp = signature(value); name = parent+name
                functions[name].append(tokens(value[bp:]))
                if parent and ':' in mask(value)[ep:bp]: initializers[name] = tokens(value[ep:bp])
                elif '::' in name and name.split('::')[-1] == name.split('::')[-2]:
                    initializers[name] = tokens(value[ep:bp])
            elif kind == 'type' and re.match(r'struct\s+\w+', mask(value)):
                name = re.match(r'struct\s+(\w+)', mask(value))[1]
                clean = mask(value); start = clean.index('{'); end = balanced(clean, start, '{', '}')
                if not parent: classes[name] = tokens(class_declaration(value))
                visit(value[start+1:end], parent+name+'::')
    visit(text)
    return functions, classes, initializers


def expand_planning(old, current):
    """Inline the finite P3 planning seam for comparison with the P2 body.

    Only scalar result export and optional's in-place model storage disappear;
    every original planning statement, literal, branch and CUDA call remains.
    """
    run = raw_body(current, 'run_real')
    prepare = raw_body(current, 'prepare_stage2_plan')[1:-1]
    export = '''    runtime.freeb=freeb;runtime.totalb=totalb;runtime.reserve=reserve;runtime.cap=cap;
    runtime.D=D;runtime.P_baby=P_baby;runtime.s4_on=s4_on;
    return 0;
'''
    assert prepare.endswith(export), 'planning result export changed'
    prepare = prepare[:-len(export)].rstrip()
    stored = 'auto &phase_model=runtime.phase_model.emplace('
    assert prepare.count(stored) == 1
    prepare = prepare.replace(stored, 'DPhaseModel phase_model(')
    seam = '''    Stage2PlanRuntime planning;
    const int plan_code=prepare_stage2_plan(L,nw,sigma,B1,B2,D_in,choose_d,curves,d_plan_only,plan_result,planning);
    if(plan_code || d_plan_only)return plan_code;
    const auto freeb=planning.freeb,totalb=planning.totalb,reserve=planning.reserve,cap=planning.cap;
    const auto D=planning.D,P_baby=planning.P_baby;
    const bool s4_on=planning.s4_on;
'''
    assert run.count(seam) == 1, 'curve planning seam changed'
    run = run.replace(seam, prepare.strip()+'\n')
    # The query keeps the original modulus/session setup and enters the same
    # planner before any curve, point or arithmetic-check work can begin.
    prefix = raw_body(old, 'run_real')[1:]
    prefix = prefix[:prefix.index('    CK(cudaSetDevice(g_device));')]
    prefix = prefix.replace('PolyLayer L;', 'ecm_stage2::ModulusContext L;')
    for statement in ('    L.ntt_lengths.enabled=g_ntt_counts_enabled && !d_plan_only;\n',
                      '    L.device = g_device;\n', '    L.cost.S = (long)L.S;\n',
                      '    const size_t nw = L.W;\n'):
        assert prefix.count(statement) == 1; prefix = prefix.replace(statement, '')
    prefix = prefix.replace('n_is_hex?16:10', '16')
    query = '{'+prefix+'''    Stage2PlanRuntime planning;
    return prepare_stage2_plan(L,L.W,sigma,B1,B2,D_in,D_in==0,1,true,&result,planning);
}'''
    assert tokens(query) == tokens(raw_body(current,'query_stage2_plan')), 'query setup/order changed'
    plan = raw_body(current,'ecm_cuda_stage2_plan')
    query_call = 'query_stage2_plan(n_hex,sigma,b1,b2,d,p,carrier_exponent)'
    assert plan.count(query_call) == 1
    plan = plan.replace(query_call, 'run_real(n_hex,true,sigma,b1,b2,d,d==0,false,1,nullptr,nullptr,true,&p,carrier_exponent)')
    return tokens(run), tokens(plan)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', default='d19123b')
    p.add_argument('--output', type=__import__('pathlib').Path, required=True)
    a = p.parse_args(); out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    frozen = lambda path: subprocess.check_output(['git', 'show', a.baseline+':'+path], cwd=ROOT).decode('utf-8-sig')
    originals = ['src/cuda/ecm_cuda_stage2.cu', 'src/cuda/stage2/stage2_baby_host.cuh',
                 'src/cuda/stage2/scaled_frontier.cuh']
    old = '\n'.join(frozen(path) for path in originals)
    paths = sorted(p for p in STAGE2.iterdir() if p.suffix in ('.cpp','.h','.cuh'))
    current = '\n'.join(p.read_text(encoding='utf-8-sig') for p in paths)
    before, classes, init = collect(old); after, current_classes, current_init = collect(current)
    after['run_real'], after['ecm_cuda_stage2_plan'] = [[v] for v in expand_planning(old,current)]
    checked = []
    for name, bodies in before.items():
        assert name in after, 'missing function: '+name
        assert bodies == after[name], 'changed body: '+name
        checked.append(dict(function=name, bodies=len(bodies), sha256=hashlib.sha256(json.dumps(bodies).encode()).hexdigest()))
    for name, fields in classes.items():
        assert fields == current_classes[name], 'changed fields/defaults/method declarations: '+name
    for name, initializer in init.items():
        assert initializer == current_init[name], 'changed constructor initializer: '+name
    # Calibration remains bound to each engine's own NTT runtime. Its host
    # algorithm and resource teardown match the original shared implementation.
    old_tune = frozen('src/cuda/ecm_stage2_tune.cuh')
    production_tune = (STAGE2/'tune.cpp').read_text()
    development_tune = (ROOT/'tools/bench/stage2_tune.cuh').read_text()
    for tune in (production_tune,development_tune):
        assert tokens(raw_body(old_tune,'run'))==tokens(raw_body(tune,'run')), 'calibration algorithm changed'
        def resource(text):
            start=text.index('struct Resources {');end=balanced(mask(text),text.index('{',start),'{','}')
            return text[start:end+1]
        assert tokens(resource(old_tune))==tokens(resource(tune)), 'calibration resources changed'
    assert tokens(raw_body(old_tune,'quote'))==tokens(raw_body((ROOT/'src/cuda/common/json_quote.h').read_text(),'quote'))
    for old_path,new_path,names in [
        ('src/cuda/stage2/s4_input_pack.cuh',STAGE2/'product.cpp',
         'supported launch enabled account s4_launch_pack_batch'),
        ('src/cuda/stage2/scale_plain.cuh',STAGE2/'scale_plain.cuh',
         's2g_launch_plain_scale s2g_plain_scale s2g_plain_scale_fixture'),
        ('src/cuda/stage2/giant_chunk_plan.cuh',ROOT/'src/core/ecm_stage2_giant_chunk.h','plan'),
        ('src/cuda/stage2/giant_chunk_plan.cuh',STAGE2/'diagnostics.cpp','fixture'),
    ]:
        for name in names.split():
            assert tokens(raw_body(frozen(old_path),name))==tokens(raw_body(new_path.read_text(),name)), 'shared helper changed: '+name
    # All snapshots remain together after ProductionDefaults, in original order.
    def globals_(text):
        result = []
        for kind, _, _, value in units(text):
            if kind == 'decl' and re.match(r'static\s+', value) and not re.match(r'static\s+(?:void|CPoly)\s+\w+\s*\(', value):
                if any(t.startswith('g_') for t in tokens(value)): result.append(tokens(re.sub(r'^static\s+', '', value)))
            elif kind == 'type' and value.rstrip().endswith('} g_giant_seed;'):
                result.append(tokens('GiantSeedStats g_giant_seed;'))
        return result
    expected = [v for v in globals_(frozen(originals[0])) if 'g_last_state' not in v and 'constexpr' not in v and 'g_s4_batch_budget_mb' not in v]
    options = (STAGE2/'options.cpp').read_text(encoding='utf-8-sig')
    before_defaults = next(v for k,_,_,v in units(frozen(originals[0])) if k=='namespace')
    after_defaults = next(v for k,_,_,v in units(options) if k=='namespace')
    assert tokens(before_defaults)==tokens(after_defaults), 'production defaults changed'
    actual = []
    for kind, _, _, value in units(options):
        if kind == 'decl' and any(t.startswith('g_') for t in tokens(value)): actual.append(tokens(value))
    assert actual == expected, 'global initialization order/defaults changed: '+str(next(((x,y) for x,y in zip(expected,actual) if x!=y),(len(expected),len(actual))))
    # Existing arithmetic, launch interfaces and NTT runtime retain repository
    # bytes (Git's declared CRLF/LF checkout conversion is not a source change).
    unchanged = []
    files = subprocess.check_output(['git','ls-tree','-r','--name-only',a.baseline,'kernels/cuda','include/ecm/cuda','src/cuda/ntt'], cwd=ROOT).decode().splitlines()
    for name in files:
        blob = subprocess.check_output(['git','rev-parse',a.baseline+':'+name],cwd=ROOT).strip()
        current_blob = subprocess.check_output(['git','hash-object','--path='+name,name],cwd=ROOT).strip()
        assert current_blob == blob, 'lower-level source changed: '+name
        unchanged.append(name)
    for path in paths:
        clean = mask(path.read_text(encoding='utf-8-sig'))
        assert not re.search(r'__global__|__device__|<<<', clean), path
        assert not re.search(r'#\s*include\s*"[^"\n]*(?:tools/|\.cpp"|\.cu")',path.read_text()), path
        if path.suffix=='.cpp' and path.stem not in ('engine','batch'):
            assert '#include "engine.h"' not in path.read_text(), 'lower module depends on curve flow: '+str(path)
    planner=(STAGE2/'plan.cpp').read_text()
    assert not re.search(r'\b(?:run_real|run_batched|PolyLayer|LadderCtx|mont_selftest)\b',mask(planner)), 'query depends on curve execution'
    for path in STAGE2.glob('*.h'):
        for kind,_,_,value in units(path.read_text()):
            assert kind!='func' or value.startswith(('inline constexpr','template')), 'implementation body in private API: '+str(path)
    result = dict(complete=True, functions=checked, classes=list(classes), constructor_initializers=list(init),
                  initialization_order=True, unchanged_lower_level=unchanged, gpu_calls=0)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(complete=True,functions=len(checked),classes=len(classes),unchanged_lower_level=len(unchanged))))


if __name__ == '__main__': main()
